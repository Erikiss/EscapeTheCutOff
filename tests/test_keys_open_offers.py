"""Key, not value (docs/DESIGN_keys.md sections 6, 7 and 10): the check of 8cc83c5.

- The grading scope (A): with --judges the grading judges decide every rule grade ``incorrect`` or ``abstain`` of an
  answer that mentions the expected answer, on every arm and set ("It is not Y, it is Zhang.", "Dr. Zhang", "Zhang et
  al.", a given name alone, "I think it's X, but I'm not sure."); they were final grades by rule with no judge asked. A
  rule ``correct``, an answer that does not name the expected answer and a control keep the rule grade; a failed or
  tied vote leaves the row ungraded; without judges nothing changes. A resumed run grades stored rule grades again
  without a recipient call, idempotently; the cap, --limit and a 401 from a grading judge work as before.
- Open offers (B): every answered row stores ``offers_expected``; a verdict of predictions 1, 1b, 2, 3a or 4 that the
  open offers (ungraded answers that offer the expected answer), resolved against it, would overturn is not testable,
  and one they cannot change is the verdict of 8cc83c5; a results file without the field reads as before.
"""

from __future__ import annotations

import asyncio
import json
import random
from types import SimpleNamespace
from typing import Any

import pytest
from keys_testlib import EVENTS, FABLE, NEON, RESULTS, TODAY, fixture_data
from test_keys_report import Study
from test_keys_run import MODELS, ScriptedBackend
from test_keys_run_judges import J1, J2, calls, judged

from cutoffmix.backends.base import BackendError
from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.io import load_events, load_results, read_jsonl, write_jsonl
from cutoffmix.bench.keys import build_plan, render_report, run_plan
from cutoffmix.bench.keys import run as run_mod
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.cli import _keys_run_start, main
from cutoffmix.types import ChatResult

OTHER = "Wei Chen"                                                  # a name the fixture events cannot place
CORRECT = '{"label": "correct", "hedged": false}'                   # the grading judge's vote
INCORRECT = '{"label": "incorrect", "hedged": false}'
UNDECIDED = "not testable (undecided answers; grade with --judges)"


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return {e.id: e for e in events}, rows


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


def name_parts(event) -> tuple[str, str]:
    given, surname = event.expected_current.rsplit(" ", 1)
    return given, surname


# answers that name the expected answer X in a form offers_answer does not accept (or only for some events), graded
# incorrect or abstain by the paper rules: the forms of the check of 8cc83c5
SCOPE = [
    "It is not {Y}, it is {given}.",
    "It's not {Y} - it's {given}.",
    "I think it's {given}, but I'm not sure.",
    "It is not {Y}, it is Dr. {surname}.",
    "The first author is {surname} et al., not {Y}.",
    "Possibly {surname} or {Y}.",
]


def answering(template: str):
    """The recipient answers ``template`` on every arm of every set (``{given}``, ``{surname}``: the parts of the
    event's expected answer; a control has none and answers with a name that is neither its donor's nor the
    expected answer of any event)."""
    def answer(req, events):
        event = events[req.metadata["event_id"]]
        if event.control:
            return f"It is not {OTHER}, it is Zed Quux."
        given, surname = name_parts(event)
        return template.format(Y=OTHER, given=given, surname=surname, X=event.expected_current)
    return answer


def scope_of(events, rows, answer) -> dict[str, bool]:
    """Per planned id: whether the grading judges decide its answer with judges (the rules unsure, or a rule grade
    incorrect or abstain of an answer that mentions the expected answer)."""
    out = {}
    for p in rows:
        event = events[p["event_id"]]
        text = answer(p, event)
        out[p["id"]] = run_mod._rule_grade(event, text, judges=True) is None
    return out


def planned_answer(template):
    def answer(p, event):
        return answering(template)(type("R", (), {"metadata": {"event_id": p["event_id"]}})(), {event.id: event})
    return answer


# ---- A: the grading scope ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("template", SCOPE)
def test_with_judges_a_rule_grade_against_the_named_expected_answer_goes_to_the_grading_judges(tmp_path, template):
    events, rows = fixture_plan()
    reply = answering(template)
    backend = judged(lambda req: reply(req, events), grade=CORRECT, votes={"j1": "ENDORSE"})
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1])
    scoped = scope_of(events, rows, planned_answer(template))
    graded = 0
    for row in got.rows:
        event = events[row["event_id"]]
        ruled = grade_rule(event, "current_d", row["response"])
        named = text_mod.mentions_answer(row["response"], event)
        if ruled is not None and ruled[0] in ("incorrect", "abstain") and named:
            assert scoped[row["id"]] and row["grader"] == "judge:j1", row     # every arm, every set
            graded += 1
        if not named:
            assert row["grader"] == "rule" or ruled is None, row             # no name: the rules (or unsure)
    assert len(calls(backend, "judge")) == sum(scoped.values()) >= graded
    # the right name of a full row the grading judges graded correct goes to the adoption judges as before
    full = [r for r in got.rows
            if r["arm"] == "full" and r["set"] in ("dark", "both_right") and r["grade"] == "correct"]
    assert all(r["adoption_grader"] == "judge:j1" for r in full)
    if template.startswith(("It is not {Y}, it is {given}", "It's not", "I think")):  # every real event's rows
        assert graded == sum(1 for p in rows if not events[p["event_id"]].control) == 28


@pytest.mark.parametrize("template", SCOPE)
def test_without_judges_the_grading_step_is_unchanged(tmp_path, template):
    events, rows = fixture_plan()
    reply = answering(template)
    backend = judged(lambda req: reply(req, events))
    got = run(backend, rows, tmp_path / "keys.jsonl", events)
    assert calls(backend, "judge") == [] and calls(backend, "adoption_judge") == []
    planned = {p["id"]: p for p in rows}
    for row in got.rows:
        event = events[row["event_id"]]
        ruled = run_mod._rule_grade(event, row["response"])                    # without judges
        grade, _, grader = run_mod._graded(planned[row["id"]], row)
        assert (grade, grader) == (("ungraded", "rule") if ruled is None else (ruled[0], "rule"))


@pytest.mark.parametrize("template", ["It is {X}.", "The first author is {X}, with {Y}.", "I don't know.",
                                      "It is not {Y}, it is Zed Quux.", "{Y}."])
def test_a_rule_correct_an_answer_without_the_name_and_a_control_keep_the_rule_grade(tmp_path, template):
    events, rows = fixture_plan()
    reply = answering(template)
    backend = judged(lambda req: reply(req, events), grade=INCORRECT, votes={"j1": "ENDORSE"})
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1])
    assert calls(backend, "judge") == []                                     # no grading judge for any of them
    assert all(row["grader"] == "rule" for row in got.rows)
    control = [r for r in got.rows if events[r["event_id"]].control]
    assert control and not any(text_mod.mentions_answer(r["response"], events[r["event_id"]]) for r in control)


@pytest.mark.parametrize("judges, votes", [
    ([J1], {"j1": BackendError("upstream 500", status=500)}),                 # a failed call
    ([J1], {"j1": "no idea"}),                                               # an unparseable reply
    ([J1, J2], {"j1": CORRECT, "j2": INCORRECT}),                            # a tie
])
def test_a_failed_or_tied_grading_vote_leaves_the_row_ungraded_never_the_rule_grade(tmp_path, judges, votes):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark"]
    reply = answering("It is not {Y}, it is {given}.")

    def reply_to(model, req):
        if req.metadata.get("purpose") == "judge":
            vote = votes[model.key]
            if isinstance(vote, BaseException):
                return vote
            return ChatResult(text=vote, usage={"cost": 0.0}, finish_reason="stop")
        return reply(req, events)
    got = run(ScriptedBackend(reply_to), rows, tmp_path / "keys.jsonl", events, judges=judges)
    assert all(grade_rule(events[r["event_id"]], "current_d", r["response"])[0] == "incorrect" for r in got.rows)
    keys = "+".join(j.key for j in judges)
    assert {(r["grade"], r["grader"]) for r in got.rows} == {("ungraded", f"judge:{keys}")}


def test_a_stored_rule_grade_now_in_the_grading_scope_is_judged_without_a_recipient_call(tmp_path):
    events, rows = fixture_plan()
    reply = answering("It is not {Y}, it is {given}.")
    out = tmp_path / "keys.jsonl"
    first = run(judged(lambda req: reply(req, events)), rows, out, events)    # without judges: incorrect by rule
    in_scope = [r for r in first.rows if r["grader"] == "rule" and r["grade"] == "incorrect"
                and text_mod.mentions_answer(r["response"], events[r["event_id"]])]
    assert len(in_scope) == 28
    backend = judged(lambda req: reply(req, events), grade=CORRECT, votes={"j1": "ENDORSE"})
    seen = []
    second = run(backend, rows, out, events, judges=[J1], on_start=seen.append)
    assert calls(backend) == [] and second.sent == 0                         # no recipient call
    assert len(calls(backend, "judge")) == len(in_scope) == second.regraded == seen[0].regrade
    assert seen[0].rule_grade == seen[0].relabel == seen[0].offer_flag == seen[0].failed == 0
    assert "28 for the judges" in _keys_run_start(seen[0], None)
    last = {r["id"]: r for r in second.rows}
    assert all(last[r["id"]]["grader"] == "judge:j1" and last[r["id"]]["grade"] == "correct" for r in in_scope)
    full = [last[r["id"]] for r in in_scope if r["arm"] == "full" and r["set"] in ("dark", "both_right")]
    assert full and all(r["adoption_grader"] == "judge:j1" for r in full)   # labelled after the re-grade
    assert len(calls(backend, "adoption_judge")) == len(full)
    for judges in ([J1], None):                                              # a grade of the judges stands
        again = judged(lambda req: reply(req, events), grade=INCORRECT, votes={"j1": "REJECT"})
        assert run(again, rows, out, events, judges=judges).regraded == 0 and again.calls == []


def test_a_stored_rule_grade_the_rules_give_otherwise_now_is_graded_again_without_a_call(tmp_path):
    # a row written before the rules doubted their incorrect next to the expected answer (as 863ca47 wrote it): final
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark"]
    reply = answering("It is not {Y}, it is {X}.")
    out = tmp_path / "keys.jsonl"
    run(judged(lambda req: reply(req, events)), rows, out, events)
    old = [{**r, "grade": "incorrect", "hedged": False} for r in read_jsonl(out)]
    write_jsonl(out, old)
    backend = judged(lambda req: reply(req, events))
    seen = []
    got = run(backend, rows, out, events, on_start=seen.append)
    assert backend.calls == [] and seen[0].rule_grade == got.regraded == len(rows)
    assert "20 whose rule grade is out of date" in _keys_run_start(seen[0], None)
    assert {(r["grade"], r["grader"], r["offers_expected"]) for r in got.rows} == {("ungraded", "rule", True)}
    assert run(judged(lambda req: reply(req, events)), rows, out, events).regraded == 0    # idempotent


def test_the_cap_and_the_limit_count_the_grading_scope(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark"]
    reply = answering("It is not {Y}, it is {given}.")
    backend = judged(lambda req: reply(req, events), grade=CORRECT, votes={"j1": "ENDORSE"})
    got = run(backend, rows, tmp_path / "capped.jsonl", events, judges=[J1], concurrency=1, max_cost=0.05)
    assert got.stopped_by == "max_cost" and got.sent == 2                     # 0.01 + 0.02 per item: stops at 0.06
    assert got.judges_usd == pytest.approx(0.04) and all(
        [e["purpose"] for e in r["judge_usage"]] == ["judge"] for r in got.rows if r["arm"] != "full")
    limited = judged(lambda req: reply(req, events), grade=CORRECT, votes={"j1": "ENDORSE"})
    got = run(limited, rows, tmp_path / "limited.jsonl", events, judges=[J1], limit=3)
    assert got.stopped_by == "limit" and got.sent == 3 == len(calls(limited, "judge"))


@pytest.mark.parametrize("status", [401, 402])
def test_an_error_no_retry_can_fix_from_a_grading_judge_in_the_scope_stops_the_run(tmp_path, status):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark"]
    reply = answering("It is not {Y}, it is {given}.")

    def reply_to(model, req):
        if req.metadata.get("purpose"):
            return BackendError(f"OpenRouter HTTP {status} for judge/j1: nope", status=status)
        return reply(req, events)
    backend = ScriptedBackend(reply_to)
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1], concurrency=1)
    assert got.stopped_by == "error" and f"[{status}]" in str(got.error)
    assert got.sent == 1 and got.not_sent == len(rows) - 1
    assert got.rows[0]["grade"] == "ungraded" and got.rows[0]["grader"] == "judge:j1"
    # and on resume: the stored rule grades in the scope are re-graded, and the same error stops the re-grades
    out = tmp_path / "resumed.jsonl"
    run(judged(lambda req: reply(req, events)), rows, out, events)
    again = run(ScriptedBackend(reply_to), rows, out, events, judges=[J1], concurrency=2)
    assert again.stopped_by == "error" and again.regraded == 2 and again.not_regraded == len(rows) - 2


@pytest.fixture
def gateway(monkeypatch):
    """``make_backend`` without the network: a run of the real models gets ``made.backend`` (a dummy key is set)."""
    import cutoffmix.backends as mod

    made = SimpleNamespace(backend=ScriptedBackend())
    monkeypatch.setattr(mod, "make_backend", lambda cfg, **kw: made.backend)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    return made


def test_the_cli_grades_the_stored_scope_first_and_stops_at_the_cap_or_a_401(tmp_path, capsys, gateway):
    events = load_events(EVENTS)
    by_id = {e.id: e for e in events}
    plan_rows, _ = build_plan(events, load_results(RESULTS), pairs=[(FABLE, NEON)], today=TODAY)
    rows = [r for r in plan_rows if r["set"] == "dark"][:10]
    plan = tmp_path / "plan.jsonl"
    write_jsonl(plan, rows)

    def backend(judge_error=None):
        def reply(model, req):
            purpose = req.metadata.get("purpose")
            if purpose and judge_error is not None:
                return judge_error
            if purpose:
                text = CORRECT if purpose == "judge" else "ENDORSE"
                return ChatResult(text=text, usage={"cost": 0.01}, finish_reason="stop")
            given = by_id[req.metadata["event_id"]].expected_current.rsplit(" ", 1)[0]
            return f"It is not {OTHER}, it is {given}."
        return ScriptedBackend(reply)

    def cli(out, *extra):
        return main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out), *extra])
    out = tmp_path / "keys_results.jsonl"
    gateway.backend = backend()
    assert cli(out, "--max-cost", "5") == 0                                  # the rules alone: incorrect, final
    capsys.readouterr()
    gateway.backend = backend()
    assert cli(out, "--max-cost", "0.115", "--judges", "gpt-6.2-sol", "--concurrency", "1") == 3
    got = capsys.readouterr()
    assert "grading 10 stored answers again first, without a new call (10 for the judges)" in got.out
    assert got.err.endswith("stopped at --max-cost $0.115: 0 of 10 planned calls not sent, 8 stored answers not "
                            "graded again; rerun with a higher --max-cost to resume\n")
    assert [req.metadata.get("purpose") for _, req in gateway.backend.calls] == ["judge", "judge"]
    gateway.backend = backend()
    assert cli(out, "--max-cost", "5", "--judges", "gpt-6.2-sol") == 0       # the rest, then nothing
    assert all(req.metadata.get("purpose") for _, req in gateway.backend.calls)
    gateway.backend = backend()
    assert cli(out, "--max-cost", "5", "--judges", "gpt-6.2-sol") == 0 and gateway.backend.calls == []
    # a 401 from a grading judge in the scope stops the re-grades: exit status 4
    fresh = tmp_path / "fresh.jsonl"
    gateway.backend = backend()
    assert cli(fresh, "--max-cost", "5") == 0
    capsys.readouterr()
    gateway.backend = backend(BackendError("OpenRouter HTTP 401 for openai/gpt-6.2-sol: No auth credentials found",
                                           status=401))
    assert cli(fresh, "--max-cost", "5", "--judges", "gpt-6.2-sol", "--concurrency", "2") == 4
    err = capsys.readouterr().err
    assert "stopped after an error that no retry can fix (BackendError: [401] OpenRouter HTTP 401 for " \
        "openai/gpt-6.2-sol: No auth credentials found): 0 of 10 planned calls not sent, 8 stored answers not graded " \
        "again" in err


# ---- B: offers_expected on every answered row ------------------------------------------------------------------------


def test_every_answered_row_stores_whether_it_offers_the_expected_answer(tmp_path):
    events, rows = fixture_plan()

    def reply(model, req):
        meta = req.metadata
        event = events[meta["event_id"]]
        if meta["arm"] == "blank":
            return ChatResult(text="", finish_reason="content_filter", usage={"cost": 0.0})
        if meta["arm"] == "wrong_key":
            return RuntimeError("upstream 503")
        if event.control or meta["arm"] == "alone":
            return "I don't know."
        return f"It is not {OTHER}, it is {event.expected_current}."
    got = run(ScriptedBackend(reply), rows, tmp_path / "keys.jsonl", events)
    for row in got.rows:
        event = events[row["event_id"]]
        if row["error"] or row["grader"] == "refusal" or event.control:
            assert row["offers_expected"] is None, row
        else:
            assert row["offers_expected"] is (row["arm"] != "alone"), row
    assert {r["grade"] for r in got.rows if r["offers_expected"]} == {"ungraded"}    # open offers without judges


def test_rows_stored_without_offers_expected_get_it_without_a_call(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"

    def reply(model, req):
        if req.metadata["arm"] == "blank":
            return ChatResult(text="", finish_reason="content_filter", usage={"cost": 0.0})
        return f"It might be {events[req.metadata['event_id']].expected_current or 'Made Upname'}."
    run(ScriptedBackend(reply), rows, out, events)
    old = []
    for row in read_jsonl(out):
        row.pop("offers_expected")
        old.append(row)
    write_jsonl(out, old)
    backend = ScriptedBackend(reply)
    seen = []
    got = run(backend, rows, out, events, on_start=seen.append)
    refusals = [r for r in old if r["grader"] == "refusal"]
    assert backend.calls == [] and got.sent == 0 and refusals
    assert seen[0].offer_flag == got.regraded == len(rows) - len(refusals)    # refusals are not written again
    assert f"{got.regraded} whose offers_expected is missing or out of date" in _keys_run_start(seen[0], None)
    last = {r["id"]: r for r in read_jsonl(out)}
    assert all("offers_expected" not in last[r["id"]] for r in refusals)
    assert all(last[r["id"]]["offers_expected"] is (None if events[r["event_id"]].control else True) for r in old
               if r["grader"] != "refusal")
    assert run(ScriptedBackend(reply), rows, out, events).regraded == 0      # filled once


# ---- B: the report's check of the open offers ----------------------------------------------------------------------

OPEN = {"grade": "ungraded", "offers_expected": True}                   # an open offer
SHUT = {"grade": "abstain", "offers_expected": False}                   # a miss
HIT = {"grade": "correct", "offers_expected": True}


def flagged(study: Study, offered: bool = False) -> Study:
    """``study`` with ``offers_expected`` on every row that lacks it (False, or True for correct rows when
    ``offered``), as a run of this version writes them."""
    for row in study.results:
        row.setdefault("offers_expected", offered and row["grade"] == "correct")
    return study


def dark(n: int, outcome) -> Study:
    """``n`` dark items of one pair; ``outcome(i, arm)`` gives each row (True correct, False a miss, or fields)."""
    study = Study()
    for i in range(n):
        study.item("dark", f"d{i}", {arm: outcome(i, arm) for arm in ("alone", "key", "full", "wrong_key", "blank")})
    return study


def verdicts(study: Study) -> dict[Any, str]:
    return {p["id"]: p["verdict"] for p in study.report()["predictions"]}


def without_offers(study: Study) -> Study:
    """The same rows with every open offer read as a plain miss (as 8cc83c5 read every ungraded row)."""
    copy = Study()
    copy.plan = study.plan
    copy.results = [{**r, "offers_expected": False} for r in study.results]
    return copy


def test_a_support_of_1_that_rests_on_alones_open_offers_is_not_testable():
    # the key recovers 20 of 40 items; alone's answers on those 20 offer the expected answer and are left open
    study = flagged(dark(40, lambda i, arm: {"key": i < 20, "alone": OPEN if i < 20 else False}.get(
        arm, arm == "full")))
    v = verdicts(study)
    assert verdicts(without_offers(study))[1] == "supported"                 # 8cc83c5: open offers count as misses
    assert v[1] == UNDECIDED and v[2] == "not testable (prediction 1 is not testable)"
    assert v["1b"] == "supported"                                            # blank holds no open offer
    one = study.report()["predictions"][0]
    assert one["comparisons"]["key_vs_alone"]["checked"] == UNDECIDED
    assert one["comparisons"]["key_vs_wrong_key"]["checked"] == "supported"
    assert "key vs alone n=40, b=20, c=0, p=1.907e-06 [supported; Wilson upper bound of b/n 0.648] -> not testable " \
        "(undecided answers; grade with --judges)" in one["evidence"]
    assert "open offers (ungraded answers that offer the expected answer): key 0, alone 20, wrong key 0, blank 0" \
        in one["evidence"]
    assert "key vs alone with alone's counted correct b=0, c=0, with key's b=20, c=0" in one["evidence"]
    # two open offers where the key is wrong cannot overturn it: the verdict of 8cc83c5
    kept = flagged(dark(40, lambda i, arm: {"key": i < 20, "alone": OPEN if i in (30, 31) else False}.get(
        arm, arm == "full")))
    assert verdicts(kept)[1] == verdicts(without_offers(kept))[1] == "supported"


def test_a_refutation_of_1_and_1b_that_rests_on_the_keys_open_offers_is_not_testable():
    study = flagged(dark(96, lambda i, arm: OPEN if arm == "key" and i < 10 else arm == "full"))
    assert verdicts(without_offers(study))[1] == verdicts(without_offers(study))["1b"] == "not supported"
    v = verdicts(study)
    assert v[1] == v["1b"] == UNDECIDED
    one = study.report()["predictions"][0]
    assert one["comparisons"]["key_vs_alone"]["verdict"] == "refuted"         # the three-way verdict, as before
    assert one["comparisons"]["key_vs_alone"]["against"]["refute"]["b"] == 10
    # one open offer of 96 cannot: b = 1, Wilson upper bound 0.057
    kept = flagged(dark(96, lambda i, arm: OPEN if arm == "key" and i == 0 else arm == "full"))
    assert verdicts(kept)[1] == verdicts(kept)["1b"] == "not supported"


def test_an_inconclusive_verdict_stays_inconclusive_beside_open_offers():
    study = flagged(dark(20, lambda i, arm: {"key": i < 3, "alone": OPEN if i == 5 else False,
                                             "blank": OPEN if i == 6 else False}.get(arm, arm == "full")))
    v = verdicts(study)
    assert v[1] == v["1b"] == "inconclusive (20 items; about 71 would be needed for a gain of 0.10)"
    assert v[1] == verdicts(without_offers(study))[1]


def test_the_confound_check_is_resolved_against_the_support_too():
    # unconfounded on the rows, but blank's 12 open offers counted correct would make blank beat the wrong key
    plain = flagged(dark(40, lambda i, arm: {"key": True, "blank": OPEN if i < 12 else False}.get(arm, arm == "full")))
    one = plain.report()["predictions"][0]
    assert one["confounded"] is False and verdicts(without_offers(plain))[1] == "supported"
    assert one["verdict"] == UNDECIDED
    assert "blank vs wrong key with wrong key's counted correct b=0, c=0, with blank's b=12, c=0" in one["evidence"]
    # confounded on the rows, but the wrong key's 12 open offers counted correct would undo it; key vs wrong key
    # itself stays supported (b = 28)
    confounded = flagged(dark(40, lambda i, arm: {"key": True, "blank": i < 12, "wrong_key": OPEN if i < 12 else False}
                              .get(arm, arm == "full")))
    one = confounded.report()["predictions"][0]
    assert one["confounded"] is True and one["comparisons"]["key_vs_wrong_key"]["checked"] == "supported"
    assert verdicts(without_offers(confounded))[1] == "supported (confounded: the wrong key misleads)"
    assert one["verdict"] == UNDECIDED
    # too few open offers to change either: the verdicts of 8cc83c5
    for study in (flagged(dark(40, lambda i, arm: {"key": True, "blank": OPEN if i < 3 else False}.get(
                      arm, arm == "full"))),
                  flagged(dark(40, lambda i, arm: {"key": True, "blank": i < 12, "wrong_key": OPEN if i < 2 else False}
                               .get(arm, arm == "full")))):
        assert verdicts(study)[1] == verdicts(without_offers(study))[1] != UNDECIDED


def p2_study(open_unit: bool) -> Study:
    """The complete study that supports every prediction (test_keys_report_rules), with one unit's alone row an open
    offer (an unrecognised item where the key is not correct either) when ``open_unit``."""
    alone = {0, 1, 30}
    study = Study().dark(60, correct={"key": alone | set(range(2, 20)) | {31}, "alone": alone, "blank": {0}},
                         recognised=lambda i: i < 30)
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent")
    study.both_right(20, wrong={"alone": (3,), "key": (7,)})
    flagged(study)
    if open_unit:
        row = next(r for r in study.results if r["set"] == "dark" and r["event_id"] == "d40" and r["arm"] == "alone")
        row.update(OPEN)
    return study


def test_a_support_of_2_stands_only_without_an_open_offer_in_its_units():
    assert verdicts(p2_study(False)) == {1: "supported", "1b": "supported", 2: "supported", 3: "supported",
                                         4: "supported"}
    study = p2_study(True)
    v = verdicts(study)
    assert v[1] == "supported" and v[2] == UNDECIDED                          # prediction 1 survives the open offer
    assert verdicts(without_offers(study))[2] == "supported"
    two = study.report()["predictions"][2]
    assert two["units_with_open_offers"] == 1 and two["open_offers"]["alone"]["open_offers"] == 1
    assert "in the units the test reads, open offers (ungraded answers that offer the expected answer): key 0, " \
        "alone 1, wrong key 0 (1 units hold one)" in two["evidence"]


def test_a_support_of_3a_that_rests_on_the_keys_open_offers_is_not_testable():
    # the check's case: full copies the note, the key's answers correct another name and give the right one, which the
    # rules leave open without judges; 3a read supported (b = 40), the judged run reads refuted
    study = flagged(dark(40, lambda i, arm: {"full": True, "key": OPEN}.get(arm, False)))
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent")
    flagged(study)
    half = study.report()["predictions"][3]["halves"]["3a"]
    assert half["comparison"]["verdict"] == "supported" and half["undecided"]["undecided"] == 0
    assert (half["against"]["support"]["b"], half["against"]["support"]["c"]) == (0, 0)
    assert half["verdict"] == UNDECIDED
    assert without_offers(study).report()["predictions"][3]["halves"]["3a"]["verdict"] == "supported"
    assert "3a, open offers (ungraded answers that offer the expected answer): full 0, key 40; key's counted correct " \
        "b=0, c=0" in study.report()["predictions"][3]["evidence"]
    # two of them cannot overturn it
    kept = flagged(dark(40, lambda i, arm: {"full": True, "key": OPEN if i < 2 else False}.get(arm, False)))
    kept.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent")
    flagged(kept)
    assert kept.report()["predictions"][3]["halves"]["3a"]["verdict"] == "supported"


def harm(outcome) -> Study:
    study = Study()
    for i in range(30):
        study.item("both_right", f"r{i}", {arm: outcome(i, arm) for arm in ("alone", "key", "full")}, recognised=True)
    return flagged(study)


def test_a_verdict_of_4_stands_only_against_the_open_offers():
    # re-asking alone leaves 5 open offers, the key misses 5 others: excess harm 0, but with alone's open offers
    # counted correct it is 5/30
    supported = harm(lambda i, arm: {"alone": OPEN if i < 5 else True, "key": not 5 <= i < 10}.get(arm, True))
    four = supported.report()["predictions"][4]
    assert verdicts(without_offers(supported))[4] == "supported" and four["verdict"] == UNDECIDED
    assert "resolved against the verdict: alone's counted correct b=5, c=0, excess harm 16.7 %" in four["evidence"]
    # 5 open offers of the key make an excess harm of 5/30 that the key's answers may not cause
    against = harm(lambda i, arm: {"key": OPEN if i < 5 else True}.get(arm, True))
    assert verdicts(without_offers(against))[4] == "not supported" and verdicts(against)[4] == UNDECIDED
    # where they cannot change it: 9 misses of the key, 4 of them open (5/30 is still 10 % or more)
    kept = harm(lambda i, arm: {"key": OPEN if i < 4 else not i < 9}.get(arm, True))
    assert verdicts(kept)[4] == verdicts(without_offers(kept))[4] == "not supported"
    assert "open offers (ungraded answers that offer the expected answer): key 4, alone 0" in \
        kept.report()["predictions"][4]["evidence"]


def test_a_results_file_without_offers_expected_reads_as_before():
    # rows written before the field: counted apart and read as not offering, so the verdicts are those of 8cc83c5
    study = dark(40, lambda i, arm: {"key": i < 20, "alone": {"grade": "ungraded"} if i < 20 else False}.get(
        arm, arm == "full"))
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent")
    report = study.report()
    assert report["totals"]["offer_unknown"] == report["totals"]["answered"] == 260
    assert report["totals"]["open_offers"] == 0
    assert any("260 answered rows carry no offers_expected (written before it was stored)" in f
               for f in report["flags"])
    assert {p["id"]: p["verdict"] for p in report["predictions"]} == verdicts(without_offers(study))
    assert verdicts(study)[1] == "supported"
    one = report["predictions"][0]
    assert "(rows without offers_expected, read as not offering: key 40, alone 40, wrong key 40, blank 40)" in \
        one["evidence"]
    md = render_report(report)
    assert "A verdict stands only if the open offers (ungraded answers that offer the expected answer) and the " \
        "undecided rows, resolved against it, leave it as it is." in md


def test_open_offers_are_counted_in_the_totals_the_arms_and_a_flag():
    study = flagged(dark(40, lambda i, arm: {"key": i < 20, "alone": OPEN if i < 20 else False}.get(
        arm, arm == "full")))
    report = study.report()
    assert report["totals"]["open_offers"] == 20 and report["totals"]["offer_unknown"] == 0
    arms = report["pairs"]["fable->sol"]["dark"]["arms"]
    assert arms["alone"]["open_offers"] == 20 and arms["key"]["open_offers"] == 0
    assert any(f.startswith("20 open offers (ungraded answers that offer the expected answer)")
               for f in report["flags"])
    json.dumps(report)                                                        # still JSON-serialisable


# ---- B: no verdict stands that a resolution of the open offers would overturn (I3) --------------------------------

STANDING = ("supported", "not supported", "supported (confounded: the wrong key misleads)")


def random_study(rng: random.Random) -> Study:
    """Dark items of one pair (some recognised), both_right items and an adoption set, each row correct, a miss, an
    open offer or ungraded without an offer, at random."""
    n = rng.randint(12, 60)
    p = {arm: (rng.random() * 0.6, rng.random() * 0.25) for arm in ("alone", "key", "full", "wrong_key", "blank")}

    def outcome(i, arm):
        hit, offer = p[arm]
        x = rng.random()
        return True if x < hit else OPEN if x < hit + offer else ({"grade": "ungraded"} if rng.random() < 0.1
                                                                   else False)
    study = Study()
    for i in range(n):
        study.item("dark", f"d{i}", {arm: outcome(i, arm) for arm in p}, recognised=rng.random() < 0.6)
    for i in range(rng.randint(8, 30)):
        study.item("both_right", f"r{i}", {arm: outcome(i, arm) if arm != "full" else True
                                           for arm in ("alone", "key", "full")}, recognised=True)
    study.adoption(26, full=lambda i: "endorsed" if i < 12 else "absent")
    return flagged(study)


def resolved(study: Study, rng: random.Random, how: str) -> Study:
    """Every open offer decided: correct (``all``), a miss (``none``) or either at random."""
    copy = Study()
    copy.plan = study.plan
    copy.results = []
    for row in study.results:
        if row.get("grade") == "ungraded" and row.get("offers_expected") is True:
            right = {"all": True, "none": False}.get(how, rng.random() < 0.5)
            row = {**row, "grade": "correct" if right else "incorrect"}
        copy.results.append(dict(row))
    return copy


def test_no_standing_verdict_is_overturned_by_any_resolution_of_the_open_offers():
    rng = random.Random(8)
    checked = 0
    for _ in range(40):
        study = random_study(rng)
        report = study.report()
        stood = {p["id"]: p["verdict"] for p in report["predictions"]}
        stood["3a"] = report["predictions"][3]["halves"]["3a"]["verdict"]
        for how in ("all", "none", "random", "random", "random"):
            other = resolved(study, rng, how).report()
            got = {p["id"]: p["verdict"] for p in other["predictions"]}
            got["3a"] = other["predictions"][3]["halves"]["3a"]["verdict"]
            for key in (1, "1b", "3a", 4):
                if stood[key] in STANDING:
                    assert got[key] == stood[key], (key, how, stood[key], got[key])
                    checked += 1
            if stood[2] == "supported":
                assert got[2] == "supported"
    assert checked > 100
