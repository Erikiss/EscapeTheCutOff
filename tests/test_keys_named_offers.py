"""Key, not value (docs/DESIGN_keys.md sections 6, 7 and 10): the review of b86d67f.

- An open offer in any form: ``offers_expected`` holds where the answer offers the expected answer or names it in any
  other form other than inside another person's name (``text.names_answer``), so the open offers cover the grading
  scope. A grading vote the judges lose (a failed call, a tie, a reply without a verdict) on an answer that names the
  expected answer leaves an open offer, and so, without judges, does an answer the rules cannot grade that names it
  ("Zhang et al.", "X?"); the review's scenarios read not testable, on the real plan through the real run_plan and
  build_report. An answer that names only another person who shares a name part (a real miss) is none.
- Answers the judges left undecided: the run counts them (``RunOutcome.judges_undecided``) and the CLI prints them,
  also for a reply without a verdict, which no failed call records; the report counts and flags them.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from types import SimpleNamespace

import pytest
from keys_testlib import EVENTS, FABLE, NEON, RESULTS, SOL, TODAY
from test_keys_contract import DESIGN, readme
from test_keys_report import Study
from test_keys_run import ScriptedBackend
from test_keys_run_judges import J1, J2, fixture_plan, judged

from cutoffmix.backends.base import BackendError
from cutoffmix.bench.grading import _COMMON_SURNAMES, grade_rule, normalize_text
from cutoffmix.bench.io import load_events, load_results, read_jsonl, write_jsonl
from cutoffmix.bench.keys import build_plan, build_report, run_plan
from cutoffmix.bench.keys import run as run_mod
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.cli import main
from cutoffmix.types import ChatResult, ModelSpec

Y = "Sanjay Ramaswamy"                                                  # another person, unknown to every event
CORRECT = '{"label": "correct", "hedged": false}'
UNDECIDED = "not testable (undecided answers; grade with --judges)"
RECIPIENTS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in (SOL, NEON, FABLE)]
DAYA = "paper:2401.14196"                                               # Daya Guo: a both_right item of each pair


@pytest.fixture(scope="module")
def real():
    events = load_events(EVENTS)
    plan, _ = build_plan(events, load_results(RESULTS), today=TODAY)
    return SimpleNamespace(events={e.id: e for e in events}, plan=plan)


def real_events(real) -> list:
    """The events of the real plan, in order of their ids."""
    return [real.events[e] for e in sorted({p["event_id"] for p in real.plan})]


def parts(event) -> tuple[str, str, str]:
    x = event.expected_current
    return x, x.split()[-1], " ".join(x.split()[:-1])


def common(event) -> bool:
    """The paper rules accept the surname alone only when it is distinctive: more than four letters, not common."""
    surname = normalize_text(event.expected_current.split()[-1])
    return len(surname) <= 4 or surname in _COMMON_SURNAMES


def answer_of(req) -> str:
    return req.messages[-1]["content"].split("MODEL ANSWER:\n", 1)[-1]


def run_real(real, rows, answer, tmp_path, *, grading=None, adoption="ENDORSE", judges=True):
    """The real run_plan on ``rows`` of the real plan: the recipient answers ``answer(meta, event)``; with ``judges``
    one grading judge answers ``grading(req)`` (a text, or an exception to raise: a failed call) and the adoption judge
    ``adoption``."""
    def reply(model, req):
        purpose = req.metadata.get("purpose")
        if purpose == "judge":
            out = grading(req) if grading is not None else CORRECT
            return out if isinstance(out, BaseException) else ChatResult(text=out, usage={"cost": 0.0},
                                                                         finish_reason="stop")
        if purpose == "adoption_judge":
            return ChatResult(text=adoption, usage={"cost": 0.0}, finish_reason="stop")
        meta = req.metadata
        return answer(meta, real.events[meta["event_id"]])
    backend = ScriptedBackend(reply, cost=0.0)
    got = asyncio.run(run_plan(backend, RECIPIENTS, rows, tmp_path / "keys.jsonl", events=real.events,
                               judges=[J1] if judges else None, concurrency=16, backend_name="scripted"))
    return got, backend


def verdicts(rows, results) -> dict[str, str]:
    report = build_report(rows, results)
    out = {str(p["id"]): p["verdict"] for p in report["predictions"]}
    out["3a"] = report["predictions"][3]["halves"]["3a"]["verdict"]
    return out


def narrow(real, results):
    """The rows as b86d67f stored them: ``offers_expected`` by ``offers_answer`` alone (its open offers)."""
    out = []
    for r in results:
        event = real.events[r["event_id"]]
        if r.get("offers_expected") is not None:
            r = {**r, "offers_expected": text_mod.offers_answer(r["response"], event)}
        out.append(r)
    return out


def judges_left_open(row) -> bool:
    """``common.judges_left_open``, looked up at call time (this file also runs against b86d67f, which lacks it)."""
    from cutoffmix.bench.keys import common
    return common.judges_left_open(row)


def design_row(title: str) -> str:
    text = DESIGN.read_text("utf-8")
    table = text[text.index("### Refinements during the implementation review"):]
    return next(line for line in table.splitlines() if line.startswith(f"| {title} |"))


# ---- which answers name the expected answer -------------------------------------------------------------------------

NAMES_IT = ["It is not {Y}, it is Dr. {S}.", "The first author is {S} et al., not {Y}.", "It is not {Y}, it is {G}.",
            "The first author is {S}.", "I'm not sure. {X}?", "I think it's {G}, but I'm not sure.",
            "The first author is Dr. {S}, with Bo Li as senior author.", "**{S}** et al., I think.",
            "The first author is {X}, with Yutong {S} as senior author.", "It is not {X}.", "I doubt it is {X}.",
            "It is not {X}, it is {Y}."]
ANOTHER_ONLY = ["The first author is Yutong {S}.", "I'm not certain, but I believe the first author is Yutong {S}.",
                "I'm not sure. Yutong {S}?", "It is not Yutong {S}.", "The first author is **Yutong {S}**.",
                "It came from Yutong {S}'s group."]


@pytest.mark.parametrize("template", NAMES_IT)
def test_every_form_that_names_the_expected_answer_offers_it(real, template):
    events = [e for e in real_events(real) if not e.control]
    assert len(events) == 107
    for event in events:
        x, s, g = parts(event)
        text = template.format(X=x, S=s, G=g, Y=Y)
        assert text_mod.names_answer(text, event) and run_mod._offers_expected(event, text), (event.id, text)


@pytest.mark.parametrize("template", ANOTHER_ONLY)
def test_a_mention_only_inside_another_persons_name_is_no_offer(real, template):
    # the replay's false mentions ("Yutong Mei" for Lingrui Mei); where the shared surname is distinctive the paper
    # rules accept it alone (offers_answer, as before), so only the common surnames are checked
    events = [e for e in real_events(real) if not e.control and common(e)]
    assert len(events) == 81
    for event in events:
        x, s, g = parts(event)
        text = template.format(X=x, S=s, G=g)
        assert text_mod.mentions_answer(text, event) and not text_mod.names_answer(text, event), (event.id, text)
        assert run_mod._offers_expected(event, text) is False


@pytest.mark.parametrize("event_id, text, named", [
    ("paper:2507.03724", "The first author is Zhaoyang Li.", False),       # agrees with the form "Z. Li" of Zhiyu Li
    ("paper:2507.03724", "The first author is Z. Li.", True),
    ("paper:2607.14935", "Kunchang Li.", False),                           # the donor's name beside Xinhao Li
    ("paper:2404.08636", "It is not Sanjay Ramaswamy, it is Mohamed El.", True),   # a part of Mohamed El Banani
    ("paper:2401.02385", "The first author is Zhang Peiyuan.", True),      # surname first
    ("paper:2507.13334", "It is not Longwen Zhang, it is Mei.", True),     # a distractor blanked, Mei stays
    ("paper:2507.13334", "It is Longwen Zhang.", False),
])
def test_the_names_of_other_people_are_blanked_but_not_the_expected_answers_own_forms(real, event_id, text, named):
    assert text_mod.names_answer(text, real.events[event_id]) is named


def test_the_stored_answers_the_rules_leave_undecided_name_the_expected_answer_only_inside_another_name(real):
    results = [r for r in load_results(RESULTS) if r.probe == "current_d"]
    undecided = [r for r in results if grade_rule(real.events[r.event_id], "current_d", r.response or "") is None]
    named = [r for r in undecided if run_mod._offers_expected(real.events[r.event_id], r.response or "")]
    mentioned = sorted(r.event_id for r in undecided if text_mod.mentions_answer(r.response or "",
                                                                                 real.events[r.event_id]))
    assert (len(undecided), len(named), mentioned) == (57, 0, ["paper:2507.03724", "paper:2607.14935"])
    six = " ".join(DESIGN.read_text("utf-8").split())
    assert "the two that mention it do so only inside another person's name, \"Zhaoyang Li\" for Zhiyu Li, a " \
        "different given name that the paper rules leave open by their test of initials, and \"KunChang Li\" for " \
        "Xinhao Li" in six                                     # check of 8739803: the paper rules leave it open


# ---- a lost grading vote on an answer that names the expected answer (review of b86d67f) --------------------------


@pytest.mark.parametrize("loss", ["failed call", "no verdict"])
def test_two_lost_votes_on_both_right_key_rows_cannot_decide_prediction_4(real, tmp_path, loss):
    rows = [p for p in real.plan if p["set"] == "both_right"]

    def answer(meta, event):
        x, _, g = parts(event)
        return f"It is not {Y}, it is {g}." if meta["arm"] == "key" else f"The first author is {x}."
    lost = []

    def grading(req):
        if answer_of(req) == f"It is not {Y}, it is Daya.":
            lost.append(req)
            return BackendError("upstream 500", status=500) if loss == "failed call" else "I need more context."
        return CORRECT
    got, _ = run_real(real, rows, answer, tmp_path, grading=grading)
    daya = [r for r in got.rows if r["event_id"] == DAYA and r["arm"] == "key"]
    assert len(lost) == len(daya) == 2
    assert {(r["grade"], r["grader"], r["offers_expected"]) for r in daya} == {("ungraded", "judge:j1", True)}
    assert got.judges_undecided == 2 and got.judge_failed == (2 if loss == "failed call" else 0)
    report = build_report(rows, got.rows)
    four = report["predictions"][4]
    assert four["verdict"] == UNDECIDED
    assert (four["comparison"]["b"], four["comparison"]["c"], four["items"]) == (2, 0, 20)
    assert verdicts(rows, narrow(real, got.rows))["4"] == "not supported"      # b86d67f: 2 of 20, no check, no flag
    assert any(f.startswith("2 open offers (ungraded answers that name the expected answer)") for f in report["flags"])
    assert any(f.startswith("2 answered rows are undecided after the judges' votes") for f in report["flags"])
    assert "2 lost votes of 20 on `both_right` key rows \"It is not Y, it is Daya.\" made 4 not supported" in \
        design_row("an open offer in any form")
    assert "Zwei ausgefallene Stimmen bei den 20 `key`-Antworten von `both_right` machten Vorhersage 4 zu „nicht " \
        "gestützt“" in readme()


def test_one_lost_vote_on_a_decisive_pair_key_row_cannot_refute_prediction_1(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "dark" and p["recipient"] == SOL]      # the decisive pair's
    decisive = sorted({p["event_id"] for p in rows if not p["pointer"]})
    three = decisive[:3]

    def answer(meta, event):
        if meta["arm"] == "key" and meta["model"] == SOL and event.id in three:
            return f"It is not {Y}, it is {parts(event)[2]}."
        return "I don't know."

    def grading(req):
        if answer_of(req) == f"It is not {Y}, it is {parts(real.events[three[0]])[2]}.":
            return BackendError("upstream 500", status=500)
        return CORRECT
    got, _ = run_real(real, rows, answer, tmp_path, grading=grading)
    assert got.judges_undecided == 1
    report = build_report(rows, got.rows)
    ka = report["predictions"][0]["comparisons"]["key_vs_alone"]
    assert (ka["n_items"], ka["b"], ka["against"]["refute"]["b"]) == (71, 2, 3)
    v = verdicts(rows, got.rows)
    assert v["1"] == v["1b"] == UNDECIDED
    old = verdicts(rows, narrow(real, got.rows))
    assert old["1"] == old["1b"] == "not supported"                           # b86d67f: refuted on b = 2 of 71
    assert "1 on a decisive-pair key row made 1 and 1b not supported (b 3 to 2 of 71)" in \
        design_row("an open offer in any form")


@pytest.mark.parametrize("other_person", [False, True])
def test_lost_votes_on_dr_surname_corrections_cannot_support_3a(real, tmp_path, other_person):
    rows = [p for p in real.plan if p["set"] == "dark"]
    candidates = sorted({p["event_id"] for p in rows if common(real.events[p["event_id"]])})
    chosen = set(candidates[:6])

    def answer(meta, event):
        x, s, _ = parts(event)
        if event.id in chosen and meta["arm"] == "key":
            return f"The first author is Yutong {s}." if other_person else f"It is not {Y}, it is Dr. {s}."
        if event.id in chosen and meta["arm"] == "full":
            return f"The first author is {x}."
        return "I don't know."
    got, _ = run_real(real, rows, answer, tmp_path, grading=lambda req: BackendError("down", status=500))
    key = [r for r in got.rows if r["arm"] == "key" and r["event_id"] in chosen]
    assert len(key) == 7 and {(r["grade"], r["grader"]) for r in key} == {("ungraded", "judge:j1")}
    assert got.judges_undecided == 7
    v = verdicts(rows, got.rows)
    if other_person:   # another person of the surname: a real miss, no open offer, and the support stands
        assert {r["offers_expected"] for r in key} == {False} and v["3a"] == "supported"
    else:
        assert {r["offers_expected"] for r in key} == {True} and v["3a"] == UNDECIDED
        assert verdicts(rows, narrow(real, got.rows))["3a"] == "supported"     # b86d67f: b = 7, c = 0, no flag
        assert f"{len(key)} on keyed-pair key rows \"It is not Y, it is Dr. Zhang.\" made 3a supported" in \
            design_row("an open offer in any form")


# ---- without judges: an answer the rules cannot grade that names the expected answer --------------------------------


def test_without_judges_et_al_key_rows_cannot_refute_prediction_1(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "dark" and p["recipient"] == SOL]      # the decisive pair's
    decisive = sorted({p["event_id"] for p in rows if not p["pointer"] and common(real.events[p["event_id"]])})
    chosen = set(decisive[:8])

    def answer(meta, event):
        if meta["arm"] == "key" and meta["model"] == SOL and event.id in chosen:
            return f"The first author is {parts(event)[1]} et al."
        return "I don't know."
    got, _ = run_real(real, rows, answer, tmp_path, judges=False)
    key = [r for r in got.rows if r["arm"] == "key" and r["recipient"] == SOL and r["event_id"] in chosen]
    assert len(key) == 8 and {(r["grade"], r["offers_expected"]) for r in key} == {("ungraded", True)}
    v = verdicts(rows, got.rows)
    assert v["1"] == v["1b"] == UNDECIDED
    old = verdicts(rows, narrow(real, got.rows))
    assert old["1"] == old["1b"] == "not supported"


def test_without_judges_a_question_on_both_right_alone_cannot_support_prediction_4(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "both_right"]
    missed = set(sorted({p["event_id"] for p in rows})[:2])

    def answer(meta, event):
        x = event.expected_current
        if meta["arm"] == "alone":
            return f"I'm not sure. {x}?"
        return "I don't know." if meta["arm"] == "key" and event.id in missed else f"The first author is {x}."
    got, _ = run_real(real, rows, answer, tmp_path, judges=False)
    alone = [r for r in got.rows if r["arm"] == "alone"]
    assert len(alone) == 20 and {(r["grade"], r["offers_expected"]) for r in alone} == {("ungraded", True)}
    assert verdicts(rows, got.rows)["4"] == UNDECIDED
    assert verdicts(rows, narrow(real, got.rows))["4"] == "supported"


def test_without_judges_hedged_surname_key_rows_cannot_support_3a(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "dark"]

    def answer(meta, event):
        x, s, _ = parts(event)
        return {"key": f"I think it's {s}, but I'm not sure.", "full": f"The first author is {x}."}.get(
            meta["arm"], "I don't know.")
    got, _ = run_real(real, rows, answer, tmp_path, judges=False)
    key = Counter((r["grade"], r["offers_expected"]) for r in got.rows if r["arm"] == "key")
    assert key == {("ungraded", True): 92, ("correct", True): 24}
    assert verdicts(rows, got.rows)["3a"] == UNDECIDED
    assert verdicts(rows, narrow(real, got.rows))["3a"] == "supported"
    assert f"and {key[('ungraded', True)]} `dark` key rows \"I think it's Zhang, but I'm not sure.\" made 3a " \
        "supported" in design_row("an open offer in any form")


FULL_FORMS = ["The note says {X}, but I believe the first author is Sanjay Ramaswamy.", "I doubt it is {X}.",
              "{X} seems unlikely.", "{X} is a co-author, not the first author.",
              "I am unable to verify that {X} is the first author."]


@pytest.mark.parametrize("template", FULL_FORMS)
def test_the_full_rows_whose_right_name_the_rules_leave_to_the_judges_are_open_offers(real, tmp_path, template):
    # a relay, a doubt, a rejection, another role, a disclaimer: correct by the paper rules, the right name left to
    # the judges, ungraded without them; b86d67f stored the doubts and rejections as offering nothing
    rows = [p for p in real.plan if p["set"] in ("dark", "both_right") and p["arm"] == "full"]
    got, _ = run_real(real, rows, lambda meta, event: template.format(X=event.expected_current), tmp_path,
                      judges=False)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert len(full) == 136 and {(r["grade"], r["offers_expected"]) for r in full} == {("ungraded", True)}
    assert build_report(rows, got.rows)["totals"]["open_offers"] == 136
    six = " ".join(DESIGN.read_text("utf-8").split())
    assert f"a relay, a doubt, a rejection, another role or a disclaimer, {len(full)} of {len(full)} such rows on " \
        "each form" in six


# ---- the answers the judges' votes left undecided ---------------------------------------------------------------


@pytest.mark.parametrize("judges, grade, votes, case", [
    ([J1], CORRECT, {"j1": "ENDORSE"}, "every vote read"),
    ([J1], "no idea", {"j1": "ENDORSE"}, "a grading reply without a verdict"),
    ([J1], CORRECT, {"j1": "maybe"}, "an adoption reply without a verdict"),
    ([J1, J2], CORRECT, {"j1": "ENDORSE", "j2": "REJECT"}, "a tie on adoption"),
    ([J1], CORRECT, {"j1": BackendError("upstream 500", status=500)}, "a failed call"),
])
def test_the_run_counts_the_answers_the_judges_left_undecided(tmp_path, judges, grade, votes, case):
    events, rows = fixture_plan()
    by_id = {e.id: e for e in events}

    def answer(req):
        event = by_id[req.metadata["event_id"]]
        if event.control or req.metadata["arm"] == "alone":
            return "I can't reliably verify the first author from the title alone."    # the rules leave it open
        return f"The note says {event.expected_current}."                               # reported speech
    got = asyncio.run(run_plan(judged(answer, grade=grade, votes=votes), [ModelSpec(key=k, provider_id=f"x/{k}")
                                                                          for k in ("a", "b")],
                               rows, tmp_path / "keys.jsonl", events=events, judges=judges))
    counted = sum(1 for r in got.rows if judges_left_open(r))
    assert got.judges_undecided == counted
    assert (counted == 0) == (case == "every vote read")
    assert (got.judge_failed > 0) == (case == "a failed call")              # only a failed call records a failure


def test_a_reply_without_a_verdict_is_printed_and_flagged(tmp_path, capsys, monkeypatch, real):
    from cutoffmix import backends

    rows = [p for p in real.plan if p["set"] == "both_right"]
    plan = tmp_path / "plan.jsonl"
    write_jsonl(plan, rows)

    def reply(model, req):
        purpose = req.metadata.get("purpose")
        if purpose == "judge":
            return ChatResult(text="I need more context.", usage={"cost": 0.001}, finish_reason="stop")
        if purpose == "adoption_judge":
            return ChatResult(text="ENDORSE", usage={"cost": 0.001}, finish_reason="stop")
        x, _, g = parts(real.events[req.metadata["event_id"]])
        return f"It is not {Y}, it is {g}." if req.metadata["arm"] == "key" else f"The first author is {x}."
    monkeypatch.setattr(backends, "make_backend", lambda cfg, **kw: ScriptedBackend(reply, cost=0.001))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    out = tmp_path / "keys_results.jsonl"
    assert main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out),
                 "--max-cost", "5", "--judges", "gpt-6.2-sol"]) == 0           # the exit status does not change
    err = capsys.readouterr().err
    assert "judge calls failed" not in err                                     # no call failed
    assert "20 answers are undecided after the judges' votes (a failed call, a tie or a reply without a verdict): " \
        "rerun with --judges before reporting; it asks the judges again, without a new recipient call" in err
    report_json = tmp_path / "report.json"
    assert main(["bench", "keys", "report", "--plan", str(plan), "--results", str(out), "--json",
                 str(report_json)]) == 0
    report = json.loads(report_json.read_text("utf-8"))
    assert report["totals"]["judges_undecided"] == 20 == report["totals"]["open_offers"]
    assert report["predictions"][4]["verdict"] == UNDECIDED                    # key 0 of 20 correct, all open offers
    assert any(f.startswith("20 answered rows are undecided after the judges' votes") for f in report["flags"])
    # a rerun whose judge answers asks again without a recipient call, and nothing stays undecided
    capsys.readouterr()

    def answering(model, req):
        if req.metadata.get("purpose") == "judge":
            return ChatResult(text=CORRECT, usage={"cost": 0.001}, finish_reason="stop")
        return reply(model, req)
    monkeypatch.setattr(backends, "make_backend", lambda cfg, **kw: ScriptedBackend(answering, cost=0.001))
    assert main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out),
                 "--max-cost", "5", "--judges", "gpt-6.2-sol"]) == 0
    assert "undecided after the judges' votes" not in capsys.readouterr().err
    last = {r["id"]: r for r in read_jsonl(out)}
    assert not any(judges_left_open(r) for r in last.values())


def test_the_report_counts_and_flags_the_rows_the_judges_left_undecided():
    study = Study()
    for i in range(12):
        study.item("both_right", f"r{i}", {"alone": True, "full": True,
                                           "key": {"grade": "ungraded", "grader": "judge:j1",
                                                   "offers_expected": False} if i < 2 else True}, recognised=True)
    report = study.report()
    assert report["totals"]["judges_undecided"] == 2 and report["totals"]["open_offers"] == 0
    assert any(f.startswith("2 answered rows are undecided after the judges' votes (a failed call, a tie or a reply "
                            "without a verdict): rerun 'bench keys run' with --judges before reporting")
               for f in report["flags"])
    # a label the adoption judges left undecided counts too; a rule's open label and a refusal do not
    assert judges_left_open({"grade": "incorrect", "grader": "rule", "adoption": "endorsed", "adopted": None,
                             "adoption_grader": "judge:j1"})
    assert not judges_left_open({"grade": "ungraded", "grader": "rule", "adoption": "reported", "adopted": None,
                                 "adoption_grader": "rule"})
    assert not judges_left_open({"grade": "ungraded", "grader": "refusal"})
    assert not judges_left_open({"grade": "ungraded", "grader": "judge:j1", "error": "upstream 503"})


def test_a_stored_row_gets_the_wider_offers_expected_on_resume_without_a_call(real, tmp_path):
    # rows written by b86d67f: offers_expected by offers_answer alone; a resumed run refreshes them without a call
    rows = [p for p in real.plan if p["set"] == "both_right"]

    def answer(meta, event):
        return f"I'm not sure. {event.expected_current}?"
    run_real(real, rows, answer, tmp_path, judges=False)
    out = tmp_path / "keys.jsonl"
    write_jsonl(out, narrow(real, read_jsonl(out)))
    assert {r["offers_expected"] for r in read_jsonl(out)} == {False}
    backend = ScriptedBackend(lambda model, req: pytest.fail("no call"))
    again = asyncio.run(run_plan(backend, RECIPIENTS, rows, out, events=real.events, backend_name="scripted"))
    assert backend.calls == [] and again.regraded == len(rows) and again.start.offer_flag == len(rows)
    assert {r["offers_expected"] for r in again.rows} == {True}
