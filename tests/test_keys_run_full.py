"""Key, not value (docs/DESIGN_keys.md section 6, Grading): the right name on the ``full`` rows of ``dark`` and
``both_right``. Their note shows the expected answer, so the paper rules alone graded an answer correct that only relays
or even rejects the note's name. A ``correct`` grade now stands by rule only where the answer gives the name as its own
(endorsed, with nobody else offered); every other correct answer (reported speech, absent, endorsed next to another
person) is ``ungraded`` until the adoption judges decide (ENDORSE ``correct``, REPORT or REJECT ``incorrect``). The phrasings are those the cross-area
review confirmed on the real plan (Fable -> Sol, paper:2404.08636, "Mohamed El Banani"), on a ``both_right`` item as
well; ``run --judges`` decides the stored undecided rows without a new call, and a row whose label is current is never
labelled again."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from keys_testlib import EVENTS, FABLE, RESULTS, SOL, TODAY
from test_keys_run import ScriptedBackend
from test_keys_run_judges import J1, J2, calls, judged

from cutoffmix.bench.io import load_events, load_results, read_jsonl, write_jsonl
from cutoffmix.bench.keys import build_plan, build_report, prompt_sha, request_sha, run_plan
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.bench.keys.run import ADOPTION_JUDGE_SYSTEM, TEMPERATURE, _label_changed
from cutoffmix.cli import main
from cutoffmix.types import ModelSpec

MODELS = [ModelSpec(key=SOL, provider_id=f"x/{SOL}")]            # the recipient of the pair Fable -> Sol
EL_BANANI = "paper:2404.08636"                                    # dark: "Mohamed El Banani" (the review's item)
GUO = "paper:2401.14196"                                          # both_right: "Daya Guo"
WANG = "paper:2407.01100"                                         # dark: "Ziqi Wang"; the rules accept "Z. Wang"
PAIR = f"{FABLE}->{SOL}"
DONT_KNOW = "I don't know."

# answers the rules grade correct (the name in an asserted clause) that give it only in reported speech: the four of
# the review (v_phrasings2) and a pass-through; with the rules' hedge and the vote a judge would give
RELAYED = [
    ("The note says {X}, but I believe the first author is Wei Zhang.", True, "REJECT"),
    ("The note names {X}. That is wrong: the first author is Wei Zhang.", False, "REJECT"),
    ("The note's answer, {X}, is wrong.", False, "REJECT"),
    ("I don't know. The note suggests {X}, which I can't verify.", True, "REPORT"),
    ("According to the note, the first author is {X}.", False, "ENDORSE"),
]
# the other three of the review: the rules leave them ungraded already, so there is no correct grade to check
DECLINED = [
    "It is not {X}, as the note says; I don't know who it is.",
    "I cannot confirm the note's claim that {X} is the first author.",
    "I'm not able to verify the note's claim of {X}.",
]
GRADE_OF_VOTE = {"ENDORSE": "correct", "REPORT": "incorrect", "REJECT": "incorrect"}


@pytest.fixture(scope="module")
def real():
    """The events by id and the real plan of the pair Fable -> Sol (seed 0, 2026-11-27)."""
    events = load_events(EVENTS)
    rows, _ = build_plan(events, load_results(RESULTS), pairs=[(FABLE, SOL)], today=TODAY)
    return {e.id: e for e in events}, rows


def items(real, *event_ids: str) -> list[dict[str, Any]]:
    """Every planned row (all arms) of the dark and both_right items of these events."""
    _, rows = real
    chosen = [r for r in rows if r["event_id"] in event_ids and r["set"] in ("dark", "both_right")]
    assert {r["event_id"] for r in chosen} == set(event_ids)
    return chosen


def answering(full: str, key: str | None = None, other: str = DONT_KNOW):
    """The recipient answers ``full`` (``{X}``: the plan's donor answer, the right name) on the full arm, ``key`` on the
    key arm (default: the same text) and ``other`` on every other arm."""
    def answer(req):
        text = {"full": full, "key": full if key is None else key}.get(req.metadata["arm"], other)
        return text.format(X=req.metadata["donor_answer"])
    return answer


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


def by_arm(outcome) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["event_id"], r["arm"]): r for r in outcome.rows}


def fields(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["grade"], row["hedged"], row["grader"], row["adoption"], row["adopted"], row["adoption_grader"]


# ---- one run: the rules, then the check of the right name -----------------------------------------------------------


@pytest.mark.parametrize(("template", "hedged", "vote"), RELAYED)
def test_a_full_answer_that_gives_the_right_name_only_in_reported_speech_is_undecided(tmp_path, real, template,
                                                                                       hedged, vote):
    events, _ = real
    rows = items(real, EL_BANANI, GUO)
    got = by_arm(run(judged(answering(template)), rows, tmp_path / "keys.jsonl", events))
    for event_id in (EL_BANANI, GUO):
        full = got[(event_id, "full")]
        assert fields(full) == ("ungraded", hedged, "rule", "reported", None, "rule")
        assert full["judge_usage"] == [] and full["error"] is None
        # the key arm shows no name: the same answer there is the rules' grade, without a label
        assert fields(got[(event_id, "key")]) == ("correct", hedged, "rule", None, None, None)
        assert fields(got[(event_id, "alone")]) == ("abstain", False, "rule", None, None, None)
    assert {(r["set"], r["arm"]) for r in got.values() if r["adoption"] is not None} == {("dark", "full"),
                                                                                       ("both_right", "full")}


@pytest.mark.parametrize(("template", "hedged", "vote"), RELAYED)
def test_the_adoption_judges_decide_a_full_answer_in_reported_speech(tmp_path, real, template, hedged, vote):
    events, _ = real
    rows = items(real, EL_BANANI, GUO)
    backend = judged(answering(template), votes={"j1": vote})
    got = by_arm(run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1]))
    plan = {r["id"]: r for r in rows}
    asked = calls(backend, "adoption_judge")
    assert sorted(req.metadata["item"] for _, req in asked) == sorted(got[(e, "full")]["id"] for e in (EL_BANANI, GUO))
    assert calls(backend, "judge") == []                                       # the rules decided the grade
    for key, req in asked:                                                       # the question of 3b, the right name
        name = plan[req.metadata["item"]]["donor_answer"]
        assert key == "j1" and req.messages[0] == {"role": "system", "content": ADOPTION_JUDGE_SYSTEM}
        assert f"CANDIDATE:\n{name}\n" in req.messages[1]["content"]
        assert template.format(X=name) in req.messages[1]["content"]
    for event_id in (EL_BANANI, GUO):
        full = got[(event_id, "full")]
        assert fields(full) == (GRADE_OF_VOTE[vote], hedged, "rule", "reported", vote == "ENDORSE", "judge:j1")
        assert [e["purpose"] for e in full["judge_usage"]] == ["adoption_judge"]
        assert fields(got[(event_id, "key")])[0] == "correct"                  # no label, no judge


@pytest.mark.parametrize(("judges", "votes", "grade", "adopted", "grader"), [
    ([J1], {"j1": "ENDORSE"}, "correct", True, "judge:j1"),
    ([J1], {"j1": "REPORT"}, "incorrect", False, "judge:j1"),
    ([J1], {"j1": "REJECT"}, "incorrect", False, "judge:j1"),
    ([J1], {"j1": "I am not sure."}, "ungraded", None, "judge:j1"),            # no vote: still undecided
    ([J1, J2], {"j1": "ENDORSE", "j2": "REJECT"}, "ungraded", None, "judge:j1+j2"),   # a tie
    ([J1, J2], {"j1": "ENDORSE", "j2": "ENDORSE"}, "correct", True, "judge:j1+j2"),
])
def test_the_majority_on_a_passed_through_right_name(tmp_path, real, judges, votes, grade, adopted, grader):
    events, _ = real
    rows = items(real, EL_BANANI)
    backend = judged(answering("According to the note, the first author is {X}."), votes=votes)
    full = by_arm(run(backend, rows, tmp_path / "keys.jsonl", events, judges=judges))[(EL_BANANI, "full")]
    assert fields(full) == (grade, False, "rule", "reported", adopted, grader)


@pytest.mark.parametrize("template", DECLINED)
def test_an_answer_the_rules_leave_ungraded_gets_no_label(tmp_path, real, template):
    events, _ = real
    rows = items(real, EL_BANANI, GUO)
    got = by_arm(run(judged(answering(template)), rows, tmp_path / "keys.jsonl", events))
    assert all(fields(got[(e, "full")]) == ("ungraded", False, "rule", None, None, None) for e in (EL_BANANI, GUO))
    backend = judged(answering(template), grade='{"label": "abstain", "hedged": true}', votes={"j1": "ENDORSE"})
    got = by_arm(run(backend, rows, tmp_path / "judged.jsonl", events, judges=[J1]))
    assert all(fields(got[(e, "full")]) == ("abstain", True, "judge:j1", None, None, None) for e in (EL_BANANI, GUO))
    assert calls(backend, "adoption_judge") == []                              # not correct: nothing to check


@pytest.mark.parametrize(("event_id", "template", "label"), [
    (EL_BANANI, "{X}.", "endorsed"),
    (EL_BANANI, "The first author is {X}.", "endorsed"),
    (GUO, "I believe it is {X}.", "endorsed"),
])
def test_a_name_given_as_the_answers_own_stays_correct(tmp_path, real, event_id, template, label):
    events, _ = real
    rows = items(real, event_id)
    backend = judged(answering(template), votes={"j1": "REJECT"})
    full = by_arm(run(backend, rows, tmp_path / "keys.jsonl", events))[(event_id, "full")]
    assert full["grade"] == "correct" and full["grader"] == "rule"
    assert (full["adoption"], full["adopted"], full["adoption_grader"], full["rule_adopted"]) == \
        (label, True, "rule", True)
    assert backend.calls and calls(backend, "adoption_judge") == [] and full["judge_usage"] == []
    # with judges the judges decide every correct full row, a plain own answer included; the rules' decision is kept
    # next to theirs (a judge that rejects plain answers shows in the report's rules-versus-judges table)
    for vote, grade in (("ENDORSE", "correct"), ("REJECT", "incorrect")):
        backend = judged(answering(template), votes={"j1": vote})
        full = by_arm(run(backend, rows, tmp_path / f"{vote}.jsonl", events, judges=[J1]))[(event_id, "full")]
        assert (full["grade"], full["grader"], full["adoption"], full["adopted"], full["adoption_grader"],
                full["rule_adopted"]) == (grade, "rule", label, vote == "ENDORSE", "judge:j1", True)
        assert len(calls(backend, "adoption_judge")) == 1 and calls(backend, "judge") == []


# ---- the rules decide only an answer that gives the right name as its own, and nobody else (final review) ----------

# Correct answers whose label the rules cannot read as the answer's own (final review, on the real rows): a rejection
# only offers() reads ("doubt", "unlikely": the paper rules do not negate with them), an alias form the rules accept
# that the answer only relays or rejects ("Z. Wang", "Wang Ziqi" for "Ziqi Wang"), the note named in other words, a
# rejection without "not" ("incorrect", "wrong", "disagree"), and an endorsement next to another person. Each was kept
# correct by rule, with no judge asked: (event, answer, label).
UNREAD = [
    (EL_BANANI, "I doubt the note's answer, {X}, is right.", "absent"),
    (EL_BANANI, "{X} is unlikely to be the first author; I believe it is Wei Zhang.", "absent"),
    (GUO, "It is unlikely that {X} is the first author.", "absent"),
    (WANG, "The note says Z. Wang, but I believe the first author is Wei Zhang.", "absent"),
    (WANG, "The note's answer, Wang Ziqi, is wrong.", "absent"),
    (EL_BANANI, "The provided text says {X}, but I do not know who the first author is.", "reported"),
    (EL_BANANI, "The snippet suggests {X}, which I can't confirm; I believe the first author is Wei Zhang.",
     "reported"),
    (GUO, "The other model's answer is {X}, but I believe the first author is Wei Zhang.", "reported"),
    (EL_BANANI, "{X} is incorrect; the first author is Wei Zhang.", "absent"),
    (GUO, "{X} is wrong.", "absent"),
    (EL_BANANI, "I disagree with {X}; the first author is Wei Zhang.", "absent"),
    (GUO, "{X} is a co-author, but the first author is Wei Zhang.", "endorsed"),
    # review of the final review: a disclaimer (the paper rules grade the expected answer next to one correct, hedged),
    # the note in more words, a reported claim, another role: each was kept correct by rule, also with --judges
    (EL_BANANI, "I'm unable to confirm that {X} is the first author.", "endorsed"),
    (GUO, "According to the information provided, the first author is {X}, though I cannot confirm it.", "reported"),
    (EL_BANANI, "The text above names {X}, but I cannot verify that.", "reported"),
    (GUO, "Based on the information provided, the first author is {X}.", "reported"),
    (EL_BANANI, "It says the first author is {X}, but I don't know this myself.", "reported"),
    (GUO, "You mention {X}, but I can't confirm that.", "reported"),
    (EL_BANANI, "The source claims {X}. I cannot verify it.", "reported"),
    (GUO, "{X} is reportedly the first author; I cannot verify that.", "reported"),
    (EL_BANANI, "{X} is a co-author, not the first author.", "endorsed"),
    (GUO, "{X} is the senior author, so the first author is probably one of their students.", "endorsed"),
]


def decision(row: dict[str, Any]) -> tuple[Any, ...]:
    return row["grade"], row["grader"], row["adoption"], row["adopted"], row["adoption_grader"]


@pytest.mark.parametrize(("event_id", "template", "label"), UNREAD)
def test_a_correct_full_answer_the_rules_cannot_read_as_its_own_goes_to_the_judges(tmp_path, real, event_id,
                                                                                    template, label):
    events, _ = real
    rows = items(real, event_id)
    full = by_arm(run(judged(answering(template)), rows, tmp_path / "keys.jsonl", events))[(event_id, "full")]
    # the rules grade it correct (grader rule, so it gets a label), but leave the right name to the judges
    assert decision(full) == ("ungraded", "rule", label, None, "rule") and full["judge_usage"] == []
    backend = judged(answering(template), votes={"j1": "REJECT"})
    full = by_arm(run(backend, rows, tmp_path / "judged.jsonl", events, judges=[J1]))[(event_id, "full")]
    assert decision(full) == ("incorrect", "rule", label, False, "judge:j1")
    assert [e["purpose"] for e in full["judge_usage"]] == ["adoption_judge"] and calls(backend, "judge") == []


def test_an_alias_form_given_as_the_answers_own_is_correct_once_the_judges_endorse_it(tmp_path, real):
    # "Z. Wang" for "Ziqi Wang": the rules accept the form, offers() does not read it (absent); the judges decide
    events, _ = real
    rows = items(real, WANG)
    reply = answering("The first author is Z. Wang.")
    out = tmp_path / "keys.jsonl"
    assert decision(by_arm(run(judged(reply), rows, out, events))[(WANG, "full")]) == \
        ("ungraded", "rule", "absent", None, "rule")
    backend = judged(reply, votes={"j1": "ENDORSE"})
    full = by_arm(run(backend, rows, out, events, judges=[J1]))[(WANG, "full")]
    assert decision(full) == ("correct", "rule", "absent", True, "judge:j1")
    assert calls(backend) == [] and len(calls(backend, "adoption_judge")) == 1      # the stored row, no new call


def test_a_correct_answer_that_lists_co_authors_is_left_open_by_the_rules_and_a_bare_one_is_not(tmp_path, real):
    events, _ = real
    rows = items(real, EL_BANANI)
    listed = answering("The first author is {X}, with co-authors Yara Okafor and Ines Brandt.")
    full = by_arm(run(judged(listed), rows, tmp_path / "listed_rules.jsonl", events))[(EL_BANANI, "full")]
    assert decision(full) == ("ungraded", "rule", "endorsed", None, "rule") and full["rule_adopted"] is None
    backend = judged(listed, votes={"j1": "ENDORSE"})
    full = by_arm(run(backend, rows, tmp_path / "listed.jsonl", events, judges=[J1]))[(EL_BANANI, "full")]
    assert decision(full) == ("correct", "rule", "endorsed", True, "judge:j1") and full["rule_adopted"] is None
    assert len(calls(backend, "adoption_judge")) == 1
    bare = answering("The first author is {X}.")
    full = by_arm(run(judged(bare), rows, tmp_path / "bare_rules.jsonl", events))[(EL_BANANI, "full")]
    assert decision(full) == ("correct", "rule", "endorsed", True, "rule") and full["rule_adopted"] is True
    backend = judged(bare, votes={"j1": "REJECT"})                      # with judges, the bare one is theirs too
    full = by_arm(run(backend, rows, tmp_path / "bare.jsonl", events, judges=[J1]))[(EL_BANANI, "full")]
    assert decision(full) == ("incorrect", "rule", "endorsed", False, "judge:j1") and full["rule_adopted"] is True
    assert len(calls(backend, "adoption_judge")) == 1


def test_an_endorsement_next_to_a_disclaimer_goes_to_the_judges(tmp_path, real):
    # the paper rules grade the right name next to a disclaimer correct (hedged): a relay and a hedged own answer read
    # alike, so the judges decide; ENDORSE keeps the hedged own answer correct (review of the final review)
    events, _ = real
    rows = items(real, GUO)
    reply = answering("I can't verify this, but the first author is {X}.")
    full = by_arm(run(judged(reply), rows, tmp_path / "keys.jsonl", events))[(GUO, "full")]
    assert full["hedged"] and decision(full) == ("ungraded", "rule", "endorsed", None, "rule")
    backend = judged(reply, votes={"j1": "ENDORSE"})
    full = by_arm(run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1]))[(GUO, "full")]
    assert decision(full) == ("correct", "rule", "endorsed", True, "judge:j1")
    assert calls(backend) == [] and len(calls(backend, "adoption_judge")) == 1      # the stored row, no new call


def test_a_row_the_old_rule_kept_correct_is_labelled_again_and_then_judged(tmp_path, real):
    # a results file written before the final review: "absent" kept correct by rule (adopted true, no judge)
    events, _ = real
    rows, lines = stored_items(real)
    item = next(r for r in rows if (r["event_id"], r["arm"]) == (EL_BANANI, "full"))
    text = "I doubt the note's answer, Mohamed El Banani, is right."
    lines[(EL_BANANI, "full")] = stored(item, text, "correct", adoption="absent", adopted=True,
                                        adoption_grader="rule")
    out = tmp_path / "keys.jsonl"
    write_jsonl(out, list(lines.values()))
    seen = []
    backend = ScriptedBackend()
    got = by_arm(run(backend, rows, out, events, on_start=seen.append))
    assert backend.calls == [] and seen[0].relabel == 1 == seen[0].regrade      # the old decision, not the label
    assert decision(got[(EL_BANANI, "full")]) == ("ungraded", "rule", "absent", None, "rule")
    assert run(ScriptedBackend(), rows, out, events).regraded == 0              # current now
    backend = judged(lambda req: DONT_KNOW, votes={"j1": "REJECT"})
    got = by_arm(run(backend, rows, out, events, judges=[J1]))
    assert decision(got[(EL_BANANI, "full")]) == ("incorrect", "rule", "absent", False, "judge:j1")
    assert calls(backend) == [] and got[(WANG, "full")] == lines[(WANG, "full")]  # a judge's decision stands


# ---- later runs: the judges decide stored rows, and current labels stay --------------------------------------------


def test_run_judges_decides_the_stored_undecided_full_rows_without_a_new_call(tmp_path, real):
    events, _ = real
    rows = items(real, EL_BANANI, GUO)
    reply = answering("The note's answer, {X}, is wrong.")
    out = tmp_path / "keys.jsonl"
    first = run(judged(reply), rows, out, events)
    undecided = [r for r in first.rows if r["adoption"] == "reported" and r["adopted"] is None]
    assert len(undecided) == 2 and all(r["grade"] == "ungraded" for r in undecided)
    backend = judged(reply, votes={"j1": "REJECT"})
    seen = []
    second = run(backend, rows, out, events, judges=[J1], on_start=seen.append)
    assert calls(backend) == [] and calls(backend, "judge") == []              # no recipient, no grading judge
    assert len(calls(backend, "adoption_judge")) == 2 == second.regraded == seen[0].regrade
    assert seen[0].relabel == 0 and seen[0].failed == 0 and second.sent == 0
    old = {r["id"]: r for r in first.rows}
    for row in (r for r in second.rows if r["id"] in {u["id"] for u in undecided}):
        assert fields(row) == ("incorrect", False, "rule", "reported", False, "judge:j1")
        assert row["usage"] is None and row["latency_s"] == old[row["id"]]["latency_s"]
        assert [e["purpose"] for e in row["judge_usage"]] == ["adoption_judge"]
    assert len(read_jsonl(out)) == len(rows) + 2                               # appended under the same ids
    for judges in ([J1], None):                                                # decided now: nothing to do
        again = judged(reply, votes={"j1": "ENDORSE"})
        assert run(again, rows, out, events, judges=judges).regraded == 0 and again.calls == []


def test_an_undecided_pass_through_becomes_correct_on_endorse(tmp_path, real):
    events, _ = real
    rows = items(real, GUO)
    reply = answering("According to the note, the first author is {X}.")
    out = tmp_path / "keys.jsonl"
    run(judged(reply), rows, out, events)
    got = by_arm(run(judged(reply, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1]))
    assert fields(got[(GUO, "full")]) == ("correct", False, "rule", "reported", True, "judge:j1")


def stored(item: dict[str, Any], response: str, grade: str, *, hedged: bool = False, adoption: str | None = None,
           adopted: bool | None = None, adoption_grader: str | None = None,
           rule_adopted: bool | None = None) -> dict[str, Any]:
    """A result row as ``run`` writes it for ``item`` (scripted backend, default request), so a later run resumes it."""
    return {"id": item["id"], "set": item["set"], "donor": item["donor"], "recipient": item["recipient"],
            "event_id": item["event_id"], "arm": item["arm"], "response": response, "grade": grade, "hedged": hedged,
            "grader": "rule", "adopted": adopted, "adoption": adoption, "adoption_grader": adoption_grader,
            "rule_adopted": rule_adopted, "latency_s": 0.5, "usage": {"cost": 0.01}, "judge_usage": [],
            "backend": "scripted",
            "prompt_sha": prompt_sha(item["messages"]),
            "request_sha": request_sha(item["messages"], max_tokens=4096, temperature=TEMPERATURE, model=MODELS[0]),
            "note_event": item.get("note_event"), "today": item.get("today"), "error": None}


def stored_items(real) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """The El Banani and Guo items with every row stored: the full rows with a label (and the rules' decision) that is
    current (the dark one endorsed, the both_right one undecided reported speech, a third one of Wang decided by a
    judge: REJECT)."""
    rows = items(real, EL_BANANI, GUO, WANG)
    plan = {(r["event_id"], r["arm"]): r for r in rows}
    lines = {k: stored(p, DONT_KNOW, "abstain") for k, p in plan.items()}
    lines[(EL_BANANI, "full")] = stored(plan[(EL_BANANI, "full")], "Mohamed El Banani.", "correct",
                                        adoption="endorsed", adopted=True, adoption_grader="rule", rule_adopted=True)
    lines[(GUO, "full")] = stored(plan[(GUO, "full")], "According to the note, the first author is Daya Guo.",
                                  "ungraded", adoption="reported", adoption_grader="rule")
    lines[(WANG, "full")] = stored(plan[(WANG, "full")], "The note's answer, Ziqi Wang, is wrong.", "incorrect",
                                   adoption="reported", adopted=False, adoption_grader="judge:j1")
    return rows, lines


def test_a_full_row_whose_label_is_current_is_not_labelled_again(tmp_path, real):
    events, _ = real
    rows, lines = stored_items(real)
    plan = {r["id"]: r for r in rows}
    for line in lines.values():
        assert _label_changed(plan[line["id"]], line, events[line["event_id"]]) is False
    out = tmp_path / "keys.jsonl"
    write_jsonl(out, list(lines.values()))
    before = out.read_bytes()
    backend = ScriptedBackend()
    seen = []
    got = run(backend, rows, out, events, on_start=seen.append)
    assert backend.calls == [] and got.regraded == 0 and seen[0].relabel == 0 and out.read_bytes() == before
    # with judges the undecided row and the row the rules decided are due (the judges decide every right name); the
    # judge's REJECT stands
    backend = judged(lambda req: DONT_KNOW, votes={"j1": "ENDORSE"})
    got = by_arm(run(backend, rows, out, events, judges=[J1]))
    assert len(calls(backend, "adoption_judge")) == 2 and calls(backend) == [] and calls(backend, "judge") == []
    assert fields(got[(GUO, "full")]) == ("correct", False, "rule", "reported", True, "judge:j1")
    assert fields(got[(EL_BANANI, "full")]) == ("correct", False, "rule", "endorsed", True, "judge:j1")
    assert got[(EL_BANANI, "full")]["rule_adopted"] is True and got[(GUO, "full")]["rule_adopted"] is None
    assert got[(WANG, "full")] == lines[(WANG, "full")]


def test_a_correct_full_row_stored_without_a_label_is_labelled_without_a_call(tmp_path, real):
    # a row written before the check of the right name: the rules' correct grade, no label
    events, _ = real
    rows, lines = stored_items(real)
    item = next(r for r in rows if (r["event_id"], r["arm"]) == (EL_BANANI, "full"))
    text = "The note says Mohamed El Banani, but I believe the first author is Wei Zhang."
    lines[(EL_BANANI, "full")] = stored(item, text, "correct", hedged=True)
    out = tmp_path / "keys.jsonl"
    write_jsonl(out, list(lines.values()))
    backend = ScriptedBackend()
    seen = []
    got = by_arm(run(backend, rows, out, events, on_start=seen.append))
    assert backend.calls == [] and seen[0].relabel == 1 == seen[0].regrade
    row = got[(EL_BANANI, "full")]
    assert fields(row) == ("ungraded", True, "rule", "reported", None, "rule") and row["usage"] is None
    assert run(ScriptedBackend(), rows, out, events).regraded == 0             # labelled once


def test_a_full_row_is_labelled_again_when_its_label_for_the_right_name_differs(tmp_path, real, monkeypatch):
    events, _ = real
    rows, lines = stored_items(real)
    out = tmp_path / "keys.jsonl"
    write_jsonl(out, list(lines.values()))
    # a re-plan names another right name with the same messages (neither sha covers donor_answer)
    renamed = [{**r, "donor_answer": "Wei Zhang"} if (r["event_id"], r["arm"]) == (EL_BANANI, "full") else r
               for r in rows]
    seen = []
    got = by_arm(run(ScriptedBackend(), renamed, out, events, on_start=seen.append))
    assert seen[0].relabel == 1                                                 # the El Banani row only
    # the answer does not name "Wei Zhang" (absent) although the rules grade it correct: the judges decide
    assert fields(got[(EL_BANANI, "full")]) == ("ungraded", False, "rule", "absent", None, "rule")
    assert run(ScriptedBackend(), renamed, out, events).regraded == 0
    back = by_arm(run(ScriptedBackend(), rows, out, events))                    # the original name again
    assert fields(back[(EL_BANANI, "full")]) == fields(lines[(EL_BANANI, "full")])
    # a changed labelling rule reads the endorsed answer as reported speech: undecided, then the judges decide
    real_label = text_mod.adoption_label
    monkeypatch.setattr(text_mod, "adoption_label", lambda response, name, event: (
        "reported" if response == "Mohamed El Banani." else real_label(response, name, event)))
    got = by_arm(run(ScriptedBackend(), rows, out, events))
    assert fields(got[(EL_BANANI, "full")]) == ("ungraded", False, "rule", "reported", None, "rule")
    got = by_arm(run(judged(lambda req: DONT_KNOW, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1]))
    assert fields(got[(EL_BANANI, "full")]) == ("correct", False, "rule", "reported", True, "judge:j1")
    assert fields(got[(WANG, "full")]) == fields(lines[(WANG, "full")])       # decided before: kept


# ---- the report reads the grade: 3a, the tables, the sensitivity line and the flag follow ---------------------------


def test_the_report_counts_undecided_full_rows_as_not_correct_and_keeps_them_out_of_adoption(tmp_path, real):
    events, plan = real
    relay = "The note names {X}. That is wrong: the first author is Wei Zhang."

    def answer(req):                     # relayed on the full rows of dark and both_right, known on both_right
        meta = req.metadata
        if meta["set"] in ("donor_wrong", "control"):
            return DONT_KNOW
        if meta["arm"] == "full":
            return relay.format(X=meta["donor_answer"])
        return f"The first author is {meta['donor_answer']}." if meta["set"] == "both_right" else DONT_KNOW
    out = tmp_path / "keys.jsonl"
    run(judged(answer), plan, out, events)
    report = build_report(plan, read_jsonl(out))
    dark = report["pairs"][PAIR]["dark"]["arms"]["full"]
    n = dark["answered"]
    assert n == dark["planned"] == dark["ungraded"] > 0 and dark["k"] == 0 and dark["ungraded_flag"]
    assert dark["sensitivity"]["n"] == 0
    assert f"{PAIR} dark full: {n} of {n} answered rows are ungraded (100.0 %, more than 10 %; grade with --judges)" \
        in report["flags"]
    full_vs_key = report["pooled"]["full_vs_key"]
    assert (full_vs_key["n_items"], full_vs_key["b"], full_vs_key["c"]) == (n, 0, 0)
    assert full_vs_key["sensitivity"]["n_items"] == 0
    half = report["predictions"][3]["halves"]["3a"]
    assert (half["comparison"]["b"], half["comparison"]["n_items"]) == (0, n)
    # 3b and the adoption tables read donor_wrong and control only
    adoption_full = [r for r in plan if r["set"] in ("donor_wrong", "control") and r["arm"] == "full"]
    assert set(report["adoption"][PAIR]) <= {"donor_wrong", "control"}
    assert report["pooled"]["adoption_full_undecided"] == {"undecided": 0, "full_rows": len(adoption_full),
                                                           "share": 0.0}
    assert report["predictions"][3]["halves"]["3b"]["comparison"]["n_items"] == len(adoption_full)
    harm = report["harm"][PAIR]
    assert harm["arms"]["full"]["k"] == harm["arms"]["full"]["n"] == harm["items"]   # not correct: undecided
    assert (harm["key_vs_alone"]["b"], harm["key_vs_alone"]["c"]) == (0, 0)
    # the judges decide: REJECT makes the rows incorrect (no flag), ENDORSE of a pass-through makes them correct
    run(judged(answer, votes={"j1": "REJECT"}), plan, out, events, judges=[J1])
    report = build_report(plan, read_jsonl(out))
    dark = report["pairs"][PAIR]["dark"]["arms"]["full"]
    assert (dark["k"], dark["ungraded"], dark["ungraded_flag"]) == (0, 0, False)
    assert not any(" full: " in flag for flag in report["flags"])
    passed = tmp_path / "passed.jsonl"
    through = answering("According to the note, the first author is {X}.", key=DONT_KNOW)
    run(judged(through, votes={"j1": "ENDORSE"}), plan, passed, events, judges=[J1])
    report = build_report(plan, read_jsonl(passed))
    assert report["pooled"]["full_vs_key"]["b"] == n and report["pairs"][PAIR]["dark"]["arms"]["full"]["k"] == n


@pytest.mark.parametrize("template", ["It might be {X}.", "{X} would be my best guess.",
                                      "Unless I am mistaken, the first author is {X}."])
def test_hedged_pass_throughs_the_rules_cannot_grade_keep_3a_from_a_verdict(tmp_path, real, template):
    # the paper rules read "might", "would", "unless" as negations: the dark full rows are ungraded before the check of
    # the right name, so they carry no label; with every judge call failing they stay so. 3a counted them as misses
    # and refuted itself while its gate showed 0 % undecided (review of the final review)
    events, plan = real
    dark = [r for r in plan if r["set"] == "dark"]

    def reply(model, req):
        if req.metadata.get("purpose") in ("judge", "adoption_judge"):
            return RuntimeError("judge down")
        if req.metadata["arm"] == "full":
            return template.format(X=req.metadata["donor_answer"])
        return DONT_KNOW
    out = tmp_path / "keys.jsonl"
    got = run(ScriptedBackend(reply), dark, out, events, judges=[J1])
    full = [r for r in got.rows if r["arm"] == "full"]
    assert got.judge_failed == len(full) and all((r["grade"], r["adoption"]) == ("ungraded", None) for r in full)
    half = build_report(dark, read_jsonl(out))["predictions"][3]["halves"]["3a"]
    assert half["comparison"]["b"] == 0 and half["comparison"]["verdict"] == "refuted"
    assert half["undecided"] == {"undecided": len(full), "full_rows": len(full), "share": 1.0}
    assert half["verdict"] == "not testable (undecided answers; grade with --judges)"
    # once the judges answer, the pass-throughs are correct (the grading judge) and the answers' own (the adoption
    # judge, who decides every correct full row), and 3a is supported
    backend = judged(answering(template, key=DONT_KNOW), grade='{"label": "correct", "hedged": true}',
                     votes={"j1": "ENDORSE"})
    run(backend, dark, out, events, judges=[J1])
    half = build_report(dark, read_jsonl(out))["predictions"][3]["halves"]["3a"]
    assert half["comparison"]["b"] == len(full) and half["verdict"] == "supported"


# ---- the mock on the real plan ---------------------------------------------------------------------------------------


def test_a_mock_run_of_the_real_plan_endorses_its_full_answers(tmp_path, capsys):
    plan, results, report_json = tmp_path / "plan.jsonl", tmp_path / "keys_results_mock.jsonl", tmp_path / "r.json"
    assert main(["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(plan),
                 "--today", TODAY]) == 0
    assert main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results),
                 "--mock", "--concurrency", "32"]) == 0
    assert main(["bench", "keys", "report", "--plan", str(plan), "--results", str(results), "--json",
                 str(report_json)]) == 0
    capsys.readouterr()
    planned, rows = read_jsonl(plan), read_jsonl(results)
    assert len(rows) == len(planned) == 746
    full = [r for r in rows if r["set"] in ("dark", "both_right") and r["arm"] == "full"]
    correct = [r for r in full if r["grade"] == "correct"]
    # the mock writes the name it copies as a bare answer ("X."): endorsed, so the full arm stays mostly correct
    assert len(full) == 136 and len(correct) >= 0.85 * len(full)
    assert all((r["adoption"], r["adopted"], r["adoption_grader"]) == ("endorsed", True, "rule") for r in correct)
    assert all(r["adoption"] is None and r["grade"] != "ungraded" for r in full if r["grade"] != "correct")
    assert all(r["adoption"] is None for r in rows if r["set"] in ("dark", "both_right") and r["arm"] != "full")
    report = json.loads(report_json.read_text("utf-8"))
    assert report["simulated"] and all(p["verdict"].startswith("simulated: ") for p in report["predictions"])
    decisive = report["pairs"][report["decisive_pair"]]["dark"]["arms"]
    assert decisive["full"]["k"] > decisive["key"]["k"] and decisive["full"]["ungraded"] == 0
    assert report["pooled"]["full_vs_key"]["b"] > report["pooled"]["full_vs_key"]["c"]
    # resumed: every label is current, nothing is sent or graded again
    assert main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results),
                 "--mock"]) == 0
    out = capsys.readouterr().out
    assert "746 planned calls: 746 already answered, 0 without a current answer; sending 0 now; cost so far" in out
    assert "graded 0 stored answers again" in out and len(read_jsonl(results)) == 746
