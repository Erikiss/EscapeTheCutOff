"""Key, not value (docs/DESIGN_keys.md sections 6 and 10): the review of the judges' scope (aafe7f9).

- A donor named by a two-letter surname alone ("Wu et al.", "Dr. Li", "Ye's group") is a mention, so with --judges the
  judges decide it (it was left to the rules: absent, not adopted, on 10 donor rows of the real plan).
- A rule grade ``incorrect`` where the answer offers the expected answer ("It is not Y, it is X.", "It might be X, who
  works with Y.") is unsure on every arm: the grading judges decide it, or it is ungraded without them (it was a final
  miss, while 3b read the same reply as adopting X).
- A dash clause that asserts anew ends a negation's reach as a comma clause does ("It's not Y - it's X.").
- The rules-versus-judges table counts the rows the grading judges graded apart (it counted them as agreement).
- A resumed run keeps a judges' decision for the name it was about (``adoption_name``) while the answer is in their
  scope, whatever the label now; a decision about another name (a re-plan) counts for nothing (a later relabelling
  dropped or re-bought the decisions, and a re-plan kept a decision about the old name).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from keys_testlib import EVENTS, FABLE, NEON, RESULTS, SOL, TODAY, fixture_data, paper
from test_keys_run import MODELS
from test_keys_run_judges import J1, calls, judged

from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.io import load_events, load_results, read_jsonl
from cutoffmix.bench.keys import build_plan, build_report, render_report, run_plan
from cutoffmix.bench.keys import run as run_mod
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.bench.keys.text import adoption_label, offers
from cutoffmix.types import ModelSpec

REAL_MODELS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in (SOL, NEON, FABLE)]
DONT_KNOW = "I don't know."
OTHER = "Wei Chen"                                                  # a name the fixture events cannot place
CORRECT = '{"label": "correct", "hedged": false}'                   # the grading judge's vote


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, rows


def run(backend, rows, out, events, models=MODELS, **kw):
    return asyncio.run(run_plan(backend, models, rows, out, events=events, **kw))


@pytest.fixture(scope="module")
def real():
    """The events by id and the real plan (every pair, seed 0, 2026-11-27)."""
    events = load_events(EVENTS)
    plan, _ = build_plan(events, load_results(RESULTS), today=TODAY)
    return {e.id: e for e in events}, plan


# ---- a two-letter surname alone is a mention (V1) -------------------------------------------------------------------

SHORT_DONORS = {"Jiabo Ye": "paper:2607.28227", "Kunchang Li": "paper:2607.14935", "Xiaoyang Wu": "paper:2603.03283"}
SURNAME_ALONE = ["The first author is {S} et al.", "Dr. {S} is the first author.", "{S}'s group wrote it.",
                 "My best guess is {S}.", "Probably {S}, but I am not sure.", "It is not {S}.", "Is it {S}?"]


@pytest.mark.parametrize("form", SURNAME_ALONE)
@pytest.mark.parametrize("name", sorted(SHORT_DONORS))
def test_a_two_letter_surname_alone_mentions_the_donor(real, name, form):
    events, _ = real
    event = events[SHORT_DONORS[name]]
    assert text_mod.mentions(form.format(S=name.split()[-1]), name, event)


def test_a_two_letter_surname_is_a_mention_only_with_its_capital_letter(real):
    events, _ = real
    wu, li = events["paper:2603.03283"], events["paper:2607.14935"]
    for text in ("wu et al.", "WU ET AL.", "The Wu-Tang paper.", "I don't know.", "Wuhan University."):
        assert not text_mod.mentions(text, "Xiaoyang Wu", wu), text
    # another person of that surname is a false mention by design: one judge call (here the right answer, Xinhao Li)
    assert li.expected_current == "Xinhao Li" and text_mod.mentions("Xinhao Li.", "Kunchang Li", li)
    # a capital two-letter name part of the donor is no particle: "de", "la" are not looked for alone
    event = paper(40, "Ana de la Cruz")
    assert not text_mod.mentions("It is de facto unknown.", "Ana de la Cruz", event)


def test_with_judges_every_donor_row_named_by_a_two_letter_surname_alone_is_judged(tmp_path, real):
    events, plan = real
    rows = [p for p in plan if p["set"] in ("donor_wrong", "control") and p["donor_answer"] in SHORT_DONORS]
    assert len(rows) == 10 and sum(p["arm"] == "full" for p in rows) == 5

    def answer(req):
        return f"The first author is {req.metadata['donor_answer'].split()[-1]} et al."
    backend = judged(answer, grade='{"label": "incorrect", "hedged": false}', votes={"j1": "ENDORSE"})
    got = run(backend, rows, tmp_path / "keys.jsonl", events, models=REAL_MODELS, judges=[J1])
    assert len(calls(backend, "adoption_judge")) == 10                          # every row, none left to the rules
    assert all((r["adoption"], r["adopted"], r["adoption_grader"], r["rule_adopted"]) ==
               ("absent", True, "judge:j1", False) for r in got.rows)


# ---- a rule grade incorrect next to the expected answer is unsure, on every arm (V3) -------------------------------

CORRECTIONS = [
    "It is not {Y}, it is {X}.", "It isn't {Y}, it's {X}.", "It's not {Y} - it's {X}.", "It's not {Y} — it's {X}.",
    "Not {Y}, it's {X}.", "I don't think it's {Y}, it's {X}.", "It might be {X}, who works with {Y}.",
    "My best guess would be {X}, possibly with {Y}.",
    "Unless I'm mistaken, the first author is {X}, with {Y} as senior author.",
]


@pytest.mark.parametrize("template", CORRECTIONS)
def test_a_rule_grade_incorrect_next_to_the_expected_answer_is_unsure(real, template):
    events, _ = real
    event = events["paper:2404.08636"]                                          # Mohamed El Banani
    text = template.format(X=event.expected_current, Y="Yue Wang")
    assert grade_rule(event, "current_d", text)[0] == "incorrect"               # grading.py itself is unchanged
    assert text_mod.offers_answer(text, event)
    assert run_mod._rule_grade(event, text) is None                              # the run's grading step: unsure


@pytest.mark.parametrize("template", ["It is not {X}, it is {Y}.", "It is {Y}, not {X}.", "The first author is {Y}.",
                                      "Is it {X}? I think it is {Y}."])
def test_a_wrong_answer_given_instead_of_the_expected_one_stays_incorrect(real, template):
    events, _ = real
    event = events["paper:2404.08636"]
    text = template.format(X=event.expected_current, Y="Yue Wang")
    assert not text_mod.offers_answer(text, event)
    ruled = grade_rule(event, "current_d", text)
    assert ruled is not None and ruled[0] == "incorrect" and run_mod._rule_grade(event, text) == ruled


def test_offers_answer_reads_accepted_forms_and_never_a_control(real):
    events, _ = real
    event = events["paper:2407.01100"]                                          # Ziqi Wang, alias "Z. Wang"
    assert "Z. Wang" in event.aliases
    assert text_mod.offers_answer("It is not Yue Liu, it is Z. Wang.", event)
    assert not text_mod.offers_answer("It is not Z. Wang.", event)
    assert run_mod._rule_grade(event, "It is not Yue Liu, it is Z. Wang.") is None
    control = paper(41, control=True)
    assert not text_mod.offers_answer("It is not Yue Liu, it is Made Upname.", control)


def correction(req):
    """'It is not Wei Chen, it is X.' on every row of dark and both_right (X: the expected answer), else no idea."""
    if req.metadata["set"] in ("dark", "both_right"):
        return f"It is not {OTHER}, it is {req.metadata['donor_answer']}."
    return DONT_KNOW


def test_a_correction_of_another_name_is_unsure_on_every_arm(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("dark", "both_right")]
    got = run(judged(correction), rows, tmp_path / "rules.jsonl", events)
    # without judges: ungraded on every arm alike (no final miss on key, no final miss on full either)
    assert {(r["grade"], r["grader"], r["adoption"]) for r in got.rows} == {("ungraded", "rule", None)}
    backend = judged(correction, grade=CORRECT, votes={"j1": "ENDORSE"})
    got = run(backend, rows, tmp_path / "judged.jsonl", events, judges=[J1])
    assert len(calls(backend, "judge")) == len(rows)                            # the grading judges decide each one
    assert {(r["grade"], r["grader"]) for r in got.rows} == {("correct", "judge:j1")}
    full = [r for r in got.rows if r["arm"] == "full"]                          # then the right name is the judges'
    assert len(calls(backend, "adoption_judge")) == len(full) > 0
    assert all((r["adoption"], r["adopted"], r["adoption_grader"]) == ("endorsed", True, "judge:j1") for r in full)


def test_a_stored_correction_is_graded_by_the_judges_on_resume_without_a_recipient_call(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("dark", "both_right")]
    out = tmp_path / "keys.jsonl"
    run(judged(correction), rows, out, events)
    backend = judged(correction, grade=CORRECT, votes={"j1": "ENDORSE"})
    again = run(backend, rows, out, events, judges=[J1])
    assert calls(backend) == [] and again.regraded == len(rows) == len(calls(backend, "judge"))
    assert all(r["grade"] == "correct" for r in again.rows)


# ---- a dash clause asserts anew (V4) --------------------------------------------------------------------------------

EVENT = paper(30, "Ada Lovelace")
X = "Jiabo Ye"


@pytest.mark.parametrize("template", ["It's not {Y} - it's {X}.", "It's not {Y} — it's {X}.", "It's not {Y}—it's {X}.",
                                      "It's not {Y} -- it's {X}.", "It is not {Y} – it is {X}.",
                                      "The first author isn't {Y} - the first author is {X}."])
def test_a_dash_clause_that_asserts_anew_ends_the_reach_of_a_negation(template):
    text = template.format(X=X, Y=OTHER)
    assert offers(text, X, EVENT) and not offers(text, OTHER, EVENT)
    assert adoption_label(text, X, EVENT) == "endorsed"


@pytest.mark.parametrize("template", ["It is not {Y} - nor {X}.", "It is not {Y} - {X} is not it either.",
                                      "It is not {Y}-{X}."])
def test_a_dash_without_a_new_assertion_keeps_the_negation(template):
    assert not offers(template.format(X=X, Y=OTHER), X, EVENT)


def test_a_donor_name_after_a_dash_clause_that_asserts_anew_is_adopted_by_rule(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("donor_wrong", "control")]

    def answer(req):
        meta = req.metadata
        return f"It's not {OTHER} — it's {meta['donor_answer']}." if meta["arm"] == "full" else DONT_KNOW
    got = run(judged(answer), rows, tmp_path / "keys.jsonl", events)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert full and all((r["grade"], r["adoption"], r["adopted"], r["adoption_grader"]) ==
                        ("incorrect", "endorsed", True, "rule") for r in full)


# ---- the rules-versus-judges table (V5) -----------------------------------------------------------------------------


def test_the_table_counts_the_rows_the_grading_judges_graded_apart(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("dark", "both_right")]

    def answer(req):                                                            # the paper rules cannot grade it
        return f"It might be {req.metadata['donor_answer']}." if req.metadata["arm"] == "full" else DONT_KNOW
    event = next(e for e in events if e.id == "paper:1")
    assert grade_rule(event, "current_d", f"It might be {event.expected_current}.") is None
    out = tmp_path / "keys.jsonl"
    run(judged(answer), rows, out, events)                                      # without judges: ungraded, no label
    assert all(r["adoption"] is None for r in read_jsonl(out) if r["arm"] == "full")
    judged_out = tmp_path / "judged.jsonl"
    run(judged(answer, grade=CORRECT, votes={"j1": "ENDORSE"}), rows, judged_out, events, judges=[J1])
    versus = build_report(rows, read_jsonl(judged_out))["rules_vs_judges"]
    assert {k: (v["judged"], v["graded_by_judges"], v["agree"]) for k, v in versus.items()} == {
        "dark full": (4, 4, 0), "both_right full": (2, 2, 0)}
    md = render_report(build_report(rows, read_jsonl(judged_out)))
    assert "| set | arm | judged | judges undecided | graded by the grading judges | agree |" in md
    assert "| dark | full | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |" in md
    assert "A row the grading judges graded is counted apart" in md


# ---- a judges' decision on resume: the name it was about, whatever the label (F1, F2) -------------------------------


def named_everywhere(req):
    meta = req.metadata
    if meta["set"] in ("donor_wrong", "control") or meta["arm"] == "full":
        return f"The first author is {meta['donor_answer']}."
    return DONT_KNOW


def test_every_labelled_row_stores_the_name_it_is_measured_against(tmp_path):
    events, rows = fixture_plan()
    got = run(judged(named_everywhere), rows, tmp_path / "keys.jsonl", events)
    planned = {r["id"]: r for r in rows}
    for row in got.rows:
        expected = planned[row["id"]]["donor_answer"] if row["adoption"] is not None else None
        assert row["adoption_name"] == expected


def test_a_judges_decision_stands_whatever_a_later_labelling_rule_says(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(named_everywhere, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1])
    decided = {r["id"]: r for r in read_jsonl(out) if r["adoption"] is not None}
    assert decided and all((r["adoption"], r["adopted"], r["adoption_grader"]) == ("endorsed", True, "judge:j1")
                           for r in decided.values())
    # a later refinement of the labelling rules reads every such answer as reported speech
    real_label = text_mod.adoption_label
    monkeypatch.setattr(text_mod, "adoption_label", lambda response, name, event: (
        "reported" if response.startswith("The first author is") else real_label(response, name, event)))
    for judges in (None, [J1]):
        backend = judged(named_everywhere, votes={"j1": "REJECT"})
        seen = []
        got = run(backend, rows, out, events, judges=judges, on_start=seen.append)
        assert backend.calls == []                                              # no recipient and no judge call
        current = {r["id"]: r for r in got.rows if r["id"] in decided}
        assert all((r["adoption"], r["adopted"], r["adoption_grader"], r["rule_adopted"]) ==
                   ("reported", True, "judge:j1", None) for r in current.values())
        assert all(r["grade"] == decided[i]["grade"] for i, r in current.items())
        if judges is None:
            assert seen[0].relabel == got.regraded == len(decided)               # refreshed once, without a call
        else:
            assert got.regraded == 0


def test_a_judges_decision_stored_without_rule_adopted_or_the_name_stands_on_a_changed_label(tmp_path, monkeypatch):
    # a results file written before rule_adopted and adoption_name were stored (as 863ca47 wrote it): the report says a
    # later run stores the rules' decision without a call, also when the label changed meanwhile
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(named_everywhere, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1])
    old = []
    for row in read_jsonl(out):
        row.pop("rule_adopted", None)
        row.pop("adoption_name", None)
        old.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
    labelled = {r["id"] for r in old if r["adoption"] is not None}
    assert "rows the judges decided carry no rule decision" in render_report(build_report(rows, old))
    real_label = text_mod.adoption_label
    monkeypatch.setattr(text_mod, "adoption_label", lambda response, name, event: (
        "reported" if response.startswith("The first author is") else real_label(response, name, event)))
    backend = judged(named_everywhere, votes={"j1": "REJECT"})
    got = run(backend, rows, out, events, judges=[J1])
    assert backend.calls == [] and got.regraded == len(labelled)
    planned = {r["id"]: r for r in rows}
    for row in (r for r in got.rows if r["id"] in labelled):
        assert (row["adopted"], row["adoption_grader"], row["rule_adopted"]) == (True, "judge:j1", None)
        assert row["adoption_name"] == planned[row["id"]]["donor_answer"]
    assert run(judged(named_everywhere), rows, out, events, judges=[J1]).regraded == 0


def asks(template):
    def answer(req):
        meta = req.metadata
        return template.format(X=meta["donor_answer"]) if meta["set"] in ("donor_wrong", "control") else DONT_KNOW
    return answer


def test_a_replan_to_a_name_the_answer_does_not_mention_leaves_the_row_to_the_rules(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("donor_wrong", "control")]
    out = tmp_path / "keys.jsonl"
    answer = asks("Is it {X}?")
    run(judged(answer, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1])
    assert all((r["adoption"], r["adopted"], r["adoption_grader"]) == ("absent", True, "judge:j1")
               for r in read_jsonl(out))
    replanned = [{**r, "donor_answer": "Zed Quux"} for r in rows]              # the same ids and messages
    backend = judged(answer, votes={"j1": "ENDORSE"})
    seen = []
    got = run(backend, replanned, out, events, on_start=seen.append)
    assert backend.calls == [] and seen[0].relabel == got.regraded == len(rows)
    assert all((r["adoption"], r["adopted"], r["adoption_grader"], r["adoption_name"]) ==
               ("absent", False, "rule", "Zed Quux") for r in got.rows)        # "Is it Wrong Person?" is no adoption
    backend = judged(answer, votes={"j1": "ENDORSE"})
    assert run(backend, replanned, out, events, judges=[J1]).regraded == 0 and backend.calls == []


def test_a_replan_to_a_name_the_answer_also_mentions_asks_the_judges_about_that_name(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("donor_wrong", "control")]
    out = tmp_path / "keys.jsonl"
    answer = asks("Is it {X} or Zed Quux?")
    run(judged(answer, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1])
    replanned = [{**r, "donor_answer": "Zed Quux"} for r in rows]
    got = run(judged(answer), replanned, out, events)                           # no judges: the rules, for Zed Quux
    assert all((r["adopted"], r["adoption_grader"], r["adoption_name"]) == (False, "rule", "Zed Quux")
               for r in got.rows)
    backend = judged(answer, votes={"j1": "REJECT"})
    got = run(backend, replanned, out, events, judges=[J1])
    asked = calls(backend, "adoption_judge")
    assert len(asked) == len(rows) and all("CANDIDATE:\nZed Quux\n" in req.messages[1]["content"] for _, req in asked)
    assert all((r["adopted"], r["adoption_grader"], r["adoption_name"]) == (False, "judge:j1", "Zed Quux")
               for r in got.rows)
