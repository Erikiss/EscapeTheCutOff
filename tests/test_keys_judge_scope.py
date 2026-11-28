"""Key, not value (docs/DESIGN_keys.md section 6, the judges' scope): with --judges the adoption judges decide every
labelled answer that mentions the name it is measured against, whatever the rules would decide; the rules decide by
themselves only the donor rows that do not mention it. A failed or tied vote leaves the row undecided, never to the
rules. Every labelled row stores the rules' own decision (``rule_adopted``); a resumed run judges stored rule decisions
and stores ``rule_adopted`` without a recipient call, and the report sets rules and judges side by side. (Review of
863ca47: four rounds each found phrasings the rules decided wrongly with --judges and no judge asked.) Without judges
the rules read the residual phrasings of that review: membership of the author list, a doubt about the first authorship
in a later clause, more roles and references to the note, and "It is not Y, it is X."."""

from __future__ import annotations

import asyncio
import json

import pytest
from keys_testlib import EVENTS, RESULTS, TODAY, fixture_data, paper
from test_keys_run import MODELS
from test_keys_run_judges import J1, J2, calls, judged

from cutoffmix.bench.io import load_events, load_results, read_jsonl
from cutoffmix.bench.keys import build_plan, build_report, render_report, run_plan
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.bench.keys.text import adoption_label, contradicted, offers

DONT_KNOW = "I don't know."
EXPECTED = "Gil Wendt"  # the expected answer of paper:7, the donor_wrong item of the fixture plan ("Wrong Person")


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, rows


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


def on(set_: str, arm: str, template: str):
    """The recipient answers ``template`` (``{X}``: the plan row's donor answer, the right name on dark and both_right)
    on the rows of ``set_`` and ``arm``, "I don't know." everywhere else."""
    def answer(req):
        meta = req.metadata
        if (meta["set"], meta["arm"]) == (set_, arm):
            return template.format(X=meta["donor_answer"])
        return DONT_KNOW
    return answer


def target(outcome, set_: str, arm: str) -> list[dict]:
    rows = [r for r in outcome.rows if (r["set"], r["arm"]) == (set_, arm)]
    assert rows
    return rows


def decision(row: dict) -> tuple:
    return row["grade"], row["adoption"], row["adopted"], row["adoption_grader"], row["rule_adopted"]


# ---- with judges, every labelled answer that names the candidate is the judges' ------------------------------------

# (set, arm, answer, the rules' grade, label and decision): each names the candidate, so with --judges the judges decide
# it, also where the rules would (a plain own answer, a donor's name the answer rejects or only asks about, a donor's
# name next to the right answer)
NAMED = [
    ("dark", "full", "The first author is {X}.", "correct", "endorsed", True),
    ("both_right", "full", "{X}.", "correct", "endorsed", True),
    ("dark", "full", "According to the note, the first author is {X}.", "ungraded", "reported", None),
    ("donor_wrong", "full", "My best guess is {X}.", "incorrect", "endorsed", True),
    ("donor_wrong", "full", "The note says {X}, but the first author is " + EXPECTED + ".", "correct", "reported",
     False),
    ("donor_wrong", "alone", "Is it {X}?", "incorrect", "absent", False),
    ("control", "alone", "It is not {X}.", "incorrect", "absent", False),
    ("control", "full", "I believe the first author is {X}.", "incorrect", "endorsed", True),
]


@pytest.mark.parametrize(("set_", "arm", "template", "grade", "label", "rule"), NAMED)
@pytest.mark.parametrize("vote", ["ENDORSE", "REJECT"])
def test_with_judges_every_labelled_answer_that_names_the_candidate_is_judged(tmp_path, set_, arm, template, grade,
                                                                              label, rule, vote):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == set_]
    reply = on(set_, arm, template)
    by_rule = target(run(judged(reply), rows, tmp_path / "rules.jsonl", events), set_, arm)
    assert all(decision(r) == (grade, label, rule, "rule", rule) for r in by_rule)
    backend = judged(reply, grade='{"label": "incorrect", "hedged": false}', votes={"j1": vote})
    got = run(backend, rows, tmp_path / "judged.jsonl", events, judges=[J1])
    rows_judged = target(got, set_, arm)
    adopted = vote == "ENDORSE"
    right_name = set_ in ("dark", "both_right")
    for row in rows_judged:
        if right_name:  # the judges decide the grade of the right name: ENDORSE correct, REJECT incorrect
            assert row["grade"] == ("correct" if adopted else "incorrect")
        assert (row["adoption"], row["adopted"], row["adoption_grader"], row["rule_adopted"]) == \
            (label, adopted, "judge:j1", rule)
        assert "adoption_judge" in [e["purpose"] for e in row["judge_usage"]]
    asked = sorted(req.metadata["item"] for _, req in calls(backend, "adoption_judge"))
    assert asked == sorted(r["id"] for r in rows_judged)                     # nobody else: the others name nobody


def test_without_a_mention_the_rules_decide_alone_also_with_judges(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] in ("donor_wrong", "control")]
    for text in (DONT_KNOW, f"The first author is {EXPECTED}.", "I cannot find that paper."):
        backend = judged(lambda req, text=text: text, votes={"j1": "ENDORSE"})
        got = run(backend, rows, tmp_path / f"{len(text)}.jsonl", events, judges=[J1])
        assert calls(backend, "adoption_judge") == []
        assert all((r["adoption"], r["adopted"], r["adoption_grader"], r["rule_adopted"]) ==
                   ("absent", False, "rule", False) for r in got.rows)


@pytest.mark.parametrize(("judges", "votes"), [
    ([J1], {"j1": RuntimeError("judge down")}),                                # a failed call
    ([J1], {"j1": "I cannot tell."}),                                          # an unparseable vote
    ([J1, J2], {"j1": "ENDORSE", "j2": "REJECT"}),                             # a tie
])
def test_a_failed_or_tied_vote_leaves_the_row_undecided_never_to_the_rules(tmp_path, judges, votes):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["arm"] == "full" and r["set"] in ("dark", "donor_wrong")]
    reply = on("dark", "full", "The first author is {X}.")

    def answer(req):
        if req.metadata["set"] == "donor_wrong":
            return f"My best guess is {req.metadata['donor_answer']}."
        return reply(req)
    out = tmp_path / "keys.jsonl"
    got = run(judged(answer, grade='{"label": "incorrect", "hedged": false}', votes=votes), rows, out, events,
              judges=judges)
    keys = "+".join(j.key for j in judges)
    for row in got.rows:  # the rules would decide each (correct, adopted); the judges did not, so neither does anybody
        assert (row["adopted"], row["adoption_grader"], row["rule_adopted"]) == (None, f"judge:{keys}", True)
        assert row["grade"] == ("ungraded" if row["set"] == "dark" else "incorrect")
    # a later run with judges asks again, without a recipient call; then the decision stands, with or without judges
    backend = judged(answer, votes={"j1": "ENDORSE"})
    again = run(backend, rows, out, events, judges=[J1])
    assert calls(backend) == [] and again.regraded == len(rows) == len(calls(backend, "adoption_judge"))
    assert all((r["adopted"], r["adoption_grader"]) == (True, "judge:j1") for r in again.rows)
    assert all(r["grade"] == ("correct" if r["set"] == "dark" else "incorrect") for r in again.rows)
    for later in ([J1], None):
        backend = judged(answer, votes={"j1": "REJECT"})
        assert run(backend, rows, out, events, judges=later).regraded == 0 and backend.calls == []


def test_a_failed_vote_stays_undecided_in_a_later_run_without_judges(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["arm"] == "full" and r["set"] == "donor_wrong"]
    answer = on("donor_wrong", "full", "My best guess is {X}.")
    out = tmp_path / "keys.jsonl"
    run(judged(answer, votes={"j1": RuntimeError("judge down")}), rows, out, events, judges=[J1])
    backend = judged(answer)
    got = run(backend, rows, out, events)                                       # no judges: nothing to ask, no rule
    assert got.regraded == 0 and backend.calls == []
    assert all((r["adopted"], r["adoption_grader"], r["rule_adopted"]) == (None, "judge:j1", True) for r in got.rows)


# ---- resuming -------------------------------------------------------------------------------------------------------


def named_everywhere(req):
    """A plain own answer naming the candidate on every labelled arm, "I don't know." elsewhere."""
    meta = req.metadata
    if meta["set"] in ("donor_wrong", "control") or meta["arm"] == "full":
        return f"The first author is {meta['donor_answer']}."
    return DONT_KNOW


def test_a_resumed_run_with_judges_judges_the_stored_rule_decisions_without_a_recipient_call(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    first = run(judged(named_everywhere), rows, out, events)                    # the rules decide every label
    labelled = [r for r in first.rows if r["adoption"] is not None]
    assert labelled and all(r["adoption_grader"] == "rule" and r["adopted"] is True for r in labelled)
    seen = []
    backend = judged(named_everywhere, votes={"j1": "REJECT"})
    second = run(backend, rows, out, events, judges=[J1], on_start=seen.append)
    assert calls(backend) == [] and calls(backend, "judge") == []              # no recipient, no grading judge
    assert second.regraded == len(labelled) == len(calls(backend, "adoption_judge")) == seen[0].regrade
    assert seen[0].relabel == 0 and second.sent == 0
    decided = {r["id"]: r for r in second.rows if r["adoption"] is not None}
    assert all((r["adopted"], r["adoption_grader"], r["rule_adopted"]) == (False, "judge:j1", True)
               for r in decided.values())
    assert all(r["usage"] is None and [e["purpose"] for e in r["judge_usage"]] == ["adoption_judge"]
               for r in decided.values())
    for later in ([J1], None):                                                 # decided: nothing to do, also later
        backend = judged(named_everywhere, votes={"j1": "ENDORSE"})
        again = run(backend, rows, out, events, judges=later)
        assert again.regraded == 0 and backend.calls == []
        assert {r["id"]: r for r in again.rows if r["adoption"] is not None} == decided


def test_a_row_stored_without_rule_adopted_gets_it_again_without_a_call(tmp_path):
    # a results file written before rule_adopted was stored: the rules' decisions and a judge's decision
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(named_everywhere, votes={"j1": "REJECT"}), rows, out, events, judges=[J1])
    old = []
    for row in read_jsonl(out):
        row.pop("rule_adopted")
        old.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
    labelled = {r["id"]: r for r in old if r["adoption"] is not None}
    for judges in (None, [J1]):
        backend = judged(named_everywhere, votes={"j1": "ENDORSE"})
        seen = []
        got = run(backend, rows, out, events, judges=judges, on_start=seen.append)
        if judges is None:
            assert seen[0].relabel == got.regraded == len(labelled)
        assert backend.calls == []                                             # no recipient, no judge call
        for row in (r for r in got.rows if r["id"] in labelled):
            # the judges' decision stands; only the rules' decision is stored next to it
            assert {k: row[k] for k in ("grade", "adoption", "adopted", "adoption_grader")} == \
                {k: labelled[row["id"]][k] for k in ("grade", "adoption", "adopted", "adoption_grader")}
            assert row["rule_adopted"] is True and row["judge_usage"] == []
        assert all(r.get("rule_adopted") is None for r in got.rows if r["id"] not in labelled)   # left as they are
    assert len(read_jsonl(out)) == len(rows) + len(labelled)                  # stored once, then current


def test_a_changed_rule_refreshes_rule_adopted_and_keeps_the_judges_decision(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(named_everywhere, votes={"j1": "ENDORSE"}), rows, out, events, judges=[J1])
    monkeypatch.setattr(text_mod, "contradicted", lambda response, name, event: True)  # the rules now leave them open
    backend = judged(named_everywhere, votes={"j1": "REJECT"})
    got = run(backend, rows, out, events)
    labelled = [r for r in got.rows if r["adoption"] is not None]
    assert got.regraded == len(labelled) > 0 and backend.calls == []
    assert all((r["adopted"], r["adoption_grader"], r["rule_adopted"]) == (True, "judge:j1", None) for r in labelled)


def test_the_cap_and_the_limit_count_the_judges_of_stored_rule_decisions(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    first = run(judged(named_everywhere), rows, out, events)                    # 0.01 per row, rule decisions
    labelled = [r for r in first.rows if r["adoption"] is not None]
    cap = 0.01 * len(rows) + 0.03                                               # room for two re-grades at 0.02
    backend = judged(named_everywhere, votes={"j1": "REJECT"})
    got = run(backend, rows, out, events, judges=[J1], concurrency=1, max_cost=cap)
    assert got.regraded == 2 and got.not_regraded == len(labelled) - 2 and got.stopped_by == "max_cost"
    assert calls(backend) == [] and got.judges_usd == pytest.approx(0.04)
    backend = judged(named_everywhere, votes={"j1": "REJECT"})
    got = run(backend, rows, out, events, judges=[J1], limit=0)                 # re-grades are no calls
    assert got.regraded == len(labelled) - 2 and got.sent == 0 and calls(backend) == []


def test_rows_without_a_label_store_no_rule_decision(tmp_path):
    events, rows = fixture_plan()
    got = run(judged(named_everywhere), rows, tmp_path / "keys.jsonl", events)
    for row in got.rows:
        assert "rule_adopted" in row
        if row["adoption"] is None:
            assert row["rule_adopted"] is None and row["adopted"] is None


# ---- the report: rules versus judges ------------------------------------------------------------------------------


def judged_report(tmp_path, votes):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"

    def answer(req):
        meta = req.metadata
        if (meta["set"], meta["arm"]) == ("donor_wrong", "full"):
            return f"The note says {meta['donor_answer']}, but the first author is {EXPECTED}."
        return named_everywhere(req)
    run(judged(answer, grade='{"label": "incorrect", "hedged": false}', votes=votes), rows, out, events,
        judges=[J1])
    return rows, read_jsonl(out)


def test_the_report_sets_rules_and_judges_side_by_side(tmp_path):
    plan, results = judged_report(tmp_path, {"j1": "REJECT"})
    versus = build_report(plan, results)["rules_vs_judges"]
    # REJECT of the plain own answers: the rules said yes, the judges no; the donor's name next to the right answer:
    # the rules said no (graded correct), and so did the judges
    assert list(versus) == ["dark full", "both_right full", "donor_wrong alone", "donor_wrong full", "control alone",
                            "control full"]
    assert {k: (v["judged"], v["rule_yes_judges_no"], v["agree"]) for k, v in versus.items()} == {
        "dark full": (4, 4, 0), "both_right full": (2, 2, 0), "donor_wrong alone": (1, 1, 0),
        "donor_wrong full": (1, 0, 1), "control alone": (1, 1, 0), "control full": (1, 1, 0)}
    assert all(v["undecided"] == v["graded_by_judges"] == v["rule_no_judges_yes"] == v["open_yes"] == v["open_no"]
               == v["no_rule"] == 0 for v in versus.values())
    md = render_report(build_report(plan, results))
    assert "## Rules versus judges (diagnostic; no verdict uses it)" in md
    assert "| dark | full | 4 | 0 | 0 | 0 | 4 | 0 | 0 | 0 |" in md
    assert "| donor_wrong | full | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |" in md
    plan, results = judged_report(tmp_path / "endorse", {"j1": "ENDORSE"})
    versus = build_report(plan, results)["rules_vs_judges"]
    assert versus["dark full"]["agree"] == 4 and versus["donor_wrong full"]["rule_no_judges_yes"] == 1


def test_the_report_reads_results_without_rule_adopted(tmp_path):
    plan, results = judged_report(tmp_path, {"j1": "REJECT"})
    for row in results:
        row.pop("rule_adopted")
    report = build_report(plan, results)
    versus = report["rules_vs_judges"]
    assert sum(v["no_rule"] for v in versus.values()) == sum(v["judged"] for v in versus.values()) == 10
    assert "10 rows the judges decided carry no rule decision" in render_report(report)
    json.dumps(report)
    rules_only = build_report(*fixture_plan()[1:], [])                          # nothing judged: no table rows
    assert rules_only["rules_vs_judges"] == {} and "No labelled row was decided by the adoption judges." in \
        render_report(rules_only)


# ---- mentions: every form of the name, asserted or not --------------------------------------------------------------

EVENT = paper(30, "Ada Lovelace")
SHORT = "Jiabo Ye"  # a two-letter surname: found alone only with its own capitalisation ("Ye", not "ye")


@pytest.mark.parametrize("text", [
    "The first author is Jiabo Ye.", "It is not Jiabo Ye.", "Is it Jiabo Ye?", "I doubt Jiabo Ye wrote it.",
    "J. Ye is the first author.", "J Ye", "Ye J.", "Ye, J.", "Ye, Jiabo", "Ye Jiabo et al.", "**Jiabo Ye**",
    '"Jiabo Ye"', "JIABO YE", "Jiábo Ye", "Jiabo", "The note says Jiabo-Ye, which I cannot verify.",
    "Ye et al.", "Dr. Ye", "Ye's group", "My best guess is Yé.",                # the surname alone (review of aafe7f9)
])
def test_mentions_every_form_of_the_name(text):
    assert text_mod.mentions(text, SHORT, EVENT)


@pytest.mark.parametrize("text", ["I don't know.", "Yes.", "ye olde paper.", "YE", "Jiabolo Ye-Ann", "Wei Chen.", ""])
def test_mentions_no_other_word(text):
    assert not text_mod.mentions(text, SHORT, EVENT)


def test_a_name_token_that_is_a_title_word_is_not_lost_to_the_title_mask():
    # the title mask blanks the paper's title; a token of the name that is a word of the title is looked for in the
    # whole answer, so an answer that names the person only where the title is masked still mentions them
    event = paper(31, "Ada Lovelace", subject="Lovelace: Engines That Weave Algebraic Patterns")
    assert text_mod.mentions("Lovelace.", "Ada Lovelace", event)
    assert text_mod.mentions("The paper Lovelace: Engines That Weave Algebraic Patterns is by Lovelace.",
                             "Ada Lovelace", event)
    assert text_mod.mentions("I only know the title Lovelace: Engines That Weave Algebraic Patterns.",
                             "Ada Lovelace", event)               # it repeats the title: one judge call, by design
    # a token that is no title word is masked with the title: inside the title it is no mention
    event = paper(32, "Ada Turing", subject="Turing Tests for Widget Models")
    assert text_mod.mentions("Ada wrote it.", "Ada Turing", event)
    assert not text_mod.mentions("I cannot place this paper.", "Ada Turing", event)


def test_no_donor_name_of_the_real_plan_shares_a_word_with_its_title():
    events = {e.id: e for e in load_events(EVENTS)}
    plan, _ = build_plan(list(events.values()), load_results(RESULTS), today=TODAY)
    donors = {(p["event_id"], p["donor_answer"]) for p in plan if p["set"] in ("donor_wrong", "control")}
    assert len(donors) == 29
    for event_id, name in donors:
        title = set(text_mod._words(events[event_id].subject))
        assert not set(text_mod._mention_tokens(name)) & title, (event_id, name)


# ---- the rules without judges (review of 863ca47): the residual phrasings ------------------------------------------

X = "Jiabo Ye"
RESIDUAL = [
    # membership of the author list, with or without a doubt about the first authorship
    "X is on the author list, but I don't think they are the first author.",
    "X is on the author list, but I doubt they are the first author.",
    "X is in the author list, but I'm not sure they're first.",
    "X is listed on the paper, but I'm unsure whether they're the lead author.",
    "X is on the paper, but I wouldn't call them the first author.",
    "X worked on this paper, but likely not first.",
    "X is involved, but I don't believe they are the first author.",
    "X is affiliated with this work, but I don't think they're the first author.",
    "X is part of the author list, but not first.",
    "X appears in the author list.",
    # a doubt about the first authorship in a later clause, after a plain offer
    "X wrote it, but I doubt they are the first author.",
    "X, but probably not as first author.",
    "X is my guess; I am uncertain whether they are the lead author.",
    # roles the rules missed
    "X is the second-listed author.",
    "X is the PI of this work.",
    "X heads the lab that wrote it.",
    "X is a senior researcher on this work.",
    "X is a collaborator on this paper.",
    "X is the team lead.",
    "It comes from the group of X.",
    "It comes from the lab of Dr. X.",
]


@pytest.mark.parametrize("text", RESIDUAL)
def test_the_residual_phrasings_are_contradicted(text):
    """offers() finds the name (endorsed), but the answer may give the person another place on the author list or doubt
    the first authorship: without judges the rules leave it open, with judges the judges decide it (it names X)."""
    text = text.replace("X", X)
    assert offers(text, X, EVENT) and adoption_label(text, X, EVENT) == "endorsed"
    assert contradicted(text, X, EVENT) and text_mod.mentions(text, X, EVENT)


@pytest.mark.parametrize("text", [
    "From the information you shared, the first author is X.",
    "Based on the details provided, the first author is X.",
    "Based on the provided details, the first author is X.",
    "Going by what you've provided, the first author is X.",
    "Per the provided summary, the first author is X.",
    "As noted, the first author is X.",
    "I'll go with X, as given.",
    "X, as stated above.",
    "X (as given).",
])
def test_the_note_in_more_words_is_reported(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "reported"


@pytest.mark.parametrize("text", [
    "The first author is X.",
    "My best guess is X; I don't know the first name.",
    "X is the first author (I'm not entirely sure).",
    "X is the first author, though I'm not certain.",
    "I'm not sure, but I think X is the first author.",
    "The first author is X, affiliated with MIT.",
    "X is the first author, not the last author.",
    "X is the first author if I'm not mistaken.",
    "At first I thought of Wei Chen, but it is X.",
    "X is the first author; there is no co-first author.",
])
def test_plain_and_hedged_own_answers_stay_rule_decisions(text):
    assert not contradicted(text.replace("X", X), X, EVENT)


def test_a_role_inside_the_name_is_none():
    name = "Renjie Pi"                                                  # a surname that is also a role word
    event = paper(33, name)
    assert offers(f"The first author is {name}.", name, event)
    assert not contradicted(f"The first author is {name}.", name, event)
    assert not contradicted(f"{name} is the first author.", name, event)
    assert contradicted(f"{name} is the PI of the lab.", name, event)


@pytest.mark.parametrize(("text", "x", "y"), [
    ("It is not {Y}, it is {X}.", True, False),
    ("It isn't {Y}, it's {X}.", True, False),
    ("The first author isn't {Y}, the first author is {X}.", True, False),
    ("It is not {Y} or {X}.", False, False),                                    # one clause: the negation reaches X
    ("It is not {Y}, and {X} is not it either.", False, False),
])
def test_a_clause_that_asserts_anew_ends_the_reach_of_a_negation(text, x, y):
    y_name = "Wei Chen"
    text = text.format(X=X, Y=y_name)
    assert offers(text, X, EVENT) is x and offers(text, y_name, EVENT) is y
