"""Key, not value (docs/DESIGN_keys.md section 6): the judges of a keys run. Their calls are metered per row
(``judge_usage``) and counted by the cost cap; stored answers the rules left undecided are graded again without a new
recipient call; answers that name the donor's wrong name only in reported speech go to the adoption judge."""

from __future__ import annotations

import asyncio
import json

import pytest
from keys_testlib import TODAY, fixture_data
from test_keys_run import MODELS, ScriptedBackend

from cutoffmix.bench.io import read_jsonl
from cutoffmix.bench.keys import build_plan, run_plan
from cutoffmix.bench.keys.run import ADOPTION_JUDGE_SYSTEM, parse_adoption_vote
from cutoffmix.types import ChatResult, ModelSpec

UNDECIDED = "I can't reliably verify the first author from the title alone."     # the rules leave this ungraded
J1, J2, J3 = (ModelSpec(key=k, provider_id=f"judge/{k}") for k in ("j1", "j2", "j3"))


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, rows


def judged(answer, *, grade='{"label": "abstain", "hedged": false}', votes=None, recipient_cost=0.01,
           judge_cost=0.02):
    """A backend whose recipient answers ``answer(req)``, whose grading judges answer ``grade`` and whose adoption
    judges answer ``votes[judge key]``; every call costs ``recipient_cost`` or ``judge_cost``."""
    votes = votes or {}

    def reply(model, req):
        purpose = req.metadata.get("purpose")
        if purpose == "judge":
            return ChatResult(text=grade, usage={"cost": judge_cost}, finish_reason="stop")
        if purpose == "adoption_judge":
            vote = votes.get(model.key, "REPORT")
            if isinstance(vote, BaseException):
                return vote
            return ChatResult(text=vote, usage={"cost": judge_cost}, finish_reason="stop")
        out = answer(req)
        if isinstance(out, (ChatResult, BaseException)):
            return out
        return ChatResult(text=out, usage={"cost": recipient_cost}, finish_reason="stop", latency_s=0.5)
    return ScriptedBackend(reply)


def calls(backend, purpose=None):
    return [(key, req) for key, req in backend.calls if req.metadata.get("purpose") == purpose]


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


# ---- metering and the cap -----------------------------------------------------------------------------------------


def test_judge_usage_is_metered_per_row_and_counted_by_the_cap(tmp_path):
    events, rows = fixture_plan()
    backend = judged(lambda req: UNDECIDED)
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, judges=[J1], concurrency=1, max_cost=0.05)
    assert len(calls(backend)) == 2 and len(calls(backend, "judge")) == 2      # 0.03 per item: stops at 0.06
    assert got.stopped_by == "max_cost" and got.sent == 2
    assert got.cost_usd == pytest.approx(0.06) and got.judges_usd == pytest.approx(0.04)
    assert got.recipients_usd == pytest.approx(0.02)
    for row in read_jsonl(out):
        assert row["grade"] == "abstain" and row["grader"] == "judge:j1"
        assert row["judge_usage"] == [{"judge": "j1", "purpose": "judge", "cost": 0.02, "usage": {"cost": 0.02}}]
        assert row["usage"] == {"cost": 0.01}                                   # the recipient's own usage
    resumed = judged(lambda req: UNDECIDED)
    again = run(resumed, rows, out, events, judges=[J1], max_cost=0.06)          # the stored judges count too
    assert resumed.calls == [] and again.stopped_by == "max_cost" and again.start.cost_usd == pytest.approx(0.06)


def test_judges_of_an_item_in_flight_finish_when_the_cap_is_reached(tmp_path):
    events, rows = fixture_plan()
    backend = judged(lambda req: UNDECIDED, grade="no idea")                     # unparseable: a third judge is asked
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1, J2, J3], concurrency=1, max_cost=0.001)
    assert len(calls(backend)) == 1 and len(calls(backend, "judge")) == 3
    row = got.rows[0]
    assert row["grade"] == "ungraded" and row["grader"] == "judge:j1+j2+j3" and len(row["judge_usage"]) == 3
    assert got.cost_usd == pytest.approx(0.07) and got.stopped_by == "max_cost"


def test_a_rule_graded_answer_calls_no_judge(tmp_path):
    events, rows = fixture_plan()
    backend = judged(lambda req: "I don't know.")
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1])
    assert calls(backend, "judge") == [] and all(r["judge_usage"] == [] for r in got.rows)
    assert got.judges_usd == 0.0


# ---- re-grading stored rows ---------------------------------------------------------------------------------------


def test_stored_undecided_answers_are_graded_again_without_a_new_call(tmp_path):
    events, rows = fixture_plan()

    def answer(req):
        if req.metadata["arm"] == "full":
            return ChatResult(text="", finish_reason="content_filter", usage={"cost": 0.0})
        return UNDECIDED
    out = tmp_path / "keys.jsonl"
    first = run(judged(answer), rows, out, events)                               # the rules only, as without --judges
    undecided = [r for r in first.rows if r["grade"] == "ungraded" and r["grader"] == "rule"]
    refusals = [r for r in first.rows if r["grader"] == "refusal"]
    assert undecided and refusals and len(undecided) + len(refusals) == len(rows)
    backend = judged(answer)
    seen = []
    second = run(backend, rows, out, events, judges=[J1], on_start=seen.append)
    assert calls(backend) == []                                                  # no recipient call at all
    assert len(calls(backend, "judge")) == len(undecided) == second.regraded == seen[0].regrade
    assert second.sent == 0 and second.stopped_by is None and second.not_sent == 0
    stored = read_jsonl(out)
    assert len(stored) == len(rows) + len(undecided)                             # appended under the same ids
    old = {r["id"]: r for r in first.rows}
    last = {r["id"]: r for r in stored}
    for row in undecided:
        new = last[row["id"]]
        assert new["grade"] == "abstain" and new["grader"] == "judge:j1" and new["response"] == UNDECIDED
        assert new["usage"] is None and new["latency_s"] == old[row["id"]]["latency_s"]
        assert [e["purpose"] for e in new["judge_usage"]] == ["judge"]
        assert {k: new[k] for k in ("backend", "prompt_sha", "request_sha", "note_event", "today")} == \
               {k: old[row["id"]][k] for k in ("backend", "prompt_sha", "request_sha", "note_event", "today")}
    assert all(last[r["id"]] == r for r in refusals)                             # a refusal is never judged
    # each paid call counts once: the re-graded rows carry only their judges
    assert second.recipients_usd == pytest.approx(first.recipients_usd) == pytest.approx(0.01 * len(undecided))
    assert second.judges_usd == pytest.approx(0.02 * len(undecided))
    third = judged(answer)
    assert run(third, rows, out, events, judges=[J1]).regraded == 0 and third.calls == []
    assert run(judged(answer), rows, out, events).regraded == 0                  # nothing without judges


def test_a_regrade_recomputes_adopted(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "control"]
    monkeypatch.setattr("cutoffmix.bench.keys.text.adoption_label",
                        lambda response, name, event: "endorsed" if name in response else "absent")
    text = "I can't find this paper. The note says the first author is Made Upname, but I can't verify that."
    out = tmp_path / "keys.jsonl"
    first = run(judged(lambda req: text), rows, out, events)
    # endorsed, but the rules leave the grade open: no rule decision (the grader read no wrong name given)
    assert {r["grade"] for r in first.rows} == {"ungraded"} and {r["adopted"] for r in first.rows} == {None}
    second = run(judged(lambda req: text, grade='{"label": "correct", "hedged": true}'), rows, out, events,
                 judges=[J1])
    assert second.regraded == len(rows)
    assert {r["grade"] for r in second.rows} == {"correct"} and {r["adopted"] for r in second.rows} == {False}


def test_the_cap_stops_the_regrades_too(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(lambda req: UNDECIDED), rows, out, events)                        # 0.01 per row
    backend = judged(lambda req: UNDECIDED)
    cap = 0.01 * len(rows) + 0.03                                               # room for two re-grades at 0.02
    got = run(backend, rows, out, events, judges=[J1], concurrency=1, max_cost=cap)
    assert got.regraded == 2 and got.not_regraded == len(rows) - 2 and got.stopped_by == "max_cost"
    assert calls(backend) == [] and got.not_sent == 0


# ---- the adoption judge -------------------------------------------------------------------------------------------


@pytest.fixture
def reported(monkeypatch):
    """``adoption_label`` reads "the note says ..." as reported speech (the plan side implements the real rule)."""
    def label(response, name, event):
        if name in response and "the note says" in response.lower():
            return "reported"
        return "endorsed" if name in response else "absent"
    monkeypatch.setattr("cutoffmix.bench.keys.text.adoption_label", label)


def adoption_plan():
    events, rows = fixture_plan()
    return events, [r for r in rows if r["set"] in ("donor_wrong", "control")]


def note_says(req):
    if req.metadata["arm"] == "full":
        return f"The note says the first author is {req.metadata['donor_answer']}, but I cannot verify it."
    return "No idea."


def test_reported_speech_stays_undecided_without_judges(tmp_path, reported):
    events, rows = adoption_plan()
    got = run(judged(note_says), rows, tmp_path / "keys.jsonl", events)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert full and all(r["adoption"] == "reported" and r["adopted"] is None and r["adoption_grader"] == "rule"
                        for r in full)
    alone = [r for r in got.rows if r["arm"] == "alone"]
    assert all(r["adoption"] == "absent" and r["adopted"] is False for r in alone)


@pytest.mark.parametrize(("vote", "adopted"), [("ENDORSE", True), ("REPORT", False), ("REJECT", False)])
def test_the_adoption_judge_decides_reported_speech(tmp_path, reported, vote, adopted):
    events, rows = adoption_plan()
    backend = judged(note_says, grade='{"label": "incorrect", "hedged": true}', votes={"j1": vote})
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=[J1])
    full = {r["id"]: r for r in got.rows if r["arm"] == "full"}
    assert full and all(r["adopted"] is adopted and r["adoption"] == "reported" and r["adoption_grader"] == "judge:j1"
                        for r in full.values())
    asked = calls(backend, "adoption_judge")
    assert sorted(req.metadata["item"] for _, req in asked) == sorted(full)
    plan_rows = {r["id"]: r for r in rows}
    for key, req in asked:
        assert key == "j1" and req.metadata == {"purpose": "adoption_judge", "item": req.metadata["item"]}
        assert req.messages[0] == {"role": "system", "content": ADOPTION_JUDGE_SYSTEM}
        assert plan_rows[req.metadata["item"]]["donor_answer"] in req.messages[1]["content"]
        assert full[req.metadata["item"]]["response"] in req.messages[1]["content"]
        assert req.temperature == 0.0 and req.max_tokens == 1024
    for row in full.values():                                                    # both judge calls are metered
        assert [e["purpose"] for e in row["judge_usage"]] == ["judge", "adoption_judge"]
    assert got.judges_usd == pytest.approx(0.04 * len(full))
    assert calls(backend, "adoption_judge") and all(r["adoption_grader"] == "rule"
                                                    for r in got.rows if r["arm"] == "alone")


@pytest.mark.parametrize(("judges", "votes", "adopted", "grader"), [
    ([J1, J2], {"j1": "ENDORSE", "j2": "ENDORSE"}, True, "judge:j1+j2"),
    ([J1, J2], {"j1": "ENDORSE", "j2": "REPORT"}, None, "judge:j1+j2"),          # a tie without a third judge
    ([J1, J2, J3], {"j1": "ENDORSE", "j2": "REPORT", "j3": "REPORT"}, False, "judge:j2+j3"),
    ([J1, J2, J3], {"j1": "ENDORSE", "j2": "REPORT", "j3": "REJECT"}, None, "judge:j1+j2+j3"),
    ([J1], {"j1": "I am not sure."}, None, "judge:j1"),                          # unparseable
    ([J1], {"j1": "The answer does not endorse the candidate."}, None, "judge:j1"),   # a negation is no ENDORSE
    ([J1, J2], {"j1": "Not ENDORSE.", "j2": "ENDORSE"}, None, "judge:j1+j2"),   # one vote is no majority of two
    ([J1], {"j1": RuntimeError("judge down")}, None, "judge:j1"),                # a failed judge is no vote
])
def test_the_adoption_majority(tmp_path, reported, judges, votes, adopted, grader):
    events, rows = adoption_plan()
    backend = judged(note_says, grade='{"label": "incorrect", "hedged": true}', votes=votes)
    got = run(backend, rows, tmp_path / "keys.jsonl", events, judges=judges)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert full and all(r["adopted"] is adopted and r["adoption_grader"] == grader for r in full)
    per_item = len(calls(backend, "adoption_judge")) / len(full)
    assert per_item == (3 if len(judges) == 3 else len(judges))               # the third only on disagreement


def test_reported_rows_are_judged_again_later(tmp_path, reported):
    events, rows = adoption_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(note_says, grade='{"label": "incorrect", "hedged": true}'), rows, out, events)
    backend = judged(note_says, grade='{"label": "incorrect", "hedged": true}', votes={"j1": "ENDORSE"})
    got = run(backend, rows, out, events, judges=[J1])
    full = [r for r in got.rows if r["arm"] == "full"]
    assert calls(backend) == [] and got.regraded == len(full)
    assert len(calls(backend, "adoption_judge")) == len(full)
    assert all(r["adopted"] is True and r["adoption_grader"] == "judge:j1" for r in full)
    assert all(r["usage"] is None for r in full)
    assert run(judged(note_says), rows, out, events, judges=[J1]).regraded == 0  # decided now


# Rejections of the donor's name and endorsements the answer contradicts (final review), with the real labelling rule:
# (answer on the full arm, label, adopted by rule; None: the judges decide). They were endorsed and adopted by rule,
# which inflated 3b, also where the grader read the answer as an abstention.
CONTRADICTED = [
    ("{X} is incorrect; I do not know who the first author is.", "absent", False),
    ("I disagree with {X}; I do not know who the first author is.", "absent", False),
    ("The provided text says {X}, but I do not know who the first author is.", "reported", None),
    ("{X} is a researcher in this area, but I cannot find this paper.", "endorsed", None),   # graded no name given
    ("It is {X}; Yara Okafor is also possible.", "endorsed", None),                         # someone else as well
    ("My best guess is {X}.", "endorsed", True),                                           # nothing contradicts it
    # another role, or a rejection in words offers() does not read as negations (review of the final review): graded
    # incorrect, endorsed and adopted by rule, also with --judges
    ("{X} is a co-author, not the first author.", "endorsed", None),
    ("{X} is one of the authors, but not the first author.", "endorsed", None),
    ("{X} is listed among the authors, but I am not sure they are the first author.", "endorsed", None),
    ("{X} is the corresponding author, so the first author is probably someone in that group.", "endorsed", None),
    ("{X} is the last author of this paper.", "endorsed", None),
    ("{X} is a co-author of this paper, but I'm not sure who the first author is.", "endorsed", None),
    ("{X} is the senior author; I'm not certain who the first author is.", "endorsed", None),
    ("{X} supervised this work, but I'm not sure he is the first author.", "endorsed", None),
    ("{X} contributed to the paper, though was not the lead author.", "endorsed", None),
    ("{X} as the first author is questionable.", "endorsed", None),
    ("I dispute that {X} is the first author.", "endorsed", None),
    ("I reject {X} as the first author.", "endorsed", None),
    ("The paper is misattributed to {X}.", "endorsed", None),
    ("{X} as the first author is an error.", "endorsed", None),
    ("Rather than {X}, I would say the first author is unknown to me.", "endorsed", None),
    ("Instead of {X}, the first author should be someone else.", "endorsed", None),
    ("The claim that {X} is the first author is untrue.", "reported", None),
    # a first-person hedge rejects nothing: these were absent, not adopted by rule
    ("Unless I'm mistaken, the first author is {X}.", "endorsed", True),
    ("I could be wrong, the first author is {X}.", "endorsed", True),
    ("If I'm not mistaken, the first author is {X}.", "endorsed", True),
]


def on_full(template):
    def answer(req):
        return template.format(X=req.metadata["donor_answer"]) if req.metadata["arm"] == "full" else "No idea."
    return answer


@pytest.mark.parametrize(("template", "label", "adopted"), CONTRADICTED)
def test_a_rejected_or_contradicted_donor_name_is_no_adoption_by_rule(tmp_path, template, label, adopted):
    events, rows = adoption_plan()
    got = run(judged(on_full(template)), rows, tmp_path / "keys.jsonl", events)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert full and all((r["adoption"], r["adopted"], r["adoption_grader"]) == (label, adopted, "rule") for r in full)
    backend = judged(on_full(template), votes={"j1": "REJECT"})
    got = run(backend, rows, tmp_path / "judged.jsonl", events, judges=[J1])
    full = [r for r in got.rows if r["arm"] == "full"]
    expected = (label, adopted, "rule") if adopted is not None else (label, False, "judge:j1")
    assert all((r["adoption"], r["adopted"], r["adoption_grader"]) == expected for r in full)
    assert len(calls(backend, "adoption_judge")) == (len(full) if adopted is None else 0)


def test_a_stored_adoption_the_old_rule_decided_is_labelled_again_without_a_call(tmp_path):
    # a results file written before the review of the final review: "X is a co-author, not the first author" was
    # adopted by rule; the rules now leave it to the judges, so a resume labels it again from the stored response
    events, rows = adoption_plan()
    out = tmp_path / "keys.jsonl"
    template = "{X} is a co-author, not the first author."
    run(judged(on_full(template)), rows, out, events)
    old = []
    for row in read_jsonl(out):
        if row["arm"] == "full":
            assert (row["adoption"], row["adopted"]) == ("endorsed", None)
            row.update(adopted=True, adoption_grader="rule")                     # as the old rule wrote it
        old.append(row)
    out.write_text("".join(json.dumps(r) + "\n" for r in old), encoding="utf-8")
    backend = judged(on_full(template))
    got = run(backend, rows, out, events)
    full = [r for r in got.rows if r["arm"] == "full"]
    assert calls(backend) == [] and got.start.relabel == got.regraded == len(full) > 0
    assert all((r["adoption"], r["adopted"], r["adoption_grader"], r["usage"]) == ("endorsed", None, "rule", None)
               for r in full)


def test_stored_undecided_endorsements_are_judged_again_on_resume(tmp_path):
    # the grading judge read the answer as an abstention and the adoption judge failed: endorsed, undecided
    events, rows = adoption_plan()
    answer = on_full("{X} is a researcher in this area, but I cannot find this paper.")
    out = tmp_path / "keys.jsonl"
    first = run(judged(answer, votes={"j1": RuntimeError("judge down")}), rows, out, events, judges=[J1])
    full = [r for r in first.rows if r["arm"] == "full"]
    assert full and all((r["grade"], r["adoption"], r["adopted"]) == ("abstain", "endorsed", None) for r in full)
    backend = judged(answer, votes={"j1": "REJECT"})
    second = run(backend, rows, out, events, judges=[J1])
    assert calls(backend) == [] and calls(backend, "judge") == []              # no recipient, no grading judge
    assert len(calls(backend, "adoption_judge")) == len(full) == second.regraded
    assert all((r["adopted"], r["adoption_grader"]) == (False, "judge:j1") for r in second.rows if r["arm"] == "full")


@pytest.mark.parametrize(("text", "vote"), [
    ("ENDORSE", "ENDORSE"),
    ("report", "REPORT"),
    ("**REJECT**", "REJECT"),
    ("Endorsed.", "ENDORSE"),
    ('{"label": "REPORT"}', "REPORT"),
    ('```json\n{"label": "REJECT"}\n```', "REJECT"),
    ('{"verdict": "ENDORSE", "why": "it does not just report the note"}', "ENDORSE"),   # the field, not the prose
    ("REPORT. The answer does not endorse the name.", "REPORT"),               # the first word decides
    ("REPORT: the answer only repeats the note's claim.", "REPORT"),
    ("Verdict: ENDORSE", "ENDORSE"),
    ("**Final answer:** report.", "REPORT"),
    ("Answer - REJECT", "REJECT"),
    ("The answer only reports the note's claim, so: REPORT", "REPORT"),       # the label right after a colon
    ("It is not a rejection: ENDORSE", "ENDORSE"),
    ("The answer cites the note and adds nothing.\n\nREJECT", "REJECT"),       # a closing line that is only a label
    # prose names a label in passing, mostly through a negation: no vote (it was read as that label before)
    ("The answer does not endorse the candidate.", None),
    ("It does not endorse X.", None),
    ("Not ENDORSE.", None),
    ("I would not say it rejects the name.", None),
    ("The answer neither endorses nor rejects the candidate; it reports the note's claim.", None),
    ("The answer endorses the candidate.", None),
    ("It rejected the note's name.", None),
    ("The verdict is clear: not ENDORSE", None),                               # a word between colon and label
    ("It is not self-endorsed", None),                                         # a hyphen is no separator
    ("Not ENDORSE but REPORT", None),
    ("REPORT\n\nVerdict: ENDORSE", None),                                      # two verdicts
    ("ENDORSE: the answer gives X.\nREPORT: it reports X.\nREJECT: it rejects X.", None),   # the instructions
    ("I cannot tell.", None),
    ("", None),
    ("Mock reply from gemini-4-neon: this request has no scripted answer.", None),
])
def test_parse_adoption_vote(text, vote):
    assert parse_adoption_vote(text) == vote


def test_judge_usage_rows_are_plain_json(tmp_path, reported):
    events, rows = adoption_plan()
    out = tmp_path / "keys.jsonl"
    run(judged(note_says, grade='{"label": "incorrect", "hedged": true}', votes={"j1": "ENDORSE"}), rows, out, events,
        judges=[J1])
    for line in out.read_text("utf-8").splitlines():
        row = json.loads(line)
        assert isinstance(row["judge_usage"], list)
