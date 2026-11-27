"""Key, not value (docs/DESIGN_keys.md section 6): what a run keeps when its grading raises after a call was paid for,
and what a resumed run repairs from the stored responses without a new recipient call (answers whose grading failed,
adoption labels measured against a donor answer the plan no longer names)."""

from __future__ import annotations

import pytest
from test_keys_run import ROW_FIELDS, ScriptedBackend, fixture_plan, refused, run
from test_keys_run_judges import J1, UNDECIDED, calls, judged

from cutoffmix.bench.io import read_jsonl
from cutoffmix.bench.keys import prompt_sha
from cutoffmix.bench.keys import text as text_mod
from cutoffmix.bench.keys.run import ADOPTION_SETS, GRADING_FAILED

REAL_LABEL = text_mod.adoption_label
ODD = "Odd answer naming Made Upname."


def broken_on_odd(response, name, event):
    """``adoption_label`` that breaks its contract on one answer (a bug in the labelling code)."""
    return "undecided" if "Odd" in response else REAL_LABEL(response, name, event)


def odd_control_full(model, req):
    if req.metadata["set"] == "control" and req.metadata["arm"] == "full":
        return ODD
    return "I don't know."


# ---- a grading that raises after the call -------------------------------------------------------------------------


def test_an_answer_whose_grading_raises_is_kept_counted_and_graded_again_without_a_new_call(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    monkeypatch.setattr(text_mod, "adoption_label", broken_on_odd)
    out = tmp_path / "keys.jsonl"
    backend = ScriptedBackend(odd_control_full)
    got = run(backend, rows, out, events, concurrency=1)
    assert len(backend.calls) == len(rows) == got.sent                          # the run went on after the failure
    stored = read_jsonl(out)
    failed = [r for r in stored if r["error"]]
    assert len(stored) == len(rows) and len(failed) == 1 == got.not_graded and got.not_sent == 0
    row = failed[0]
    assert list(row) == ROW_FIELDS and row["arm"] == "full" and row["set"] == "control"
    assert row["error"].startswith(GRADING_FAILED + "ValueError: adoption_label returned 'undecided'")
    assert row["response"] == ODD and row["usage"] == {"cost": 0.01} and row["judge_usage"] == []
    assert row["grade"] == "ungraded" and row["grader"] == "none"
    assert row["adopted"] is None and row["adoption"] is None and row["adoption_grader"] is None
    assert row["backend"] == "scripted" and row["prompt_sha"] and row["request_sha"]
    assert got.cost_usd == pytest.approx(0.01 * len(rows))                      # the paid answer is counted
    # the next run sends nothing: it grades the stored answer again (still failing, so one more failure row)
    again = ScriptedBackend(odd_control_full)
    seen = []
    second = run(again, rows, out, events, on_start=seen.append)
    assert again.calls == [] and second.sent == 0 and second.regraded == 1 and second.not_graded == 1
    assert seen[0].answered == len(rows) and seen[0].pending == 0 and seen[0].regrade == seen[0].failed == 1
    assert second.cost_usd == pytest.approx(0.01 * len(rows)) and second.stopped_by is None
    # once the labelling works again, the stored answer is graded, still without a new call
    monkeypatch.setattr(text_mod, "adoption_label", REAL_LABEL)
    fixed = ScriptedBackend(odd_control_full)
    third = run(fixed, rows, out, events)
    assert fixed.calls == [] and third.regraded == 1 and third.not_graded == 0 and third.not_sent == 0
    last = {r["id"]: r for r in read_jsonl(out)}[row["id"]]
    assert last["error"] is None and last["usage"] is None and last["response"] == ODD and last["latency_s"] == 0.5
    assert (last["grade"], last["grader"]) == ("incorrect", "rule")             # the fake paper has no author
    label = REAL_LABEL(ODD, "Made Upname", next(e for e in events if e.id == "paper:20"))
    adopted = {"endorsed": True, "absent": False, "reported": None}[label]
    assert (last["adoption"], last["adopted"], last["adoption_grader"]) == (label, adopted, "rule")
    assert third.recipients_usd == pytest.approx(0.01 * len(rows))             # each paid call counted once
    assert run(ScriptedBackend(odd_control_full), rows, out, events).regraded == 0


def test_a_grading_function_that_raises_keeps_the_answer_too(tmp_path, monkeypatch):
    events, rows = fixture_plan()

    async def grade(event, probe, response, **kwargs):
        raise RuntimeError("grader broke")
    monkeypatch.setattr("cutoffmix.bench.keys.run.grade_response", grade)
    got = run(ScriptedBackend(), rows[:3], tmp_path / "keys.jsonl", events)
    assert got.sent == 3 and got.not_graded == 3 and got.cost_usd == pytest.approx(0.03)
    assert all(r["error"] == GRADING_FAILED + "RuntimeError: grader broke" and r["response"] == "I don't know."
               for r in got.rows)


def test_judge_calls_made_before_a_grading_failure_stay_counted(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "control"]

    def label(response, name, event):
        raise RuntimeError("labelling broke")
    monkeypatch.setattr(text_mod, "adoption_label", label)
    backend = judged(lambda req: UNDECIDED)                                     # the rules cannot decide: a judge first
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, judges=[J1])
    assert len(calls(backend)) == len(rows) == len(calls(backend, "judge")) == got.not_graded
    for row in got.rows:
        assert row["error"] == GRADING_FAILED + "RuntimeError: labelling broke" and row["usage"] == {"cost": 0.01}
        assert row["judge_usage"] == [{"judge": "j1", "purpose": "judge", "cost": 0.02, "usage": {"cost": 0.02}}]
    assert got.cost_usd == pytest.approx(0.03 * len(rows)) and got.judges_usd == pytest.approx(0.02 * len(rows))
    # a re-grade that fails again keeps its judge calls as well, and sends no recipient call
    again = judged(lambda req: UNDECIDED)
    second = run(again, rows, out, events, judges=[J1])
    assert calls(again) == [] and len(calls(again, "judge")) == len(rows) == second.regraded == second.not_graded
    assert all(r["usage"] is None and len(r["judge_usage"]) == 1 for r in second.rows)
    assert second.judges_usd == pytest.approx(0.04 * len(rows)) and second.recipients_usd == pytest.approx(0.02)


def test_an_answer_whose_grading_failed_under_another_request_is_stale(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    monkeypatch.setattr(text_mod, "adoption_label", broken_on_odd)
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(odd_control_full), rows, out, events)
    monkeypatch.setattr(text_mod, "adoption_label", REAL_LABEL)
    refused(ScriptedBackend(odd_control_full), rows, out, events, f"{len(rows)} of the {len(rows)} planned calls were "
            "answered under another request", max_tokens=2048)                  # the failed answer counts as one
    backend = ScriptedBackend(odd_control_full)
    seen = []
    got = run(backend, rows, out, events, max_tokens=2048, resend_stale=True, on_start=seen.append)
    assert len(backend.calls) == len(rows) and got.not_graded == 0 and not any(r["error"] for r in got.rows)
    assert seen[0].stale == len(rows) and seen[0].errors == 0 and seen[0].regrade == 0


def test_regrades_without_judges_cost_nothing_and_the_cap_does_not_stop_them(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    monkeypatch.setattr(text_mod, "adoption_label", broken_on_odd)
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(odd_control_full), rows, out, events)                   # 0.01 per call
    monkeypatch.setattr(text_mod, "adoption_label", REAL_LABEL)
    backend = ScriptedBackend(odd_control_full)
    got = run(backend, rows, out, events, max_cost=0.01)                        # reached long ago
    assert backend.calls == [] and got.regraded == 1 and got.not_graded == 0 and got.stopped_by is None


# ---- a donor answer that changed in a re-plan -------------------------------------------------------------------


def donor_wrong_plan():
    events, rows = fixture_plan()
    return events, [r for r in rows if r["set"] == "donor_wrong"]


def copies_the_offer(model, req):                                               # "My best guess is Wrong Person."
    return "My best guess is Wrong Person."


def labels(outcome):
    return [(r["arm"], r["adoption"], r["adopted"]) for r in outcome.rows]


def test_a_changed_donor_answer_relabels_the_stored_rows_without_a_call(tmp_path):
    events, rows = donor_wrong_plan()
    out = tmp_path / "keys.jsonl"
    first = run(ScriptedBackend(copies_the_offer), rows, out, events)
    assert labels(first) == [("alone", "endorsed", True), ("full", "endorsed", True)]
    replanned = [{**r, "donor_answer": "Other Name"} for r in rows]             # another name, the same messages
    assert [prompt_sha(r["messages"]) for r in replanned] == [prompt_sha(r["messages"]) for r in rows]
    backend = ScriptedBackend(copies_the_offer)
    seen = []
    second = run(backend, replanned, out, events, on_start=seen.append)
    assert backend.calls == [] and second.sent == 0 and second.regraded == 2 == seen[0].relabel == seen[0].regrade
    fresh = run(ScriptedBackend(copies_the_offer), replanned, tmp_path / "fresh.jsonl", events)
    assert labels(second) == labels(fresh) == [("alone", "absent", False), ("full", "absent", False)]
    assert all(r["usage"] is None and r["response"] == "My best guess is Wrong Person." for r in second.rows)
    assert second.recipients_usd == pytest.approx(first.recipients_usd)        # no call billed twice
    assert run(ScriptedBackend(copies_the_offer), replanned, out, events).regraded == 0   # labelled once
    back = run(ScriptedBackend(copies_the_offer), rows, out, events)             # the original name again
    assert back.regraded == 2 and labels(back) == labels(first)


def test_an_unchanged_label_is_kept_and_a_label_that_cannot_be_computed_is_a_grading_failure(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(copies_the_offer), rows, out, events)
    quiet = run(ScriptedBackend(copies_the_offer), [{**r, "donor_answer": "Nobody Else"} if r["set"] == "dark" else r
                                                    for r in rows], out, events)
    assert quiet.regraded == 0                                                  # dark rows have no adoption

    def label(response, name, event):
        raise RuntimeError("labelling broke")
    monkeypatch.setattr(text_mod, "adoption_label", label)
    backend = ScriptedBackend(copies_the_offer)
    got = run(backend, rows, out, events)
    adoption_rows = [r for r in rows if r["set"] in ADOPTION_SETS]
    assert backend.calls == [] and got.regraded == len(adoption_rows) == got.not_graded
    assert {r["id"] for r in got.rows if r["error"]} == {r["id"] for r in adoption_rows}
