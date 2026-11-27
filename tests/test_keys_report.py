"""Key, not value (docs/DESIGN_keys.md): the statistics and the report with its prediction verdicts."""

from __future__ import annotations

import json
import math

import pytest

from cutoffmix.bench.keys import (
    ARMS_BY_SET,
    build_report,
    fisher_exact,
    mcnemar_exact,
    render_report,
    wilson,
)

# ---- statistics ---------------------------------------------------------------------------------------------------


def test_wilson_known_values():
    assert wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_exact_tests_known_values():
    assert mcnemar_exact(10, 2) == pytest.approx(0.03857, abs=1e-5)
    assert mcnemar_exact(2, 10) == mcnemar_exact(10, 2)
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact(10, 0, 0, 10) == pytest.approx(2 / math.comb(20, 10))
    assert fisher_exact(0, 0, 0, 0) == 1.0


# ---- report -----------------------------------------------------------------------------------------------------


def plan_row(set_: str, pair: tuple[str, str], event: str, arm: str, recognised: bool | None = None) -> dict:
    return {"id": f"{set_}|{pair[0]}|{pair[1]}|{event}|{arm}", "set": set_, "donor": pair[0], "recipient": pair[1],
            "event_id": event, "arm": arm, "recognised": recognised, "donor_grade": "correct", "donor_answer": "X Y",
            "note": None, "note_event": None, "messages": []}


def result_row(row: dict, grade: str, adopted: bool | None = None, error: str | None = None) -> dict:
    return {"id": row["id"], "set": row["set"], "donor": row["donor"], "recipient": row["recipient"],
            "event_id": row["event_id"], "arm": row["arm"], "response": "" if error else "text", "grade": grade,
            "hedged": False, "grader": "rule", "adopted": adopted, "latency_s": 1.0,
            "usage": {"cost": 0.001}, "error": error}


def synthetic(dark_items: int = 20, adopt_items: int = 8, harm_items: int = 10):
    """Dark items: key right on 9 of 10 recognised and 1 of 10 unrecognised, alone right on 1, wrong key never,
    full on 18; donor_wrong: adopted under full on every item, never alone; both_right: key never breaks recall."""
    plan, results = [], []
    pair = ("a", "b")
    for i in range(dark_items):
        rec = i < dark_items // 2
        outcome = {"alone": i == 0, "key": (i < 9) if rec else (i == dark_items // 2), "wrong_key": False,
                   "full": i < dark_items - 2, "blank": False}
        for arm in ARMS_BY_SET["dark"]:
            row = plan_row("dark", pair, f"e{i}", arm, rec)
            plan.append(row)
            results.append(result_row(row, "correct" if outcome[arm] else "abstain"))
    for i in range(adopt_items):
        for arm in ARMS_BY_SET["donor_wrong"]:
            row = plan_row("donor_wrong", pair, f"w{i}", arm)
            plan.append(row)
            results.append(result_row(row, "incorrect" if arm == "full" else "abstain", adopted=arm == "full"))
    for i in range(harm_items):
        for arm in ARMS_BY_SET["both_right"]:
            row = plan_row("both_right", pair, f"r{i}", arm)
            plan.append(row)
            results.append(result_row(row, "correct"))
    return plan, results


def test_report_numbers_and_verdicts():
    plan, results = synthetic()
    results.append(result_row(plan[0], "ungraded", error="boom"))              # a later failed retry: last row counts
    results.append(result_row(plan[0], "abstain"))
    report = build_report(plan, results)
    dark = report["pairs"]["a->b"]["dark"]
    assert dark["items"] == 20 and dark["arms"]["key"]["k"] == 10 and dark["arms"]["key"]["n"] == 20
    assert dark["arms"]["alone"]["k"] == 0                                    # the last row of e0/alone is a miss
    assert report["paired"]["a->b"]["key_vs_wrong_key"] == {"b": 10, "c": 0, "p": round(mcnemar_exact(10, 0), 6),
                                                            "n_items": 20}
    rec = report["recognised"]["a->b"]
    assert rec["recognised"]["key"]["k"] == 9 and rec["unrecognised"]["key"]["k"] == 1
    assert report["adoption"]["a->b"]["donor_wrong"]["full"]["k"] == 8
    assert report["harm"]["a->b"]["key"]["k"] == 0 and report["totals"]["planned"] == len(plan)
    verdicts = {p["id"]: p["verdict"] for p in report["predictions"]}
    assert verdicts == {1: "supported", 2: "supported", 3: "supported", 4: "supported"}
    md = render_report(report)
    assert "# Key, not value: report" in md and "**supported**" in md and "a->b" in md
    json.dumps(report)                                                           # plain data


def test_report_verdicts_when_the_key_does_nothing_or_n_is_small():
    plan, results = synthetic()
    flat = [dict(r, grade="abstain") if r["arm"] == "key" else r for r in results]
    verdicts = {p["id"]: p["verdict"] for p in build_report(plan, flat)["predictions"]}
    # every key answer is now a miss: no recovery, nothing to tell apart, and the key breaks the correct recalls
    assert verdicts == {1: "not supported", 2: "not supported", 3: "supported", 4: "not supported"}
    small_plan, small_results = synthetic(dark_items=6, adopt_items=2, harm_items=3)
    small = {p["id"]: p["verdict"] for p in build_report(small_plan, small_results)["predictions"]}
    assert small == {1: "not testable (n too small)", 2: "not testable (n too small)",
                     3: "not testable (n too small)", 4: "not testable (n too small)"}
    empty = build_report(plan, [])
    assert empty["totals"]["rows"] == 0 and empty["totals"]["missing"] == len(plan)
    assert all(p["verdict"].startswith("not testable") for p in empty["predictions"])
    render_report(empty)


def test_report_leaves_ungraded_and_errors_out_of_n():
    plan, results = synthetic()
    broken = [dict(r, grade="ungraded") if r["arm"] == "wrong_key" and r["event_id"] == "e3" else r for r in results]
    report = build_report(plan, broken)
    arms = report["pairs"]["a->b"]["dark"]["arms"]
    assert arms["wrong_key"]["n"] == 19 and arms["wrong_key"]["ungraded"] == 1
    assert report["paired"]["a->b"]["n_items"] == 19                           # e3 lacks a graded wrong key
    assert report["totals"]["ungraded"] == 1
