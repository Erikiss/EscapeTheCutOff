"""Tests for cutoffmix.bench.simulate: synthetic events, simulated answers and estimator recovery."""

from __future__ import annotations

import calendar

import numpy as np
import pytest

from cutoffmix.backends.mock import MockModelTruth, default_mock_truths
from cutoffmix.bench.estimate import fit_index
from cutoffmix.bench.grading import grade_rule
from cutoffmix.bench.simulate import recovery_report, simulate_results, synthetic_events, true_tau
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.timeutil import month_index, parse_fractional_month


def truth(key: str, tau: dict[str, float] | None = None, default_tau: float = 28.0, **kw: float) -> MockModelTruth:
    """A simulated model's truth with the defaults these tests use (no topic overrides, default tau 28)."""
    return MockModelTruth(key, dict(tau or {}), default_tau, **kw)


def test_synthetic_events_cover_every_cell() -> None:
    events = synthetic_events(["ai_ml", "sports"], range(22, 26), 3, seed=4)
    assert len(events) == 2 * 4 * 3 and len({e.id for e in events}) == len(events)
    cells = {(e.topic, e.month) for e in events}
    assert cells == {(t, f"2025-{m:02d}") for t in ("ai_ml", "sports") for m in (11, 12)} | \
        {(t, f"2026-{m:02d}") for t in ("ai_ml", "sports") for m in (1, 2)}
    for e in events:
        year, month, day = (int(x) for x in e.date.split("-"))
        assert e.month == e.date[:7] and 1 <= day <= calendar.monthrange(year, month)[1]
        assert month_index(year, month) in range(22, 26)
        letters = [c[0] for c in e.mcq_choices]
        assert letters == ["A", "B", "C", "D"] and e.mcq_answer in letters
        correct = next(c for c in e.mcq_choices if c[0] == e.mcq_answer)
        assert correct == f"{e.mcq_answer}) {e.expected_current}" and e.aliases == [e.expected_current]
        assert sum(e.reject_aliases[0] in c for c in e.mcq_choices) == 1
        assert e.origin == "synthetic" and e.predictability == "low" and not e.control
        assert e.question_current and e.question_recall and e.expected_recall == e.expected_current
    assert synthetic_events(["ai_ml"], [30], 2, seed=4) == synthetic_events(["ai_ml"], [30], 2, seed=4)
    assert synthetic_events(["ai_ml"], [30], 2, seed=4) != synthetic_events(["ai_ml"], [30], 2, seed=5)
    with pytest.raises(ValueError):
        synthetic_events(["ai_ml"], [30], 0)


def test_simulated_answers_follow_the_truth_and_regrade_identically() -> None:
    truths = {"early": truth("early", default_tau=20.0, stale_rate=1.0), "late": truth("late", default_tau=30.0)}
    events = synthetic_events(["sports", "government"], range(14, 35), 6, seed=1)
    results = simulate_results(truths, events, ["current", "recall", "mcq"], seed=2)
    assert len(results) == len(events) * 2 * 3
    assert results == simulate_results(truths, events, ["current", "recall", "mcq"], seed=2)
    by_id = {e.id: e for e in events}
    for r in results:  # the rule grader reads every simulated answer the way the simulator labelled it
        assert grade_rule(by_id[r.event_id], r.probe, r.response)[0] == r.grade, r
    assert {r.grade for r in results if r.model == "early" and r.probe != "mcq"} <= {"correct", "outdated"}
    assert {r.grade for r in results if r.probe == "mcq"} <= {"correct", "outdated", "incorrect"}

    def known(model: str, lo: float, hi: float) -> float:
        sel = [r.grade == "correct" for r in results if r.model == model and r.probe == "current"
               and lo <= parse_fractional_month(by_id[r.event_id].date) < hi]
        return float(np.mean(sel))

    assert known("early", 14, 18) > 0.85 and known("early", 23, 35) < 0.05
    assert known("late", 14, 28) > 0.85 and known("late", 32, 35) < 0.1


def test_estimator_recovers_the_mock_truths() -> None:
    """Acceptance: 15 topics, months 0..34, 8 items per topic-month (so >= 8 per month in every transition band):
    RMSE(tau) < 0.8 months and the per-topic best model matches the truth on >= 85% of topics whose true gap is
    >= 1 month."""
    truths = default_mock_truths()
    events = synthetic_events(TOPIC_IDS, range(35), 8, seed=7)
    results = simulate_results(truths, events, ["current", "mcq"], seed=8)
    index = fit_index(events, results, bootstrap=100, seed=9)
    for probe in ("current", "mcq"):
        report = recovery_report(index, truths, probe, min_gap=1.0, min_n=8 * 35)
        assert report["n_cells"] == 3 * 15
        assert report["rmse"] < 0.8, report["rmse"]
        assert report["n_topics_gap"] >= 10
        assert report["argmax_agreement_gap"] >= 0.85, report["argmax_agreement_gap"]
        assert report["p_best_true_mean"] > 0.7
        assert 0.6 <= report["coverage90"] <= 1.0
        assert sum(b["n"] for b in report["p_best_calibration"]) == 3 * 15


def test_recovery_report_on_an_oracle_index() -> None:
    from cutoffmix.router.index import CutoffIndex, ProbeIndex

    truths = {"a": truth("a", {"sports": 30.0}, 27.0), "b": truth("b", {"ai_ml": 31.0}, 28.0)}
    topics = ["sports", "ai_ml", "government"]
    tau = np.array([[true_tau(truths[m], z) for z in topics] for m in ("a", "b")])
    ones = np.ones_like(tau)
    probe = ProbeIndex(probe="current", tau=tau, var=0.01 * ones, hazard=0 * ones, abstain=0 * ones,
                       n=(100 * ones).astype(int), draws=None, global_tau=tau.mean(axis=1),
                       slope=np.ones(2), plateau=np.full(2, 0.95))
    index = CutoffIndex(["a", "b"], topics, {"current": probe})
    report = recovery_report(index, truths, "current")
    assert report["rmse"] == 0 and report["bias"] == 0 and report["coverage90"] == 1.0
    assert report["argmax_agreement"] == 1.0 and report["argmax_agreement_gap"] == 1.0
    assert report["n_topics_gap"] == 3 and report["p_best_brier"] < 1e-6
    assert report["rmse_by_model"] == {"a": 0.0, "b": 0.0}
    assert report["cells"][0] == {"model": "a", "topic": "sports", "true": 30.0, "true_label": "2026-07",
                                  "est": 30.0, "sd": pytest.approx(0.1), "n": 100, "err": 0.0}
    assert recovery_report(index, truths, "current", min_gap=5.0)["n_topics_gap"] == 0
    with pytest.raises(ValueError, match="truth"):
        recovery_report(index, {"zzz": truths["a"]}, "current")
