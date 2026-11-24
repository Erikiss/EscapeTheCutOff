"""Tests for cutoffmix.bench.estimate (4PL index fit, shrinkage, censoring, bootstrap, hazard, kc-style cutoffs)."""

from __future__ import annotations

import dataclasses
import math
import time

import numpy as np
import pytest

from cutoffmix.backends.mock import MockModelTruth
from cutoffmix.bench.estimate import A_GRID, S_GRID, TAU_MAX, TAU_MIN, fit_index, kc_style_cutoffs
from cutoffmix.bench.simulate import simulate_results, synthetic_events
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.timeutil import parse_fractional_month as _month
from cutoffmix.types import Event, ProbeResult


def truth(key: str, tau: dict[str, float] | None = None, default_tau: float = 28.0, **kw: float) -> MockModelTruth:
    """A simulated model's truth with the defaults these tests use (no topic overrides, default tau 28)."""
    return MockModelTruth(key, dict(tau or {}), default_tau, **kw)


TOPICS = ["ai_ml", "sports", "government"]
TRUTHS = {
    "A": truth("A", {"ai_ml": 31.0}, default_tau=27.0),
    "B": truth("B", {"sports": 30.0}, default_tau=29.0, slope=0.5),
}


@pytest.fixture(scope="module")
def data() -> tuple[list[Event], list[ProbeResult]]:
    events = synthetic_events(TOPICS, range(14, 35), 10, seed=11)
    return events, simulate_results(TRUTHS, events, ["current", "mcq"], seed=12)


@pytest.fixture(scope="module")
def fitted(data: tuple[list[Event], list[ProbeResult]]):
    events, results = data
    return fit_index(events, results, bootstrap=40, seed=5)


def cell(index, probe: str, model: str, topic: str, field_name: str = "tau") -> float:
    p = index.probes[probe]
    return float(getattr(p, field_name)[index.models.index(model), index.topics.index(topic)])


def test_recovers_cell_cutoffs_and_model_parameters(fitted) -> None:
    assert fitted.models == ["A", "B"] and fitted.topics[: len(TOPIC_IDS)] == list(TOPIC_IDS)
    assert list(fitted.probes) == ["current", "mcq"]
    for probe in ("current", "mcq"):
        p = fitted.probes[probe]
        for model, truth in TRUTHS.items():
            for topic in TOPICS:
                est = cell(fitted, probe, model, topic)
                assert abs(est - truth.tau.get(topic, truth.default_tau)) < 1.0, (probe, model, topic, est)
                assert cell(fitted, probe, model, topic, "n") == 21 * 10
        assert set(p.slope) <= set(S_GRID) and set(p.plateau) <= set(A_GRID)
        assert np.all(p.plateau >= 0.85)
    # the ai_ml reversal is found: A is newer than B there, B newer elsewhere
    assert cell(fitted, "current", "A", "ai_ml") > cell(fitted, "current", "B", "ai_ml") + 1
    assert cell(fitted, "current", "B", "government") > cell(fitted, "current", "A", "government") + 1


def test_empty_cells_get_the_prior(fitted) -> None:
    p = fitted.probes["current"]
    sigma = fitted.meta["hyper"]["current"]["sigma_eps"]
    assert fitted.meta["hyper"]["current"]["gamma"]["space_astronomy"] == 0.0
    for m, model in enumerate(fitted.models):
        assert cell(fitted, "current", model, "space_astronomy", "n") == 0
        assert cell(fitted, "current", model, "space_astronomy") == pytest.approx(p.global_tau[m], abs=0.01)
        assert cell(fitted, "current", model, "space_astronomy", "var") == pytest.approx(sigma ** 2, rel=0.02)


def test_draws_are_a_paired_bootstrap_with_prior_draws_for_empty_cells(fitted, data) -> None:
    p = fitted.probes["current"]
    assert p.draws is not None and p.draws.shape == (40, 2, len(fitted.topics))
    assert np.isfinite(p.draws).all()
    a, t_ai, t_space = 0, fitted.topics.index("ai_ml"), fitted.topics.index("space_astronomy")
    assert 0.0 < p.draws[:, a, t_ai].std() < 0.6  # data-rich cell: tight
    assert p.draws[:, a, t_space].std() == pytest.approx(math.sqrt(p.var[a, t_space]), rel=0.35)
    events, results = data
    again = fit_index(events, results, bootstrap=40, seed=5)
    np.testing.assert_array_equal(again.probes["current"].draws, p.draws)
    other = fit_index(events, results, bootstrap=40, seed=6)
    assert not np.array_equal(other.probes["current"].draws, p.draws)
    assert fit_index(events, results, bootstrap=0).probes["current"].draws is None


def test_topics_with_too_few_events_get_posterior_draws(data) -> None:
    # resampling a single event always gives the same data: bootstrap means would make P_best look certain
    events, results = data
    one = Event(id="one-health", date="2026-03-15", month="2026-03", topic="health_medicine", category="other",
                subject="x", fact="x", question_current="q?", expected_current="a", origin="synthetic")
    extra = [ProbeResult("one-health", m, "current", "a", "correct") for m in TRUTHS]
    index = fit_index([*events, one], [*results, *extra], probes=["current"], bootstrap=200, seed=1)
    p = index.probes["current"]
    z = index.topics.index("health_medicine")
    assert np.all(p.n[:, z] == 1)
    np.testing.assert_allclose(p.draws[:, :, z].std(axis=0), np.sqrt(p.var[:, z]), rtol=0.25)


def test_hazard_and_abstain_rates_follow_the_answer_style() -> None:
    truths = {"asserts": truth("asserts", stale_rate=1.0), "abstains": truth("abstains", stale_rate=0.0)}
    events = synthetic_events(["sports", "government"], range(18, 35), 10, seed=2)
    index = fit_index(events, simulate_results(truths, events, ["current"], seed=3), bootstrap=0)
    p = index.probes["current"]
    for arr in (p.hazard, p.abstain):
        assert np.all((arr >= 0) & (arr <= 1))
    assert np.all(p.hazard[0] > 0.8) and np.all(p.abstain[0] < 0.2)
    assert np.all(p.hazard[1] < 0.2) and np.all(p.abstain[1] > 0.8)



def test_model_level_hazard_is_shrunk_to_the_pooled_rate() -> None:
    # the freshest model has only a handful of items after its cutoff (the data end near it): its model-level
    # hazard must lean on the rate pooled over all models instead of Beta(1, 1) plus those few items
    truths = {
        "fresh": truth("fresh", default_tau=33.6, slope=0.3, stale_rate=1.0),
        "old1": truth("old1", default_tau=26.0, stale_rate=0.1),
        "old2": truth("old2", default_tau=26.5, stale_rate=0.1),
    }
    events = synthetic_events(["sports", "government"], range(18, 35), 6, seed=2)
    results = simulate_results(truths, events, ["current"], seed=3)
    index = fit_index(events, results, models=list(truths), bootstrap=0)
    p = index.probes["current"]
    by_id = {e.id: e for e in events}
    counts: dict[str, tuple[int, int]] = {}  # model -> (items after the cell cutoff + 2s, of which stale/incorrect)
    for mi, model in enumerate(index.models):
        after = [r for r in results if r.model == model
                 and _month(by_id[r.event_id].date) > p.tau[mi, index.topics.index(by_id[r.event_id].topic)]
                 + 2 * p.slope[mi]]
        counts[model] = (len(after), sum(r.grade in ("outdated", "incorrect") for r in after))
    assert 0 < counts["fresh"][0] <= 10 and counts["fresh"][1] == counts["fresh"][0]  # few items, all stale
    n_all, k_all = (sum(c[i] for c in counts.values()) for i in (0, 1))
    pooled = (k_all + 1) / (n_all + 2)
    empty = index.topics.index("space_astronomy")  # no data: the cell rate is the model-level rate
    for mi, model in enumerate(index.models):
        n_m, k_m = counts[model]
        assert p.hazard[mi, empty] == pytest.approx((k_m + 5 * pooled) / (n_m + 5), abs=1e-9)
    assert p.hazard[0, empty] < 0.7  # Beta(1, 1) alone gave (k + 1) / (n + 2) ~ 0.9
    assert np.all(p.hazard[1:, empty] < 0.15)

def test_censored_cells_stay_finite_in_the_prior_driven_region() -> None:
    events = synthetic_events(["sports", "ai_ml", "government"], range(10, 31), 6, seed=4)
    results = simulate_results({"m": truth("m", default_tau=22.0)}, events, ["current"], seed=1)
    by_id = {e.id: e for e in events}
    forced = []
    for r in results:
        topic = by_id[r.event_id].topic
        if topic == "sports":
            r = dataclasses.replace(r, grade="correct")  # knows everything: right-censored
        elif topic == "ai_ml":
            r = dataclasses.replace(r, grade="outdated")  # knows nothing: left-censored
        forced.append(r)
    index = fit_index(events, forced, bootstrap=20)
    p = index.probes["current"]
    sports, ai, gov = (cell(index, "current", "m", z) for z in ("sports", "ai_ml", "government"))
    assert np.isfinite(p.tau).all() and np.isfinite(p.var).all() and np.isfinite(p.draws).all()
    assert 30 < sports < TAU_MAX and TAU_MIN < ai < 10
    assert ai < gov < sports and abs(gov - 22.0) < 3.0  # forced cells also distort the pooled slope/plateau
    assert cell(index, "current", "m", "sports", "var") > cell(index, "current", "m", "government", "var")


def test_a_censored_topic_does_not_move_the_topic_offset(data) -> None:
    # one early event that every model answers correctly says little about the cutoffs (months later); its flat
    # likelihood must neither shift the topic offset nor drag the cell to the top of the τ grid
    events, results = data
    early = Event(id="early-health", date="2024-08-15", month="2024-08", topic="health_medicine", category="other",
                  subject="x", fact="x", question_current="q?", expected_current="a", origin="synthetic")
    extra = [ProbeResult("early-health", m, "current", "a", "correct") for m in TRUTHS]
    index = fit_index([*events, early], [*results, *extra], probes=["current"], bootstrap=0)
    assert index.meta["hyper"]["current"]["gamma"]["health_medicine"] == 0.0
    p = index.probes["current"]
    z = index.topics.index("health_medicine")
    assert np.all(p.n[:, z] == 1) and np.allclose(p.tau[:, z], p.global_tau, atol=0.5)


def test_all_correct_and_all_wrong_models_are_stable() -> None:
    events = synthetic_events(["sports"], range(35), 5, seed=9)
    results = [ProbeResult(e.id, "perfect", "current", "x", "correct") for e in events]
    results += [ProbeResult(e.id, "clueless", "mcq", "x", "incorrect") for e in events]
    index = fit_index(events, results, bootstrap=10)
    perfect, clueless = index.probes["current"], index.probes["mcq"]
    assert np.isfinite(perfect.tau[0]).all() and np.isfinite(perfect.draws[:, 0]).all()
    assert 34 < perfect.global_tau[0] < TAU_MAX and np.all(perfect.tau[0] < TAU_MAX)
    assert np.isfinite(clueless.tau[1]).all() and TAU_MIN <= clueless.global_tau[1] < 2  # despite the 0.25 floor
    assert np.isnan(perfect.tau[1]).all() and np.isnan(clueless.tau[0]).all()  # no rows, no prior
    # few items: the prior (centred on the median event month) keeps the estimate finite and moderate
    sparse = [e for e in events if e.month in ("2024-01", "2026-11")]
    few = fit_index(sparse, [r for r in results if r.event_id in {e.id for e in sparse}], bootstrap=0)
    assert np.isfinite(few.probes["mcq"].global_tau[1]) and TAU_MIN < few.probes["mcq"].global_tau[1] < TAU_MAX


def test_mcq_floor_is_part_of_the_model() -> None:
    events = synthetic_events(["sports", "government"], range(14, 35), 10, seed=21)
    truths = {"m": truth("m", default_tau=26.0, plateau=0.97)}
    results = simulate_results(truths, events, ["mcq"], seed=22)
    late = [r for r in results if r.event_id.split("-")[-2] == "11" and r.event_id.split("-")[-3] == "2026"]
    assert 0.1 < np.mean([r.grade == "correct" for r in late]) < 0.45  # chance floor in the data
    mcq = fit_index(events, results, bootstrap=0).probes["mcq"]
    assert abs(mcq.global_tau[0] - 26.0) < 1.0
    # the same answers fitted without the floor need a long tail and put the cutoff months too late
    as_open = fit_index(events, [dataclasses.replace(r, probe="current") for r in results], bootstrap=0)
    assert as_open.probes["current"].global_tau[0] > mcq.global_tau[0] + 1.5
    assert as_open.probes["current"].slope[0] > mcq.slope[0]


def test_exclusions_and_duplicate_rows_do_not_change_the_fit(data) -> None:
    events, results = data
    base = fit_index(events, results, bootstrap=0)
    controls = [dataclasses.replace(e, id=f"ctl{i}", control=True, category="control_alive")
                for i, e in enumerate(events[:30])]
    high = [dataclasses.replace(e, id=f"hi{i}", predictability="high") for i, e in enumerate(events[:30])]
    noise = [ProbeResult(e.id, "A", "current", "x", "correct") for e in controls + high]
    stale_copies = [dataclasses.replace(r, grade="abstain") for r in results[:200]]  # superseded by later rows
    extra = [ProbeResult(results[0].event_id, "A", "current", "", "ungraded", error="boom"),
             ProbeResult("missing-event", "A", "current", "x", "correct")]
    noisy = fit_index(events + controls + high, stale_copies + results + noise, bootstrap=0)
    for probe in base.probes:
        np.testing.assert_allclose(noisy.probes[probe].tau, base.probes[probe].tau)
        np.testing.assert_array_equal(noisy.probes[probe].n, base.probes[probe].n)
    # an ungraded row that comes last replaces the graded one for that key
    last_ungraded = fit_index(events, results + extra[:1], bootstrap=0)
    assert last_ungraded.probes["current"].n.sum() == base.probes["current"].n.sum() - 1


def test_priors_models_without_data_and_metadata(data) -> None:
    events, results = data
    index = fit_index(events, results, models=["A", "B", "ghost", "nobody"], topics=TOPICS, probes=["current"],
                      priors={"ghost": 30.0, "A": "2026-03"}, bootstrap=10)
    p = index.probes["current"]
    sigma = index.meta["hyper"]["current"]["sigma_eps"]
    assert index.topics == TOPICS and list(index.probes) == ["current"]
    gamma = np.array([index.meta["hyper"]["current"]["gamma"][t] for t in TOPICS])
    np.testing.assert_allclose(p.tau[2], 30.0 + gamma, atol=0.05)
    np.testing.assert_allclose(p.var[2], sigma ** 2 + 9.0, rtol=0.05)
    assert (p.n[2] == 0).all() and np.isnan(p.tau[3]).all() and np.isnan(p.draws[:, 3]).all()
    scores = index.scores({"sports": 1.0})
    assert scores["nobody"] == -math.inf and math.isfinite(scores["ghost"])
    meta = index.meta
    assert meta["priors"] == {"A": 26.5, "ghost": 30.0} and meta["prior_sd"] == 3.0
    assert meta["n_events"] == len(events) and meta["n_results"] == len(events) * 2
    assert meta["sources"] == ["synthetic"] and meta["events_by_origin"] == {"synthetic": len(events)}
    assert meta["data_range"] == {"first": min(e.date for e in events), "last": max(e.date for e in events)}
    assert meta["probes"] == ["current"]
    assert "2024-01 -> 0" in meta["month_axis"] and meta["method"] and meta["created"].endswith("+00:00")


def test_index_survives_a_json_round_trip(fitted) -> None:
    from cutoffmix.router.index import CutoffIndex

    back = CutoffIndex.from_json(fitted.to_json())
    for probe, p in fitted.probes.items():
        np.testing.assert_allclose(back.probes[probe].tau, p.tau, atol=1e-4)
        np.testing.assert_allclose(back.probes[probe].draws, p.draws, atol=1e-3)


def test_input_validation(data) -> None:
    events, results = data
    with pytest.raises(ValueError, match="no models"):
        fit_index(events, [])
    with pytest.raises(ValueError, match="no usable"):
        fit_index(events, results, probes=["recall"])
    for kwargs in ({"prior_sd": 0}, {"bootstrap": -1}, {"grid_step": 0}, {"grid_step": 2}):
        with pytest.raises(ValueError):
            fit_index(events, results, **kwargs)
    bad = [dataclasses.replace(events[0], date="sometime")]
    with pytest.raises(ValueError, match="unparseable date"):
        fit_index(bad, [ProbeResult(bad[0].id, "A", "current", "x", "correct")])


def test_fit_is_fast_enough() -> None:
    """3 models x 15 topics x 4,200 events x 2 probes with bootstrap=200 (the contract bound is 60 s)."""
    truths = {k: truth(k, default_tau=tau) for k, tau in (("f", 27.0), ("s", 28.5), ("n", 27.5))}
    events = synthetic_events(TOPIC_IDS, range(35), 8, seed=1)
    results = simulate_results(truths, events, ["current", "mcq"], seed=2)
    start = time.perf_counter()
    index = fit_index(events, results, bootstrap=200)
    elapsed = time.perf_counter() - start
    assert index.probes["mcq"].draws.shape == (200, 3, 15)
    assert elapsed < 30.0, f"fit took {elapsed:.1f} s"


# ------------------------------------------------------------------------------------------------- kc style


def _curve_results(model: str, events: list[Event], rates: list[float]) -> list[ProbeResult]:
    out = []
    for month_events, rate in zip(_by_month(events), rates, strict=True):
        k = round(rate * len(month_events))
        out += [ProbeResult(e.id, model, "current", "x", "correct" if i < k else "outdated")
                for i, e in enumerate(month_events)]
    return out


def _by_month(events: list[Event]) -> list[list[Event]]:
    months: dict[str, list[Event]] = {}
    for e in events:
        months.setdefault(e.month, []).append(e)
    return [months[m] for m in sorted(months)]


def test_kc_style_cutoffs_reproduce_the_benchmark_rules() -> None:
    events = synthetic_events(["sports"], range(24, 30), 10, seed=0)  # 2026-01 .. 2026-06
    results = (_curve_results("pinned", events, [1, 1, 0.9, 0.4, 0.2, 0])
               + _curve_results("never", events, [1] * 6)
               + _curve_results("notail", events, [1, 0.2, 1, 0.3, 1, 1])
               + _curve_results("before", events, [0.1] * 6))
    results.append(ProbeResult(events[0].id, "before", "mcq", "", "ungraded"))
    out = kc_style_cutoffs(events, results)
    pinned = out["pinned"]["current"]
    assert (pinned["cutoff_status"], pinned["crossover"], pinned["last_above"]) == ("pinned", "2026-03", "2026-03")
    assert pinned["curve"][3] == {"month": "2026-04", "n": 10, "known_rate": 0.4, "wrong_rate": 0.6,
                                  "abstain_rate": 0.0}
    assert (out["never"]["current"]["cutoff_status"], out["never"]["current"]["crossover"]) == \
        ("never_below", "2026-06+")
    assert out["notail"]["current"]["cutoff_status"] == "no_tail"
    assert out["notail"]["current"]["last_above"] == "2026-06" and out["notail"]["current"]["crossover"] is None
    assert out["before"]["current"]["cutoff_status"] == "before_window"
    assert "mcq" not in out["before"]  # ungraded rows are dropped from every denominator
    strict = kc_style_cutoffs(events, results, min_tail=4)
    assert strict["pinned"]["current"]["cutoff_status"] == "no_tail"
