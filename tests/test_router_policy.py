import math
import random
from types import SimpleNamespace

import pytest

from cutoffmix.router.policy import (
    ArgmaxHysteresisPolicy,
    FixedPolicy,
    RandomPolicy,
    RoundRobinPolicy,
    SoftmaxPolicy,
    ThompsonPolicy,
    make_policy,
)

MODELS = ["fable", "sol", "neon"]
SCORES = {"fable": 27.0, "sol": 28.5, "neon": 28.0}


def _expected_softmax(scores, current, r, T0=0.5, beta=0.25, r_min=0.05):
    t = T0 / max(r, r_min)
    w = {m: math.exp((s + (beta if m == current else 0.0)) / t) for m, s in scores.items()}
    z = sum(w.values())
    return {m: v / z for m, v in w.items()}


@pytest.mark.parametrize("r", [0.0, 0.05, 0.3, 1.0])
@pytest.mark.parametrize("current", [None, "fable", "neon"])
def test_softmax_probabilities_match_formula(r, current):
    pol = SoftmaxPolicy(T0=0.5, beta=0.25, r_min=0.05)
    probs = pol.probabilities(SCORES, current=current, r=r, available=MODELS)
    assert probs == pytest.approx(_expected_softmax(SCORES, current, r))
    assert pol.temperature(r) == pytest.approx(0.5 / max(r, 0.05))


def test_softmax_is_numerically_stable_and_handles_inf():
    pol = SoftmaxPolicy()
    probs = pol.probabilities({"fable": 1e9, "sol": 1e9 - 1, "neon": -math.inf}, current=None, r=1.0, available=MODELS)
    assert all(math.isfinite(p) for p in probs.values()) and sum(probs.values()) == pytest.approx(1.0)
    assert probs["neon"] == 0.0
    uniform = pol.probabilities({"fable": -math.inf}, current="sol", r=1.0, available=MODELS)
    assert uniform == pytest.approx({m: 1 / 3 for m in MODELS})


def test_softmax_mixes_when_time_insensitive_and_routes_sharply_when_sensitive():
    pol = SoftmaxPolicy()
    flat = pol.probabilities(SCORES, current=None, r=0.0, available=MODELS)
    sharp = pol.probabilities(SCORES, current=None, r=1.0, available=MODELS)
    assert max(flat.values()) - min(flat.values()) < 0.1
    assert sharp["sol"] > 0.7


def test_softmax_sampling_follows_probabilities_and_decisions():
    pol = SoftmaxPolicy()
    rng = random.Random(0)
    counts = {m: 0 for m in MODELS}
    decisions = set()
    n = 4000
    for _ in range(n):
        d = pol.choose(SCORES, current="neon", r=0.4, available=MODELS, rng=rng)
        counts[d.model] += 1
        decisions.add(d.decision)
        assert d.decision == ("hold" if d.model == "neon" else "sample")
    expected = _expected_softmax(SCORES, "neon", 0.4)
    for m in MODELS:
        assert counts[m] / n == pytest.approx(expected[m], abs=0.03)
    assert decisions == {"hold", "sample"}


def test_single_and_empty_availability():
    rng = random.Random(0)
    for pol in (SoftmaxPolicy(), ThompsonPolicy(), ArgmaxHysteresisPolicy(), RandomPolicy(), RoundRobinPolicy()):
        d = pol.choose(SCORES, current="sol", r=1.0, available=["fable"], rng=rng)
        assert (d.model, d.decision, d.probs) == ("fable", "forced", {"fable": 1.0})
        with pytest.raises(ValueError):
            pol.choose(SCORES, current=None, r=1.0, available=[], rng=rng)


def test_thompson_uses_sampled_scores():
    draws = iter([{"fable": 30.0, "sol": 1.0, "neon": 1.0}] + [{"fable": 1.0, "sol": 30.0, "neon": 1.0}] * 100)
    d = ThompsonPolicy(n_prob_draws=4).choose(SCORES, current="sol", r=1.0, available=MODELS, rng=random.Random(0),
                                              sample_scores=lambda: next(draws))
    assert d.model == "fable" and d.decision == "sample"
    assert d.probs == {"fable": 0.25, "sol": 0.75, "neon": 0.0}
    held = ThompsonPolicy().choose(SCORES, current="sol", r=1.0, available=MODELS, rng=random.Random(0))
    assert held.model == "sol" and held.decision == "hold" and held.probs["sol"] == 1.0


def test_thompson_breaks_ties_randomly():
    zeros = {m: 0.0 for m in MODELS}
    d = ThompsonPolicy(n_prob_draws=3000).choose(zeros, current=None, r=1.0, available=MODELS, rng=random.Random(1))
    assert all(abs(p - 1 / 3) < 0.05 for p in d.probs.values())


def test_argmax_hysteresis():
    pol = ArgmaxHysteresisPolicy(delta=0.75)
    rng = random.Random(0)
    first = pol.choose(SCORES, current=None, r=1.0, available=MODELS, rng=rng)
    assert (first.model, first.decision) == ("sol", "argmax")
    hold = pol.choose(SCORES, current="neon", r=1.0, available=MODELS, rng=rng)  # 28.5 - 28.0 < 0.75
    assert (hold.model, hold.decision) == ("neon", "hold")
    switch = pol.choose(SCORES, current="fable", r=1.0, available=MODELS, rng=rng)  # 1.5 >= 0.75
    assert (switch.model, switch.decision) == ("sol", "argmax")
    gone = pol.choose(SCORES, current="sol", r=1.0, available=["fable", "neon"], rng=rng)
    assert (gone.model, gone.decision) == ("neon", "argmax")
    best = pol.choose(SCORES, current="sol", r=1.0, available=MODELS, rng=rng)
    assert (best.model, best.decision) == ("sol", "hold")


def test_baseline_policies():
    rng = random.Random(0)
    assert FixedPolicy("neon").choose(SCORES, current="sol", r=1, available=MODELS, rng=rng).model == "neon"
    fallback = FixedPolicy("neon").choose(SCORES, current=None, r=1, available=["fable", "sol"], rng=rng)
    assert (fallback.model, fallback.decision) == ("sol", "forced")
    rr = RoundRobinPolicy()
    seq, cur = [], None
    for _ in range(4):
        cur = rr.choose(SCORES, current=cur, r=1, available=MODELS, rng=rng).model
        seq.append(cur)
    assert seq == ["fable", "sol", "neon", "fable"]
    rnd = RandomPolicy().choose(SCORES, current=None, r=1, available=MODELS, rng=rng)
    assert rnd.decision == "sample" and rnd.probs == pytest.approx({m: 1 / 3 for m in MODELS})
    with pytest.raises(ValueError):
        FixedPolicy("")


def test_make_policy():
    cfg = SimpleNamespace(T0=1.0, beta=0.5, r_min=0.1, delta=0.3)
    soft = make_policy("softmax", cfg)
    assert isinstance(soft, SoftmaxPolicy) and (soft.T0, soft.beta, soft.r_min) == (1.0, 0.5, 0.1)
    assert isinstance(make_policy("softmax"), SoftmaxPolicy)
    assert isinstance(make_policy("thompson", cfg), ThompsonPolicy)
    arg = make_policy("argmax", cfg)
    assert isinstance(arg, ArgmaxHysteresisPolicy) and arg.delta == 0.3
    fixed = make_policy("fixed:gpt-6.2-sol", cfg)
    assert isinstance(fixed, FixedPolicy) and fixed.model == "gpt-6.2-sol"
    assert isinstance(make_policy("round_robin", cfg), RoundRobinPolicy)
    assert isinstance(make_policy("random", cfg), RandomPolicy)
    with pytest.raises(ValueError, match="unknown routing policy"):
        make_policy("greedy", cfg)
    with pytest.raises(ValueError):
        SoftmaxPolicy(T0=0)
    with pytest.raises(ValueError):
        ArgmaxHysteresisPolicy(delta=-1)
