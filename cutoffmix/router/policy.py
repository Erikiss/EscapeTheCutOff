"""Routing policies: turn per-model window scores ``U_m`` into the model for the next segment.

* :class:`SoftmaxPolicy` (default): ``P(m) ∝ exp((U_m + β·1[m = current]) / T)`` with ``T = T0 / max(r, r_min)``.
  Time-insensitive windows (small r) mix almost uniformly; time-sensitive windows route sharply.
* :class:`ThompsonPolicy`: argmax of one posterior draw of the scores.
* :class:`ArgmaxHysteresisPolicy`: deterministic best model, switching only when it beats the current one by ``delta``.
* Baselines: :class:`FixedPolicy`, :class:`RoundRobinPolicy`, :class:`RandomPolicy`.

Every policy chooses among ``available`` only. Models with a ``-inf`` (or missing) score are never chosen unless no
available model has a finite score, in which case the choice is uniform. With a single available model the decision
is ``"forced"``.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from cutoffmix.types import Decision

if TYPE_CHECKING:
    from cutoffmix.config import RouterConfig

__all__ = [
    "ArgmaxHysteresisPolicy",
    "FixedPolicy",
    "PolicyDecision",
    "RandomPolicy",
    "RoundRobinPolicy",
    "RoutingPolicy",
    "SoftmaxPolicy",
    "ThompsonPolicy",
    "make_policy",
]

ScoreSampler = Callable[[], dict[str, float]]


@dataclass
class PolicyDecision:
    """The chosen model, how it was chosen, and the choice probability of every available model."""

    model: str
    decision: Decision
    probs: dict[str, float]


class RoutingPolicy(Protocol):
    def choose(
        self,
        scores: dict[str, float],
        *,
        current: str | None,
        r: float,
        available: list[str],
        rng: random.Random,
        sample_scores: ScoreSampler | None = None,
    ) -> PolicyDecision:
        """Pick the next model among ``available``."""
        ...


def _check_available(available: list[str]) -> None:
    if not available:
        raise ValueError("no model available to route to")


def _finite(scores: Mapping[str, float], model: str) -> float | None:
    v = scores.get(model)
    return v if v is not None and math.isfinite(v) else None


def _forced(model: str) -> PolicyDecision:
    return PolicyDecision(model, "forced", {model: 1.0})


def _sample(probs: Mapping[str, float], rng: random.Random) -> str:
    u = rng.random()
    acc = 0.0
    last = ""
    for m, p in probs.items():
        if p <= 0:
            continue
        acc += p
        last = m
        if u < acc:
            return m
    return last  # floating-point slack: u >= sum of probabilities


def _best(scores: Mapping[str, float], available: list[str]) -> list[str]:
    """Available models sharing the best finite score (all available if none is finite)."""
    finite = {m: v for m in available if (v := _finite(scores, m)) is not None}
    if not finite:
        return list(available)
    top = max(finite.values())
    return [m for m, v in finite.items() if v == top]


class SoftmaxPolicy:
    """Temperature-scaled softmax over scores with a stickiness bonus for the current model."""

    def __init__(self, T0: float = 0.5, beta: float = 0.25, r_min: float = 0.05):
        if T0 <= 0 or r_min <= 0:
            raise ValueError(f"T0 and r_min must be > 0, got T0={T0}, r_min={r_min}")
        self.T0 = T0
        self.beta = beta
        self.r_min = r_min

    def temperature(self, r: float) -> float:
        """``T = T0 / max(r, r_min)`` (months)."""
        return self.T0 / max(r, self.r_min)

    def probabilities(self, scores: Mapping[str, float], *, current: str | None, r: float,
                      available: list[str]) -> dict[str, float]:
        """Choice probabilities over ``available`` (numerically stable; uniform if no score is finite)."""
        _check_available(available)
        t = self.temperature(r)
        logits = {}
        for m in available:
            v = _finite(scores, m)
            if v is not None:
                logits[m] = (v + (self.beta if m == current else 0.0)) / t
        if not logits:
            return {m: 1.0 / len(available) for m in available}
        top = max(logits.values())
        weights = {m: math.exp(logits[m] - top) if m in logits else 0.0 for m in available}
        total = sum(weights.values())
        return {m: w / total for m, w in weights.items()}

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """Sample from :meth:`probabilities`; ``"hold"`` when the sample is the current model."""
        _check_available(available)
        if len(available) == 1:
            return _forced(available[0])
        probs = self.probabilities(scores, current=current, r=r, available=available)
        model = _sample(probs, rng)
        return PolicyDecision(model, "hold" if model == current else "sample", probs)


class ThompsonPolicy:
    """Argmax of one posterior draw of the scores (``sample_scores()``); ties broken at random.

    ``probs`` reports each model's share of wins over ``n_prob_draws`` draws (the first of which is the one acted
    on). Without a sampler the deterministic ``scores`` are used.
    """

    def __init__(self, n_prob_draws: int = 32):
        if n_prob_draws < 1:
            raise ValueError(f"n_prob_draws must be >= 1, got {n_prob_draws}")
        self.n_prob_draws = n_prob_draws

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """Act on the argmax of one draw; ``"hold"`` when it is the current model."""
        _check_available(available)
        if len(available) == 1:
            return _forced(available[0])
        draw = sample_scores or (lambda: scores)
        wins = {m: 0 for m in available}
        model = ""
        for i in range(self.n_prob_draws):
            winner = rng.choice(_best(draw(), available))
            wins[winner] += 1
            if i == 0:
                model = winner
        probs = {m: c / self.n_prob_draws for m, c in wins.items()}
        return PolicyDecision(model, "hold" if model == current else "sample", probs)


class ArgmaxHysteresisPolicy:
    """Deterministic best score; leave the current model only if the best beats it by at least ``delta`` months."""

    def __init__(self, delta: float = 0.5):
        if delta < 0:
            raise ValueError(f"delta must be >= 0, got {delta}")
        self.delta = delta

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """``"argmax"`` for a switch (or first pick), ``"hold"`` when the margin is below ``delta``."""
        _check_available(available)
        if len(available) == 1:
            return _forced(available[0])
        candidates = _best(scores, available)
        if current in candidates:
            return PolicyDecision(current, "hold", {current: 1.0})
        best = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
        if current in available:
            cur = _finite(scores, current)
            top = _finite(scores, best)
            if cur is not None and top is not None and top - cur < self.delta:
                return PolicyDecision(current, "hold", {current: 1.0})
        return PolicyDecision(best, "argmax", {best: 1.0})


class FixedPolicy:
    """Always the same model; if it is unavailable, fall back to the best-scoring available model."""

    def __init__(self, model: str):
        if not model:
            raise ValueError("FixedPolicy needs a model key")
        self.model = model

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """Decision ``"forced"``."""
        _check_available(available)
        if self.model in available:
            return _forced(self.model)
        return _forced(rng.choice(_best(scores, available)))


class RoundRobinPolicy:
    """Cycle through ``available`` in order: the model after ``current`` (the first one if there is none)."""

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """Decision ``"forced"``."""
        _check_available(available)
        nxt = available[(available.index(current) + 1) % len(available)] if current in available else available[0]
        return _forced(nxt)


class RandomPolicy:
    """Uniformly random model among ``available``."""

    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: ScoreSampler | None = None) -> PolicyDecision:
        """Decision ``"sample"`` (``"forced"`` with a single available model)."""
        _check_available(available)
        if len(available) == 1:
            return _forced(available[0])
        return PolicyDecision(rng.choice(available), "sample", {m: 1.0 / len(available) for m in available})


_POLICY_NAMES = "softmax, thompson, argmax, fixed:<model key>, round_robin, random"


def make_policy(name: str, cfg: RouterConfig | None = None) -> RoutingPolicy:
    """Build a policy by name; parameters (T0, beta, r_min, delta) come from ``cfg`` (defaults if ``None``)."""
    key = name.strip()
    if key == "softmax":
        if cfg is None:
            return SoftmaxPolicy()
        return SoftmaxPolicy(cfg.T0, cfg.beta, cfg.r_min)
    if key == "thompson":
        return ThompsonPolicy()
    if key == "argmax":
        return ArgmaxHysteresisPolicy() if cfg is None else ArgmaxHysteresisPolicy(cfg.delta)
    if key.startswith("fixed:"):
        return FixedPolicy(key.split(":", 1)[1].strip())
    if key == "round_robin":
        return RoundRobinPolicy()
    if key == "random":
        return RandomPolicy()
    raise ValueError(f"unknown routing policy {name!r}; expected one of: {_POLICY_NAMES}")
