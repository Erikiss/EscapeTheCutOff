"""The router: window text -> topic posterior + time sensitivity -> per-model scores -> policy decision."""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.index import CutoffIndex, load_index
from cutoffmix.router.policy import RoutingPolicy, make_policy
from cutoffmix.router.topics import KeywordTopicClassifier, load_classifier
from cutoffmix.types import Decision, TopicClassifier

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig, RouterConfig

__all__ = ["RouteDecision", "Router"]


@dataclass
class RouteDecision:
    """One routing decision with everything the UI/trace shows about it."""

    model: str
    decision: Decision
    probs: dict[str, float]  # choice probability per available model
    scores: dict[str, float]  # U_m per available model (all 0.0 without an index)
    posterior: dict[str, float]  # smoothed topic posterior used for the scores
    r: float  # time sensitivity of the window


def _default_router_config() -> "RouterConfig":
    from cutoffmix.config import RouterConfig

    return RouterConfig()


class Router:
    """Routes each segment of the shared notes to the model with the freshest knowledge of the window's topic.

    Public attributes: ``models``, ``index`` (may be ``None``: all scores 0, so the policy mixes uniformly),
    ``classifier``, ``gate``, ``policy``, ``cfg`` (a ``RouterConfig``), ``probe`` (index probe used for scoring)
    and ``rng`` (``random.Random`` shared with the policy).
    """

    def __init__(
        self,
        models: list[str],
        *,
        index: CutoffIndex | None,
        classifier: TopicClassifier | None = None,
        gate: TimeSensitivityGate | None = None,
        policy: RoutingPolicy | None = None,
        cfg: "RouterConfig | None" = None,
        seed: int | None = None,
    ):
        if not models:
            raise ValueError("the router needs at least one model")
        self.models: list[str] = list(models)
        self.cfg = cfg if cfg is not None else _default_router_config()
        self.index = index
        self.classifier: TopicClassifier = classifier if classifier is not None else KeywordTopicClassifier()
        self.gate = gate if gate is not None else TimeSensitivityGate()
        self.policy: RoutingPolicy = policy if policy is not None else make_policy(self.cfg.policy, self.cfg)
        self.seed = seed if seed is not None else self.cfg.seed
        self.probe: str | None = index.primary(self.cfg.probe_preference).probe if index is not None else None
        self._posterior: dict[str, float] | None = None
        self.rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

    @classmethod
    def from_config(cls, cfg: "AppConfig") -> "Router":
        """Router for ``cfg.model_keys()`` with the configured index, classifier, policy and seed."""
        rc = cfg.router
        return cls(
            cfg.model_keys(),
            index=load_index(rc.index_path),
            classifier=load_classifier(rc.classifier, rc.classifier_path),
            cfg=rc,
            seed=rc.seed,
        )

    def reset(self) -> None:
        """Start a new run: forget the smoothed topic posterior and re-seed the random state (if seeded)."""
        self._posterior = None
        self.rng = random.Random(self.seed)
        self._np_rng = np.random.default_rng(self.seed)

    @staticmethod
    def window(question: str, segments: list[str], n: int) -> str:
        """Routing window: the question plus the last ``n`` segments, separated by blank lines."""
        parts = [question.strip(), *(s.strip() for s in (segments[-n:] if n > 0 else []))]
        return "\n\n".join(p for p in parts if p)

    def _classify(self, text: str) -> dict[str, float]:
        return dict(zip(self.classifier.topics, self.classifier.posterior(text)))

    def _scores(self, posterior: Mapping[str, float], models: list[str]) -> dict[str, float]:
        if self.index is None:
            return {m: 0.0 for m in models}
        return self.index.scores(posterior, probe=self.probe, kappa=self.cfg.kappa, lam=self.cfg.lam, models=models)

    def decide(
        self, window_text: str, *, current: str | None, available: list[str] | None = None, update: bool = True
    ) -> RouteDecision:
        """Choose the model for the next segment.

        The topic posterior is an EMA over windows (weight ``cfg.ema`` on the newest); ``update=False`` computes the
        decision without storing the new posterior. ``available`` defaults to all router models.
        """
        avail = list(self.models if available is None else available)
        if not avail:
            raise ValueError("no model available to route to")
        new = self._classify(window_text)
        prev = self._posterior
        if prev is None:
            posterior = new
        else:
            w = self.cfg.ema
            posterior = {t: w * new.get(t, 0.0) + (1.0 - w) * prev.get(t, 0.0) for t in {**prev, **new}}
        if update:
            self._posterior = posterior
        r = self.gate.score(window_text)
        scores = self._scores(posterior, avail)
        sampler = None
        if self.index is not None:
            index = self.index

            def sampler() -> dict[str, float]:
                return index.sample_scores(posterior, self._np_rng, probe=self.probe, models=avail)

        choice = self.policy.choose(scores, current=current, r=r, available=avail, rng=self.rng, sample_scores=sampler)
        return RouteDecision(choice.model, choice.decision, choice.probs, scores, posterior, r)

    def score_draft(self, window_text: str, draft: str, model: str) -> float:
        """Look-ahead score: ``U_model`` on the topic posterior of ``window + draft`` (0.0 without an index)."""
        posterior = self._classify(f"{window_text}\n\n{draft}" if window_text else draft)
        return self._scores(posterior, [model])[model]

    def decide_speculative(self, window_text: str, drafts: Mapping[str, str], *, current: str | None) -> RouteDecision:
        """Pick one of several parallel drafts (speculative strategy) with decision ``"speculative"``.

        Each draft is scored with :meth:`score_draft`, then the policy chooses among the drafting models with the
        window's time sensitivity. ``posterior`` is that of ``window + chosen draft``; the stored EMA is untouched.
        """
        if not drafts:
            raise ValueError("no drafts to choose from")
        scores = {m: self.score_draft(window_text, d, m) for m, d in drafts.items()}
        r = self.gate.score(window_text)
        choice = self.policy.choose(scores, current=current, r=r, available=list(drafts), rng=self.rng)
        posterior = self._classify(f"{window_text}\n\n{drafts[choice.model]}")
        return RouteDecision(choice.model, "speculative", choice.probs, scores, posterior, r)
