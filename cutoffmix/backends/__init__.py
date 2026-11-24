"""Model backends: the real OpenRouter gateway and a deterministic mock, behind the ``types.Backend`` contract."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from cutoffmix.backends.base import Backend, BackendError, ChatResult, StreamEvent
from cutoffmix.backends.mock import MockBackend, MockModelTruth, default_mock_truths
from cutoffmix.backends.openrouter import OpenRouterBackend, parse_sse

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig
    from cutoffmix.types import Event

__all__ = [
    "Backend", "BackendError", "ChatResult", "MockBackend", "MockModelTruth", "OpenRouterBackend", "StreamEvent",
    "default_mock_truths", "make_backend", "parse_sse",
]


def make_backend(cfg: AppConfig, *, mock: bool = False, seed: int = 0, events: Sequence[Event] | None = None,
                 mock_latency_s: float | None = None) -> Backend:
    """The backend for a configuration: OpenRouter (key from ``cfg.api_key_env`` at the first request) or, with
    ``mock=True``, a :class:`MockBackend`. Configured models without a default mock truth get a neutral one.

    Mock only: ``events`` (benchmark events) let the mock answer probes about them; ``mock_latency_s`` overrides
    the per-chunk streaming delay of every simulated model.
    """
    if not mock:
        return OpenRouterBackend(base_url=cfg.base_url, app_title=cfg.app_title, api_key_env=cfg.api_key_env)
    truths = default_mock_truths()
    neutral = sum(t.default_tau for t in truths.values()) / len(truths)
    for key in cfg.model_keys():
        truths.setdefault(key, MockModelTruth(key=key, tau={}, default_tau=neutral))
    if mock_latency_s is not None:
        for truth in truths.values():
            truth.latency_s = mock_latency_s
    by_id = {e.id: e for e in events} if events else None
    return MockBackend(truths, seed=seed, events=by_id, today=cfg.today)
