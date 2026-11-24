"""Backend contract re-exports, the shared error type and small helpers used by every backend."""

from __future__ import annotations

from cutoffmix.types import Backend, ChatResult, ContinuationRequest, ModelSpec, StreamEvent

__all__ = ["Backend", "BackendError", "ChatResult", "StreamEvent", "continuation_max_tokens", "is_retryable_status"]

RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


def is_retryable_status(status: int | None) -> bool:
    """True for HTTP statuses worth retrying (timeouts, rate limits, provider/server errors)."""
    return status in RETRYABLE_STATUS


class BackendError(Exception):
    """A failed backend call.

    ``status`` is the HTTP status (or the gateway's error code), ``error_type`` OpenRouter's
    ``error.metadata.error_type`` when present, ``retryable`` whether the same request may succeed later, and
    ``retry_after`` the server-requested delay in seconds, if any.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        error_type: str | None = None,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.retryable = retryable
        self.retry_after = retry_after

    def __str__(self) -> str:
        parts = [p for p in (str(self.status) if self.status is not None else None, self.error_type) if p]
        prefix = f"[{' '.join(parts)}] " if parts else ""
        return prefix + super().__str__()


def continuation_max_tokens(model: ModelSpec, req: ContinuationRequest) -> int:
    """``req.max_tokens``, else the visible target plus the model's hidden-reasoning slack (they share the budget)."""
    return req.max_tokens if req.max_tokens is not None else req.target_tokens + model.max_tokens_slack
