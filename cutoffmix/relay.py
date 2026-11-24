"""The relay: one shared chain of working notes, written segment by segment by the routed model (DESIGN.md §5).

Each segment is a fresh request (static system prompt + the notes so far, see :mod:`cutoffmix.prompts`). For every
segment the router reads the window (question + last N segments) and picks the model whose topic-specific knowledge
cutoff is the most recent; the chosen model's visible paragraph is streamed, cleaned and committed to the notes.
When a segment contains the final-answer marker, or the segment cap is reached, the answer phase runs.

Event protocol of :meth:`Relay.run` (every attempt that streams from a backend is bracketed):

- ``segment_start`` (think) or ``phase`` (answer) opens an attempt; ``data`` carries ``probs``, ``r``, ``attempt``
  and ``strategy``. A retry or re-route of the same segment opens a new attempt with the same ``segment`` index.
- ``delta`` events stream display text. The first characters are buffered so a preamble or an echo of the notes can
  be stripped, and nothing at or after the sentinel (or, while taking notes, the final-answer marker) is ever shown.
- ``segment_end`` closes the attempt. Its ``text`` is authoritative (clients replace the streamed text with it).
  A failed attempt ends with ``data["dropped"] = True`` and a ``reason`` (refused | error | length | empty) and is not
  part of the trace.
- ``error`` when no model is left, then always one final ``done`` whose ``data`` is ``trace.to_dict()`` plus
  ``usage`` (per model), ``usage_total``, ``latency``, ``status`` and bookkeeping. A stream closed before its usage
  chunk (sentinel, cut, marker, overflow) has no token counts or cost: such calls are listed in ``usage_missing``
  (with the backend's ``generation_id`` when it reported one) and ``usage_complete`` is then False, so
  ``usage_total`` is a lower bound.
- When every model left for a think segment returns nothing usable (``empty``: it stopped at once, wrote only the
  sentinel, or only echoed the notes) and the notes are not empty, the notes count as concluded and the answer phase
  runs instead of failing the run.

The trace holds exactly the committed segments (one per non-dropped ``segment_end``, the answer as a segment with
phase ``"answer"``), so it always agrees with the event stream.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import re
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cutoffmix.prompts import clean_segment, split_final, strip_preamble, trim_overlap
from cutoffmix.types import (
    Backend,
    ContinuationRequest,
    Decision,
    ModelSpec,
    Phase,
    RelayEvent,
    RelayTrace,
    Segment,
)

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig
    from cutoffmix.router.index import CutoffIndex
    from cutoffmix.router.router import RouteDecision, Router

__all__ = [
    "BUFFER_CHARS",
    "Relay",
    "RelayRuntime",
    "mock_truth_index",
    "to_jsonable",
]

BUFFER_CHARS = 40  # display text held back at the start of a segment so preambles/echoes can be stripped
_MARKER_LAG_EXTRA = 8  # display lag beyond the sentinel/marker length (markdown variants: **Final answer:**)
_CUT_FACTOR = 1.5  # stop reading a think segment once its visible text exceeds hard_cut_chars * this
_RUNAWAY_FACTOR = 4  # safety: stop reading even a non-cancellable stream that keeps going past this many hard cuts
_NO_LIMIT = 1_000_000_000  # "no hard cut" for clean_segment on answers
_MAX_BACKOFF_S = 30.0
_ECHO_MIN_CHARS = 12  # same minimum as prompts.trim_overlap
_USAGE_COUNT_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
_CLOSED_FINISHES = ("sentinel", "cut")  # finish reasons of a stream the backend closed before the provider finished

Outcome = Literal["ok", "refused", "error", "length", "empty"]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """A JSON-safe copy of ``obj``: non-finite floats become ``None``, tuples/sets lists, numpy values plain Python,
    dataclasses dicts and anything else its ``str()``. Use it before ``json.dumps`` (scores can be ``-inf``)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    tolist = getattr(obj, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        return to_jsonable(tolist())
    return str(obj)


def _rounded(values: Mapping[str, float], digits: int) -> dict[str, float]:
    return {k: round(float(v), digits) if math.isfinite(float(v)) else float(v) for k, v in values.items()}


def _add_usage(acc: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Add every numeric field of ``usage`` (recursively, e.g. ``completion_tokens_details``) into ``acc``."""
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            prev = acc.get(key, 0)
            acc[key] = (prev if isinstance(prev, (int, float)) else 0) + value
        elif isinstance(value, Mapping):
            sub = acc.get(key)
            if not isinstance(sub, dict):
                sub = acc[key] = {}
            _add_usage(sub, value)


def _has_counts(usage: Mapping[str, Any] | None) -> bool:
    """Whether ``usage`` holds token counts or a cost (not just a ``generation_id``)."""
    return usage is not None and any(key in usage for key in _USAGE_COUNT_KEYS)


def _sentinel_regex(sentinel: str) -> re.Pattern[str] | None:
    """Tolerant pattern for mangled sentinels (``<END_STEP>``, ``<<end step>>``); needs an opening bracket."""
    core = sentinel.strip("<>[]{}() ")
    if len(core) < 3:
        return None
    core_re = r"[_ ]".join(re.escape(part) for part in re.split(r"[_ ]", core))
    return re.compile(r"[<\[]{1,2}[ \t]*" + core_re, re.IGNORECASE)


async def _aclose(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------------------
# Streaming display filter and final clean-up
# --------------------------------------------------------------------------------------


class _StreamFilter:
    """Turns the raw text of one streamed attempt into display deltas.

    - Holds back the first :data:`BUFFER_CHARS` characters (longer while they still look like an echo of the notes)
      and then strips a preamble / echoed overlap once, fixing where display text starts.
    - Never emits text at or after the sentinel (exact or mangled) or, in the think phase, the final-answer marker.
    - Lags the end of the received text by a few characters, so a sentinel or marker split across deltas is
      recognised before any of it could be shown.
    - Caps think-phase display text at ``hard_cut_chars``.

    ``stop_reason`` tells the caller why the remaining stream is no longer needed (``"sentinel"``, ``"mangled"``,
    ``"marker"`` or ``"overflow"``).
    """

    def __init__(self, *, phase: Phase, notes_tail: str, sentinel: str, final_marker: str, hard_cut_chars: int,
                 stop_at_marker: bool) -> None:
        self.phase = phase
        self.notes_tail = notes_tail
        self.sentinel = sentinel
        self.final_marker = final_marker
        self.hard_cut: int | None = hard_cut_chars if phase == "think" else None
        self.stop_at_marker = stop_at_marker
        self.lag = max(len(sentinel), len(final_marker)) + _MARKER_LAG_EXTRA
        self._mangled = _sentinel_regex(sentinel)
        self.raw = ""
        self.start: int | None = None
        self.emitted = 0  # end offset of the display text emitted so far
        self.stop_reason: str | None = None

    def feed(self, piece: str) -> str:
        """Add a raw delta; return the display text that may be shown now (possibly empty)."""
        self.raw += piece
        return self._advance(final=False)

    def flush(self) -> str:
        """The stream is over: return the display text still held back."""
        return self._advance(final=True)

    def _visible(self) -> tuple[str, bool]:
        """The raw text that may ever be displayed, and whether it is closed (nothing more will be added)."""
        raw = self.raw
        cut = raw.find(self.sentinel)
        reason = "sentinel" if cut >= 0 else None
        mangled = self._mangled.search(raw) if self._mangled is not None else None
        if mangled is not None and (cut < 0 or mangled.start() < cut):
            cut = mangled.start()
            # the exact sentinel may still be arriving ("<<END_STEP" + ">>"): hide it, but do not stop yet
            reason = None if self.sentinel.startswith(raw[cut:]) else "mangled"
        text = raw[:cut] if cut >= 0 else raw
        closed = cut >= 0
        if self.phase == "think":
            notes, answer = split_final(text, self.final_marker)
            if answer is not None:
                text, closed = notes, True
                if self.stop_at_marker and reason is None:
                    reason = "marker"
        if reason is None and self.hard_cut is not None and len(text) - (self.start or 0) > self.hard_cut * _CUT_FACTOR:
            reason = "overflow"
        if reason is not None and self.stop_reason is None:
            self.stop_reason = reason
        return text, closed

    def _find_start(self, text: str, done: bool) -> int | None:
        if not done and len(text) < BUFFER_CHARS:
            return None
        s = strip_preamble(text)
        if self.phase == "think":
            s = trim_overlap(self.notes_tail, s).lstrip()
            echo_window = self.hard_cut if self.hard_cut is not None else BUFFER_CHARS
            if (not done and len(text) < echo_window and len(s) >= _ECHO_MIN_CHARS
                    and s in self.notes_tail):
                return None  # still repeating the notes: wait until the echo ends
        else:
            notes, answer = split_final(s, self.final_marker)
            if answer is not None and not notes.strip():
                s = answer
        if not s.strip():
            return len(text) if done else None
        return text.rfind(s)

    def _advance(self, *, final: bool) -> str:
        text, closed = self._visible()
        done = final or closed
        if self.start is None:
            self.start = self._find_start(text, done)
            if self.start is None:
                return ""
            self.emitted = max(self.emitted, self.start)
        limit = len(text) if done else len(text) - self.lag
        if self.hard_cut is not None:
            limit = min(limit, self.start + self.hard_cut)
        if limit <= self.emitted:
            return ""
        out = text[self.emitted:limit]
        self.emitted = limit
        return out


def _finalize_think(raw: str, notes_tail: str, *, sentinel: str, final_marker: str,
                    hard_cut_chars: int) -> tuple[str, str | None, bool]:
    """Authoritative ``(segment text, answer after the final marker or None, saw_sentinel)`` of a think attempt."""
    cut = raw.find(sentinel)
    pre = raw[:cut] if cut >= 0 else raw
    notes, answer = split_final(pre, final_marker)
    text, saw = clean_segment(notes if answer is not None else pre, notes_tail, sentinel=sentinel,
                              hard_cut_chars=hard_cut_chars)
    if answer is not None:
        answer = clean_segment(answer, "", sentinel=sentinel, hard_cut_chars=_NO_LIMIT)[0]
    return text, answer, saw or cut >= 0


def _finalize_answer(raw: str, *, sentinel: str, final_marker: str) -> str:
    """Authoritative text of an answer attempt: cut at the sentinel, preamble and a leading final marker removed."""
    text = clean_segment(raw, "", sentinel=sentinel, hard_cut_chars=_NO_LIMIT)[0]
    notes, answer = split_final(text, final_marker)
    return answer if answer is not None and not notes.strip() else text


# --------------------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------------------


@dataclass
class _Attempt:
    """One streamed call (or draft) and everything learned from it."""

    model: str
    phase: Phase
    index: int
    attempt: int = 0
    raw: str = ""
    text: str = ""
    answer: str | None = None  # think phase: text after the final marker (None without a marker)
    finish: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    retryable: bool = True
    retry_after: float | None = None
    note: str | None = None  # the backend's done.error, e.g. "refused: ..."
    saw_sentinel: bool = False
    aborted: bool = False  # the stream was closed (by the relay or the backend) before the provider finished
    streamed: bool = False  # the backend yielded at least one event (a call that failed before it is not billed)
    started_at: float = 0.0
    ended_at: float = 0.0
    ttft_s: float | None = None  # time to the first text from the model
    duration_s: float = 0.0

    @property
    def outcome(self) -> Outcome:
        if self.finish == "content_filter":
            return "refused"
        if self.error is not None or self.finish == "error":
            return "error"
        usable = bool(self.text) or (self.phase == "think" and self.answer is not None)
        if usable:
            return "ok"
        return "length" if self.finish == "length" else "empty"


@dataclass
class _Produced:
    """Result slot filled by a segment generator (async generators cannot return values)."""

    attempt: _Attempt | None = None
    error: str | None = None
    excluded: set[str] = field(default_factory=set)
    outcomes: dict[str, Outcome] = field(default_factory=dict)  # last failed outcome per model


@dataclass
class _RunState:
    question: str
    run_id: str
    today: str | None
    keys: list[str]
    trace: RelayTrace
    t0: float = field(default_factory=time.perf_counter)
    notes: list[str] = field(default_factory=list)
    current: str | None = None
    disabled: set[str] = field(default_factory=set)
    refusals: dict[str, int] = field(default_factory=dict)
    usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage_missing: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    first_delta_s: float | None = None
    error: str | None = None

    def available(self, excluded: set[str] | frozenset[str] = frozenset()) -> list[str]:
        return [k for k in self.keys if k not in self.disabled and k not in excluded]

    def think_count(self) -> int:
        return sum(1 for s in self.trace.segments if s.phase == "think")

    def count_call(self, att: _Attempt) -> None:
        acc = self.usage.setdefault(att.model, {"calls": 0})
        acc["calls"] += 1
        if att.usage:
            _add_usage(acc, att.usage)
        if att.streamed and not _has_counts(att.usage):  # closed before the usage chunk: tokens and cost unknown
            self.usage_missing.append({
                "index": att.index, "model": att.model, "phase": att.phase, "attempt": att.attempt,
                "finish_reason": att.finish, "generation_id": (att.usage or {}).get("generation_id")})

    def mark_delta(self) -> None:
        if self.first_delta_s is None:
            self.first_delta_s = round(time.perf_counter() - self.t0, 4)

    def done_data(self, status: str) -> dict[str, Any]:
        total: dict[str, Any] = {}
        for acc in self.usage.values():
            _add_usage(total, acc)
        per_model: dict[str, dict[str, Any]] = {}
        for row in self.timings:
            agg = per_model.setdefault(row["model"], {"segments": 0, "ttft": [], "duration": []})
            agg["segments"] += 1
            if row["ttft_s"] is not None:
                agg["ttft"].append(row["ttft_s"])
            agg["duration"].append(row["duration_s"])
        latency = {
            "total_s": round(time.perf_counter() - self.t0, 4),
            "first_delta_s": self.first_delta_s,
            "segments": self.timings,
            "per_model": {
                m: {"segments": a["segments"],
                    "mean_ttft_s": round(sum(a["ttft"]) / len(a["ttft"]), 4) if a["ttft"] else None,
                    "mean_duration_s": round(sum(a["duration"]) / len(a["duration"]), 4)}
                for m, a in per_model.items()
            },
        }
        data = self.trace.to_dict() | {
            "usage": self.usage,
            "latency": latency,
            "usage_total": total,
            "usage_complete": not self.usage_missing,
            "usage_missing": self.usage_missing,
            "run_id": self.run_id,
            "status": status,
            "dropped": self.dropped,
            "refusals": self.refusals,
            "disabled": sorted(self.disabled),
        }
        if self.error is not None:
            data["error"] = self.error
        return data


# --------------------------------------------------------------------------------------
# Relay
# --------------------------------------------------------------------------------------


class Relay:
    """Runs one question through the multi-model relay (see the module docstring for the event protocol).

    ``router`` must provide ``reset()``, ``window()`` and ``decide()`` as in DESIGN.md §3.4 and, for the speculative
    strategy, ``decide_speculative()`` (which scores each draft with ``score_draft()``). A router keeps per-run
    state, so use one Relay (and one Router) per concurrent run; sequential runs may reuse them (the router is reset
    at the start of each run).
    ``retry_base_delay`` is the first back-off delay in seconds after a failed call (doubled per retry, capped at
    30 s; a server-sent ``Retry-After`` wins).
    """

    def __init__(self, cfg: AppConfig, backend: Backend, router: Router, *, models: list[ModelSpec] | None = None,
                 retry_base_delay: float = 0.5) -> None:
        self.cfg = cfg
        self.backend = backend
        self.router = router
        self.models: list[ModelSpec] = list(models) if models is not None else list(cfg.models)
        if not self.models:
            raise ValueError("the relay needs at least one model")
        self._spec = {m.key: m for m in self.models}
        if len(self._spec) != len(self.models):
            raise ValueError("model keys must be unique")
        if retry_base_delay < 0:
            raise ValueError(f"retry_base_delay must be >= 0, got {retry_base_delay}")
        self.retry_base_delay = retry_base_delay

    # ---- public API ---------------------------------------------------------------------------

    async def run(self, question: str, *, run_id: str | None = None) -> AsyncGenerator[RelayEvent, None]:
        """Stream the relay events for ``question`` (ends with exactly one ``done`` event).

        Closing the iterator early (``aclose()``, or cancelling the consuming task) closes the in-flight backend
        stream(s) before returning.
        """
        state = self._new_state(question, run_id=run_id)
        async with contextlib.aclosing(self._run(state)) as events:
            async for event in events:
                yield event

    async def run_to_trace(self, question: str, **kw: Any) -> RelayTrace:
        """Run to completion and return the trace (``kw``: ``run_id``)."""
        state = self._new_state(question, **kw)
        async with contextlib.aclosing(self._run(state)) as events:
            async for _ in events:
                pass
        return state.trace

    # ---- run loop -----------------------------------------------------------------------------

    def _new_state(self, question: str, *, run_id: str | None = None) -> _RunState:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("the question must be a non-empty string")
        today = self.cfg.today_str() if self.cfg.relay.inject_date else None
        return _RunState(question=question.strip(), run_id=run_id or _new_run_id(), today=today,
                         keys=[m.key for m in self.models], trace=RelayTrace(question=question.strip()))

    async def _run(self, st: _RunState) -> AsyncGenerator[RelayEvent, None]:
        rc = self.cfg.relay
        self.router.reset()
        st.trace.started_at = time.time()
        final: _Attempt | None = None
        while st.error is None and st.think_count() < rc.max_segments:
            index = len(st.trace.segments)
            window = self.router.window(st.question, st.notes, rc.window_segments)
            out = _Produced()
            async with contextlib.aclosing(self._think(st, index, window, out)) as events:
                async for event in events:
                    yield event
            if out.attempt is None:
                if st.notes and out.outcomes and all(o == "empty" for o in out.outcomes.values()):
                    break  # no model has anything to add: the notes are concluded, answer from them
                st.error = out.error or f"segment {index} failed"
            elif out.attempt.answer is not None:
                final = out.attempt
                break
        if st.error is None:
            out = _Produced()
            async with contextlib.aclosing(self._answer(st, final, out)) as events:
                async for event in events:
                    yield event
            if out.attempt is None:
                st.error = out.error or "the answer phase failed"
        if st.error is not None:
            yield RelayEvent(type="error", text=st.error, data={"dropped": st.dropped[-10:]})
        st.trace.ended_at = time.time()
        yield RelayEvent(type="done", data=st.done_data("error" if st.error is not None else "ok"))

    async def _think(self, st: _RunState, index: int, window: str, out: _Produced) -> AsyncGenerator[RelayEvent, None]:
        """One think segment: speculative drafting when the gate opens, else the sequential path."""
        rc = self.cfg.relay
        decision: RouteDecision | None = None
        update = True
        if rc.strategy == "speculative":
            avail = st.available()
            if len(avail) > 1:
                decision = self.router.decide(window, current=st.current, available=avail)
                update = False
                if decision.r >= rc.speculative_gate:
                    async with contextlib.aclosing(self._speculative(st, index, window, out)) as events:
                        async for event in events:
                            yield event
                    if out.attempt is not None:
                        return
                    decision = None  # no usable draft: re-decide among the remaining models
        async with contextlib.aclosing(
                self._produce(st, "think", index, window, out, decision=decision, update=update)) as events:
            async for event in events:
                yield event

    async def _answer(self, st: _RunState, final: _Attempt | None, out: _Produced) -> AsyncGenerator[RelayEvent, None]:
        """The answer phase: the inline answer after the marker, or one routed answer call."""
        rc = self.cfg.relay
        index = len(st.trace.segments)
        if rc.answer_mode == "inline" and final is not None and final.answer:
            now = time.time()
            att = _Attempt(model=final.model, phase="answer", index=index, text=final.answer, finish="inline",
                           started_at=now, ended_at=now, ttft_s=0.0)
            yield RelayEvent(type="phase", segment=index, model=att.model, phase="answer", decision="forced",
                             data={"answer_mode": "inline", "attempt": 0})
            st.mark_delta()
            yield RelayEvent(type="delta", segment=index, model=att.model, phase="answer", text=att.text)
            yield self._commit(st, att, "forced", posterior={}, scores={}, extra={"answer_mode": "inline"})
            out.attempt = att
            return
        window = self.router.window(st.question, st.notes, len(st.notes))
        async with contextlib.aclosing(self._produce(st, "answer", index, window, out, update=False)) as events:
            async for event in events:
                yield event

    # ---- sequential path (also used for the routed answer) ---------------------------------------

    async def _produce(self, st: _RunState, phase: Phase, index: int, window: str, out: _Produced, *,
                       decision: RouteDecision | None = None, update: bool = True) -> AsyncGenerator[RelayEvent, None]:
        """Decide, stream, and handle refusals/length/errors until one attempt is committed or no model is left."""
        rc = self.cfg.relay
        excluded = set(out.excluded)
        reasons: dict[str, str] = {m: str(o) for m, o in out.outcomes.items()}
        attempt_no = 0
        while True:
            avail = st.available(excluded)
            if not avail:
                detail = "; ".join(f"{m}: {r}" for m, r in reasons.items()) or "all models are disabled"
                out.error = f"no model left for {'the answer' if phase == 'answer' else f'segment {index}'} ({detail})"
                return
            if decision is None or decision.model not in avail:
                decision = self.router.decide(window, current=st.current, available=avail, update=update)
            update = False
            d, decision = decision, None
            spec = self._spec[d.model]
            base_tokens = (rc.answer_max_tokens if phase == "answer" else rc.segment_tokens) + spec.max_tokens_slack
            max_tokens: int | None = None
            length_retried = False
            failures = 0
            while True:
                att = _Attempt(model=spec.key, phase=phase, index=index, attempt=attempt_no)
                attempt_no += 1
                yield RelayEvent(
                    type="segment_start" if phase == "think" else "phase", segment=index, model=spec.key, phase=phase,
                    decision=d.decision, topic_posterior=_rounded(d.posterior, 4), scores=_rounded(d.scores, 3),
                    data={"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4), "attempt": att.attempt,
                          "strategy": "sequential", "available": avail})
                req = self._request(st, phase, max_tokens)
                async with contextlib.aclosing(self._stream(st, spec, req, att, emit=True)) as events:
                    async for event in events:
                        yield event
                result = att.outcome
                if result == "ok":
                    extra = {"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4)}
                    yield self._commit(st, att, d.decision, posterior=d.posterior, scores=d.scores, extra=extra)
                    out.attempt = att
                    return
                yield self._drop(st, att, result)
                reasons[spec.key] = result if att.error is None else f"{result}: {att.error}"
                out.outcomes[spec.key] = result
                if result == "length" and not length_retried:
                    length_retried = True
                    max_tokens = 2 * base_tokens
                    continue
                if result == "error" and att.retryable and failures < rc.max_retries:
                    failures += 1
                    await asyncio.sleep(self._backoff(failures, att.retry_after))
                    continue
                if result == "refused":
                    self._count_refusal(st, spec.key)
                excluded.add(spec.key)
                break

    # ---- speculative path ------------------------------------------------------------------------

    async def _speculative(self, st: _RunState, index: int, window: str,
                           out: _Produced) -> AsyncGenerator[RelayEvent, None]:
        """Draft the segment with every available model concurrently; the router picks among the usable drafts."""
        avail = st.available()
        req = self._request(st, "think", None)
        tasks = [asyncio.create_task(self._draft(st, self._spec[m], req, index)) for m in avail]
        try:
            drafts: list[_Attempt] = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ok: dict[str, _Attempt] = {}
        failed: dict[str, str] = {}
        for att in drafts:
            result = att.outcome
            if result == "ok":
                ok[att.model] = att
                continue
            failed[att.model] = result
            out.outcomes[att.model] = result
            self._record_drop(st, att, result)
            if result == "refused":
                self._count_refusal(st, att.model)
            if result in ("refused", "empty"):
                out.excluded.add(att.model)  # errors and length overruns get the sequential retry path
        if not ok:
            return
        d = self.router.decide_speculative(window, {m: a.text or a.answer or "" for m, a in ok.items()},
                                           current=st.current)
        att = ok[d.model]
        extra = {"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4)}
        yield RelayEvent(type="segment_start", segment=index, model=att.model, phase="think", decision=d.decision,
                         topic_posterior=_rounded(d.posterior, 4), scores=_rounded(d.scores, 3),
                         data=extra | {"attempt": 0, "strategy": "speculative", "available": list(ok)})
        if att.text:
            st.mark_delta()
            yield RelayEvent(type="delta", segment=index, model=att.model, phase="think", text=att.text)
        extra |= {"drafts": {m: a.text for m, a in ok.items()}, "failed_drafts": failed}
        yield self._commit(st, att, d.decision, posterior=d.posterior, scores=d.scores, extra=extra)
        out.attempt = att

    async def _draft(self, st: _RunState, spec: ModelSpec, req: ContinuationRequest, index: int) -> _Attempt:
        """A non-emitting attempt (retried once with doubled max_tokens after an empty ``length`` finish)."""
        att = _Attempt(model=spec.key, phase="think", index=index)
        async with contextlib.aclosing(self._stream(st, spec, req, att, emit=False)) as events:
            async for _ in events:
                pass
        if att.outcome != "length":
            return att
        retry = _Attempt(model=spec.key, phase="think", index=index, attempt=1)
        req2 = dataclasses.replace(req, max_tokens=2 * (req.target_tokens + spec.max_tokens_slack))
        async with contextlib.aclosing(self._stream(st, spec, req2, retry, emit=False)) as events:
            async for _ in events:
                pass
        return retry

    # ---- one streamed call -------------------------------------------------------------------------

    def _request(self, st: _RunState, phase: Phase, max_tokens: int | None) -> ContinuationRequest:
        rc = self.cfg.relay
        return ContinuationRequest(
            question=st.question, segments=list(st.notes), phase=phase,
            target_tokens=rc.answer_max_tokens if phase == "answer" else rc.segment_tokens, max_tokens=max_tokens,
            today=st.today, sentinel=rc.sentinel, final_marker=rc.final_marker, hard_cut_chars=rc.hard_cut_chars,
            run_id=st.run_id, metadata={"segment": len(st.trace.segments), "phase": phase})

    async def _stream(self, st: _RunState, spec: ModelSpec, req: ContinuationRequest, att: _Attempt, *,
                      emit: bool) -> AsyncGenerator[RelayEvent, None]:
        """Stream one call into ``att``; yields display ``delta`` events when ``emit`` (drafts stay silent)."""
        rc = self.cfg.relay
        notes_tail = "\n\n".join(req.segments[-3:])
        filt = _StreamFilter(phase=req.phase, notes_tail=notes_tail, sentinel=rc.sentinel,
                             final_marker=rc.final_marker, hard_cut_chars=rc.hard_cut_chars,
                             stop_at_marker=rc.answer_mode == "routed")
        runaway = rc.hard_cut_chars * _RUNAWAY_FACTOR if req.phase == "think" else None
        t0 = time.perf_counter()
        att.started_at = time.time()
        stream: Any = None
        try:
            stream = self.backend.stream_continuation(spec, req)
            async for ev in stream:
                att.streamed = True
                if ev.kind == "text":
                    if att.error is not None or filt.stop_reason == "sentinel":
                        continue  # nothing after an error or the sentinel is used
                    if att.ttft_s is None:
                        att.ttft_s = round(time.perf_counter() - t0, 4)
                    piece = filt.feed(ev.text)
                    if piece and emit:
                        st.mark_delta()
                        yield RelayEvent(type="delta", segment=att.index, model=spec.key, phase=req.phase,
                                         text=piece)
                    if filt.stop_reason is not None and (
                            (filt.stop_reason != "sentinel" and spec.cancellable)
                            or (runaway is not None and len(filt.raw) > runaway)):
                        att.aborted = True  # closing the stream stops generation (and billing)
                        break
                elif ev.kind == "error":
                    att.error = ev.error or "the backend reported a stream error"
                elif ev.kind == "done":
                    att.finish = ev.finish_reason
                    att.usage = dict(ev.usage) if ev.usage else None
                    att.note = ev.error
                    att.aborted = att.aborted or ev.finish_reason in _CLOSED_FINISHES
                    break
        except Exception as e:  # noqa: BLE001 -- any backend failure is handled by the retry/exclude policy
            att.error = f"{type(e).__name__}: {e}"
            att.retryable = bool(getattr(e, "retryable", True))
            retry_after = getattr(e, "retry_after", None)
            att.retry_after = float(retry_after) if isinstance(retry_after, (int, float)) else None
        finally:
            if stream is not None:
                await _aclose(stream)
            att.duration_s = round(time.perf_counter() - t0, 4)
            att.ended_at = time.time()
            st.count_call(att)
        att.raw = filt.raw
        if req.phase == "think":
            att.text, att.answer, att.saw_sentinel = _finalize_think(
                filt.raw, notes_tail, sentinel=rc.sentinel, final_marker=rc.final_marker,
                hard_cut_chars=rc.hard_cut_chars)
        else:
            att.text = _finalize_answer(filt.raw, sentinel=rc.sentinel, final_marker=rc.final_marker)
            att.saw_sentinel = filt.stop_reason in ("sentinel", "mangled")
        if emit and att.outcome == "ok":
            tail = filt.flush()
            if tail:
                st.mark_delta()
                yield RelayEvent(type="delta", segment=att.index, model=spec.key, phase=req.phase, text=tail)

    # ---- commit / drop -----------------------------------------------------------------------------

    def _commit(self, st: _RunState, att: _Attempt, decision: Decision, *, posterior: Mapping[str, float],
                scores: Mapping[str, float], extra: dict[str, Any]) -> RelayEvent:
        """Append the attempt to the trace (and the notes) and return its authoritative ``segment_end``."""
        seg = Segment(index=att.index, model=att.model, phase=att.phase, text=att.text, decision=decision,
                      topic_posterior=_rounded(posterior, 4), scores=_rounded(scores, 3),
                      started_at=att.started_at, ended_at=att.ended_at, aborted=att.aborted, usage=att.usage,
                      note=att.finish or "")
        st.trace.segments.append(seg)
        if att.phase == "think":
            if att.text:
                st.notes.append(att.text)
            st.current = att.model
        else:
            st.trace.answer = att.text
        st.timings.append({"index": att.index, "model": att.model, "phase": att.phase, "ttft_s": att.ttft_s,
                           "duration_s": att.duration_s})
        data = extra | {"finish_reason": att.finish, "attempt": att.attempt, "ttft_s": att.ttft_s,
                        "duration_s": att.duration_s, "usage": att.usage, "aborted": att.aborted,
                        "saw_sentinel": att.saw_sentinel}
        if att.phase == "think":
            data["final"] = att.answer is not None
        return RelayEvent(type="segment_end", segment=att.index, model=att.model, phase=att.phase, text=att.text,
                          decision=decision, data=data)

    def _record_drop(self, st: _RunState, att: _Attempt, reason: Outcome) -> dict[str, Any]:
        row = {"index": att.index, "model": att.model, "phase": att.phase, "attempt": att.attempt, "reason": reason,
               "finish_reason": att.finish, "error": att.error or att.note}
        st.dropped.append(row)
        return row

    def _drop(self, st: _RunState, att: _Attempt, reason: Outcome) -> RelayEvent:
        row = self._record_drop(st, att, reason)
        return RelayEvent(type="segment_end", segment=att.index, model=att.model, phase=att.phase,
                          data={"dropped": True, **{k: v for k, v in row.items() if k not in ("index", "model")}})

    def _count_refusal(self, st: _RunState, model: str) -> None:
        st.refusals[model] = st.refusals.get(model, 0) + 1
        if st.refusals[model] >= self.cfg.relay.disable_after_refusals:
            st.disabled.add(model)

    def _backoff(self, failures: int, retry_after: float | None) -> float:
        delay = retry_after if retry_after is not None else self.retry_base_delay * 2 ** (failures - 1)
        return max(0.0, min(delay, _MAX_BACKOFF_S))


# --------------------------------------------------------------------------------------
# Runtime assembly (CLI and server)
# --------------------------------------------------------------------------------------


def mock_truth_index(truths: Mapping[str, Any], models: Sequence[str]) -> CutoffIndex | None:
    """A cutoff index built directly from simulated-model truths (``MockModelTruth``), for ``--mock`` demos.

    tau = the true per-topic cutoff, sd = 0.5 months, hazard = the truth's stale rate. ``meta["source"]`` says that
    it is not a measurement. None if no model has a truth.
    """
    import numpy as np

    from cutoffmix.router.index import CutoffIndex, ProbeIndex
    from cutoffmix.taxonomy import TOPIC_IDS

    keys = [m for m in models if m in truths]
    if not keys:
        return None
    tau = np.array([[truths[m].tau.get(z, truths[m].default_tau) for z in TOPIC_IDS] for m in keys], dtype=float)
    hazard = np.array([[truths[m].stale_rate] * len(TOPIC_IDS) for m in keys], dtype=float)
    probe = ProbeIndex(
        probe="current_d", tau=tau, var=np.full(tau.shape, 0.25), hazard=hazard, abstain=1.0 - hazard,
        n=np.zeros(tau.shape, dtype=int), draws=None,
        global_tau=np.array([truths[m].default_tau for m in keys], dtype=float),
        slope=np.array([truths[m].slope for m in keys], dtype=float),
        plateau=np.array([truths[m].plateau for m in keys], dtype=float),
    )
    meta = {"source": "mock truths", "note": "simulated models (--mock); true cutoffs, not a measurement"}
    return CutoffIndex(keys, list(TOPIC_IDS), {"current_d": probe}, meta)


RouterFactory = Callable[["AppConfig"], "Router"]


class RelayRuntime:
    """What every run needs, set up once: config, backend, cutoff index and topic classifier.

    :meth:`relay` builds a fresh Router and Relay per run (a router keeps per-run state), so one runtime can serve
    concurrent requests. The index and classifier are loaded on first use. With ``mock=True`` and no
    ``router.index_path`` the index is :func:`mock_truth_index` of the mock backend's truths, so simulated runs
    route on the simulated cutoffs. ``router_factory`` replaces the default Router construction (tests).
    """

    def __init__(self, cfg: AppConfig, *, backend: Backend | None = None, mock: bool = False, seed: int = 0,
                 router_factory: RouterFactory | None = None, mock_latency_s: float | None = None) -> None:
        self.cfg = cfg
        self.mock = mock
        self._owns_backend = backend is None
        if backend is None:
            from cutoffmix.backends import make_backend

            backend = make_backend(cfg, mock=mock, seed=seed, mock_latency_s=mock_latency_s)
        self.backend: Backend = backend
        self.router_factory: RouterFactory = router_factory or self._default_router
        self._routing: tuple[CutoffIndex | None, str | None, Any] | None = None

    # ---- routing components ----

    def _load_routing(self) -> tuple[CutoffIndex | None, str | None, Any]:
        if self._routing is None:
            from cutoffmix.router.index import load_index
            from cutoffmix.router.topics import load_classifier

            rc = self.cfg.router
            truths = getattr(self.backend, "truths", None)
            source: str | None
            if self.mock and rc.index_path is None and isinstance(truths, Mapping):
                index, source = mock_truth_index(truths, self.cfg.model_keys()), "mock truths"
            else:
                index = load_index(rc.index_path)
                source = rc.index_path or ("packaged index_prior.json" if index is not None else None)
            self._routing = (index, source, load_classifier(rc.classifier, rc.classifier_path))
        return self._routing

    @property
    def index(self) -> CutoffIndex | None:
        """The cutoff index used for routing (None: all scores 0, the policy mixes uniformly)."""
        return self._load_routing()[0]

    @property
    def index_source(self) -> str | None:
        """Where the index came from: a path, ``"mock truths"`` or None."""
        return self._load_routing()[1]

    def _default_router(self, cfg: AppConfig) -> Router:
        from cutoffmix.router.router import Router

        index, _, classifier = self._load_routing()
        return Router(cfg.model_keys(), index=index, classifier=classifier, cfg=cfg.router, seed=cfg.router.seed)

    # ---- per run ----

    def configure(self, *, policy: str | None = None, strategy: str | None = None,
                  segment_tokens: int | None = None) -> AppConfig:
        """A validated copy of the config with per-run overrides (raises ``ValueError``/``ConfigError``)."""
        cfg = self.cfg
        relay_changes: dict[str, Any] = {}
        if strategy is not None:
            relay_changes["strategy"] = strategy
        if segment_tokens is not None:
            relay_changes["segment_tokens"] = segment_tokens
        relay = dataclasses.replace(cfg.relay, **relay_changes) if relay_changes else cfg.relay
        router = dataclasses.replace(cfg.router, policy=policy) if policy is not None else cfg.router
        if relay is cfg.relay and router is cfg.router:
            return cfg
        return dataclasses.replace(cfg, relay=relay, router=router)

    def relay(self, cfg: AppConfig | None = None) -> Relay:
        """A new Relay (with a new Router) for one run with ``cfg`` (default: the runtime's config)."""
        run_cfg = cfg or self.cfg
        return Relay(run_cfg, self.backend, self.router_factory(run_cfg))

    async def aclose(self) -> None:
        """Close the backend if this runtime created it."""
        if self._owns_backend:
            await _aclose(self.backend)
