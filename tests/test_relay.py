"""Tests for cutoffmix.relay: a scripted fake backend and fake router, plus integration with the real ones."""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from cutoffmix.config import AppConfig, RelayConfig, RouterConfig
from cutoffmix.relay import BUFFER_CHARS, Relay, RelayRuntime, to_jsonable
from cutoffmix.types import ContinuationRequest, ModelSpec, RelayEvent, StreamEvent

SENTINEL = "<<END_STEP>>"
MODELS = [
    ModelSpec(key="a", provider_id="x/a", display="Model A", color="#d97757", max_tokens_slack=100),
    ModelSpec(key="b", provider_id="x/b", display="Model B", color="#10a37f", supports_stop=False,
              max_tokens_slack=100),
    ModelSpec(key="c", provider_id="x/c", display="Model C", color="#4285f4", cancellable=False,
              max_tokens_slack=50),
]


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------

Script = Callable[[str, ContinuationRequest, int], Iterable[Any]]


class FakeBackend:
    """Scripted ``types.Backend``: ``script(model_key, request, call_no)`` returns items to play back; an item is a
    StreamEvent (yielded), an exception (raised) or a number (seconds to sleep)."""

    def __init__(self, script: Script) -> None:
        self.script = script
        self.calls: list[tuple[str, ContinuationRequest]] = []
        self.open = 0
        self.max_open = 0
        self.closed = 0
        self.consumed: list[int] = []  # items played back per call

    async def stream_continuation(self, model: ModelSpec, req: ContinuationRequest):
        call_no = len(self.calls)
        self.calls.append((model.key, req))
        self.consumed.append(0)
        self.open += 1
        self.max_open = max(self.max_open, self.open)
        try:
            for item in self.script(model.key, req, call_no):
                self.consumed[call_no] += 1
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, (int, float)):
                    await asyncio.sleep(item)
                    continue
                yield item
        finally:
            self.open -= 1
            self.closed += 1

    async def chat(self, model, req):  # pragma: no cover - the relay never calls it
        raise NotImplementedError


@dataclass
class FakeDecision:
    model: str
    decision: str
    probs: dict[str, float]
    scores: dict[str, float]
    posterior: dict[str, float]
    r: float


class FakeRouter:
    """Implements the Router contract used by the relay; ``pick(available, current, n)`` chooses the model."""

    def __init__(self, models: list[str], *, pick: Callable[[list[str], str | None, int], str] | None = None,
                 r: float = 0.9, draft_scores: dict[str, float] | None = None) -> None:
        self.models = models
        self.pick = pick or (lambda avail, current, n: avail[0])
        self.r = r
        self.draft_scores = draft_scores or {}
        self.decide_calls: list[dict[str, Any]] = []
        self.scored: list[tuple[str, str]] = []
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    @staticmethod
    def window(question: str, segments: list[str], n: int) -> str:
        return "\n\n".join([question, *(segments[-n:] if n > 0 else [])])

    def decide(self, window_text, *, current, available=None, update=True):
        avail = list(self.models if available is None else available)
        self.decide_calls.append({"window": window_text, "current": current, "available": avail, "update": update})
        model = self.pick(avail, current, len(self.decide_calls) - 1)
        return FakeDecision(model, "hold" if model == current else "sample", {m: 1 / len(avail) for m in avail},
                            {m: 27.0 for m in avail}, {"government": 0.7, "sports": 0.3}, self.r)

    def score_draft(self, window_text, draft, model):
        self.scored.append((model, draft))
        return self.draft_scores.get(model, 0.0)

    def decide_speculative(self, window_text, drafts, *, current):
        scores = {m: self.score_draft(window_text, d, m) for m, d in drafts.items()}
        best = max(drafts, key=lambda m: scores[m])  # argmax over the drafts' look-ahead scores
        return FakeDecision(best, "speculative", {m: float(m == best) for m in drafts}, scores, {"government": 1.0},
                            self.r)


class FlakyError(Exception):
    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = None


def text_events(text: str, *, chunk: int = 8, finish: str = "stop", usage: dict | None = None) -> list[Any]:
    pieces: list[Any] = [StreamEvent(kind="text", text=text[i:i + chunk]) for i in range(0, len(text), chunk)]
    return [*pieces, StreamEvent(kind="done", finish_reason=finish, usage=usage)]


def para(n: int) -> str:
    return f"Step {n}: the evidence for item {n} is consistent with every record known so far."


def make_cfg(**relay: Any) -> AppConfig:
    relay.setdefault("segment_tokens", 64)
    return AppConfig(models=list(MODELS), relay=RelayConfig(**relay), router=RouterConfig(), today="2026-11-24")


def make_relay(script: Script, *, router: FakeRouter | None = None, **relay: Any) -> tuple[Relay, FakeBackend,
                                                                                              FakeRouter]:
    backend = FakeBackend(script)
    router = router or FakeRouter([m.key for m in MODELS])
    return Relay(make_cfg(**relay), backend, router, retry_base_delay=0.0), backend, router


async def collect(relay: Relay, question: str = "Who is the chancellor right now?") -> list[RelayEvent]:
    return [ev async for ev in relay.run(question, run_id="run-1")]


def deltas_by_segment(events: list[RelayEvent], phase: str = "think") -> dict[int, list[str]]:
    """Streamed text per attempt (keyed by the order of segment starts)."""
    out: dict[int, list[str]] = {}
    attempt = -1
    for ev in events:
        if ev.type in ("segment_start", "phase") and ev.phase == phase:
            attempt += 1
            out[attempt] = []
        elif ev.type == "delta" and ev.phase == phase:
            out[attempt].append(ev.text)
    return out


def committed(events: list[RelayEvent], phase: str = "think") -> list[RelayEvent]:
    return [ev for ev in events if ev.type == "segment_end" and ev.phase == phase
            and not (ev.data or {}).get("dropped")]


def finishing_script(final_at: int = 2, answer: str = "The answer is 42.") -> Script:
    """Think segment k gets para(k) (+ sentinel); think call ``final_at`` concludes; answer calls get ``answer``."""

    def script(model: str, req: ContinuationRequest, n: int) -> list[Any]:
        if req.phase == "answer":
            return text_events(answer, usage={"prompt_tokens": 5, "completion_tokens": 3, "cost": 0.001})
        k = len(req.segments)
        body = "Wrapping up the shared notes here. FINAL ANSWER: 42" if k == final_at else para(k)
        return text_events(body + SENTINEL, usage={"prompt_tokens": 10, "completion_tokens": 20,
                                                   "completion_tokens_details": {"reasoning_tokens": 7},
                                                   "cost": 0.002})

    return script


# --------------------------------------------------------------------------------------
# Sequential path
# --------------------------------------------------------------------------------------


async def test_sequential_run_streams_segments_and_keeps_the_trace_consistent():
    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: avail[n % len(avail)])
    relay, backend, router = make_relay(finishing_script(), router=router)
    events = await collect(relay)

    types = [ev.type for ev in events]
    assert types[0] == "segment_start" and types[-1] == "done" and types.count("done") == 1
    assert "error" not in types
    ends = committed(events)
    streamed = deltas_by_segment(events)
    for i, end in enumerate(ends):
        assert "".join(streamed[i]) == end.text  # streamed text agrees with the authoritative text
    assert [e.text for e in ends] == [para(0), para(1), "Wrapping up the shared notes here."]
    assert ends[-1].data["final"] is True
    assert all("<" not in ev.text and "FINAL" not in ev.text for ev in events if ev.type == "delta")

    done = events[-1].data
    assert done["status"] == "ok" and done["answer"] == "The answer is 42."
    assert [s["text"] for s in done["segments"]] == [e.text for e in ends] + ["The answer is 42."]
    assert [s["phase"] for s in done["segments"]] == ["think"] * 3 + ["answer"]
    assert [s["model"] for s in done["segments"][:3]] == ["a", "b", "c"]
    assert done["switches"] == 2

    # requests: append-only notes, one run id, today's date, the configured target length
    think_reqs = [req for _, req in backend.calls if req.phase == "think"]
    assert [req.segments for req in think_reqs] == [[], [para(0)], [para(0), para(1)]]
    assert {req.run_id for _, req in backend.calls} == {"run-1"}
    assert all(req.today == "2026-11-24" and req.target_tokens == 64 for req in think_reqs)
    answer_req = backend.calls[-1][1]
    assert answer_req.phase == "answer" and answer_req.target_tokens == 1024
    assert answer_req.segments == [para(0), para(1), "Wrapping up the shared notes here."]

    # router use: reset once, EMA update for every think decision, none for the answer decision
    assert router.resets == 1
    assert [c["update"] for c in router.decide_calls] == [True, True, True, False]
    assert router.decide_calls[1]["current"] == "a"


async def test_done_event_aggregates_usage_and_latency_per_model():
    relay, _, _ = make_relay(finishing_script())  # FakeRouter picks "a" every time
    done = (await collect(relay))[-1].data
    usage_a = done["usage"]["a"]
    assert usage_a["calls"] == 4
    # the concluding segment is closed at the final marker (routed answer mode), so its usage never arrives
    assert done["segments"][2]["aborted"] is True and done["segments"][2]["usage"] is None
    assert usage_a["prompt_tokens"] == 2 * 10 + 5
    assert usage_a["completion_tokens_details"] == {"reasoning_tokens": 14}
    assert usage_a["cost"] == pytest.approx(2 * 0.002 + 0.001)
    assert done["usage_total"]["calls"] == 4
    latency = done["latency"]
    assert latency["total_s"] >= 0 and latency["first_delta_s"] is not None
    assert [row["index"] for row in latency["segments"]] == [0, 1, 2, 3]
    assert latency["per_model"]["a"]["segments"] == 4
    assert all(row["ttft_s"] is not None for row in latency["segments"])


async def test_preamble_and_echoed_overlap_never_reach_the_stream():
    first = "Alpha beta gamma delta epsilon zeta eta theta iota."

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Done.")
        if not req.segments:
            return text_events(first + SENTINEL)
        if len(req.segments) == 1:
            return text_events(f"Next paragraph: {first} Kappa lambda mu nu xi omicron pi rho.{SENTINEL}")
        return text_events("FINAL ANSWER: done" + SENTINEL)

    relay, _, _ = make_relay(script)
    events = await collect(relay)
    ends = committed(events)
    assert ends[1].text == "Kappa lambda mu nu xi omicron pi rho."
    streamed = "".join(deltas_by_segment(events)[1])
    assert streamed == ends[1].text
    assert "Next paragraph" not in streamed and "Alpha" not in streamed


async def test_sentinel_split_across_deltas_is_never_shown_and_usage_is_kept():
    chunks = ["The result holds for every case we ", "checked so far, as shown.<<END_", "STEP>> leaked text after"]

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Answer.")
        if req.segments:
            return text_events("FINAL ANSWER: x" + SENTINEL)
        return [*(StreamEvent(kind="text", text=c) for c in chunks),
                StreamEvent(kind="done", finish_reason="sentinel", usage={"prompt_tokens": 11})]

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: "b")  # b: no `stop` support
    relay, _, _ = make_relay(script, router=router)
    events = await collect(relay)
    shown = "".join(ev.text for ev in events if ev.type == "delta" and ev.segment == 0)
    assert shown == "The result holds for every case we checked so far, as shown."
    assert committed(events)[0].text == shown
    assert committed(events)[0].data["saw_sentinel"] is True
    assert events[-1].data["usage"]["b"]["prompt_tokens"] >= 11  # read on to the done event after the sentinel


@pytest.mark.parametrize("ending", [
    "\n\n",  # live Fable: a blank line, then the server-side stop at the sentinel
    "\n" + SENTINEL,  # live Sol: the sentinel on its own line
    " \n \n" + SENTINEL + " trailing",
])
async def test_trailing_whitespace_is_not_streamed(ending):
    body = para(0)

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("The answer is 42." + "\n\n")
        if req.segments:
            return text_events("FINAL ANSWER: 42")
        return text_events(body + ending, chunk=3)

    relay, _, _ = make_relay(script, answer_mode="routed")
    events = await collect(relay)
    ends = committed(events) + committed(events, "answer")
    assert [e.text for e in ends] == [body, "", "The answer is 42."]
    for end in ends:  # the CLI redraws a block (and clients rewrite it) whenever these differ
        streamed = "".join(ev.text for ev in events
                           if ev.type == "delta" and ev.segment == end.segment and ev.phase == end.phase)
        assert streamed == end.text


async def test_final_marker_split_across_deltas_inline_answer():
    chunks = ["Everything needed is in the notes now. FINAL AN", "SWER: Olaf Sch", "olz is chancellor.", SENTINEL]

    def script(model, req, n):
        return [*(StreamEvent(kind="text", text=c) for c in chunks), StreamEvent(kind="done", finish_reason="stop")]

    relay, backend, _ = make_relay(script, answer_mode="inline")
    events = await collect(relay)
    think_text = "".join(ev.text for ev in events if ev.type == "delta" and ev.phase == "think")
    assert think_text == "Everything needed is in the notes now."
    assert "FINAL" not in think_text and "Scholz" not in think_text
    phase = next(ev for ev in events if ev.type == "phase")
    assert phase.phase == "answer" and phase.data["answer_mode"] == "inline"
    answer_deltas = [ev.text for ev in events if ev.type == "delta" and ev.phase == "answer"]
    assert answer_deltas == ["Olaf Scholz is chancellor."]
    assert events[-1].data["answer"] == "Olaf Scholz is chancellor."
    assert len(backend.calls) == 1  # no separate answer call in inline mode


async def test_final_marker_in_routed_mode_stops_reading_and_routes_the_answer():
    tail = ["filler " * 10] * 20

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Routed answer.")
        return [StreamEvent(kind="text", text="All points are settled in the notes. FINAL ANSWER: maybe"),
                *(StreamEvent(kind="text", text=t) for t in tail), StreamEvent(kind="done", finish_reason="stop")]

    relay, backend, _ = make_relay(script)  # model "a" is cancellable
    events = await collect(relay)
    assert backend.consumed[0] < 5  # the stream was closed right after the marker
    seg = events[-1].data["segments"][0]
    assert seg["aborted"] is True and seg["text"] == "All points are settled in the notes."
    assert backend.calls[1][1].phase == "answer"
    assert events[-1].data["answer"] == "Routed answer."


async def test_non_cancellable_model_is_read_to_the_end_after_the_marker():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Routed answer.")
        return [StreamEvent(kind="text", text="All points are settled in the notes. FINAL ANSWER: maybe"),
                StreamEvent(kind="text", text=" more"), StreamEvent(kind="done", finish_reason="stop",
                                                                    usage={"prompt_tokens": 3})]

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: "c")  # c: not cancellable
    relay, backend, _ = make_relay(script, router=router)
    events = await collect(relay)
    assert backend.consumed[0] == 3
    assert events[-1].data["segments"][0]["aborted"] is False
    assert events[-1].data["usage"]["c"]["prompt_tokens"] == 3  # usage of the done event was received


async def test_hard_cut_caps_the_stream_and_the_segment():
    long_text = ("This sentence is part of a very long paragraph that never ends. " * 40)

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if req.segments:
            return text_events("FINAL ANSWER: x" + SENTINEL)
        return text_events(long_text, chunk=20)

    relay, backend, _ = make_relay(script, hard_cut_chars=300)
    events = await collect(relay)
    shown = "".join(ev.text for ev in events if ev.type == "delta" and ev.segment == 0)
    assert len(shown) <= 300
    first = committed(events)[0]
    assert len(first.text) <= 300 and first.text.endswith("ends.")
    assert first.data["aborted"] is True
    assert backend.consumed[0] < len(long_text) // 20  # stopped reading after 1.5x the hard cut


async def test_buffer_holds_back_the_first_characters():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if req.segments:
            return text_events("FINAL ANSWER: x")
        return [StreamEvent(kind="text", text=c) for c in para(0)] + [StreamEvent(kind="done", finish_reason="stop")]

    relay, _, _ = make_relay(script)
    events: list[RelayEvent] = []
    async for ev in relay.run("q?"):
        events.append(ev)
    first_delta = next(ev for ev in events if ev.type == "delta")
    # the first delta only arrives once BUFFER_CHARS characters were received
    assert len(first_delta.text) >= BUFFER_CHARS - max(len(SENTINEL), len("FINAL ANSWER:")) - 8


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------


async def test_refusal_drops_the_attempt_reroutes_and_disables_the_model():
    def script(model, req, n):
        if model == "a":
            return [StreamEvent(kind="done", finish_reason="content_filter", error="refused: policy")]
        if req.phase == "answer":
            return text_events("Answer by someone else.")
        return text_events(("FINAL ANSWER: ok" if len(req.segments) >= 2 else para(len(req.segments))) + SENTINEL)

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: "a" if "a" in avail else avail[0])
    relay, _, _ = make_relay(script, router=router, disable_after_refusals=2)
    events = await collect(relay)
    drops = [ev for ev in events if ev.type == "segment_end" and (ev.data or {}).get("dropped")]
    assert [(d.model, d.data["reason"]) for d in drops] == [("a", "refused"), ("a", "refused")]
    assert all(not ev.text for ev in drops)
    done = events[-1].data
    assert done["refusals"] == {"a": 2} and done["disabled"] == ["a"]
    assert all(s["model"] != "a" for s in done["segments"])
    # after the second refusal "a" is never offered again
    later = router.decide_calls[3:]
    assert later and all("a" not in c["available"] for c in later)
    # a re-decision inside the same segment does not update the topic EMA
    assert router.decide_calls[1]["update"] is False


async def test_length_without_text_retries_once_with_doubled_max_tokens():
    def script(model, req, n):
        if n == 0:
            return [StreamEvent(kind="done", finish_reason="length")]
        if req.phase == "answer":
            return text_events("A.")
        return text_events(("FINAL ANSWER: x" if req.segments else para(0)) + SENTINEL)

    relay, backend, _ = make_relay(script)
    events = await collect(relay)
    assert backend.calls[0][1].max_tokens is None
    assert backend.calls[1][1].max_tokens == 2 * (64 + 100)
    assert backend.calls[0][0] == backend.calls[1][0] == "a"
    drop = next(ev for ev in events if ev.type == "segment_end" and (ev.data or {}).get("dropped"))
    assert drop.data["reason"] == "length"
    assert events[-1].data["status"] == "ok"


async def test_errors_are_retried_then_the_model_is_excluded():
    def script(model, req, n):
        if model == "a":
            return [FlakyError("503 upstream")]
        if req.phase == "answer":
            return text_events("A.")
        return text_events(("FINAL ANSWER: x" if req.segments else para(0)) + SENTINEL)

    relay, backend, router = make_relay(script, max_retries=2)
    events = await collect(relay)
    first_segment_calls = [m for m, req in backend.calls if req.phase == "think" and not req.segments]
    assert first_segment_calls == ["a", "a", "a", "b"]  # 1 try + 2 retries, then re-routed
    assert router.decide_calls[1]["available"] == ["b", "c"]
    assert committed(events)[0].model == "b"
    errors = [ev.data["error"] for ev in events if ev.type == "segment_end" and (ev.data or {}).get("dropped")]
    assert errors[0] == "FlakyError: 503 upstream"


async def test_non_retryable_errors_and_error_events_exclude_without_retry():
    def script(model, req, n):
        if model == "a":
            return [FlakyError("401 no key", retryable=False)]
        if model == "b":
            return [StreamEvent(kind="text", text="partial text that will be dropped"),
                    StreamEvent(kind="error", error="provider failed"),
                    StreamEvent(kind="done", finish_reason="error")]
        if req.phase == "answer":
            return text_events("A.")
        return text_events(("FINAL ANSWER: x" if req.segments else para(0)) + SENTINEL)

    relay, backend, _ = make_relay(script, max_retries=0)
    events = await collect(relay)
    assert [m for m, _ in backend.calls[:3]] == ["a", "b", "c"]
    reasons = [(ev.model, ev.data["reason"]) for ev in events if ev.type == "segment_end"
               and (ev.data or {}).get("dropped")]
    assert reasons[:2] == [("a", "error"), ("b", "error")]
    assert committed(events)[0].model == "c"


async def test_no_model_left_emits_error_then_done():
    def script(model, req, n):
        return [StreamEvent(kind="done", finish_reason="content_filter")]

    relay, _, _ = make_relay(script)
    events = await collect(relay)
    assert [ev.type for ev in events[-2:]] == ["error", "done"]
    assert "no model left" in events[-2].text
    done = events[-1].data
    assert done["status"] == "error" and done["segments"] == [] and done["answer"] == ""
    assert "error" in done


async def test_max_segments_forces_the_answer_phase():
    def script(model, req, n):
        return text_events(("Answer after the cap." if req.phase == "answer" else para(len(req.segments)))
                           + SENTINEL)

    relay, backend, _ = make_relay(script, max_segments=2)
    events = await collect(relay)
    assert len(committed(events)) == 2
    assert [req.phase for _, req in backend.calls] == ["think", "think", "answer"]
    assert events[-1].data["answer"] == "Answer after the cap."


async def test_empty_question_is_rejected():
    relay, _, _ = make_relay(finishing_script())
    with pytest.raises(ValueError):
        await collect(relay, "   ")


# --------------------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------------------


def slow_script(model, req, n):
    for i in range(200):
        yield 0.005
        yield StreamEvent(kind="text", text=f"word{i} " * 3)
    yield StreamEvent(kind="done", finish_reason="stop")


async def test_closing_the_run_closes_the_backend_stream():
    relay, backend, _ = make_relay(slow_script)
    gen = relay.run("A question about the present?")
    async for ev in gen:
        if ev.type == "delta":
            break
    assert backend.open == 1
    await gen.aclose()
    assert backend.open == 0 and backend.closed == 1


async def test_cancelling_the_consumer_task_closes_the_backend_stream():
    relay, backend, _ = make_relay(slow_script)
    seen = asyncio.Event()

    async def consume():
        async for ev in relay.run("A question about the present?"):
            if ev.type == "delta":
                seen.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(seen.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.open == 0


# --------------------------------------------------------------------------------------
# Speculative strategy
# --------------------------------------------------------------------------------------


def drafting_script(delay: float = 0.03) -> Script:
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Speculative answer.")
        if req.segments:
            return [delay, *text_events("FINAL ANSWER: x" + SENTINEL)]
        return [delay, *text_events(f"Draft by {model}: the evidence points one way, as recorded." + SENTINEL)]

    return script


async def test_speculative_drafts_concurrently_and_keeps_the_best_draft():
    router = FakeRouter([m.key for m in MODELS], r=0.9, draft_scores={"a": 1.0, "b": 5.0, "c": 2.0})
    relay, backend, router = make_relay(drafting_script(), router=router, strategy="speculative")
    events = await collect(relay)
    assert backend.max_open == 3  # all three drafts were in flight at once
    starts = [ev for ev in events if ev.type == "segment_start"]
    assert starts[0].decision == "speculative" and starts[0].model == "b"
    assert starts[0].scores == {"a": 1.0, "b": 5.0, "c": 2.0}
    assert len([ev for ev in events if ev.type == "delta" and ev.segment == 0]) == 1
    end = committed(events)[0]
    assert end.text == "Draft by b: the evidence points one way, as recorded."
    assert set(end.data["drafts"]) == {"a", "b", "c"}
    assert end.decision == "speculative"
    assert events[-1].data["segments"][0]["decision"] == "speculative"
    assert events[-1].data["usage_total"]["calls"] == 3 + 3 + 1


async def test_speculative_below_the_gate_uses_the_sequential_path():
    router = FakeRouter([m.key for m in MODELS], r=0.2, draft_scores={"b": 5.0})
    relay, backend, _ = make_relay(drafting_script(0.0), router=router, strategy="speculative")
    events = await collect(relay)
    assert backend.max_open == 1
    assert all(ev.decision != "speculative" for ev in events if ev.type == "segment_start")


async def test_speculative_ignores_failed_drafts():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if model == "a":
            return [StreamEvent(kind="done", finish_reason="content_filter")]
        if model == "b":
            return [FlakyError("boom")]
        return text_events(("FINAL ANSWER: x" if req.segments else para(0)) + SENTINEL)

    router = FakeRouter([m.key for m in MODELS], r=0.9, draft_scores={"a": 9.0, "b": 8.0, "c": 1.0})
    relay, _, _ = make_relay(script, router=router, strategy="speculative")
    events = await collect(relay)
    end = committed(events)[0]
    assert end.model == "c" and set(end.data["drafts"]) == {"c"}
    assert end.data["failed_drafts"] == {"a": "refused", "b": "error"}
    # "a" refuses its draft in both drafted segments, so it is disabled for the rest of the run
    assert events[-1].data["refusals"] == {"a": 2} and events[-1].data["disabled"] == ["a"]


async def test_cancelling_during_drafting_closes_every_draft_stream():
    def script(model, req, n):
        return [5.0, *text_events("never")]

    router = FakeRouter([m.key for m in MODELS], r=0.9)
    relay, backend, _ = make_relay(script, router=router, strategy="speculative")
    task = asyncio.create_task(collect(relay))
    for _ in range(200):
        if backend.open == 3:
            break
        await asyncio.sleep(0.005)
    assert backend.open == 3
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert backend.open == 0 and backend.closed == 3


# --------------------------------------------------------------------------------------
# Helpers and runtime
# --------------------------------------------------------------------------------------


def test_to_jsonable_makes_scores_json_safe():
    data = {"scores": {"a": -math.inf, "b": 1.5}, "t": (1, 2), "nan": math.nan, "nested": [{"x": math.inf}]}
    assert to_jsonable(data) == {"scores": {"a": None, "b": 1.5}, "t": [1, 2], "nan": None, "nested": [{"x": None}]}


def test_runtime_configure_applies_and_validates_overrides():
    runtime = RelayRuntime(make_cfg(), backend=FakeBackend(finishing_script()),
                           router_factory=lambda cfg: FakeRouter(cfg.model_keys()))
    cfg = runtime.configure(policy="fixed:b", strategy="speculative", segment_tokens=32)
    assert cfg.router.policy == "fixed:b" and cfg.relay.strategy == "speculative" and cfg.relay.segment_tokens == 32
    assert runtime.cfg.relay.strategy == "sequential"  # the base config is untouched
    assert runtime.configure() is runtime.cfg
    for bad in ({"policy": "fixed:zzz"}, {"strategy": "parallel"}, {"segment_tokens": 0}, {"policy": "best"}):
        with pytest.raises(ValueError):
            runtime.configure(**bad)
    relay = runtime.relay(cfg)
    assert isinstance(relay, Relay) and relay.cfg is cfg and isinstance(relay.router, FakeRouter)


async def test_run_to_trace_returns_the_committed_trace():
    relay, _, _ = make_relay(finishing_script())
    trace = await relay.run_to_trace("Who is the chancellor right now?", run_id="r")
    assert trace.answer == "The answer is 42."
    assert [s.phase for s in trace.segments] == ["think", "think", "think", "answer"]
    assert trace.started_at <= trace.ended_at


# --------------------------------------------------------------------------------------
# Integration with the real mock backend and router
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", ["sequential", "speculative"])
async def test_integration_mock_backend_and_router_without_index(strategy):
    from cutoffmix.backends.mock import MockBackend, default_mock_truths
    from cutoffmix.router.router import Router

    cfg = AppConfig(
        models=[ModelSpec(key=k, provider_id=f"x/{k}", supports_stop=k != "gpt-6.2-sol")
                for k in ("claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon")],
        relay=RelayConfig(strategy=strategy, speculative_gate=0.0), router=RouterConfig(seed=3), today="2026-11-24")
    backend = MockBackend(default_mock_truths(latency_s=0.0), seed=1, today="2026-11-24", segments_to_answer=4)
    router = Router(cfg.model_keys(), index=None, cfg=cfg.router, seed=3)
    events = [ev async for ev in Relay(cfg, backend, router).run("Wer ist aktuell Bundeskanzler von Deutschland?")]
    assert events[-1].type == "done"
    done = events[-1].data
    assert done["status"] == "ok" and done["answer"]
    think = [s for s in done["segments"] if s["phase"] == "think"]
    assert len(think) == 5 and all(s["model"] in cfg.model_keys() for s in think)
    for i, end in enumerate(committed(events)):
        if strategy == "sequential":
            assert "".join(deltas_by_segment(events)[i]) == end.text
        assert SENTINEL not in end.text and "FINAL ANSWER" not in end.text


async def test_all_empty_without_notes_or_next_to_other_failures_still_ends_with_an_error():
    # nothing usable for the very first segment: there are no notes to answer from
    relay, _, _ = make_relay(lambda model, req, n: [StreamEvent(kind="done", finish_reason="stop")])
    events = await collect(relay)
    assert events[-1].data["status"] == "error" and "no model left for segment 0" in events[-2].text

    # an error next to the empty outcomes is a failure, not a sign that the notes are complete
    def script(model, req, n):
        if not req.segments:
            return text_events(para(0))
        if model == "a":
            return [FlakyError("boom", retryable=False)]
        return [StreamEvent(kind="done", finish_reason="stop")]

    relay, _, _ = make_relay(script)
    events = await collect(relay)
    assert events[-1].data["status"] == "error" and "no model left for segment 1" in events[-2].text
    assert "a: error" in events[-2].text and "b: empty" in events[-2].text


# --------------------------------------------------------------------------------------
# The real OpenRouter backend behind httpx.MockTransport (no network)
# --------------------------------------------------------------------------------------

FABLE_KEY, SOL_KEY, NEON_KEY = "claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"
OR_USAGE = {"prompt_tokens": 1000, "completion_tokens": 400, "total_tokens": 1400, "cost": 0.004}


def or_stream(*pieces: str, refusal: str | None = None, finish: str = "stop") -> str:
    """An OpenRouter SSE body: one chunk per piece, the finish chunk, the usage chunk (always last), [DONE]."""

    def chunk(delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> str:
        out: dict[str, Any] = {"id": "gen-chunk", "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
        if usage is not None:
            out["usage"] = usage
        return "data: " + json.dumps(out) + "\n\n"

    body = ": OPENROUTER PROCESSING\n\n"
    if refusal is not None:
        body += chunk({"refusal": refusal})
    body += "".join(chunk({"content": p}) for p in pieces)
    body += chunk({"content": ""}, finish) + chunk({}, finish, OR_USAGE)
    return body + "data: [DONE]\n\n"


def real_relay(handler: Callable[[str, str, int], str], *, pick: Callable[[list[str], str | None, int], str],
               **relay: Any) -> tuple[Relay, list[dict]]:
    """A Relay over the real OpenRouterBackend; ``handler(model key, phase, notes so far)`` returns the SSE body.
    Each response carries ``X-Generation-Id: gen-<call number>``."""
    from cutoffmix.backends.openrouter import OpenRouterBackend
    from cutoffmix.config import load_config
    from cutoffmix.prompts import ANSWER_TAIL

    cfg = load_config(overrides={"today": "2026-11-24", "relay": relay})
    by_provider = {m.provider_id: m.key for m in cfg.models}
    bodies: list[dict] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        blocks = body["messages"][1]["content"]
        phase = "answer" if blocks[-1]["text"] == ANSWER_TAIL else "think"
        notes = 0 if blocks[1]["text"].startswith("(no notes") else len(blocks) - 2
        sse = handler(by_provider[body["model"]], phase, notes)
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream",
                                                                  "x-generation-id": f"gen-{len(bodies)}"})

    backend = OpenRouterBackend(api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    return Relay(cfg, backend, FakeRouter(cfg.model_keys(), pick=pick), retry_base_delay=0.0), bodies


async def test_real_backend_calls_closed_before_their_usage_chunk_are_reported():
    runaway = " trailing words that keep going past the sentinel"  # beyond the backend's grace window

    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("The answer is X.")
        text = "The notes agree on every point. FINAL ANSWER: X" if notes >= 2 else para(notes)
        return or_stream(text, SENTINEL, runaway)  # Sol has no `stop`: it writes the sentinel itself

    relay, bodies = real_relay(handler, pick=lambda avail, cur, n: SOL_KEY)
    done = (await collect(relay))[-1].data
    assert done["status"] == "ok" and done["answer"] == "The answer is X." and len(bodies) == 4
    # the think calls were closed before their usage chunk (twice at the sentinel, once at the final marker)
    assert done["usage_total"]["calls"] == 4 and done["usage_total"]["cost"] == pytest.approx(0.004)
    assert done["usage_complete"] is False
    # the relay closed the concluding call itself at the final marker: its id still arrived (on the first event)
    assert [(m["index"], m["model"], m["finish_reason"], m["generation_id"]) for m in done["usage_missing"]] == [
        (0, SOL_KEY, "sentinel", "gen-1"), (1, SOL_KEY, "sentinel", "gen-2"), (2, SOL_KEY, "marker", "gen-3")]
    segs = done["segments"]
    assert [s["aborted"] for s in segs] == [True, True, True, False]  # the backend's runaway close counts too
    assert segs[0]["usage"] == {"generation_id": "gen-1"} and segs[2]["usage"] == {"generation_id": "gen-3"}
    assert segs[2]["note"] == "marker"
    assert segs[3]["usage"] == OR_USAGE | {"generation_id": "gen-4"}


async def test_real_backend_sentinel_as_the_last_content_keeps_the_usage_complete():
    # the live Sol shape: the sentinel is its very last content, finish "stop" and the usage chunk follow at once
    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("The answer is X.")
        if notes >= 2:
            return or_stream("FINAL ANSWER: X")  # inline answer: read to the end, no close at the marker
        return or_stream(para(notes) + "\n", "<<", "END", "_STEP", ">>")

    relay, bodies = real_relay(handler, pick=lambda avail, cur, n: SOL_KEY, answer_mode="inline")
    events = await collect(relay)
    done = events[-1].data
    assert done["status"] == "ok" and len(bodies) == 3
    assert done["usage_complete"] is True and done["usage_missing"] == []
    assert done["usage_total"]["calls"] == 3 and done["usage_total"]["cost"] == pytest.approx(3 * 0.004)
    segs = done["segments"]
    assert [s["text"] for s in segs[:2]] == [para(0), para(1)]
    assert [s["note"] for s in segs[:2]] == ["sentinel", "sentinel"]  # the backend's finish_reason contract
    assert [s["usage"] for s in segs[:2]] == [OR_USAGE | {"generation_id": "gen-1"},
                                              OR_USAGE | {"generation_id": "gen-2"}]
    assert [s["aborted"] for s in segs] == [False, False, False, False]  # the provider finished every stream
    assert not any(SENTINEL in (ev.text or "") or "END_STEP" in (ev.text or "") for ev in events)


async def test_real_backend_complete_usage_when_every_stream_ends_normally():
    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("Answer.")
        return or_stream("FINAL ANSWER: x" if notes else para(0))  # Fable's `stop` ends at the sentinel

    relay, _ = real_relay(handler, pick=lambda avail, cur, n: FABLE_KEY, answer_mode="inline")
    done = (await collect(relay))[-1].data
    assert done["usage_complete"] is True and done["usage_missing"] == []
    assert done["usage_total"]["cost"] == pytest.approx(2 * 0.004)


async def test_real_backend_non_cancellable_overflow_is_read_through_for_its_usage():
    long = "This sentence keeps the paragraph going past the cut. " * 14  # 1.5x < length < 4x the hard cut

    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("Answer.")
        if notes:
            return or_stream("FINAL ANSWER: x", SENTINEL)
        return or_stream(*(long[i:i + 50] for i in range(0, len(long), 50)))

    relay, _ = real_relay(handler, pick=lambda avail, cur, n: NEON_KEY, hard_cut_chars=300)
    done = (await collect(relay))[-1].data
    seg = done["segments"][0]
    assert seg["model"] == NEON_KEY and 0 < len(seg["text"]) <= 300
    # closing would not have stopped Google's billing; reading on collected the usage chunk
    assert seg["note"] == "stop" and seg["aborted"] is False and seg["usage"]["cost"] == pytest.approx(0.004)


async def test_real_backend_openai_style_refusals_disable_the_model():
    def handler(key, phase, notes):
        if key == SOL_KEY:
            return or_stream(refusal="I can't help with that.")  # refusal text, no content, finish "stop"
        if phase == "answer":
            return or_stream("Answer.")
        return or_stream("FINAL ANSWER: x" if notes >= 2 else para(notes))

    relay, bodies = real_relay(handler, pick=lambda avail, cur, n: SOL_KEY if SOL_KEY in avail else avail[0])
    done = (await collect(relay))[-1].data
    assert done["status"] == "ok" and done["answer"] == "Answer."
    assert done["refusals"] == {SOL_KEY: 2} and done["disabled"] == [SOL_KEY]
    assert sum(b["model"] == "openai/gpt-6.2-sol" for b in bodies) == 2  # never billed again after that
    assert all(r["reason"] == "refused" for r in done["dropped"])


async def test_real_backend_all_empty_segment_concludes_the_notes():
    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("X.")
        if notes < 2:
            return or_stream(para(notes))
        # nothing left to add: Fable and Neon hit `stop` at once (no content), Sol writes only the sentinel
        return or_stream(SENTINEL) if key == SOL_KEY else or_stream()

    relay, _ = real_relay(handler, pick=lambda avail, cur, n: avail[0])
    events = await collect(relay)
    done = events[-1].data
    assert not any(ev.type == "error" for ev in events)
    assert done["status"] == "ok" and done["answer"] == "X." and "error" not in done
    assert [s["phase"] for s in done["segments"]] == ["think", "think", "answer"]
    assert [(r["model"], r["reason"]) for r in done["dropped"]] == [
        (FABLE_KEY, "empty"), (SOL_KEY, "empty"), (NEON_KEY, "empty")]
