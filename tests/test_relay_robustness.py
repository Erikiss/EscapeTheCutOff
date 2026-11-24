"""Regression tests for relay and backend robustness findings from the first live runs (no network).

Covers: speculative drafting only for score-based policies, generation ids of streams the relay closes itself,
wall-clock budgets per attempt, echoes of earlier notes, text after a mangled sentinel, paragraphs cut off by
``max_tokens`` and odd chunk shapes in the OpenRouter backend.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import test_openrouter as tor
from test_relay import (
    MODELS,
    SENTINEL,
    SOL_KEY,
    FakeBackend,
    FakeRouter,
    collect,
    committed,
    deltas_by_segment,
    make_relay,
    or_stream,
    para,
    text_events,
)

from cutoffmix.config import AppConfig, RelayConfig, RouterConfig
from cutoffmix.relay import Relay
from cutoffmix.types import StreamEvent

FABLE_KEY, NEON_KEY = "claude-fable-5.2", "gemini-4-neon"
LIVE = Path(__file__).parent / "fixtures" / "live_sse"
LIVE_NOTES = [s["text"] for s in json.loads((LIVE / "relay_trace_round_robin.json").read_text("utf-8"))["segments"][:4]]
NEW = "Neu: Artemis III ist für 2027 geplant."


def live_relay(handler, *, pick, **relay: Any) -> tuple[Relay, list[tuple[str, str, int]]]:
    """A Relay over the real OpenRouterBackend (httpx.MockTransport). ``handler(model key, phase, notes)`` returns an
    SSE body (str) or an ``httpx.AsyncByteStream``; every response carries ``X-Generation-Id: gen-<call number>``."""
    from cutoffmix.backends.openrouter import OpenRouterBackend
    from cutoffmix.config import load_config
    from cutoffmix.prompts import ANSWER_TAIL

    cfg = load_config(overrides={"today": "2026-11-24", "relay": relay})
    by_provider = {m.provider_id: m.key for m in cfg.models}
    calls: list[tuple[str, str, int]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        blocks = body["messages"][1]["content"]
        phase = "answer" if blocks[-1]["text"] == ANSWER_TAIL else "think"
        notes = [] if blocks[1]["text"].startswith("(no notes") else [b["text"].strip() for b in blocks[1:-1]]
        key = by_provider[body["model"]]
        calls.append((key, phase, len(notes)))
        out = handler(key, phase, notes)
        headers = {"content-type": "text/event-stream", "x-generation-id": f"gen-{len(calls)}"}
        if isinstance(out, str):
            return httpx.Response(200, content=out.encode(), headers=headers)
        return httpx.Response(200, stream=out, headers=headers)

    backend = OpenRouterBackend(api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    return Relay(cfg, backend, FakeRouter(cfg.model_keys(), pick=pick), retry_base_delay=0.0), calls


class KeepAlive(httpx.AsyncByteStream):
    """An SSE body that only ever sends OpenRouter's keep-alive comments (optionally after some data lines)."""

    def __init__(self, prefix: bytes = b"") -> None:
        self.prefix = prefix
        self.closed = False

    async def __aiter__(self):
        if self.prefix:
            yield self.prefix
        while True:
            yield b": OPENROUTER PROCESSING\n\n"
            await asyncio.sleep(0.01)

    async def aclose(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------------------
# Speculative drafting only where drafts can change the choice
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("policy", ["fixed:b", "round_robin", "random"])
async def test_speculative_strategy_does_not_draft_for_policies_that_ignore_the_drafts(policy):
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Answer.")
        return text_events(("FINAL ANSWER: x" if len(req.segments) >= 2 else para(len(req.segments))) + SENTINEL)

    backend = FakeBackend(script)
    cfg = AppConfig(models=list(MODELS), relay=RelayConfig(strategy="speculative", segment_tokens=64),
                    router=RouterConfig(policy=policy), today="2026-11-24")
    relay = Relay(cfg, backend, FakeRouter([m.key for m in MODELS], r=0.9), retry_base_delay=0.0)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok"
    assert backend.max_open == 1  # no parallel drafts: one call per segment, as with the sequential strategy
    assert len(backend.calls) == 3 + 1
    assert all(ev.decision != "speculative" for ev in events if ev.type == "segment_start")


# --------------------------------------------------------------------------------------
# Generation ids and finish reasons of streams the relay closes itself
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("model, body, finish", [
    (SOL_KEY, "All points are settled in the notes. FINAL ANSWER: X and more words", "marker"),
    (FABLE_KEY, "All points are settled in the notes. FINAL ANSWER: X and more words", "marker"),
    (SOL_KEY, para(0) + "<END_STEP> and then the model just keeps on writing", "mangled"),
    (SOL_KEY, "This sentence keeps the paragraph going past the cut. " * 30, "overflow"),
])
async def test_relay_side_closes_keep_the_generation_id_and_say_why(model, body, finish):
    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("The answer is X.")
        if notes:
            return or_stream("FINAL ANSWER: X", SENTINEL)
        return or_stream(*(body[i:i + 25] for i in range(0, len(body), 25)), " tail " * 50)

    relay, _ = live_relay(handler, pick=lambda avail, cur, n: model)
    events = await collect(relay)
    done = events[-1].data
    assert done["status"] == "ok"
    first = done["segments"][0]
    assert first["model"] == model and first["aborted"] is True
    assert first["note"] == finish and first["usage"] == {"generation_id": "gen-1"}
    assert done["usage_missing"][0] == {"index": 0, "model": model, "phase": "think", "attempt": 0,
                                        "finish_reason": finish, "generation_id": "gen-1"}
    end = committed(events)[0]
    assert end.data["finish_reason"] == finish and end.data["usage"] == {"generation_id": "gen-1"}


async def test_backend_reports_the_generation_id_on_its_first_event():
    data = tor.sse_bytes(tor.chunk(reasoning="Checking."), tor.chunk("The result "), tor.chunk("is 42."),
                         tor.chunk("", finish="stop"), tor.chunk(None, finish="stop", usage=tor.USAGE))
    rec = tor.Recorder(httpx.Response(200, stream=tor.ChunkStream([data]), headers={"X-Generation-Id": "gen-hdr-7"}))
    events = await tor.collect(tor.backend(rec), tor.FABLE, tor.req())
    assert [(e.kind, e.usage) for e in events[:3]] == [
        ("reasoning", {"generation_id": "gen-hdr-7"}), ("text", None), ("text", None)]
    assert events[-1].usage == tor.USAGE | {"generation_id": "gen-hdr-7"}
    # without the header, the chunks' id is reported on the first event
    data = tor.sse_bytes(tor.chunk("Text."), tor.chunk("", finish="stop"))
    events = await tor.collect(tor.backend(tor.Recorder(httpx.Response(200, stream=tor.ChunkStream([data])))),
                               tor.SOL, tor.req())
    assert events[0].kind == "text" and events[0].usage == {"generation_id": "gen-1"}


# --------------------------------------------------------------------------------------
# Wall-clock budget per attempt
# --------------------------------------------------------------------------------------


async def test_a_stream_of_keepalive_comments_times_out_and_is_rerouted():
    streams: list[KeepAlive] = []

    def handler(key, phase, notes):
        if key == SOL_KEY:
            streams.append(KeepAlive())
            return streams[-1]
        if phase == "answer":
            return or_stream("Answer.")
        return or_stream("FINAL ANSWER: X" if notes else para(0))

    relay, calls = live_relay(handler, pick=lambda avail, cur, n: SOL_KEY if SOL_KEY in avail else avail[0],
                              first_content_timeout_s=0.2, max_retries=1)
    events = await asyncio.wait_for(collect(relay), 10)
    done = events[-1].data
    assert done["status"] == "ok" and done["answer"] == "Answer."
    assert [c[0] for c in calls[:3]] == [SOL_KEY, SOL_KEY, FABLE_KEY]  # one retry, then re-routed
    assert [(d["model"], d["reason"]) for d in done["dropped"][:2]] == [(SOL_KEY, "error")] * 2
    assert "timeout: no text within 0.2 s" in done["dropped"][0]["error"]
    assert streams and all(s.closed for s in streams)
    # the provider may bill a call the relay gave up on: its usage is reported as unknown (Sol is excluded for one
    # segment only, so every segment and the answer try it again)
    timed_out = [m for m in done["usage_missing"] if m["model"] == SOL_KEY]
    assert len(timed_out) == sum(c[0] == SOL_KEY for c in calls) >= 2
    assert {m["finish_reason"] for m in timed_out} == {"timeout"} and done["usage_complete"] is False


async def test_a_trickling_stream_times_out_as_a_whole():
    def script(model, req, n):
        if model == "a":
            return (x for i in range(10_000) for x in (0.02, StreamEvent(kind="text", text=f"word{i} ")))
        if req.phase == "answer":
            return text_events("A.")
        return text_events(("FINAL ANSWER: x" if req.segments else para(0)) + SENTINEL)

    relay, backend, _ = make_relay(script, segment_timeout_s=0.3, max_retries=0)
    events = await asyncio.wait_for(collect(relay), 10)
    done = events[-1].data
    assert done["status"] == "ok" and committed(events)[0].model == "b"
    assert done["dropped"][0]["reason"] == "error"
    assert done["dropped"][0]["error"] == "timeout: the stream did not end within 0.3 s"
    assert backend.open == 0


async def test_a_hanging_draft_does_not_hold_the_speculative_segment():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Speculative answer.")
        if model == "c":
            return [3600.0]  # never sends anything
        return text_events(("FINAL ANSWER: x" if req.segments else f"Draft by {model}: the record says so.")
                           + SENTINEL)

    router = FakeRouter([m.key for m in MODELS], r=0.9, draft_scores={"a": 1.0, "b": 5.0, "c": 9.0})
    relay, backend, _ = make_relay(script, router=router, strategy="speculative", first_content_timeout_s=0.2)
    events = await asyncio.wait_for(collect(relay), 10)
    end = committed(events)[0]
    assert end.model == "b" and end.text == "Draft by b: the record says so."
    assert set(end.data["drafts"]) == {"a", "b"} and end.data["failed_drafts"] == {"c": "error"}
    assert events[-1].data["status"] == "ok" and backend.open == 0


async def test_a_timeout_after_the_sentinel_keeps_the_complete_paragraph():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if req.segments:
            return text_events("FINAL ANSWER: x" + SENTINEL)
        return [StreamEvent(kind="text", text=para(0) + SENTINEL), 3600.0]  # the usage chunk never comes

    relay, _, _ = make_relay(script, segment_timeout_s=0.3)
    events = await asyncio.wait_for(collect(relay), 10)
    done = events[-1].data
    assert done["status"] == "ok" and done["dropped"] == []
    seg = done["segments"][0]
    assert seg["text"] == para(0) and seg["note"] == "sentinel" and seg["aborted"] is True


def test_timeouts_are_validated():
    for bad in ({"segment_timeout_s": 0}, {"first_content_timeout_s": -1.0}):
        with pytest.raises(ValueError):
            RelayConfig(**bad)


# --------------------------------------------------------------------------------------
# Echoes of earlier notes
# --------------------------------------------------------------------------------------


def echo_handler(echoed: list[int], tail: str):
    """Sol writes the four live notes, then repeats notes ``echoed`` and continues with ``tail``."""

    def handler(key, phase, notes):
        if phase == "answer":
            return or_stream("Artemis II.")
        if len(notes) < 4:
            return or_stream(LIVE_NOTES[len(notes)], "\n" + SENTINEL)
        if len(notes) == 4 and key == SOL_KEY:
            body = "\n\n".join(notes[i] for i in echoed) + tail
            return or_stream(*(body[i:i + 40] for i in range(0, len(body), 40)))
        return or_stream("FINAL ANSWER: Artemis II.", SENTINEL)

    return handler


@pytest.mark.parametrize("echoed", [[2, 3], [0, 1, 2, 3], [2], [3]])
async def test_a_repeat_of_earlier_notes_is_neither_shown_nor_committed(echoed):
    relay, _ = live_relay(echo_handler(echoed, "\n\n" + NEW + "\n" + SENTINEL), pick=lambda avail, cur, n: SOL_KEY)
    events = await collect(relay)
    ends = committed(events)
    assert [e.text for e in ends[:4]] == LIVE_NOTES
    assert ends[4].text == NEW
    assert "".join(deltas_by_segment(events)[4]) == NEW


@pytest.mark.parametrize("echoed, tail", [
    ([2, 3], "\n" + SENTINEL),  # nothing new after the repeat
    ([0, 1, 2, 3, 0, 1, 2, 3], ""),  # a repeat that runs past the cut
])
async def test_an_attempt_that_only_repeats_the_notes_is_dropped_as_empty(echoed, tail):
    relay, _ = live_relay(echo_handler(echoed, tail),
                          pick=lambda avail, cur, n: SOL_KEY if SOL_KEY in avail else avail[0])
    events = await collect(relay)
    done = events[-1].data
    assert done["status"] == "ok"
    assert (SOL_KEY, "empty") in [(d["model"], d["reason"]) for d in done["dropped"]]
    think = [s["text"] for s in done["segments"] if s["phase"] == "think"]
    assert think[:4] == LIVE_NOTES and len(think) == len(set(think))  # no note is committed twice
    sol_deltas = [ev.text for ev in events if ev.type == "delta" and ev.segment == 4 and ev.model == SOL_KEY]
    assert sol_deltas == []


async def test_backend_cut_leaves_room_for_an_echo_of_the_notes():
    notes = ["A long earlier note that the model may repeat verbatim. " * 18]  # ~1000 chars
    text = notes[0] + " New result follows here. " * 20  # ~1500 chars: past 1.5 x 600, within that + the notes
    data = tor.sse_bytes(*(tor.chunk(text[i:i + 100]) for i in range(0, len(text), 100)), tor.chunk("", finish="stop"))
    r = tor.req(notes, hard_cut_chars=600)
    events = await tor.collect(tor.backend(tor.Recorder(httpx.Response(200, stream=tor.ChunkStream([data])))),
                               tor.SOL, r)
    assert tor.texts(events) == text and events[-1].finish_reason == "stop"


# --------------------------------------------------------------------------------------
# Mangled sentinels and cut-off paragraphs
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("answer_mode", ["inline", "routed"])
async def test_text_after_a_mangled_sentinel_is_not_an_answer(answer_mode):
    first = "Alar Karis ist seit dem 11. Oktober 2021 Präsident."

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Routed answer.")
        if not req.segments:  # model "c" is not cancellable: it is read to the end
            return text_events(first + "<END_STEP>FINAL ANSWER: Toomas Hendrik Ilves ist Präsident.")
        return text_events("FINAL ANSWER: Alar Karis.")

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: "c")
    relay, _, _ = make_relay(script, router=router, answer_mode=answer_mode)
    events = await collect(relay)
    ends = committed(events)
    assert ends[0].text == first and ends[0].data["final"] is False and ends[0].data["saw_sentinel"] is True
    assert len(ends) == 2  # the think phase went on to a second segment
    assert events[-1].data["answer"] == ("Alar Karis." if answer_mode == "inline" else "Routed answer.")
    assert not any("Toomas" in ev.text for ev in events if ev.type == "delta")


async def test_a_paragraph_cut_off_by_max_tokens_loses_its_half_sentence():
    fragment = "Alar Karis trat sein Amt am 11. Oktober 2021 an. Seine Amtszeit endet am 11. Okt"

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if req.segments:
            return text_events("FINAL ANSWER: x" + SENTINEL)
        return text_events(fragment, finish="length")

    relay, _, _ = make_relay(script)
    events = await collect(relay)
    assert committed(events)[0].text == "Alar Karis trat sein Amt am 11. Oktober 2021 an."
    assert events[-1].data["segments"][0]["text"] == "Alar Karis trat sein Amt am 11. Oktober 2021 an."


async def test_a_cut_off_paragraph_without_a_complete_sentence_is_retried_with_more_tokens():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if req.segments:
            return text_events("FINAL ANSWER: x" + SENTINEL)
        if n == 0:
            return text_events("Die Amtszeit von Alar Karis endet am", finish="length")
        return text_events(para(0) + SENTINEL)

    relay, backend, _ = make_relay(script)
    events = await collect(relay)
    assert committed(events)[0].text == para(0)
    assert backend.calls[1][1].max_tokens == 2 * (64 + 100)
    assert events[-1].data["dropped"][0]["reason"] == "length"


# --------------------------------------------------------------------------------------
# OpenRouter chunk shapes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("odd, text", [
    ({"choices": [None]}, ""),
    ({"choices": [{"index": 0, "delta": "oops", "finish_reason": None}]}, ""),
    ({"choices": "nope"}, ""),
    ({"choices": [{"index": 0, "delta": {"content": [{"type": "text", "text": "parts"}]}}]}, "parts"),
    ({"choices": [{"index": 0, "delta": {"content": 7, "reasoning": ["r"]}}]}, ""),
    ({"usage": [1, 2], "choices": []}, ""),
])
async def test_odd_chunk_shapes_never_raise_after_the_first_event(odd, text):
    data = tor.sse_bytes(tor.chunk("ok "), {"id": "gen-1", **odd}, tor.chunk("", finish="stop"))
    events = await tor.collect(tor.backend(tor.Recorder(httpx.Response(200, stream=tor.ChunkStream([data])))),
                               tor.SOL, tor.req())
    assert tor.texts(events) == "ok " + text
    assert [e.kind for e in events].count("done") == 1 and events[-1].finish_reason == "stop"
