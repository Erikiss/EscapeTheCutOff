"""Replays of the first live OpenRouter captures (``tests/fixtures/live_sse``) through the real backend and relay.

Each ``<model>.sse`` file holds the raw SSE bytes of one real think-segment call (request body, status and the
``X-Generation-Id`` header in ``<model>.meta.json``). The bytes are served through ``httpx.MockTransport`` in small
network chunks whose boundaries also fall inside UTF-8 characters and inside the sentinel. No network.
"""

from __future__ import annotations

import io
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from cutoffmix.backends.openrouter import OpenRouterBackend
from cutoffmix.cli import TerminalPrinter
from cutoffmix.config import AppConfig, load_config
from cutoffmix.prompts import clean_segment, strip_preamble
from cutoffmix.relay import Relay, RelayRuntime
from cutoffmix.types import ContinuationRequest, RelayEvent, StreamEvent

LIVE = Path(__file__).parent / "fixtures" / "live_sse"
FABLE, SOL, NEON = "claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"
MODELS = (FABLE, SOL, NEON)
S = "<<END_STEP>>"
TODAY = "2026-11-24"
QUESTION = "Wer ist aktuell Präsident von Estland?"  # the captured requests' PROBLEM
NOTE = "Die Frage betrifft das estnische Staatsoberhaupt zum 24. November 2026; die Amtszeit beträgt fünf Jahre."
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")


# --------------------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Capture:
    key: str
    raw: bytes
    meta: dict[str, Any]
    content: str  # all delta.content, in order
    reasoning: tuple[str, ...]  # the delta.reasoning pieces
    reasoning_details: tuple[str, ...]  # the text of reasoning_details entries
    finish: str | None  # the provider's finish_reason
    usage: dict[str, Any]

    @property
    def generation_id(self) -> str:
        return self.meta["headers"]["x-generation-id"]

    @property
    def body(self) -> dict[str, Any]:
        return self.meta["body"]

    @property
    def visible(self) -> str:
        """What the backend may emit: the content up to the end of the sentinel."""
        end = self.content.find(S)
        return self.content if end < 0 else self.content[: end + len(S)]

    @property
    def note(self) -> str:
        """The content as it should be committed to the notes."""
        return self.content.replace(S, "").strip()


def load_capture(key: str) -> Capture:
    raw = (LIVE / f"{key}.sse").read_bytes()
    meta = json.loads((LIVE / f"{key}.meta.json").read_text(encoding="utf-8"))
    content: list[str] = []
    reasoning: list[str] = []
    details: list[str] = []
    finish: str | None = None
    usage: dict[str, Any] = {}
    for line in raw.decode("utf-8").splitlines():
        if not line.startswith("data: {"):
            continue
        chunk = json.loads(line[len("data: "):])
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            if delta.get("reasoning"):
                reasoning.append(delta["reasoning"])
            details += [d["text"] for d in delta.get("reasoning_details") or [] if d.get("text")]
            finish = choice.get("finish_reason") or finish
    return Capture(key, raw, meta, "".join(content), tuple(reasoning), tuple(details), finish, usage)


CAPTURES = {key: load_capture(key) for key in MODELS}


def test_the_captures_hold_what_the_tests_rely_on():
    fable, sol, neon = (CAPTURES[k] for k in MODELS)
    assert all(c.meta["status"] == 200 and c.finish == "stop" and c.usage for c in CAPTURES.values())
    # Sol has no `stop`: it writes the sentinel itself as its very last content
    assert "stop" not in sol.body and sol.content.endswith("\n" + S)
    # Fable and Neon stop at the sentinel server-side
    assert fable.body["stop"] == neon.body["stop"] == [S] and S not in fable.content + neon.content
    # Fable's reasoning summary is streamed although the request excludes reasoning
    assert fable.body["reasoning"]["exclude"] is True and len("".join(fable.reasoning)) > 300
    assert "".join(fable.reasoning) == "".join(fable.reasoning_details)
    assert not sol.reasoning and not neon.reasoning


# --------------------------------------------------------------------------------------
# Network replay
# --------------------------------------------------------------------------------------


def network_chunks(raw: bytes, seed: int) -> list[bytes]:
    """Split ``raw`` the way a network delivers it: chunks of 1-96 bytes, plus cuts inside multi-byte UTF-8
    characters and inside every piece of the sentinel."""
    rng = random.Random(seed)
    cuts: set[int] = set()
    pos = 0
    while pos < len(raw):
        pos += rng.randint(1, 96)
        cuts.add(pos)
    leads = [i for i, b in enumerate(raw) if b >= 0xC0]  # first byte of a multi-byte character
    cuts.update(i + 1 for i in leads[:: max(1, len(leads) // 12)])
    for piece in (S.encode(), b'"<<"', b'"END"', b'"_STEP"', b'">>"'):
        start = raw.find(piece)
        while start >= 0:
            cuts.add(start + len(piece) // 2)
            start = raw.find(piece, start + 1)
    bounds = sorted(c for c in cuts if 0 < c < len(raw))
    return [raw[a:b] for a, b in zip([0, *bounds], [*bounds, len(raw)], strict=True)]


def splits_utf8(chunks: list[bytes]) -> bool:
    def decodes(c: bytes) -> bool:
        try:
            c.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    return not all(decodes(c) for c in chunks)


class Replay(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.sent = 0
        self.closed = False

    async def __aiter__(self):
        for c in self.chunks:
            self.sent += 1
            yield c

    async def aclose(self) -> None:
        self.closed = True


class FixtureTransport:
    """MockTransport handler: answers every request with the capture of the requested model. ``truncate`` maps a
    call number (1-based) to the byte offset at which that response's connection drops."""

    def __init__(self, truncate: dict[int, int] | None = None) -> None:
        self.by_provider = {c.body["model"]: c for c in CAPTURES.values()}
        self.truncate = truncate or {}
        self.bodies: list[dict[str, Any]] = []
        self.streams: list[Replay] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        capture = self.by_provider[body["model"]]
        raw = capture.raw[: self.truncate.get(len(self.bodies), len(capture.raw))]
        stream = Replay(network_chunks(raw, seed=len(self.bodies)))
        self.streams.append(stream)
        return httpx.Response(capture.meta["status"], headers=capture.meta["headers"], stream=stream)


def fixture_backend(truncate: dict[int, int] | None = None) -> tuple[OpenRouterBackend, FixtureTransport,
                                                                      httpx.AsyncClient]:
    transport = FixtureTransport(truncate)
    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return OpenRouterBackend(api_key="test-key", client=client), transport, client


def relay_cfg(max_segments: int = 3, **router: Any) -> AppConfig:
    relay = {"strategy": "sequential", "max_segments": max_segments, "answer_mode": "routed"}
    return load_config(overrides={"today": TODAY, "relay": relay, "router": router})


# --------------------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key, finish", [(FABLE, "stop"), (SOL, "sentinel"), (NEON, "stop")])
async def test_backend_replays_the_live_capture(key, finish):
    capture = CAPTURES[key]
    backend, transport, client = fixture_backend()
    cfg = load_config(overrides={"today": TODAY})
    req = ContinuationRequest(question=QUESTION, segments=[NOTE], today=TODAY, run_id="capture1")
    try:
        events: list[StreamEvent] = [e async for e in backend.stream_continuation(cfg.model(key), req)]
    finally:
        await client.aclose()

    sent = transport.bodies[0]
    assert sent["model"] == capture.body["model"] and sent.get("stop") == capture.body.get("stop")
    assert sent["reasoning"]["exclude"] is True
    stream = transport.streams[0]
    assert splits_utf8(stream.chunks)  # the replay really cuts inside multi-byte characters
    assert stream.closed

    kinds = [e.kind for e in events]
    assert kinds.count("done") == 1 and kinds[-1] == "done" and "error" not in kinds
    text = "".join(e.text for e in events if e.kind == "text")
    assert text == capture.visible
    assert text.replace(S, "") == capture.content.replace(S, "")
    assert clean_segment(text, NOTE, sentinel=S, hard_cut_chars=600) == (capture.note, S in capture.content)
    done = events[-1]
    assert done.finish_reason == finish and done.error is None
    assert done.usage is not None and all(k in done.usage for k in USAGE_KEYS)
    assert done.usage["cost"] == capture.usage["cost"]
    assert done.usage == capture.usage | {"generation_id": capture.generation_id}
    assert "".join(e.text for e in events if e.kind == "reasoning") == "".join(capture.reasoning)


def test_the_sol_replay_cuts_inside_the_sentinel():
    chunks = network_chunks(CAPTURES[SOL].raw, seed=1)
    joined = b"".join(chunks)
    ends = {sum(len(c) for c in chunks[: i + 1]) for i in range(len(chunks))}
    for piece in (b'"<<"', b'"_STEP"'):
        start = joined.find(piece)
        assert any(start < end < start + len(piece) for end in ends)


# --------------------------------------------------------------------------------------
# Relay
# --------------------------------------------------------------------------------------


async def run_relay(cfg: AppConfig) -> tuple[list[RelayEvent], FixtureTransport]:
    backend, transport, client = fixture_backend()
    runtime = RelayRuntime(cfg, backend=backend)
    try:
        events = [ev async for ev in runtime.relay().run(QUESTION, run_id="capture1")]
    finally:
        await client.aclose()
    return events, transport


async def test_round_robin_relay_over_the_three_captures():
    events, transport = await run_relay(relay_cfg(policy="round_robin"))
    done = events[-1].data
    assert events[-1].type == "done" and not any(ev.type == "error" for ev in events)
    assert done["status"] == "ok" and "error" not in done and done["dropped"] == []
    segs = done["segments"]
    assert [(s["model"], s["phase"]) for s in segs] == [
        (FABLE, "think"), (SOL, "think"), (NEON, "think"), (FABLE, "answer")]
    assert len(transport.bodies) == 4 and all(s.closed for s in transport.streams)

    # every stream ended normally or at the sentinel with its usage chunk read: the cost is complete
    assert done["usage_complete"] is True and done["usage_missing"] == []
    assert [s["note"] for s in segs] == ["stop", "sentinel", "stop", "stop"]
    assert [s["aborted"] for s in segs] == [False] * 4
    for s in segs:
        capture = CAPTURES[s["model"]]
        assert s["usage"] == capture.usage | {"generation_id": capture.generation_id}
    expected_cost = sum(CAPTURES[s["model"]].usage["cost"] for s in segs)
    assert done["usage_total"]["cost"] == pytest.approx(expected_cost)
    assert done["usage_total"]["calls"] == 4

    # the notes are the captured paragraphs, without sentinel or preamble, and the stream agrees with them
    notes = [s["text"] for s in segs if s["phase"] == "think"]
    assert notes == [CAPTURES[k].note for k in MODELS]
    for note in notes:
        assert S not in note and "END_STEP" not in note and "<<" not in note
        assert strip_preamble(note) == note
    for ev in events:
        assert "END_STEP" not in (ev.text or "") and "<<" not in (ev.text or "")
    for end in (ev for ev in events if ev.type == "segment_end"):
        streamed = "".join(ev.text for ev in events if ev.type == "delta" and ev.segment == end.segment)
        assert streamed == end.text
    assert done["answer"] == CAPTURES[FABLE].note


# --------------------------------------------------------------------------------------
# Hidden reasoning never leaks
# --------------------------------------------------------------------------------------


def reasoning_fragments(capture: Capture) -> list[str]:
    """Multi-word pieces of the capture's hidden reasoning (stream deltas and clauses) that are not part of its
    visible text; single words such as "election" also occur in routing metadata (topic names)."""
    full = "".join(capture.reasoning)
    clauses = [c.strip() for c in re.split(r"(?<=[.,;])\s+|\n+", full)]
    pieces = {p.strip() for p in (*capture.reasoning, *capture.reasoning_details, *clauses)}
    return sorted(p for p in pieces if len(p) >= 10 and " " in p and p not in capture.content)


def assert_no_reasoning(haystack: str, capture: Capture) -> None:
    fragments = reasoning_fragments(capture)
    assert len(fragments) >= 20  # the check has teeth
    leaked = [f for f in fragments if f in haystack]
    assert leaked == []


def printed(cfg: AppConfig, events: list[RelayEvent], *, live: bool) -> str:
    out, err = io.StringIO(), io.StringIO()
    printer = TerminalPrinter(cfg.models, out=out, err=err, color=False, live=live, width=100)
    for ev in events:
        printer.handle(ev)
    return out.getvalue() + err.getvalue()


async def test_fable_reasoning_never_reaches_notes_trace_events_or_cli():
    fable = CAPTURES[FABLE]
    cfg = relay_cfg(1, policy=f"fixed:{FABLE}")  # one think segment, then the answer: both replay Fable
    events, transport = await run_relay(cfg)
    done = events[-1].data
    assert done["status"] == "ok"
    assert {b["model"] for b in transport.bodies} == {fable.body["model"]}  # every call replayed Fable's capture
    assert [s["text"] for s in done["segments"]] == [fable.note] * len(done["segments"])
    dumped = json.dumps([ev.to_dict() for ev in events], ensure_ascii=False)
    assert fable.note in dumped
    assert_no_reasoning(dumped, fable)
    assert_no_reasoning(json.dumps(done, ensure_ascii=False), fable)
    for live in (False, True):
        output = printed(cfg, events, live=live)
        assert fable.note[:60] in output
        assert_no_reasoning(output, fable)


async def test_a_connection_drop_inside_a_reasoning_chunk_does_not_leak_it():
    fable = CAPTURES[FABLE]
    cut = fable.raw.index(b'"reasoning":"alling that Alar') + 26  # the first response ends inside this delta
    cfg = relay_cfg(1, policy=f"fixed:{FABLE}")
    backend, transport, client = fixture_backend(truncate={1: cut})
    relay = Relay(cfg, backend, RelayRuntime(cfg, backend=backend).router_factory(cfg), retry_base_delay=0.0)
    try:
        events = [ev async for ev in relay.run(QUESTION, run_id="capture1")]
    finally:
        await client.aclose()
    done = events[-1].data
    assert done["status"] == "ok" and len(transport.bodies) == 3  # the failed call is retried, then the answer
    assert [(d["model"], d["reason"]) for d in done["dropped"]] == [(FABLE, "error")]
    assert "malformed SSE data payload" in done["dropped"][0]["error"]
    assert_no_reasoning(json.dumps([ev.to_dict() for ev in events], ensure_ascii=False), fable)


def sse_data(text: str) -> list[Any]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


async def test_fable_reasoning_never_reaches_the_server_streams():
    pytest.importorskip("fastapi")
    from cutoffmix.server import create_app

    fable = CAPTURES[FABLE]
    backend, transport, client = fixture_backend()
    app = create_app(relay_cfg(1, policy=f"fixed:{FABLE}"), backend=backend, allowed_hosts=("test",))
    question = {"role": "user", "content": QUESTION}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
            relay = await c.post("/api/relay", json={"question": QUESTION})
            stream = await c.post("/v1/chat/completions", json={"model": f"cutoffmix-fixed-{FABLE}",
                                                                "messages": [question], "stream": True})
            plain = await c.post("/v1/chat/completions", json={"model": f"cutoffmix-fixed-{FABLE}",
                                                               "messages": [question]})
    finally:
        await client.aclose()
    assert relay.status_code == stream.status_code == plain.status_code == 200
    assert transport.bodies and {b["model"] for b in transport.bodies} == {fable.body["model"]}

    rich = sse_data(relay.text)
    assert rich[-1]["type"] == "done" and rich[-1]["data"]["status"] == "ok"
    assert_no_reasoning(relay.text, fable)

    chunks = sse_data(stream.text)
    notes = "".join(ch["choices"][0]["delta"].get("reasoning") or "" for ch in chunks if ch.get("choices"))
    answer = "".join(ch["choices"][0]["delta"].get("content") or "" for ch in chunks if ch.get("choices"))
    assert notes.startswith(fable.note)  # delta.reasoning carries the shared notes, not hidden reasoning
    assert answer == fable.note
    assert_no_reasoning(stream.text, fable)

    message = plain.json()["choices"][0]["message"]
    assert message["content"] == fable.note
    assert_no_reasoning(plain.text, fable)
