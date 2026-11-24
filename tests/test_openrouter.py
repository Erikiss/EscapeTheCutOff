"""Tests for cutoffmix.backends.openrouter, using httpx.MockTransport (no network)."""

from __future__ import annotations

import asyncio
import email.utils
import json
import time
from collections.abc import Callable

import httpx
import pytest

from cutoffmix.backends import openrouter as orb
from cutoffmix.backends.base import BackendError
from cutoffmix.backends.openrouter import OpenRouterBackend, parse_sse
from cutoffmix.config import default_config
from cutoffmix.prompts import ANSWER_TAIL, RELAY_TAIL, answer_system_prompt, relay_system_prompt
from cutoffmix.types import ChatRequest, ContinuationRequest, ModelSpec, StreamEvent

CFG = default_config()
FABLE, SOL, NEON = (CFG.model(k) for k in ("claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"))
S = "<<END_STEP>>"


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def chunk(content: str | None = None, *, reasoning: str | None = None, refusal: str | None = None,
          finish: str | None = None, usage: dict | None = None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    if refusal is not None:
        delta["refusal"] = refusal
    out = {"id": "gen-1", "object": "chat.completion.chunk", "provider": "Test",
           "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage is not None:
        out["usage"] = usage
    return out


USAGE = {"prompt_tokens": 50, "completion_tokens": 20, "total_tokens": 70,
         "completion_tokens_details": {"reasoning_tokens": 12}}


def sse_bytes(*events: dict | str, done: bool = True) -> bytes:
    out = ": OPENROUTER PROCESSING\n\n"
    for e in events:
        out += "data: " + (e if isinstance(e, str) else json.dumps(e, ensure_ascii=False)) + "\n\n"
    if done:
        out += "data: [DONE]\n\n"
    return out.encode("utf-8")


def split_every(data: bytes, n: int) -> list[bytes]:
    """Network chunks at fixed byte offsets: boundaries fall mid-line and even inside UTF-8 characters."""
    return [data[i: i + n] for i in range(0, len(data), n)]


class ChunkStream(httpx.AsyncByteStream):
    """Response body that records how far it was read and whether it was closed; optionally never ends."""

    def __init__(self, chunks: list[bytes], *, hang_after: bool = False) -> None:
        self.chunks = chunks
        self.hang_after = hang_after
        self.sent = 0
        self.closed = False

    async def __aiter__(self):
        for c in self.chunks:
            self.sent += 1
            yield c
        if self.hang_after:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


class Recorder:
    """MockTransport handler that records requests and replays canned responses in order."""

    def __init__(self, *responses: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        r = self.responses[min(len(self.requests), len(self.responses)) - 1]
        return r(request) if callable(r) else r

    def body(self, i: int = -1) -> dict:
        return json.loads(self.requests[i].content)


def backend(handler, **kw) -> OpenRouterBackend:
    kw.setdefault("api_key", "test-key")
    return OpenRouterBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), **kw)


def req(segments: list[str] | None = None, **kw) -> ContinuationRequest:
    return ContinuationRequest(question="Who leads the lab?", segments=list(segments or []), today="2026-11-24",
                               run_id="run42", **kw)


async def collect(b: OpenRouterBackend, model: ModelSpec, r: ContinuationRequest) -> list[StreamEvent]:
    return [e async for e in b.stream_continuation(model, r)]


def texts(events: list[StreamEvent]) -> str:
    return "".join(e.text for e in events if e.kind == "text")


def has_cache_control(obj) -> bool:
    return "cache_control" in json.dumps(obj)


# --------------------------------------------------------------------------------------
# parse_sse
# --------------------------------------------------------------------------------------


def test_parse_sse_skips_comments_and_stops_at_done():
    lines = [": OPENROUTER PROCESSING", "", "data: {\"a\": 1}", "", ": keep-alive", "", "event: x", "data:{\"b\": 2}",
             "", "data: [DONE]", "", "data: {\"never\": true}", ""]
    assert list(parse_sse(lines)) == [{"a": 1}, {"b": 2}]


def test_parse_sse_multiline_data_crlf_and_unterminated_last_event():
    lines = ["data: {\"text\":\r\n", "data: \"x\"}\r\n", "\r\n", "data: {\"last\": 1}"]
    assert list(parse_sse(lines)) == [{"text": "x"}, {"last": 1}]


def test_parse_sse_malformed_payload():
    with pytest.raises(ValueError, match="malformed SSE"):
        list(parse_sse(["data: {not json", ""]))
    with pytest.raises(ValueError, match="not a JSON object"):
        list(parse_sse(["data: [1, 2]", ""]))


# --------------------------------------------------------------------------------------
# request bodies
# --------------------------------------------------------------------------------------


def test_fable_body():
    b = OpenRouterBackend(api_key="k")
    body = b.build_continuation_body(FABLE, req(["s1", "s2"], metadata={"temperature": 0.3}))
    assert body["model"] == "anthropic/claude-fable-5.2" and body["stream"] is True
    assert body["max_tokens"] == 128 + 2048
    assert body["stop"] == [S]
    assert body["reasoning"] == {"effort": "low", "exclude": True}
    assert body["provider"] == {"only": ["anthropic", "azure"]}
    for absent in ("temperature", "verbosity", "prompt_cache_key", "session_id"):
        assert absent not in body
    system, user = body["messages"]
    assert system["role"] == "system" and user["role"] == "user"
    assert system["content"] == [{"type": "text", "text": relay_system_prompt(128, "2026-11-24", S, "FINAL ANSWER:"),
                                  "cache_control": {"type": "ephemeral"}}]
    blocks = user["content"]
    assert [b.get("cache_control") for b in blocks] == [None, None, {"type": "ephemeral"}, None]
    assert blocks[2]["text"] == "s2\n\n" and blocks[-1]["text"] == RELAY_TAIL


def test_fable_body_without_segments_and_answer_phase():
    b = OpenRouterBackend(api_key="k")
    first = b.build_continuation_body(FABLE, req([]))
    assert "cache_control" in first["messages"][0]["content"][0]
    assert not has_cache_control(first["messages"][1])
    answer = b.build_continuation_body(FABLE, req(["s1"], phase="answer", max_tokens=3000))
    assert "stop" not in answer and not has_cache_control(answer)
    assert answer["max_tokens"] == 3000
    assert answer["messages"][0]["content"][0]["text"] == answer_system_prompt("2026-11-24")
    assert answer["messages"][1]["content"][-1]["text"] == ANSWER_TAIL


def test_sol_body():
    body = OpenRouterBackend(api_key="k").build_continuation_body(SOL, req(["s1"], metadata={"temperature": 0.3}))
    assert body["model"] == "openai/gpt-6.2-sol"
    assert "stop" not in body and "temperature" not in body
    assert body["prompt_cache_key"] == body["session_id"] == "run42:gpt-6.2-sol"
    assert body["verbosity"] == "low"
    assert body["provider"] == {"only": ["openai", "azure"]}
    assert not has_cache_control(body)
    no_run = OpenRouterBackend(api_key="k").build_continuation_body(SOL, ContinuationRequest(question="Q"))
    assert "prompt_cache_key" not in no_run and "session_id" not in no_run


def test_neon_body():
    b = OpenRouterBackend(api_key="k")
    body = b.build_continuation_body(NEON, req(["s1"]))
    assert body["model"] == "google/gemini-4-neon"
    assert body["stop"] == [S]
    assert body["provider"] == {"only": ["google-vertex"]}
    assert body["max_tokens"] == 128 + 1536
    assert "temperature" not in body and "prompt_cache_key" not in body and not has_cache_control(body)
    assert b.build_continuation_body(NEON, req(["s1"], metadata={"temperature": 0.2}))["temperature"] == 0.2


def test_continuation_bodies_are_append_only():
    b = OpenRouterBackend(api_key="k")

    def strip(body):
        msgs = json.loads(json.dumps(body["messages"]))
        for m in msgs:
            for blk in m["content"]:
                blk.pop("cache_control", None)
        return msgs

    for model in (FABLE, SOL, NEON):
        m2 = strip(b.build_continuation_body(model, req(["s1", "s2"])))
        m3 = strip(b.build_continuation_body(model, req(["s1", "s2", "s3"])))
        assert m3[0] == m2[0]
        assert m3[1]["content"][: len(m2[1]["content"]) - 1] == m2[1]["content"][:-1]


def test_extra_body_merge_and_verbosity_guard():
    claude = ModelSpec(key="c", provider_id="anthropic/x", provider_routing={"only": ["anthropic"]},
                       extra_body={"verbosity": "low", "provider": {"sort": "price"}})
    body = OpenRouterBackend(api_key="k").build_continuation_body(claude, req())
    assert "verbosity" not in body
    assert body["provider"] == {"only": ["anthropic"], "sort": "price"}


def test_chat_bodies():
    b = OpenRouterBackend(api_key="k")
    msgs = [{"role": "user", "content": "Q?"}]
    neon = b.build_chat_body(NEON, ChatRequest(messages=msgs, max_tokens=4096, response_format={"type": "json_object"}))
    assert neon["stream"] is False and neon["temperature"] == 0.0 and neon["max_tokens"] == 4096
    assert neon["response_format"] == {"type": "json_object"} and neon["messages"] == msgs
    assert "stop" not in neon
    fable = b.build_chat_body(FABLE, ChatRequest(messages=msgs, reasoning={"effort": "medium"}))
    assert "temperature" not in fable and fable["reasoning"] == {"effort": "medium"}
    assert b.build_chat_body(SOL, ChatRequest(messages=msgs))["verbosity"] == "low"


# --------------------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------------------


async def test_stream_normal_with_split_chunks_and_usage_chunk():
    data = sse_bytes(
        chunk(reasoning="Let me check."),
        chunk("Die Antwort ist "),
        chunk("Äpfel — 42."),
        chunk("", finish="stop"),
        chunk(None, finish="stop", usage=USAGE),  # usage chunk repeats finish_reason
    )
    stream = ChunkStream(split_every(data, 7))
    rec = Recorder(httpx.Response(200, stream=stream))
    events = await collect(backend(rec, referer="https://example.org"), FABLE, req(["s1"]))
    assert texts(events) == "Die Antwort ist Äpfel — 42."
    assert [e.text for e in events if e.kind == "reasoning"] == ["Let me check."]
    assert [e.kind for e in events].count("done") == 1 and events[-1].kind == "done"
    assert events[-1].finish_reason == "stop" and events[-1].text == ""
    assert events[-1].usage == USAGE | {"generation_id": "gen-1"}  # the chunks' id: no X-Generation-Id header here
    assert stream.closed and stream.sent == len(stream.chunks)
    request = rec.requests[0]
    assert request.url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    assert request.headers["x-title"] == "cutoffmix"
    assert request.headers["http-referer"] == "https://example.org"
    assert rec.body() == OpenRouterBackend(api_key="k").build_continuation_body(FABLE, req(["s1"]))


async def test_no_referer_header_by_default():
    rec = Recorder(httpx.Response(200, stream=ChunkStream([sse_bytes(chunk("x", finish="stop"))])))
    await collect(backend(rec), FABLE, req())
    assert "http-referer" not in rec.requests[0].headers


async def test_sentinel_mid_delta_closes_stream_promptly():
    data = sse_bytes(chunk("The result is 42."), chunk(" Done<<END_"), chunk("STEP>> leaked words"),
                     chunk("more leaked words"), done=False)
    stream = ChunkStream(split_every(data, 11), hang_after=True)  # never ends unless we stop reading
    events = await asyncio.wait_for(collect(backend(Recorder(httpx.Response(200, stream=stream))), SOL, req()), 5)
    assert texts(events) == "The result is 42. Done<<END_STEP>>"
    assert events[-1].kind == "done" and events[-1].finish_reason == "sentinel"
    assert events[-1].usage == {"generation_id": "gen-1"}  # no usage chunk, but the generation can be looked up
    assert stream.closed and stream.sent < len(stream.chunks)


async def test_hard_cut_closes_stream():
    data = sse_bytes(*[chunk("x" * 100) for _ in range(10)], done=False)
    stream = ChunkStream(split_every(data, 64), hang_after=True)
    r = req(hard_cut_chars=100)
    events = await asyncio.wait_for(collect(backend(Recorder(httpx.Response(200, stream=stream))), SOL, r), 5)
    assert len(texts(events)) == 200  # first delta beyond 1.5 * hard_cut_chars is still emitted, then we stop
    assert events[-1].finish_reason == "cut" and stream.closed


async def test_answer_phase_is_not_hard_cut():
    data = sse_bytes(*[chunk("y" * 100) for _ in range(5)], chunk("", finish="stop"))
    r = req(["s1"], hard_cut_chars=100, phase="answer")
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, r)
    assert len(texts(events)) == 500 and events[-1].finish_reason == "stop"


async def test_refusal_stream():
    data = sse_bytes(chunk(refusal="I can't help with that."), chunk("", finish="content_filter"),
                     chunk(None, finish="content_filter", usage=USAGE))
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), FABLE, req())
    assert texts(events) == ""
    done = events[-1]
    assert done.kind == "done" and done.finish_reason == "content_filter" and done.usage["prompt_tokens"] == 50
    assert "I can't help with that." in done.error


async def test_generation_id_header_is_reported_when_the_usage_chunk_is_cut_off():
    # Sol writes the sentinel itself; the usage chunk would only follow it, so it is never read
    data = sse_bytes(chunk("The result is 42."), chunk(S), chunk(" more"), chunk("", finish="stop"),
                     chunk(None, finish="stop", usage=USAGE))
    stream = ChunkStream(split_every(data, 16))
    rec = Recorder(httpx.Response(200, stream=stream, headers={"X-Generation-Id": "gen-hdr-7"}))
    events = await collect(backend(rec), SOL, req())
    done = events[-1]
    assert done.finish_reason == "sentinel" and stream.closed and stream.sent < len(stream.chunks)
    assert done.usage == {"generation_id": "gen-hdr-7"}  # the header wins over the chunk id


async def test_non_cancellable_model_is_not_cut_and_keeps_its_usage():
    data = sse_bytes(*[chunk("x" * 100) for _ in range(5)], chunk("", finish="stop"),
                     chunk(None, finish="stop", usage=USAGE))
    r = req(hard_cut_chars=100)
    assert not NEON.cancellable and SOL.cancellable
    neon = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), NEON, r)
    assert len(texts(neon)) == 500  # closing would not stop Google's billing: read on, the relay bounds it
    assert neon[-1].finish_reason == "stop" and neon[-1].usage["prompt_tokens"] == 50
    sol = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, r)
    assert sol[-1].finish_reason == "cut" and "prompt_tokens" not in (sol[-1].usage or {})


async def test_openai_style_refusal_counts_as_content_filter():
    # OpenAI's native shape: refusal text, no content, finish_reason "stop"
    data = sse_bytes(chunk(refusal="I can't help with "), chunk(refusal="that."), chunk("", finish="stop"),
                     chunk(None, finish="stop", usage=USAGE))
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, req())
    done = events[-1]
    assert texts(events) == "" and done.finish_reason == "content_filter"
    assert done.error == "refused: I can't help with that."
    # refusal text next to real content is not a refusal of the segment
    data = sse_bytes(chunk(refusal="partly"), chunk("A usable note."), chunk("", finish="stop"))
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, req())
    assert events[-1].finish_reason == "stop" and events[-1].error is None


async def test_error_chunk_mid_stream():
    err = {"id": "gen-2", "object": "chat.completion.chunk", "provider": "OpenAI",
           "error": {"code": 502, "message": "Provider disconnected",
                     "metadata": {"error_type": "provider_unavailable", "provider_code": "x"}},
           "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]}
    data = sse_bytes(chunk("partial "), err)
    stream = ChunkStream(split_every(data, 13))
    events = await collect(backend(Recorder(httpx.Response(200, stream=stream))), SOL, req())
    assert [e.kind for e in events] == ["text", "error", "done"]
    assert "provider_unavailable" in events[1].error and "Provider disconnected" in events[1].error
    assert events[1].finish_reason == "error" and events[-1].finish_reason == "error"
    assert stream.closed


async def test_length_with_empty_content():
    data = sse_bytes(chunk(reasoning="hmm"), chunk("", finish="length"), chunk(None, finish="length", usage=USAGE))
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), NEON, req())
    assert texts(events) == "" and [e.kind for e in events] == ["reasoning", "done"]
    assert events[-1].finish_reason == "length" and events[-1].usage == USAGE | {"generation_id": "gen-1"}


async def test_premature_eof_reports_error():
    data = sse_bytes(chunk("half a sent"), done=False)
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, req())
    assert [e.kind for e in events] == ["text", "error", "done"] and events[-1].finish_reason == "error"


async def test_malformed_chunk_mid_stream_does_not_raise():
    data = sse_bytes(chunk("ok "), "{broken json")
    events = await collect(backend(Recorder(httpx.Response(200, stream=ChunkStream([data])))), SOL, req())
    assert [e.kind for e in events] == ["text", "error", "done"] and "malformed" in events[1].error


async def test_consumer_leaving_early_closes_response():
    data = sse_bytes(*[chunk(f"part {i}. ") for i in range(20)], done=False)
    stream = ChunkStream(split_every(data, 32), hang_after=True)
    gen = backend(Recorder(httpx.Response(200, stream=stream))).stream_continuation(SOL, req())
    first = await gen.__anext__()
    assert first.kind == "text"
    await gen.aclose()
    assert stream.closed and stream.sent < len(stream.chunks)


async def test_http_error_raises_before_first_event():
    body = {"error": {"code": 429, "message": "Rate limit exceeded", "metadata": {"error_type": "rate_limit_exceeded"}}}
    rec = Recorder(httpx.Response(429, json=body, headers={"Retry-After": "7"}))
    with pytest.raises(BackendError) as ei:
        await collect(backend(rec), SOL, req())
    e = ei.value
    assert (e.status, e.error_type, e.retryable, e.retry_after) == (429, "rate_limit_exceeded", True, 7.0)
    assert "Rate limit exceeded" in str(e)
    rec400 = Recorder(httpx.Response(400, json={"error": {"code": 400, "message": "prefill not supported"}}))
    with pytest.raises(BackendError) as ei:
        await collect(backend(rec400), FABLE, req())
    assert ei.value.status == 400 and not ei.value.retryable


async def test_transport_error_raises_retryable_backend_error():
    def boom(request):
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(BackendError) as ei:
        await collect(backend(boom), SOL, req())
    assert ei.value.retryable and ei.value.error_type == "transport"


async def test_missing_api_key_raises_at_first_request(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    rec = Recorder(httpx.Response(200, stream=ChunkStream([b""])))
    b = OpenRouterBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(rec)))  # constructing is fine
    with pytest.raises(BackendError, match="OPENROUTER_API_KEY") as ei:
        await collect(b, SOL, req())
    assert ei.value.error_type == "missing_api_key" and rec.requests == []
    monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
    rec.responses = [httpx.Response(200, stream=ChunkStream([sse_bytes(chunk("x", finish="stop"))]))]
    await collect(b, SOL, req())
    assert rec.requests[0].headers["authorization"] == "Bearer env-key"


# --------------------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------------------


def chat_payload(content="Paris.", finish="stop") -> dict:
    return {"id": "gen-3", "choices": [{"index": 0, "finish_reason": finish,
                                        "message": {"role": "assistant", "content": content, "reasoning": "r"}}],
            "usage": USAGE}


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(orb.asyncio, "sleep", fake_sleep)
    return recorded


async def test_chat_success():
    rec = Recorder(httpx.Response(200, json=chat_payload()))
    res = await backend(rec).chat(NEON, ChatRequest(messages=[{"role": "user", "content": "Capital of France?"}]))
    assert (res.text, res.reasoning, res.finish_reason, res.usage) == ("Paris.", "r", "stop", USAGE)
    assert res.raw["id"] == "gen-3" and res.latency_s >= 0
    assert rec.body()["stream"] is False and rec.body()["temperature"] == 0.0


async def test_chat_retries_with_retry_after_then_succeeds(sleeps):
    rec = Recorder(
        httpx.Response(429, json={"error": {"code": 429, "message": "slow down"}}, headers={"Retry-After": "3"}),
        httpx.Response(503, json={"error": {"code": 503, "message": "overloaded"}}),
        httpx.Response(200, json=chat_payload("ok")),
    )
    res = await backend(rec, retry_base_delay=0.5).chat(SOL, ChatRequest(messages=[{"role": "user", "content": "Q"}]))
    assert res.text == "ok" and len(rec.requests) == 3
    assert sleeps[0] == 3.0 and 0.25 <= sleeps[1] <= 1.0  # Retry-After, then jittered backoff 0.5 * 2**1


async def test_chat_does_not_retry_client_errors(sleeps):
    rec = Recorder(httpx.Response(400, json={"error": {"code": 400, "message": "bad"}}))
    with pytest.raises(BackendError) as ei:
        await backend(rec).chat(SOL, ChatRequest(messages=[]))
    assert ei.value.status == 400 and len(rec.requests) == 1 and sleeps == []


async def test_chat_gives_up_after_max_retries(sleeps):
    rec = Recorder(httpx.Response(502, text="bad gateway"))
    with pytest.raises(BackendError) as ei:
        await backend(rec, max_retries=2, retry_base_delay=0).chat(SOL, ChatRequest(messages=[]))
    assert ei.value.status == 502 and len(rec.requests) == 3 and len(sleeps) == 2


async def test_chat_402_retry_only_with_retry_after(sleeps):
    budget = httpx.Response(402, json={"error": {"code": 402, "message": "in-flight budget"}},
                            headers={"Retry-After": "1"})
    rec = Recorder(budget, httpx.Response(200, json=chat_payload()))
    assert (await backend(rec).chat(SOL, ChatRequest(messages=[]))).text == "Paris."
    rec2 = Recorder(httpx.Response(402, json={"error": {"code": 402, "message": "no credits"}}))
    with pytest.raises(BackendError) as ei:
        await backend(rec2).chat(SOL, ChatRequest(messages=[]))
    assert not ei.value.retryable and len(rec2.requests) == 1


async def test_chat_error_payload_and_refusal(sleeps):
    err = {"error": {"code": 400, "message": "invalid", "metadata": {"error_type": "invalid_request"}}}
    with pytest.raises(BackendError) as ei:
        await backend(Recorder(httpx.Response(200, json=err))).chat(SOL, ChatRequest(messages=[]))
    assert ei.value.error_type == "invalid_request" and not ei.value.retryable
    refusal = chat_payload(content=None, finish="content_filter")
    res = await backend(Recorder(httpx.Response(200, json=refusal))).chat(FABLE, ChatRequest(messages=[]))
    assert res.text == "" and res.finish_reason == "content_filter"


def test_retry_after_parsing():
    assert orb._retry_after(None) is None and orb._retry_after("soon") is None
    assert orb._retry_after("2.5") == 2.5 and orb._retry_after("-3") == 0.0
    assert orb._retry_after("100000") == orb.MAX_RETRY_AFTER_S
    future = email.utils.formatdate(time.time() + 30, usegmt=True)
    assert 25 <= orb._retry_after(future) <= 30


async def test_aclose_only_closes_owned_client():
    external = httpx.AsyncClient(transport=httpx.MockTransport(Recorder(httpx.Response(200))))
    await OpenRouterBackend(api_key="k", client=external).aclose()
    assert not external.is_closed
    await external.aclose()
    owned = OpenRouterBackend(api_key="k")
    client = owned._http()
    await owned.aclose()
    assert client.is_closed
