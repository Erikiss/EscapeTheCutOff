"""Tests for cutoffmix.server: a scripted fake backend and fake router, plus the packaged default index (no network)."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi")
import httpx

from cutoffmix import server
from cutoffmix.config import AppConfig, RelayConfig, RouterConfig
from cutoffmix.server import create_app, flatten_messages
from cutoffmix.types import ModelSpec, StreamEvent

SENTINEL = "<<END_STEP>>"
MODELS = [
    ModelSpec(key="a", provider_id="x/a", display="Model A", color="#d97757"),
    ModelSpec(key="b", provider_id="x/b", display="Model B", color="#10a37f"),
]


class FakeBackend:
    """Scripted backend: think segment k streams ``para(k)``; segment ``final_at`` concludes; answers stream
    ``ANSWER``. ``delay`` sleeps before every chunk; ``refuse`` makes every call a refusal."""

    def __init__(self, *, final_at: int = 2, delay: float = 0.0, refuse: bool = False) -> None:
        self.final_at = final_at
        self.delay = delay
        self.refuse = refuse
        self.requests: list[Any] = []
        self.open = 0
        self.closed = 0

    async def stream_continuation(self, model, req):
        self.requests.append(req)
        self.open += 1
        try:
            if self.refuse:
                yield StreamEvent(kind="done", finish_reason="content_filter")
                return
            if req.phase == "answer":
                text = "The answer is 42."
            elif len(req.segments) >= self.final_at:
                text = "The notes are complete. FINAL ANSWER: 42" + SENTINEL
            else:
                text = f"Step {len(req.segments)}: the record supports this intermediate result." + SENTINEL
            for i in range(0, len(text), 8):
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield StreamEvent(kind="text", text=text[i:i + 8])
            yield StreamEvent(kind="done", finish_reason="stop",
                              usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.01})
        finally:
            self.open -= 1
            self.closed += 1

    async def chat(self, model, req):  # pragma: no cover - not used by the server
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
    def __init__(self, models: list[str]) -> None:
        self.models = models

    def reset(self) -> None:
        pass

    @staticmethod
    def window(question, segments, n):
        return "\n\n".join([question, *segments[-n:]])

    def decide(self, window_text, *, current, available=None, update=True):
        avail = list(available or self.models)
        return FakeDecision(avail[0], "sample", {m: 1 / len(avail) for m in avail}, {m: float("-inf") for m in avail},
                            {"government": 1.0}, 0.9)

    def decide_speculative(self, window_text, drafts, *, current):
        first = next(iter(drafts))
        return FakeDecision(first, "speculative", {m: float(m == first) for m in drafts}, {m: 0.0 for m in drafts},
                            {"government": 1.0}, 0.9)


class FakeIndex:
    def __init__(self) -> None:
        self.models = ["a", "b"]
        self.topics = ["government", "sports"]
        self.probes = {"current_d": None, "mcq": None}
        self.meta = {"method": "test"}

    def primary(self, preference=("current_d",)):
        return SimpleNamespace(probe="current_d")

    def table(self, probe=None):
        return [{"topic": t, "model": m, "tau": 27.5 if m == "a" else float("nan"), "tau_label": "2026-04",
                 "sd": 1.0, "n": 3, "hazard": 0.2, "abstain": None, "p_best": 0.5}
                for t in self.topics for m in self.models]


def make_cfg() -> AppConfig:
    return AppConfig(models=list(MODELS), relay=RelayConfig(), router=RouterConfig(), today="2026-11-24")


def make_app(backend: FakeBackend | None = None, *, index: Any = None):
    backend = backend or FakeBackend()
    app = create_app(make_cfg(), backend=backend)
    runtime = app.state.runtime
    runtime.seen_cfgs = []

    def factory(cfg):
        runtime.seen_cfgs.append(cfg)
        return FakeRouter(cfg.model_keys())

    runtime.router_factory = factory
    runtime._load_routing = lambda: (index, "test-index" if index is not None else None, None)
    return app, backend


def client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


def sse_payloads(body: str) -> list[Any]:
    out = []
    for block in body.split("\n\n"):
        data = "\n".join(line[5:].lstrip() for line in block.split("\n") if line.startswith("data:"))
        if data:
            out.append(data if data == "[DONE]" else json.loads(data))
    return out


# --------------------------------------------------------------------------------------
# UI and metadata endpoints
# --------------------------------------------------------------------------------------


async def test_ui_is_a_single_self_contained_page():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "<title>" in html and "/api/relay" in html and "/api/index" in html and "prefers-color-scheme" in html
    assert not re.search(r"https?://", html), "the UI must not load anything from the network"
    assert "<link" not in html and "@import" not in html


async def test_api_config_lists_models_defaults_and_policies():
    app, _ = make_app()
    async with client(app) as c:
        cfg = (await c.get("/api/config")).json()
    assert [m["key"] for m in cfg["models"]] == ["a", "b"]
    assert cfg["models"][0]["color"] == "#d97757" and cfg["models"][0]["display"] == "Model A"
    assert cfg["defaults"]["strategy"] == "sequential" and cfg["defaults"]["segment_tokens"] == 128
    assert "fixed:b" in cfg["policies"] and "softmax" in cfg["policies"]
    assert cfg["today"] == "2026-11-24" and cfg["mock"] is False
    assert len(cfg["topics"]) == 15


async def test_api_index_returns_json_safe_rows():
    app, _ = make_app(index=FakeIndex())
    async with client(app) as c:
        body = (await c.get("/api/index")).json()
        bad = await c.get("/api/index", params={"probe": "recall"})
        mcq = (await c.get("/api/index", params={"probe": "mcq"})).json()
    assert body["available"] is True and body["probe"] == "current_d" and body["source"] == "test-index"
    assert len(body["rows"]) == 4 and body["rows"][1]["tau"] is None  # NaN became null
    assert bad.status_code == 400 and "recall" in bad.json()["detail"]
    assert mcq["probe"] == "mcq"


async def test_api_index_without_an_index():
    app, _ = make_app(index=None)
    async with client(app) as c:
        body = (await c.get("/api/index")).json()
    assert body["available"] is False and body["rows"] == []


async def test_default_index_is_the_packaged_prior_and_mock_mode_uses_the_mock_truths():
    from cutoffmix.config import default_config

    async with client(create_app(default_config())) as c:  # no request reaches a model: no key needed
        real = (await c.get("/api/index")).json()
    assert real["available"] is True and real["source"] == "packaged index_prior.json"
    assert real["meta"]["sources"] == ["apoorvumang/knowledge-cutoff (CC BY 4.0)"]
    assert real["meta"]["data_range"]["last"] <= "2026-06-30" and "PRIOR" in real["meta"]["note"]
    async with client(create_app(default_config(), mock=True)) as c:
        mock = (await c.get("/api/index")).json()
    assert mock["source"] == "mock truths" and "not a measurement" in mock["meta"]["note"]


async def test_v1_models():
    app, _ = make_app()
    async with client(app) as c:
        data = (await c.get("/v1/models")).json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["cutoffmix", "cutoffmix-speculative", "cutoffmix-fixed-a",
                                               "cutoffmix-fixed-b"]


# --------------------------------------------------------------------------------------
# /api/relay
# --------------------------------------------------------------------------------------


async def test_api_relay_streams_relay_events_as_sse():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "Who is the chancellor right now?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache" and r.headers["x-accel-buffering"] == "no"
    events = sse_payloads(r.text)
    assert events[0]["type"] == "segment_start" and events[-1]["type"] == "done"
    assert events[0]["scores"] == {"a": None, "b": None}  # -inf scores are sent as null
    done = events[-1]["data"]
    assert done["answer"] == "The answer is 42." and done["status"] == "ok"
    # 4 calls; the concluding one is closed at the final marker, so 3 report usage
    assert done["usage_total"]["calls"] == 4 and done["usage_total"]["cost"] == pytest.approx(0.03)


async def test_api_relay_applies_per_request_overrides():
    app, backend = make_app()
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "q?", "segment_tokens": 64, "strategy": "speculative",
                                             "policy": "fixed:b"})
    assert r.status_code == 200
    cfg = app.state.runtime.seen_cfgs[-1]
    assert cfg.relay.segment_tokens == 64 and cfg.relay.strategy == "speculative" and cfg.router.policy == "fixed:b"
    assert {req.target_tokens for req in backend.requests if req.phase == "think"} == {64}


@pytest.mark.parametrize("payload, needle", [
    ({}, "question"),
    ({"question": "   "}, "question"),
    ({"question": "q", "strategy": "parallel"}, "strategy"),
    ({"question": "q", "policy": "fixed:zzz"}, "zzz"),
    ({"question": "q", "segment_tokens": "many"}, "segment_tokens"),
    ({"question": "q", "segment_tokens": 0}, "segment_tokens"),
    ({"question": "q", "segment_tokens": 99999}, "segment_tokens"),
])
async def test_api_relay_rejects_invalid_requests(payload, needle):
    app, backend = make_app()
    async with client(app) as c:
        r = await c.post("/api/relay", json=payload)
    assert r.status_code == 400 and needle in r.json()["detail"]
    assert backend.requests == []


async def test_api_relay_rejects_non_json():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/api/relay", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


async def test_idle_streams_get_heartbeats(monkeypatch):
    monkeypatch.setattr(server, "POLL_S", 0.01)
    monkeypatch.setattr(server, "HEARTBEAT_S", 0.02)
    app, _ = make_app(FakeBackend(final_at=1, delay=0.03))
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "q?"})
    assert ": ping" in r.text
    assert sse_payloads(r.text)[-1]["type"] == "done"


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_client_disconnect_stops_the_relay(monkeypatch, spec_version):
    monkeypatch.setattr(server, "POLL_S", 0.01)
    backend = FakeBackend(final_at=50, delay=0.02)
    app, _ = make_app(backend)
    body = json.dumps({"question": "q?"}).encode()
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    disconnected = asyncio.Event()

    async def receive():
        if pending:
            return pending.pop(0)
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and b'"type":"delta"' in message.get("body", b""):
            disconnected.set()

    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/api/relay", "raw_path": b"/api/relay", "query_string": b"",
             "root_path": "", "headers": [(b"content-type", b"application/json"), (b"host", b"test")],
             "client": ("127.0.0.1", 1234), "server": ("test", 80)}
    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert disconnected.is_set()
    assert backend.open == 0 and backend.closed >= 1  # the in-flight model stream was closed
    assert len(backend.requests) <= 2  # the relay stopped instead of running all 50 segments


# --------------------------------------------------------------------------------------
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------------------


async def test_chat_completions_non_stream():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "cutoffmix", "messages": [{"role": "user", "content": "Who is the chancellor?"}]})
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "chat.completion" and data["model"] == "cutoffmix" and data["id"].startswith("chatcmpl-")
    message = data["choices"][0]["message"]
    assert message["role"] == "assistant" and message["content"] == "The answer is 42."
    assert message["reasoning"].startswith("Step 0:") and "\n\nStep 1:" in message["reasoning"]
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45,
                             "cost": pytest.approx(0.03)}
    assert data["cutoffmix"]["answer"] == "The answer is 42." and "segments" in data["cutoffmix"]


async def test_chat_completions_stream_chunks():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "cutoffmix", "stream": True, "messages": [{"role": "user", "content": "Who is it?"}]})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks = sse_payloads(r.text)
    assert chunks[-1] == "[DONE]"
    chunks = chunks[:-1]
    assert len({c["id"] for c in chunks}) == 1
    assert all(c["object"] == "chat.completion.chunk" and c["model"] == "cutoffmix" for c in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    reasoning = "".join(d.get("reasoning", "") for d in deltas)
    content = "".join(d.get("content", "") for d in deltas)
    assert reasoning.startswith("Step 0:") and "\n\nStep 1:" in reasoning and "FINAL" not in reasoning
    assert content == "The answer is 42."
    assert any(d.get("cutoffmix", {}).get("type") == "segment_start" for d in deltas)
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert len(finish) == 1 and finish[0]["choices"][0]["finish_reason"] == "stop"
    usage = chunks[-1]
    assert usage["choices"] == [] and usage["usage"]["total_tokens"] == 45
    assert usage["cutoffmix"]["answer"] == "The answer is 42."


@pytest.mark.parametrize("model_id, field, value", [
    ("cutoffmix-speculative", "strategy", "speculative"),
    ("cutoffmix-fixed-b", "policy", "fixed:b"),
])
async def test_chat_completions_model_ids_select_strategy_and_policy(model_id, field, value):
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": model_id, "messages": [{"role": "user",
                                                                                       "content": "q?"}]})
    assert r.status_code == 200
    cfg = app.state.runtime.seen_cfgs[-1]
    assert (cfg.relay.strategy if field == "strategy" else cfg.router.policy) == value


async def test_chat_completions_body_overrides_win():
    app, _ = make_app()
    async with client(app) as c:
        await c.post("/v1/chat/completions", json={"model": "cutoffmix-speculative", "strategy": "sequential",
                                                   "segment_tokens": 32,
                                                   "messages": [{"role": "user", "content": "q?"}]})
    cfg = app.state.runtime.seen_cfgs[-1]
    assert cfg.relay.strategy == "sequential" and cfg.relay.segment_tokens == 32


async def test_chat_completions_errors_use_the_openai_format():
    app, _ = make_app()
    async with client(app) as c:
        unknown = await c.post("/v1/chat/completions", json={"model": "gpt-4", "messages": [{"role": "user",
                                                                                          "content": "q"}]})
        empty = await c.post("/v1/chat/completions", json={"model": "cutoffmix", "messages": []})
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "model_not_found"
    assert empty.status_code == 400 and empty.json()["error"]["type"] == "invalid_request_error"


async def test_chat_completions_report_relay_failures():
    app, _ = make_app(FakeBackend(refuse=True))
    async with client(app) as c:
        plain = await c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "q"}]})
        streamed = await c.post("/v1/chat/completions", json={"stream": True,
                                                              "messages": [{"role": "user", "content": "q"}]})
    assert plain.status_code == 502 and "no model left" in plain.json()["error"]["message"]
    chunks = [c for c in sse_payloads(streamed.text) if c != "[DONE]"]
    finish = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finish[0]["choices"][0]["finish_reason"] == "error" and "no model left" in finish[0]["error"]["message"]


def test_flatten_messages():
    assert flatten_messages([{"role": "user", "content": "  Who is it?  "}]) == "Who is it?"
    text = flatten_messages([
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": [{"type": "text", "text": "Who won?"}, {"type": "image_url", "image_url": {}}]},
        {"role": "assistant", "content": "Team A."},
        {"role": "user", "content": "And before that?"},
    ])
    assert text == "Instructions: Answer briefly.\n\nUser: Who won?\n\nAssistant: Team A.\n\nUser: And before that?"
    for bad in (None, [], [{"role": "user", "content": ""}], ["text"]):
        with pytest.raises(ValueError):
            flatten_messages(bad)
