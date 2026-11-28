"""Tests for cutoffmix.server: a scripted fake backend and fake router, plus the packaged default index (no network)."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
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


TEST_HOSTS = ("test",)  # the host name of client() below


def make_cfg() -> AppConfig:
    return AppConfig(models=list(MODELS), relay=RelayConfig(), router=RouterConfig(), today="2026-11-24")


def make_app(backend: FakeBackend | None = None, *, index: Any = None):
    backend = backend or FakeBackend()
    app = create_app(make_cfg(), backend=backend, allowed_hosts=TEST_HOSTS)
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


class BrokenStream(Exception):
    retryable = False  # the relay drops the attempt and re-routes instead of retrying the same model


class FlakyBackend(FakeBackend):
    """The first think call streams half a paragraph (past the relay's preamble buffer) and then fails."""

    PARTIAL = "This half paragraph is long enough to get past the preamble buffer of the relay before it"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.failed = False

    async def stream_continuation(self, model, req):
        if not self.failed and req.phase == "think":
            self.failed = True
            self.requests.append(req)
            for i in range(0, len(self.PARTIAL), 8):
                yield StreamEvent(kind="text", text=self.PARTIAL[i:i + 8])
            raise BrokenStream("connection reset by peer")
        async for event in super().stream_continuation(model, req):
            yield event


def http_scope(path: str, headers: list[tuple[bytes, bytes]] | None = None,
               spec_version: str = "2.4") -> dict[str, Any]:
    """An ASGI HTTP scope for a JSON POST to ``path`` (for driving the app with a hand-written receive/send)."""
    return {"type": "http", "asgi": {"version": "3.0", "spec_version": spec_version}, "http_version": "1.1",
            "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
            "root_path": "", "client": ("127.0.0.1", 1234), "server": ("test", 80),
            "headers": headers or [(b"content-type", b"application/json"), (b"host", b"test")]}


def js_function(html: str, name: str) -> str:
    """The source of the JavaScript function ``name`` in ``html`` (brace matching from its first ``{``)."""
    start = html.index(f"function {name}(")
    depth = 0
    for j in range(html.index("{", start), len(html)):
        depth += {"{": 1, "}": -1}.get(html[j], 0)
        if depth == 0:
            return html[start:j + 1]
    raise AssertionError(f"unbalanced function {name}")


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
    assert "@import" not in html
    links = re.findall(r"<link\b[^>]*>", html)
    assert all(re.search(r'href="data:', tag) for tag in links), links  # only inline resources (the icon)


async def test_ui_link_with_a_question_never_starts_a_run_by_itself():
    """A cross-site link to ``/?q=...&run=1`` must not start a paid run: the question is only prefilled."""
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    boot = js_function(html, "boot")
    assert 'params.get("q")' in boot
    assert "run()" not in boot and 'get("run")' not in boot


async def test_ui_segment_tokens_input_accepts_every_value_the_server_accepts():
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    tag = re.search(r'<input id="segtok"[^>]*>', html)
    assert tag is not None
    attrs = dict(re.findall(r'(\w+)="([^"]*)"', tag.group(0)))
    assert attrs["min"] == "1" and attrs["max"] == str(server.MAX_SEGMENT_TOKENS)
    assert attrs["step"] in ("1", "any")  # e.g. a configured 200 or 100 must not block the Run button


async def test_ui_new_run_resets_the_stopped_hint():
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    reset = js_function(html, "resetRun")
    assert '$("notes-hint").textContent = NOTES_HINT' in reset
    default = re.search(r'id="notes-hint">([^<]*)<', html)
    assert default is not None and f'const NOTES_HINT = "{default.group(1)}";' in html


def run_js_function(source: str, name: str, args: list[Any]) -> list[Any]:
    """``[name(a) for a in args]`` evaluated by Node (the test is skipped without a ``node`` on PATH)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    script = f"{source}\nconsole.log(JSON.stringify({json.dumps(args)}.map((a) => {name}(a))));"
    done = subprocess.run([node, "-"], input=script, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


# Routing scores U of a real round-robin relay (tests/fixtures/live_sse/relay_trace_round_robin.json, segments 1-3)
LIVE_SCORES = [
    {"claude-fable-5.2": 25.724, "gpt-6.2-sol": 27.698, "gemini-4-neon": 26.148},
    {"claude-fable-5.2": 25.626, "gpt-6.2-sol": 27.606, "gemini-4-neon": 26.054},
    {"claude-fable-5.2": 25.565, "gpt-6.2-sol": 27.548, "gemini-4-neon": 25.995},
]


async def test_ui_shows_routing_scores_as_gaps_to_the_best_model_not_as_dates():
    """U already subtracts the uncertainty and stale-answer penalties, so it is no calendar month: the inspector
    shows it as months behind the best model of the same decision."""
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    inspector = js_function(html, "renderInspector")
    assert "fmtTau" not in inspector and "scoreGaps(" in inspector
    assert "freshness score" not in html and "freshness of the window" not in html
    title = re.search(r'const SCORE_TITLE = ([^;]*);', html)
    assert title is not None and "Not a date" in title.group(1)

    gaps = run_js_function(js_function(html, "scoreGaps"), "scoreGaps", LIVE_SCORES)
    for scores, labels in zip(LIVE_SCORES, gaps, strict=True):
        best = max(scores.values())
        assert set(labels) == set(scores)
        for model, label in labels.items():
            assert not re.search(r"\d{4}-\d{2}", label), label
            if scores[model] == best:
                assert label == "best"
            else:
                m = re.fullmatch(r"−(\d+\.\d) mo", label)
                assert m is not None, label
                assert abs(float(m.group(1)) - (best - scores[model])) <= 0.051
    edge = run_js_function(js_function(html, "scoreGaps"), "scoreGaps", [
        {"a": 26.0, "b": 26.0, "c": 25.96, "d": 25.9995},  # tie, a sub-0.1 gap, a sub-0.01 gap
        {"a": 26.0, "b": None},  # -inf on the server (the index lacks b) arrives as null
        {"a": 0.0, "b": 0.0},  # no index: every score is 0
        {},  # inline answer: no scores at all
    ])
    assert edge == [
        {"a": "tied best", "b": "tied best", "c": "−0.04 mo", "d": "−<0.01 mo"},
        {"a": "best", "b": "not in index"},
        {"a": "tied best", "b": "tied best"},
        {},
    ]


async def test_ui_inspector_lists_every_scored_model_also_for_forced_decisions():
    """Round robin and fixed policies report probability 1 for the chosen model only; the other models must still
    show their score gap, or a forced pick of a worse model looks like the best choice."""
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    inspector = js_function(html, "renderInspector")
    assert "new Set([...Object.keys(rec.probs || {}), ...Object.keys(rec.scores || {})])" in inspector


async def test_ui_index_labels_p_freshest_and_explains_the_routing_penalties():
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    assert "P(best)" not in html
    heat, tip, about = (js_function(html, n) for n in ("renderHeatmap", "showTip", "aboutIndex"))
    assert '"(freshest)"' in heat and "P_FRESHEST_TITLE" in heat
    title = re.search(r'const P_FRESHEST_TITLE = ([^;]*);', html)
    assert title is not None and "latest knowledge cutoff on this" in title.group(1)
    assert "P(freshest)" in tip and "latest cutoff on this topic" in tip
    # the routing explanation is always there, also for an index without provenance metadata
    assert "return h(\"span\", {})" not in about
    assert "uncertain estimate" in about and "stale-answer rate" in about and "κ·sd" in about and "λ·H" in about


async def test_ui_preset_select_explore_badges_and_exploration_strip():
    """The explore preset's UI (DESIGN.md §7.7): a preset select, lens/check/continue badges, the exploration strip
    and the pure helpers that read the relay's (optional) explore data."""
    app, _ = make_app()
    async with client(app) as c:
        html = (await c.get("/")).text
    assert '<select id="preset"' in html and 'id="xstats" hidden' in html
    assert "body.preset = preset" in js_function(html, "run")
    boot = js_function(html, "boot")
    assert "cfg.presets" in boot and "cfg.defaults.preset" in boot and "segment_tokens" in boot
    assert "exploreBadge(rec)" in js_function(html, "startSegment")
    end = js_function(html, "endSegment")
    assert "flags.lens_verdict" in end and "lensVerdict(rec.text)" in end
    assert "renderExplore((ev.data || {}).explore)" in js_function(html, "onEvent")
    assert "renderExplore" not in js_function(html, "finish")  # finish() keeps its stubbable contract
    assert 'd.explore && typeof d.explore === "object" ? d.explore : d' in js_function(html, "recordFrom")
    for kind in ("lens", "check", "continue"):
        assert f".badge.x.{kind}" in html
    assert "--ok-soft" in html and "--warn-soft" in html and html.count("--ok-soft:") == 2  # light and dark values

    stats = run_js_function(js_function(html, "exploreStats"), "exploreStats", [
        {"topics_visited": {"all": 2, "main_line": 1}, "lens": {"offered": 1, "written": 1, "kept": 1, "dropped": 0},
         "labels": {"coverage": {"two_or_more_items": 0.83, "where_we_stand": 1.0}}, "forced_continuations": 1,
         "premature_finals": [{}]},
        {},
        {"lens": "x", "topics_visited": ["a", "b"], "labels": {"coverage": 0.5}, "forced_continuations": "3"},
        None,
    ])
    empty = {"topics": None, "lenses": None, "lensDetail": None, "labels": None, "forced": None, "forcedDetail": None}
    assert stats == [
        {"topics": 2, "lenses": "1", "lensDetail": "1 kept · 0 dropped", "labels": "83%", "forced": "1",
         "forcedDetail": "1 early"},
        empty,
        empty | {"topics": 2, "labels": "50%"},
        empty,
    ]
    verdicts = run_js_function(js_function(html, "lensVerdict"), "lensVerdict", [
        "Lens kept: the mapping holds.", "- **Lens dropped:** it breaks.", "Claim: x.\nlens Kept : y", "Lens (ecology): a.",
        "", None])
    assert verdicts == ["kept", "dropped", "kept", None, None, None]


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


async def test_api_config_lists_the_presets_and_the_active_one():
    app, _ = make_app()
    async with client(app) as c:
        cfg = (await c.get("/api/config")).json()
    assert cfg["presets"] == [{"name": "explore", "segment_tokens": 192, "max_segments": 16},
                              {"name": "rally", "segment_tokens": 160, "max_segments": 12}]
    assert cfg["defaults"]["preset"] is None
    assert "cutoffmix-explore" in cfg["model_ids"]
    from cutoffmix.config import load_config

    explore_app = create_app(load_config(preset="explore"), backend=FakeBackend(), allowed_hosts=TEST_HOSTS)
    async with client(explore_app) as c:
        cfg = (await c.get("/api/config")).json()
    assert cfg["defaults"]["preset"] == "explore" and cfg["defaults"]["segment_tokens"] == 192


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

    async with client(create_app(default_config(), allowed_hosts=TEST_HOSTS)) as c:  # no request reaches a model: no key needed
        real = (await c.get("/api/index")).json()
    assert real["available"] is True and real["source"] == "packaged index_prior.json"
    assert real["meta"]["sources"][0] == "apoorvumang/knowledge-cutoff (CC BY 4.0)"
    assert real["meta"]["data_range"]["last"] <= "2026-11-30"
    assert "PRIOR" in real["meta"]["note"] and "MEASURED" in real["meta"]["note"]
    async with client(create_app(default_config(), mock=True, allowed_hosts=TEST_HOSTS)) as c:
        mock = (await c.get("/api/index")).json()
    assert mock["source"] == "mock truths" and "not a measurement" in mock["meta"]["note"]


async def test_v1_models():
    app, _ = make_app()
    async with client(app) as c:
        data = (await c.get("/v1/models")).json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["cutoffmix", "cutoffmix-speculative", "cutoffmix-explore",
                                               "cutoffmix-rally", "cutoffmix-fixed-a", "cutoffmix-fixed-b"]


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


async def test_api_relay_preset_runs_the_explore_configuration():
    """``"preset": "explore"`` runs the config ``serve --preset explore`` would have: the preset is layered between
    the packaged default and the server's config file; the other per-request overrides apply on top."""
    app, backend = make_app()
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "q?", "preset": "explore"})
        assert r.status_code == 200 and sse_payloads(r.text)[-1]["type"] == "done"
        cfg = app.state.runtime.seen_cfgs[-1]
        assert cfg.explore.enabled and cfg.relay.segment_tokens == 192 and cfg.relay.hard_cut_chars == 1500
        assert cfg.model("gpt-6.2-sol").supports_seed and cfg.router.policy == "softmax"
        assert {req.target_tokens for req in backend.requests if req.phase == "think"} == {192}
        r = await c.post("/api/relay", json={"question": "q?", "preset": "explore", "segment_tokens": 64,
                                             "policy": "round_robin"})
        assert r.status_code == 200
        cfg = app.state.runtime.seen_cfgs[-1]
        assert cfg.explore.enabled and cfg.relay.segment_tokens == 64 and cfg.router.policy == "round_robin"
        r = await c.post("/api/relay", json={"question": "q?"})  # the server's own config is untouched
        assert r.status_code == 200
        cfg = app.state.runtime.seen_cfgs[-1]
        assert not cfg.explore.enabled and cfg.relay.segment_tokens == 128 and cfg.model_keys() == ["a", "b"]
    assert backend.closed == backend.open + backend.closed  # the shared backend served every run


async def test_api_relay_preset_is_layered_under_the_servers_config_file(tmp_path):
    cfg_file = tmp_path / "server.yaml"
    cfg_file.write_text("relay: {segment_tokens: 150}\napp_title: mine\n", encoding="utf-8")
    from cutoffmix.config import load_config

    app = create_app(load_config(cfg_file), backend=FakeBackend(), allowed_hosts=TEST_HOSTS, config_path=cfg_file)
    runtime = app.state.runtime
    runtime.seen_cfgs = []
    runtime.router_factory = lambda cfg: (runtime.seen_cfgs.append(cfg), FakeRouter(cfg.model_keys()))[1]
    runtime._load_routing = lambda: (None, None, None)
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "q?", "preset": "explore"})
    assert r.status_code == 200
    cfg = runtime.seen_cfgs[-1]
    assert cfg.explore.enabled and cfg.relay.hard_cut_chars == 1500  # from the preset
    assert cfg.relay.segment_tokens == 150 and cfg.app_title == "mine"  # the file still wins over the preset
    async with client(app) as c:
        presets = (await c.get("/api/config")).json()["presets"]
    assert presets == [{"name": "explore", "segment_tokens": 150, "max_segments": 16},
                       {"name": "rally", "segment_tokens": 150, "max_segments": 12}]


@pytest.mark.parametrize("payload, needle", [
    ({}, "question"),
    ({"question": "   "}, "question"),
    ({"question": "q", "strategy": "parallel"}, "strategy"),
    ({"question": "q", "policy": "fixed:zzz"}, "zzz"),
    ({"question": "q", "segment_tokens": "many"}, "segment_tokens"),
    ({"question": "q", "segment_tokens": 0}, "segment_tokens"),
    ({"question": "q", "segment_tokens": 99999}, "segment_tokens"),
    ({"question": "q", "preset": "wild"}, "preset must be 'explore'"),
    ({"question": "q", "preset": 1}, "preset"),
    ({"question": "q", "preset": "default"}, "preset"),
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


@pytest.mark.parametrize("path", ["/api/relay", "/v1/chat/completions"])
@pytest.mark.parametrize("value", ["1e999", "-1e999", "NaN", "Infinity"])
async def test_non_finite_segment_tokens_are_a_client_error(path, value):
    app, backend = make_app()
    body = '{"question": "q", "messages": [{"role": "user", "content": "q"}], "segment_tokens": ' + value + "}"
    async with client(app) as c:
        r = await c.post(path, content=body.encode(), headers={"content-type": "application/json"})
    assert r.status_code == 400
    detail = r.json()["detail"] if path == "/api/relay" else r.json()["error"]["message"]
    assert "segment_tokens" in detail and backend.requests == []


@pytest.mark.parametrize("path", ["/api/relay", "/v1/chat/completions"])
@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded",
                                          "multipart/form-data; boundary=x"])
async def test_posts_need_a_json_content_type(path, content_type):
    """Simple cross-site requests (no CORS preflight) cannot carry application/json, so they are refused."""
    app, backend = make_app()
    body = json.dumps({"question": "q", "messages": [{"role": "user", "content": "q"}]}).encode()
    headers = {"content-type": content_type} if content_type else {}
    async with client(app) as c:
        r = await c.post(path, content=body, headers=headers)
    assert r.status_code == 415 and backend.requests == []
    if path.startswith("/v1"):
        assert r.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("path", ["/api/relay", "/v1/chat/completions"])
@pytest.mark.parametrize("origin", ["http://evil.example", "http://test.evil.example", "https://test:8443", "null"])
async def test_posts_from_another_origin_are_refused(path, origin):
    app, backend = make_app()
    body = {"question": "q", "messages": [{"role": "user", "content": "q"}]}
    async with client(app) as c:
        r = await c.post(path, json=body, headers={"origin": origin})
    assert r.status_code == 403 and backend.requests == []
    assert "--allow-origin" in r.text


@pytest.mark.parametrize("origin, headers, allowed", [
    ("http://test", {}, ()),  # the page served by this server
    ("http://TEST/", {}, ()),
    ("https://notes.example", {"x-forwarded-host": "notes.example"}, ()),  # behind a reverse proxy
    ("https://notes.example", {}, ("https://notes.example/",)),  # --allow-origin
])
async def test_posts_from_the_own_or_an_allowed_origin_run(origin, headers, allowed):
    backend = FakeBackend()
    app = create_app(make_cfg(), backend=backend, allowed_origins=allowed, allowed_hosts=TEST_HOSTS)
    app.state.runtime.router_factory = lambda cfg: FakeRouter(cfg.model_keys())
    app.state.runtime._load_routing = lambda: (None, None, None)
    async with client(app) as c:
        r = await c.post("/api/relay", json={"question": "q?"}, headers={"origin": origin, **headers})
    assert r.status_code == 200 and sse_payloads(r.text)[-1]["type"] == "done"


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

    await asyncio.wait_for(app(http_scope("/api/relay", spec_version=spec_version), receive, send), timeout=5)
    assert disconnected.is_set()
    assert backend.open == 0 and backend.closed >= 1  # the in-flight model stream was closed
    assert len(backend.requests) <= 2  # the relay stopped instead of running all 50 segments


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_client_disconnect_stops_a_non_stream_chat_completion(monkeypatch, spec_version):
    """A client that times out (or is killed) while waiting for a non-stream answer must not leave the relay
    making paid calls to the end."""
    # Events arrive far more often than POLL_S, so the relay is never idle long enough for an idle-time check:
    # the disconnect must be noticed while events flow (the whole run would take about 3 s).
    monkeypatch.setattr(server, "POLL_S", 0.2)
    backend = FakeBackend(final_at=50, delay=0.005)
    app, _ = make_app(backend)
    body = json.dumps({"model": "cutoffmix", "messages": [{"role": "user", "content": "q?"}]}).encode()
    pending = [{"type": "http.request", "body": body, "more_body": False}]
    sent: list[dict[str, Any]] = []

    async def receive():
        if pending:
            return pending.pop(0)
        while not backend.open:  # the client gives up while the first model call is streaming
            await asyncio.sleep(0.005)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(http_scope("/v1/chat/completions", spec_version=spec_version), receive, send),
                           timeout=3)
    assert backend.open == 0 and backend.closed >= 1  # the in-flight model stream was closed
    assert len(backend.requests) <= 12  # the relay stopped within about POLL_S instead of running all 50 segments
    assert sent and sent[0]["type"] == "http.response.start" and sent[0]["status"] == 499


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
    # the concluding segment was closed at the final marker, before its usage chunk: the totals are a lower bound
    assert data["usage"] == {"prompt_tokens": 30, "completion_tokens": 15, "total_tokens": 45,
                             "cost": pytest.approx(0.03), "cutoffmix_partial": True}
    assert data["cutoffmix"]["usage_complete"] is False
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


async def test_chat_completions_explore_model_id_and_preset_override():
    app, _ = make_app()
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={"model": "cutoffmix-explore",
                                                       "messages": [{"role": "user", "content": "q?"}]})
        assert r.status_code == 200 and r.json()["model"] == "cutoffmix-explore"
        cfg = app.state.runtime.seen_cfgs[-1]
        assert cfg.explore.enabled and cfg.relay.segment_tokens == 192
        r = await c.post("/v1/chat/completions", json={"model": "cutoffmix", "preset": "explore", "stream": True,
                                                       "messages": [{"role": "user", "content": "q?"}]})
        assert r.status_code == 200 and app.state.runtime.seen_cfgs[-1].explore.enabled
        bad = await c.post("/v1/chat/completions", json={"model": "cutoffmix", "preset": "wild",
                                                         "messages": [{"role": "user", "content": "q?"}]})
        assert bad.status_code == 400 and "preset must be 'explore'" in bad.json()["error"]["message"]
        unknown = await c.post("/v1/chat/completions", json={"model": "cutoffmix-wild",
                                                             "messages": [{"role": "user", "content": "q?"}]})
    assert unknown.status_code == 404 and "cutoffmix-explore" in unknown.json()["error"]["message"]


class ExploreRelay:
    """A scripted relay: a lens note, its check and the explore metrics in ``done`` (as DESIGN.md §7 defines)."""

    EXPLORE = {"topics_visited": {"all": 2, "main_line": 1},
               "lens": {"offered": 1, "written": 1, "kept": 1, "dropped": 0, "ignored": 0, "unverified": 0},
               "labels": {"counts": {"Claim": 2}, "coverage": {"two_or_more_items": 1.0, "where_we_stand": 1.0}},
               "forced_continuations": 0, "premature_finals": [], "seed": 1, "seed_source": "run"}

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, question, *, run_id=None):
        from cutoffmix.types import RelayEvent

        self.calls += 1
        base = dict(phase="think", decision="sample", topic_posterior={"government": 1.0})
        yield RelayEvent(type="segment_start", segment=0, model="a", data={"explore": {"kind": "lens", "field": "ecology",
                                                                                       "tier": "mid", "cue": None}},
                         **base)
        yield RelayEvent(type="delta", segment=0, model="a", phase="think", text="Lens (ecology): a mapping.")
        yield RelayEvent(type="segment_end", segment=0, model="a", phase="think", text="Lens (ecology): a mapping.",
                         decision="sample", data={"explore": {"kind": "lens", "field": "ecology", "tier": "mid",
                                                              "lens_prefix_ok": True}})
        yield RelayEvent(type="segment_start", segment=1, model="b", data={"explore": {"kind": "check", "field": "ecology",
                                                                                       "tier": None, "cue": None}},
                         **base)
        yield RelayEvent(type="segment_end", segment=1, model="b", phase="think", text="Lens kept: it holds.",
                         decision="sample", data={"explore": {"kind": "check", "field": "ecology",
                                                              "lens_verdict": "kept"}})
        yield RelayEvent(type="phase", segment=2, model="b", phase="answer", decision="sample", data={})
        yield RelayEvent(type="delta", segment=2, model="b", phase="answer", text="42")
        yield RelayEvent(type="segment_end", segment=2, model="b", phase="answer", text="42", decision="sample", data={})
        yield RelayEvent(type="done", data={"question": question, "answer": "42", "status": "ok", "switches": 1,
                                            "share_by_model": {"a": 0.5, "b": 0.5}, "usage_total": {},
                                            "segments": [], "explore": self.EXPLORE})


async def test_explore_metadata_reaches_both_streams_and_the_final_chunk(monkeypatch):
    from cutoffmix.relay import RelayRuntime

    app, _ = make_app()
    scripted = ExploreRelay()
    monkeypatch.setattr(RelayRuntime, "relay", lambda self, cfg=None: scripted)  # the preset runtime too
    async with client(app) as c:
        sse = await c.post("/api/relay", json={"question": "q?", "preset": "explore"})
        events = sse_payloads(sse.text)
        assert events[0]["data"]["explore"] == {"kind": "lens", "field": "ecology", "tier": "mid", "cue": None}
        assert events[4]["data"]["explore"]["lens_verdict"] == "kept"
        assert events[-1]["data"]["explore"] == ExploreRelay.EXPLORE
        r = await c.post("/v1/chat/completions", json={"model": "cutoffmix-explore", "stream": True,
                                                       "messages": [{"role": "user", "content": "q?"}]})
        chunks = [ch for ch in sse_payloads(r.text) if ch != "[DONE]"]
        metas = [ch["choices"][0]["delta"]["cutoffmix"] for ch in chunks if ch["choices"]
                 and "cutoffmix" in ch["choices"][0]["delta"]]
        starts = [m for m in metas if m.get("type") == "segment_start"]
        assert starts[0]["data"]["explore"]["kind"] == "lens" and starts[1]["data"]["explore"]["kind"] == "check"
        assert chunks[-1]["cutoffmix"]["explore"] == ExploreRelay.EXPLORE  # the final chunk carries the metrics
        plain = await c.post("/v1/chat/completions", json={"model": "cutoffmix-explore",
                                                           "messages": [{"role": "user", "content": "q?"}]})
    assert plain.json()["cutoffmix"]["explore"]["lens"]["kept"] == 1 and scripted.calls == 3


async def test_chat_completions_body_overrides_win():
    app, _ = make_app()
    async with client(app) as c:
        await c.post("/v1/chat/completions", json={"model": "cutoffmix-speculative", "strategy": "sequential",
                                                   "segment_tokens": 32,
                                                   "messages": [{"role": "user", "content": "q?"}]})
    cfg = app.state.runtime.seen_cfgs[-1]
    assert cfg.relay.strategy == "sequential" and cfg.relay.segment_tokens == 32


async def test_chat_completions_stream_closes_the_text_of_a_dropped_attempt():
    """Streamed reasoning of an attempt that fails later cannot be taken back: it is closed with a visible marker
    and the re-routed attempt starts a new paragraph instead of being glued onto the half sentence."""
    backend = FlakyBackend()
    app, _ = make_app(backend)
    async with client(app) as c:
        r = await c.post("/v1/chat/completions", json={
            "model": "cutoffmix", "stream": True, "messages": [{"role": "user", "content": "Who is it?"}]})
    chunks = [c for c in sse_payloads(r.text) if c != "[DONE]"]
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning", "") for c in chunks if c["choices"])
    assert reasoning.startswith("This half paragraph")  # it was streamed before the failure
    assert " [segment dropped: error]\n\nStep 0:" in reasoning
    assert "\n\nStep 1:" in reasoning and reasoning.count("[segment dropped") == 1
    done = chunks[-1]["cutoffmix"]
    assert done["status"] == "ok" and done["segments"][0]["text"].startswith("Step 0:")  # the trace is clean


async def test_chat_completions_reject_an_over_long_conversation():
    app, backend = make_app()
    text = "x" * (server.MAX_QUESTION_CHARS + 1)
    async with client(app) as c:
        for stream in (False, True):
            r = await c.post("/v1/chat/completions", json={"model": "cutoffmix", "stream": stream,
                                                           "messages": [{"role": "user", "content": text}]})
            assert r.status_code == 400
            assert r.json()["error"]["type"] == "invalid_request_error"
            assert str(server.MAX_QUESTION_CHARS) in r.json()["error"]["message"]
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": "y" * 3000} for i in range(7)]
        r = await c.post("/v1/chat/completions", json={"model": "cutoffmix", "messages": history})
        assert r.status_code == 400  # each message is short, the flattened conversation is not
    assert backend.requests == []


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
