"""Regression tests for findings on the server, CLI, web UI, gate and shipped data (no network)."""

from __future__ import annotations

import collections
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cutoffmix.cli import TerminalPrinter, main
from cutoffmix.config import AppConfig, RelayConfig, RouterConfig, load_config
from cutoffmix.relay import RelayRuntime
from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.types import ModelSpec, StreamEvent

ROOT = Path(__file__).resolve().parent.parent
LIVE_TRACE = ROOT / "tests" / "fixtures" / "live_sse" / "relay_trace_round_robin.json"
MODELS = [ModelSpec(key="a", provider_id="x/a"), ModelSpec(key="b", provider_id="x/b")]
SENTINEL = "<<END_STEP>>"


class FakeBackend:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def stream_continuation(self, model, req):
        self.requests.append(req)
        text = "The answer is 42." if req.phase == "answer" else (
            "The notes are complete. FINAL ANSWER: 42" if req.segments else "Step 0: the record holds.") + SENTINEL
        yield StreamEvent(kind="text", text=text)
        yield StreamEvent(kind="done", finish_reason="stop", usage={"prompt_tokens": 1, "total_tokens": 2})

    async def chat(self, model, req):  # pragma: no cover - not used
        raise NotImplementedError


def make_app(**kw: Any):
    pytest.importorskip("fastapi")
    from cutoffmix.router.router import Router
    from cutoffmix.server import create_app

    backend = FakeBackend()
    cfg = AppConfig(models=list(MODELS), relay=RelayConfig(), router=RouterConfig(policy="round_robin"),
                    today="2026-11-24")
    app = create_app(cfg, backend=backend, **kw)
    app.state.runtime.router_factory = lambda c: Router(c.model_keys(), index=None, cfg=c.router, seed=0)
    return app, backend


def client(app, base_url: str):
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url)


# --------------------------------------------------------------------------------------
# Host allowlist (DNS rebinding) and origin normalisation
# --------------------------------------------------------------------------------------


async def test_a_dns_rebinding_host_is_refused():
    app, backend = make_app()
    rebound = "rebind.attacker.example:32805"
    async with client(app, "http://127.0.0.1:32805") as c:
        r = await c.post("/api/relay", json={"question": "x", "segment_tokens": 8},
                         headers={"host": rebound, "origin": f"http://{rebound}"})
        assert r.status_code == 403 and "--allow-host rebind.attacker.example" in r.text
        assert (await c.get("/", headers={"host": "rebind.attacker.example"})).status_code == 403
        assert backend.requests == []
        for host in ("127.0.0.1:32805", "localhost:32805", "[::1]:32805", "LOCALHOST."):
            assert (await c.get("/", headers={"host": host})).status_code == 200
        r = await c.post("/api/relay", json={"question": "q?"},
                         headers={"host": "localhost:32805", "origin": "http://localhost:32805"})
        assert r.status_code == 200 and backend.requests


@pytest.mark.parametrize("kw, host", [
    ({"allowed_hosts": ["box.lan"]}, "box.lan:8000"),
    ({"allowed_hosts": ["192.168.1.5"]}, "192.168.1.5:8000"),
    ({"allowed_origins": ["https://Proxy.Example/"]}, "proxy.example"),  # an allowed origin's host is served
    ({"allowed_hosts": ["*"]}, "anything.example"),
])
async def test_allowed_hosts_are_served(kw, host):
    app, _ = make_app(**kw)
    async with client(app, "http://127.0.0.1") as c:
        assert (await c.get("/api/config", headers={"host": host})).status_code == 200


@pytest.mark.parametrize("allowed, origin", [
    ("https://other.example:443", "https://other.example"),  # browsers omit the default port
    ("http://Other.Example:80/", "http://other.example"),
    ("https://other.example", "https://other.example:443"),
])
async def test_allowed_origins_match_with_or_without_the_default_port(allowed, origin):
    app, backend = make_app(allowed_origins=[allowed])
    async with client(app, "http://127.0.0.1:8000") as c:
        r = await c.post("/api/relay", json={"question": "q?"}, headers={"origin": origin})
    assert r.status_code == 200 and backend.requests


def test_serve_passes_the_allowed_hosts_and_the_bound_host(monkeypatch, capsys):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from cutoffmix import server

    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(server, "create_app", lambda cfg, **kw: seen.append(kw) or "app")
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    assert main(["serve", "--mock", "--host", "192.168.1.5", "--allow-host", "box.lan"]) == 0
    assert seen[-1]["allowed_hosts"] == ["box.lan", "192.168.1.5"]
    assert main(["serve", "--mock", "--host", "0.0.0.0"]) == 0
    assert seen[-1]["allowed_hosts"] == [] and "--allow-host" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# Favicon, partial usage labels
# --------------------------------------------------------------------------------------


async def test_ui_declares_an_inline_icon_so_the_browser_requests_no_favicon():
    app, _ = make_app()
    async with client(app, "http://localhost") as c:
        html = (await c.get("/")).text
    assert re.search(r'<link rel="icon" href="data:[^"]*">', html)


def run_finish(html: str, data: dict[str, Any]) -> dict[str, str]:
    """Run the UI's ``finish(data)`` in Node with a stub DOM; returns the text of every element it touched."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    start = html.index("function finish(")
    depth, end = 0, start
    for end in range(html.index("{", start), len(html)):
        depth += {"{": 1, "}": -1}.get(html[end], 0)
        if depth == 0:
            break
    stubs = ("const S = {ttft: null, committed: []}; const els = {};"
             "function $(id) { return els[id] || (els[id] = {textContent: ''}); }"
             "function setStat(id, v) { $(id).textContent = v; } function secs(x) { return x + ' s'; }"
             "function renderLegend() {} function switches() { return 0; }")
    script = (f"{stubs}\n{html[start:end + 1]}\nfinish({json.dumps(data)});\n"
              "console.log(JSON.stringify(Object.fromEntries(Object.entries(els).map(([k, v]) => [k, v.textContent]))));")
    done = subprocess.run([node, "-"], input=script, capture_output=True, text=True, timeout=30, check=True)
    return json.loads(done.stdout)


async def test_ui_labels_partial_token_totals_too():
    app, _ = make_app()
    async with client(app, "http://localhost") as c:
        html = (await c.get("/")).text
    tokens_only = {"usage_total": {"total_tokens": 8494}, "usage_complete": False, "status": "ok"}
    assert run_finish(html, tokens_only)["s-cost-k"] == "Tokens (partial)"
    assert run_finish(html, {**tokens_only, "usage_complete": True})["s-cost-k"] == "Tokens"
    assert run_finish(html, {"usage_total": {"cost": 0.15}, "usage_complete": False})["s-cost-k"] == "Cost (partial)"


def test_cli_summary_and_openai_usage_mark_partial_totals():
    pytest.importorskip("fastapi")
    from cutoffmix.server import _openai_usage

    trace = json.loads(LIVE_TRACE.read_text(encoding="utf-8"))
    assert trace["usage_complete"] is False
    summary = TerminalPrinter([], out=None).summary(trace)  # type: ignore[arg-type]
    assert f"tokens {trace['usage_total']['total_tokens']} (partial)" in summary and "(partial)" in summary
    assert _openai_usage(trace)["cutoffmix_partial"] is True
    complete = {**trace, "usage_complete": True}
    assert "partial" not in TerminalPrinter([], out=None).summary(complete)  # type: ignore[arg-type]
    assert "cutoffmix_partial" not in _openai_usage(complete)


# --------------------------------------------------------------------------------------
# CLI: bench run --batch, index show --config
# --------------------------------------------------------------------------------------


def test_bench_run_batch_is_refused_without_a_traceback(monkeypatch, capsys):
    import cutoffmix.bench.io as bench_io

    monkeypatch.setattr(bench_io, "load_events", lambda path: [])
    assert main(["bench", "run", "--events", "e", "--out", "o", "--batch", "--mock", "--probes", "mcq"]) == 2
    err = capsys.readouterr().err
    assert "Batch API" in err and "Traceback" not in err
    with pytest.raises(SystemExit):
        main(["bench", "run", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "discounted" not in help_text and "not supported" in help_text


def test_index_show_uses_the_configured_probe_preference_and_index_path(tmp_path, capsys):
    config = tmp_path / "cfg.yaml"
    config.write_text("router:\n  probe_preference: [mcq]\n", encoding="utf-8")
    assert main(["index", "show", "--config", str(config)]) == 0
    assert "probe mcq" in capsys.readouterr().out

    from cutoffmix.backends.mock import default_mock_truths
    from cutoffmix.relay import mock_truth_index

    path = tmp_path / "index.json"
    mock_truth_index(default_mock_truths(), ["claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"]).save(path)
    config.write_text(f"router:\n  index_path: {json.dumps(str(path))}\n", encoding="utf-8")
    assert main(["index", "show", "--config", str(config)]) == 0
    assert "probe current_d" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Router: trained gate is loaded, current-year questions are time-sensitive
# --------------------------------------------------------------------------------------


def test_a_trained_gate_from_router_gate_path_is_used_for_routing(tmp_path):
    from cutoffmix.router.router import Router
    from cutoffmix.router.train import build_gate_corpus

    texts, labels = build_gate_corpus([], n_synthetic=60, seed=0)
    path = tmp_path / "gate_linear.npz"
    TimeSensitivityGate.train(texts, labels, dim=2**12, epochs=2).save(path)
    cfg = load_config(overrides={"today": "2026-11-24", "router": {"gate_path": str(path)}})
    runtime = RelayRuntime(cfg, backend=FakeBackend())
    for router in (runtime.relay().router, Router.from_config(cfg)):
        assert router.gate.model is not None and router.gate.year == 2026
    assert load_config().router.gate_path is None  # the lexical gate stays the default


MOON_DE = "Welche Raumsonden oder bemannten Missionen sind 2026 zum Mond gestartet, und was war das Ergebnis?"
MOON_EN = "Which probes or crewed missions launched to the Moon in 2026, and what was the outcome?"


@pytest.mark.parametrize("question", [MOON_DE, MOON_EN, "Who won the 2025 Nobel Prize in Physics?"])
def test_questions_about_this_or_last_year_open_the_speculative_gate(question):
    gate = TimeSensitivityGate(today="2026-11-24")
    assert gate.score(question) >= RelayConfig().speculative_gate
    assert gate.cues(question)["strong_temporal"] >= 1
    # years ago, the same question is no longer about the present
    assert TimeSensitivityGate(today="2030-06-01").score(question) < RelayConfig().speculative_gate


@pytest.mark.parametrize("puzzle", ["Find the remainder when 2026^2025 is divided by 7.",
                                    "Prove that 2026 is not a perfect square."])
def test_a_year_used_as_a_number_in_a_puzzle_stays_time_insensitive(puzzle):
    assert TimeSensitivityGate(today="2026-11-24").score(puzzle) < 0.2


def test_the_runtime_gate_uses_the_configured_date():
    cfg = load_config(overrides={"today": "2027-03-01"})
    assert RelayRuntime(cfg, backend=FakeBackend()).gate.year == 2027


# --------------------------------------------------------------------------------------
# Shipped data: data/events/README.md agrees with wikidata_events.jsonl
# --------------------------------------------------------------------------------------


def test_events_readme_numbers_match_the_events_file():
    events_dir = ROOT / "data" / "events"
    rows = [json.loads(line) for line in (events_dir / "wikidata_events.jsonl").read_text("utf-8").splitlines()
            if line.strip()]
    readme = " ".join((events_dir / "README.md").read_text("utf-8").split())
    letters = collections.Counter(r["mcq_answer"] for r in rows if r.get("mcq_answer"))
    m = re.search(r"Answers are spread A (\d+), B (\d+), C (\d+), D (\d+)", readme)
    assert m is not None and [int(x) for x in m.groups()] == [letters[k] for k in "ABCD"]
    sports = sum(r["topic"] == "sports" for r in rows)
    m = re.search(r"sports makes up (\d+)% of all events \((\d+) of (\d+)\) and (\d+)% of the current-question "
                  r"events \((\d+) of (\d+)\)", readme)
    assert m is not None
    pct_all, n_sports, n_all, pct_cur, n_sports_cur, n_cur = (int(x) for x in m.groups())
    assert (n_sports, n_all) == (sports, len(rows)) and pct_all == round(100 * sports / len(rows))
    assert pct_cur == round(100 * n_sports_cur / n_cur)
