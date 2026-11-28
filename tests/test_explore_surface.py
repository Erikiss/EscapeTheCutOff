"""End-to-end tests of the explore preset's surface: the simulated models under ``--preset explore`` through the CLI
and the server (DESIGN.md §7.7), including the relay's explore state machine (lens/check/continue directives, the
record gate) and the metrics in ``done``.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any

import pytest

from cutoffmix import cli
from cutoffmix.backends.mock import MockBackend, default_mock_truths
from cutoffmix.cli import main
from cutoffmix.config import load_config
from cutoffmix.relay import RelayRuntime

QUESTION = "Who is the minister of finance, and since when?"
TODAY = "2026-11-24"


def _explore(trace: dict[str, Any]) -> dict[str, Any]:
    """The §7 metrics of a ``done`` payload / trace, which every preset run must carry with the preset's name."""
    explore = trace.get("explore")
    assert isinstance(explore, dict), "the done payload of a preset run carries the explore metrics"
    assert explore["preset"] == "explore" and explore["config"]["enabled"] is True
    return explore


def test_cli_run_mock_with_the_explore_preset(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", functools.partial(RelayRuntime, mock_latency_s=0.0))
    out = tmp_path / "explore.json"
    code = main(["run", "--mock", "--preset", "explore", "--seed", "3", "--today", TODAY, "--policy", "round_robin",
                 "--json", str(out), QUESTION])
    printed = capsys.readouterr().out
    assert code == 0
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert trace["status"] == "ok" and trace["answer"]
    thinks = [s for s in trace["segments"] if s["phase"] == "think"]
    assert thinks and all("explore" in s for s in thinks)  # Segment.explore exists (None while the preset is off)
    assert "segments" in printed and "answer" in printed
    explore = _explore(trace)
    # the mock wrote structured notes under the explore tail: labels, 'Where we stand:', a lens and its check
    texts = [s["text"] for s in thinks if s["text"]]
    assert sum("Where we stand:" in t for t in texts) >= len(texts) - 2
    assert any(t.startswith("Lens (") for t in texts) and any(t.startswith(("Lens kept:", "Lens dropped:"))
                                                              for t in texts)
    assert any(s.get("explore", {}).get("kind") == "continue" for s in thinks)  # the early conclusion was set aside
    assert "Basis:" in trace["answer"] and "Not resolved:" in trace["answer"]
    assert explore["seed"] == 3 and isinstance(explore.get("lens"), dict) and explore["lens"]["offered"] >= 1
    assert explore["forced_continuations"] >= 1 and explore["config"]["enabled"] is True
    assert "lenses" in printed and "labels" in printed  # the summary line carries the metrics
    assert re.search(r"\[lens: [a-z ]+ \((near|mid|far)\)\]", printed) and "[check: " in printed


def test_cli_run_mock_without_the_preset_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "RelayRuntime", functools.partial(RelayRuntime, mock_latency_s=0.0))
    out = tmp_path / "default.json"
    assert main(["run", "--mock", "--seed", "3", "--today", TODAY, "--json", str(out), QUESTION]) == 0
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert trace["status"] == "ok" and trace.get("explore") is None
    texts = [s["text"] for s in trace["segments"] if s["phase"] == "think"]
    assert texts and not any("Where we stand:" in t or t.startswith("Lens (") for t in texts)
    assert all(s.get("explore") is None for s in trace["segments"])


@pytest.fixture
def explore_server():
    pytest.importorskip("fastapi")
    from cutoffmix.server import create_app

    cfg = load_config(overrides={"today": TODAY})
    backend = MockBackend(default_mock_truths(latency_s=0), today=TODAY, seed=3)
    return create_app(cfg, backend=backend, mock=True, allowed_hosts=("test",))


async def test_server_mock_relay_with_the_preset_override(explore_server):
    import httpx

    from tests.test_server import sse_payloads

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=explore_server), base_url="http://test") as c:
        r = await c.post("/api/relay", json={"question": QUESTION, "preset": "explore", "policy": "round_robin"})
        assert r.status_code == 200
        events = sse_payloads(r.text)
        done = events[-1]["data"]
        assert events[-1]["type"] == "done" and done["status"] == "ok"
        models = {e["model"] for e in events if e["type"] == "segment_start"}
        assert models <= {"claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"} and len(models) == 3
        cfg = (await c.get("/api/config")).json()
        assert cfg["mock"] is True and cfg["defaults"]["preset"] is None
        assert cfg["presets"] == [{"name": "explore", "segment_tokens": 192, "max_segments": 16},
                              {"name": "rally", "segment_tokens": 160, "max_segments": 12}]
    explore = _explore(done)
    assert explore["config"]["enabled"] is True and explore["lens"]["offered"] >= 1
    kinds = [e["data"].get("explore", {}).get("kind") for e in events if e["type"] == "segment_start"]
    assert "lens" in kinds and "check" in kinds
    assert any("Where we stand:" in s["text"] for s in done["segments"] if s["phase"] == "think")


async def test_server_openai_explore_model_id_end_to_end(explore_server):
    import httpx

    from tests.test_server import sse_payloads

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=explore_server), base_url="http://test") as c:
        r = await c.post("/v1/chat/completions", json={"model": "cutoffmix-explore", "stream": True,
                                                       "messages": [{"role": "user", "content": QUESTION}]})
        assert r.status_code == 200
        chunks = [ch for ch in sse_payloads(r.text) if ch != "[DONE]"]
        assert chunks[-1]["model"] == "cutoffmix-explore" and chunks[-1]["cutoffmix"]["status"] == "ok"
        content = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks if ch["choices"])
        assert content.startswith("Based on the shared notes")
    explore = _explore(chunks[-1]["cutoffmix"])
    assert isinstance(explore.get("labels"), dict) and "Basis:" in content
