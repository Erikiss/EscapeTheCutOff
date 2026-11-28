"""The rally preset's surface (DESIGN_rally.md section 7): the simulated models writing rally notes, the CLI
(``--preset rally``, ``--ring``, segment tags and the summary), the server (``cutoffmix-rally``, the ``ring`` option)
and the UI markup. Everything runs offline against the mock backend."""

from __future__ import annotations

import functools
import json
from typing import Any

import pytest

from cutoffmix import cli
from cutoffmix.backends.mock import MockBackend, default_mock_truths
from cutoffmix.cli import main, rally_note_tag, rally_summary
from cutoffmix.config import load_config
from cutoffmix.rally import month_number, parse_note
from cutoffmix.relay import RelayRuntime

TODAY = "2026-11-26"
FABLE, SOL, NEON = "claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"
QUESTIONS = ("What is the latest state of AI models?", "Why does ice float on water?",
             "Wer gewann zuletzt die Wahl und was ist neu in der Wissenschaft?")


async def _run(ring: list[str], question: str, seed: int = 0) -> dict[str, Any]:
    cfg = load_config(preset="rally", overrides={"rally": {"ring": ring}, "today": TODAY, "router": {"seed": seed}})
    runtime = RelayRuntime(cfg, backend=MockBackend(default_mock_truths(latency_s=0), today=TODAY, seed=seed),
                           mock=True, seed=seed)
    done: dict[str, Any] | None = None
    async for event in runtime.relay().run(question):
        if event.type == "done":
            done = event.data
    assert done is not None
    return done


# ---- the simulated models -----------------------------------------------------------------------------------


@pytest.mark.parametrize("ring", [[FABLE, NEON], [NEON, FABLE], [FABLE, NEON, SOL], [FABLE, SOL]])
@pytest.mark.parametrize("question", QUESTIONS)
@pytest.mark.parametrize("seed", [0, 1])
async def test_mock_rally_notes_follow_the_ring_parse_and_converge(ring, question, seed):
    done = await _run(ring, question, seed)
    rally = done["rally"]
    assert done["status"] == "ok" and rally["stop_reason"] == "converged" and 2 * len(ring) <= rally["turns"] <= 8
    notes = [s for s in done["segments"] if s["phase"] == "think"]
    assert [s["model"] for s in notes] == [ring[i % len(ring)] for i in range(len(notes))]       # strict ring order
    assert [n["model"] for n in rally["notes"]] == [s["model"] for s in notes]
    for i, segment in enumerate(notes):
        parsed = parse_note(segment["text"], TODAY)
        lines = segment["text"].rstrip().splitlines()
        assert "FINAL ANSWER" not in segment["text"] and lines[-1].startswith("Newest I know:")
        assert parsed["newest"] is not None and not parsed["future"] and segment["rally"]["newest"] == parsed["newest"]
        if i == 0:
            assert parsed["labels"]["known"] == 3 and segment["rally"]["kind"] == "open"
        else:
            labels = parsed["labels"]
            assert segment["rally"]["kind"] == "reply"
            assert labels["update"] or labels["newer"] or labels["nothing_newer"]
    assert rally["notes"][-1]["quiet"] and rally["quiet_streak"] >= len(ring)
    frontier = rally["frontier"]
    assert frontier["self_reported"] is True and len(frontier["by_turn"]) == rally["turns"]
    months = [month_number(m) for m in frontier["by_turn"]]
    assert months == sorted(months)                                                  # a running maximum never falls
    answer = done["answer"]
    assert "(single source" in answer and "Basis:" in answer and "Not resolved:" in answer


async def test_the_exchange_can_lift_the_frontier_by_a_month():
    rally = (await _run([FABLE, SOL], QUESTIONS[1]))["rally"]
    assert rally["frontier"]["gain_months"] == 1 and rally["turns"] == 6
    assert rally["adds_by_model"][SOL]["update"] == 1 and rally["adds_by_model"][FABLE]["newer"] >= 1
    kinds = [n["labels"]["newer"] for n in rally["notes"]]
    assert kinds[1] and kinds[2] and kinds[3]                                       # newer items passed back and forth


async def test_mock_rally_is_deterministic_per_seed():
    a, b = await _run([FABLE, NEON], QUESTIONS[0], 3), await _run([FABLE, NEON], QUESTIONS[0], 3)
    assert [s["text"] for s in a["segments"]] == [s["text"] for s in b["segments"]] and a["answer"] == b["answer"]


async def test_plain_mock_runs_have_no_rally_part():
    cfg = load_config(overrides={"today": TODAY})
    runtime = RelayRuntime(cfg, backend=MockBackend(default_mock_truths(latency_s=0), today=TODAY), mock=True)
    done = None
    async for event in runtime.relay().run(QUESTIONS[0]):
        if event.type == "done":
            done = event.data
    assert done is not None and "rally" not in done and all(s.get("rally") is None for s in done["segments"])


# ---- CLI ----------------------------------------------------------------------------------------------------


@pytest.fixture
def fast_cli(monkeypatch):
    monkeypatch.setattr(cli, "RelayRuntime", functools.partial(RelayRuntime, mock_latency_s=0.0))


def test_cli_run_mock_with_the_rally_preset(fast_cli, tmp_path, capsys):
    out = tmp_path / "rally.json"
    code = main(["run", "--mock", "--preset", "rally", "--seed", "1", "--today", TODAY, "--json", str(out),
                 QUESTIONS[0]])
    printed = capsys.readouterr().out
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert code == 0 and trace["status"] == "ok"
    rally = trace["rally"]
    assert rally["preset"] == "rally" and rally["ring"] == [FABLE, NEON] and rally["stop_reason"] == "converged"
    assert "[rally: open · 3 known · newest " in printed and "[rally: reply · " in printed  # counts in the headers
    assert "rally · " in printed and "converged" in printed and "ring Claude Fable 5.2 → Gemini 4 Neon" in printed
    assert "frontier (self-reported" in printed and "Gemini 4 Neon: " in printed


def test_cli_ring_switches_the_rally_on_and_sets_the_order(fast_cli, tmp_path, capsys):
    out = tmp_path / "ring.json"
    code = main(["run", "--mock", "--ring", f"{FABLE},{SOL},{NEON}", "--today", TODAY, "--json", str(out),
                 QUESTIONS[1]])
    capsys.readouterr()
    rally = json.loads(out.read_text(encoding="utf-8"))["rally"]
    assert code == 0 and rally["ring"] == [FABLE, SOL, NEON] and rally["preset"] == "rally"
    assert [n["model"] for n in rally["notes"][:3]] == [FABLE, SOL, NEON]


@pytest.mark.parametrize("args, message", [
    (["--ring", FABLE], "two or three model keys"),
    (["--ring", f"{FABLE},"], "two or three model keys"),
    (["--ring", "nope,nada"], "unknown model"),
    (["--ring", f"{FABLE},{FABLE}"], "must not repeat"),
    (["--ring", f"{FABLE},{NEON}", "--preset", "explore"], "cannot be combined with --preset explore"),
])
def test_cli_ring_errors_are_user_messages(args, message, capsys):
    assert main(["run", "--mock", *args, "question"]) == 2
    assert message in capsys.readouterr().err


def test_rally_and_explore_do_not_combine():
    from cutoffmix.config import ConfigError

    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(preset="rally", overrides={"explore": {"enabled": True}})


def test_note_tag_and_summary_helpers():
    note = {"kind": "reply", "labels": {"update": 1, "newer": 2, "confirm": 1}, "newest": "2026-03", "quiet": False}
    assert rally_note_tag(note) == "reply · +1 update · +2 newer · 1 confirm · newest 2026-03"
    quiet = {"kind": "reply", "labels": {"confirm": 1, "nothing_newer": 1}, "newest": None, "quiet": True}
    assert rally_note_tag(quiet) == "reply · 1 confirm · nothing newer · quiet · newest not stated"
    opening = {"kind": "open", "labels": {"known": 3}, "newest": "2026-08"}
    assert rally_note_tag(opening) == "open · 3 known · newest 2026-08"
    assert rally_note_tag({}) == "note · newest not stated"
    lines = rally_summary({"ring": [FABLE, NEON], "turns": 4, "stop_reason": "converged",
                           "frontier": {"by_turn": ["2026-03", None, "2026-05"], "gain_months": 2},
                           "adds_by_model": {FABLE: {"update": 1, "newer": 0}, NEON: {}}})
    assert lines[0] == f"rally · 4 notes · converged · ring {FABLE} → {NEON}"
    assert "self-reported" in lines[1] and "2026-03 → ? → 2026-05" in lines[1] and "gain +2 months" in lines[1]
    assert lines[2] == f"  {FABLE}: 1 update" and lines[3] == f"  {NEON}: no labelled lines"
    assert rally_summary({})[0] == "rally · 0 notes · ended"
    # the live probe of 2026-11-28: the opening model wrote only 'Known:' lines, which are no reply labels
    opened = rally_summary({"ring": [FABLE, NEON], "turns": 2, "stop_reason": "max_turns",
                            "notes": [{"model": FABLE, "labels": {"known": 5, "confirm": 0}},
                                      {"model": NEON, "labels": {"known": 0, "confirm": 2, "newer": 1}}],
                            "adds_by_model": {FABLE: {"confirm": 0, "newer": 0}, NEON: {"confirm": 2, "newer": 1}}})
    assert opened[-2] == f"  {FABLE}: 5 known" and opened[-1] == f"  {NEON}: 1 newer, 2 confirm"


# ---- server -------------------------------------------------------------------------------------------------


@pytest.fixture
def rally_server():
    pytest.importorskip("fastapi")
    from cutoffmix.server import create_app

    cfg = load_config(overrides={"today": TODAY})
    backend = MockBackend(default_mock_truths(latency_s=0), today=TODAY, seed=1)
    return create_app(cfg, backend=backend, mock=True, allowed_hosts=("test",))


def _client(app):
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_server_lists_the_rally_preset_and_model_id(rally_server):
    async with _client(rally_server) as c:
        config = (await c.get("/api/config")).json()
        models = (await c.get("/v1/models")).json()
    assert {"name": "rally", "segment_tokens": 160, "max_segments": 12} in config["presets"]
    assert "cutoffmix-rally" in config["model_ids"] and "cutoffmix-rally" in [m["id"] for m in models["data"]]


async def test_server_openai_rally_model_id_stream_and_non_stream(rally_server):
    from tests.test_server import sse_payloads

    body = {"model": "cutoffmix-rally", "messages": [{"role": "user", "content": QUESTIONS[0]}]}
    async with _client(rally_server) as c:
        plain = await c.post("/v1/chat/completions", json=body)
        streamed = await c.post("/v1/chat/completions", json={**body, "stream": True})
    assert plain.status_code == 200 and streamed.status_code == 200
    rally = plain.json()["cutoffmix"]["rally"]
    assert rally["stop_reason"] == "converged" and rally["ring"] == [FABLE, NEON] and rally["preset"] == "rally"
    assert "(single source" in plain.json()["choices"][0]["message"]["content"]
    chunks = [ch for ch in sse_payloads(streamed.text) if ch != "[DONE]"]
    assert chunks[-1]["cutoffmix"]["rally"]["frontier"]["self_reported"] is True


async def test_server_relay_takes_a_ring_and_selects_the_rally_preset(rally_server):
    from tests.test_server import sse_payloads

    async with _client(rally_server) as c:
        response = await c.post("/api/relay", json={"question": QUESTIONS[1], "ring": [FABLE, SOL]})
    assert response.status_code == 200
    events = sse_payloads(response.text)
    done = events[-1]["data"]
    assert events[-1]["type"] == "done" and done["rally"]["ring"] == [FABLE, SOL]
    starts = [e for e in events if e["type"] == "segment_start" and e["phase"] == "think"]
    assert [e["model"] for e in starts[:4]] == [FABLE, SOL, FABLE, SOL] and starts[0]["data"]["rally"]["kind"] == "open"
    ends = [e for e in events if e["type"] == "segment_end" and e["phase"] == "think"]
    assert ends[0]["data"]["rally"]["labels"]["known"] == 3


@pytest.mark.parametrize("extra", [
    {"ring": [FABLE]}, {"ring": "x"}, {"ring": [FABLE, FABLE]}, {"ring": ["nope", NEON]}, {"ring": [FABLE, 3]},
    {"ring": [FABLE, NEON], "preset": "explore"},
])
async def test_server_refuses_a_bad_ring(rally_server, extra):
    async with _client(rally_server) as c:
        response = await c.post("/api/relay", json={"question": "q", **extra})
        openai = await c.post("/v1/chat/completions", json={
            "model": "cutoffmix", "messages": [{"role": "user", "content": "q"}], **extra})
    assert response.status_code == 400 and "ring" in response.json()["detail"]
    assert openai.status_code == 400 and "ring" in openai.json()["error"]["message"]


async def test_server_ring_with_the_explore_model_id_is_refused(rally_server):
    async with _client(rally_server) as c:
        response = await c.post("/v1/chat/completions", json={"model": "cutoffmix-explore", "ring": [FABLE, NEON],
                                                              "messages": [{"role": "user", "content": "q"}]})
    assert response.status_code == 400 and "mutually exclusive" in response.json()["error"]["message"]


# ---- UI -----------------------------------------------------------------------------------------------------


async def test_ui_has_the_rally_strip_and_badge(rally_server):
    async with _client(rally_server) as c:
        html = (await c.get("/")).text
    for needle in ('id="rally"', 'id="r-chart"', "function renderRally", "function rallyBadge", "function rallyChart",
                   "renderRally((ev.data || {}).rally)", "Self-reported", "badge x rally",
                   "rally: two or three models"):
        assert needle in html, needle
    assert 'id="rally" hidden' in html                                              # only shown when done.rally exists
    assert "grid-template-columns: minmax(0, 1fr)" in html  # a long badge cannot widen the stream on a phone
