"""Tests for the cutoffmix command line (fake backend/router; real modules only where noted and available)."""

from __future__ import annotations

import functools
import io
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from cutoffmix import cli
from cutoffmix.cli import TerminalPrinter, build_parser, format_tau, main, use_color
from cutoffmix.relay import RelayRuntime
from cutoffmix.types import ModelSpec, RelayEvent, StreamEvent

SENTINEL = "<<END_STEP>>"


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeBackend:
    async def stream_continuation(self, model, req):
        if req.phase == "answer":
            text = "Der Bundeskanzler ist laut den Notizen im Amt."
        elif len(req.segments) >= 2:
            text = "Die Notizen sind vollständig. FINAL ANSWER: ok" + SENTINEL
        else:
            text = f"Schritt {len(req.segments)}: Die Quellen bestätigen das Zwischenergebnis." + SENTINEL
        for i in range(0, len(text), 8):
            yield StreamEvent(kind="text", text=text[i:i + 8])
        yield StreamEvent(kind="done", finish_reason="stop", usage={"prompt_tokens": 7, "total_tokens": 9})

    async def chat(self, model, req):  # pragma: no cover
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
        self.n = 0

    def reset(self) -> None:
        self.n = 0

    @staticmethod
    def window(question, segments, n):
        return question

    def decide(self, window_text, *, current, available=None, update=True):
        avail = list(available or self.models)
        model = avail[self.n % len(avail)]
        self.n += 1
        return FakeDecision(model, "sample", {m: 1 / len(avail) for m in avail}, {m: 27.0 for m in avail},
                            {"government": 0.9, "sports": 0.1}, 0.8)

    def decide_speculative(self, window_text, drafts, *, current):  # pragma: no cover - CLI runs are sequential
        raise NotImplementedError


def fake_runtime(cfg, *, mock=False, seed=0):
    return RelayRuntime(cfg, backend=FakeBackend(), router_factory=lambda c: FakeRouter(c.model_keys()))


MODELS = [ModelSpec(key="a", provider_id="x/a", display="Model A", color="#d97757"),
          ModelSpec(key="b", provider_id="x/b", display="Model B", color="#10a37f")]


def events_for_printer() -> list[RelayEvent]:
    return [
        RelayEvent(type="segment_start", segment=0, model="a", phase="think", decision="sample",
                   topic_posterior={"government": 0.8, "sports": 0.2}, data={"r": 0.9, "probs": {"a": 0.6}}),
        RelayEvent(type="delta", segment=0, model="a", phase="think", text="First result. Trailing par"),
        RelayEvent(type="segment_end", segment=0, model="a", phase="think", text="First result.", decision="sample",
                   data={}),
        RelayEvent(type="segment_start", segment=1, model="b", phase="think", decision="sample", data={"r": 0.5}),
        RelayEvent(type="segment_end", segment=1, model="b", phase="think",
                   data={"dropped": True, "reason": "refused", "error": None}),
        RelayEvent(type="phase", segment=1, model="b", phase="answer", decision="hold", data={"r": 0.5}),
        RelayEvent(type="delta", segment=1, model="b", phase="answer", text="The answer."),
        RelayEvent(type="segment_end", segment=1, model="b", phase="answer", text="The answer.", data={}),
        RelayEvent(type="done", data={"segments": [{"phase": "think", "model": "a", "text": "First result."}],
                                      "switches": 0, "share_by_model": {"a": 1.0},
                                      "latency": {"first_delta_s": 0.5, "total_s": 2.0},
                                      "usage_total": {"total_tokens": 99, "cost": 0.0123}}),
    ]


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [
    ["run", "Wer ist aktuell Kanzler?", "--mock", "--policy", "argmax", "--strategy", "speculative", "--segments",
     "5", "--json", "out.json", "--config", "c.yaml"],
    ["run", "q?", "--mock", "--preset", "explore", "--seed", "3"],
    ["serve", "--host", "0.0.0.0", "--port", "9000", "--mock"],
    ["serve", "--mock", "--preset", "explore", "--config", "c.yaml"],
    ["serve", "--allow-origin", "https://a.example", "--allow-origin", "https://b.example"],
    ["index", "show", "--path", "i.json", "--probe", "mcq"],
    ["bench", "import-kc", "--out", "data/kc"],
    ["bench", "build-wikidata", "--out", "wd.jsonl"],
    ["bench", "build-wikipedia", "--out", "wp.jsonl", "--start", "2026-10-01", "--end", "2026-10-07",
     "--probe-model", "deepseek/deepseek-v4-pro"],
    ["bench", "build-papers", "--out", "papers.jsonl", "--dense-from", "2025-10", "--dense-to", "2026-11",
     "--per-month", "12", "--anchor-from", "2024-01", "--anchor-to", "2025-09", "--anchor-every", "3",
     "--anchor-per-month", "6", "--interp-share", "0.5", "--controls-share", "0.1", "--seed", "0", "--meta", "m.jsonl",
     "--cache", "cache", "--candidates", "60"],
    ["bench", "run", "--events", "e.jsonl", "--out", "r.jsonl", "--models", "a", "b", "--probes", "current_d", "mcq",
     "--dry-run", "--batch", "--limit", "10"],
    ["bench", "grade", "--events", "e.jsonl", "--in", "r.jsonl", "--out", "g.jsonl"],
    ["bench", "grade", "--events", "e.jsonl", "f.jsonl", "--results", "r.jsonl", "--out", "g.jsonl", "--regrade-all",
     "--judges", "a", "--concurrency", "2", "--mock"],
    ["bench", "fit", "--events", "e.jsonl", "--results", "r.jsonl", "--out", "i.json", "--priors", "a=27.75"],
    ["router", "train"],
    ["simulate", "--out", "sim"],
])
def test_parser_accepts_every_documented_command(argv):
    args, extra = build_parser().parse_known_args(argv)
    assert callable(args.func) and extra == []


def test_unknown_arguments_are_rejected():
    with pytest.raises(SystemExit) as exc:
        main(["index", "show", "--bogus"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def test_run_prints_segments_and_writes_the_trace(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    out = tmp_path / "trace.json"
    code = main(["run", "--mock", "--today", "2026-11-24", "--json", str(out), "Wer ist aktuell Bundeskanzler?"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Claude Fable 5.2" in printed and "GPT-6.2 Sol" in printed
    assert "Schritt 0: Die Quellen bestätigen das Zwischenergebnis." in printed
    assert "Der Bundeskanzler ist laut den Notizen im Amt." in printed
    assert "segments 3" in printed and "\033[" not in printed  # captured output is not a terminal
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert trace["answer"] == "Der Bundeskanzler ist laut den Notizen im Amt." and trace["status"] == "ok"
    assert {"segments", "usage", "latency", "switches", "share_by_model"} <= set(trace)


def test_run_without_api_key_explains_what_to_do(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    code = main(["run", "Who is it?"])
    err = capsys.readouterr().err
    assert code == 2 and "OPENROUTER_API_KEY" in err and "--mock" in err


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_run_rejects_an_empty_question_without_a_traceback(monkeypatch, capsys, question):
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    code = main(["run", "--mock", question])
    err = capsys.readouterr().err
    assert code == 2 and "question must not be empty" in err and "Traceback" not in err


def test_serve_passes_the_allowed_origins_to_the_app(monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from cutoffmix import server

    seen: dict[str, Any] = {}

    def fake_create_app(cfg, *, mock=False, allowed_origins=(), allowed_hosts=(), config_path=None):
        seen.update(mock=mock, allowed_origins=list(allowed_origins), allowed_hosts=list(allowed_hosts),
                    config_path=config_path, explore=cfg.explore.enabled)
        return "app"

    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(app=app, **kw))
    assert main(["serve", "--mock", "--port", "8123", "--allow-origin", "https://notes.example"]) == 0
    assert seen["allowed_origins"] == ["https://notes.example"] and seen["app"] == "app" and seen["port"] == 8123
    assert seen["config_path"] is None and seen["explore"] is False


def test_serve_preset_loads_the_explore_config_and_passes_the_config_path(monkeypatch, tmp_path, capsys):
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("fastapi")
    from cutoffmix import server

    seen: dict[str, Any] = {}
    cfg_file = tmp_path / "my.yaml"
    cfg_file.write_text("app_title: demo\n", encoding="utf-8")

    def fake_create_app(cfg, *, mock=False, allowed_origins=(), allowed_hosts=(), config_path=None):
        seen.update(config_path=config_path, explore=cfg.explore.enabled, tokens=cfg.relay.segment_tokens,
                    title=cfg.app_title)
        return "app"

    monkeypatch.setattr(server, "create_app", fake_create_app)
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: None)
    assert main(["serve", "--mock", "--preset", "explore", "--config", str(cfg_file)]) == 0
    assert seen == {"config_path": str(cfg_file), "explore": True, "tokens": 192, "title": "demo"}
    assert "preset explore" in capsys.readouterr().err


def test_run_reports_invalid_configuration(monkeypatch, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    code = main(["run", "--mock", "--policy", "fixed:nobody", "q?"])
    assert code == 2 and "invalid configuration" in capsys.readouterr().err


def test_run_preset_is_merged_before_the_config_file(monkeypatch, tmp_path):
    import cutoffmix.config as config_mod

    seen: dict[str, Any] = {}
    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text("relay: {segment_tokens: 150}\n", encoding="utf-8")
    real = config_mod.load_config

    def spy(path=None, overrides=None, *, preset=None):
        seen.update(path=path, overrides=overrides, preset=preset)
        cfg = real(path, overrides, preset=preset)
        seen["cfg"] = cfg
        return cfg

    monkeypatch.setattr(config_mod, "load_config", spy)
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    assert main(["run", "--mock", "--preset", "explore", "--seed", "5", "--config", str(cfg_file), "q?"]) == 0
    assert seen["path"] == str(cfg_file) and seen["preset"] == "explore" and seen["overrides"] == {"router": {"seed": 5}}
    cfg = seen["cfg"]
    assert cfg.explore.enabled and cfg.relay.hard_cut_chars == 1500 and cfg.router.seed == 5
    assert cfg.relay.segment_tokens == 150  # the user's file wins over the preset
    assert cfg.router.policy == "softmax"  # the preset does not change the routing policy


def test_run_rejects_an_unknown_preset(monkeypatch, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", fake_runtime)
    code = main(["run", "--mock", "--preset", "wild", "q?"])
    err = capsys.readouterr().err
    assert code == 2 and "unknown preset 'wild'" in err and "explore" in err and "Traceback" not in err


def test_run_mock_end_to_end_with_the_real_mock_backend(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", functools.partial(RelayRuntime, mock_latency_s=0.0))
    out = tmp_path / "t.json"
    code = main(["run", "--mock", "--seed", "1", "--today", "2026-11-24", "--json", str(out),
                 "Wer ist aktuell Bundeskanzler von Deutschland?"])
    assert code == 0
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert trace["answer"] and trace["status"] == "ok"
    assert all(s["model"] in ("claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon") for s in trace["segments"])
    assert "answer" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------------------


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


def test_use_color_respects_no_color_and_terminals():
    assert use_color(Tty(), env={}) is True
    assert use_color(Tty(), env={"NO_COLOR": "1"}) is False
    assert use_color(Tty(), env={"NO_COLOR": ""}) is True
    assert use_color(io.StringIO(), env={}) is False


def test_printer_plain_output_prints_committed_text_only():
    out = io.StringIO()
    printer = TerminalPrinter(MODELS, out=out)
    for ev in events_for_printer():
        printer.handle(ev)
    text = out.getvalue()
    assert "[Model A] sample · r=0.90 · government 80%" in text
    assert "First result." in text and "Trailing par" not in text
    assert "↳ Model B refused - re-routing" in text
    assert "[Model B] answer · hold" in text and "The answer." in text
    assert "segments 1 · switches 0 · share Model A 100% · first token 0.50s · total 2.00s · tokens 99" in text
    assert "cost $0.0123" in text and "\033" not in text


def test_printer_colours_each_model_and_redraws_corrected_segments():
    out = Tty()
    printer = TerminalPrinter(MODELS, out=out, color=True, live=True, width=80)
    for ev in events_for_printer():
        printer.handle(ev)
    text = out.getvalue()
    assert "\033[38;2;217;119;87m" in text  # model A's text colour
    assert "\033[48;2;16;163;127;" in text  # model B's chip background
    assert "Trailing par" in text  # streamed live ...
    assert "\033[J" in text  # ... then erased and redrawn with the committed text
    assert text.count("First result.") == 2


def explore_events() -> list[RelayEvent]:
    """A run under the explore preset (DESIGN.md §7.6 shapes): a lens note, its check, a forced continuation and
    the explore metrics; the last segment carries the badge keys flat, which the printer tolerates too."""
    start = dict(phase="think", decision="sample", topic_posterior={"science_research": 0.7})
    return [
        RelayEvent(type="segment_start", segment=2, model="a", data={"explore": {"kind": "lens", "field": "music theory",
                                                                                 "tier": "far", "cue": None}}, **start),
        RelayEvent(type="segment_end", segment=2, model="a", phase="think", text="Lens (music theory): a mapping.",
                   decision="sample", data={"explore": {"kind": "lens", "lens_prefix_ok": True}}),
        RelayEvent(type="segment_start", segment=3, model="b", data={"explore": {"kind": "check", "field": "music theory",
                                                                                 "tier": None, "cue": None}}, **start),
        RelayEvent(type="segment_end", segment=3, model="b", phase="think", text="Lens kept: it holds.",
                   decision="sample", data={"explore": {"kind": "check", "lens_verdict": "kept"}}),
        RelayEvent(type="segment_start", segment=4, model="a", data={"explore": {"kind": "continue", "field": None,
                                                                                 "tier": None, "cue": "Alternatively, "}},
                   **start),
        RelayEvent(type="segment_end", segment=4, model="a", phase="think", text="Alternatively, no.",
                   decision="sample", data={}),
        RelayEvent(type="segment_start", segment=5, model="b", data={"kind": "main", "field": None}, **start),
        RelayEvent(type="segment_end", segment=5, model="b", phase="think", text="Claim: yes.", decision="sample",
                   data={}),
        RelayEvent(type="done", data={"segments": [{"phase": "think"}] * 4, "switches": 3, "share_by_model": {},
                                      "latency": {}, "usage_total": {},
                                      "explore": {"topics_visited": {"all": 3, "main_line": 2}, "topic_switches": 2,
                                                  "lens": {"offered": 2, "written": 2, "kept": 1, "dropped": 1,
                                                           "ignored": 0, "unverified": 0,
                                                           "fields": ["music theory", "ecology"], "keep_rate": 0.5},
                                                  "labels": {"counts": {"Claim": 4},
                                                             "coverage": {"two_or_more_items": 0.83,
                                                                          "where_we_stand": 1.0}},
                                                  "forced_continuations": 1, "premature_finals": [{"index": 4}],
                                                  "seed": 5, "seed_source": "run"}}),
    ]


def test_printer_tags_lens_check_and_continuation_segments():
    out = io.StringIO()
    printer = TerminalPrinter(MODELS, out=out)
    for ev in explore_events():
        printer.handle(ev)
    text = out.getvalue()
    assert "[Model A] sample · science_research 70% [lens: music theory (far)]" in text
    assert "[Model B] sample · science_research 70% [check: music theory]" in text
    assert "[Model A] sample · science_research 70% [continue: 'Alternatively,']" in text
    plain = next(line for line in text.splitlines() if line.startswith("[Model B] sample") and "Claim" not in line
                 and "check" not in line)
    assert "[" not in plain[len("[Model B]"):]  # a main-line note has no tag
    assert "segments 4 · switches 3 · lenses 2 (1 kept, 1 dropped) · labels 83% · topics 3 · forced 1" in text


def test_printer_colours_the_explore_tag():
    out = Tty()
    printer = TerminalPrinter(MODELS, out=out, color=True, live=True, width=120)
    for ev in explore_events()[:2]:
        printer.handle(ev)
    assert "\033[1m[lens: music theory (far)]\033[0m" in out.getvalue()


@pytest.mark.parametrize("data, tag", [
    ({"explore": {"kind": "lens", "field": "ecology", "tier": "near", "cue": None}}, "lens: ecology (near)"),
    ({"explore": {"kind": "lens", "field": None, "tier": None}}, "lens: ?"),
    ({"explore": {"kind": "check", "field": None}}, "check: the previous lens"),
    ({"explore": {"kind": "continue", "cue": "  Perhaps the missing piece is  "}}, "continue: 'Perhaps the missing piece is'"),
    ({"explore": {"kind": "continue"}}, "continue"),
    ({"explore": {"kind": "main"}}, ""),
    ({"kind": "check", "field": "ecology"}, "check: ecology"),  # flat keys are accepted too
    ({"explore": "odd", "kind": "lens", "field": "x"}, "lens: x"),
    ({}, ""),
])
def test_explore_tag_shapes(data, tag):
    assert TerminalPrinter.explore_tag(data) == tag


@pytest.mark.parametrize("explore, expected", [
    ({}, []),
    ({"lens": {"kept": 2, "dropped": 0}}, ["lenses 2 (2 kept, 0 dropped)"]),
    ({"lens": {"written": 1}, "labels": {"coverage": {"two_or_more_items": 0.9, "where_we_stand": 1.0}}},
     ["lenses 1", "labels 90%"]),
    ({"topics_visited": ["a", "b"], "forced_continuations": 0}, ["topics 2", "forced 0"]),
    ({"topics_visited": {"all": 3, "main_line": 1}, "premature_finals": [{}, {}]}, ["topics 3"]),
    ({"topics_visited": 4.0, "labels": "n/a", "lens": "none"}, ["topics 4"]),
    ({"lens": {"kept": True}}, []),
    ({"forced_continuations": 1, "wws": {"missing": 0, "cut": 2, "waived": 1, "waived_at": [8]}},
     ["forced 1", "wws 2 cut, 1 waived"]),
    ({"wws": {"missing": 3, "cut": 0, "waived": 0}}, ["wws 3 missing"]),
    ({"wws": {"missing": 0, "cut": 0, "waived": 0, "waived_at": []}}, []),  # nothing to report
    ({"wws": "n/a"}, []),
])
def test_explore_summary_is_defensive_about_shapes(explore, expected):
    assert cli.explore_summary(explore) == expected


def test_format_tau():
    assert format_tau(27.75) == "2026-04.8"
    assert format_tau(27.96) == "2026-05.0"
    assert format_tau(34.0) == "2026-11.0"
    assert format_tau(float("nan")) == "n/a" and format_tau(None) == "n/a"


# --------------------------------------------------------------------------------------
# index / bench / router / simulate (real modules where they exist, patched heavy work)
# --------------------------------------------------------------------------------------


def _truth_index():
    from cutoffmix.backends.mock import default_mock_truths
    from cutoffmix.relay import mock_truth_index

    return mock_truth_index(default_mock_truths(), ["claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"])


def test_index_show_prints_the_topic_by_model_table(tmp_path, capsys):
    path = tmp_path / "index.json"
    _truth_index().save(path)
    assert main(["index", "show", "--path", str(path)]) == 0
    text = capsys.readouterr().out
    assert "probe current_d" in text and "ai_ml" in text and "gpt-6.2-sol" in text
    ai_line = next(line for line in text.splitlines() if line.startswith("ai_ml"))
    assert "2026-08.0" in ai_line and ai_line.count("*") == 1  # Fable's reversal on AI is marked best


def test_index_show_without_an_index(monkeypatch, capsys):
    import cutoffmix.router.index as index_mod

    monkeypatch.setattr(index_mod, "load_index", lambda path=None: None)
    assert main(["index", "show"]) == 1
    assert "no cutoff index" in capsys.readouterr().err


def test_router_train_passes_its_options_through(monkeypatch):
    import cutoffmix.router.train as train_mod

    seen: list[list[str]] = []
    monkeypatch.setattr(train_mod, "main", lambda argv: seen.append(list(argv)) or 0)
    assert main(["router", "train", "--events", "e.jsonl", "--epochs", "3"]) == 0
    assert seen == [["--events", "e.jsonl", "--epochs", "3"]]


def test_bench_fit_parses_priors_and_saves_the_index(monkeypatch, tmp_path, capsys):
    import cutoffmix.bench.io as bench_io
    from cutoffmix.bench import estimate

    calls: dict[str, Any] = {}

    def fake_fit(events, results, **kw):
        calls.update(kw)
        return _truth_index()

    monkeypatch.setattr(estimate, "fit_index", fake_fit)
    monkeypatch.setattr(bench_io, "load_events", lambda path: [])
    monkeypatch.setattr(bench_io, "load_results", lambda path: [])
    out = tmp_path / "idx.json"
    code = main(["bench", "fit", "--events", "e.jsonl", "--results", "r.jsonl", "--out", str(out), "--priors",
                 "gpt-6.2-sol=27.75", "claude-fable-5.2=2026-02", "--bootstrap", "0", "--models", "gpt-6.2-sol",
                 "claude-fable-5.2", "--probes", "current", "mcq", "--note", "prior only"])
    assert code == 0 and out.exists()
    assert calls["priors"] == {"gpt-6.2-sol": 27.75, "claude-fable-5.2": "2026-02"} and calls["bootstrap"] == 0
    assert calls["models"] == ["gpt-6.2-sol", "claude-fable-5.2"] and calls["probes"] == ["current", "mcq"]
    assert calls["meta"] == {"note": "prior only"}
    assert "index ->" in capsys.readouterr().out


def test_bench_fit_rejects_malformed_priors(monkeypatch, capsys):
    import cutoffmix.bench.io as bench_io

    monkeypatch.setattr(bench_io, "load_events", lambda path: [])
    monkeypatch.setattr(bench_io, "load_results", lambda path: [])
    code = main(["bench", "fit", "--events", "e", "--results", "r", "--out", "o", "--priors", "nonsense"])
    assert code == 2 and "KEY=VALUE" in capsys.readouterr().err


def test_bench_run_dry_run_prints_the_estimate(monkeypatch, capsys):
    import cutoffmix.bench.io as bench_io
    from cutoffmix.bench import runner

    events = [SimpleNamespace(id=f"e{i}") for i in range(5)]
    seen: dict[str, Any] = {}

    def fake_estimate(models, evs, probes, *, batch=False):
        seen.update(models=[m.key for m in models], n=len(evs), probes=list(probes), batch=batch)
        return {"calls": 3, "total_usd": 0.5}

    monkeypatch.setattr(bench_io, "load_events", lambda path: events)
    monkeypatch.setattr(runner, "estimate_cost", fake_estimate)
    code = main(["bench", "run", "--events", "e.jsonl", "--out", "r.jsonl", "--models", "gpt-6.2-sol",
                 "deepseek/deepseek-v4-pro", "--probes", "mcq", "--dry-run", "--batch", "--limit", "2"])
    assert code == 0
    assert seen == {"models": ["gpt-6.2-sol", "deepseek/deepseek-v4-pro"], "n": 2, "probes": ["mcq"], "batch": True}
    assert json.loads(capsys.readouterr().out)["total_usd"] == 0.5


def test_bench_run_dry_run_passes_token_options_only_when_given(monkeypatch, capsys):
    import cutoffmix.bench.io as bench_io
    from cutoffmix.bench import runner

    seen: dict[str, Any] = {}

    def fake_estimate(models, evs, probes, *, batch=False, **kw):
        seen.update(kw)
        return {"calls": 1, "total_usd": 0.1}

    monkeypatch.setattr(bench_io, "load_events", lambda path: [SimpleNamespace(id="e0")])
    monkeypatch.setattr(runner, "estimate_cost", fake_estimate)
    base = ["bench", "run", "--events", "e.jsonl", "--out", "r.jsonl", "--models", "gpt-6.2-sol", "--probes", "mcq",
            "--dry-run"]
    assert main(base) == 0 and seen == {}
    assert main([*base, "--out-tokens", "claude-fable-5.2=350", "gemini-4-neon=900", "60", "--in-tokens", "110",
                 "--max-tokens", "1500", "--out-tokens-measured"]) == 0
    assert seen == {"out_tokens": {"claude-fable-5.2": 350, "gemini-4-neon": 900, "default": 60}, "in_tokens": 110,
                    "max_tokens": 1500, "measured": True}
    capsys.readouterr()
    assert main([*base, "--out-tokens", "fable=lots"]) != 0 and "KEY=N" in capsys.readouterr().err
    assert main([*base, "--out-tokens", "-5"]) != 0 and ">= 0" in capsys.readouterr().err


def test_bench_run_passes_max_tokens_to_run_probes(monkeypatch, tmp_path):
    from cutoffmix.bench import runner
    from cutoffmix.bench.io import save_events
    from cutoffmix.types import Event

    seen: dict[str, Any] = {}

    async def fake_run(backend, models, events, probes, out, **kw):
        seen.update(kw, n=len(events), probes=list(probes))
        return []

    monkeypatch.setattr(runner, "run_probes", fake_run)
    events_path = tmp_path / "e.jsonl"
    save_events(events_path, [Event(id="x", date="2026-01-05", month="2026-01", topic="ai_ml", category="paper",
                                    subject="s", fact="f", mcq_question="q?", mcq_choices=["A) a", "B) b", "C) c",
                                                                                           "D) d"], mcq_answer="A")])
    base = ["bench", "run", "--events", str(events_path), "--out", str(tmp_path / "r.jsonl"), "--models",
            "gpt-6.2-sol", "--probes", "mcq", "--mock"]
    assert main(base) == 0 and "max_tokens" not in seen and seen["n"] == 1  # the runner's own default applies
    assert main([*base, "--max-tokens", "1500"]) == 0 and seen["max_tokens"] == 1500


def _pilot_fixture(tmp_path):
    from cutoffmix.bench.io import save_events, save_results
    from cutoffmix.types import Event, ProbeResult

    fake = Event(id="fakepaper:1", date="2024-01-18", month="2024-01", topic="ai_ml", category="fake_paper",
                 subject="Efficient Temporal Adaptation Modeling through Language-Conditioned Video Generation",
                 fact="control: no such arXiv paper exists",
                 question_current="Who is the first author of the arXiv paper titled 'Efficient Temporal Adaptation "
                                  "Modeling through Language-Conditioned Video Generation'?", control=True)
    rows = [
        ProbeResult("fakepaper:1", "gpt-6.2-sol", "current_d", "The first author is **Sho Maeoki**.", "ungraded",
                    latency_s=0.7, usage={"prompt_tokens": 100, "completion_tokens": 12, "cost": 0.001}),
        ProbeResult("fakepaper:1", "gpt-6.2-sol", "current_d", "I'm not aware of this paper.", "correct", latency_s=0.4,
                    usage={"prompt_tokens": 100, "completion_tokens": 9}),
        ProbeResult("fakepaper:1", "gemini-4-neon", "current_d", "", "ungraded", error="timeout"),
    ]
    events, results = tmp_path / "events.jsonl", tmp_path / "pilot.jsonl"
    save_events(events, [fake])
    save_results(results, rows)
    return events, results, rows


def test_bench_grade_regrades_offline_into_a_new_file_and_prints_before_and_after(tmp_path, capsys):
    from cutoffmix.bench.io import load_results

    events, src, rows = _pilot_fixture(tmp_path)
    before = src.read_text(encoding="utf-8")
    out = tmp_path / "regraded.jsonl"
    code = main(["bench", "grade", "--events", str(events), "--in", str(src), "--out", str(out)])
    assert code == 0 and src.read_text(encoding="utf-8") == before  # never in place
    graded = load_results(out)
    assert [r.grade for r in graded] == ["incorrect", "correct", "ungraded"]
    assert (graded[0].grader, graded[0].latency_s, graded[0].usage) == ("rule", 0.7, rows[0].usage)
    assert graded[1] == rows[1] and graded[2] == rows[2]  # untouched rows, an API error among them
    text = capsys.readouterr().out
    lines = {tuple(line.split()[:2]): line.split()[2:] for line in text.splitlines() if line.startswith("  ")}
    assert lines[("model", "probe")] == ["correct", "incorrect", "ungraded", "changed"]
    assert lines[("gpt-6.2-sol", "current_d")] == ["1", "0->1", "1->0", "1"]
    assert lines[("gpt-6.2-sol", "all")] == ["1", "0->1", "1->0", "1"]
    assert lines[("gemini-4-neon", "all")] == ["0", "0", "1", "0"]
    assert f"3 results (1 changed) -> {out}" in text
    again = tmp_path / "again.jsonl"  # --results is the old spelling of --in; --regrade-all keeps sure grades
    assert main(["bench", "grade", "--events", str(events), "--results", str(out), "--out", str(again),
                 "--regrade-all"]) == 0
    assert [r.grade for r in load_results(again)] == ["incorrect", "correct", "ungraded"]
    assert "3 results (0 changed)" in capsys.readouterr().out


def test_bench_grade_refuses_to_regrade_in_place(tmp_path, capsys):
    events, src, _ = _pilot_fixture(tmp_path)
    before = src.read_text(encoding="utf-8")
    code = main(["bench", "grade", "--events", str(events), "--in", str(src), "--out", str(src)])
    assert code == 2 and "never re-graded in place" in capsys.readouterr().err
    assert src.read_text(encoding="utf-8") == before
    with pytest.raises(SystemExit):  # --out is required
        main(["bench", "grade", "--events", str(events), "--in", str(src)])


def test_bench_run_requires_a_key_unless_mocked(monkeypatch, capsys):
    import cutoffmix.bench.io as bench_io

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(bench_io, "load_events", lambda path: [])
    assert main(["bench", "run", "--events", "e", "--out", "o"]) == 2
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err


def test_simulate_writes_events_results_index_and_report(tmp_path, capsys):
    out = tmp_path / "sim"
    code = main(["simulate", "--out", str(out), "--from", "2026-01", "--to", "2026-06", "--per-cell", "1",
                 "--probes", "mcq", "--bootstrap", "0"])
    assert code == 0
    assert {p.name for p in out.iterdir()} == {"events.jsonl", "results.jsonl", "index.json", "recovery.json"}
    report = json.loads((out / "recovery.json").read_text(encoding="utf-8"))
    assert "mcq" in report and "rmse" in report["mcq"]
    assert "RMSE" in capsys.readouterr().out
