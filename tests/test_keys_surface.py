"""Surface of the key-not-value experiment (DESIGN_keys.md): the mock's keyed answers, the `bench keys` CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from cutoffmix.backends.mock import FULL_COPY, KEY_DARK_RECALL, KEY_DARK_WINDOW_MONTHS, MockBackend, default_mock_truths
from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.cli import main
from cutoffmix.timeutil import parse_fractional_month
from cutoffmix.types import ChatRequest, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events" / "papers_events.jsonl"
RESULTS = ROOT / "data" / "results" / "papers_probes_2026-11-25.jsonl"
NEON = ModelSpec(key="gemini-4-neon", provider_id="google/gemini-4-neon")


@pytest.fixture(scope="module")
def events():
    return load_events(EVENTS)


def _ask(backend, event, arm=None, donor_answer=None, model=NEON):
    meta = {"event_id": event.id, "probe": "current_d", "model": model.key}
    if arm is not None:
        meta |= {"arm": arm, "set": "dark", "donor": "claude-fable-5.2", "item": f"x-{event.id}-{arm}"}
    if donor_answer is not None:
        meta["donor_answer"] = donor_answer
    req = ChatRequest(messages=[{"role": "user", "content": event.question_current}], max_tokens=50,
                      temperature=0.0, metadata=meta)
    return asyncio.run(backend.chat(model, req))


def _backend(events):
    return MockBackend(default_mock_truths(latency_s=0.0), events={e.id: e for e in events}, seed=0, today="2026-11-27")


# ---- mock ----------------------------------------------------------------------------------------------------------


def test_unkeyed_and_alone_probes_are_the_ordinary_answer(events):
    backend = _backend(events)
    for event in [e for e in events if not e.control][:40]:
        plain, alone = _ask(backend, event), _ask(backend, event, "alone")
        assert plain.text == alone.text and plain.raw == alone.raw
        assert set(plain.raw) == {"mock", "model", "event_id", "probe", "outcome", "p_correct"}   # unchanged shape


def test_keyed_answers_are_deterministic(events):
    for event in [e for e in events if not e.control][:30]:
        for arm in ("key", "full", "wrong_key"):
            a, b = _ask(_backend(events), event, arm), _ask(_backend(events), event, arm)
            assert a.text == b.text and a.raw == b.raw
            assert a.raw["arm"] == arm and a.raw["keyed"] in ("own", "dark_memory", "copied")


def test_wrong_key_is_the_ordinary_answer(events):
    backend = _backend(events)
    for event in [e for e in events if not e.control]:
        alone, wrong = _ask(backend, event), _ask(backend, event, "wrong_key")
        assert wrong.text == alone.text and wrong.raw["outcome"] == alone.raw["outcome"]
        assert wrong.raw["keyed"] == "own"


def test_a_key_recovers_some_misses_just_past_the_cutoff_and_nothing_else(events):
    backend = _backend(events)
    tau = backend.truths[NEON.key].tau_for("ai_ml")
    inside, recovered, outside_changed, broken = 0, 0, 0, 0
    for event in [e for e in events if not e.control]:
        alone, key = _ask(backend, event), _ask(backend, event, "key")
        t = parse_fractional_month(event.date)
        if alone.raw["outcome"] == "correct":
            broken += key.text != alone.text
            continue
        if tau < t <= tau + KEY_DARK_WINDOW_MONTHS:
            inside += 1
            recovered += key.raw["outcome"] == "correct"
            if key.raw["outcome"] == "correct":
                assert key.raw["keyed"] == "dark_memory" and key.text.rstrip(".") == event.expected_current
        else:
            outside_changed += key.text != alone.text
    assert inside >= 10 and 0 < recovered < inside                      # some, not all (p = KEY_DARK_RECALL)
    assert abs(recovered / inside - KEY_DARK_RECALL) < 0.3
    assert outside_changed == 0 and broken == 0                         # a key never adds anything else


def test_a_full_note_is_copied_with_the_donors_name(events):
    backend = _backend(events)
    real = [e for e in events if not e.control]
    copied = [_ask(backend, e, "full", donor_answer="Zed Quux") for e in real]
    share = sum(r.text == "Zed Quux." for r in copied) / len(copied)
    assert abs(share - FULL_COPY) < 0.1
    assert all(r.raw["outcome"] == "incorrect" for r in copied if r.raw["keyed"] == "copied")
    right = [(e, _ask(backend, e, "full")) for e in real]                 # a right donor: the right name is copied
    assert all(r.text.rstrip(".") == e.expected_current and r.raw["outcome"] == "correct"
               for e, r in right if r.raw["keyed"] == "copied")


# ---- CLI -------------------------------------------------------------------------------------------------------------

@pytest.fixture
def keys():
    return pytest.importorskip("cutoffmix.bench.keys", reason="cutoffmix/bench/keys.py (the experiment core) is missing")


def _plan(tmp_path, capsys, *extra):
    out = tmp_path / "plan.jsonl"
    code = main(["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(out),
                 "--today", "2026-11-27", *extra])
    printed = capsys.readouterr().out
    assert code == 0, printed
    return out, printed


def test_plan_on_the_stored_paper_results(tmp_path, capsys, keys):
    out, printed = _plan(tmp_path, capsys)
    rows = read_jsonl(out)
    assert rows and "nothing was sent" in printed and "cost_estimate" in printed and "total_usd" in printed
    dark: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        if row["set"] == "dark":
            dark.setdefault((row["donor"], row["recipient"]), set()).add(row["event_id"])
    from cutoffmix.bench.io import load_results
    _, summary = keys.build_plan(load_events(EVENTS), load_results(RESULTS), today="2026-11-27")
    for (donor, recipient), candidates in ((("claude-fable-5.2", "gemini-4-neon"), 50),
                                           (("claude-fable-5.2", "gpt-6.2-sol"), 96),
                                           (("gemini-4-neon", "gpt-6.2-sol"), 47)):
        info = summary["pairs"][f"{donor}->{recipient}"]
        assert info["dark_candidates"] == candidates
        assert len(dark[(donor, recipient)]) == info["dark"] >= candidates - info["leaks"]
    assert ("gemini-4-neon", "claude-fable-5.2") not in dark                  # 1 item: below --min-items
    for row in rows:
        assert row["arm"] in keys.ARMS_BY_SET[row["set"]]
        assert row["messages"][0]["role"] == "system" and "2026-11-27" in row["messages"][0]["content"]
    again, _ = _plan(tmp_path / "again", capsys)
    assert read_jsonl(again) == rows                                            # deterministic


def test_plan_refuses_unknown_models_and_bad_pairs(tmp_path, capsys, keys):
    out = tmp_path / "plan.jsonl"
    assert main(["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(out),
                 "--models", "claude-fable-5.2,nope"]) != 0
    assert "unknown model" in capsys.readouterr().err
    assert main(["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(out),
                 "--pairs", "gemini-4-neon:nope"]) != 0
    with pytest.raises(SystemExit):
        main(["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(out),
              "--pairs", "gemini-4-neon"])
    assert not out.exists()


def test_run_mock_then_report_end_to_end(tmp_path, capsys, keys):
    plan, _ = _plan(tmp_path, capsys, "--pairs", "claude-fable-5.2:gemini-4-neon")
    results = tmp_path / "results.jsonl"
    code = main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results),
                 "--mock", "--limit", "60"])
    out = capsys.readouterr().out
    assert code == 0 and "simulated models" in out
    rows = read_jsonl(results)
    assert len(rows) == 60 and all(r["grade"] for r in rows)
    # resumable: a second run skips the 60 answered ids and sends the next 60 (no id twice)
    assert main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results),
                 "--mock", "--limit", "60"]) == 0
    capsys.readouterr()
    ids = [r["id"] for r in read_jsonl(results)]
    assert len(ids) == 120 and len(set(ids)) == 120
    report_json, report_md = tmp_path / "report.json", tmp_path / "report.md"
    code = main(["bench", "keys", "report", "--plan", str(plan), "--results", str(results), "--json", str(report_json),
                 "--md", str(report_md)])
    text = capsys.readouterr().out
    assert code == 0
    for arm in ("alone", "key", "full", "wrong_key"):
        assert arm in text
    assert "predict" in text.lower()
    assert json.loads(report_json.read_text("utf-8"))
    assert report_md.read_text("utf-8").strip() == text.strip()


def test_run_with_the_real_models_needs_a_key_and_a_cost_cap(tmp_path, capsys, monkeypatch, keys):
    plan, _ = _plan(tmp_path, capsys, "--pairs", "gemini-4-neon:gpt-6.2-sol")
    results = tmp_path / "results.jsonl"
    argv = ["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results)]
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(argv) != 0
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    assert main(argv) != 0                                                     # refused before any call
    assert "--max-cost" in capsys.readouterr().err
    assert main([*argv, "--max-cost", "0"]) != 0
    assert "--max-cost" in capsys.readouterr().err
    assert not results.exists()
