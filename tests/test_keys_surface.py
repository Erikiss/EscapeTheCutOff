"""Surface of the key-not-value experiment (DESIGN_keys.md): the mock's keyed answers, the `bench keys` CLI."""

from __future__ import annotations

import asyncio
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from cutoffmix.backends.mock import FULL_COPY, KEY_DARK_RECALL, KEY_DARK_WINDOW_MONTHS, MockBackend, default_mock_truths
from cutoffmix.bench.grading import parse_judge_output
from cutoffmix.bench.io import load_events, read_jsonl, write_jsonl
from cutoffmix.cli import main
from cutoffmix.timeutil import parse_fractional_month
from cutoffmix.types import ChatRequest, ChatResult, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events" / "papers_events.jsonl"
RESULTS = ROOT / "data" / "results" / "papers_probes_2026-11-25.jsonl"
NEON = ModelSpec(key="gemini-4-neon", provider_id="google/gemini-4-neon")


@pytest.fixture(scope="module")
def events():
    return load_events(EVENTS)


def _ask(backend, event, arm=None, donor_answer=None, model=NEON, set_="dark"):
    meta = {"event_id": event.id, "probe": "current_d", "model": model.key}
    if arm is not None:
        meta |= {"arm": arm, "set": set_, "donor": "claude-fable-5.2", "item": f"x-{event.id}-{arm}"}
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


def test_blank_and_the_both_right_alone_arm_are_the_ordinary_answer(events):
    backend = _backend(events)                                          # arms outside KEY_ARMS: the plain probe
    for event in [e for e in events if not e.control][:60]:
        plain = _ask(backend, event)
        for arm, set_ in (("blank", "dark"), ("alone", "both_right")):
            other = _ask(backend, event, arm, set_=set_)
            assert other.text == plain.text and other.raw == plain.raw


def test_judge_requests_get_no_scripted_answer(events):
    from cutoffmix.bench.keys.run import parse_adoption_vote

    backend = _backend(events)                                          # so a mock run with judges decides nothing
    for purpose in ("judge", "adoption_judge"):
        req = ChatRequest(messages=[{"role": "user", "content": "Does the answer endorse X?"}], max_tokens=50,
                          temperature=0.0, metadata={"purpose": purpose, "item": "x"})
        reply = asyncio.run(backend.chat(NEON, req))
        assert reply.raw["outcome"] == "unscripted"
        assert parse_judge_output(reply.text) is None and parse_adoption_vote(reply.text) is None


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


PAIR = "claude-fable-5.2:gemini-4-neon"


class Gateway:
    """Stands in for OpenRouter (no network): answers every call with ``text`` at ``cost`` USD and keeps the
    requests."""

    def __init__(self, text: str = "I don't know.", cost: float = 0.01):
        self.text, self.cost, self.calls = text, cost, []

    async def chat(self, model, req):
        self.calls.append((model.key, req))
        return ChatResult(text=self.text, usage={"cost": self.cost}, finish_reason="stop", latency_s=0.1)

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        raise NotImplementedError


class CountingMock:
    """The real mock backend, with its requests kept."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    async def chat(self, model, req):
        self.calls.append((model.key, req))
        return await self.inner.chat(model, req)

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        return self.inner.stream_continuation(model, req)


@pytest.fixture
def backends(monkeypatch):
    """``make_backend`` without the network: the real path gets :class:`Gateway` (a dummy key is set), ``--mock`` the
    mock backend wrapped in :class:`CountingMock` (``made.mock`` is the latest)."""
    import cutoffmix.backends as mod

    real_make = mod.make_backend
    made = SimpleNamespace(gateway=Gateway(), mock=None)

    def make(cfg, *, mock=False, events=None, **kw):
        if not mock:
            return made.gateway
        made.mock = CountingMock(real_make(cfg, mock=True, events=events, **kw))
        return made.mock

    monkeypatch.setattr(mod, "make_backend", make)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    return made


def _run(plan, out, *extra):
    return main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out), *extra])


def _one_line(err: str) -> bool:
    return err.startswith("cutoffmix: ") and err.count("\n") == 1 and "Traceback" not in err


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
        funnel = info["funnel"]["dark"]
        assert info["dark_candidates"] == funnel["candidates"] == candidates
        assert candidates == funnel["leaks"] + funnel["no_cue"] + funnel["no_wrong_key"] + funnel["kept"]
        if info["keyed"]:
            assert len(dark[(donor, recipient)]) == info["dark"] == funnel["kept"] >= 5
        else:                                                                   # too few items survive the filters
            assert (donor, recipient) not in dark and info["dark"] == 0 and funnel["kept"] < 5
    # Neon answers in one sentence: masked, its notes carry no cue, so Neon->Sol is unkeyed (design section 2)
    assert summary["pairs"]["gemini-4-neon->gpt-6.2-sol"]["keyed"] is False
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
    planned = len(read_jsonl(plan))
    results = tmp_path / "results.jsonl"
    code = main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(results),
                 "--mock", "--limit", "60"])
    captured = capsys.readouterr()
    out, err = captured.out, captured.err
    assert code == 0 and "simulated models" in out                             # a --limit stop is no failure
    assert f"{planned} planned calls: 0 already answered, {planned} without a current answer; sending 60 now " \
           "(--limit 60)" in out
    assert "sent 60 calls, graded 0 stored answers again · cost $0.0000 (recipients $0.0000 + judges $0.0000" in out
    assert err.endswith(f"\r  60/60\nstopped at --limit 60: {planned - 60} of {planned} planned calls not sent; "
                        "rerun to send the next ones\n")
    rows = read_jsonl(results)
    assert len(rows) == 60 and all(r["grade"] for r in rows) and {r["backend"] for r in rows} == {"mock"}
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


# ---- CLI: provenance, safety, cost (design section 6) ----------------------------------------------------------------


def test_mock_and_real_rows_never_share_a_results_file(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    mock_out = tmp_path / "keys_results_mock.jsonl"
    assert _run(plan, mock_out, "--mock", "--limit", "30") == 0
    capsys.readouterr()
    before = mock_out.read_bytes()
    code = _run(plan, mock_out, "--max-cost", "1")                              # the real models on the mock's file
    err = capsys.readouterr().err
    assert code == 2 and _one_line(err) and "30 row(s) from another backend (mock: 30)" in err and "--out" in err
    assert backends.gateway.calls == [] and mock_out.read_bytes() == before
    real_out = tmp_path / "keys_results.jsonl"
    assert _run(plan, real_out, "--max-cost", "1", "--limit", "5") == 0
    capsys.readouterr()
    assert len(backends.gateway.calls) == 5 and {r["backend"] for r in read_jsonl(real_out)} == {"openrouter"}
    before = real_out.read_bytes()
    code = _run(plan, real_out, "--mock")                                       # and the mock on the real file
    err = capsys.readouterr().err
    assert code == 2 and _one_line(err) and "(openrouter: 5)" in err
    assert backends.mock.calls == [] and real_out.read_bytes() == before


@pytest.mark.parametrize("kind", ["probe_results", "events", "plan"])
def test_run_refuses_an_out_that_is_not_a_keys_results_file(tmp_path, capsys, keys, backends, kind):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    if kind == "plan":
        target = plan
    else:
        target = tmp_path / f"{kind}_copy.jsonl"
        shutil.copy(RESULTS if kind == "probe_results" else EVENTS, target)
    before = target.read_bytes()
    code = _run(plan, target, "--mock")
    err = capsys.readouterr().err
    assert code == 2 and _one_line(err) and f"{target.name}:1 is not a 'bench keys run' result row" in err
    assert target.read_bytes() == before and backends.mock.calls == []


def test_a_cap_stop_says_so_and_exits_3(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    planned = len(read_jsonl(plan))
    results = tmp_path / "keys_results.jsonl"
    code = _run(plan, results, "--max-cost", "0.025", "--concurrency", "1")    # 0.01 per call: 3 calls
    captured = capsys.readouterr()
    assert code == 3 and len(backends.gateway.calls) == 3
    assert f"sending {planned} now; cost so far $0.0000 (openrouter)" in captured.out
    assert "3 rows -> " in captured.out and "cost $0.0300 (recipients $0.0300 + judges $0.0000" in captured.out
    assert re.search(r"\r  3/\d+\nstopped at --max-cost \$0\.025: " + f"{planned - 3} of {planned} planned calls not "
                     r"sent; rerun with a higher --max-cost to resume\n$", captured.err)
    model, req = backends.gateway.calls[0]
    assert model == "gemini-4-neon" and req.max_tokens == 4096 and req.temperature == 0.0
    assert _run(plan, results, "--max-cost", "0.025") == 3                     # the stored rows count: nothing sent
    assert len(backends.gateway.calls) == 3
    capsys.readouterr()
    assert _run(plan, results, "--max-cost", "100") == 0 and len(backends.gateway.calls) == planned
    out = capsys.readouterr().out
    assert f"{planned} planned calls: 3 already answered" in out


def test_run_errors_are_one_line_without_a_traceback(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    rows = read_jsonl(plan)
    write_jsonl(plan, [*rows, rows[0]])                                          # a repeated id
    assert _run(plan, tmp_path / "keys_results_mock.jsonl", "--mock") == 2
    err = capsys.readouterr().err
    assert _one_line(err) and err.startswith(f"cutoffmix: --plan {plan}: the plan repeats 1 id")
    assert backends.mock is None                                                # refused before a backend is made
    assert _run(tmp_path / "missing_plan.jsonl", tmp_path / "x.jsonl", "--mock") == 2
    assert _one_line(capsys.readouterr().err)


def test_run_refuses_stale_rows_unless_resend_stale(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    planned = len(read_jsonl(plan))
    results = tmp_path / "keys_results_mock.jsonl"
    assert _run(plan, results, "--mock", "--limit", "10") == 0
    _plan(tmp_path, capsys, "--pairs", PAIR, "--today", "2026-11-28", "--force")   # every prompt changes
    before = results.read_bytes()
    code = _run(plan, results, "--mock")
    err = capsys.readouterr().err
    assert code == 2 and _one_line(err) and "--resend-stale" in err and "10 of the" in err
    assert backends.mock.calls == [] and results.read_bytes() == before
    assert _run(plan, results, "--mock", "--resend-stale", "--limit", "5") == 0
    out = capsys.readouterr().out
    assert f"0 already answered, {planned} without a current answer (10 answered under another request); " \
           "sending 5 now (--limit 5)" in out
    assert len(backends.mock.calls) == 5


def test_plan_does_not_replace_a_plan_with_other_messages_without_force(tmp_path, capsys, keys):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    before = plan.read_bytes()
    argv = ["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(plan),
            "--pairs", PAIR]
    assert main([*argv, "--today", "2026-11-28"]) == 2
    err = capsys.readouterr().err
    assert _one_line(err) and "carry other messages" in err and "--force" in err and plan.read_bytes() == before
    _plan(tmp_path, capsys, "--pairs", PAIR)                                    # the same messages: replaced
    assert plan.read_bytes() == before
    assert main([*argv, "--today", "2026-11-28", "--force"]) == 0
    capsys.readouterr()
    assert plan.read_bytes() != before
    stored = tmp_path / "results_copy.jsonl"                                    # not a plan at all
    shutil.copy(RESULTS, stored)
    copy_before = stored.read_bytes()
    assert main([*argv[:8], str(stored), "--pairs", PAIR, "--today", "2026-11-27"]) == 2
    err = capsys.readouterr().err
    assert _one_line(err) and "is not a plan" in err and stored.read_bytes() == copy_before


# ---- CLI: files of the wrong kind, an --out the run cannot use, answers whose grading failed ------------------------


def test_plan_force_never_replaces_a_file_that_is_not_a_plan(tmp_path, capsys, keys):
    results = tmp_path / "papers_probes_copy.jsonl"                             # the stored probe results (paid)
    shutil.copy(RESULTS, results)
    events = tmp_path / "events_copy.jsonl"
    shutil.copy(EVENTS, events)
    plain = tmp_path / "other_results.jsonl"
    shutil.copy(RESULTS, plain)
    folder = tmp_path / "plans"
    folder.mkdir()
    before = {p: p.read_bytes() for p in (results, events, plain)}

    def plan(out, *extra):
        return main(["bench", "keys", "plan", "--events", str(events), "--results", str(results), "--out", str(out),
                     "--pairs", PAIR, "--today", "2026-11-27", *extra])
    for out, needle in ((results, "is the --results file"), (events, "is the --events file"),
                        (plain, "is not a plan (row 1 has no id or messages)"), (folder, "is not a regular file")):
        for extra in ([], ["--force"]):
            assert plan(out, *extra) == 2
            err = capsys.readouterr().err
            assert _one_line(err) and needle in err, (out, extra, err)
    assert all(p.read_bytes() == data for p, data in before.items()) and list(folder.iterdir()) == []
    new = tmp_path / "plan.jsonl"
    assert plan(new) == 0 and plan(new, "--seed", "1", "--force") == 0          # an earlier plan: --force replaces it
    capsys.readouterr()


def test_plan_refuses_a_plan_whose_donor_answers_differ_without_force(tmp_path, capsys, keys):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    rows = read_jsonl(plan)
    i = next(n for n, r in enumerate(rows) if r["set"] == "donor_wrong")
    original = rows[i]["donor_answer"]
    rows[i] = {**rows[i], "donor_answer": "Some Other Name"}                    # an earlier name rule, same messages
    write_jsonl(plan, rows)
    before = plan.read_bytes()
    argv = ["bench", "keys", "plan", "--events", str(EVENTS), "--results", str(RESULTS), "--out", str(plan),
            "--pairs", PAIR, "--today", "2026-11-27"]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert _one_line(err) and f"1 of its {len(rows)} ids carry other messages or another donor answer (1 only the " \
        "donor answer" in err and "--force" in err and plan.read_bytes() == before
    assert main([*argv, "--force"]) == 0
    capsys.readouterr()
    assert read_jsonl(plan)[i]["donor_answer"] == original


def test_a_wrong_or_missing_input_file_is_a_one_line_error(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    new_plan = tmp_path / "p.jsonl"

    def keys_plan(events, results):
        return ["bench", "keys", "plan", "--events", str(events), "--results", str(results), "--out", str(new_plan),
                "--pairs", PAIR, "--today", "2026-11-27"]

    def keys_run(plan_file, events):
        return ["bench", "keys", "run", "--plan", str(plan_file), "--events", str(events), "--out",
                str(tmp_path / "keys_results_mock.jsonl"), "--mock"]
    missing = tmp_path / "missing.jsonl"
    cases = [
        (keys_plan(EVENTS, missing), f"--results {missing}: no such file"),
        (keys_plan(EVENTS, EVENTS), f"--results {EVENTS} is not a probe results file (TypeError: "),
        (keys_plan(RESULTS, RESULTS), f"--events {RESULTS} is not an events file (TypeError: "),
        (keys_plan(missing, RESULTS), f"--events {missing}: no such file"),
        (keys_run(plan, plan), f"--events {plan} is not an events file (TypeError: "),
        (keys_run(plan, RESULTS), f"--events {RESULTS} is not an events file"),
        (keys_run(EVENTS, EVENTS), f"--plan {EVENTS}: plan row 1 has no set, donor, recipient, event_id, arm, "
                                   "messages"),
        (keys_run(RESULTS, EVENTS), f"--plan {RESULTS}: plan row 1 has no id, set, donor, recipient, arm, messages"),
        (keys_run(missing, EVENTS), f"--plan {missing}: no such file"),
    ]
    for argv, needle in cases:
        assert main(argv) == 2, argv
        err = capsys.readouterr().err
        assert _one_line(err) and needle in err, (argv, err)
    assert backends.mock is None and not new_plan.exists() and not (tmp_path / "keys_results_mock.jsonl").exists()


def test_run_with_an_out_it_cannot_write_sends_nothing(tmp_path, capsys, keys, backends):
    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    folder = tmp_path / "keys_results.jsonl"
    folder.mkdir()                                                              # e.g. --out runs
    code = _run(plan, folder, "--max-cost", "1")
    captured = capsys.readouterr()
    assert code == 2 and _one_line(captured.err) and "cannot write --out" in captured.err
    assert "not a regular file" in captured.err and "planned calls:" not in captured.out
    under_a_file = plan / "keys_results.jsonl"                                  # its folder is a file: an OSError
    code = _run(plan, under_a_file, "--max-cost", "1")
    captured = capsys.readouterr()
    assert code == 2 and _one_line(captured.err) and "Traceback" not in captured.err
    assert backends.gateway.calls == []


def test_run_says_which_answers_could_not_be_graded(tmp_path, capsys, keys, backends, monkeypatch):
    from cutoffmix.bench.grading import grade_paper
    from cutoffmix.bench.keys import text as text_mod

    plan, _ = _plan(tmp_path, capsys, "--pairs", PAIR)
    rows = read_jsonl(plan)
    real = text_mod.adoption_label

    def broken(response, name, event):
        raise ValueError("labelling broke")
    monkeypatch.setattr(text_mod, "adoption_label", broken)
    results = tmp_path / "keys_results_mock.jsonl"
    assert _run(plan, results, "--mock") == 0                                   # the run goes on: nothing is lost
    captured = capsys.readouterr()
    assert len(backends.mock.calls) == len(rows)
    # labelled, so failed: every adoption row, and every full row of dark and both_right the rules grade correct
    by_id = {e.id: e for e in load_events(EVENTS)}
    stored = {r["id"]: r for r in read_jsonl(results)}
    adoption = {r["id"] for r in rows if r["set"] in ("donor_wrong", "control")}
    right_name = {r["id"] for r in rows if r["set"] in ("dark", "both_right") and r["arm"] == "full"
                  and (grade_paper(by_id[r["event_id"]], "current_d", stored[r["id"]]["response"]) or ("",))[0]
                  == "correct"}
    failed = adoption | right_name
    assert adoption and right_name and {i for i, r in stored.items() if r["error"]} == failed
    assert captured.err.endswith(f"\n{len(failed)} answers could not be graded (ValueError: labelling broke); they "
                                 "are kept and counted in the cost, and the next run grades them again from the "
                                 "stored responses, without a new call\n")
    monkeypatch.setattr(text_mod, "adoption_label", real)
    assert _run(plan, results, "--mock") == 0
    captured = capsys.readouterr()
    assert backends.mock.calls == [] and "could not be graded" not in captured.err
    assert f"{len(rows)} planned calls: {len(rows)} already answered, 0 without a current answer; sending 0 now; " \
           f"grading {len(failed)} stored answers again first, without a new call ({len(failed)} whose grading " \
           "failed)" in captured.out
    assert f"sent 0 calls, graded {len(failed)} stored answers again" in captured.out
    labelled = {r["id"]: r for r in read_jsonl(results)}
    assert all(labelled[i]["adoption"] == "endorsed" and labelled[i]["grade"] == "correct" for i in right_name)
