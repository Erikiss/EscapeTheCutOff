"""Key, not value (docs/DESIGN_keys.md section 6, Safety, Resume and Cost): the robustness of 'bench keys run' and
'bench keys report'. The report checks its inputs and never writes over them (B); a run keeps a complete last row
without its newline and cuts only a torn one, with a warning (C); failed calls and failed judge calls are reported, and
an error no retry can fix stops the run (D); a completion that failed inside an HTTP 200, or came back empty, is sent
again (E); the first Ctrl-C lets the calls in flight finish and write their rows (F); a wrong key never equals its key
on letters and digits (G)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from keys_testlib import EVENTS, FABLE, NEON, RESULTS, ROOT, TODAY, paper, result
from test_keys_run import MODELS, ScriptedBackend, fixture_plan, run
from test_keys_run_judges import J1, UNDECIDED

from cutoffmix.backends.base import BackendError
from cutoffmix.backends.openrouter import OpenRouterBackend
from cutoffmix.bench.io import load_events, load_results, read_jsonl, write_jsonl
from cutoffmix.bench.keys import build_plan, run_plan
from cutoffmix.cli import main
from cutoffmix.types import ChatResult, ModelSpec

# ---- helpers ------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fable_neon(tmp_path_factory):
    """The Fable to Neon calls of the real plan (design section 9), written once: ``(path, rows)``."""
    rows, _ = build_plan(load_events(EVENTS), load_results(RESULTS), pairs=[(FABLE, NEON)], today=TODAY)
    path = tmp_path_factory.mktemp("plan") / "keys_plan.jsonl"
    write_jsonl(path, rows)
    return path, rows


@pytest.fixture(scope="module")
def answered(tmp_path_factory, fable_neon):
    """A results file of 'bench keys run' with the first 10 of those calls answered."""
    path = tmp_path_factory.mktemp("results") / "keys_results.jsonl"
    neon = ModelSpec(key=NEON, provider_id="google/gemini-4-neon")
    asyncio.run(run_plan(ScriptedBackend(), [neon], fable_neon[1][:10], path, events=load_events(EVENTS)))
    return path


def first_calls(tmp_path: Path, fable_neon, n: int) -> Path:
    """A plan of the first ``n`` of those calls (a plan, as 'bench keys plan' writes one)."""
    path = tmp_path / "plan.jsonl"
    write_jsonl(path, fable_neon[1][:n])
    return path


@pytest.fixture
def gateway(monkeypatch):
    """``make_backend`` without the network: a run of the real models gets ``made.backend`` (a dummy key is set)."""
    import cutoffmix.backends as mod

    made = SimpleNamespace(backend=ScriptedBackend())
    monkeypatch.setattr(mod, "make_backend", lambda cfg, **kw: made.backend)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-used")
    return made


def _one_line(err: str) -> bool:
    return err.startswith("cutoffmix: ") and err.count("\n") == 1 and "Traceback" not in err


def _report(plan, results, *extra) -> int:
    return main(["bench", "keys", "report", "--plan", str(plan), "--results", str(results), *extra])


def _run(plan, out, *extra) -> int:
    return main(["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out),
                 "--max-cost", "5", *extra])


def _recipient_calls(backend) -> list[str]:
    return [req.metadata["item"] for _, req in backend.calls if not req.metadata.get("purpose")]


# ---- B: the report's inputs and outputs ---------------------------------------------------------------------------


def test_report_inputs_are_checked_and_a_wrong_file_is_a_one_line_error(tmp_path, capsys, fable_neon, answered):
    plan = fable_neon[0]
    results = tmp_path / "keys_results.jsonl"
    results.touch()                                                             # a report before the first answer
    assert _report(plan, results) == 0 and "Planned calls 231" in capsys.readouterr().out
    missing, folder = tmp_path / "missing.jsonl", tmp_path / "folder"
    folder.mkdir()
    not_json = tmp_path / "notes.jsonl"
    not_json.write_text("this is not json\n", encoding="utf-8")
    empty_plan = tmp_path / "empty_plan.jsonl"
    empty_plan.touch()
    run_rows = "is not a results file of 'bench keys run' (ValueError: line 1 is not a 'bench keys run' result row"
    cases = [
        ((plan, missing), f"--results {missing}: no such file"),
        ((missing, results), f"--plan {missing}: no such file"),
        ((plan, folder), f"--results {folder}: cannot read it"),
        ((plan, not_json), (f"--results {not_json} is not a results file of 'bench keys run' (ValueError: line 1: "
                            "invalid JSON")),
        ((EVENTS, results), f"--plan {EVENTS}: plan row 1 has no set, donor, recipient, event_id, arm, messages"),
        ((RESULTS, results), f"--plan {RESULTS}: plan row 1 has no id, set, donor, recipient, arm, messages"),
        ((answered, plan), f"--plan {answered}: plan row 1 has no messages"),            # plan and results swapped
        ((empty_plan, results), f"the plan {empty_plan} has no calls"),
        ((plan, RESULTS), f"--results {RESULTS} {run_rows} (no id, arm; has probe, model))"),   # the probe results
        ((plan, EVENTS), f"--results {EVENTS} {run_rows} (no arm, grade))"),
        ((plan, plan), f"--results {plan} {run_rows} (no grade; has messages))"),
    ]
    target = tmp_path / "report.json"
    for (plan_file, results_file), needle in cases:
        assert _report(plan_file, results_file, "--json", str(target)) == 2, (plan_file, results_file)
        captured = capsys.readouterr()
        assert _one_line(captured.err) and needle in captured.err, (plan_file, results_file, captured.err)
        assert captured.out == ""                                               # no report on a file of another kind
    assert not target.exists()


def test_report_never_writes_over_its_inputs(tmp_path, capsys, fable_neon, answered):
    plan, results = tmp_path / "keys_plan.jsonl", tmp_path / "keys_results.jsonl"
    shutil.copy(fable_neon[0], plan)
    shutil.copy(answered, results)
    before = {p: p.read_bytes() for p in (plan, results)}
    link = tmp_path / "report.json"
    link.symlink_to(results)                                                    # a symlink to the results ...
    hard = tmp_path / "report.md"
    os.link(plan, hard)                                                         # ... and a hard link to the plan
    (tmp_path / "sub").mkdir()
    cases = [
        (["--json", str(results)], f"--json {results} is the --results file: write the report to a file of its own"),
        (["--md", str(results)], f"--md {results} is the --results file"),
        (["--json", str(plan)], f"--json {plan} is the --plan file"),
        (["--md", str(plan)], f"--md {plan} is the --plan file"),
        (["--json", str(link)], f"--json {link} is the --results file"),
        (["--md", str(hard)], f"--md {hard} is the --plan file"),
        (["--md", str(tmp_path / "sub" / ".." / "keys_results.jsonl")], "is the --results file"),
        (["--json", str(tmp_path / "new.out"), "--md", str(tmp_path / "sub" / ".." / "new.out")],
         f"--json {tmp_path / 'new.out'} and --md {tmp_path / 'sub' / '..' / 'new.out'} are the same file"),
    ]
    for extra, needle in cases:
        assert _report(plan, results, *extra) == 2, extra
        captured = capsys.readouterr()
        assert _one_line(captured.err) and needle in captured.err, (extra, captured.err)
        assert captured.out == ""                                               # refused before anything is written
        assert all(p.read_bytes() == data for p, data in before.items())
    assert not (tmp_path / "new.out").exists()
    json_out, md_out = tmp_path / "out" / "report.json", tmp_path / "out" / "report.md"
    assert _report(plan, results, "--json", str(json_out), "--md", str(md_out)) == 0   # two files of their own
    text = capsys.readouterr().out
    assert json.loads(json_out.read_text("utf-8"))["totals"]["planned"] == 231
    assert md_out.read_text("utf-8").strip() == text.strip()
    assert _report(plan, results, "--json", str(tmp_path / "sub")) == 2         # a directory: one line, no traceback
    assert _one_line(capsys.readouterr().err)
    assert all(p.read_bytes() == data for p, data in before.items())


# ---- C: the last line of the results file -------------------------------------------------------------------------


def test_a_complete_last_row_without_its_newline_is_kept(tmp_path, caplog):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows[:3], out, events)
    data = out.read_bytes().rstrip(b"\n")
    out.write_bytes(data)                                                       # e.g. a tool that drops the newline
    backend, starts = ScriptedBackend(), []
    with caplog.at_level(logging.WARNING):
        got = run(backend, rows[:4], out, events, on_start=starts.append)
    assert _recipient_calls(backend) == [rows[3]["id"]]                         # the kept row is not sent again
    assert starts[0].answered == 3 and starts[0].cost_usd == pytest.approx(0.03)   # it counts, its cost too
    assert out.read_bytes().startswith(data + b"\n") and out.read_bytes().endswith(b"\n")
    assert [r["id"] for r in read_jsonl(out)] == [r["id"] for r in rows[:4]] and got.not_sent == 0
    assert "truncated" not in caplog.text                                       # nothing was cut
    from cutoffmix.bench.keys.run import read_results

    out.write_bytes(out.read_bytes().rstrip(b"\n"))
    assert [r["id"] for r in read_results(out)] == [r["id"] for r in rows[:4]]  # the report counts it as the run


def test_a_torn_last_line_is_cut_with_a_warning(tmp_path, caplog):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows[:2], out, events)
    torn = '{"id": "' + rows[2]["id"] + '", "grade": "abst'                     # a crash in the middle of a write
    with out.open("a", encoding="utf-8") as f:
        f.write(torn)
    backend = ScriptedBackend()
    with caplog.at_level(logging.WARNING):
        run(backend, rows[:3], out, events)                                     # the run cuts it, saying so
    assert _recipient_calls(backend) == [rows[2]["id"]] and len(read_jsonl(out)) == 3
    assert f"{out}: cutting truncated last line 3 (interrupted append)" in caplog.text
    from cutoffmix.bench.keys.run import read_results

    caplog.clear()
    with out.open("a", encoding="utf-8") as f:
        f.write(torn)
    with caplog.at_level(logging.WARNING):
        assert len(read_results(out)) == 3                                      # the report skips it, saying so
    assert "skipping truncated last line 4 (interrupted append)" in caplog.text


# ---- D: failed calls, an error no retry can fix, failed judge calls -----------------------------------------------


class FailFirst:
    """A backend whose first call fails with ``error`` after 10 ms, while the second (an answer after 50 ms) is still
    in flight; every later call answers at once."""

    def __init__(self, error: Exception):
        self.error, self.calls = error, []

    async def chat(self, model, req):
        self.calls.append(req.metadata["item"])
        if len(self.calls) == 1:
            await asyncio.sleep(0.01)
            raise self.error
        await asyncio.sleep(0.05 if len(self.calls) == 2 else 0)
        return ChatResult(text="I don't know.", usage={"cost": 0.01}, finish_reason="stop", latency_s=0.05)

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        raise NotImplementedError


def _http(status: int, *, retryable: bool = False, retry_after: float | None = None) -> BackendError:
    return BackendError(f"OpenRouter HTTP {status} for x/b: nope", status=status, retryable=retryable,
                        retry_after=retry_after)


@pytest.mark.parametrize("error, stops", [
    (_http(401), True), (_http(402), True), (_http(403), True),                 # no key, no credits, no access
    (_http(402, retryable=True, retry_after=5.0), False),                       # the gateway's temporary budget limit
    (_http(400), False), (_http(500, retryable=True), False), (RuntimeError("socket closed"), False),
])
def test_an_error_no_retry_can_fix_stops_the_run_and_the_calls_in_flight_finish(tmp_path, error, stops):
    events, rows = fixture_plan()
    backend = FailFirst(error)
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, concurrency=2)
    stored = read_jsonl(out)
    assert [r["id"] for r in stored if r["error"]] == [rows[0]["id"]]           # one error row, the failed call's
    if stops:
        assert backend.calls == [rows[0]["id"], rows[1]["id"]]                  # nothing after the two in flight
        assert len(stored) == 2 and stored[1]["grade"] == "abstain"             # the one in flight was written
        assert got.stopped_by == "error" and got.error == f"BackendError: {error}"
        assert got.not_sent == len(rows) - 1
    else:
        assert len(backend.calls) == len(stored) == len(rows) and got.stopped_by is None and got.error is None
        assert got.not_sent == 1


def test_failed_judge_calls_are_recorded_and_counted(tmp_path):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark" and r["arm"] != "full"][:6]

    def reply(model, req):
        return RuntimeError("judge down") if req.metadata.get("purpose") else UNDECIDED
    backend = ScriptedBackend(reply)
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, judges=[J1])
    judge_calls = [req for _, req in backend.calls if req.metadata.get("purpose") == "judge"]
    assert len(judge_calls) == len(rows) == len(read_jsonl(out))
    for row in read_jsonl(out):                                                 # the failed call is in its row
        assert row["grade"] == "ungraded" and row["error"] is None
        assert row["judge_usage"] == [{"judge": "j1", "purpose": "judge", "cost": 0.0, "usage": None,
                                       "error": "RuntimeError: judge down"}]
    assert got.judge_failed == len(rows) and got.judge_error == "RuntimeError: judge down"
    retry = ScriptedBackend(lambda model, req: '{"label": "abstain", "hedged": false}' if req.metadata.get("purpose")
                            else UNDECIDED)
    again = run(retry, rows, out, events, judges=[J1])                          # the next run with judges asks again
    assert again.regraded == len(rows) and again.judge_failed == 0 and _recipient_calls(retry) == []
    assert {r["grade"] for r in again.rows} == {"abstain"}


def test_run_counts_failed_calls_apart_and_exits_4(tmp_path, capsys, gateway, fable_neon):
    plan = first_calls(tmp_path, fable_neon, 30)
    rows = read_jsonl(plan)
    failing = {r["id"] for r in rows[::3]}
    gateway.backend = ScriptedBackend(lambda m, req: RuntimeError("upstream 503") if req.metadata["item"] in failing
                                      else "I don't know.")
    out = tmp_path / "keys_results.jsonl"
    assert _run(plan, out) == 4
    captured = capsys.readouterr()
    lines = captured.out.splitlines()
    assert lines[1] == f"  abstain {30 - len(failing)}  errors {len(failing)}"  # errors are no 'ungraded' answers
    assert lines[2].startswith(f"30 rows -> {out} · sent 30 calls")
    assert captured.err.endswith(f"\r  30/30\n{len(failing)} of 30 planned calls failed (first error: RuntimeError: "
                                 "upstream 503); rerun to send them again\n")
    gateway.backend = ScriptedBackend()
    assert _run(plan, out) == 0                                                 # the rerun sends only those
    assert sorted(_recipient_calls(gateway.backend)) == sorted(failing)
    assert capsys.readouterr().out.splitlines()[1] == "  abstain 30"


def test_run_stops_after_an_error_no_retry_can_fix_and_exits_4(tmp_path, capsys, gateway, fable_neon):
    plan = first_calls(tmp_path, fable_neon, 30)
    gateway.backend = ScriptedBackend(lambda m, req: BackendError(
        "OpenRouter HTTP 401 for google/gemini-4-neon: No auth credentials found", status=401))
    out = tmp_path / "keys_results.jsonl"
    assert _run(plan, out, "--concurrency", "1") == 4
    captured = capsys.readouterr()
    assert len(gateway.backend.calls) == 1 and len(read_jsonl(out)) == 1        # no call after the 401
    assert captured.out.splitlines()[1] == "  errors 1"
    assert captured.err.endswith("stopped after an error that no retry can fix (BackendError: [401] OpenRouter HTTP "
                                 "401 for google/gemini-4-neon: No auth credentials found): 30 of 30 planned calls not "
                                 "sent; fix its cause (the API key, the credits or the access to the model) and rerun "
                                 "to send them\n")


def test_run_says_how_many_judge_calls_failed(tmp_path, capsys, gateway, fable_neon):
    plan = first_calls(tmp_path, fable_neon, 10)
    gateway.backend = ScriptedBackend(lambda m, req: RuntimeError("judge down") if req.metadata.get("purpose")
                                      else UNDECIDED)
    assert _run(plan, tmp_path / "keys_results.jsonl", "--judges", "gpt-6.2-sol") == 0
    failed = sum(1 for _, req in gateway.backend.calls if req.metadata.get("purpose"))
    assert failed > 0
    assert capsys.readouterr().err.endswith(f"\n{failed} judge calls failed (first error: RuntimeError: judge down); "
                                            "an answer they leave undecided stays undecided, and the next run with "
                                            "--judges asks again\n")


# ---- E: a completion that failed inside an HTTP 200, a native refusal, an empty answer ----------------------------


@pytest.mark.parametrize("text, finish", [("", "stop"), ("  \n", "stop"), ("", None), ("", "tool_calls")])
def test_an_empty_answer_is_a_send_error_and_is_sent_again(tmp_path, text, finish):
    events, rows = fixture_plan()
    rows = rows[:3]
    empty = ScriptedBackend(lambda m, req: ChatResult(text=text, usage={"cost": 0.002}, finish_reason=finish,
                                                      latency_s=0.2))
    out = tmp_path / "keys.jsonl"
    got = run(empty, rows, out, events)
    for row in got.rows:
        assert row["error"] == f"empty answer (finish_reason {finish})" and row["grade"] == "ungraded"
        assert row["usage"] == {"cost": 0.002} and row["grader"] == "none"      # paid: the cap counts it
    assert got.not_sent == 3 and got.cost_usd == pytest.approx(0.006)
    again, starts = ScriptedBackend(), []
    second = run(again, rows, out, events, on_start=starts.append)
    assert starts[0].errors == 3 and len(again.calls) == 3 and second.not_sent == 0
    assert second.cost_usd == pytest.approx(0.036)                              # every paid call counted once


def test_an_empty_answer_cut_at_max_tokens_stays_an_answer(tmp_path):
    events, rows = fixture_plan()
    rows = rows[:3]
    cut = ScriptedBackend(lambda m, req: ChatResult(text="", usage={"cost": 0.05}, finish_reason="length"))
    out = tmp_path / "keys.jsonl"
    got = run(cut, rows, out, events)
    assert all(r["error"] is None and r["grade"] == "ungraded" and r["grader"] == "rule" for r in got.rows)
    assert got.not_sent == 0 and run(ScriptedBackend(), rows, out, events).sent == 0   # max_tokens is pre-registered


def test_gateway_shapes_through_the_run(tmp_path):
    """The real OpenRouterBackend over an httpx.MockTransport (no network): a native refusal is a refusal; a
    completion that failed inside an HTTP 200 is retried and, while it keeps failing, a send error that the next run
    sends again; an empty answer cut at max_tokens is an answer."""
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "dark" and r["arm"] == "alone"][:4]
    shape = dict(zip((json.dumps(r["messages"]) for r in rows), ("refusal", "fails", "fails_once", "length"),
                     strict=True))
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        kind = shape[json.dumps(json.loads(request.content)["messages"])]
        seen.append(kind)
        message: dict = {"role": "assistant", "content": "I don't know."}
        finish, extra = "stop", {}
        if kind == "refusal":
            message = {"role": "assistant", "content": None, "refusal": "I'm sorry, but I can't help with that."}
        elif kind == "fails" or (kind == "fails_once" and seen.count(kind) == 1):
            message, finish = {"role": "assistant", "content": ""}, "error"
            extra = {"error": {"code": 502, "message": "Upstream provider error"}}
        elif kind == "length":
            message, finish = {"role": "assistant", "content": ""}, "length"
        choice = {"index": 0, "finish_reason": finish, "message": message, **extra}
        return httpx.Response(200, json={"id": "gen", "choices": [choice], "usage": {"cost": 0.001}})

    gateway = OpenRouterBackend(api_key="test-key", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                                max_retries=1, retry_base_delay=0)
    out = tmp_path / "keys.jsonl"
    got = run(gateway, rows, out, events, concurrency=1)
    by_kind = {shape[json.dumps(p["messages"])]: r for p, r in zip(rows, got.rows, strict=True)}
    assert by_kind["refusal"]["grader"] == "refusal" and by_kind["refusal"]["error"] is None
    assert by_kind["fails"]["error"].startswith("BackendError: [502 completion_error] x/b: the completion failed")
    assert by_kind["fails_once"]["error"] is None and by_kind["fails_once"]["grade"] == "abstain"   # the retry
    assert by_kind["length"]["error"] is None and by_kind["length"]["grader"] == "rule"
    assert seen.count("fails") == 2 and got.not_sent == 1
    seen.clear()
    again = run(gateway, rows, out, events)
    assert seen == ["fails", "fails"] and again.start.errors == 1               # only the failed call, sent again


# ---- F: Ctrl-C ----------------------------------------------------------------------------------------------------


def test_the_stop_event_lets_the_items_in_flight_finish_with_their_judges(tmp_path):
    events, rows = fixture_plan()

    class Slow(ScriptedBackend):
        def __init__(self, stop: asyncio.Event):
            super().__init__(lambda m, req: '{"label": "abstain", "hedged": false}' if req.metadata.get("purpose")
                             else UNDECIDED)
            self.stop, self.arrived = stop, 0

        async def chat(self, model, req):
            if not req.metadata.get("purpose"):
                self.arrived += 1
                if self.arrived == 2:
                    self.stop.set()                                             # Ctrl-C while two calls are in flight
            await asyncio.sleep(0.02)
            return await super().chat(model, req)

    async def go():
        stop = asyncio.Event()
        backend = Slow(stop)
        return backend, await run_plan(backend, MODELS, rows, tmp_path / "keys.jsonl", events=events, concurrency=2,
                                       judges=[J1], stop=stop)
    backend, got = asyncio.run(go())
    assert len(_recipient_calls(backend)) == 2 and got.stopped_by == "interrupted" and got.sent == 2
    stored = read_jsonl(tmp_path / "keys.jsonl")
    assert len(stored) == 2 and all(len(r["judge_usage"]) == 1 and r["grade"] == "abstain" for r in stored)
    assert got.not_sent == len(rows) - 2


CHILD = '''"""'bench keys run' against a slow scripted gateway, as a subprocess the test can interrupt."""
import asyncio
import json
import signal
import sys

signal.signal(signal.SIGINT, signal.default_int_handler)  # as in a terminal, also under a runner that ignores SIGINT

import cutoffmix.backends as backends
from cutoffmix import cli
from cutoffmix.types import ChatResult

log, delay = sys.argv[1], float(sys.argv[2])


class Slow:
    async def chat(self, model, req):
        with open(log, "a", encoding="utf-8") as f:  # every call, logged as it is sent
            f.write(json.dumps({"item": req.metadata.get("item"), "purpose": req.metadata.get("purpose")}) + "\\n")
        await asyncio.sleep(delay)
        return ChatResult(text="I don't know.", usage={"cost": 0.001}, finish_reason="stop", latency_s=delay)

    def stream_continuation(self, model, req):
        raise NotImplementedError


backends.make_backend = lambda cfg, **kw: Slow()
sys.exit(cli.main(sys.argv[3:]))
'''


def _child(tmp_path: Path, plan: Path, out: Path, log: Path, delay: float, *extra: str, **popen) -> subprocess.Popen:
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    env = dict(os.environ, OPENROUTER_API_KEY="sk-test-not-used",
               PYTHONPATH=os.pathsep.join([str(ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]))
    argv = ["bench", "keys", "run", "--plan", str(plan), "--events", str(EVENTS), "--out", str(out), "--max-cost", "5",
            *extra]
    return subprocess.Popen([sys.executable, str(script), str(log), str(delay), *argv], cwd=tmp_path, env=env,
                            text=True, **popen)


def _logged(log: Path) -> list[str]:
    return [json.loads(line)["item"] for line in log.read_text("utf-8").splitlines()] if log.exists() else []


def _wait(condition, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform == "win32", reason="the first Ctrl-C is handled on POSIX only")
def test_the_first_ctrl_c_lets_the_calls_in_flight_finish_and_the_rerun_resumes(tmp_path, fable_neon):
    plan = first_calls(tmp_path, fable_neon, 24)
    out, log = tmp_path / "keys_results.jsonl", tmp_path / "calls.jsonl"
    proc = _child(tmp_path, plan, out, log, 1.0, "--concurrency", "3", stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE)
    _wait(lambda: len(_logged(log)) >= 4, proc)                                 # the second wave is in flight
    assert proc.poll() is None, proc.communicate()
    os.kill(proc.pid, signal.SIGINT)
    stdout, stderr = proc.communicate(timeout=60)
    sent, rows = _logged(log), read_jsonl(out)
    assert proc.returncode == 130, stderr
    assert sorted(r["id"] for r in rows) == sorted(sent) and len(sent) < 24     # every call sent wrote its row
    assert "interrupted: no new calls are started; the calls in flight finish and are written (Ctrl-C again " \
           "aborts them)\n" in stderr
    assert stderr.endswith(f"interrupted: {24 - len(rows)} of 24 planned calls not sent; rerun to resume\n")
    assert f"  abstain {len(rows)}\n{len(rows)} rows -> {out} · sent {len(rows)} calls" in stdout   # the summary
    log2 = tmp_path / "calls2.jsonl"
    resumed = _child(tmp_path, plan, out, log2, 0.0, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = resumed.communicate(timeout=60)
    assert resumed.returncode == 0, stderr
    assert not set(sent) & set(_logged(log2)) and len(sent) + len(_logged(log2)) == 24   # no call sent twice
    ids = [r["id"] for r in read_jsonl(out)]
    assert len(ids) == len(set(ids)) == 24


@pytest.mark.skipif(sys.platform == "win32", reason="the first Ctrl-C is handled on POSIX only")
def test_a_second_ctrl_c_aborts_at_once(tmp_path, fable_neon):
    plan = first_calls(tmp_path, fable_neon, 10)
    out, log, err = tmp_path / "keys_results.jsonl", tmp_path / "calls.jsonl", tmp_path / "stderr.txt"
    with err.open("w", encoding="utf-8") as err_file:
        proc = _child(tmp_path, plan, out, log, 30.0, "--concurrency", "2", stdout=subprocess.DEVNULL,
                      stderr=err_file)
        _wait(lambda: len(_logged(log)) >= 2, proc)                             # two calls in flight, 30 s each
        os.kill(proc.pid, signal.SIGINT)
        _wait(lambda: "no new calls are started" in err.read_text("utf-8"), proc)
        assert proc.poll() is None and "no new calls are started" in err.read_text("utf-8")
        started = time.monotonic()
        os.kill(proc.pid, signal.SIGINT)
        assert proc.wait(timeout=20) == 130 and time.monotonic() - started < 20  # not after the 30 s calls
    assert err.read_text("utf-8").endswith("cutoffmix: interrupted\n")
    assert len(_logged(log)) == 2 and (not out.exists() or read_jsonl(out) == [])   # aborted: nothing written


# ---- G: a wrong key is never the key itself -------------------------------------------------------------------------


def test_a_wrong_key_never_equals_its_key_on_letters_and_digits():
    from cutoffmix.bench.grading import normalize_text

    notes = {1: ("Alba Quist", "It is Alba Quist, with Yara Okafor."),
             2: ("Bodo Rinne", "It is Bodo Rinne with Yara Okafor"),                  # a comma and a period less
             3: ("Cora Sell", "It is **Cora Sell**, with Ines Brandt.")}

    def plan(n):
        events = [paper(i, notes[i][0]) for i in range(1, n + 1)]
        results = [result(f"paper:{i}", "a", "correct", notes[i][1]) for i in range(1, n + 1)]
        results += [result(f"paper:{i}", "b", "abstain", "I don't know.") for i in range(1, n + 1)]
        return build_plan(events, results, pairs=[("a", "b")], min_items=1, today=TODAY)

    def letters(note):
        return "".join(ch for ch in normalize_text(note) if ch.isalnum())
    rows, summary = plan(2)                                                     # the two keys differ in a comma only
    assert summary["no_wrong_key"] == 2 and not [r for r in rows if r["set"] == "dark"]
    rows, summary = plan(3)
    keys = {r["event_id"]: r["note"] for r in rows if r["arm"] == "key"}
    wrong = {r["event_id"]: r for r in rows if r["arm"] == "wrong_key"}
    assert summary["no_wrong_key"] == 0 and len(wrong) == 3
    assert wrong["paper:1"]["note_event"] == wrong["paper:2"]["note_event"] == "paper:3"
    assert all(letters(wrong[e]["note"]) != letters(keys[e]) for e in keys)
