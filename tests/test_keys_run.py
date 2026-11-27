"""Key, not value (docs/DESIGN_keys.md): the runner with a scripted backend (sending, grading, resuming, errors,
refusals, the limit and the cost cap)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from keys_testlib import TODAY, fixture_data

from cutoffmix.bench.io import read_jsonl
from cutoffmix.bench.keys import build_plan, run_plan
from cutoffmix.types import ChatResult, ModelSpec

# ---- run --------------------------------------------------------------------------------------------------------


class ScriptedBackend:
    """``chat`` answers from ``reply(model, req)``: a ChatResult, a string (answer text) or an exception to raise."""

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[tuple[str, Any]] = []

    async def chat(self, model, req):
        self.calls.append((model.key, req))
        out = self.reply(model, req)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, str):
            return ChatResult(text=out, usage={"cost": 0.01}, finish_reason="stop", latency_s=0.5)
        return out

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        raise NotImplementedError


MODELS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in ("a", "b")]


def fixture_plan():
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, rows


def run(backend, rows, out, events, **kw):
    return asyncio.run(run_plan(backend, MODELS, rows, out, events=events, **kw))


def test_run_sends_grades_and_resumes(tmp_path):
    events, rows = fixture_plan()
    by_id = {e.id: e for e in events}

    def reply(model, req):
        event = by_id[req.metadata["event_id"]]
        if req.metadata["set"] in ("donor_wrong", "control"):
            return f"I think it is {req.metadata['donor_answer']}." if req.metadata["arm"] == "full" else "No idea."
        if req.metadata["arm"] in ("key", "full"):
            return f"The first author is {event.expected_current}."
        return "I don't know."
    backend = ScriptedBackend(reply)
    out = tmp_path / "keys.jsonl"
    got = run(backend, rows, out, events, concurrency=3)
    assert len(got) == len(rows) == len(backend.calls) and [r["id"] for r in got] == [r["id"] for r in rows]
    model, req = backend.calls[0]
    assert model == "b" and req.temperature == 0.0 and req.max_tokens == 4096
    assert set(req.metadata) == {"event_id", "probe", "model", "arm", "set", "donor", "item", "donor_answer"}
    assert req.metadata["probe"] == "current_d" and req.metadata["model"] == "b" and req.metadata["donor"] == "a"
    by_arm = {(r["set"], r["arm"]): r for r in got}
    assert by_arm[("dark", "key")]["grade"] == "correct" and by_arm[("dark", "alone")]["grade"] == "abstain"
    assert by_arm[("donor_wrong", "full")]["adopted"] is True and by_arm[("donor_wrong", "alone")]["adopted"] is False
    assert by_arm[("control", "full")]["adopted"] is True and by_arm[("dark", "key")]["adopted"] is None
    assert len(read_jsonl(out)) == len(rows)
    again = run(ScriptedBackend(reply), rows, out, events)
    assert len(again) == len(rows) and len(read_jsonl(out)) == len(rows)       # nothing sent twice


def test_run_records_errors_and_refusals_and_retries_errors(tmp_path):
    events, rows = fixture_plan()

    def reply(model, req):
        if req.metadata["arm"] == "wrong_key":
            return RuntimeError("upstream 503")
        if req.metadata["arm"] == "full":
            return ChatResult(text="", finish_reason="content_filter", usage={"cost": 0.0})
        return "I don't know."
    out = tmp_path / "keys.jsonl"
    got = run(ScriptedBackend(reply), rows, out, events)
    errors = [r for r in got if r["error"]]
    assert errors and all(r["arm"] == "wrong_key" and r["grade"] == "ungraded" and "503" in r["error"]
                          for r in errors)
    refused = [r for r in got if r["grader"] == "refusal"]
    assert refused and all(r["arm"] == "full" and r["grade"] == "ungraded" for r in refused)
    second = ScriptedBackend(lambda model, req: "I don't know.")
    run(second, rows, out, events)
    assert {req.metadata["arm"] for _, req in second.calls} == {"wrong_key"}    # only the failed calls again
    last = {r["id"]: r for r in read_jsonl(out)}
    assert not any(r["error"] for r in last.values())


def test_run_honours_limit_and_cost_cap(tmp_path):
    events, rows = fixture_plan()
    backend = ScriptedBackend(lambda model, req: "I don't know.")
    run(backend, rows, tmp_path / "limit.jsonl", events, limit=2)
    assert len(backend.calls) == 2
    capped = ScriptedBackend(lambda model, req: "I don't know.")
    run(capped, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert len(capped.calls) == 3                                               # 0.01 each: stops once 0.03 >= 0.025
    resumed = ScriptedBackend(lambda model, req: "I don't know.")
    run(resumed, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert resumed.calls == []                                                  # the earlier rows count toward the cap
    with pytest.raises(ValueError, match="not configured"):
        asyncio.run(run_plan(backend, MODELS[:1], rows, tmp_path / "x.jsonl", events=events))
    with pytest.raises(ValueError, match="not in the events file"):
        asyncio.run(run_plan(backend, MODELS, rows, tmp_path / "x.jsonl", events=events[:2]))


def test_run_cuts_an_interrupted_last_line(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(lambda model, req: "I don't know."), rows[:2], out, events)
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "' + rows[2]["id"] + '", "grade": "abst')               # a crash in the middle of a write
    backend = ScriptedBackend(lambda model, req: "I don't know.")
    run(backend, rows[:3], out, events)
    assert len(backend.calls) == 1 and len(read_jsonl(out)) == 3


