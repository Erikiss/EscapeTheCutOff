"""Key, not value (docs/DESIGN_keys.md section 6): the runner with a scripted backend (sending, grading, provenance,
resuming, the safety checks on --out, the plan and the lock, errors, refusals, the limit and the cost cap)."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from typing import Any

import pytest
from keys_testlib import TODAY, fixture_data, result

from cutoffmix.backends.mock import MockBackend, default_mock_truths
from cutoffmix.backends.openrouter import OpenRouterBackend
from cutoffmix.bench.io import read_jsonl, write_jsonl
from cutoffmix.bench.keys import (
    RunOutcome,
    RunStart,
    build_plan,
    judge_cost,
    prompt_sha,
    request_sha,
    run_plan,
)
from cutoffmix.bench.keys.run import _backend_name, _locked
from cutoffmix.types import ChatRequest, ChatResult, ModelSpec

# ---- helpers ------------------------------------------------------------------------------------------------------

ROW_FIELDS = ["id", "set", "donor", "recipient", "event_id", "arm", "response", "grade", "hedged", "grader", "adopted",
              "adoption", "adoption_grader", "rule_adopted", "latency_s", "usage", "judge_usage", "backend",
              "prompt_sha", "request_sha", "note_event", "today", "error"]       # design section 6, in this order


class ScriptedBackend:
    """``chat`` answers from ``reply(model, req)``: a ChatResult, a string (answer text, ``cost`` USD) or an exception
    to raise."""

    def __init__(self, reply=None, cost: float = 0.01):
        self.reply = reply or (lambda model, req: "I don't know.")
        self.cost = cost
        self.calls: list[tuple[str, Any]] = []

    async def chat(self, model, req):
        self.calls.append((model.key, req))
        out = self.reply(model, req)
        if isinstance(out, BaseException):
            raise out
        if isinstance(out, str):
            return ChatResult(text=out, usage={"cost": self.cost}, finish_reason="stop", latency_s=0.5)
        return out

    def stream_continuation(self, model, req):  # pragma: no cover - not used here
        raise NotImplementedError


MODELS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in ("a", "b")]


def fixture_plan(**extra):
    events, results = fixture_data()
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    return events, [{**r, **extra} for r in rows]


def run(backend, rows, out, events, models=MODELS, **kw) -> RunOutcome:
    return asyncio.run(run_plan(backend, models, rows, out, events=events, **kw))


def refused(backend, rows, out, events, match, **kw):
    """The run stops with ``ValueError`` before any call and leaves ``out`` byte-identical."""
    before = out.read_bytes() if out.exists() else None
    with pytest.raises(ValueError, match=match):
        run(backend, rows, out, events, **kw)
    assert backend.calls == []
    assert (out.read_bytes() if out.exists() else None) == before


# ---- sending and grading ------------------------------------------------------------------------------------------


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
    assert isinstance(got, RunOutcome) and got.sent == len(rows) == len(backend.calls) and got.regraded == 0
    assert [r["id"] for r in got.rows] == [r["id"] for r in rows]
    assert got.not_sent == 0 and got.stopped_by is None and got.planned == len(rows)
    model, req = backend.calls[0]
    assert model == "b" and req.temperature == 0.0 and req.max_tokens == 4096
    assert set(req.metadata) == {"event_id", "probe", "model", "arm", "set", "donor", "item", "donor_answer"}
    assert req.metadata["probe"] == "current_d" and req.metadata["model"] == "b" and req.metadata["donor"] == "a"
    by_arm = {(r["set"], r["arm"]): r for r in got.rows}
    assert by_arm[("dark", "key")]["grade"] == "correct" and by_arm[("dark", "alone")]["grade"] == "abstain"
    assert by_arm[("donor_wrong", "full")]["adopted"] is True and by_arm[("donor_wrong", "alone")]["adopted"] is False
    assert by_arm[("donor_wrong", "full")]["adoption"] == "endorsed"
    assert by_arm[("donor_wrong", "alone")]["adoption"] == "absent"
    graders = {by_arm[(s, a)]["adoption_grader"] for s in ("donor_wrong", "control") for a in ("alone", "full")}
    assert graders == {"rule"}
    assert by_arm[("control", "full")]["adopted"] is True and by_arm[("dark", "key")]["adopted"] is None
    assert by_arm[("dark", "key")]["adoption"] is None and by_arm[("dark", "key")]["adoption_grader"] is None
    assert len(read_jsonl(out)) == len(rows)
    again = run(ScriptedBackend(reply), rows, out, events)
    assert again.sent == 0 and again.start.answered == len(rows) and len(again.rows) == len(rows)
    assert len(read_jsonl(out)) == len(rows)                                     # nothing sent twice


def test_adopted_is_false_for_a_correct_answer(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    rows = [r for r in rows if r["set"] == "control"]
    monkeypatch.setattr("cutoffmix.bench.keys.text.adoption_label", lambda response, name, event: "endorsed")
    got = run(ScriptedBackend(lambda m, r: "I cannot find a paper with that title; it may not exist."), rows,
              tmp_path / "keys.jsonl", events)
    assert {r["grade"] for r in got.rows} == {"correct"}                        # an abstention on a fake paper
    assert {r["adopted"] for r in got.rows} == {False} and {r["adoption"] for r in got.rows} == {"endorsed"}


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
    errors = [r for r in got.rows if r["error"]]
    assert errors and all(r["arm"] == "wrong_key" and r["grade"] == "ungraded" and "503" in r["error"]
                          for r in errors)
    assert all(r["usage"] is None and r["judge_usage"] == [] and r["backend"] == "scripted" for r in errors)
    refused_rows = [r for r in got.rows if r["grader"] == "refusal"]
    assert refused_rows and all(r["arm"] == "full" and r["grade"] == "ungraded" for r in refused_rows)
    assert all(r["adopted"] is None and r["adoption"] is None for r in refused_rows)   # no answer, no adoption
    assert got.not_sent == len(errors)                                          # an error row is no answer
    second = ScriptedBackend(lambda model, req: "I don't know.")
    starts: list[RunStart] = []
    again = run(second, rows, out, events, on_start=starts.append)
    assert {req.metadata["arm"] for _, req in second.calls} == {"wrong_key"}    # only the failed calls again
    assert starts[0].errors == len(errors) == again.sent and again.not_sent == 0
    last = {r["id"]: r for r in read_jsonl(out)}
    assert not any(r["error"] for r in last.values())



def test_run_cuts_an_interrupted_last_line(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows[:2], out, events)
    with out.open("a", encoding="utf-8") as f:
        f.write('{"id": "' + rows[2]["id"] + '", "grade": "abst')               # a crash in the middle of a write
    backend = ScriptedBackend()
    run(backend, rows[:3], out, events)
    assert len(backend.calls) == 1 and len(read_jsonl(out)) == 3


# ---- provenance ---------------------------------------------------------------------------------------------------


def test_every_row_records_its_provenance(tmp_path):
    events, rows = fixture_plan(today=TODAY)
    out = tmp_path / "keys.jsonl"
    got = run(ScriptedBackend(), rows, out, events)
    specs = {m.key: m for m in MODELS}
    planned = {r["id"]: r for r in rows}
    for row in read_jsonl(out):
        assert list(row) == ROW_FIELDS
        plan_row = planned[row["id"]]
        assert row["backend"] == "scripted"                                    # derived from the backend object
        assert row["prompt_sha"] == prompt_sha(plan_row["messages"])
        assert row["request_sha"] == request_sha(plan_row["messages"], max_tokens=4096, temperature=0.0,
                                                 model=specs[plan_row["recipient"]])
        assert row["note_event"] == plan_row["note_event"] and row["today"] == TODAY
        assert row["judge_usage"] == []
    wrong = [r for r in got.rows if r["arm"] == "wrong_key"]
    assert wrong and all(r["note_event"] and r["note_event"] != r["event_id"] for r in wrong)
    named = run(ScriptedBackend(), rows[:1], tmp_path / "named.jsonl", events, backend_name="openrouter")
    assert named.rows[0]["backend"] == "openrouter" and named.start.backend == "openrouter"
    _, planned_rows = fixture_plan()
    plain = [{k: v for k, v in planned_rows[0].items() if k != "today"}]       # a plan row without the field
    assert run(ScriptedBackend(), plain, tmp_path / "plain.jsonl", events).rows[0]["today"] is None


def test_request_sha_covers_the_messages_the_budget_and_the_model_settings():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    spec = ModelSpec(key="b", provider_id="x/b", reasoning={"effort": "low"}, provider_routing={"only": ["p"]},
                     extra_body={"verbosity": "low"}, supports_temperature=True)   # a model that takes a temperature
    base = request_sha(messages, max_tokens=4096, temperature=0.0, model=spec)
    assert base == request_sha([dict(m) for m in messages], max_tokens=4096, temperature=0.0, model=spec)
    assert len(base) == 40
    changed = [
        request_sha([messages[0], {"role": "user", "content": "q2"}], max_tokens=4096, temperature=0.0, model=spec),
        request_sha(messages, max_tokens=2048, temperature=0.0, model=spec),
        request_sha(messages, max_tokens=4096, temperature=None, model=spec),
        request_sha(messages, max_tokens=4096, temperature=0.0, model=dataclasses.replace(spec, provider_id="x/c")),
        request_sha(messages, max_tokens=4096, temperature=0.0,
                    model=dataclasses.replace(spec, reasoning={"effort": "high"})),
        request_sha(messages, max_tokens=4096, temperature=0.0,
                    model=dataclasses.replace(spec, provider_routing={"only": ["q"]})),
        request_sha(messages, max_tokens=4096, temperature=0.0, model=dataclasses.replace(spec, extra_body={})),
    ]
    assert base not in changed and len(set(changed)) == len(changed)
    same = request_sha(messages, max_tokens=4096, temperature=0.0, model=dataclasses.replace(spec, key="other",
                                                                                              color="#000"))
    assert same == base                                                         # display settings do not count


def test_request_sha_hashes_the_temperature_as_the_gateway_sends_it():
    """A temperature reaches the provider only for a model with ``supports_temperature`` (build_chat_body), so the
    hash follows the body: flipping the setting changes it, a temperature that is never sent does not."""
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    off = ModelSpec(key="b", provider_id="x/b")                                 # as configured for the three models
    on = dataclasses.replace(off, supports_temperature=True)
    req = ChatRequest(messages=messages, max_tokens=4096, temperature=0.0)
    gateway = OpenRouterBackend()                                               # builds bodies offline, never calls
    assert "temperature" not in gateway.build_chat_body(off, req)
    assert gateway.build_chat_body(on, req)["temperature"] == 0
    sha = {(model.supports_temperature, t): request_sha(messages, max_tokens=4096, temperature=t, model=model)
           for model in (off, on) for t in (0.0, None)}
    assert sha[(False, 0.0)] == sha[(False, None)] == sha[(True, None)]       # no temperature in the body
    assert sha[(True, 0.0)] != sha[(False, 0.0)]


def test_a_model_that_starts_taking_a_temperature_makes_its_rows_stale(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows, out, events)
    sampled = [MODELS[0], dataclasses.replace(MODELS[1], supports_temperature=True)]   # e.g. a config override
    refused(ScriptedBackend(), rows, out, events, f"{len(rows)} of the {len(rows)} planned calls were answered under "
            "another request", models=sampled)
    backend = ScriptedBackend()
    assert run(backend, rows, out, events, models=sampled, resend_stale=True).sent == len(rows) == len(backend.calls)


def test_the_default_backend_name_comes_from_the_backend():
    assert _backend_name(MockBackend(default_mock_truths(latency_s=0.0))) == "mock"
    assert _backend_name(OpenRouterBackend()) == "openrouter"                   # constructed offline, never called
    assert _backend_name(ScriptedBackend()) == "scripted"


# ---- resume: provenance must match --------------------------------------------------------------------------------


def test_mock_rows_and_real_rows_never_share_a_file(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows, out, events, backend_name="mock", limit=5)
    refused(ScriptedBackend(), rows, out, events, r"5 row\(s\) from another backend \(mock: 5\)",
            backend_name="openrouter")
    real = tmp_path / "real.jsonl"
    run(ScriptedBackend(), rows, real, events, backend_name="openrouter", limit=3)
    refused(ScriptedBackend(), rows, real, events, "another backend.*openrouter: 3.*new --out", backend_name="mock")
    full = tmp_path / "full_mock.jsonl"                                         # a complete mock run: still refused
    run(ScriptedBackend(), rows, full, events, backend_name="mock")
    refused(ScriptedBackend(), rows, full, events, "another backend", backend_name="openrouter")


def test_rows_without_provenance_are_refused(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows[:4], out, events)
    old = [{k: v for k, v in r.items() if k not in ("backend", "prompt_sha", "request_sha")} for r in read_jsonl(out)]
    write_jsonl(out, old)                                                       # rows as version 1 wrote them
    refused(ScriptedBackend(), rows, out, events, r"4 row\(s\) without provenance.*4 of them for ids of this plan")
    foreign = tmp_path / "foreign.jsonl"                                        # also for an id the plan does not have
    write_jsonl(foreign, [{**old[0], "id": "not-in-the-plan"}])
    refused(ScriptedBackend(), rows, foreign, events, r"1 row\(s\) without provenance.*0 of them")


def test_a_changed_request_stops_the_run_unless_resend_stale(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows, out, events)
    refused(ScriptedBackend(), rows, out, events, f"{len(rows)} of the {len(rows)} planned calls were answered under "
            "another request.*--resend-stale", max_tokens=2048)
    settings = [MODELS[0], ModelSpec(key="b", provider_id="x/b", reasoning={"effort": "high"})]
    refused(ScriptedBackend(), rows, out, events, "another request", models=settings)
    events2, results2 = fixture_data()                                          # another date in the system prompt
    replanned, _ = build_plan(events2, results2, pairs=[("a", "b")], min_items=3, today="2026-11-30")
    assert [r["id"] for r in replanned] == [r["id"] for r in rows]
    refused(ScriptedBackend(), replanned, out, events, "another request")
    backend = ScriptedBackend()
    seen: list[RunStart] = []
    got = run(backend, replanned, out, events, resend_stale=True, limit=4, on_start=seen.append)
    assert len(backend.calls) == 4 and got.sent == 4 and got.stopped_by == "limit"
    assert seen[0].stale == len(rows) == seen[0].pending and seen[0].answered == 0
    assert got.not_sent == len(rows) - 4 and len(got.rows) == 4                 # stale rows are no current rows
    assert all(r["prompt_sha"] == prompt_sha(p["messages"]) for r, p in zip(got.rows, replanned, strict=False))
    rest = ScriptedBackend()
    done = run(rest, replanned, out, events, resend_stale=True)
    assert len(rest.calls) == len(rows) - 4 and done.not_sent == 0 and len(done.rows) == len(rows)
    assert run(ScriptedBackend(), replanned, out, events).sent == 0           # the newest rows count


# ---- safety -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["probe_results", "events", "plan", "no_trailing_newline"])
def test_out_must_hold_keys_results_only(tmp_path, kind):
    events, rows = fixture_plan()
    out = tmp_path / f"{kind}.jsonl"
    if kind in ("probe_results", "no_trailing_newline"):
        write_jsonl(out, fixture_data()[1])                                     # stored probe rows: probe, model
        if kind == "no_trailing_newline":
            out.write_bytes(out.read_bytes().rstrip(b"\n"))                     # must not be cut either
        match = rf"{kind}.jsonl:1 is not a 'bench keys run' result row \(no id, arm; has probe, model\)"
    elif kind == "events":
        write_jsonl(out, [e.to_dict() for e in events])
        match = r"events.jsonl:1 is not a 'bench keys run' result row \(no arm, grade\)"
    else:
        write_jsonl(out, rows)                                                  # a plan given as --out
        match = r"plan.jsonl:1 is not a 'bench keys run' result row \(no grade; has messages\)"
    refused(ScriptedBackend(), rows, out, events, match)


def test_a_one_row_foreign_file_without_a_newline_is_not_cut(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "stored.jsonl"
    out.write_text(json.dumps(result("paper:1", "b", "abstain")), encoding="utf-8")   # one complete row, no newline
    refused(ScriptedBackend(), rows, out, events, r"stored.jsonl:1 is not a 'bench keys run' result row")
    partial = tmp_path / "partial.jsonl"                                        # an interrupted first append is cut
    partial.write_text('{"id": "' + rows[0]["id"] + '", "grade": "abst', encoding="utf-8")
    backend = ScriptedBackend()
    run(backend, rows[:1], partial, events)
    assert len(backend.calls) == 1 and len(read_jsonl(partial)) == 1


def test_a_mixed_keys_file_names_the_first_foreign_row(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows[:3], out, events)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result("paper:1", "b", "abstain")) + "\n")
    refused(ScriptedBackend(), rows, out, events, r"keys.jsonl:4 is not a 'bench keys run' result row")


def test_a_plan_with_repeated_ids_is_refused(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    refused(ScriptedBackend(), [*rows, rows[0]], out, events, "the plan repeats 1 id")
    assert not out.exists()
    broken = [{k: v for k, v in r.items() if k != "messages"} for r in rows]
    refused(ScriptedBackend(), broken, out, events, "plan row 1 has no messages")


def test_a_second_run_on_the_same_out_is_refused(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    with _locked(out):                                                          # another process holds the lock
        refused(ScriptedBackend(), rows, out, events, f"another run is writing {out}")
    assert (tmp_path / "keys.jsonl.lock").exists()

    async def both():
        gate = asyncio.Event()
        first = ScriptedBackend()

        async def slow_chat(model, req):
            await gate.wait()
            return await ScriptedBackend.chat(first, model, req)
        first.chat = slow_chat
        task = asyncio.create_task(run_plan(first, MODELS, rows, out, events=events, concurrency=2))
        await asyncio.sleep(0)                                                  # the first run holds the lock now
        second = ScriptedBackend()
        with pytest.raises(ValueError, match="another run is writing"):
            await run_plan(second, MODELS, rows, out, events=events)
        gate.set()
        return second, await task
    second, outcome = asyncio.run(both())
    assert second.calls == [] and outcome.sent == len(rows)
    ids = [r["id"] for r in read_jsonl(out)]
    assert len(ids) == len(set(ids)) == len(rows)                               # every call sent once
    assert run(ScriptedBackend(), rows, out, events).sent == 0                  # the lock ends with the run


def test_a_symlink_to_the_results_file_shares_its_lock(tmp_path):
    events, rows = fixture_plan()
    real = tmp_path / "keys_results.jsonl"
    real.touch()
    link = tmp_path / "results_link.jsonl"
    link.symlink_to(real)
    (tmp_path / "sub").mkdir()
    spellings = [link, tmp_path / "sub" / ".." / "keys_results.jsonl"]          # one file, three names

    async def both(other):
        gate = asyncio.Event()
        first = ScriptedBackend()

        async def slow_chat(model, req):
            await gate.wait()
            return await ScriptedBackend.chat(first, model, req)
        first.chat = slow_chat
        task = asyncio.create_task(run_plan(first, MODELS, rows, real, events=events, concurrency=2))
        await asyncio.sleep(0)                                                  # the first run holds the lock now
        second = ScriptedBackend()
        with pytest.raises(ValueError, match="another run is writing"):
            await run_plan(second, MODELS, rows, other, events=events)
        gate.set()
        return second, await task
    second, outcome = asyncio.run(both(link))
    assert second.calls == [] and outcome.sent == len(rows)
    ids = [r["id"] for r in read_jsonl(real)]
    assert len(ids) == len(set(ids)) == len(rows)                               # every call sent once
    for other in spellings:
        with _locked(real):
            refused(ScriptedBackend(), rows, other, events, "another run is writing")
    assert sorted(p.name for p in tmp_path.iterdir() if p.name.endswith(".lock")) == ["keys_results.jsonl.lock"]


def test_an_out_that_cannot_take_rows_is_refused_before_any_call(tmp_path, monkeypatch):
    events, rows = fixture_plan()
    folder = tmp_path / "keys_results.jsonl"
    folder.mkdir()                                                              # --out naming a directory
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match=r"cannot write --out .*keys_results.jsonl: it exists and is not a regular"):
        run(backend, rows, folder, events)
    assert backend.calls == [] and list(folder.iterdir()) == [] and not list(tmp_path.glob("*.lock"))
    binary = tmp_path / "image.png"
    binary.write_bytes(bytes(range(256)) * 4)
    refused(ScriptedBackend(), rows, binary, events, r"image.png is not a text file")
    stored = tmp_path / "stored.jsonl"                                          # an earlier results file ...
    run(ScriptedBackend(), rows[:2], stored, events)
    empty = tmp_path / "empty.jsonl"
    empty.touch()
    new = tmp_path / "new.jsonl"
    locked_out = {stored, empty, new}
    real_open = type(stored).open

    def read_only(self, mode="r", *args, **kwargs):                             # ... that this user may not write
        if self in locked_out and any(c in mode for c in "aw+"):
            raise PermissionError(13, "Permission denied", str(self))
        return real_open(self, mode, *args, **kwargs)
    monkeypatch.setattr(type(stored), "open", read_only)
    for out in (stored, empty, new):
        refused(ScriptedBackend(), rows, out, events, rf"cannot write --out .*{out.name} \(Permission denied\)")
    assert not new.exists()


# ---- cost ---------------------------------------------------------------------------------------------------------


def test_the_cap_counts_every_row_of_the_file_and_the_judges(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(), rows, out, events, limit=2)                          # 2 rows at 0.01
    stored = read_jsonl(out)
    judged = {**stored[0], "judge_usage": [{"judge": "j", "purpose": "judge", "cost": 0.005, "usage": {"cost": 0.005}},
                                           {"cost": 0.005}]}                     # a bare usage entry counts as well
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(judged) + "\n")                                      # the same id once more (a double send)
    assert judge_cost(judged) == pytest.approx(0.01) and judge_cost({"judge_usage": None}) == 0.0
    idle = ScriptedBackend()
    capped = run(idle, rows, out, events, max_cost=0.04)                        # 3 x 0.01 + 0.01 of judges = 0.04
    assert idle.calls == [] and capped.stopped_by == "max_cost" and capped.start.cost_usd == pytest.approx(0.04)
    assert capped.sent == 0 and capped.not_sent == len(rows) - 2
    one = ScriptedBackend()
    got = run(one, rows, out, events, max_cost=0.045, concurrency=1)
    assert len(one.calls) == 1 and got.stopped_by == "max_cost"
    assert got.cost_usd == pytest.approx(0.05) and got.recipients_usd == pytest.approx(0.04)
    assert got.judges_usd == pytest.approx(0.01) and got.run_cost_usd == pytest.approx(0.01)


def test_the_outcome_says_whether_the_limit_or_the_cap_stopped_the_run(tmp_path):
    events, rows = fixture_plan()
    backend = ScriptedBackend()
    limited = run(backend, rows, tmp_path / "limit.jsonl", events, limit=2)
    assert len(backend.calls) == 2 and limited.sent == 2 and limited.stopped_by == "limit"
    assert limited.not_sent == len(rows) - 2 and limited.start.to_send == 2 and limited.start.pending == len(rows)
    capped = ScriptedBackend()
    got = run(capped, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert len(capped.calls) == 3 and got.stopped_by == "max_cost"            # 0.01 each: stops once 0.03 >= 0.025
    assert got.not_sent == len(rows) - 3 and got.cost_usd == pytest.approx(0.03)
    resumed = ScriptedBackend()
    again = run(resumed, rows, tmp_path / "cap.jsonl", events, concurrency=1, max_cost=0.025)
    assert resumed.calls == [] and again.stopped_by == "max_cost"            # the earlier rows count toward the cap
    finished = run(ScriptedBackend(), rows, tmp_path / "cap.jsonl", events, max_cost=1.0)
    assert finished.stopped_by is None and finished.not_sent == 0 and finished.sent == len(rows) - 3
    nothing = run(ScriptedBackend(), rows, tmp_path / "cap.jsonl", events, limit=0)
    assert nothing.stopped_by is None and nothing.sent == 0                     # nothing was left to stop
    zero = run(ScriptedBackend(), rows, tmp_path / "zero.jsonl", events, limit=0)
    assert zero.stopped_by == "limit" and zero.sent == 0 and zero.not_sent == len(rows)


def test_run_checks_its_arguments(tmp_path):
    events, rows = fixture_plan()
    backend = ScriptedBackend()
    with pytest.raises(ValueError, match="not configured"):
        asyncio.run(run_plan(backend, MODELS[:1], rows, tmp_path / "x.jsonl", events=events))
    with pytest.raises(ValueError, match="not in the events file"):
        asyncio.run(run_plan(backend, MODELS, rows, tmp_path / "x.jsonl", events=events[:2]))
    for kw, match in (({"concurrency": 0}, "concurrency"), ({"limit": -1}, "limit"), ({"max_cost": -1.0}, "max_cost")):
        with pytest.raises(ValueError, match=match):
            asyncio.run(run_plan(backend, MODELS, rows, tmp_path / "x.jsonl", events=events, **kw))
    assert backend.calls == [] and not (tmp_path / "x.jsonl").exists()


def test_on_start_reports_what_the_run_found_before_the_first_call(tmp_path):
    events, rows = fixture_plan()
    out = tmp_path / "keys.jsonl"
    run(ScriptedBackend(lambda m, r: RuntimeError("down") if r.metadata["item"] == rows[1]["id"] else "No idea."),
        rows[:5], out, events)
    seen: list[tuple[RunStart, int]] = []
    backend = ScriptedBackend()
    run(backend, rows, out, events, limit=3, on_start=lambda s: seen.append((s, len(backend.calls))))
    start, calls_then = seen[0]
    assert calls_then == 0 and start.planned == len(rows) and start.answered == 4 and start.errors == 1
    assert start.pending == len(rows) - 4 and start.to_send == 3 and start.regrade == 0 and start.stale == 0
    assert start.cost_usd == pytest.approx(0.04) and start.backend == "scripted"
