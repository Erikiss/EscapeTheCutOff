"""Tests for cutoffmix.bench.runner (probe runs, resume, concurrency, grading hooks, cost estimate)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from cutoffmix.backends.base import BackendError
from cutoffmix.backends.mock import MockBackend, default_mock_truths
from cutoffmix.bench.estimate import fit_index
from cutoffmix.bench.runner import DEFAULT_PRICING, estimate_cost, run_probes
from cutoffmix.bench.simulate import synthetic_events
from cutoffmix.types import ChatRequest, ChatResult, Event, ModelSpec, ProbeResult


def make_event(i: int, *, recall: bool = True, mcq_answer: str = "B") -> Event:
    return Event(
        id=f"ev{i}", date=f"2026-{1 + i % 9:02d}-15", month=f"2026-{1 + i % 9:02d}", topic="government",
        category="office_change", subject=f"Office {i}", fact=f"New Holder{i} replaced Old Holder{i}.",
        question_current=f"Who holds office {i}?", expected_current=f"New Holder{i}",
        question_recall=f"Who took office {i} in 2026?" if recall else "", expected_recall=f"New Holder{i}",
        mcq_question=f"Who holds office {i}?",
        mcq_choices=[f"A) Old Holder{i}", f"B) New Holder{i}", "C) Someone Else", "D) Nobody Known"],
        mcq_answer=mcq_answer, aliases=[f"New Holder{i}"], reject_aliases=[f"Old Holder{i}"],
    )


FABLE = ModelSpec(key="fable", provider_id="anthropic/claude-fable-5.2")
SOL = ModelSpec(key="sol", provider_id="openai/gpt-6.2-sol")
J1, J2 = ModelSpec(key="j1", provider_id="judge/one"), ModelSpec(key="j2", provider_id="judge/two")
Reply = Callable[[ModelSpec, ChatRequest], "ChatResult | Exception"]


def knows_everything(model: ModelSpec, req: ChatRequest) -> ChatResult:
    event_id, probe = req.metadata["event_id"], req.metadata["probe"]
    text = "B" if probe == "mcq" else f"New Holder{event_id[2:]} holds it."
    return ChatResult(text=text, usage={"prompt_tokens": 10, "completion_tokens": 3}, finish_reason="stop",
                      latency_s=0.25)


class ScriptedBackend:
    """Answers chat calls with ``reply`` after ``delay`` seconds; records calls and peak concurrency."""

    def __init__(self, reply: Reply = knows_everything, delay: float = 0.0) -> None:
        self.reply, self.delay = reply, delay
        self.calls: list[tuple[str, ChatRequest]] = []
        self.in_flight = self.max_in_flight = 0

    def stream_continuation(self, model: ModelSpec, req: Any) -> Any:
        raise NotImplementedError

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        self.calls.append((model.provider_id, req))
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            out = self.reply(model, req)
            if isinstance(out, Exception):
                raise out
            return out
        finally:
            self.in_flight -= 1

    def probe_calls(self) -> list[tuple[str, str, str]]:
        return [(pid, r.metadata["event_id"], r.metadata["probe"]) for pid, r in self.calls if "event_id" in r.metadata]


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


async def test_run_grades_appends_and_records_metadata(tmp_path: Path) -> None:
    out = tmp_path / "res" / "results.jsonl"
    backend = ScriptedBackend()
    events = [make_event(i) for i in range(3)]
    results = await run_probes(backend, [FABLE, SOL], events, ["current", "mcq"], out, today="2026-11-24")
    assert len(results) == 12 and len(rows(out)) == 12
    assert {r.grade for r in results} == {"correct"}
    assert {r.grader for r in results} == {"rule", "mcq"}
    assert all(r.latency_s == 0.25 and r.usage == {"prompt_tokens": 10, "completion_tokens": 3} for r in results)
    assert [(r.event_id, r.model, r.probe) for r in results[:4]] == [
        ("ev0", "fable", "current"), ("ev0", "fable", "mcq"), ("ev0", "sol", "current"), ("ev0", "sol", "mcq")]
    req = backend.calls[0][1]
    assert req.max_tokens == 4096 and req.temperature == 0.0
    assert req.metadata == {"event_id": "ev0", "probe": "current", "model": "fable"}


async def test_resume_skips_done_rows_and_survives_a_torn_last_line(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    events = [make_event(i) for i in range(3)]
    first = await run_probes(ScriptedBackend(), [FABLE, SOL], events, ["current", "mcq"], out)
    again = ScriptedBackend()
    assert [r.to_dict() for r in await run_probes(again, [FABLE, SOL], events, ["current", "mcq"], out)] == \
        [r.to_dict() for r in first]
    assert again.calls == []
    # simulate a crash: keep 5 complete rows plus half of the sixth
    lines = out.read_text(encoding="utf-8").splitlines(keepends=True)
    out.write_text("".join(lines[:5]) + lines[5][:20], encoding="utf-8")
    resumed = ScriptedBackend()
    results = await run_probes(resumed, [FABLE, SOL], events, ["current", "mcq"], out)
    assert len(resumed.calls) == 7 and len(results) == 12
    stored = rows(out)  # every line parses again
    assert len(stored) == 12 and len({(r["event_id"], r["model"], r["probe"]) for r in stored}) == 12


async def test_resume_false_starts_a_fresh_file(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    events = [make_event(0)]
    await run_probes(ScriptedBackend(), [FABLE], events, ["current"], out)
    backend = ScriptedBackend()
    await run_probes(backend, [FABLE], events, ["current"], out, resume=False)
    assert len(backend.calls) == 1 and len(rows(out)) == 1


async def test_api_errors_become_ungraded_rows_and_are_retried(tmp_path: Path) -> None:
    out = tmp_path / "results.jsonl"
    events = [make_event(i) for i in range(3)]

    def flaky(model: ModelSpec, req: ChatRequest) -> ChatResult | Exception:
        if req.metadata["event_id"] == "ev1":
            return BackendError("rate limited", status=429, retryable=True)
        return knows_everything(model, req)

    results = await run_probes(ScriptedBackend(flaky), [FABLE], events, ["current"], out)
    failed = [r for r in results if r.error]
    assert len(results) == 3 and len(failed) == 1
    assert failed[0].grade == "ungraded" and "rate limited" in failed[0].error and failed[0].grader == "none"
    no_retry = ScriptedBackend()
    await run_probes(no_retry, [FABLE], events, ["current"], out, retry_errors=False)
    assert no_retry.calls == []
    retry = ScriptedBackend()
    results = await run_probes(retry, [FABLE], events, ["current"], out)
    assert retry.probe_calls() == [(FABLE.provider_id, "ev1", "current")]
    assert all(r.grade == "correct" and r.error is None for r in results)
    assert len(rows(out)) == 4  # the retry is appended; the last row per key wins


async def test_concurrency_is_bounded(tmp_path: Path) -> None:
    backend = ScriptedBackend(delay=0.01)
    events = [make_event(i) for i in range(12)]
    await run_probes(backend, [FABLE, SOL], events, ["current"], tmp_path / "r.jsonl", concurrency=3)
    assert len(backend.calls) == 24 and backend.max_in_flight == 3
    with pytest.raises(ValueError, match="concurrency"):
        await run_probes(backend, [FABLE], events, ["current"], tmp_path / "r2.jsonl", concurrency=0)


async def test_batch_variant_changes_only_the_provider_id(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    results = await run_probes(backend, [FABLE, SOL], [make_event(0)], ["current"], tmp_path / "r.jsonl",
                               use_batch_variant=True)
    assert sorted(pid for pid, _ in backend.calls) == ["anthropic/claude-fable-5.2:batch", "openai/gpt-6.2-sol:batch"]
    assert sorted(r.model for r in results) == ["fable", "sol"]


async def test_judges_are_called_only_for_unsure_answers(tmp_path: Path) -> None:
    def answers(model: ModelSpec, req: ChatRequest) -> ChatResult:
        if model.key in ("j1", "j2"):
            return ChatResult(text='{"label": "incorrect", "hedged": false}')
        sure = req.metadata["event_id"] == "ev0"
        return ChatResult(text="New Holder0." if sure else "Probably Someone Else.")

    events = [make_event(0), make_event(1)]
    backend = ScriptedBackend(answers)
    results = await run_probes(backend, [FABLE], events, ["current"], tmp_path / "a.jsonl", judges=[J1, J2])
    assert [(r.grade, r.grader) for r in results] == [("correct", "rule"), ("incorrect", "judge:j1+j2")]
    assert sorted(pid for pid, _ in backend.calls if pid.startswith("judge/")) == ["judge/one", "judge/two"]
    unjudged = await run_probes(ScriptedBackend(answers), [FABLE], events, ["current"], tmp_path / "b.jsonl")
    assert [(r.grade, r.grader) for r in unjudged] == [("correct", "rule"), ("ungraded", "rule")]
    assert unjudged[1].response == "Probably Someone Else." and unjudged[1].error is None


async def test_refusals_and_empty_answers_are_ungraded(tmp_path: Path) -> None:
    def odd(model: ModelSpec, req: ChatRequest) -> ChatResult:
        if req.metadata["probe"] == "mcq":
            return ChatResult(text="", finish_reason="length")
        return ChatResult(text="I can't help with that.", finish_reason="content_filter")

    results = await run_probes(ScriptedBackend(odd), [FABLE], [make_event(0)], ["current", "mcq"],
                               tmp_path / "r.jsonl")
    assert [(r.probe, r.grade, r.grader) for r in results] == [
        ("current", "ungraded", "refusal"), ("mcq", "ungraded", "mcq")]


async def test_probes_without_a_question_or_with_a_bad_answer_letter_are_skipped(tmp_path: Path) -> None:
    backend = ScriptedBackend()
    events = [make_event(0, recall=False), make_event(1, mcq_answer="Z"), make_event(2)]
    results = await run_probes(backend, [FABLE], events, ["recall", "mcq"], tmp_path / "r.jsonl")
    assert sorted((r.event_id, r.probe) for r in results) == [
        ("ev0", "mcq"), ("ev1", "recall"), ("ev2", "mcq"), ("ev2", "recall")]


async def test_current_d_uses_the_fixed_run_date_and_progress_is_reported(tmp_path: Path) -> None:
    seen: list[tuple[int, int, str]] = []
    backend = ScriptedBackend()
    await run_probes(backend, [FABLE, SOL], [make_event(0), make_event(1)], ["current_d"], tmp_path / "r.jsonl",
                     today="2026-11-24", progress=lambda done, total, r: seen.append((done, total, r.event_id)))
    assert all("2026-11-24" in req.messages[0]["content"] for _, req in backend.calls)
    assert [(d, t) for d, t, _ in seen] == [(1, 4), (2, 4), (3, 4), (4, 4)]


async def test_unexpected_failures_propagate_after_writing_finished_rows(tmp_path: Path) -> None:
    out = tmp_path / "r.jsonl"

    def explode(done: int, total: int, result: ProbeResult) -> None:
        if done == 2:
            raise RuntimeError("progress sink broke")

    with pytest.raises(RuntimeError, match="progress sink broke"):
        await run_probes(ScriptedBackend(), [FABLE], [make_event(i) for i in range(6)], ["current"], out,
                         concurrency=1, progress=explode)
    assert len(rows(out)) == 2


async def test_input_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown probe"):
        await run_probes(ScriptedBackend(), [FABLE], [make_event(0)], ["direct"], tmp_path / "r.jsonl")
    with pytest.raises(ValueError, match="duplicate model"):
        await run_probes(ScriptedBackend(), [FABLE, FABLE], [make_event(0)], ["current"], tmp_path / "r.jsonl")


async def test_end_to_end_with_the_mock_backend(tmp_path: Path) -> None:
    truths = default_mock_truths(latency_s=0.0)
    events = synthetic_events(["ai_ml", "sports"], range(18, 35), 2, seed=3)
    backend = MockBackend(truths, seed=1, events={e.id: e for e in events}, today="2026-11-24")
    models = [ModelSpec(key=k, provider_id=f"mock/{k}") for k in truths]
    results = await run_probes(backend, models, events, ["current", "mcq"], tmp_path / "mock.jsonl", concurrency=16)
    assert len(results) == len(events) * 3 * 2
    assert not [r for r in results if r.grade == "ungraded"]  # the rule grader understands every mock answer
    assert {r.grade for r in results if r.probe == "current"} <= {"correct", "outdated", "abstain"}
    index = fit_index(events, results, bootstrap=0)
    assert set(index.probes) == {"current", "mcq"} and index.models == list(truths)


def test_estimate_cost_uses_builtin_list_prices() -> None:
    events = [make_event(0), make_event(1, recall=False)]
    est = estimate_cost([FABLE, SOL], events, ["current", "recall", "mcq"])
    assert est["calls"] == 2 * 5  # 5 applicable probes per model
    fable = est["per_model"]["fable"]
    assert fable["calls"] == 5 and fable["in_tokens"] == 1000 and fable["out_tokens"] == 3000
    assert fable["usd"] == pytest.approx(5 * (200 * 8 + 600 * 40) / 1e6)
    assert est["per_model"]["sol"]["usd"] == pytest.approx(5 * (200 * 2 + 600 * 10) / 1e6)
    assert est["total_usd"] == pytest.approx(fable["usd"] + est["per_model"]["sol"]["usd"])
    assert DEFAULT_PRICING["google/gemini-4-neon"] == (3.0, 15.0)


def test_estimate_cost_custom_pricing_batch_discount_and_missing_prices() -> None:
    events = [make_event(0)]
    custom = estimate_cost([FABLE], events, ["current"], pricing={"fable": {"input": 1.0, "output": 2.0}},
                           out_tokens=1000, in_tokens=1000, batch=True)
    assert custom["per_model"]["fable"]["usd"] == pytest.approx((1000 * 1.0 + 1000 * 2.0) / 1e6 * 0.5)
    by_provider = estimate_cost([SOL], events, ["current"], pricing={"openai/gpt-6.2-sol": (4.0, 4.0)})
    assert by_provider["per_model"]["sol"]["usd_per_m_out"] == 4.0
    with pytest.raises(ValueError, match="no price"):
        estimate_cost([ModelSpec(key="x", provider_id="vendor/x")], events, ["current"])
