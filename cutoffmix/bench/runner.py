"""Benchmark probe runner and cost estimate (DESIGN.md §3.5, evaluation side).

:func:`run_probes` asks every model every applicable probe of every event, grades each answer as it arrives and
appends one :class:`~cutoffmix.types.ProbeResult` JSON line per answer to ``out_path``, so an interrupted run resumes
where it stopped. Grading: MCQ deterministically, open probes by rules, and judges only for answers the rules are
unsure about (``ungraded`` when no judges are given, for a later :func:`cutoffmix.bench.grading.grade_results`
pass).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import logging
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from cutoffmix.bench.grading import grade_response
from cutoffmix.bench.io import append_jsonl, read_jsonl
from cutoffmix.bench.prompts import probe_messages
from cutoffmix.types import Backend, ChatRequest, Event, Grade, ModelSpec, ProbeKind, ProbeResult

__all__ = ["BATCH_SUFFIX", "DEFAULT_PRICING", "PROBES", "ProgressCallback", "estimate_cost", "run_probes"]

PROBES: tuple[str, ...] = ("current", "current_d", "recall", "mcq")
BATCH_SUFFIX = ":batch"
BATCH_DISCOUNT = 0.5  # OpenRouter `:batch` variants cost 50% of list price
# OpenRouter list prices, USD per million tokens (input, output), checked 2026-11-24; keyed by provider id.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "anthropic/claude-fable-5.2": (8.0, 40.0),
    "openai/gpt-6.2-sol": (2.0, 10.0),
    "google/gemini-4-neon": (3.0, 15.0),
}

ProgressCallback = Callable[[int, int, ProbeResult], None]
"""``progress(done, total, result)`` is called after every new result (``total`` = calls this run makes)."""

_Key = tuple[str, str, str]  # (event_id, model key, probe)

log = logging.getLogger(__name__)


def _validate(models: Sequence[ModelSpec], probes: Sequence[str]) -> None:
    unknown = [p for p in probes if p not in PROBES]
    if unknown:
        raise ValueError(f"unknown probe(s) {unknown}; expected a subset of {PROBES}")
    keys = [m.key for m in models]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate model keys: {keys}")


def _probe_messages(event: Event, probe: str, today: str, deployment_system: str | None) -> list[dict] | None:
    """The probe's chat messages, or ``None`` if the event has no question for this probe (or, for MCQ, no valid
    answer letter, which is logged because it is a data error)."""
    if probe == "mcq" and event.mcq_choices:
        letters = {c.strip()[:1].upper() for c in event.mcq_choices}
        if event.mcq_answer.strip().upper()[:1] not in letters:
            log.warning("event %s: mcq_answer %r is not one of its choices; MCQ probe skipped",
                        event.id, event.mcq_answer)
            return None
    try:
        return probe_messages(event, probe, today=today, deployment_system=deployment_system)
    except ValueError:
        return None


def _drop_partial_line(path: Path) -> None:
    """Cut an interrupted last line (no trailing newline) so that new lines never get glued onto it."""
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        f.seek(0)
        data = f.read()
        f.truncate(data.rfind(b"\n") + 1)


def _load_existing(path: Path) -> dict[_Key, ProbeResult]:
    """Results already in ``path`` (the last row wins for repeated keys)."""
    if not path.is_file():
        return {}
    out: dict[_Key, ProbeResult] = {}
    for row in read_jsonl(path):
        r = ProbeResult.from_dict(row)
        out[(r.event_id, r.model, r.probe)] = r
    return out


async def _run_one(
    backend: Backend,
    model: ModelSpec,
    call_model: ModelSpec,
    event: Event,
    probe: str,
    messages: list[dict],
    *,
    max_tokens: int,
    judges: Sequence[ModelSpec] | None,
) -> ProbeResult:
    """Ask one probe and grade the answer; API failures become ``ungraded`` rows with ``error`` set."""
    kind = cast(ProbeKind, probe)  # validated by run_probes
    req = ChatRequest(messages=messages, max_tokens=max_tokens, temperature=0.0,
                      metadata={"event_id": event.id, "probe": probe, "model": model.key})
    start = time.perf_counter()
    try:
        result = await backend.chat(call_model, req)
    except Exception as e:  # every failure is recorded, the run goes on
        return ProbeResult(event_id=event.id, model=model.key, probe=kind, response="", grade="ungraded",
                           grader="none", latency_s=time.perf_counter() - start,
                           error=f"{type(e).__name__}: {e}"[:500])
    latency = result.latency_s if result.latency_s > 0 else time.perf_counter() - start
    text = result.text or ""
    grade: Grade
    if result.finish_reason == "content_filter":  # a policy refusal says nothing about knowledge
        grade, hedged, grader = "ungraded", False, "refusal"
    else:
        grade, hedged, grader = await grade_response(event, probe, text, backend=backend, judges=judges)
    return ProbeResult(event_id=event.id, model=model.key, probe=kind, response=text, grade=grade, hedged=hedged,
                       grader=grader, latency_s=round(latency, 4), usage=result.usage)


async def run_probes(
    backend: Backend,
    models: Sequence[ModelSpec],
    events: Sequence[Event],
    probes: Sequence[str],
    out_path: str | os.PathLike[str],
    *,
    today: str | None = None,
    concurrency: int = 8,
    resume: bool = True,
    use_batch_variant: bool = False,
    judges: Sequence[ModelSpec] | None = None,
    deployment_system: str | None = None,
    max_tokens: int = 4096,
    progress: ProgressCallback | None = None,
    retry_errors: bool = True,
) -> list[ProbeResult]:
    """Run ``probes`` x ``models`` over ``events`` and append graded results to ``out_path`` (JSONL) as they arrive.

    - Probes an event has no question for are skipped (no row).
    - ``resume=True`` skips every (event, model, probe) already in ``out_path``, except rows with an API ``error``
      when ``retry_errors`` is true (the new row is appended; readers keep the last row per key).
      ``resume=False`` starts a fresh file (``out_path`` is truncated).
    - At most ``concurrency`` calls are in flight (``asyncio.Semaphore``); judge calls happen inside that slot.
    - ``use_batch_variant`` calls ``provider_id + ":batch"`` (OpenRouter's discounted batch variants, probing only);
      results are still recorded under the model key.
    - ``today`` (ISO date, default: the local date, fixed for the whole run) and ``deployment_system`` shape the
      ``current_d`` system prompt. ``max_tokens`` includes the hidden reasoning budget.
    - Each row records the response, grade, hedged flag, grader, latency, usage and error (if any).

    Returns the results for every requested (event, model, probe) that has a row, previously stored ones included,
    in event x model x probe order.
    """
    _validate(models, probes)
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
    path = Path(out_path)
    day = today or _dt.date.today().isoformat()
    if resume:
        _drop_partial_line(path)
        existing = _load_existing(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        existing = {}

    order: list[_Key] = []
    pending: list[tuple[ModelSpec, Event, str, list[dict]]] = []
    for event in events:
        for model in models:
            for probe in probes:
                messages = _probe_messages(event, probe, day, deployment_system)
                if messages is None:
                    continue
                key = (event.id, model.key, probe)
                order.append(key)
                done = existing.get(key)
                if done is None or (retry_errors and done.error):
                    pending.append((model, event, probe, messages))

    call_models = {m.key: _batch_variant(m) if use_batch_variant else m for m in models}
    results = dict(existing)
    total, finished = len(pending), 0
    sem = asyncio.Semaphore(concurrency)
    failures: list[Exception] = []

    async def worker(model: ModelSpec, event: Event, probe: str, messages: list[dict]) -> None:
        nonlocal finished
        try:
            r = await _run_one(backend, model, call_models[model.key], event, probe, messages,
                               max_tokens=max_tokens, judges=judges)
            append_jsonl(path, r.to_dict())
            results[(r.event_id, r.model, r.probe)] = r
            finished += 1
            if progress is not None:
                progress(finished, total, r)
        except Exception as e:  # re-raised by run_probes after the in-flight calls finish
            failures.append(e)
        finally:
            sem.release()

    tasks: set[asyncio.Task[None]] = set()
    try:
        for item in pending:
            await sem.acquire()  # bounds the number of live tasks, not only the calls in flight
            if failures:
                sem.release()
                break
            task = asyncio.create_task(worker(*item))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        await asyncio.gather(*list(tasks))
    finally:
        for task in list(tasks):
            task.cancel()
    if failures:
        raise failures[0]
    return [results[k] for k in order if k in results]


def _batch_variant(model: ModelSpec) -> ModelSpec:
    if model.provider_id.endswith(BATCH_SUFFIX):
        return model
    return dataclasses.replace(model, provider_id=model.provider_id + BATCH_SUFFIX)


def _price(model: ModelSpec, pricing: Mapping[str, Any]) -> tuple[float, float] | None:
    """``(input, output)`` USD per million tokens for ``model``, looked up by key, then provider id."""
    entry = pricing.get(model.key, pricing.get(model.provider_id))
    if entry is None:
        return None
    if isinstance(entry, Mapping):
        try:
            return float(entry["input"]), float(entry["output"])
        except KeyError as e:
            raise ValueError(f"pricing for {model.key!r} needs 'input' and 'output' (USD per M tokens)") from e
    inp, out = entry
    return float(inp), float(out)


def estimate_cost(
    models: Sequence[ModelSpec],
    events: Sequence[Event],
    probes: Sequence[str],
    *,
    out_tokens: int = 600,
    in_tokens: int = 200,
    pricing: Mapping[str, Any] | None = None,
    batch: bool = False,
) -> dict[str, Any]:
    """Estimate calls, tokens and USD for :func:`run_probes` (judge calls not included).

    ``pricing`` maps a model key or provider id to ``(input, output)`` or ``{"input": .., "output": ..}`` in USD per
    million tokens (e.g. the OpenRouter list prices from a config); without it, :data:`DEFAULT_PRICING` (Fable
    8/40, Sol 2/10, Neon 3/15). ``out_tokens`` must include the hidden reasoning, which is billed as output.
    ``batch=True`` applies the 50% ``:batch`` discount. Only probes the event has a question for are counted.
    Raises ``ValueError`` for a model without a price.
    """
    _validate(models, probes)
    table: Mapping[str, Any] = DEFAULT_PRICING if pricing is None else pricing
    prices = {m.key: _price(m, table) for m in models}
    missing = [k for k, p in prices.items() if p is None]
    if missing:
        raise ValueError(f"no price for model(s) {missing}; pass pricing={{key: (input_usd_per_M, output_usd_per_M)}}")
    day = _dt.date.today().isoformat()
    calls_per_model = sum(1 for e in events for p in probes if _probe_messages(e, p, day, None) is not None)
    factor = BATCH_DISCOUNT if batch else 1.0
    per_model: dict[str, dict[str, Any]] = {}
    for m in models:
        inp, out = prices[m.key]  # type: ignore[misc]
        usd = calls_per_model * (in_tokens * inp + out_tokens * out) / 1e6 * factor
        per_model[m.key] = {"calls": calls_per_model, "in_tokens": calls_per_model * in_tokens,
                            "out_tokens": calls_per_model * out_tokens, "usd_per_m_in": inp * factor,
                            "usd_per_m_out": out * factor, "usd": round(usd, 4)}
    return {
        "calls": calls_per_model * len(models),
        "per_model": per_model,
        "total_usd": round(sum(v["usd"] for v in per_model.values()), 4),
        "assumptions": {"in_tokens": in_tokens, "out_tokens": out_tokens, "batch": batch,
                        "pricing": "custom" if pricing is not None else "built-in OpenRouter list prices"},
    }

