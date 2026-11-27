"""Sending and grading the planned calls of the key experiment (docs/DESIGN_keys.md section 6)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cutoffmix.bench.grading import grade_response
from cutoffmix.bench.io import append_jsonl, read_jsonl
from cutoffmix.bench.keys.common import PROBE, row_cost
from cutoffmix.bench.keys.text import _asserts
from cutoffmix.bench.runner import _drop_partial_line
from cutoffmix.types import Backend, ChatRequest, Event, ModelSpec

ProgressCallback = Callable[[int, int, Mapping[str, Any]], None]
"""``progress(done, total, row)`` after every new result row (``total`` = calls this run sends at most)."""


# ---------------------------------------------------------------------------------------------------------- run


def _existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    return {str(r.get("id")): r for r in read_jsonl(path)}


async def run_plan(
    backend: Backend,
    models: Sequence[ModelSpec],
    plan: Sequence[Mapping[str, Any]],
    out_path: str | Path,
    *,
    events: Mapping[str, Event] | Sequence[Event],
    concurrency: int = 8,
    max_cost: float | None = None,
    limit: int | None = None,
    judges: Sequence[ModelSpec] | None = None,
    max_tokens: int = 4096,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Send the planned calls, grade each answer and append one row per answer to ``out_path`` (JSONL).

    - Resumable: an id that already has a row without ``error`` is not sent again (an interrupted last line is cut
      first); readers keep the last row per id.
    - At most ``concurrency`` calls are in flight. ``limit`` caps the calls this run sends. ``max_cost`` (USD) stops
      sending new calls once the summed :func:`row_cost` of the rows in ``out_path`` (earlier ones and this run's)
      reaches it; calls in flight finish.
    - Each call is ``ChatRequest(messages, max_tokens, temperature=0.0, metadata={event_id, probe: "current_d",
      model, arm, set, donor, item, donor_answer})`` to the recipient model, and the answer is graded with
      :func:`cutoffmix.bench.grading.grade_response` (the paper rules; ``judges`` only when given). A policy refusal
      (``content_filter``) is ``ungraded`` with grader ``refusal``; an exception becomes an ``ungraded`` row with
      ``error``. ``adopted`` (sets ``donor_wrong`` and ``control``): the answer asserts the donor's wrong name and is
      not ``correct``; None elsewhere.

    Returns the rows of every planned id that has one (earlier rows included), in plan order.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be >= 0, got {limit}")
    if max_cost is not None and max_cost < 0:
        raise ValueError(f"max_cost must be >= 0, got {max_cost}")
    by_event = dict(events) if isinstance(events, Mapping) else {e.id: e for e in events}
    specs = {m.key: m for m in models}
    missing_models = sorted({str(r["recipient"]) for r in plan} - set(specs))
    if missing_models:
        raise ValueError(f"the plan names recipient model(s) that are not configured: {missing_models}")
    missing_events = sorted({str(r["event_id"]) for r in plan} - set(by_event))
    if missing_events:
        raise ValueError(f"the plan names {len(missing_events)} event(s) not in the events file, "
                         f"e.g. {missing_events[0]}")
    path = Path(out_path)
    _drop_partial_line(path)
    done = _existing(path)
    spent = sum(row_cost(r) for r in done.values())
    pending = [r for r in plan if str(r["id"]) not in done or done[str(r["id"])].get("error")]
    if limit is not None:
        pending = pending[:limit]
    total, finished = len(pending), 0
    sem = asyncio.Semaphore(concurrency)
    failures: list[Exception] = []

    async def one(item: Mapping[str, Any]) -> dict[str, Any]:
        event = by_event[str(item["event_id"])]
        spec = specs[str(item["recipient"])]
        meta = {"event_id": event.id, "probe": PROBE, "model": spec.key, "arm": item["arm"], "set": item["set"],
                "donor": item["donor"], "item": item["id"], "donor_answer": item.get("donor_answer")}
        req = ChatRequest(messages=[dict(m) for m in item["messages"]], max_tokens=max_tokens, temperature=0.0,
                          metadata=meta)
        base = {k: item[k] for k in ("id", "set", "donor", "recipient", "event_id", "arm")}
        start = time.perf_counter()
        try:
            result = await backend.chat(spec, req)
        except Exception as e:  # every failure is recorded, the run goes on
            return {**base, "response": "", "grade": "ungraded", "hedged": False, "grader": "none", "adopted": None,
                    "latency_s": round(time.perf_counter() - start, 4), "usage": None,
                    "error": f"{type(e).__name__}: {e}"[:500]}
        latency = result.latency_s if result.latency_s > 0 else time.perf_counter() - start
        text = result.text or ""
        if result.finish_reason == "content_filter":  # a policy refusal says nothing about knowledge
            grade, hedged, grader = "ungraded", False, "refusal"
        else:
            grade, hedged, grader = await grade_response(event, PROBE, text, backend=backend, judges=judges)
        adopted = None
        if item["set"] in ("donor_wrong", "control") and item.get("donor_answer"):
            adopted = grade != "correct" and _asserts(text, str(item["donor_answer"]), event)
        return {**base, "response": text, "grade": grade, "hedged": hedged, "grader": grader, "adopted": adopted,
                "latency_s": round(latency, 4), "usage": result.usage, "error": None}

    async def worker(item: Mapping[str, Any]) -> None:
        nonlocal finished, spent
        try:
            row = await one(item)
            append_jsonl(path, row)
            done[str(row["id"])] = row
            spent += row_cost(row)
            finished += 1
            if progress is not None:
                progress(finished, total, row)
        except Exception as e:  # re-raised after the calls in flight finish
            failures.append(e)
        finally:
            sem.release()

    tasks: set[asyncio.Task[None]] = set()
    try:
        for item in pending:
            await sem.acquire()  # bounds the live tasks, not only the calls in flight
            if failures or (max_cost is not None and spent >= max_cost):
                sem.release()
                break
            task = asyncio.create_task(worker(item))
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        await asyncio.gather(*list(tasks))
    finally:
        for task in list(tasks):
            task.cancel()
    if failures:
        raise failures[0]
    return [done[str(r["id"])] for r in plan if str(r["id"]) in done]

