"""The plan of the key experiment (docs/DESIGN_keys.md sections 3 to 6) and its cost estimate."""

from __future__ import annotations

import datetime as _dt
import hashlib
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from cutoffmix.bench.grading import normalize_text
from cutoffmix.bench.keys.common import (
    _KEY_SPLIT,
    BLANK_NOTE,
    _NOT_CORRECT,
    ARMS_BY_SET,
    KEY_USER_TEMPLATE,
    PROBE,
    SETS,
    _grade,
    _index,
    row_cost,
)
from cutoffmix.bench.keys.text import _contains_answer, asserted_name, mask_note, mask_title
from cutoffmix.bench.prompts import probe_messages
from cutoffmix.types import Event, ProbeResult

# --------------------------------------------------------------------------------------------------------- plan


def _item_id(set_: str, donor: str, recipient: str, event_id: str, arm: str) -> str:
    return hashlib.sha1(f"{set_}|{donor}|{recipient}|{event_id}|{arm}".encode()).hexdigest()[:16]


def _messages(event: Event, today: str, note: str | None) -> list[dict[str, Any]]:
    messages = probe_messages(event, PROBE, today=today)
    if note is not None:
        messages[1] = {"role": "user", "content": KEY_USER_TEMPLATE.format(question=event.question_current, note=note)}
    return messages


def build_plan(
    events: Sequence[Event],
    results: Sequence[ProbeResult | Mapping[str, Any]],
    *,
    models: Sequence[str] | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    both_right: int = 10,
    min_items: int = 5,
    seed: int = 0,
    today: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The planned calls (design sections 3 to 6) and a summary.

    ``results`` are stored probe rows (the last row per (event, model, probe) counts; only ``current_d`` and ``mcq``
    are read). ``models`` default to the models of the ``current_d`` rows in order of appearance, ``pairs`` to every
    ordered pair of them (donor, recipient). Per pair: ``dark`` (real paper, donor ``correct``, recipient a miss),
    ``donor_wrong`` (real paper, donor ``incorrect`` with a name, recipient a miss), ``both_right`` (a seeded sample
    of ``both_right`` real papers both answered correctly) and ``control`` (fake paper the donor named an author
    for). A ``dark`` or ``both_right`` item whose key still gives the answer away is left out (``leaks``). A pair with
    fewer than ``min_items`` ``dark`` items gets no ``dark`` or ``both_right`` set (it is listed under ``excluded``);
    its ``donor_wrong`` and ``control`` sets are planned all the same, because they measure something else (whether a
    wrong donor answer is adopted) and most wrong answers come from the donors with the fewest ``dark`` items.

    Rows: ``id`` (16 hex of sha1 of "set|donor|recipient|event|arm"), ``set``, ``donor``, ``recipient``,
    ``event_id``, ``arm``, ``recognised`` (the recipient's stored ``mcq`` grade is ``correct``; None without one),
    ``donor_grade``, ``donor_answer`` (the name the donor gave: the expected answer for the correct sets, the
    asserted wrong name for the others), ``note`` (None for ``alone``), ``note_event`` (the wrong key's source
    event, else None) and ``messages``. The order and every draw are fixed by the inputs and ``seed``.
    """
    if both_right < 0:
        raise ValueError(f"both_right must be >= 0, got {both_right}")
    if min_items < 1:
        raise ValueError(f"min_items must be >= 1, got {min_items}")
    day = today or _dt.date.today().isoformat()
    _dt.date.fromisoformat(day)  # a malformed date fails here, not in the middle of a run
    cur, mcq, order = _index(results)
    keys = list(models) if models is not None else order
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate models: {keys}")
    pair_list = list(pairs) if pairs is not None else [(a, b) for a in keys for b in keys if a != b]
    for a, b in pair_list:
        if a == b:
            raise ValueError(f"a pair needs two different models, got {a}:{b}")
        unknown = [m for m in (a, b) if m not in keys]
        if unknown:
            raise ValueError(f"pair {a}:{b} names model(s) without stored results or not in --models: {unknown}")

    masked_cache: dict[tuple[str, str], tuple[str, bool]] = {}

    def masked(model: str, event: Event) -> tuple[str, bool]:
        key = (model, event.id)
        if key not in masked_cache:
            masked_cache[key] = mask_note(str(cur[(event.id, model)].get("response") or ""), event)
        return masked_cache[key]

    def grade(event_id: str, model: str) -> str | None:
        return _grade(cur.get((event_id, model)))

    def miss(event_id: str, model: str) -> bool:
        return grade(event_id, model) in _NOT_CORRECT

    rows: list[dict[str, Any]] = []
    summary_pairs: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    real = [e for e in events if not e.control and (e.expected_current or "").strip() and e.question_current]
    fakes = [e for e in events if e.control and e.question_current]

    for donor, recipient in pair_list:
        name = f"{donor}->{recipient}"
        info: dict[str, Any] = {"donor": donor, "recipient": recipient, "dark_candidates": 0, "leaks": 0,
                                "no_wrong_key": 0, "no_name": 0, **{s: 0 for s in SETS}, "keyed": True}
        summary_pairs[name] = info

        def add(set_: str, event: Event, *, notes: Mapping[str, tuple[str | None, str | None]],
                donor_answer: str | None, donor: str = donor, recipient: str = recipient,
                info: dict[str, Any] = info) -> None:
            recognised_row = mcq.get((event.id, recipient))
            recognised = None if recognised_row is None else recognised_row.get("grade") == "correct"
            for arm in ARMS_BY_SET[set_]:
                note, note_event = notes.get(arm, (None, None))
                rows.append({"id": _item_id(set_, donor, recipient, event.id, arm), "set": set_, "donor": donor,
                             "recipient": recipient, "event_id": event.id, "arm": arm, "recognised": recognised,
                             "donor_grade": grade(event.id, donor), "donor_answer": donor_answer, "note": note,
                             "note_event": note_event, "messages": _messages(event, day, note)})
            info[set_] += 1

        dark = [e for e in real if grade(e.id, donor) == "correct" and miss(e.id, recipient)
                and str(cur[(e.id, donor)].get("response") or "").strip()]
        info["dark_candidates"] = len(dark)
        keyed = len(dark) >= min_items
        if not keyed:
            excluded.append({"donor": donor, "recipient": recipient, "dark": len(dark)})
            info["keyed"] = False
        else:
            # wrong keys come from the donor's other correct answers, each masked with its own event's forms
            pool = [e for e in real if grade(e.id, donor) == "correct"
                    and str(cur[(e.id, donor)].get("response") or "").strip()]
            for event in dark:
                key_note, leaked = masked(donor, event)
                if leaked:
                    info["leaks"] += 1
                    continue
                answer = normalize_text(event.expected_current)
                others = [o for o in pool if o.id != event.id and normalize_text(o.expected_current) != answer]
                random.Random(f"{seed}|wrong_key|{donor}|{recipient}|{event.id}").shuffle(others)
                wrong: tuple[str, str] | None = None
                for other in others:
                    note, other_leaked = masked(donor, other)
                    note = mask_title(note, event)
                    if not other_leaked and not _contains_answer(note, event):
                        wrong = (note, other.id)
                        break
                if wrong is None:
                    info["no_wrong_key"] += 1
                    continue
                full = mask_title(str(cur[(event.id, donor)].get("response") or ""), event)
                add("dark", event, notes={"key": (key_note, None), "full": (full, None), "wrong_key": wrong,
                                          "blank": (BLANK_NOTE, None)},
                    donor_answer=event.expected_current)

        for event in real:
            if grade(event.id, donor) != "incorrect" or not miss(event.id, recipient):
                continue
            response = str(cur[(event.id, donor)].get("response") or "")
            wrong_name = asserted_name(response, event)
            if wrong_name is None:
                info["no_name"] += 1
                continue
            add("donor_wrong", event, notes={"full": (mask_title(response, event), None)}, donor_answer=wrong_name)

        if keyed and both_right > 0:
            candidates = [e for e in real if grade(e.id, donor) == "correct" and grade(e.id, recipient) == "correct"
                          and str(cur[(e.id, donor)].get("response") or "").strip()]
            usable = []
            for event in candidates:
                if masked(donor, event)[1]:
                    info["leaks"] += 1
                else:
                    usable.append(event)
            chosen = set(e.id for e in random.Random(f"{seed}|both_right|{donor}|{recipient}").sample(
                usable, min(both_right, len(usable))))
            for event in usable:
                if event.id in chosen:
                    response = str(cur[(event.id, donor)].get("response") or "")
                    add("both_right", event, notes={"key": (masked(donor, event)[0], None),
                                                    "full": (mask_title(response, event), None)},
                        donor_answer=event.expected_current)

        for event in fakes:
            if grade(event.id, donor) != "incorrect" or (event.id, recipient) not in cur:
                continue
            response = str(cur[(event.id, donor)].get("response") or "")
            fake_name = asserted_name(response, event)
            if fake_name is None:
                info["no_name"] += 1
                continue
            add("control", event, notes={"full": (mask_title(response, event), None)}, donor_answer=fake_name)
        info["calls"] = sum(1 for r in rows if r["donor"] == donor and r["recipient"] == recipient)

    by_set_arm: dict[str, dict[str, int]] = {s: dict(Counter(r["arm"] for r in rows if r["set"] == s)) for s in SETS}
    summary = {
        "today": day, "seed": seed, "models": keys, "min_items": min_items, "both_right": both_right,
        "pairs": summary_pairs, "excluded": excluded, "calls": len(rows), "calls_by_set_arm": by_set_arm,
        "calls_by_recipient": dict(Counter(r["recipient"] for r in rows)),
        "leaks": sum(p["leaks"] for p in summary_pairs.values()),
    }
    return rows, summary


# --------------------------------------------------------------------------------------------------------- cost


def _prompt_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


def _base_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    """Characters of the plain probe these messages extend (the note block of a keyed arm cut off)."""
    return sum(len(str(m.get("content", "")).split(_KEY_SPLIT, 1)[0]) for m in messages)


def _prompt_share(row: Mapping[str, Any]) -> float | None:
    details = ((row.get("usage") or {}).get("cost_details") or {})
    total, prompt = details.get("upstream_inference_cost"), details.get("upstream_inference_prompt_cost")
    if isinstance(total, (int, float)) and isinstance(prompt, (int, float)) and total > 0:
        return max(0.0, min(1.0, float(prompt) / float(total)))
    return None


def estimate_cost(plan: Sequence[Mapping[str, Any]],
                  results: Sequence[ProbeResult | Mapping[str, Any]]) -> dict[str, Any]:
    """Estimated cost of the planned calls from the stored ``current_d`` usage of each recipient.

    Approximation: a stored call's cost (:func:`row_cost`) is split into a prompt part and a completion part by the
    shares in ``usage.cost_details`` (upstream prompt cost / upstream total; a share of 0.25 when the split is not
    stored). A planned call costs the recipient's mean completion part plus the prompt part of the stored call on the
    same event (the recipient's mean when there is none), scaled by the planned prompt's characters over those of the
    plain probe it extends. Hidden reasoning is in the completion part and is assumed to be unchanged by the note.
    """
    cur, _, _ = _index(results)
    per_model: dict[str, list[tuple[float, float]]] = {}
    per_call: dict[tuple[str, str], tuple[float, float]] = {}
    for (event_id, model), row in cur.items():
        if not row.get("usage"):
            continue
        cost = row_cost(row)
        share = _prompt_share(row)
        split = (cost * (0.25 if share is None else share), cost * (1 - (0.25 if share is None else share)))
        per_model.setdefault(model, []).append(split)
        per_call[(event_id, model)] = split
    by_model: dict[str, dict[str, Any]] = {}
    total = 0.0
    for row in plan:
        model = str(row["recipient"])
        stored = per_model.get(model) or []
        mean_prompt = sum(p for p, _ in stored) / len(stored) if stored else 0.0
        mean_completion = sum(c for _, c in stored) / len(stored) if stored else 0.0
        prompt = per_call.get((str(row["event_id"]), model), (mean_prompt, mean_completion))[0]
        messages = row.get("messages") or []
        ratio = _prompt_chars(messages) / max(_base_chars(messages), 1)
        value = mean_completion + prompt * ratio
        entry = by_model.setdefault(model, {"calls": 0, "mean_stored_cost_usd": round(mean_prompt + mean_completion, 6),
                                            "stored_calls": len(stored), "estimate_usd": 0.0})
        entry["calls"] += 1
        entry["estimate_usd"] += value
        total += value
    for entry in by_model.values():
        entry["estimate_usd"] = round(entry["estimate_usd"], 4)
    return {"by_model": by_model, "calls": len(plan), "total_usd": round(total, 4),
            "method": "stored current_d usage per recipient; prompt part scaled by the planned prompt length"}

