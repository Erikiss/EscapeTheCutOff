"""Shared constants and helpers of the key experiment (docs/DESIGN_keys.md): placeholders, the prompt template,
the sets and arms, and reading stored probe results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from cutoffmix.types import ProbeResult

WITHHELD = "[withheld]"
TITLE = "[title]"
PROBE = "current_d"  # every planned call is the index probe, with or without a note
# Identical for the arms key, full and wrong_key, so that only the note differs between them (design section 4).
KEY_USER_TEMPLATE = (
    "{question}\n\n"
    "Another analyst wrote this note about the question:\n"
    "<<<\n"
    "{note}\n"
    ">>>\n"
    "Use the note only as a pointer and answer from what you know yourself. If you do not know, say so."
)
_KEY_SPLIT = "\n\nAnother analyst wrote this note about the question:\n"
# The content-free note of the ``blank`` arm: the wrapper and the shape of a key, nothing the question does not say
# (design section 4; the "random map" control of Plato's Key). Written for the paper probe of the v1 data.
BLANK_NOTE = f'The first author of the arXiv paper "{TITLE}" is {WITHHELD}.'
ARMS_BY_SET: dict[str, tuple[str, ...]] = {
    "dark": ("alone", "key", "full", "wrong_key", "blank"),
    "donor_wrong": ("alone", "full"),
    "both_right": ("alone", "key", "full"),
    "control": ("alone", "full"),
}
SETS: tuple[str, ...] = tuple(ARMS_BY_SET)
_NOT_CORRECT = frozenset({"abstain", "incorrect", "outdated"})  # a stored miss (``ungraded`` is no outcome)


def _as_dict(row: ProbeResult | Mapping[str, Any]) -> Mapping[str, Any]:
    return row.to_dict() if isinstance(row, ProbeResult) else row


def _index(results: Iterable[ProbeResult | Mapping[str, Any]]) -> tuple[dict[tuple[str, str], Mapping[str, Any]],
                                                                         dict[tuple[str, str], Mapping[str, Any]],
                                                                         list[str]]:
    """The last stored row per (event, model) for ``current_d`` and for ``mcq``, and the models in the order they
    first appear among the ``current_d`` rows."""
    cur: dict[tuple[str, str], Mapping[str, Any]] = {}
    mcq: dict[tuple[str, str], Mapping[str, Any]] = {}
    order: list[str] = []
    for raw in results:
        row = _as_dict(raw)
        probe = row.get("probe")
        key = (str(row.get("event_id")), str(row.get("model")))
        if probe == PROBE:
            cur[key] = row
            if key[1] not in order:
                order.append(key[1])
        elif probe == "mcq":
            mcq[key] = row
    return cur, mcq, order


def _grade(row: Mapping[str, Any] | None) -> str | None:
    return None if row is None else row.get("grade")


def row_cost(row: Mapping[str, Any] | None) -> float:
    """USD a stored or new result row cost: ``usage.cost``, or, for a bring-your-own-key call (OpenRouter reports
    ``cost`` 0 for those), the provider cost in ``usage.cost_details.upstream_inference_cost``; 0 without usage."""
    usage = (row or {}).get("usage") or {}
    cost = usage.get("cost")
    value = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else 0.0
    if value <= 0 and usage.get("is_byok"):
        upstream = (usage.get("cost_details") or {}).get("upstream_inference_cost")
        if isinstance(upstream, (int, float)) and not isinstance(upstream, bool):
            value = float(upstream)
    return value



def prompt_sha(messages: Sequence[Mapping[str, Any]]) -> str:
    """sha1 (hex) of the canonical JSON of a call's messages: a result row records it, so a row answered under other
    messages than the plan's (another seed, date, template or mask) is recognised as stale (design section 6)."""
    canonical = json.dumps([dict(m) for m in messages], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()
