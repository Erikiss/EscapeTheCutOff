"""Key, not value: does another model's cue unlock a model's own memory? (docs/DESIGN_keys.md)

A text-level analogue of Hazy Research's "Plato's Key" (25 November 2026) for models that are reached only through an
API. For a donor model A that recalled a fact and a recipient model B that missed it, B is asked again alone, with A's
note whose answer is masked (``key``), with A's whole note (``full``), and with A's masked note about another event
(``wrong_key``). Further sets measure whether B adopts a *wrong* donor answer (``donor_wrong``, ``control``) and
whether a note breaks a correct recall (``both_right``).

- :func:`build_plan` turns stored probe results into the planned calls (offline, deterministic for its inputs).
- :func:`estimate_cost` prices them from the usage stored with those results.
- :func:`run_plan` sends and grades them (resumable, bounded concurrency, optional cost cap).
- :func:`build_report` and :func:`render_report` compute the comparisons and evaluate the pre-registered predictions
  of section 7 of the design.

The masking helpers (:func:`mask_title`, :func:`mask_note`) work on the original text through an accent-free,
casefolded copy whose characters keep their positions in the original, so a name written "Pióro" or "Pioro",
"Jun-sol" or "Junsol" is found either way and replaced in place.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import math
import random
import re
import time
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from cutoffmix.bench.grading import (
    _COMMON_SURNAMES,
    _PUNCT_MAP,
    _compile_forms,
    _hits,
    _plain,
    _surname_given,
    _unknown_names,
    grade_paper,
    grade_response,
    normalize_text,
)
from cutoffmix.bench.io import append_jsonl, read_jsonl
from cutoffmix.bench.prompts import probe_messages
from cutoffmix.bench.runner import _drop_partial_line
from cutoffmix.types import Backend, ChatRequest, Event, ModelSpec, ProbeResult

__all__ = [
    "ARMS_BY_SET",
    "KEY_USER_TEMPLATE",
    "PROBE",
    "SETS",
    "TITLE",
    "WITHHELD",
    "asserted_name",
    "build_plan",
    "build_report",
    "estimate_cost",
    "fisher_exact",
    "mask_note",
    "mask_title",
    "mcnemar_exact",
    "render_report",
    "row_cost",
    "run_plan",
    "wilson",
]

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
ARMS_BY_SET: dict[str, tuple[str, ...]] = {
    "dark": ("alone", "key", "full", "wrong_key"),
    "donor_wrong": ("alone", "full"),
    "both_right": ("key", "full"),
    "control": ("alone", "full"),
}
SETS: tuple[str, ...] = tuple(ARMS_BY_SET)
_NOT_CORRECT = frozenset({"abstain", "incorrect", "outdated"})  # a stored miss (``ungraded`` is no outcome)
_MIN_TESTABLE_PAIRED = 10  # paired dark items a comparison needs before a prediction counts as testable
_MIN_TESTABLE_GROUP = 5  # rows per group (recognised / unrecognised, adoption items) for the other predictions
_HARM_LIMIT = 0.10  # prediction 4: harm under 10 %

ProgressCallback = Callable[[int, int, Mapping[str, Any]], None]
"""``progress(done, total, row)`` after every new result row (``total`` = calls this run sends at most)."""


# --------------------------------------------------------------------------------------------------- text helpers


def _fold(text: str) -> tuple[str, list[int]]:
    """``(folded, index)``: ``text`` without accents, casefolded, typographic quotes and dashes as ASCII, and for
    every folded character the position in ``text`` it came from (``_PUNCT_MAP`` maps one character to one)."""
    chars: list[str] = []
    index: list[int] = []
    for i, ch in enumerate(text.translate(_PUNCT_MAP)):
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        for c in base.casefold():
            chars.append(c)
            index.append(i)
    return "".join(chars), index


def _folded(text: str) -> str:
    return _fold(text)[0]


def _span(text: str, index: list[int], start: int, end: int) -> tuple[int, int]:
    """The span of ``text`` that the folded characters ``start:end`` came from (with trailing combining marks)."""
    lo, hi = index[start], index[end - 1] + 1
    while hi < len(text) and unicodedata.combining(text[hi]):
        hi += 1
    return lo, hi


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", _folded(name or "")) if t]


def _loose(tokens: Sequence[str]) -> re.Pattern[str]:
    """A folded-text pattern for a name: a space or hyphen may appear or disappear inside and between its tokens
    ("Jun-sol Kim" = "Junsol Kim" = "Jun Sol Kim"), as in the paper grader's loose forms. The dot of an initial is
    optional inside the name ("M. Pióro" = "M Pióro") and never part of the match at its end, where it is usually
    the full stop of the sentence ("... is Pióro M.")."""
    parts = []
    for i, token in enumerate(tokens):
        core = token.rstrip(".")
        if not core:
            continue
        part = r"[\s\-]?".join(re.escape(ch) for ch in core)
        parts.append(part + (r"\.?" if token != core and i < len(tokens) - 1 else ""))
    return re.compile(r"(?<!\w)" + r"[\s\-]*".join(parts) + r"(?!\w)")


def _letters(token: str) -> int:
    return sum(ch.isalpha() for ch in token)


def _answer_tokens(event: Event) -> list[str]:
    """The name tokens of the expected answer (and of each part of a hyphenated token), as written."""
    out: list[str] = []
    for token in (event.expected_current or "").split():
        token = token.strip(".,;:()'\"")
        for part in [token, *token.split("-")]:
            if _letters(part) >= 2 and part not in out:
                out.append(part)
    return out


def _answer_spans(text: str, event: Event) -> list[tuple[int, int]]:
    """Spans of ``text`` that write the expected answer: any accepted form (``expected_current``, ``aliases``), and
    each name token as a whole word. Tokens of four letters or more are matched accent- and case-free; shorter ones
    ("Li", "He", "Wu") only with their own capitalisation, so the pronoun "he" stays."""
    if event.control or not (event.expected_current or "").strip():
        return []
    folded, index = _fold(text)
    spans: list[tuple[int, int]] = []
    patterns: list[re.Pattern[str]] = []
    for form in [event.expected_current, *event.aliases]:
        tokens = _tokens(form)
        if tokens and sum(_letters(t) for t in tokens) >= 2:
            patterns.append(_loose(tokens))
    for token in _answer_tokens(event):
        if _letters(token) >= 4:
            patterns.append(_loose(_tokens(token)))
        else:  # short tokens: case-sensitive on the original text, also in an accent-free spelling
            plain = "".join(c for c in unicodedata.normalize("NFKD", token) if not unicodedata.combining(c))
            for variant in {token, plain}:
                for m in re.finditer(r"(?<!\w)" + re.escape(variant) + r"(?!\w)", text):
                    spans.append((m.start(), m.end()))
    for pat in patterns:
        for m in pat.finditer(folded):
            spans.append(_span(text, index, m.start(), m.end()))
    return spans


def _title_patterns(subject: str) -> list[re.Pattern[str]]:
    """Folded-text patterns for the paper title: the whole title, the part before a colon when it has at least four
    letters ("SmolVLM", "OLMo 2"), and the part after a colon when it has at least three words; any punctuation or
    spacing between the words."""
    parts = [subject]
    if ":" in subject:
        main, sub = subject.split(":", 1)
        if sum(ch.isalpha() for ch in main) >= 4:
            parts.append(main)
        if len(re.findall(r"\w+", sub)) >= 3:
            parts.append(sub)
    out = []
    for part in parts:
        words = re.findall(r"\w+", _folded(part))
        if words:
            out.append(re.compile(r"(?<!\w)" + r"\W+".join(map(re.escape, words)) + r"(?!\w)"))
    return out


def _title_spans(text: str, event: Event) -> list[tuple[int, int]]:
    if not (event.subject or "").strip():
        return []
    folded, index = _fold(text)
    return [_span(text, index, m.start(), m.end()) for pat in _title_patterns(event.subject)
            for m in pat.finditer(folded)]


def _replace(text: str, spans: Iterable[tuple[int, int, str]]) -> str:
    """``text`` with every span replaced by its placeholder; overlapping spans merge (the answer placeholder wins)."""
    merged: list[list[Any]] = []
    for lo, hi, mark in sorted(spans):
        if merged and lo < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
            if mark == WITHHELD:
                merged[-1][2] = WITHHELD
        else:
            merged.append([lo, hi, mark])
    out, pos = [], 0
    for lo, hi, mark in merged:
        out.append(text[pos:lo])
        out.append(mark)
        pos = hi
    out.append(text[pos:])
    joined = "".join(out)
    joined = re.sub(r"\[withheld\](?:[\s\-]+\[withheld\])+", WITHHELD, joined)
    return re.sub(r"\[title\](?:[\s:\-]+\[title\])+", TITLE, joined)


def mask_title(text: str, event: Event) -> str:
    """``text`` with the paper's title (whole, or its main title or subtitle alone) replaced by :data:`TITLE`."""
    return _replace(text, ((lo, hi, TITLE) for lo, hi in _title_spans(text, event)))


def _contains_answer(text: str, event: Event) -> bool:
    """Whether ``text`` still gives away ``event``'s answer: the paper rule grader reads it as ``correct``, or an
    accepted form or an answer token of three letters or more is still there as a whole word."""
    if event.control or not (event.expected_current or "").strip():
        return False
    ruled = grade_paper(event, PROBE, text) if text.strip() else None
    if ruled is not None and ruled[0] == "correct":
        return True
    folded = _folded(text)
    for form in [event.expected_current, *event.aliases]:
        tokens = _tokens(form)
        if tokens and sum(_letters(t) for t in tokens) >= 3 and _loose(tokens).search(folded):
            return True
    return any(re.search(r"(?<!\w)" + re.escape(_folded(t)) + r"(?!\w)", folded)
               for t in _answer_tokens(event) if _letters(t) >= 3)


def mask_note(text: str, event: Event) -> tuple[str, bool]:
    """``(masked, leaked)``: ``text`` with every form of ``event``'s answer replaced by :data:`WITHHELD` and its
    title by :data:`TITLE` (design section 5); ``leaked`` when the masked text still gives the answer away (see
    :func:`_contains_answer`), e.g. a spelling no form covers."""
    spans = [(lo, hi, WITHHELD) for lo, hi in _answer_spans(text, event)]
    spans += [(lo, hi, TITLE) for lo, hi in _title_spans(text, event)]
    masked = _replace(text, spans)
    return masked, _contains_answer(masked, event)


def _date(event: Event) -> str:
    return event.date if len(event.date or "") >= 7 else "2000-01-01"


_QUOTED = re.compile(r'"[^"\n]{2,300}"|\u201c[^\u201c\u201d\n]{2,300}\u201d|\u201e[^\u201c\u201d\n]{2,300}[\u201c\u201d]')


def asserted_name(response: str, event: Event) -> str | None:
    """The person a (wrong) response names as first author: of the distractors (``reject_aliases``) it asserts and
    the capitalised names the paper grader cannot place (``grading._unknown_names``), the one that comes first in the
    text; None when it names nobody. Text in double quotes is skipped first: a response that guesses from a related
    paper often quotes that paper's title ("Vidu: a Highly Consistent ... Generator"), whose capitalised words are no
    name. This is what the paper grader reads as an ``incorrect`` answer."""
    response = _QUOTED.sub(lambda m: " " * len(m.group(0)), response or "")
    text = normalize_text(_plain(response))
    found: list[tuple[int, str]] = []
    for alias in event.reject_aliases:
        pats = _compile_forms([alias], loose=True)
        strong, _ = _hits(text, {"correct": pats, "outdated": [], "incorrect": []}, _date(event))
        if "correct" in strong:
            at = min((m.start() for p in pats for m in [p.search(text)] if m), default=len(text))
            found.append((at, alias))
    for name in _unknown_names(response or "", event):
        m = next((m for p in _compile_forms([name], loose=True) for m in [p.search(text)] if m), None)
        found.append((m.start() if m else len(text), name))
    return min(found)[1] if found else None


def _asserts(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` asserts ``name`` (a full form, or its surname alone when that surname is distinctive:
    longer than four letters and not common), outside a negated clause."""
    text = normalize_text(_plain(response or ""))
    pats = _compile_forms([name], loose=True)
    surname, _ = _surname_given(name)
    s = normalize_text(surname)
    if len(s) > 4 and s not in _COMMON_SURNAMES:
        pats += _compile_forms([surname], loose=True)
    strong, _ = _hits(text, {"correct": pats, "outdated": [], "incorrect": []}, _date(event))
    return "correct" in strong


# ------------------------------------------------------------------------------------------------------ results


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
                add("dark", event, notes={"key": (key_note, None), "full": (full, None), "wrong_key": wrong},
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


# ------------------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of ``k`` successes in ``n`` trials (``(0.0, 1.0)`` for ``n == 0``)."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: the binomial probability (p = 1/2) of a split of the ``b + c`` discordant pairs
    at least as uneven as the one observed; 1.0 without discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the table ``[[a, b], [c, d]]``: the summed probability of every table with the
    same margins that is no more likely than the observed one."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-9)))


# ------------------------------------------------------------------------------------------------------- report


def _usable(row: Mapping[str, Any] | None) -> bool:
    return row is not None and not row.get("error") and row.get("grade") not in (None, "ungraded")


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci": [round(lo, 4), round(hi, 4)]}


def _paired(items: Mapping[str, Mapping[str, Mapping[str, Any]]], first: str, second: str,
            ok: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
    b = sum(1 for arms in items.values() if ok(arms[first]) and not ok(arms[second]))
    c = sum(1 for arms in items.values() if ok(arms[second]) and not ok(arms[first]))
    return {"b": b, "c": c, "p": round(mcnemar_exact(b, c), 6), "n_items": len(items)}


def _correct(row: Mapping[str, Any]) -> bool:
    return row.get("grade") == "correct"


def build_report(plan: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join plan and results (the last row per id counts) into the numbers of design section 6 and the verdicts on
    the pre-registered predictions of section 7. Rows with an ``error`` or the grade ``ungraded`` are left out of
    every ``n`` and counted separately; adoption only needs an answer (no error, a non-empty response)."""
    last: dict[str, Mapping[str, Any]] = {}
    for row in results:
        last[str(row.get("id"))] = row
    planned = {str(p["id"]): p for p in plan}
    joined = [(p, last.get(i)) for i, p in planned.items()]
    rows = [r for _, r in joined if r is not None]
    totals = {
        "planned": len(planned), "rows": len(rows),
        "missing": sum(1 for _, r in joined if r is None),
        "errors": sum(1 for r in rows if r.get("error")),
        "ungraded": sum(1 for r in rows if not r.get("error") and r.get("grade") == "ungraded"),
        "refusals": sum(1 for r in rows if r.get("grader") == "refusal"),
        "cost_usd": round(sum(row_cost(r) for r in rows), 4),
        "by_recipient": dict(Counter(str(p["recipient"]) for p, r in joined if r is not None)),
    }
    pair_names = list(dict.fromkeys(f"{p['donor']}->{p['recipient']}" for p in plan))

    def pick(pair: str, set_: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any] | None]]:
        return [(p, r) for p, r in joined if f"{p['donor']}->{p['recipient']}" == pair and p["set"] == set_]

    pairs: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    recognised: dict[str, dict[str, Any]] = {}
    adoption: dict[str, dict[str, Any]] = {}
    harm: dict[str, dict[str, Any]] = {}
    pooled_dark: dict[str, dict[str, Mapping[str, Any]]] = {}
    pooled_rec: dict[bool, list[Mapping[str, Any]]] = {True: [], False: []}
    pooled_adopt: dict[str, dict[str, Mapping[str, Any]]] = {}
    pooled_harm: list[Mapping[str, Any]] = []
    for pair in pair_names:
        pairs[pair] = {}
        for set_ in SETS:
            chosen = pick(pair, set_)
            if not chosen:
                continue
            arms = {}
            for arm in ARMS_BY_SET[set_]:
                at = [(p, r) for p, r in chosen if p["arm"] == arm]
                ok = [r for _, r in at if _usable(r)]
                entry = _rate(sum(1 for r in ok if _correct(r)), len(ok))
                entry.update(planned=len(at), missing=sum(1 for _, r in at if r is None),
                             errors=sum(1 for _, r in at if r is not None and r.get("error")),
                             ungraded=sum(1 for _, r in at if r is not None and not r.get("error")
                                          and r.get("grade") == "ungraded"))
                arms[arm] = entry
            pairs[pair][set_] = {"items": len({p["event_id"] for p, _ in chosen}), "arms": arms}
        # dark: paired comparisons on the items with every arm graded, and the split by recognition
        items: dict[str, dict[str, Mapping[str, Any]]] = {}
        flags: dict[str, Any] = {}
        for p, r in pick(pair, "dark"):
            items.setdefault(str(p["event_id"]), {})[str(p["arm"])] = r
            flags[str(p["event_id"])] = p.get("recognised")
        complete = {e: a for e, a in items.items()
                    if all(_usable(a.get(arm)) for arm in ARMS_BY_SET["dark"])}
        if items:
            paired[pair] = {"n_items": len(complete),
                            "key_vs_alone": _paired(complete, "key", "alone", _correct),
                            "key_vs_wrong_key": _paired(complete, "key", "wrong_key", _correct),
                            "full_vs_key": _paired(complete, "full", "key", _correct)}
            for event_id, arms_rows in complete.items():
                pooled_dark[f"{pair}|{event_id}"] = arms_rows
            groups: dict[str, dict[str, Any]] = {}
            for label, flag in (("recognised", True), ("unrecognised", False)):
                group: dict[str, Any] = {}
                for arm in ARMS_BY_SET["dark"]:
                    ok = [a[arm] for e, a in items.items() if flags.get(e) is flag and _usable(a.get(arm))]
                    group[arm] = _rate(sum(1 for r in ok if _correct(r)), len(ok))
                    if arm == "key":
                        pooled_rec[flag].extend(ok)
                groups[label] = group
            recognised[pair] = groups
        # adoption of the donor's wrong name: alone vs full, on answers (no error, a response)
        for set_ in ("donor_wrong", "control"):
            chosen = pick(pair, set_)
            if not chosen:
                continue
            entry_set: dict[str, Any] = {}
            by_item: dict[str, dict[str, Mapping[str, Any]]] = {}
            for arm in ARMS_BY_SET[set_]:
                ok = [r for p, r in chosen if p["arm"] == arm and r is not None and not r.get("error")
                      and str(r.get("response") or "").strip()]
                entry_set[arm] = _rate(sum(1 for r in ok if r.get("adopted")), len(ok))
            for p, r in chosen:
                if r is not None and not r.get("error") and str(r.get("response") or "").strip():
                    by_item.setdefault(str(p["event_id"]), {})[str(p["arm"])] = r
            for event_id, arms_rows in by_item.items():
                if all(a in arms_rows for a in ARMS_BY_SET[set_]):
                    pooled_adopt[f"{pair}|{set_}|{event_id}"] = arms_rows
            adoption.setdefault(pair, {})[set_] = entry_set
        chosen = pick(pair, "both_right")
        if chosen:
            entry_h: dict[str, Any] = {}
            for arm in ARMS_BY_SET["both_right"]:
                ok = [r for p, r in chosen if p["arm"] == arm and _usable(r)]
                entry_h[arm] = _rate(sum(1 for r in ok if not _correct(r)), len(ok))
                if arm == "key":
                    pooled_harm.extend(ok)
            harm[pair] = entry_h

    pooled = {
        "dark_items": len(pooled_dark),
        "full_vs_key": _paired(pooled_dark, "full", "key", _correct) if pooled_dark else None,
        "recognised_key": _rate(sum(1 for r in pooled_rec[True] if _correct(r)), len(pooled_rec[True])),
        "unrecognised_key": _rate(sum(1 for r in pooled_rec[False] if _correct(r)), len(pooled_rec[False])),
        "adoption_items": len(pooled_adopt),
        "adoption_full_vs_alone": _paired(pooled_adopt, "full", "alone", lambda r: bool(r.get("adopted")))
        if pooled_adopt else None,
        "harm_key": _rate(sum(1 for r in pooled_harm if not _correct(r)), len(pooled_harm)),
    }
    report = {"totals": totals, "pairs": pairs, "paired": paired, "recognised": recognised, "adoption": adoption,
              "harm": harm, "pooled": pooled}
    report["predictions"] = _predictions(report)
    return report


def _fmt_rate(entry: Mapping[str, Any] | None) -> str:
    if not entry or not entry.get("n"):
        return "n/a"
    lo, hi = entry["ci"]
    return f"{entry['k']}/{entry['n']} ({100 * entry['rate']:.1f} %, {100 * lo:.1f}-{100 * hi:.1f})"


def _fmt_test(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    return f"b={entry['b']}, c={entry['c']}, p={entry['p']:.4g}"


def _predictions(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Verdicts on the four pre-registered predictions (design section 7), conservative by construction."""
    out: list[dict[str, Any]] = []
    paired = report["paired"]
    pooled = report["pooled"]
    pairs = report["pairs"]
    # 1. key beats alone and wrong_key on dark items, p < 0.05 on (at least) the largest pair
    if paired:
        largest = max(sorted(paired), key=lambda k: paired[k]["n_items"])
        entry = paired[largest]
        if entry["n_items"] < _MIN_TESTABLE_PAIRED:
            verdict = "not testable (n too small)"
        else:
            wins = all(entry[c]["b"] > entry[c]["c"] and entry[c]["p"] < 0.05
                       for c in ("key_vs_alone", "key_vs_wrong_key"))
            verdict = "supported" if wins else "not supported"
        arms = pairs.get(largest, {}).get("dark", {}).get("arms", {})
        evidence = (f"largest pair {largest}, {entry['n_items']} items with all four arms graded; key vs alone "
                    f"{_fmt_test(entry['key_vs_alone'])}; key vs wrong key {_fmt_test(entry['key_vs_wrong_key'])}; "
                    f"correct: key {_fmt_rate(arms.get('key'))}, wrong key {_fmt_rate(arms.get('wrong_key'))}, "
                    f"alone {_fmt_rate(arms.get('alone'))} (expected: key 10-30 %, wrong key under 10 %)")
    else:
        verdict, evidence = "not testable (n too small)", "no dark items in the results"
    out.append({"id": 1, "text": "On dark items, key beats alone and wrong_key (paired McNemar, p < 0.05, largest "
                                 "pair).", "verdict": verdict, "evidence": evidence})
    # 2. recovery concentrated on recognised items (pooled key arm), two-sided Fisher exact p < 0.05
    rec, unrec = pooled["recognised_key"], pooled["unrecognised_key"]
    if rec["n"] < _MIN_TESTABLE_GROUP or unrec["n"] < _MIN_TESTABLE_GROUP:
        verdict = "not testable (n too small)"
        p = None
    else:
        p = fisher_exact(rec["k"], rec["n"] - rec["k"], unrec["k"], unrec["n"] - unrec["k"])
        verdict = "supported" if rec["rate"] > unrec["rate"] and p < 0.05 else "not supported"
    out.append({"id": 2, "text": "The key recovers recognised items (the recipient passes the MCQ) more often than "
                                 "unrecognised ones.", "verdict": verdict,
                "evidence": f"key arm, all pairs: recognised {_fmt_rate(rec)}, unrecognised {_fmt_rate(unrec)}"
                            + (f", Fisher exact p={p:.4g}" if p is not None else "")})
    # 3. full beats key on dark items, and full makes the recipient adopt a wrong donor answer more often than alone
    fk, ad = pooled["full_vs_key"], pooled["adoption_full_vs_alone"]
    if not fk or fk["n_items"] < _MIN_TESTABLE_PAIRED or not ad or ad["n_items"] < _MIN_TESTABLE_GROUP:
        verdict = "not testable (n too small)"
    else:
        ok = fk["b"] > fk["c"] and fk["p"] < 0.05 and ad["b"] > ad["c"] and ad["p"] < 0.05
        verdict = "supported" if ok else "not supported"
    out.append({"id": 3, "text": "full beats key on dark items, and full makes the recipient adopt a wrong donor "
                                 "answer far more often than alone (paired, p < 0.05 each).", "verdict": verdict,
                "evidence": f"full vs key, all pairs ({fk['n_items'] if fk else 0} items): {_fmt_test(fk)}; "
                            f"adoption full vs alone ({ad['n_items'] if ad else 0} items): {_fmt_test(ad)}"})
    # 4. harm under 10 % in both_right under key
    hk = pooled["harm_key"]
    if hk["n"] < _MIN_TESTABLE_PAIRED:
        verdict = "not testable (n too small)"
    else:
        verdict = "supported" if hk["rate"] < _HARM_LIMIT else "not supported"
    out.append({"id": 4, "text": "A key rarely breaks a correct recall (both_right, harm under 10 %).",
                "verdict": verdict, "evidence": f"not correct under key, all pairs: {_fmt_rate(hk)} (the observed "
                                                f"rate must be under 10 %; the interval is reported)"})
    return out


def render_report(report: Mapping[str, Any]) -> str:
    """The report as Markdown: totals, per pair and set the correct rate per arm with its 95 % Wilson interval, the
    paired comparisons, the split by recognition, adoption, harm and the prediction verdicts."""
    t = report["totals"]
    lines = ["# Key, not value: report", "",
             f"Planned calls {t['planned']}, rows {t['rows']} (missing {t['missing']}, errors {t['errors']}, "
             f"ungraded {t['ungraded']}, refusals {t['refusals']}), cost {t['cost_usd']:.4f} USD.", ""]
    lines += ["## Correct answers per arm", "",
              "| pair | set | items | " + " | ".join(ARMS_BY_SET["dark"]) + " |",
              "|---|---|---:|" + "---|" * len(ARMS_BY_SET["dark"])]
    for pair, sets in report["pairs"].items():
        for set_, entry in sets.items():
            cells = [_fmt_rate(entry["arms"].get(arm)) if arm in entry["arms"] else "" for arm in ARMS_BY_SET["dark"]]
            lines.append(f"| {pair} | {set_} | {entry['items']} | " + " | ".join(cells) + " |")
    lines += ["", "## Paired comparisons on dark items (exact McNemar; b = first arm right and second wrong)", "",
              "| pair | items | key vs alone | key vs wrong key | full vs key |", "|---|---:|---|---|---|"]
    for pair, entry in report["paired"].items():
        lines.append(f"| {pair} | {entry['n_items']} | {_fmt_test(entry['key_vs_alone'])} | "
                     f"{_fmt_test(entry['key_vs_wrong_key'])} | {_fmt_test(entry['full_vs_key'])} |")
    lines += ["", "## Dark items by recognition (the recipient passes the MCQ)", "",
              "| pair | group | " + " | ".join(ARMS_BY_SET["dark"]) + " |",
              "|---|---|" + "---|" * len(ARMS_BY_SET["dark"])]
    for pair, groups in report["recognised"].items():
        for label, group in groups.items():
            lines.append(f"| {pair} | {label} | " + " | ".join(_fmt_rate(group.get(a)) for a in ARMS_BY_SET["dark"])
                         + " |")
    lines += ["", "## Adoption of the donor's wrong name", "", "| pair | set | alone | full |", "|---|---|---|---|"]
    for pair, sets in report["adoption"].items():
        for set_, entry in sets.items():
            lines.append(f"| {pair} | {set_} | {_fmt_rate(entry.get('alone'))} | {_fmt_rate(entry.get('full'))} |")
    lines += ["", "## Harm on items both answered correctly (share not correct)", "", "| pair | key | full |",
              "|---|---|---|"]
    for pair, entry in report["harm"].items():
        lines.append(f"| {pair} | {_fmt_rate(entry.get('key'))} | {_fmt_rate(entry.get('full'))} |")
    lines += ["", "## Pre-registered predictions (docs/DESIGN_keys.md section 7)", ""]
    for p in report["predictions"]:
        lines.append(f"{p['id']}. {p['text']} **{p['verdict']}**: {p['evidence']}")
    return "\n".join(lines) + "\n"
