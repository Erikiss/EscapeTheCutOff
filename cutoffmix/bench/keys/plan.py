"""The plan of the key experiment (docs/DESIGN_keys.md sections 3 to 6) and its cost estimate."""

from __future__ import annotations

import datetime as _dt
import hashlib
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from cutoffmix.bench.grading import _compact, _plain, normalize_text
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
from cutoffmix.bench.keys.text import _contains_answer, asserted_name, cue_tokens, mask_note, mask_title, offers
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


def _flat(note: str) -> str:
    """A note's letters and digits only, accents folded and casefolded (``normalize_text``): two notes that differ
    only in punctuation, markdown, quotes, whitespace, case or accents are the same note (a wrong key must never be
    the key itself, design section 3), as names are compared letters-only for the wrong key's person."""
    return "".join(ch for ch in normalize_text(note) if ch.isalnum())


# ---- pointers (design section 3): a key that ties [withheld] to another work

_W = r"\[withheld\]"
_SENT = r"[^.;!?\n]"  # stays inside one sentence
_RELATED = (r"(?:related|original|earlier|previous|prior|predecessor|preceding|first|follow-?up|companion|same|other"
            r"|older)")
# The other work that a relation word or a verb speaks of: a work noun at most two words on ("the related OmniSVG
# work", "previous Gadget reports", "previous papers"), or a name or model id right after it ("the original Gadget
# paper", "the earlier VibeThinker-1.5B paper"). The paper's own arXiv listing, submission, abstract page, preprint
# or version, "prior knowledge" and "the project" are no other work.
_WORK_NOUN = r"(?:papers?|works?|models?|series|reports?)(?:'s)?(?![\w\-])"
_NO_WORK = (r"(?:arxiv|listings?|submissions?|abstracts?|pages?|preprints?|versions?|knowledge|information|of|to|for"
            r"|in|on|at|by|with|this|that|these|those|the|its|it|an?)(?![\w\-])")
_NAMED = r"(?-i:[A-Z])(?!rxiv(?!\w))[\w.\-]*"                     # "Gadget", "NEO", "VibeThinker-1.5B"
_MODEL_ID = r"(?=[\w.\-]*\d)(?!v\d)[^\W\d_][\w.\-]{3,}"           # "gadget-1.5b"; a version tag "v1" is none
_CAMEL = r"(?-i:[\w\-]*[a-z][A-Z][\w.\-]*)"                       # "OmniSVG", "AgentDoG"
_OPEN = "[\"'\u201c\u201d\u2018\u2019(]*"
_WORK = (r"\s+" + _OPEN + r"(?:(?:(?!" + _NO_WORK + r")(?:[^\s.;!?]|\.(?=\w))+\s+){0,2}?" + _WORK_NOUN
         + "|" + _NAMED + "|" + _MODEL_ID + ")")
# After "also" + verb: a relation word and its work, a CamelCase or model-id handle, or a named work ("the Gadget
# paper"); "who also led the project", "who also wrote the code release" and "who is also the first author on the
# v1 preprint of this paper" speak of this paper.
_ALSO_WORK = (r"\s+(?:(?:the|an?|his|her|their)\s+)?" + _OPEN + r"(?:" + _RELATED + r"(?!\w)" + _WORK + "|" + _CAMEL
              + "|" + _MODEL_ID + "|" + _NAMED + r"\s+(?:(?:[^\s.;!?]|\.(?=\w))+\s+)?" + _WORK_NOUN + ")")
_POINTER_RELATION = re.compile(
    # "[withheld], who also led the related X work"; "[withheld], who was first author of the earlier X paper";
    # "[withheld], who has served as the first author on previous X reports"
    _W + _SENT + r"{0,120}?(?<!\w)(?:also\s+(?:led|leads|wrote|authored|first-authored|(?:the\s+)?first[\s\-]+author"
    r"(?:\s+(?:of|on))?)" + _ALSO_WORK
    + r"|(?:led|first-authored|authored|(?:the\s+)?first[\s\-]+author\s+(?:of|on))\s+"
    r"(?:the\s+|an?\s+|his\s+|her\s+|their\s+)?" + _RELATED + r"(?!\w)" + _WORK + ")"
    # "the original X paper lists [withheld]"; "the original X paper (2024) had [withheld] as first author";
    # "previous papers in the X series have been first-authored by ... [withheld]"; "the related X work listed
    # [withheld] as first author". The present "has" and "have" describe this paper's own listing ("the original
    # arXiv listing has [withheld] as the first author").
    + r"|(?<!\w)(?:original|earlier|previous|prior|related|predecessor|preceding)(?!\w)" + _WORK + _SENT
    + r"{0,160}?(?<!\w)(?:lists|listed|list|had|credits|credited|names|named|(?:led|written|authored|first-authored)"
    r"\s+by)(?!\w)" + _SENT + r"{0,60}?" + _W
    # "(works that) have been first-authored by researchers such as ... [withheld]"; "this paper was first-authored
    # by [withheld]" speaks of this paper
    + r"|(?<!\w)(?:(?:have|had)\s+(?:\w+\s+)?been|were)\s+first-authored\s+by(?!\w)" + _SENT + r"{0,100}?" + _W
    # "as a best guess based on the original X paper, it may be [withheld]"
    + r"|(?<!\w)based\s+on\s+(?:the|its|their|an?)\s+(?:related|original|earlier|previous|prior|predecessor)(?!\w)"
    + _WORK + _SENT + r"{0,160}?" + _W,
    re.IGNORECASE)
_TITLE_TOKEN_TRIM = ".,;:()[]{}\"'\u201c\u201d\u2018\u2019!?*"


def _handles(event: Event) -> list[str]:
    """Short handles of a paper title (design section 3): the main title before a colon (four letters or more) and
    the CamelCase or model-id tokens of the title of four characters or more ("OmniSVG", "VibeThinker-1.5B",
    "InCoder-32B"); an all-capital acronym alone ("LLMs", "VBVR") is no handle."""
    subject = event.subject or ""
    out: list[str] = []
    if ":" in subject:
        main = subject.split(":", 1)[0].strip()
        if sum(ch.isalpha() for ch in main) >= 4:
            out.append(main)
    for raw in subject.split():
        token = raw.strip(_TITLE_TOKEN_TRIM)
        camel = re.search(r"[a-z][A-Z]", token) is not None
        model_id = re.search(r"[^\W\d_][\-.]?\d|\d[\-.]?[^\W\d_]", token) is not None
        if len(token) >= 4 and (camel or model_id) and token not in out:
            out.append(token)
    return out


def _names_handle(note: str, handle: str) -> bool:
    return re.search(r"(?<![\w\-])" + re.escape(handle) + r"(?![\w\-])", note) is not None


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
    ordered pair of them (donor, recipient); a repeated model or pair is refused. Per pair, in this order:

    - ``dark`` candidates (real paper, donor ``correct``, recipient a miss) lose an item whose key still gives the
      answer away (``leaks``, :func:`~cutoffmix.bench.keys.text.mask_note`), whose key carries no cue token
      (``no_cue``, :func:`~cutoffmix.bench.keys.text.cue_tokens`) or for which no wrong key is left
      (``no_wrong_key``). A wrong key is drawn (seeded) among the donor's other correct, non-leaking notes that carry
      a cue themselves, never the same event, never one with this item's expected answer (compared letters-only, so
      "Jun-sol Kim" and "Junsol Kim" are one person), never one equal to the key on letters and digits (so notes that
      differ only in punctuation, markdown, quotes or whitespace are one note), masked with that event's answer forms
      and this item's title, and never one that still contains this item's answer.
    - A key that ties ``[withheld]`` to another work, or names a short handle of another event of ``events`` with
      the same expected answer, is a ``pointer``; ``related_known`` lists the recipient's stored ``current_d`` and
      ``mcq`` grades on those events. Pointer items are planned all the same.
    - ``min_items`` is applied after these filters: a pair with fewer surviving ``dark`` items gets no ``dark`` and
      no ``both_right`` set (``keyed`` False, listed under ``excluded`` with its counts). Its ``donor_wrong`` and
      ``control`` sets are planned all the same: most wrong answers come from the donors with the fewest dark items.
    - ``donor_wrong`` (real paper, donor ``incorrect``, recipient a miss) and ``control`` (fake paper, donor
      ``incorrect``): ``donor_answer`` is the name the donor gives (``asserted_name``; ``no_name`` without one), kept
      only if the donor's own full note offers it (``offers``; else ``unasserted``).
    - ``both_right``: a seeded sample of ``both_right`` real papers both answered correctly, drawn among those whose
      key passes the leak and cue rules.

    Rows: ``id`` (16 hex of sha1 of "set|donor|recipient|event|arm"), ``set``, ``donor``, ``recipient``,
    ``event_id``, ``arm``, ``recognised`` (the recipient's stored ``mcq`` grade is ``correct``; None without one),
    ``donor_grade``, ``donor_answer`` (the expected answer for ``dark`` and ``both_right``, the offered wrong name
    for the others), ``note`` (None for ``alone``), ``note_event`` (the wrong key's source event, else None),
    ``today``, ``cues`` (the key's cue tokens), ``pointer``, ``related_known`` (the last three None outside ``dark``
    and ``both_right``) and ``messages``. The order and every draw are fixed by the inputs and ``seed``.

    The summary has per pair ``dark_candidates``, the drops ``leaks``, ``no_cue`` (both over the ``dark`` and the
    ``both_right`` candidates), ``no_wrong_key``, ``no_name``, ``unasserted``, the planned ``pointers``, the items
    per set, ``keyed`` and ``calls``; ``funnel`` gives per set the candidates, the drops and what is kept, so the
    counts add up. The drops are also totalled over the pairs.
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
    pair_list = [tuple(p) for p in pairs] if pairs is not None else [(a, b) for a in keys for b in keys if a != b]
    repeated = sorted({f"{a}:{b}" for (a, b), n in Counter(pair_list).items() if n > 1})
    if repeated:
        raise ValueError(f"duplicate pairs: {', '.join(repeated)}")
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

    def answered(event: Event, model: str) -> bool:
        return bool(str(cur[(event.id, model)].get("response") or "").strip())

    real = [e for e in events if not e.control and (e.expected_current or "").strip() and e.question_current]
    fakes = [e for e in events if e.control and e.question_current]
    same_answer: dict[str, list[Event]] = {}
    for e in real:
        same_answer.setdefault(_compact(e.expected_current), []).append(e)

    def pointer(note: str, event: Event, recipient: str) -> tuple[bool, list[dict[str, Any]]]:
        """``(pointer, related_known)`` of a key (design section 3)."""
        plain = _plain(note)
        related = []
        for other in same_answer.get(_compact(event.expected_current), []):
            if other.id != event.id and any(_names_handle(plain, h) for h in _handles(other)):
                known = mcq.get((other.id, recipient))
                related.append({"event_id": other.id, "current_d": grade(other.id, recipient),
                                "mcq": None if known is None else known.get("grade")})
        return bool(related) or _POINTER_RELATION.search(plain) is not None, related

    rows: list[dict[str, Any]] = []
    summary_pairs: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []

    for donor, recipient in pair_list:
        name = f"{donor}->{recipient}"
        funnel: dict[str, dict[str, int]] = {
            "dark": {"candidates": 0, "leaks": 0, "no_cue": 0, "no_wrong_key": 0, "kept": 0},
            "both_right": {"candidates": 0, "leaks": 0, "no_cue": 0, "usable": 0, "sampled": 0},
            "donor_wrong": {"candidates": 0, "no_name": 0, "unasserted": 0, "kept": 0},
            "control": {"candidates": 0, "no_name": 0, "unasserted": 0, "kept": 0},
        }
        info: dict[str, Any] = {"donor": donor, "recipient": recipient, "dark_candidates": 0, "leaks": 0,
                                "no_cue": 0, "no_wrong_key": 0, "pointers": 0, "no_name": 0, "unasserted": 0,
                                **{s: 0 for s in SETS}, "keyed": True, "calls": 0, "funnel": funnel}
        summary_pairs[name] = info
        start = len(rows)

        def add(set_: str, event: Event, *, notes: Mapping[str, tuple[str | None, str | None]],
                donor_answer: str | None, key: Mapping[str, Any] | None = None, donor: str = donor,
                recipient: str = recipient, info: dict[str, Any] = info) -> None:
            recognised_row = mcq.get((event.id, recipient))
            recognised = None if recognised_row is None else recognised_row.get("grade") == "correct"
            for arm in ARMS_BY_SET[set_]:
                note, note_event = notes.get(arm, (None, None))
                rows.append({"id": _item_id(set_, donor, recipient, event.id, arm), "set": set_, "donor": donor,
                             "recipient": recipient, "event_id": event.id, "arm": arm, "recognised": recognised,
                             "donor_grade": grade(event.id, donor), "donor_answer": donor_answer, "note": note,
                             "note_event": note_event, "today": day,
                             "cues": None if key is None else list(key["cues"]),
                             "pointer": None if key is None else key["pointer"],
                             "related_known": None if key is None else [dict(r) for r in key["related_known"]],
                             "messages": _messages(event, day, note)})
            info[set_] += 1
            if key is not None and key["pointer"]:
                info["pointers"] += 1

        def keyed_note(event: Event, drops: dict[str, int], donor: str = donor) -> tuple[str, list[str]] | None:
            """The key and its cue tokens, or None (counted in ``drops``) when it leaks or carries no cue."""
            note, leaked = masked(donor, event)
            if leaked:
                drops["leaks"] += 1
                return None
            cues = cue_tokens(note, event)
            if not cues:
                drops["no_cue"] += 1
                return None
            return note, cues

        # ---- dark, pass 1: the items that survive the leak, cue and wrong-key rules
        dark = [e for e in real if grade(e.id, donor) == "correct" and miss(e.id, recipient) and answered(e, donor)]
        funnel["dark"]["candidates"] = len(dark)
        # wrong keys come from the donor's other correct answers, each masked with its own event's forms
        pool = [e for e in real if grade(e.id, donor) == "correct" and answered(e, donor)]
        survivors: list[tuple[Event, dict[str, Any], tuple[str, str]]] = []
        for event in dark:
            kept = keyed_note(event, funnel["dark"])
            if kept is None:
                continue
            key_note, cues = kept
            # never a note about the same person: names compared letters-only ("Jun-sol Kim" is "Junsol Kim"), as
            # the masking and the pointer rule compare them
            answer = _compact(event.expected_current)
            others = [o for o in pool if o.id != event.id and _compact(o.expected_current) != answer]
            random.Random(f"{seed}|wrong_key|{donor}|{recipient}|{event.id}").shuffle(others)
            wrong: tuple[str, str] | None = None
            for other in others:
                note, other_leaked = masked(donor, other)
                if other_leaked:
                    continue
                note = mask_title(note, event)
                if (cue_tokens(note, other) and _flat(note) != _flat(key_note)
                        and not _contains_answer(note, event)):
                    wrong = (note, other.id)
                    break
            if wrong is None:
                funnel["dark"]["no_wrong_key"] += 1
                continue
            is_pointer, related = pointer(key_note, event, recipient)
            survivors.append((event, {"note": key_note, "cues": cues, "pointer": is_pointer,
                                      "related_known": related}, wrong))
        funnel["dark"]["kept"] = len(survivors)

        # ---- min_items after the filters, then pass 2: the rows
        keyed = len(survivors) >= min_items
        if not keyed:
            info["keyed"] = False
            excluded.append({"donor": donor, "recipient": recipient, "dark": len(survivors),
                             "dark_candidates": len(dark),
                             **{k: funnel["dark"][k] for k in ("leaks", "no_cue", "no_wrong_key")}})
        else:
            for event, key, wrong in survivors:
                full = mask_title(str(cur[(event.id, donor)].get("response") or ""), event)
                add("dark", event, notes={"key": (key["note"], None), "full": (full, None), "wrong_key": wrong,
                                          "blank": (BLANK_NOTE, None)},
                    donor_answer=event.expected_current, key=key)

        # ---- donor_wrong: the donor's wrong name, kept only if its own full note offers it
        for event in real:
            if grade(event.id, donor) != "incorrect" or not miss(event.id, recipient):
                continue
            funnel["donor_wrong"]["candidates"] += 1
            full = _offered_note(cur[(event.id, donor)], event, funnel["donor_wrong"])
            if full is not None:
                add("donor_wrong", event, notes={"full": (full[0], None)}, donor_answer=full[1])

        # ---- both_right: a seeded sample among the candidates whose key passes the leak and cue rules
        if keyed and both_right > 0:
            candidates = [e for e in real if grade(e.id, donor) == "correct" and grade(e.id, recipient) == "correct"
                          and answered(e, donor)]
            funnel["both_right"]["candidates"] = len(candidates)
            usable: list[tuple[Event, str, list[str]]] = []
            for event in candidates:
                kept = keyed_note(event, funnel["both_right"])
                if kept is not None:
                    usable.append((event, *kept))
            funnel["both_right"]["usable"] = len(usable)
            chosen = {e.id for e, _, _ in random.Random(f"{seed}|both_right|{donor}|{recipient}").sample(
                usable, min(both_right, len(usable)))}
            for event, key_note, cues in usable:
                if event.id in chosen:
                    is_pointer, related = pointer(key_note, event, recipient)
                    response = str(cur[(event.id, donor)].get("response") or "")
                    add("both_right", event, notes={"key": (key_note, None),
                                                    "full": (mask_title(response, event), None)},
                        donor_answer=event.expected_current,
                        key={"cues": cues, "pointer": is_pointer, "related_known": related})
            funnel["both_right"]["sampled"] = len(chosen)

        # ---- control: a fake paper the donor named an author for
        for event in fakes:
            if grade(event.id, donor) != "incorrect" or (event.id, recipient) not in cur:
                continue
            funnel["control"]["candidates"] += 1
            full = _offered_note(cur[(event.id, donor)], event, funnel["control"])
            if full is not None:
                add("control", event, notes={"full": (full[0], None)}, donor_answer=full[1])

        funnel["donor_wrong"]["kept"], funnel["control"]["kept"] = info["donor_wrong"], info["control"]
        info["dark_candidates"] = funnel["dark"]["candidates"]
        for drop in ("leaks", "no_cue"):
            info[drop] = funnel["dark"][drop] + funnel["both_right"][drop]
        info["no_wrong_key"] = funnel["dark"]["no_wrong_key"]
        for drop in ("no_name", "unasserted"):
            info[drop] = funnel["donor_wrong"][drop] + funnel["control"][drop]
        info["calls"] = len(rows) - start

    by_set_arm: dict[str, dict[str, int]] = {s: dict(Counter(r["arm"] for r in rows if r["set"] == s)) for s in SETS}
    summary = {
        "today": day, "seed": seed, "models": keys, "min_items": min_items, "both_right": both_right,
        "pairs": summary_pairs, "excluded": excluded, "calls": len(rows), "calls_by_set_arm": by_set_arm,
        "calls_by_recipient": dict(Counter(r["recipient"] for r in rows)),
        **{drop: sum(p[drop] for p in summary_pairs.values())
           for drop in ("leaks", "no_cue", "no_wrong_key", "pointers", "no_name", "unasserted")},
    }
    return rows, summary


def _offered_note(stored: Mapping[str, Any], event: Event, drops: dict[str, int]) -> tuple[str, str] | None:
    """``(full note, donor_answer)`` of a donor's wrong answer (design section 3): the name it gives
    (:func:`~cutoffmix.bench.keys.text.asserted_name`, text in double quotes skipped), kept only if the full note
    (the answer with the title masked) offers it (:func:`~cutoffmix.bench.keys.text.offers`); None, counted in
    ``drops`` as ``no_name`` or ``unasserted``, otherwise (it rules the name out, or names it only in a condition)."""
    response = str(stored.get("response") or "")
    wrong_name = asserted_name(response, event)
    if wrong_name is None:
        drops["no_name"] += 1
        return None
    full = mask_title(response, event)
    if not offers(full, wrong_name, event):
        drops["unasserted"] += 1
        return None
    return full, wrong_name


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
