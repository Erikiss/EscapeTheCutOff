"""Probe generation and the V3 validation gates shared by every event builder.

* :func:`generate_probes` asks a generator model (via any :class:`cutoffmix.types.Backend`) to turn free-text
  Wikipedia Current-events items into probe questions (prompt: :data:`cutoffmix.bench.prompts.PROBEGEN_SYSTEM`).
  It needs an API key for a real backend; template sources such as Wikidata do not use it.
* :func:`validate_event` implements the format gates: no answer in a question, no year/month tokens in
  ``question_current``, four unique MCQ choices, a valid answer letter, consistent ids/dates/topics.
* :func:`assemble_mcq`, :func:`name_variants` and :func:`stable_hash` are the deterministic helpers that the
  template builders (:mod:`cutoffmix.bench.sources.wikidata`) share with the generator: the correct letter is chosen
  by a stable hash of the event id, so answer positions are balanced and reproducible.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import json
import logging
import random
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence

from cutoffmix.bench.prompts import probegen_messages
from cutoffmix.taxonomy import EVENT_TYPES, TOPIC_IDS
from cutoffmix.types import Backend, ChatRequest, Event, ModelSpec

log = logging.getLogger(__name__)

LETTERS = ("A", "B", "C", "D")
PREDICTABILITY = ("low", "medium", "high")
MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December")

_YEAR = re.compile(r"(?<!\d)(?:1[89]|20)\d{2}s?(?!\d)")
_MONTH = re.compile(
    r"\b(?:January|February|March|April|June|July|August|September|October|November|December)\b"
    r"|\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)\."
    r"|\bMay\s+\d|\b\d{1,2}(?:st|nd|rd|th)?\s+May\b"
)
_CHOICE_PREFIX = re.compile(r"^\s*\(?([A-Da-d])[).:]\s+")
_SUFFIX = re.compile(r",?\s+(?:Jr\.?|Sr\.?|II|III|IV)$")


def stable_hash(text: str) -> int:
    """A process-independent 64-bit hash (``hash()`` is salted per process)."""
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def month_year(date: str) -> str:
    """``'2026-03-17'`` (or ``'2026-03'``) -> ``'March 2026'``."""
    return f"{MONTH_NAMES[int(date[5:7]) - 1]} {date[:4]}"


def long_date(date: str) -> str:
    """``'2026-03-07'`` -> ``'March 7, 2026'``."""
    return f"{MONTH_NAMES[int(date[5:7]) - 1]} {int(date[8:10])}, {date[:4]}"


def ascii_fold(text: str) -> str:
    """Strip diacritics: ``'Ülle Madise'`` -> ``'Ulle Madise'``."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def name_variants(label: str) -> list[str]:
    """Simple accepted surface forms of a name: the label, without a parenthetical qualifier, without a
    generational suffix (Jr., III), and diacritic-free versions of each. Order-preserving, case-insensitively unique."""
    base = re.sub(r"\s+", " ", label).strip()
    forms = [base, re.sub(r"\s*\([^)]*\)", "", base).strip()]
    forms += [_SUFFIX.sub("", f).strip() for f in list(forms)]
    forms += [ascii_fold(f) for f in list(forms)]
    out: list[str] = []
    seen: set[str] = set()
    for f in forms:
        key = f.casefold()
        if len(f) >= 2 and key not in seen:
            seen.add(key)
            out.append(f)
    return out


def date_tokens(text: str) -> list[str]:
    """Year, decade and month-name tokens in ``text`` (``'May'`` only next to a number, since it is also a verb)."""
    return [m.group(0) for m in _YEAR.finditer(text)] + [m.group(0) for m in _MONTH.finditer(text)]


def assemble_mcq(event_id: str, correct: str, wrong: Sequence[str]) -> tuple[list[str], str]:
    """Build ``["A) ...", ..., "D) ..."]`` and the answer letter from one correct and three wrong options.

    The correct option's letter is ``stable_hash(event_id) % 4``; the wrong options fill the other letters in an
    order shuffled with a second stable hash, so the stale option is not always next to the answer.
    """
    if len(wrong) != 3:
        raise ValueError(f"{event_id}: need exactly 3 wrong options, got {len(wrong)}")
    texts = [correct, *wrong]
    if any(not t.strip() for t in texts) or len({t.strip().casefold() for t in texts}) != 4:
        raise ValueError(f"{event_id}: MCQ options must be 4 unique non-empty strings: {texts}")
    pos = stable_hash(event_id) % 4
    others = [w.strip() for w in wrong]
    random.Random(stable_hash(event_id + "|mcq-order")).shuffle(others)
    ordered = others[:pos] + [correct.strip()] + others[pos:]
    return [f"{LETTERS[i]}) {t}" for i, t in enumerate(ordered)], LETTERS[pos]


def _contains(haystack: str, needle: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(needle.casefold()) + r"(?!\w)", haystack.casefold()) is not None


def validate_event(e: Event) -> list[str]:
    """Return the reasons why ``e`` fails the V3 format gates (an empty list means valid).

    Checks: id, ISO date and matching month, known topic/category/predictability; at least one probe question,
    each with its expected answer (control events may leave it empty: there is no right answer to a fake paper, the
    ``fact`` is what the judges read); no year or month tokens in ``question_current``; no accepted answer alias
    inside any question; MCQ (when present) with exactly four ``"A) "``..``"D) "`` choices, unique texts and a
    valid answer letter; no overlap between ``aliases`` and ``reject_aliases``.
    """
    errors: list[str] = []
    if not e.id.strip():
        errors.append("empty id")
    try:
        _dt.date.fromisoformat(e.date)
        if e.month != e.date[:7]:
            errors.append(f"month {e.month!r} does not match date {e.date!r}")
    except ValueError:
        errors.append(f"date {e.date!r} is not an ISO date (YYYY-MM-DD)")
    if e.topic not in TOPIC_IDS:
        errors.append(f"unknown topic {e.topic!r}")
    if e.category not in EVENT_TYPES:
        errors.append(f"unknown category {e.category!r}")
    if e.predictability not in PREDICTABILITY:
        errors.append(f"bad predictability {e.predictability!r}")

    has_mcq = bool(e.mcq_question or e.mcq_choices or e.mcq_answer)
    if not (e.question_current or e.question_recall or has_mcq):
        errors.append("no probe question")
    if e.question_current and not e.expected_current and not e.control:
        errors.append("question_current without expected_current")
    if e.question_recall and not (e.expected_recall or e.expected_current) and not e.control:
        errors.append("question_recall without an expected answer")
    if e.question_current:
        tokens = date_tokens(e.question_current)
        if tokens:
            errors.append(f"question_current contains date tokens {tokens}")

    questions = [q for q in (e.question_current, e.question_recall, e.mcq_question) if q]
    for alias in e.aliases:
        if len(alias.strip()) >= 3 and any(_contains(q, alias.strip()) for q in questions):
            errors.append(f"answer alias {alias!r} appears in a question")
            break

    if has_mcq:
        if not e.mcq_question:
            errors.append("mcq choices without mcq_question")
        if len(e.mcq_choices) != 4:
            errors.append(f"mcq needs 4 choices, got {len(e.mcq_choices)}")
        else:
            bodies = []
            for letter, choice in zip(LETTERS, e.mcq_choices):
                if not choice.startswith(f"{letter}) ") or not choice[3:].strip():
                    errors.append(f"choice {choice!r} must start with '{letter}) ' and have text")
                bodies.append(choice[3:].strip().casefold())
            if len(set(bodies)) != 4:
                errors.append("mcq choices are not unique")
        if e.mcq_answer not in LETTERS:
            errors.append(f"mcq_answer {e.mcq_answer!r} is not one of A-D")

    accepted = {a.strip().casefold() for a in e.aliases}
    clash = sorted(r for r in e.reject_aliases if r.strip().casefold() in accepted)
    if clash:
        errors.append(f"reject_aliases overlap aliases: {clash}")
    return errors


# --------------------------------------------------------------------------------------
# LLM-generated probes
# --------------------------------------------------------------------------------------


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in a model reply (code fences and surrounding prose are tolerated)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the reply")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in the reply: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("reply JSON is not an object")
    return obj


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Iterable):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def item_event_id(item: Mapping[str, Any]) -> str:
    """Stable id of a Current-events item: ``wp-<date>-<10 hex of sha1(text)>``."""
    digest = hashlib.sha1(str(item["text"]).encode("utf-8")).hexdigest()[:10]
    return f"wp-{item['date']}-{digest}"


def event_from_probe_json(item: Mapping[str, Any], data: Mapping[str, Any]) -> Event | None:
    """Turn the generator's JSON for one Current-events item into an :class:`Event`.

    Returns None when the generator marked the item unusable. Raises ``ValueError`` for malformed output (missing
    questions, not four options, an answer letter that matches no option). The MCQ is re-lettered with
    :func:`assemble_mcq`; the topic is ``item["topic"]`` or :func:`~cutoffmix.bench.sources.wikipedia.ce_item_topic`.
    """
    if data.get("usable") is False:
        return None
    q_current = str(data.get("q_current") or "").strip()
    a_current = str(data.get("a_current") or "").strip()
    if not q_current or not a_current:
        raise ValueError("missing q_current/a_current")
    eid = item_event_id(item)
    stale = str(data.get("stale_answer") or "").strip()

    raw_choices = [_CHOICE_PREFIX.sub("", c) for c in _strings(data.get("mcq_choices"))]
    if len(raw_choices) != 4:
        raise ValueError(f"expected 4 mcq_choices, got {len(raw_choices)}")
    answer = str(data.get("mcq_answer") or "").strip().upper()[:1]
    if answer not in LETTERS:
        raise ValueError(f"mcq_answer {data.get('mcq_answer')!r} is not a letter A-D")
    correct = raw_choices[LETTERS.index(answer)]
    wrong = [c for i, c in enumerate(raw_choices) if i != LETTERS.index(answer)]
    choices, letter = assemble_mcq(eid, correct, wrong)

    reject = name_variants(stale) if stale and stale.casefold() != a_current.casefold() else []
    rejected = {r.casefold() for r in reject}
    alias_src = _strings(data.get("aliases")) + ([a_current] if len(a_current.split()) <= 6 else [])
    aliases = [a for src in alias_src for a in name_variants(src)]
    aliases = [a for a in dict.fromkeys(aliases) if a.casefold() not in rejected]

    category = str(data.get("category") or "other")
    if category not in EVENT_TYPES or category in ("control_alive", "fake_event", "paper", "fake_paper"):
        category = "other"
    predictability = str(data.get("predictability") or "medium")
    topic = item.get("topic")
    if not topic:
        from cutoffmix.bench.sources.wikipedia import ce_item_topic

        topic = ce_item_topic(item)
    sources = item.get("sources") or []
    date = str(item["date"])
    return Event(
        id=eid, date=date, month=date[:7], topic=str(topic), category=category,
        subject=str(data.get("subject") or "").strip(), fact=str(item["text"]),
        predictability=predictability if predictability in PREDICTABILITY else "medium",  # type: ignore[arg-type]
        question_current=q_current, expected_current=a_current,
        question_recall=str(data.get("q_dated") or "").strip(),
        expected_recall=str(data.get("a_dated") or "").strip(),
        mcq_question=str(data.get("mcq_question") or q_current).strip(), mcq_choices=choices, mcq_answer=letter,
        aliases=aliases, reject_aliases=reject,
        source=str(sources[0].get("url", "")) if sources and isinstance(sources[0], Mapping) else "",
        origin="wikipedia",
    )


async def generate_probes(backend: Backend, model: ModelSpec, items: list[dict], *, concurrency: int = 4,
                          max_tokens: int = 4096, reasoning: dict[str, Any] | None = None) -> list[Event]:
    """Generate probes for Current-events ``items`` with ``model`` (at most ``concurrency`` calls in flight).

    Each item gets one chat call (JSON response format, hidden reasoning at effort low and excluded by default; it
    shares ``max_tokens``). Items the generator marks unusable, replies that cannot be parsed, failed calls and
    events failing :func:`validate_event` are skipped and logged. Returns the valid events in input order,
    de-duplicated by id. Use a generator outside the models being measured to avoid style bias.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    sem = asyncio.Semaphore(concurrency)
    effort = dict(reasoning) if reasoning is not None else {"effort": "low", "exclude": True}

    async def one(item: dict) -> Event | None:
        req = ChatRequest(messages=probegen_messages(item), max_tokens=max_tokens, reasoning=effort,
                          response_format={"type": "json_object"},
                          metadata={"purpose": "probegen", "item_id": item_event_id(item)})
        async with sem:
            try:
                result = await backend.chat(model, req)
            except Exception as e:  # one failed call must not abort the batch; it is reported and skipped
                log.warning("probegen %s: call failed: %s", item_event_id(item), e)
                return None
        try:
            event = event_from_probe_json(item, parse_json_object(result.text))
        except ValueError as e:
            log.warning("probegen %s: unusable reply: %s", item_event_id(item), e)
            return None
        if event is None:
            return None
        problems = validate_event(event)
        if problems:
            log.info("probegen %s: rejected by validation: %s", event.id, "; ".join(problems))
            return None
        return event

    results = await asyncio.gather(*(one(item) for item in items))
    unique: dict[str, Event] = {}
    for event in results:
        if event is not None and event.id not in unique:
            unique[event.id] = event
    return list(unique.values())
