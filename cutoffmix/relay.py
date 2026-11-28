"""The relay: one shared chain of working notes, written segment by segment by the routed model (DESIGN.md §5).

Each segment is a fresh request (static system prompt + the notes so far, see :mod:`cutoffmix.prompts`). For every
segment the router reads the window (question + last N segments) and picks the model whose topic-specific knowledge
cutoff is the most recent; the chosen model's visible paragraph is streamed, cleaned and committed to the notes.
When a segment contains the final-answer marker, or the segment cap is reached, the answer phase runs.

Event protocol of :meth:`Relay.run` (every attempt that streams from a backend is bracketed):

- ``segment_start`` (think) or ``phase`` (answer) opens an attempt; ``data`` carries ``probs``, ``r``, ``attempt``
  and ``strategy``. A retry or re-route of the same segment opens a new attempt with the same ``segment`` index.
- ``delta`` events stream display text. The first characters are buffered so a preamble or an echo of the notes can
  be stripped, and nothing at or after the sentinel (or, while taking notes, the final-answer marker) is ever shown.
- ``segment_end`` closes the attempt. Its ``text`` is authoritative (clients replace the streamed text with it).
  A failed attempt ends with ``data["dropped"] = True`` and a ``reason`` (refused | error | length | empty) and is not
  part of the trace. ``finish_reason`` is the backend's, or why the relay closed the stream itself (``marker``,
  ``mangled``, ``overflow``, ``sentinel`` for a runaway past it, ``timeout``).
- ``error`` when no model is left, then always one final ``done`` whose ``data`` is ``trace.to_dict()`` plus
  ``usage`` (per model), ``usage_total``, ``latency``, ``status`` and bookkeeping. A stream closed before its usage
  chunk (cut, marker, overflow, a runaway after the sentinel, a timeout) has no token counts or cost: such calls are
  listed in ``usage_missing`` (with the ``generation_id`` the backend reported on its first event or in ``done``)
  and ``usage_complete`` is then False, so ``usage_total`` is a lower bound.
- Every attempt has a wall-clock budget that keep-alive comments cannot reset: ``relay.first_content_timeout_s``
  until the first text, ``relay.segment_timeout_s`` for the whole stream (drafts included). An attempt that runs out
  of it fails as a retryable ``error`` (unless its segment was already complete, e.g. after the sentinel).
- When every model left for a think segment returns nothing usable (``empty``: it stopped at once, wrote only the
  sentinel, or only repeated the notes) and the notes are not empty, the notes count as concluded and the answer
  phase runs instead of failing the run.

The trace holds exactly the committed segments (one per non-dropped ``segment_end``, the answer as a segment with
phase ``"answer"``), so it always agrees with the event stream.

Explore preset (``cfg.explore.enabled``, DESIGN.md §7.6): the protocol above is unchanged, and so is every request,
event and ``done`` payload while the preset is off. With it on, every think request carries the structured-notes
system prompt and a per-segment tail (record status plus a lens, check, continuation or next-move directive),
``segment_start.data["explore"]`` badges the segment (``kind``, ``field``, ``tier``, ``cue``), ``segment_end.data``
carries the segment's ``explore`` flags (and ``premature_final`` when a conclusion was set aside), the committed
segments carry ``Segment.explore`` and ``done.data["explore"]`` (= ``trace.to_dict()["explore"]``) holds the §7
metrics. The record status of the tail reminds the writer of the 'Where we stand:' line whenever the previous note
has none (skipped, or removed by the hard cut: ``ExploreFlags.cut``), and the gate waives that line once it has been
the only missing item after :data:`WWS_WAIVER_NOTES` consecutive notes (``explore.wws`` in the metrics).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import math
import random
import re
import time
import uuid
from collections import Counter
from collections.abc import AsyncGenerator, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from cutoffmix.lenses import Field, draw_lens
from cutoffmix.prompts import (
    LABELS,
    answer_system_prompt_rally,
    WWS_ITEM,
    answer_system_prompt_explore,
    check_text,
    clean_segment,
    complete_sentences,
    continue_text,
    cut_at_sentinel,
    lens_text,
    missing_record_items,
    notes_echo_window,
    parse_labels,
    parse_vs_menu,
    pick_vs_direction,
    rally_directive,
    record_status_text,
    relay_system_prompt,
    relay_system_prompt_explore,
    relay_system_prompt_rally,
    relay_tail,
    relay_tail_rally,
    relay_tail_rally_light,
    split_final,
    strip_preamble,
    trim_overlap,
    trim_to_sentence,
    vs_direction_text,
    vs_menu_messages,
    word3gram_jaccard,
)
from cutoffmix.rally import RallyState, parse_note
from cutoffmix.taxonomy import argmax_topic
from cutoffmix.types import (
    Backend,
    ChatRequest,
    ContinuationRequest,
    Decision,
    ExploreFlags,
    ModelSpec,
    Phase,
    RelayEvent,
    RelayTrace,
    Segment,
    SegmentKind,
)

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig, ExploreConfig
    from cutoffmix.router.gate import TimeSensitivityGate
    from cutoffmix.router.index import CutoffIndex
    from cutoffmix.router.router import RouteDecision, Router

__all__ = [
    "BUFFER_CHARS",
    "GARBLE_WORD_SHARE",
    "STAGNATION_SEGMENTS",
    "VS_MENU_MAX_TOKENS",
    "VS_MENU_MODEL",
    "WWS_WAIVER_NOTES",
    "ExploreTrace",
    "RallyTrace",
    "Relay",
    "RelayRuntime",
    "mock_truth_index",
    "to_jsonable",
]

BUFFER_CHARS = 40  # display text held back at the start of a segment so preambles/echoes can be stripped
# Explore preset (DESIGN.md §7.6)
VS_MENU_MODEL = "gpt-6.2-sol"  # the next-move menu side call always runs on Sol (cheapest, honours `seed`)
VS_MENU_MAX_TOKENS = 300  # five menu lines of at most 25 words
STAGNATION_SEGMENTS = 4  # the argmax topic unchanged for this many think segments (plus no new salient token)
GARBLE_WORD_SHARE = 0.6  # a note whose share of dictionary-like words is below this counts as garbled
# The gate waives the 'Where we stand:' line once it has been the only missing item after this many consecutive
# committed notes (DESIGN_explore §6.1): a line the hard cut removed must not hold the record open for good.
WWS_WAIVER_NOTES = 2
_SEED_BITS = 31  # a random explore seed, when neither explore.seed nor the run seed is set
_MARKER_LAG_EXTRA = 8  # display lag beyond the sentinel/marker length (markdown variants: **Final answer:**)
_CUT_FACTOR = 1.5  # stop reading a think segment once its visible text exceeds hard_cut_chars * this
_RUNAWAY_FACTOR = 4  # safety: stop reading even a non-cancellable stream that keeps going past this many hard cuts
_NO_LIMIT = 1_000_000_000  # "no hard cut" for clean_segment on answers
_MAX_BACKOFF_S = 30.0
_ECHO_MIN_CHARS = 12  # same minimum as prompts.trim_overlap
_USAGE_COUNT_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
_CLOSED_FINISHES = ("sentinel", "cut")  # finish reasons of a stream the backend may close before the provider finished
_TRUNCATED_FINISHES = ("length", "cut", "overflow")  # a think paragraph that may end mid-sentence
# Policies that rank drafts by their look-ahead scores. The others (fixed, round_robin, random) pick the same model
# with or without drafts, so speculative drafting with them would only pay for drafts that are thrown away.
_DRAFT_POLICIES = frozenset({"softmax", "thompson", "argmax"})

Outcome = Literal["ok", "refused", "error", "length", "empty"]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def to_jsonable(obj: Any) -> Any:
    """A JSON-safe copy of ``obj``: non-finite floats become ``None``, tuples/sets lists, numpy values plain Python,
    dataclasses dicts and anything else its ``str()``. Use it before ``json.dumps`` (scores can be ``-inf``)."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    tolist = getattr(obj, "tolist", None)  # numpy scalars and arrays
    if callable(tolist):
        return to_jsonable(tolist())
    return str(obj)


def _rounded(values: Mapping[str, float], digits: int) -> dict[str, float]:
    return {k: round(float(v), digits) if math.isfinite(float(v)) else float(v) for k, v in values.items()}


def _add_usage(acc: dict[str, Any], usage: Mapping[str, Any]) -> None:
    """Add every numeric field of ``usage`` (recursively, e.g. ``completion_tokens_details``) into ``acc``."""
    for key, value in usage.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            prev = acc.get(key, 0)
            acc[key] = (prev if isinstance(prev, (int, float)) else 0) + value
        elif isinstance(value, Mapping):
            sub = acc.get(key)
            if not isinstance(sub, dict):
                sub = acc[key] = {}
            _add_usage(sub, value)


def _has_counts(usage: Mapping[str, Any] | None) -> bool:
    """Whether ``usage`` holds token counts or a cost (not just a ``generation_id``)."""
    return usage is not None and any(key in usage for key in _USAGE_COUNT_KEYS)


def _sentinel_regex(sentinel: str) -> re.Pattern[str] | None:
    """Tolerant pattern for mangled sentinels (``<END_STEP>``, ``<<end step>>``); needs an opening bracket."""
    core = sentinel.strip("<>[]{}() ")
    if len(core) < 3:
        return None
    core_re = r"[_ ]".join(re.escape(part) for part in re.split(r"[_ ]", core))
    return re.compile(r"[<\[]{1,2}[ \t]*" + core_re, re.IGNORECASE)


async def _aclose(stream: Any) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        await close()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:16]


# ---- explore helpers (pure) ----

_LABEL_WORDS = frozenset(word for label in LABELS for word in label.split())
_SALIENT = re.compile(r"\d+(?:[.,:]\d+)*|[^\W\d_]+(?:['’-][^\W\d_]+)*", re.UNICODE)
_NUMERIC = re.compile(r"[\d.,:%/+-]+")
_WORDISH = re.compile(r"^[^\W\d_]+(?:['’-][^\W\d_]+)*$", re.UNICODE)
_VOWEL = re.compile(r"[aeiouyAEIOUYÀ-ÿ]")
_PUNCT = "\"'()[]{}.,;:!?…«»„“”‘’*_-/"


def _without(avail: list[str], exclude: Collection[str]) -> list[str]:
    """``avail`` minus ``exclude``, unless that leaves nothing (then the only model left writes anyway)."""
    kept = [m for m in avail if m not in exclude]
    return kept or avail


def _join(items: Sequence[str]) -> str:
    return (", ".join(items[:-1]) + " and " + items[-1]) if len(items) > 1 else "".join(items)


def _lacks(missing: Sequence[str]) -> str:
    """The moderator's grounds for setting a conclusion aside (the ``{why}`` of the continuation text)."""
    return f"the record still lacks {_join(missing)}" if missing else "the moderator does not accept a conclusion yet"


def _argmax(posterior: Mapping[str, float]) -> str | None:
    """The clear argmax topic, or None for an empty or flat posterior (:func:`cutoffmix.taxonomy.argmax_topic`)."""
    return argmax_topic(posterior)


def _entropy_bits(weights: Mapping[str, float]) -> float:
    total = sum(v for v in weights.values() if v > 0)
    if total <= 0:
        return 0.0
    return round(max(0.0, -sum((v / total) * math.log2(v / total) for v in weights.values() if v > 0)), 4)


def _salient_tokens(text: str) -> set[str]:
    """Numbers and capitalised tokens (the cheap proxy for 'the note introduces something new')."""
    return {t for t in _SALIENT.findall(text) if (t[0].isdigit() or t[0].isupper()) and t not in _LABEL_WORDS}


def _word_share(text: str) -> float | None:
    """Share of tokens that look like dictionary words (letters, a vowel or a CJK character); None without tokens.
    Numbers, dates and percentages do not count either way."""
    tokens = [t.strip(_PUNCT) for t in text.split()]
    tokens = [t for t in tokens if t and not _NUMERIC.fullmatch(t)]
    if not tokens:
        return None
    good = sum(1 for t in tokens if _WORDISH.match(t)
               and (len(t) <= 2 or _VOWEL.search(t) or any(ord(c) > 0x2E80 for c in t)))
    return good / len(tokens)


def _wws_key(sentence: str) -> str:
    """The 'best answer' part of a 'Where we stand:' sentence, normalised for flip counting: the text before the
    first comma or semicolon (the confidence and the weakest assumption follow it), lower-cased, letters/digits only."""
    head = re.split(r"[,;]", sentence, maxsplit=1)[0]
    return re.sub(r"[^\w]+", " ", head.lower(), flags=re.UNICODE).strip()


# --------------------------------------------------------------------------------------
# Streaming display filter and final clean-up
# --------------------------------------------------------------------------------------


class _Echo:
    """The notes a think attempt may repeat before its new paragraph, and how to recognise such a repeat.

    Models sometimes echo one or more earlier notes verbatim (or just the end of the last one) before continuing.
    :meth:`strip` removes complete notes repeated at the start (in any order) and then a repeated end of the notes;
    :meth:`repeats` tells whether a text is (still) nothing but a piece of the notes, e.g. an echo in progress or
    one that was cut off.
    """

    def __init__(self, segments: Sequence[str], *, exempt: Callable[[str], bool] | None = None) -> None:
        self.context = notes_echo_window(list(segments))
        self._notes = sorted({n.strip() for n in segments if len(n.strip()) >= _ECHO_MIN_CHARS}, key=len,
                             reverse=True)
        self._exempt = exempt  # the rally: a short quiet note ('Nothing newer: ...') may repeat an earlier one

    def exempt(self, text: str) -> bool:
        """Whether ``text`` is a whole note that may repeat earlier wording (never a partial repeat of an earlier note)."""
        return self._exempt is not None and self._exempt(text)

    def strip(self, text: str) -> str:
        body = text.lstrip()
        if self.exempt(body):  # a quiet rally note as a whole is kept as written, even when it repeats earlier wording
            return body
        while True:
            note = next((n for n in self._notes if body.startswith(n)), None)
            if note is None:
                if self.context and not self.exempt(body):
                    return trim_overlap(self.context, body)
                return body
            body = body[len(note):].lstrip()

    def repeats(self, text: str) -> bool:
        return len(text) >= _ECHO_MIN_CHARS and text in self.context and not self.exempt(text)


class _StreamFilter:
    """Turns the raw text of one streamed attempt into display deltas.

    - Holds back the first :data:`BUFFER_CHARS` characters (longer while they are still a repeat of the notes) and
      then strips a preamble / echoed notes once, fixing where display text starts.
    - Never emits text at or after the sentinel (exact or mangled) or, in the think phase, the final-answer marker.
    - Lags the end of the received text by a few characters, so a sentinel or marker split across deltas is
      recognised before any of it could be shown.
    - Caps think-phase display text at ``hard_cut_chars``, counted from where the display text starts (an echo
      of the notes before it does not count; while the start is open, up to the length of the notes is allowed).

    ``stop_reason`` tells the caller why the remaining stream is no longer needed (``"sentinel"``, ``"mangled"``,
    ``"marker"`` or ``"overflow"``).
    """

    def __init__(self, *, phase: Phase, echo: _Echo, sentinel: str, final_marker: str, hard_cut_chars: int,
                 stop_at_marker: bool) -> None:
        self.phase = phase
        self.echo = echo
        self.sentinel = sentinel
        self.final_marker = final_marker
        self.hard_cut: int | None = hard_cut_chars if phase == "think" else None
        self.stop_at_marker = stop_at_marker
        self.lag = max(len(sentinel), len(final_marker)) + _MARKER_LAG_EXTRA
        self._mangled = _sentinel_regex(sentinel)
        self.raw = ""
        self.start: int | None = None
        self.emitted = 0  # end offset of the display text emitted so far
        self.stop_reason: str | None = None

    def feed(self, piece: str) -> str:
        """Add a raw delta; return the display text that may be shown now (possibly empty)."""
        self.raw += piece
        return self._advance(final=False)

    def flush(self) -> str:
        """The stream is over: return the display text still held back."""
        return self._advance(final=True)

    def excess(self, length: int) -> int:
        """Characters of ``length`` beyond where the display text starts (beyond a possible echo while open)."""
        return length - (self.start if self.start is not None else len(self.echo.context))

    def _visible(self) -> tuple[str, bool]:
        """The raw text that may ever be displayed, and whether it is closed (nothing more will be added)."""
        raw = self.raw
        cut = raw.find(self.sentinel)
        reason = "sentinel" if cut >= 0 else None
        mangled = self._mangled.search(raw) if self._mangled is not None else None
        if mangled is not None and (cut < 0 or mangled.start() < cut):
            cut = mangled.start()
            # the exact sentinel may still be arriving ("<<END_STEP" + ">>"): hide it, but do not stop yet
            reason = None if self.sentinel.startswith(raw[cut:]) else "mangled"
        text = raw[:cut] if cut >= 0 else raw
        closed = cut >= 0
        if self.phase == "think":
            notes, answer = split_final(text, self.final_marker)
            if answer is not None:
                text, closed = notes, True
                if self.stop_at_marker and reason is None:
                    reason = "marker"
        if reason is None and self.hard_cut is not None and self.excess(len(text)) > self.hard_cut * _CUT_FACTOR:
            reason = "overflow"
        if reason is not None and self.stop_reason is None:
            self.stop_reason = reason
        return text, closed

    def _find_start(self, text: str, done: bool) -> int | None:
        if not done and len(text) < BUFFER_CHARS:
            return None
        s = strip_preamble(text)
        if self.phase == "think":
            s = self.echo.strip(s).lstrip()
            # (so far) only a repeat of the notes: wait until the echo ends (while open, also on a short piece,
            # e.g. the first characters of the next echoed note); a finished repeat is never shown
            if s and s in self.echo.context and (not done or self.echo.repeats(s)):
                return len(text) if done else None
        else:
            notes, answer = split_final(s, self.final_marker)
            if answer is not None and not notes.strip():
                s = answer
        if not s.strip():
            return len(text) if done else None
        return text.rfind(s)

    def _advance(self, *, final: bool) -> str:
        text, closed = self._visible()
        done = final or closed
        if self.start is None:
            self.start = self._find_start(text, done)
            if self.start is None:
                return ""
            self.emitted = max(self.emitted, self.start)
        # trailing whitespace (Fable ends with a blank line, Sol puts the sentinel on its own line) is never shown:
        # the committed text is stripped, and streamed text that differs from it makes clients redraw the block
        limit = len(text.rstrip()) if done else len(text) - self.lag
        if self.hard_cut is not None:
            limit = min(limit, self.start + self.hard_cut)
        if limit <= self.emitted:
            return ""
        out = text[self.emitted:limit]
        self.emitted = limit
        return out


def _quiet_shape(text: str) -> bool:
    """A rally note that says nothing more than 'I confirm / know nothing newer', with its closing 'Newest I know:' line:
    the shape of the note the stop rule waits for, which models write in nearly the same words twice in a row."""
    got = parse_note(text)
    labels = got["labels"]
    return (labels["confirm"] + labels["nothing_newer"] + labels["unverified"] > 0
            and labels["update"] == labels["newer"] == labels["known"] == 0 and got["newest_raw"] is not None)


def _finalize_think(raw: str, echo: _Echo, *, sentinel: str, final_marker: str,
                    hard_cut_chars: int) -> tuple[str, str | None, bool, bool, str]:
    """Authoritative ``(segment text, answer after the final marker or None, saw_sentinel, cut, full)`` of a think
    attempt; ``full`` is the paragraph as the model wrote it, before the hard cut (the rally reads its labels and its
    closing 'Newest I know:' line from it).

    Nothing after the first sentinel counts, also a mangled one (a final marker written after it is not an answer).
    Notes repeated before the new paragraph are removed, and a text that is only a piece of the notes (an echo,
    possibly cut off) becomes ``""``. ``cut`` says whether the hard cut to ``hard_cut_chars`` (with its sentence
    trim) shortened the paragraph: a stream that ended normally can still lose its end here, which the explore
    preset must know (a trailing 'Where we stand:' line is the first thing the cut removes).
    """
    pre, saw = cut_at_sentinel(raw, sentinel)
    notes, answer = split_final(pre, final_marker)
    body = echo.strip(strip_preamble(notes if answer is not None else pre))
    full = clean_segment(body, "" if echo.exempt(body) else echo.context, sentinel=sentinel,
                         hard_cut_chars=_NO_LIMIT)[0]
    text = trim_to_sentence(full, hard_cut_chars)
    cut = len(text) < len(full)
    if echo.repeats(text):
        text = ""
    if answer is not None:
        answer = clean_segment(answer, "", sentinel=sentinel, hard_cut_chars=_NO_LIMIT)[0]
    return text, answer, saw, cut, full


def _finalize_answer(raw: str, *, sentinel: str, final_marker: str) -> str:
    """Authoritative text of an answer attempt: cut at the sentinel, preamble and a leading final marker removed."""
    text = clean_segment(raw, "", sentinel=sentinel, hard_cut_chars=_NO_LIMIT)[0]
    notes, answer = split_final(text, final_marker)
    return answer if answer is not None and not notes.strip() else text


# --------------------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------------------


@dataclass
class _Attempt:
    """One streamed call (or draft) and everything learned from it."""

    model: str
    phase: Phase
    index: int
    attempt: int = 0
    raw: str = ""
    text: str = ""
    answer: str | None = None  # think phase: text after the final marker (None without a marker)
    finish: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    retryable: bool = True
    retry_after: float | None = None
    note: str | None = None  # the backend's done.error, e.g. "refused: ..."
    saw_sentinel: bool = False
    cut: bool = False  # think: the committed text is shorter than what was written (hard cut or truncated stream)
    full: str = ""  # think: the paragraph before the hard cut (``text`` is its sentence-trimmed form)
    aborted: bool = False  # the stream was closed (by the relay or the backend) before the provider finished
    streamed: bool = False  # the backend yielded at least one event (a call that failed before it is not billed)
    timed_out: bool = False  # the relay gave up waiting (the provider may have billed it: its usage is unknown)
    generation_id: str | None = None  # reported by the backend on its first event, before any close
    started_at: float = 0.0
    ended_at: float = 0.0
    ttft_s: float | None = None  # time to the first text from the model
    duration_s: float = 0.0
    premature: bool = False  # explore: the attempt concluded while the record gate was closed (answer set aside)
    premature_answer: str | None = None  # the answer that was set aside (restored when the loop stop accepts it)

    @property
    def outcome(self) -> Outcome:
        if self.finish == "content_filter":
            return "refused"
        if self.error is not None or self.finish == "error":
            return "error"
        usable = bool(self.text) or (self.phase == "think" and self.answer is not None)
        if usable:
            return "ok"
        return "length" if self.finish == "length" else "empty"


@dataclass
class _Produced:
    """Result slot filled by a segment generator (async generators cannot return values)."""

    attempt: _Attempt | None = None
    error: str | None = None
    excluded: set[str] = field(default_factory=set)
    outcomes: dict[str, Outcome] = field(default_factory=dict)  # last failed outcome per model


@dataclass
class ExploreTrace(RelayTrace):
    """The trace of an explore run: ``explore`` holds the DESIGN.md §7 metrics once the run has ended, and
    ``to_dict()`` includes them under ``"explore"`` (so ``done.data["explore"]`` is the same object)."""

    explore: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {"explore": self.explore}


@dataclass
class RallyTrace(RelayTrace):
    """The trace of a rally run: ``rally`` holds the DESIGN_rally.md section 6 metrics once the run has ended, and
    ``to_dict()`` includes them under ``"rally"`` (so ``done.data["rally"]`` is the same object)."""

    rally: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {"rally": self.rally}


@dataclass
class _Plan:
    """What the tail of a think segment asks for (DESIGN.md §7.6), decided before its routing decision.

    Deterministic for the run state and the segment index (the lens draw uses ``random.Random(f"{seed}:{index}")``),
    so a re-route of the same segment gets the same plan unless the state changed (a premature conclusion schedules a
    continuation). ``exclude`` names the models the segment is routed away from when another one is available.
    """

    kind: SegmentKind = "main"
    directive: str = ""
    exclude: frozenset[str] = frozenset()
    field: Field | None = None
    tier: str | None = None
    concepts: tuple[str, ...] = ()
    cue: str | None = None
    why: str | None = None
    forcing: dict[str, Any] | None = None  # the pending continuation this segment consumes (identity matters)
    stagnating: bool = False
    wants_menu: bool = False  # a next-move menu side call is due for this segment (resolved once per index)

    def badge(self) -> dict[str, Any]:
        """What ``segment_start.data["explore"]`` shows."""
        return {"kind": self.kind, "field": self.field.label if self.field else None, "tier": self.tier,
                "cue": self.cue}


@dataclass
class _ExploreState:
    """Per-run bookkeeping of the explore preset: the lens/check schedule, the record gate, forced continuations and
    the material for the §7 metrics. ``trace`` and ``notes`` are the run state's own objects."""

    cfg: ExploreConfig
    seed: int
    seed_source: str  # "explore" (explore.seed) | "run" (router.seed) | "random"
    trace: RelayTrace
    notes: list[str]
    preset: str | None = None  # AppConfig.preset: the packaged preset the run's config was loaded with
    pending_check: bool = False
    lens_writer: str | None = None
    lens_field: Field | None = None
    cooldown: int = 0
    lenses_used: int = 0
    used_fields: list[str] = field(default_factory=list)  # field ids, drawn without replacement
    forcings: int = 0  # continuations scheduled so far (the limit counts these)
    forcing: dict[str, Any] | None = None  # pending continuation: {"why", "model", "cue", "index"}
    cue_index: int = 0
    loop_stop: bool = False
    premature_finals: list[dict[str, Any]] = field(default_factory=list)
    first_final_index: int | None = None
    gate_missing_at_final: list[str] | None = None
    window_notes: list[str] = field(default_factory=list)  # the notes the routing window sees (no lens notes)
    wws_lines: list[str] = field(default_factory=list)
    wws_only_streak: int = 0  # consecutive committed notes after which the WWS line was the only missing item
    wws_waivers: int = 0  # how often the streak reached WWS_WAIVER_NOTES (the gate then waives the line)
    wws_waived_at: list[int] = field(default_factory=list)  # the segment index of every waiver granted
    vs_menu: dict[str, Any] = field(default_factory=lambda: {"calls": 0, "parsed": 0, "chosen_p": [], "menus": [],
                                                             "errors": [], "skipped": None})
    vs_cache: dict[int, str] = field(default_factory=dict)  # menu directive per segment index

    # ---- the record ----

    def flags(self, extra: ExploreFlags | None = None) -> list[ExploreFlags]:
        """The flags of the committed think segments, plus ``extra`` (a note about to be committed)."""
        flags = [s.explore for s in self.trace.segments if s.phase == "think" and s.explore is not None]
        return [*flags, extra] if extra is not None else flags

    def main_line_count(self, extra: ExploreFlags | None = None) -> int:
        return sum(1 for f in self.flags(extra) if f.kind in ("main", "continue"))

    def label_counts(self, extra: ExploreFlags | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in self.flags(extra):
            for label, n in f.labels.items():
                counts[label] = counts.get(label, 0) + n
        return counts

    def wws_recent(self, extra: ExploreFlags | None = None) -> bool:
        return any(f.has_wws for f in self.flags(extra)[-self.cfg.where_we_stand_window:])

    def wws_previous(self) -> bool:
        """Whether the last committed note has a 'Where we stand:' line. False (a reminder is due) when the model
        skipped it or the hard cut removed it; True before the first note, whose tail lists the line anyway."""
        flags = self.flags()
        return not flags or flags[-1].has_wws

    def wws_waived(self) -> bool:
        """Whether the gate waives the 'Where we stand:' line: it has been the only missing item after
        :data:`WWS_WAIVER_NOTES` consecutive notes (a line the hard cut removed must not block the gate for good).
        The waiver lapses with the streak, i.e. once a note has the line again or another item goes missing."""
        return self.wws_only_streak >= WWS_WAIVER_NOTES

    def note_committed(self, index: int) -> None:
        """Gate bookkeeping once think segment ``index`` joined the trace: advance or reset the waiver streak (judged
        on the unwaived checklist), and log a waiver when the streak reaches :data:`WWS_WAIVER_NOTES`."""
        if self._missing() == [WWS_ITEM]:
            self.wws_only_streak += 1
            if self.wws_only_streak == WWS_WAIVER_NOTES:
                self.wws_waivers += 1
                self.wws_waived_at.append(index)
        else:
            self.wws_only_streak = 0

    def _missing(self, extra: ExploreFlags | None = None, pending: bool | None = None) -> list[str]:
        """The checklist without the waiver (what :func:`missing_record_items` says)."""
        return missing_record_items(main_line_count=self.main_line_count(extra), min_segments=self.cfg.min_segments,
                                    label_counts=self.label_counts(extra), require_labels=self.cfg.require_labels,
                                    wws_recent=self.wws_recent(extra),
                                    pending_check=self.pending_check if pending is None else pending)

    def record_missing(self, *, after: tuple[SegmentKind, str] | None = None) -> list[str]:
        """The checklist items still failing (DESIGN_explore §6.1); empty when the gate is open. With ``after``
        (kind and text of a note about to be committed) the record is judged as it will be once that note is in.
        While :meth:`wws_waived`, the 'Where we stand:' item is left out (the tail still reminds of the line)."""
        extra: ExploreFlags | None = None
        pending = self.pending_check
        if after is not None and after[1]:
            kind, text = after
            parsed = parse_labels(text)
            extra = ExploreFlags(kind=kind, labels=dict(parsed["counts"]), has_wws=bool(parsed["has_wws"]))
            pending = True if kind == "lens" else False if kind == "check" else pending
        missing = self._missing(extra, pending)
        if self.wws_waived() and WWS_ITEM in missing:
            missing.remove(WWS_ITEM)
        return missing

    def gate_open(self) -> bool:
        return not self.record_missing()

    def accepting(self) -> bool:
        """Whether a conclusion written now is accepted: the gate is open, forcing stopped on a loop, or the
        continuation limit is used up (the next conclusion is then accepted and the missing items logged)."""
        return self.gate_open() or self.loop_stop or self.forcings >= self.cfg.max_forced_continuations

    def next_cue(self) -> str:
        cue = self.cfg.cues[self.cue_index % len(self.cfg.cues)]
        self.cue_index += 1
        return cue

    # ---- topics ----

    def topics(self) -> list[str | None]:
        """The argmax topic of every committed think segment (its decision's smoothed posterior)."""
        return [_argmax(s.topic_posterior) for s in self.trace.segments if s.phase == "think"]

    def current_topic(self) -> str | None:
        topics = self.topics()
        return topics[-1] if topics else None

    def stagnating(self) -> bool:
        """The cheap proxy of DESIGN_explore §4.2: the argmax topic unchanged for :data:`STAGNATION_SEGMENTS`
        think segments and the last note introducing no number or capitalised token the notes did not have."""
        topics = self.topics()
        if len(topics) < STAGNATION_SEGMENTS or len(self.notes) < 2:
            return False
        last = topics[-STAGNATION_SEGMENTS:]
        if last[0] is None or any(t != last[0] for t in last):
            return False
        earlier = _salient_tokens("\n".join(self.notes[:-1]))
        return not (_salient_tokens(self.notes[-1]) - earlier)

    # ---- metrics (DESIGN_explore §7) ----

    def metrics(self) -> dict[str, Any]:
        think = [s for s in self.trace.segments if s.phase == "think"]
        flags = [s.explore or ExploreFlags() for s in think]
        topics = self.topics()
        main = [i for i, f in enumerate(flags) if f.kind in ("main", "continue")]
        seen = [t for t in topics if t is not None]
        posteriors = [s.topic_posterior for s in think if s.topic_posterior]
        mean: dict[str, float] = {}
        for p in posteriors:
            for topic, v in p.items():
                mean[topic] = mean.get(topic, 0.0) + v / len(posteriors)
        lenses = [f for f in flags if f.kind == "lens"]
        checks = [f for f in flags if f.kind == "check"]
        kept = sum(1 for f in checks if f.lens_verdict == "kept")
        dropped = sum(1 for f in checks if f.lens_verdict == "dropped")
        written = sum(1 for f in lenses if f.lens_prefix_ok)
        noted = [f for f, s in zip(flags, think) if s.text]
        items = [sum(n for label, n in f.labels.items() if label != "Where we stand") for f in noted]
        keys = [_wws_key(w) for w in self.wws_lines]
        cfg = dataclasses.asdict(self.cfg)
        cfg["require_labels"], cfg["cues"] = list(self.cfg.require_labels), list(self.cfg.cues)
        cfg["lens"]["tiers"] = dict(self.cfg.lens.tiers)
        return {
            "preset": self.preset,
            "seed": self.seed,
            "seed_source": self.seed_source,
            "config": cfg,
            "think_segments": len(think),
            "main_line_segments": len(main),
            "lens_segments": len(lenses),
            "check_segments": len(checks),
            "continue_segments": sum(1 for f in flags if f.kind == "continue"),
            "kinds": [f.kind for f in flags],
            "topics": topics,
            "topics_visited": {"all": len(set(seen)), "main_line": len({topics[i] for i in main} - {None})},
            "topic_entropy_bits": {"mean_posterior": _entropy_bits(mean),
                                   "argmax_histogram": _entropy_bits(Counter(seen))},
            "topic_switches": sum(1 for a, b in zip(seen, seen[1:]) if a != b),
            "lens": {
                "offered": len(lenses), "written": written, "ignored": len(lenses) - written,
                "kept": kept, "dropped": dropped, "unverified": len(checks) - kept - dropped,
                "fields": [f.field for f in lenses], "tiers": [f.tier for f in lenses],
                "keep_rate": round(kept / len(lenses), 4) if lenses else None,
            },
            "labels": {
                "counts": self.label_counts(),
                "coverage": {
                    "two_or_more_items": round(sum(1 for n in items if n >= 2) / len(noted), 4) if noted else None,
                    "where_we_stand": round(sum(1 for f in noted if f.has_wws) / len(noted), 4) if noted else None,
                },
            },
            "wws_lines": list(self.wws_lines),
            # notes without a 'Where we stand:' line: skipped by the model (missing) or lost to the cut (cut), and
            # how often the gate waived the line (DESIGN_explore §6.1)
            "wws": {
                "missing": sum(1 for f in noted if not f.has_wws and not f.cut),
                "cut": sum(1 for f in noted if not f.has_wws and f.cut),
                "waived": self.wws_waivers,
                "waived_at": list(self.wws_waived_at),
            },
            "answer_flips": sum(1 for a, b in zip(keys, keys[1:]) if a != b),
            "premature_finals": list(self.premature_finals),
            "forced_continuations": sum(1 for f in flags if f.kind == "continue"),
            "first_final_index": self.first_final_index,
            "gate_open": self.gate_open(),
            "gate_missing_at_final": self.gate_missing_at_final,
            "loop_stop": self.loop_stop,
            "ngram_overlap_last2": (round(word3gram_jaccard(self.notes[-2], self.notes[-1]), 4)
                                    if len(self.notes) >= 2 else None),
            "notes_chars": sum(len(s.text) for s in think),
            "notes_tokens_est": sum(len(s.text) for s in think) // 4,
            "garble": sum(1 for s in think if s.text and (_word_share(s.text) or 1.0) < GARBLE_WORD_SHARE),
            "vs_menu": {"mode": self.cfg.vs_menu, **self.vs_menu},
        }


@dataclass
class _RunState:
    question: str
    run_id: str
    today: str | None
    keys: list[str]
    trace: RelayTrace
    t0: float = field(default_factory=time.perf_counter)
    notes: list[str] = field(default_factory=list)
    current: str | None = None
    disabled: set[str] = field(default_factory=set)
    refusals: dict[str, int] = field(default_factory=dict)
    usage: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage_missing: list[dict[str, Any]] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    timings: list[dict[str, Any]] = field(default_factory=list)
    first_delta_s: float | None = None
    error: str | None = None
    explore: _ExploreState | None = None  # set when the explore preset is enabled
    rally: RallyState | None = None  # set when the rally preset is enabled

    def available(self, excluded: set[str] | frozenset[str] = frozenset()) -> list[str]:
        return [k for k in self.keys if k not in self.disabled and k not in excluded]

    def think_count(self) -> int:
        return sum(1 for s in self.trace.segments if s.phase == "think")

    def window_notes(self) -> list[str]:
        """The notes the routing window is built from: all of them, or, with the preset, the non-lens notes."""
        if self.explore is not None and self.explore.cfg.lens.exclude_from_window:
            return self.explore.window_notes
        return self.notes

    def count_call(self, att: _Attempt) -> None:
        acc = self.usage.setdefault(att.model, {"calls": 0})
        acc["calls"] += 1
        if att.usage:
            _add_usage(acc, att.usage)
        if (att.streamed or att.timed_out) and not _has_counts(att.usage):  # closed before the usage chunk
            self.usage_missing.append({
                "index": att.index, "model": att.model, "phase": att.phase, "attempt": att.attempt,
                "finish_reason": att.finish, "generation_id": (att.usage or {}).get("generation_id")})

    def mark_delta(self) -> None:
        if self.first_delta_s is None:
            self.first_delta_s = round(time.perf_counter() - self.t0, 4)

    def done_data(self, status: str) -> dict[str, Any]:
        total: dict[str, Any] = {}
        for acc in self.usage.values():
            _add_usage(total, acc)
        per_model: dict[str, dict[str, Any]] = {}
        for row in self.timings:
            agg = per_model.setdefault(row["model"], {"segments": 0, "ttft": [], "duration": []})
            agg["segments"] += 1
            if row["ttft_s"] is not None:
                agg["ttft"].append(row["ttft_s"])
            agg["duration"].append(row["duration_s"])
        latency = {
            "total_s": round(time.perf_counter() - self.t0, 4),
            "first_delta_s": self.first_delta_s,
            "segments": self.timings,
            "per_model": {
                m: {"segments": a["segments"],
                    "mean_ttft_s": round(sum(a["ttft"]) / len(a["ttft"]), 4) if a["ttft"] else None,
                    "mean_duration_s": round(sum(a["duration"]) / len(a["duration"]), 4)}
                for m, a in per_model.items()
            },
        }
        data = self.trace.to_dict() | {
            "usage": self.usage,
            "latency": latency,
            "usage_total": total,
            "usage_complete": not self.usage_missing,
            "usage_missing": self.usage_missing,
            "run_id": self.run_id,
            "status": status,
            "dropped": self.dropped,
            "refusals": self.refusals,
            "disabled": sorted(self.disabled),
        }
        if self.error is not None:
            data["error"] = self.error
        return data


# --------------------------------------------------------------------------------------
# Relay
# --------------------------------------------------------------------------------------


class Relay:
    """Runs one question through the multi-model relay (see the module docstring for the event protocol).

    ``router`` must provide ``reset()``, ``window()`` and ``decide()`` as in DESIGN.md §3.4 and, for the speculative
    strategy, ``decide_speculative()`` (which scores each draft with ``score_draft()``). A router keeps per-run
    state, so use one Relay (and one Router) per concurrent run; sequential runs may reuse them (the router is reset
    at the start of each run).
    ``retry_base_delay`` is the first back-off delay in seconds after a failed call (doubled per retry, capped at
    30 s; a server-sent ``Retry-After`` wins).
    """

    def __init__(self, cfg: AppConfig, backend: Backend, router: Router, *, models: list[ModelSpec] | None = None,
                 retry_base_delay: float = 0.5) -> None:
        self.cfg = cfg
        self.backend = backend
        self.router = router
        self.models: list[ModelSpec] = list(models) if models is not None else list(cfg.models)
        if not self.models:
            raise ValueError("the relay needs at least one model")
        self._spec = {m.key: m for m in self.models}
        if len(self._spec) != len(self.models):
            raise ValueError("model keys must be unique")
        if retry_base_delay < 0:
            raise ValueError(f"retry_base_delay must be >= 0, got {retry_base_delay}")
        self.retry_base_delay = retry_base_delay

    # ---- public API ---------------------------------------------------------------------------

    async def run(self, question: str, *, run_id: str | None = None) -> AsyncGenerator[RelayEvent, None]:
        """Stream the relay events for ``question`` (ends with exactly one ``done`` event).

        Closing the iterator early (``aclose()``, or cancelling the consuming task) closes the in-flight backend
        stream(s) before returning.
        """
        state = self._new_state(question, run_id=run_id)
        async with contextlib.aclosing(self._run(state)) as events:
            async for event in events:
                yield event

    async def run_to_trace(self, question: str, **kw: Any) -> RelayTrace:
        """Run to completion and return the trace (``kw``: ``run_id``)."""
        state = self._new_state(question, **kw)
        async with contextlib.aclosing(self._run(state)) as events:
            async for _ in events:
                pass
        return state.trace

    # ---- run loop -----------------------------------------------------------------------------

    def _new_state(self, question: str, *, run_id: str | None = None) -> _RunState:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("the question must be a non-empty string")
        today = self.cfg.today_str() if self.cfg.relay.inject_date else None
        question = question.strip()
        ex = self.cfg.explore
        ra = self.cfg.rally
        trace: RelayTrace = (ExploreTrace(question=question) if ex.enabled else
                             RallyTrace(question=question) if ra.enabled else RelayTrace(question=question))
        st = _RunState(question=question, run_id=run_id or _new_run_id(), today=today,
                       keys=[m.key for m in self.models], trace=trace)
        if ra.enabled:
            ring = [k for k in self.cfg.rally_ring() if k in st.keys]
            st.rally = RallyState(ra, ring, self.cfg.today_str())
        if ex.enabled:
            if ex.seed is not None:
                seed, source = ex.seed, "explore"
            elif self.cfg.router.seed is not None:
                seed, source = self.cfg.router.seed, "run"
            else:
                seed, source = random.randrange(1 << _SEED_BITS), "random"
            st.explore = _ExploreState(cfg=ex, seed=seed, seed_source=source, trace=st.trace, notes=st.notes,
                                       preset=self.cfg.preset)
        return st

    async def _run(self, st: _RunState) -> AsyncGenerator[RelayEvent, None]:
        rc = self.cfg.relay
        self.router.reset()
        st.trace.started_at = time.time()
        final: _Attempt | None = None
        while st.error is None and st.think_count() < rc.max_segments:
            index = len(st.trace.segments)
            window = self.router.window(st.question, st.window_notes(), rc.window_segments)
            out = _Produced()
            async with contextlib.aclosing(self._think(st, index, window, out)) as events:
                async for event in events:
                    yield event
            if out.attempt is None:
                if st.notes and out.outcomes and all(o == "empty" for o in out.outcomes.values()):
                    if st.rally is not None:
                        st.rally.stop_reason = "empty"
                    break  # no model has anything to add: the notes are concluded, answer from them
                if st.rally is not None and st.notes:
                    st.rally.stop_reason = "no_partner"  # nobody in the ring can take the turn: answer from the record
                    break
                st.error = out.error or f"segment {index} failed"
            elif out.attempt.answer is not None:
                final = out.attempt
                break
            elif st.rally is not None:
                partners = sum(1 for m in st.rally.ring if m in st.available())
                reason = st.rally.should_stop(partners)
                if reason is not None:
                    st.rally.stop_reason = reason
                    break
        if st.error is None:
            if st.explore is not None and not st.explore.gate_open():
                st.explore.gate_missing_at_final = st.explore.record_missing()
            out = _Produced()
            async with contextlib.aclosing(self._answer(st, final, out)) as events:
                async for event in events:
                    yield event
            if out.attempt is None:
                st.error = out.error or "the answer phase failed"
        if st.error is not None:
            yield RelayEvent(type="error", text=st.error, data={"dropped": st.dropped[-10:]})
        if st.explore is not None and isinstance(st.trace, ExploreTrace):
            st.trace.explore = st.explore.metrics()
        if st.rally is not None and isinstance(st.trace, RallyTrace):
            if st.rally.stop_reason is None:
                st.rally.stop_reason = "error" if st.error is not None else "max_segments"
            st.trace.rally = st.rally.metrics(preset=self.cfg.preset)
        st.trace.ended_at = time.time()
        yield RelayEvent(type="done", data=st.done_data("error" if st.error is not None else "ok"))

    async def _think(self, st: _RunState, index: int, window: str, out: _Produced) -> AsyncGenerator[RelayEvent, None]:
        """One think segment: speculative drafting when the gate opens (and the policy ranks drafts by their scores),
        else the sequential path."""
        rc = self.cfg.relay
        decision: RouteDecision | None = None
        update = True
        if rc.strategy == "speculative" and self.cfg.router.policy in _DRAFT_POLICIES and st.rally is None:
            plan = await self._plan(st, index) if st.explore is not None else None
            avail = st.available() if plan is None else _without(st.available(), plan.exclude)
            if len(avail) > 1:
                decision = self.router.decide(window, current=st.current, available=avail,
                                              update=plan is None or plan.kind != "lens")
                update = False
                if decision.r >= rc.speculative_gate:
                    async with contextlib.aclosing(self._speculative(st, index, window, out, plan)) as events:
                        async for event in events:
                            yield event
                    if out.attempt is not None:
                        return
                    decision = None  # no usable draft: re-decide among the remaining models
        async with contextlib.aclosing(
                self._produce(st, "think", index, window, out, decision=decision, update=update)) as events:
            async for event in events:
                yield event

    async def _answer(self, st: _RunState, final: _Attempt | None, out: _Produced) -> AsyncGenerator[RelayEvent, None]:
        """The answer phase: the inline answer after the marker, or one routed answer call."""
        rc = self.cfg.relay
        index = len(st.trace.segments)
        if rc.answer_mode == "inline" and final is not None and final.answer:
            now = time.time()
            att = _Attempt(model=final.model, phase="answer", index=index, text=final.answer, finish="inline",
                           started_at=now, ended_at=now, ttft_s=0.0)
            yield RelayEvent(type="phase", segment=index, model=att.model, phase="answer", decision="forced",
                             data={"answer_mode": "inline", "attempt": 0})
            st.mark_delta()
            yield RelayEvent(type="delta", segment=index, model=att.model, phase="answer", text=att.text)
            yield self._commit(st, att, "forced", posterior={}, scores={}, extra={"answer_mode": "inline"})
            out.attempt = att
            return
        notes = st.window_notes()
        window = self.router.window(st.question, notes, len(notes))
        async with contextlib.aclosing(self._produce(st, "answer", index, window, out, update=False)) as events:
            async for event in events:
                yield event

    # ---- sequential path (also used for the routed answer) ---------------------------------------

    async def _produce(self, st: _RunState, phase: Phase, index: int, window: str, out: _Produced, *,
                       decision: RouteDecision | None = None, update: bool = True) -> AsyncGenerator[RelayEvent, None]:
        """Decide, stream, and handle refusals/length/errors until one attempt is committed or no model is left."""
        rc = self.cfg.relay
        excluded = set(out.excluded)
        reasons: dict[str, str] = {m: str(o) for m, o in out.outcomes.items()}
        attempt_no = 0
        while True:
            avail = st.available(excluded)
            if st.rally is not None:  # the rally writes in ring order; its answer comes from a ring model if one is left
                in_ring = [m for m in st.rally.ring if m in avail]
                avail = in_ring if phase == "think" or (self.cfg.rally.answer_from_ring and in_ring) else avail
            if not avail:
                detail = "; ".join(f"{m}: {r}" for m, r in reasons.items()) or "all models are disabled"
                out.error = f"no model left for {'the answer' if phase == 'answer' else f'segment {index}'} ({detail})"
                return
            plan: _Plan | None = None
            if st.explore is not None and phase == "think":
                plan = await self._plan(st, index)
                avail = _without(avail, plan.exclude)
            if decision is None or decision.model not in avail:
                decision = self.router.decide(window, current=st.current, available=avail,
                                              update=update and (plan is None or plan.kind != "lens"))
            if st.rally is not None and phase == "think":  # hand-over: the router only reports topic and scores
                nxt = st.rally.next_model(avail)
                if nxt is not None and (decision.model != nxt or decision.decision != "forced"):
                    decision = dataclasses.replace(decision, model=nxt, decision="forced", probs={nxt: 1.0})
            update = False
            d, decision = decision, None
            spec = self._spec[d.model]
            base_tokens = (rc.answer_max_tokens if phase == "answer" else rc.segment_tokens) + spec.max_tokens_slack
            max_tokens: int | None = None
            length_retried = False
            failures = 0
            while True:
                att = _Attempt(model=spec.key, phase=phase, index=index, attempt=attempt_no)
                attempt_no += 1
                start: dict[str, Any] = {"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4),
                                         "attempt": att.attempt, "strategy": "sequential", "available": avail}
                if plan is not None:
                    start["explore"] = plan.badge()
                if st.rally is not None and phase == "think":
                    last = st.rally.previous()
                    start["rally"] = {"kind": st.rally.kind(), "turn": st.rally.turns + 1, "ring": list(st.rally.ring),
                                      "self_reply": last is not None and last["model"] == spec.key}
                yield RelayEvent(
                    type="segment_start" if phase == "think" else "phase", segment=index, model=spec.key, phase=phase,
                    decision=d.decision, topic_posterior=_rounded(d.posterior, 4), scores=_rounded(d.scores, 3),
                    data=start)
                req = self._request(st, phase, max_tokens, plan, spec)
                async with contextlib.aclosing(self._stream(st, spec, req, att, emit=True)) as events:
                    async for event in events:
                        yield event
                if plan is not None:
                    self._strip_final(st, att)
                    self._resolve_final(st, att, plan)
                if st.rally is not None and phase == "think" and att.answer is not None:
                    att.premature, att.premature_answer, att.answer = True, att.answer, None  # the moderator ends the exchange
                result = att.outcome
                if result == "ok":
                    extra = {"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4)}
                    yield self._commit(st, att, d.decision, posterior=d.posterior, scores=d.scores, extra=extra,
                                       plan=plan)
                    out.attempt = att
                    return
                yield self._drop(st, att, result)
                reasons[spec.key] = result if att.error is None else f"{result}: {att.error}"
                out.outcomes[spec.key] = result
                if result == "length" and not length_retried:
                    length_retried = True
                    max_tokens = 2 * base_tokens
                    continue
                if result == "error" and att.retryable and failures < rc.max_retries:
                    failures += 1
                    await asyncio.sleep(self._backoff(failures, att.retry_after))
                    continue
                if result == "refused":
                    self._count_refusal(st, spec.key)
                if st.rally is not None and phase == "think":
                    st.rally.note_skipped(index, spec.key, result)
                    if result == "error" and st.rally.hand_over_failed(spec.key, fatal=not att.retryable):
                        st.disabled.add(spec.key)  # a partner that keeps failing is out for the run (no_partner follows)
                excluded.add(spec.key)
                break

    # ---- speculative path ------------------------------------------------------------------------

    async def _speculative(self, st: _RunState, index: int, window: str, out: _Produced,
                           plan: _Plan | None = None) -> AsyncGenerator[RelayEvent, None]:
        """Draft the segment with every available model concurrently; the router picks among the usable drafts."""
        avail = st.available() if plan is None else _without(st.available(), plan.exclude)
        tasks = [asyncio.create_task(self._draft(st, self._spec[m], self._request(st, "think", None, plan, self._spec[m]),
                                             index)) for m in avail]
        try:
            drafts: list[_Attempt] = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ok: dict[str, _Attempt] = {}
        failed: dict[str, str] = {}
        for att in drafts:
            if plan is not None:
                self._strip_final(st, att)  # a draft that concludes early keeps only its note part
            result = att.outcome
            if result == "ok":
                ok[att.model] = att
                continue
            failed[att.model] = result
            out.outcomes[att.model] = result
            self._record_drop(st, att, result)
            if result == "refused":
                self._count_refusal(st, att.model)
            if result in ("refused", "empty"):
                out.excluded.add(att.model)  # errors and length overruns get the sequential retry path
        if not ok:
            return
        d = self.router.decide_speculative(window, {m: a.text or a.answer or "" for m, a in ok.items()},
                                           current=st.current)
        att = ok[d.model]
        if plan is not None:
            self._resolve_final(st, att, plan)
        extra = {"probs": _rounded(d.probs, 4), "r": round(float(d.r), 4)}
        start = extra | {"attempt": 0, "strategy": "speculative", "available": list(ok)}
        if plan is not None:
            start["explore"] = plan.badge()
        yield RelayEvent(type="segment_start", segment=index, model=att.model, phase="think", decision=d.decision,
                         topic_posterior=_rounded(d.posterior, 4), scores=_rounded(d.scores, 3), data=start)
        if att.text:
            st.mark_delta()
            yield RelayEvent(type="delta", segment=index, model=att.model, phase="think", text=att.text)
        extra |= {"drafts": {m: a.text for m, a in ok.items()}, "failed_drafts": failed}
        yield self._commit(st, att, d.decision, posterior=d.posterior, scores=d.scores, extra=extra, plan=plan)
        out.attempt = att

    async def _draft(self, st: _RunState, spec: ModelSpec, req: ContinuationRequest, index: int) -> _Attempt:
        """A non-emitting attempt (retried once with doubled max_tokens after an empty ``length`` finish)."""
        att = _Attempt(model=spec.key, phase="think", index=index)
        async with contextlib.aclosing(self._stream(st, spec, req, att, emit=False)) as events:
            async for _ in events:
                pass
        if att.outcome != "length":
            return att
        retry = _Attempt(model=spec.key, phase="think", index=index, attempt=1)
        req2 = dataclasses.replace(req, max_tokens=2 * (req.target_tokens + spec.max_tokens_slack))
        async with contextlib.aclosing(self._stream(st, spec, req2, retry, emit=False)) as events:
            async for _ in events:
                pass
        return retry

    # ---- one streamed call -------------------------------------------------------------------------

    def _request(self, st: _RunState, phase: Phase, max_tokens: int | None,
                 plan: _Plan | None = None, spec: ModelSpec | None = None) -> ContinuationRequest:
        rc = self.cfg.relay
        index = len(st.trace.segments)
        req = ContinuationRequest(
            question=st.question, segments=list(st.notes), phase=phase,
            target_tokens=rc.answer_max_tokens if phase == "answer" else rc.segment_tokens, max_tokens=max_tokens,
            today=st.today, sentinel=rc.sentinel, final_marker=rc.final_marker, hard_cut_chars=rc.hard_cut_chars,
            run_id=st.run_id, metadata={"segment": index, "phase": phase})
        ra = st.rally
        if ra is not None:  # rally preset: its own system prompt, and a tail that names the previous writer
            if phase == "think":
                turn = ra.turns + 1
                prev = ra.previous()
                writer = self._spec[prev["model"]].label() if prev is not None and prev["model"] in self._spec else None
                light = spec is not None and spec.key in self.cfg.rally.light_schema_models
                self_reply = spec is not None and prev is not None and spec.key == prev["model"]
                directive = rally_directive(turn, writer, prev["newest"] if prev else None,
                                            newest_raw=prev["newest_raw"] if prev else None,
                                            approx=bool(prev and prev["approx"]), self_reply=self_reply, light=light)
                if light:  # the default system prompt and a prose tail: schema-heavy prompts trip Fable's input filter
                    req.system_override = relay_system_prompt(rc.segment_tokens, st.today, rc.sentinel, rc.final_marker)
                    req.tail = relay_tail_rally_light(turn, directive, rc.final_marker)
                else:
                    req.system_override = relay_system_prompt_rally(rc.segment_tokens, st.today, rc.sentinel)
                    req.tail = relay_tail_rally(turn, directive)
                req.metadata |= {"kind": ra.kind(), "light": light, "self_reply": self_reply}
            else:
                notes = [i + 1 for i, n in enumerate(ra.notes) if n["self_reply"]]
                req.system_override = answer_system_prompt_rally(st.today, answer_basis=self.cfg.rally.answer_basis,
                                                                 self_reply_notes=notes)
            return req
        ex = st.explore
        if ex is None:
            return req
        # explore preset: the structured-notes prompts, the per-segment tail, and a reproducible seed per segment
        req.metadata |= {"segment_index": index, "seed": ex.seed}
        if phase == "think":
            missing = ex.record_missing()
            light = spec is not None and spec.key in ex.cfg.light_schema_models
            if light:
                # light schema (DESIGN.md §7.8): the default system prompt and a tail that only carries the note
                # number, the directive and the conclusion permission; both are live-verified to pass Fable's filter
                req.system_override = relay_system_prompt(rc.segment_tokens, st.today, rc.sentinel, rc.final_marker)
                status = record_status_text([], not missing, rc.final_marker)
            else:
                req.system_override = relay_system_prompt_explore(rc.segment_tokens, st.today, rc.sentinel,
                                                                  rc.final_marker)
                status = record_status_text(missing, not missing, rc.final_marker,
                                            remind_wws=not ex.wws_previous())
            req.tail = relay_tail(st.think_count() + 1, status, plan.directive if plan is not None else "")
            req.metadata["kind"] = plan.kind if plan is not None else "main"
            req.metadata["light"] = light
        else:
            req.system_override = answer_system_prompt_explore(st.today, answer_basis=ex.cfg.answer_basis)
        return req

    # ---- explore preset: the plan of a think segment and the record gate (DESIGN.md §7.6) -------------

    async def _plan(self, st: _RunState, index: int) -> _Plan:
        """Decide what the tail of think segment ``index`` asks for: the pending check of a lens, a forced
        continuation, a freshly drawn lens, a next-move direction from the menu side call, or a plain note."""
        ex = st.explore
        assert ex is not None
        lens = ex.cfg.lens
        if ex.pending_check and ex.lens_field is not None:
            exclude = {ex.lens_writer} if lens.check_other_model and ex.lens_writer else set()
            if ex.forcing is not None:  # the conclusion set aside in the lens note: the check continues the line
                exclude.add(ex.forcing["model"])
            return _Plan(kind="check", directive=check_text(ex.lens_field.label), exclude=frozenset(exclude),
                         field=ex.lens_field, forcing=ex.forcing)
        if ex.forcing is not None:
            f = ex.forcing
            return _Plan(kind="continue", directive=continue_text(f["why"], f["cue"]), exclude=frozenset({f["model"]}),
                         cue=f["cue"], why=f["why"], forcing=f)
        stagnating = ex.stagnating()
        gate_open = ex.gate_open()
        eligible = (index >= lens.first_index and not gate_open and ex.cooldown == 0
                    and ex.lenses_used < lens.max)
        last_slot = eligible and ex.main_line_count() == ex.cfg.min_segments - 1
        rng = random.Random(f"{ex.seed}:{index}")
        if eligible:
            p = lens.p_stagnation if stagnating else lens.p
            if rng.random() < p or (last_slot and ex.lenses_used < lens.min):
                drawn = draw_lens(rng, ex.current_topic(), ex.used_fields, lens.tiers, stagnating)
                if drawn is not None:
                    fld, tier, concepts = drawn
                    return _Plan(kind="lens", directive=lens_text(fld.label, concepts), field=fld, tier=tier,
                                 concepts=concepts, stagnating=stagnating)
        mode = ex.cfg.vs_menu
        wants = not gate_open and ((mode == "stagnation" and stagnating) or (mode == "every_lens" and eligible))
        plan = _Plan(kind="main", stagnating=stagnating, wants_menu=wants)
        if wants:
            plan.directive = await self._menu_directive(st, index)
        return plan

    async def _menu_directive(self, st: _RunState, index: int) -> str:
        """The next-move menu side call (verbalized sampling) for segment ``index``, once per index: Sol lists five
        directions with probabilities, the client picks one from the tail. Skipped (logged) when Sol is not among the
        relay's models; a failing side call never fails the run."""
        ex = st.explore
        assert ex is not None
        if index in ex.vs_cache:
            return ex.vs_cache[index]
        book = ex.vs_menu
        directive = ""
        spec = self._spec.get(VS_MENU_MODEL)
        if spec is None:
            book["skipped"] = f"{VS_MENU_MODEL} is not configured"
        else:
            book["calls"] += 1
            req = ChatRequest(messages=vs_menu_messages(st.question, st.notes), max_tokens=VS_MENU_MAX_TOKENS,
                              metadata={"seed": ex.seed, "segment_index": index, "purpose": "vs_menu"})
            try:
                res = await self.backend.chat(spec, req)
            except Exception as e:  # noqa: BLE001 -- a side call must never fail the run
                book["errors"].append({"index": index, "error": f"{type(e).__name__}: {e}"})
            else:
                acc = st.usage.setdefault(spec.key, {"calls": 0})
                acc["calls"] += 1
                if res.usage:
                    _add_usage(acc, res.usage)
                menu = parse_vs_menu(res.text)
                pick = pick_vs_direction(menu, random.Random(f"{ex.seed}:{index}:vs"))
                row: dict[str, Any] = {"index": index, "menu": [list(line) for line in menu], "chosen": None, "p": None}
                if pick is not None:
                    book["parsed"] += 1
                    book["chosen_p"].append(pick[1])
                    row["chosen"], row["p"] = pick
                    directive = vs_direction_text(pick[0])
                book["menus"].append(row)
        ex.vs_cache[index] = directive
        return directive

    def _strip_final(self, st: _RunState, att: _Attempt) -> None:
        """Set aside a conclusion the record gate does not accept yet: the note part stays, the answer is kept
        only for :meth:`_resolve_final` (the loop stop may accept it after all)."""
        ex = st.explore
        if ex is None or att.answer is None or ex.accepting():
            return
        att.premature, att.premature_answer, att.answer = True, att.answer, None

    def _resolve_final(self, st: _RunState, att: _Attempt, plan: _Plan) -> None:
        """Gate bookkeeping once ``att`` is the segment's result (DESIGN_explore §6.1-6.2): record the first
        conclusion; for a premature one either stop forcing (the last two notes loop: the conclusion is accepted)
        or schedule the forced continuation with the next cue, the concluding model excluded."""
        ex = st.explore
        if ex is None or (att.answer is None and not att.premature):
            return
        if ex.first_final_index is None:
            ex.first_final_index = att.index
        if not att.premature:
            return
        missing = ex.record_missing(after=(plan.kind, att.text))  # what is still missing once the note is in
        notes = [*st.notes, att.text] if att.text else list(st.notes)
        overlap = word3gram_jaccard(notes[-2], notes[-1]) if len(notes) >= 2 else 0.0
        row: dict[str, Any] = {"index": att.index, "model": att.model, "kind": plan.kind, "missing": missing,
                               "overlap_last2": round(overlap, 4), "forced": False, "loop": False}
        if ex.forcings > 0 and overlap > ex.cfg.loop_overlap_jaccard:
            ex.loop_stop = True  # forcing produced a repeat: accept this conclusion rather than loop
            att.answer, att.premature = att.premature_answer, False
            row["loop"] = True
        else:
            cue = ex.next_cue()
            ex.forcings += 1
            ex.forcing = {"why": _lacks(missing), "model": att.model, "cue": cue, "index": att.index}
            row["forced"], row["cue"] = True, cue
        ex.premature_finals.append(row)

    async def _stream(self, st: _RunState, spec: ModelSpec, req: ContinuationRequest, att: _Attempt, *,
                      emit: bool) -> AsyncGenerator[RelayEvent, None]:
        """Stream one call into ``att``; yields display ``delta`` events when ``emit`` (drafts stay silent).

        Each event is awaited within the attempt's remaining wall-clock budget (``first_content_timeout_s`` until the
        first text, ``segment_timeout_s`` in total), so a provider that only sends keep-alive comments, or trickles,
        cannot hold the segment.
        """
        rc = self.cfg.relay
        echo = _Echo(req.segments if req.phase == "think" else (), exempt=_quiet_shape if st.rally is not None else None)
        filt = _StreamFilter(phase=req.phase, echo=echo, sentinel=rc.sentinel,
                             final_marker=rc.final_marker, hard_cut_chars=rc.hard_cut_chars,
                             stop_at_marker=rc.answer_mode == "routed")
        runaway = rc.hard_cut_chars * _RUNAWAY_FACTOR if req.phase == "think" else None
        loop = asyncio.get_running_loop()
        t0 = time.perf_counter()
        begin = loop.time()
        att.started_at = time.time()
        stream: Any = None
        try:
            stream = self.backend.stream_continuation(spec, req)
            while True:
                budget = rc.segment_timeout_s if att.ttft_s is not None else min(rc.first_content_timeout_s,
                                                                                  rc.segment_timeout_s)
                try:
                    ev = await asyncio.wait_for(stream.__anext__(), max(0.0, begin + budget - loop.time()))
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    self._time_out(att, filt, budget)
                    break
                att.streamed = True
                if att.generation_id is None and ev.kind != "done" and ev.usage:
                    gid = ev.usage.get("generation_id")
                    att.generation_id = gid if isinstance(gid, str) and gid else None
                if ev.kind == "text":
                    if att.error is not None or filt.stop_reason == "sentinel":
                        continue  # nothing after an error or the sentinel is used
                    if att.ttft_s is None:
                        att.ttft_s = round(time.perf_counter() - t0, 4)
                    piece = filt.feed(ev.text)
                    if piece and emit:
                        st.mark_delta()
                        yield RelayEvent(type="delta", segment=att.index, model=spec.key, phase=req.phase,
                                         text=piece)
                    if filt.stop_reason is not None and (
                            (filt.stop_reason != "sentinel" and spec.cancellable)
                            or (runaway is not None and filt.excess(len(filt.raw)) > runaway)):
                        att.aborted = True  # closing the stream stops generation (and billing)
                        att.finish = filt.stop_reason
                        break
                elif ev.kind == "error":
                    att.error = ev.error or "the backend reported a stream error"
                elif ev.kind == "done":
                    att.finish = ev.finish_reason
                    att.usage = dict(ev.usage) if ev.usage else None
                    att.note = ev.error
                    # a sentinel stream whose usage chunk (always the last chunk) arrived was read to its end
                    att.aborted = att.aborted or (ev.finish_reason in _CLOSED_FINISHES and not _has_counts(att.usage))
                    break
        except Exception as e:  # noqa: BLE001 -- any backend failure is handled by the retry/exclude policy
            att.error = f"{type(e).__name__}: {e}"
            att.retryable = bool(getattr(e, "retryable", True))
            retry_after = getattr(e, "retry_after", None)
            att.retry_after = float(retry_after) if isinstance(retry_after, (int, float)) else None
        finally:
            if stream is not None:
                await _aclose(stream)
            if att.generation_id is not None:  # closed before ``done`` (or without its id): keep the first event's
                att.usage = {"generation_id": att.generation_id} | (att.usage or {})
            att.duration_s = round(time.perf_counter() - t0, 4)
            att.ended_at = time.time()
            st.count_call(att)
        att.raw = filt.raw
        if req.phase == "think":
            att.text, att.answer, att.saw_sentinel, att.cut, att.full = _finalize_think(
                filt.raw, echo, sentinel=rc.sentinel, final_marker=rc.final_marker,
                hard_cut_chars=rc.hard_cut_chars)
            if att.answer is None and not att.saw_sentinel and att.finish in _TRUNCATED_FINISHES:
                att.text = complete_sentences(att.text)  # cut off: drop the dangling half sentence
                att.cut = True
        else:
            att.text = _finalize_answer(filt.raw, sentinel=rc.sentinel, final_marker=rc.final_marker)
            att.saw_sentinel = filt.stop_reason in ("sentinel", "mangled")
        if emit and att.outcome == "ok":
            tail = filt.flush()
            if tail:
                st.mark_delta()
                yield RelayEvent(type="delta", segment=att.index, model=spec.key, phase=req.phase, text=tail)

    @staticmethod
    def _time_out(att: _Attempt, filt: _StreamFilter, budget: float) -> None:
        """The attempt ran out of its wall-clock budget; the stream is closed next."""
        att.aborted = True
        if filt.stop_reason is not None:  # the paragraph was already complete (sentinel, marker, overflow): keep it
            att.finish = att.finish or filt.stop_reason
            return
        att.timed_out = True
        att.finish = "timeout"
        what = "no text" if att.ttft_s is None else "the stream did not end"
        att.error = f"timeout: {what} within {budget:g} s"

    # ---- commit / drop -----------------------------------------------------------------------------

    def _commit(self, st: _RunState, att: _Attempt, decision: Decision, *, posterior: Mapping[str, float],
                scores: Mapping[str, float], extra: dict[str, Any], plan: _Plan | None = None) -> RelayEvent:
        """Append the attempt to the trace (and the notes) and return its authoritative ``segment_end``."""
        seg = Segment(index=att.index, model=att.model, phase=att.phase, text=att.text, decision=decision,
                      topic_posterior=_rounded(posterior, 4), scores=_rounded(scores, 3),
                      started_at=att.started_at, ended_at=att.ended_at, aborted=att.aborted, usage=att.usage,
                      note=att.finish or "")
        flags = self._explore_commit(st, att, plan) if plan is not None and att.phase == "think" else None
        seg.explore = flags
        if st.rally is not None and att.phase == "think":
            seg.rally = st.rally.add_note(att.index, att.model, att.text, full=att.full or att.text, cut=att.cut,
                                          light=att.model in self.cfg.rally.light_schema_models)
            st.rally.hand_over_ok(att.model)
        st.trace.segments.append(seg)
        if flags is not None and st.explore is not None:
            st.explore.note_committed(seg.index)  # the waiver streak is judged with this note in the record
        if att.phase == "think":
            if att.text:
                st.notes.append(att.text)
            st.current = att.model
        else:
            st.trace.answer = att.text
        st.timings.append({"index": att.index, "model": att.model, "phase": att.phase, "ttft_s": att.ttft_s,
                           "duration_s": att.duration_s})
        data = extra | {"finish_reason": att.finish, "attempt": att.attempt, "ttft_s": att.ttft_s,
                        "duration_s": att.duration_s, "usage": att.usage, "aborted": att.aborted,
                        "saw_sentinel": att.saw_sentinel}
        if att.phase == "think":
            data["final"] = att.answer is not None
        if flags is not None:
            data["explore"] = flags.to_dict()
            if att.premature:
                data["premature_final"] = True
        if seg.rally is not None:
            data["rally"] = seg.rally
            if att.premature:
                data["premature_final"] = True
        return RelayEvent(type="segment_end", segment=att.index, model=att.model, phase=att.phase, text=att.text,
                          decision=decision, data=data)

    def _explore_commit(self, st: _RunState, att: _Attempt, plan: _Plan) -> ExploreFlags:
        """The committed think segment's explore flags, and the state transitions of the schedule: a lens awaits
        its check (its writer is remembered, its field is used up), a check starts the cooldown, a continuation
        consumes its forcing; non-lens notes join the routing window."""
        ex = st.explore
        assert ex is not None
        parsed = parse_labels(att.text)
        flags = ExploreFlags(kind=plan.kind, field=plan.field.label if plan.field is not None else None,
                             tier=plan.tier, labels=dict(parsed["counts"]), has_wws=bool(parsed["has_wws"]),
                             cut=att.cut, cue=plan.cue, light=att.model in ex.cfg.light_schema_models)
        if plan.kind == "lens":
            flags.lens_prefix_ok = parsed["lens_prefix"] is not None
            ex.pending_check, ex.lens_writer, ex.lens_field = True, att.model, plan.field
            ex.lenses_used += 1
            if plan.field is not None:
                ex.used_fields.append(plan.field.name)
        elif plan.kind == "check":
            flags.lens_verdict = parsed["lens_verdict"]
            ex.pending_check, ex.lens_writer, ex.lens_field = False, None, None
            ex.cooldown = ex.cfg.lens.cooldown
        else:
            ex.cooldown = max(0, ex.cooldown - 1)
        if plan.forcing is not None and ex.forcing is plan.forcing:
            ex.forcing = None  # consumed by this segment (a newer one scheduled meanwhile stays pending)
        if att.text and (plan.kind != "lens" or not ex.cfg.lens.exclude_from_window):
            ex.window_notes.append(att.text)
        if parsed["has_wws"]:
            ex.wws_lines.append(parsed["wws_sentence"])
        return flags

    def _record_drop(self, st: _RunState, att: _Attempt, reason: Outcome) -> dict[str, Any]:
        row = {"index": att.index, "model": att.model, "phase": att.phase, "attempt": att.attempt, "reason": reason,
               "finish_reason": att.finish, "error": att.error or att.note}
        if att.premature:
            row["premature_final"] = True  # explore: it concluded with an empty note while the gate was closed
        st.dropped.append(row)
        return row

    def _drop(self, st: _RunState, att: _Attempt, reason: Outcome) -> RelayEvent:
        row = self._record_drop(st, att, reason)
        return RelayEvent(type="segment_end", segment=att.index, model=att.model, phase=att.phase,
                          data={"dropped": True, **{k: v for k, v in row.items() if k not in ("index", "model")}})

    def _count_refusal(self, st: _RunState, model: str) -> None:
        st.refusals[model] = st.refusals.get(model, 0) + 1
        if st.refusals[model] >= self.cfg.relay.disable_after_refusals:
            st.disabled.add(model)

    def _backoff(self, failures: int, retry_after: float | None) -> float:
        delay = retry_after if retry_after is not None else self.retry_base_delay * 2 ** (failures - 1)
        return max(0.0, min(delay, _MAX_BACKOFF_S))


# --------------------------------------------------------------------------------------
# Runtime assembly (CLI and server)
# --------------------------------------------------------------------------------------


def mock_truth_index(truths: Mapping[str, Any], models: Sequence[str]) -> CutoffIndex | None:
    """A cutoff index built directly from simulated-model truths (``MockModelTruth``), for ``--mock`` demos.

    tau = the true per-topic cutoff, sd = 0.5 months, hazard = the truth's stale rate. ``meta["source"]`` says that
    it is not a measurement. None if no model has a truth.
    """
    import numpy as np

    from cutoffmix.router.index import CutoffIndex, ProbeIndex
    from cutoffmix.taxonomy import TOPIC_IDS

    keys = [m for m in models if m in truths]
    if not keys:
        return None
    tau = np.array([[truths[m].tau.get(z, truths[m].default_tau) for z in TOPIC_IDS] for m in keys], dtype=float)
    hazard = np.array([[truths[m].stale_rate] * len(TOPIC_IDS) for m in keys], dtype=float)
    probe = ProbeIndex(
        probe="current_d", tau=tau, var=np.full(tau.shape, 0.25), hazard=hazard, abstain=1.0 - hazard,
        n=np.zeros(tau.shape, dtype=int), draws=None,
        global_tau=np.array([truths[m].default_tau for m in keys], dtype=float),
        slope=np.array([truths[m].slope for m in keys], dtype=float),
        plateau=np.array([truths[m].plateau for m in keys], dtype=float),
    )
    meta = {"source": "mock truths", "note": "simulated models (--mock); true cutoffs, not a measurement"}
    return CutoffIndex(keys, list(TOPIC_IDS), {"current_d": probe}, meta)


RouterFactory = Callable[["AppConfig"], "Router"]


class RelayRuntime:
    """What every run needs, set up once: config, backend, cutoff index, topic classifier and time-sensitivity gate.

    :meth:`relay` builds a fresh Router and Relay per run (a router keeps per-run state), so one runtime can serve
    concurrent requests. The index and classifier are loaded on first use. With ``mock=True`` and no
    ``router.index_path`` the index is :func:`mock_truth_index` of the mock backend's truths, so simulated runs
    route on the simulated cutoffs. ``router_factory`` replaces the default Router construction (tests).
    """

    def __init__(self, cfg: AppConfig, *, backend: Backend | None = None, mock: bool = False, seed: int = 0,
                 router_factory: RouterFactory | None = None, mock_latency_s: float | None = None) -> None:
        self.cfg = cfg
        self.mock = mock
        self._owns_backend = backend is None
        if backend is None:
            from cutoffmix.backends import make_backend

            backend = make_backend(cfg, mock=mock, seed=seed, mock_latency_s=mock_latency_s)
        self.backend: Backend = backend
        self.router_factory: RouterFactory = router_factory or self._default_router
        self._routing: tuple[CutoffIndex | None, str | None, Any] | None = None
        self._gate: TimeSensitivityGate | None = None

    # ---- routing components ----

    def _load_routing(self) -> tuple[CutoffIndex | None, str | None, Any]:
        if self._routing is None:
            from cutoffmix.router.index import load_index
            from cutoffmix.router.topics import load_classifier

            rc = self.cfg.router
            truths = getattr(self.backend, "truths", None)
            source: str | None
            if self.mock and rc.index_path is None and isinstance(truths, Mapping):
                index, source = mock_truth_index(truths, self.cfg.model_keys()), "mock truths"
            else:
                index = load_index(rc.index_path)
                source = rc.index_path or ("packaged index_prior.json" if index is not None else None)
            self._routing = (index, source, load_classifier(rc.classifier, rc.classifier_path))
        return self._routing

    @property
    def index(self) -> CutoffIndex | None:
        """The cutoff index used for routing (None: all scores 0, the policy mixes uniformly)."""
        return self._load_routing()[0]

    @property
    def index_source(self) -> str | None:
        """Where the index came from: a path, ``"mock truths"`` or None."""
        return self._load_routing()[1]

    @property
    def gate(self) -> TimeSensitivityGate:
        """The time-sensitivity gate: the trained one at ``router.gate_path``, else the lexical baseline."""
        if self._gate is None:
            from cutoffmix.router.gate import load_gate

            self._gate = load_gate(self.cfg.router.gate_path, today=self.cfg.today_str())
        return self._gate

    def _default_router(self, cfg: AppConfig) -> Router:
        from cutoffmix.router.router import Router

        index, _, classifier = self._load_routing()
        return Router(cfg.model_keys(), index=index, classifier=classifier, gate=self.gate, cfg=cfg.router,
                      seed=cfg.router.seed)

    # ---- per run ----

    def configure(self, *, policy: str | None = None, strategy: str | None = None,
                  segment_tokens: int | None = None) -> AppConfig:
        """A validated copy of the config with per-run overrides (raises ``ValueError``/``ConfigError``)."""
        cfg = self.cfg
        relay_changes: dict[str, Any] = {}
        if strategy is not None:
            relay_changes["strategy"] = strategy
        if segment_tokens is not None:
            relay_changes["segment_tokens"] = segment_tokens
        relay = dataclasses.replace(cfg.relay, **relay_changes) if relay_changes else cfg.relay
        router = dataclasses.replace(cfg.router, policy=policy) if policy is not None else cfg.router
        if relay is cfg.relay and router is cfg.router:
            return cfg
        return dataclasses.replace(cfg, relay=relay, router=router)

    def relay(self, cfg: AppConfig | None = None) -> Relay:
        """A new Relay (with a new Router) for one run with ``cfg`` (default: the runtime's config)."""
        run_cfg = cfg or self.cfg
        return Relay(run_cfg, self.backend, self.router_factory(run_cfg))

    async def aclose(self) -> None:
        """Close the backend if this runtime created it."""
        if self._owns_backend:
            await _aclose(self.backend)
