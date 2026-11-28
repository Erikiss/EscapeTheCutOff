"""The rally preset's note parsing and bookkeeping (DESIGN_rally.md): which labelled lines a note has, the newest month
its writer says it knows, when the exchange has run dry, and the metrics of a run.

Everything here is pure: the relay (``relay.py``) feeds it the committed notes and asks it who writes next and whether
to stop. The dates are the models' own statements ("self-reported"); the project rule that self-reports do not count
is why these numbers describe what the exchange did and are no proof that it helped.

Notes come in two shapes. A *full* note uses the labels (``Confirm:``, ``Update:``, ``Newer:`` ...). A *light* note
(the writer got the light prompt, see ``rally.light_schema_models``) is prose that may carry no label at all; it
counts as quiet when it has no ``Correction:`` and its closing ``Newest I know:`` line is readable and not above what
the record already reaches. Two things never count as quiet: a note the writer sent to itself (the partner could not
take its turn) and a note whose end the hard cut removed before its closing line could be read.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cutoffmix.config import RallyConfig

__all__ = ["RallyState", "month_number", "parse_month", "parse_note", "REPLY_LABELS"]

REPLY_LABELS = ("confirm", "update", "newer", "unverified", "nothing_newer")
_LABEL_NAMES = {
    "known": "known", "confirm": "confirm", "update": "update", "updates": "update", "correction": "update",
    "corrections": "update", "newer": "newer", "unverified": "unverified", "nothing newer": "nothing_newer",
    "newest i know": "newest",
}

# ---- labels --------------------------------------------------------------------------------------------------------
#
# Every whitespace quantifier below is followed by a token that is not whitespace, so no run of spaces can be split
# between two neighbouring quantifiers: matching stays linear in the length of the text (a run of 200,000 spaces after
# a full stop is scanned once; the first version of this pattern took seconds for a run of a thousand).

_WS = r"[^\S\n]"  # whitespace except a line break (a no-break space included)
_MARK = r"(?:\*\*|__)"
_START = (
    r"(?:^|(?<=[.!?;)\]])(?=" + _WS + r"))" + _WS + r"*"  # a line start, or just after the end of a sentence
    r"(?:#{1,6}" + _WS + r"+)?"  # a markdown heading
    r"(?:(?:[-*•]|\d{1,2}[.)])" + _WS + r"+)?"  # a bullet or a number
    r"(?:" + _MARK + _WS + r"*)?"  # opening bold
)
_NOTHING = r"Nothing" + _WS + r"+newer"
_LABEL_RE = re.compile(
    _START + r"(?:"
    # 'Label:' with an optional qualifier before the colon: 'Update (March 2026):', 'Update 1:'; 'Updates:' and
    # 'Correction:' are updates; closing bold may sit before the colon ('**Update**:')
    r"(?P<label>Known|Confirm|Updates?|Corrections?|Newer|Unverified|" + _NOTHING
    + r"|Newest" + _WS + r"+I" + _WS + r"+know)"
    r"(?:" + _WS + r"*\([^)\n]{0,60}\)|" + _WS + r"+\d{1,2})?" + _WS + r"*(?:" + _MARK + _WS + r"*)?:"
    # 'Nothing newer than March 2026' (with or without the colon after the date): the only label with a qualifier
    r"|(?P<than>" + _NOTHING + r")(?:" + _WS + r"*" + _MARK + r")?" + _WS + r"+than\b"
    r")",
    re.IGNORECASE | re.MULTILINE)

# ---- dates ---------------------------------------------------------------------------------------------------------

_MONTHS = {
    **{name: i + 1 for i, name in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                              "september", "october", "november", "december"))},
    **{name: i + 1 for i, name in enumerate(("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct",
                                              "nov", "dec"))},
    "sept": 9,
    **{name: i + 1 for i, name in enumerate(("januar", "februar", "märz", "april", "mai", "juni", "juli", "august",
                                              "september", "oktober", "november", "dezember"))},
    "maerz": 3, "jän": 1, "mär": 3, "mrz": 3, "okt": 10, "dez": 12,
}
_MONTH = r"(0?[1-9]|1[0-2])"
_DAY = r"(?:0?[1-9]|[12]\d|3[01])"
_ISO = re.compile(r"(?<![\d.])(20\d{2})[-/.](0[1-9]|1[0-2])(?!\d)")  # 2026-03, 2026/03, 2026.03, 2026-03-17
# (a number glued to a letter, a dot or a hyphen is no day or month: 'H2-2025', 'v1.3.2026', '1.2.3.2026')
_DMY = re.compile(  # 17.03.2026, 1.3.2026, 17/03/2026
    r"(?<![\w.\-/])" + _DAY + r"[./]" + _MONTH + r"[./](20\d{2})(?!\d)")
_MY = re.compile(r"(?<![\w.\-/])" + _MONTH + r"[-/.](20\d{2})(?!\d)")  # 03/2026, 3.2026, 03-2026
_NAMED = re.compile(  # March 2026, March, 2026, Feb-2026, Feb.2026, March 17, 2026
    r"\b([A-Za-zäöüÄÖÜ]{3,9})(?:[ \t]*[,.\-/][ \t]*|[ \t]+)"
    r"(?:\d{1,2}(?:st|nd|rd|th)?\.?,?[ \t]+)?(20\d{2})(?!\d)")
_QUARTER = re.compile(r"\bQ([1-4])(?:[ \t]*[/\-][ \t]*|[ \t]*)(20\d{2})(?!\d)", re.IGNORECASE)  # Q2 2026, Q4-2025
# a bare year: not part of a longer number, a date or a version ('2024.5'); 'mid-2026' counts, '2025-2026' does not
_YEAR = re.compile(r"(?<![\d./])(?<!\d-)(20\d{2})(?![\d\-/]|\.\d)")


def month_number(month: str | None) -> int | None:
    """``'2026-03'`` -> ``2026 * 12 + 2``; None for anything that is not ``YYYY-MM``."""
    if month and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        return int(month[:4]) * 12 + int(month[5:7]) - 1
    return None


def _label(number: int) -> str:
    return f"{number // 12:04d}-{number % 12 + 1:02d}"


def parse_month(text: str, today: _dt.date | str | None = None) -> tuple[str | None, bool, bool]:
    """The latest date in ``text`` as ``(YYYY-MM | None, approx, future)``.

    Understood: ``2026-03``, ``2026-03-17``, ``03/2026``, ``03-2026``, day-first ``17.03.2026`` / ``1.3.2026`` /
    ``17/03/2026`` (month-first US dates such as ``03/17/2026`` are not read), ``March 2026`` / ``Mar 2026`` /
    ``März 2026`` (also with a day, and with a comma, dot, hyphen or slash before the year: ``March, 2026``,
    ``Feb-2026``), ``Q2 2026`` / ``Q4-2025`` (first month of the quarter, ``approx``) and a bare year, also in
    ``mid-2026`` or ``late 2025`` (January, ``approx``). Version numbers (``2024.5``, ``1.2.3.2026``) are no dates.
    Dates after ``today``'s month are ``future``: they never become the result (a text with only such dates gives
    ``None, False, True``). A token that has been read as a date is not read again as a bare year, so ``December
    2026`` on 2026-11-26 is only a future date and does not leave a January 2026 behind.
    """
    if isinstance(today, str):
        today = _dt.date.fromisoformat(today)
    today = today or _dt.date.today()
    limit = today.year * 12 + today.month - 1
    found: list[tuple[int, bool]] = []
    work = text

    def sweep(pattern: re.Pattern[str], read: Callable[[re.Match[str]], tuple[int, bool] | None]) -> None:
        """Collect the dates ``read`` finds in ``work`` and blank the matched spans, so later patterns skip them."""
        nonlocal work

        def blank(m: re.Match[str]) -> str:
            item = read(m)
            if item is None:
                return m.group(0)
            found.append(item)
            return "\x00" * len(m.group(0))

        work = pattern.sub(blank, work)

    sweep(_ISO, lambda m: (int(m[1]) * 12 + int(m[2]) - 1, False))
    sweep(_QUARTER, lambda m: (int(m[2]) * 12 + 3 * (int(m[1]) - 1), True))  # before 'Q4-2025' reads as 4-2025
    sweep(_DMY, lambda m: (int(m[2]) * 12 + int(m[1]) - 1, False))
    sweep(_MY, lambda m: (int(m[2]) * 12 + int(m[1]) - 1, False))
    sweep(_NAMED, lambda m: (int(m[2]) * 12 + _MONTHS[m[1].lower()] - 1, False) if m[1].lower() in _MONTHS else None)
    sweep(_YEAR, lambda m: (int(m[1]) * 12, True))
    future = any(n > limit for n, _ in found)
    usable = [(n, a) for n, a in found if n <= limit]
    if not usable:
        return None, False, future
    best = max(n for n, _ in usable)
    approx = all(a for n, a in usable if n == best)
    return _label(best), approx, future


def parse_note(text: str, today: _dt.date | str | None = None) -> dict[str, Any]:
    """Counts of the rally labels in ``text`` and what the note says about its writer's newest knowledge.

    ``labels``: counts of ``known``, ``confirm``, ``update`` (``Update:``, ``Updates:``, ``Correction:``, also with a
    qualifier such as ``Update (March 2026):``), ``newer``, ``unverified`` and ``nothing_newer`` (also the colon-less
    ``Nothing newer than March 2026`` at the start of a line or sentence).
    ``newest_raw``: the text after the last ``Newest I know:`` up to the end of its line (None without the line);
    ``newest`` / ``approx`` / ``future`` come from it (:func:`parse_month`). ``mentioned``: the latest non-future
    month anywhere in the note.
    """
    counts = {"known": 0, "confirm": 0, "update": 0, "newer": 0, "unverified": 0, "nothing_newer": 0}
    newest_raw: str | None = None
    for m in _LABEL_RE.finditer(text):
        name = _LABEL_NAMES[" ".join((m.group("label") or m.group("than")).lower().split())]
        if name == "newest":
            end = text.find("\n", m.end())
            newest_raw = text[m.end(): end if end >= 0 else len(text)].strip().lstrip("*_ \t") or None
        else:
            counts[name] += 1
    newest, approx, future = parse_month(newest_raw, today) if newest_raw else (None, False, False)
    mentioned = parse_month(text, today)[0]
    return {"labels": counts, "newest_raw": newest_raw, "newest": newest, "approx": approx, "future": future,
            "mentioned": mentioned}


class RallyState:
    """Per-run bookkeeping of the rally: the committed notes, who writes next, when to stop and the metrics."""

    def __init__(self, cfg: RallyConfig, ring: Sequence[str], today: _dt.date | str | None = None) -> None:
        if len(ring) < 2:
            raise ValueError("the rally needs at least two models in its ring")
        self.cfg = cfg
        self.ring = list(ring)
        self.today = _dt.date.fromisoformat(today) if isinstance(today, str) else (today or _dt.date.today())
        self.notes: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self.stop_reason: str | None = None
        self._failures: dict[str, int] = {}

    # ---- schedule ----

    @property
    def min_turns(self) -> int:
        return self.cfg.min_turns if self.cfg.min_turns is not None else 2 * len(self.ring)

    @property
    def quiet_turns(self) -> int:
        return self.cfg.quiet_turns if self.cfg.quiet_turns is not None else len(self.ring)

    @property
    def turns(self) -> int:
        return len(self.notes)

    def kind(self) -> str:
        """What the next note is: ``"open"`` (the first one) or ``"reply"``."""
        return "open" if not self.notes else "reply"

    def previous(self) -> dict[str, Any] | None:
        return self.notes[-1] if self.notes else None

    def next_model(self, available: Sequence[str]) -> str | None:
        """The writer of the next note: the model after the last writer in ring order that is in ``available`` (the
        ring's first available model for the first note); the last writer again only if no other ring model is left."""
        candidates = [m for m in self.ring if m in available]
        if not candidates:
            return None
        if not self.notes or self.notes[-1]["model"] not in self.ring:
            return candidates[0]
        start = self.ring.index(self.notes[-1]["model"])
        for step in range(1, len(self.ring) + 1):
            model = self.ring[(start + step) % len(self.ring)]
            if model in candidates:
                return model
        return candidates[0]

    def frontier(self) -> str | None:
        """The newest month the notes declare so far (``YYYY-MM``), a running maximum over the writers' own
        ``Newest I know:`` lines; None while no note has a readable one. Future dates never enter it."""
        best = max((n for n in (month_number(note["newest"]) for note in self.notes) if n is not None), default=None)
        return None if best is None else _label(best)

    # ---- notes ----

    def add_note(self, index: int, model: str, text: str, *, full: str | None = None, cut: bool = False,
                 light: bool = False) -> dict[str, Any]:
        """Parse and record a committed think note; returns its record (the same dict as in ``metrics()["notes"]``).

        ``text`` is the committed note, ``full`` what the writer wrote before the relay's hard cut (default: ``text``);
        labels and dates are read from ``full``, so a note whose closing line the cut removed is still understood,
        and ``cut`` (the committed note is shorter than what the writer wrote, or its stream was truncated) is kept
        in the record. ``light`` marks a note written under the light prompt (it may carry no labels at all).

        A reply is ``quiet`` (it adds nothing) when all of this holds: it is no ``self_reply`` (the previous note is
        by the same model, because the partner could not take its turn), it has no ``Update:`` / ``Correction:`` and
        no ``Newer:`` line, it does not declare a newest month above the frontier before it, and its end was
        readable (a cut note without a readable ``Newest I know:`` line may have lost its ``Newer:`` lines), and it
        says so: a ``Confirm:``, ``Unverified:`` or ``Nothing newer:`` line, or, without any label, a readable
        ``Newest I know:`` that is not above the frontier. The opening note is never quiet.
        """
        parsed = parse_note(text if full is None else full, self.today)
        labels = parsed["labels"]
        kind = self.kind()
        self_reply = bool(self.notes) and self.notes[-1]["model"] == model
        newest = parsed["newest"]
        top, declared = month_number(self.frontier()), month_number(newest)
        raised = declared is not None and top is not None and declared > top
        unreadable = cut and newest is None
        says_so = (labels["confirm"] + labels["unverified"] + labels["nothing_newer"] > 0
                   or (not any(labels.values()) and newest is not None))
        note = {"index": index, "model": model, "kind": kind, "labels": labels, "newest": newest,
                "newest_raw": parsed["newest_raw"], "approx": parsed["approx"], "future": parsed["future"],
                "mentioned": parsed["mentioned"],
                "quiet": (kind == "reply" and not self_reply and labels["update"] == 0 and labels["newer"] == 0
                          and not raised and not unreadable and says_so),
                "self_reply": self_reply, "cut": cut, "light": light}
        self.notes.append(note)
        self._failures.pop(model, None)  # a note is a successful hand-over
        return note

    def hand_over_failed(self, model: str, *, fatal: bool = False) -> bool:
        """Count a failed hand-over (the model returned an error instead of a note). True when the relay must take the
        model out of the run: the failure is ``fatal`` (not retryable) or it is the ``rally.partner_failures``-th
        consecutive one. A committed note by the model, or :meth:`hand_over_ok`, ends the streak."""
        self._failures[model] = self._failures.get(model, 0) + 1
        return bool(fatal) or self._failures[model] >= self.cfg.partner_failures

    def hand_over_ok(self, model: str) -> None:
        """The model took its turn: its streak of failed hand-overs is over."""
        self._failures.pop(model, None)

    def note_skipped(self, index: int, model: str, reason: str) -> None:
        """Record that ``model`` did not take its turn for think segment ``index`` (reported in the metrics)."""
        self.skipped.append({"index": index, "model": model, "reason": reason})

    def self_reply_streak(self) -> int:
        """How many of the latest notes in a row were written to themselves (by the writer of the note before)."""
        streak = 0
        for note in reversed(self.notes):
            if not note["self_reply"]:
                break
            streak += 1
        return streak

    def should_stop(self, partners_left: int | None = None) -> str | None:
        """``"max_turns"``, ``"converged"``, ``"no_partner"`` or None, checked in this order.

        ``converged``: at least ``min_turns`` notes, and the last ``quiet_turns`` are all quiet and come from at least
        ``min(quiet_turns, models that can still write)`` different writers (a revolution of the ring, not one model
        agreeing with itself; ``partners_left`` is that number of ring models, default: the whole ring).
        ``no_partner``: fewer than two ring models can still write, or the last two notes were both self-replies
        (one fallback note is allowed, so a hand-over that failed once can recover)."""
        n = len(self.notes)
        if n >= self.cfg.max_turns:
            return "max_turns"
        if n >= self.min_turns and n >= self.quiet_turns + 1:
            window = self.notes[-self.quiet_turns:]
            needed = min(self.quiet_turns, len(self.ring) if partners_left is None else max(partners_left, 1))
            if all(x["quiet"] for x in window) and len({x["model"] for x in window}) >= needed:
                return "converged"
        if (partners_left is not None and partners_left < 2) or self.self_reply_streak() >= 2:
            return "no_partner"
        return None

    def quiet_streak(self) -> int:
        streak = 0
        for note in reversed(self.notes):
            if not note["quiet"]:
                break
            streak += 1
        return streak

    # ---- metrics ----

    def metrics(self, *, preset: str | None = None) -> dict[str, Any]:
        """``done.data["rally"]`` (DESIGN_rally.md section 6).

        ``frontier.by_turn`` is the running maximum of the declared newest months. An approximate date (a bare year, a
        quarter) counts as the first month of its period, so it never lifts the frontier above an exact date of that
        period; when an exact date reaches the same month, the exact note is the one that holds the maximum.
        ``by_turn_approx`` says whether the note holding the maximum at that turn was approximate.
        ``gain_months`` is the final frontier minus the latest first declaration of the writers (what the exchange added
        to the best opening note), and None unless every writer's first note was dated and the number is certain: the
        note holding the final frontier and the note holding that latest first declaration are exact dates, and no
        approximate first declaration could reach it (a bare year may stand for up to eleven months after its first
        month, so a writer who opened with "2025" and later says "October 2025" has shown precision, not a lift).
        Otherwise the number is reported as ``gain_months_approx`` and ``gain_months`` stays None.
        """
        by_turn: list[str | None] = []
        by_turn_approx: list[bool] = []
        best: int | None = None
        best_approx = False
        first: dict[str, str | None] = {}
        first_approx: dict[str, bool] = {}
        adds: dict[str, dict[str, int]] = {}
        for note in self.notes:
            declared = month_number(note["newest"])
            if declared is not None and (best is None or declared > best
                                         or (declared == best and best_approx and not note["approx"])):
                best, best_approx = declared, note["approx"]
            by_turn.append(None if best is None else _label(best))
            by_turn_approx.append(best_approx if best is not None else False)
            if note["model"] not in first:
                first[note["model"]] = note["newest"]
                first_approx[note["model"]] = note["approx"]
            row = adds.setdefault(note["model"], {name: 0 for name in REPLY_LABELS})
            for name in REPLY_LABELS:
                row[name] += note["labels"][name]
        gain: int | None = None
        gain_approx: int | None = None
        openings = {m: month_number(v) for m, v in first.items()}
        if best is not None and openings and all(n is not None for n in openings.values()):
            base = max(n for n in openings.values() if n is not None)
            base_approx = all(first_approx[m] for m, n in openings.items() if n == base)
            reaches = any(first_approx[m] and n is not None and n + 11 >= base for m, n in openings.items())
            if best_approx or base_approx or reaches:
                gain_approx = best - base
            else:
                gain = best - base
        return {
            "ring": list(self.ring), "preset": preset, "turns": len(self.notes), "stop_reason": self.stop_reason,
            "quiet_streak": self.quiet_streak(), "min_turns": self.min_turns, "max_turns": self.cfg.max_turns,
            "quiet_turns": self.quiet_turns,
            "notes": [dict(n) for n in self.notes],
            "frontier": {"self_reported": True, "by_turn": by_turn, "by_turn_approx": by_turn_approx,
                         "first_declared": first, "final": by_turn[-1] if by_turn else None,
                         "gain_months": gain, "gain_months_approx": gain_approx},
            "adds_by_model": adds,
            "unlabelled_replies": sum(1 for n in self.notes if n["kind"] == "reply"
                                      and not any(n["labels"][k] for k in REPLY_LABELS)),
            "self_replies": sum(1 for n in self.notes if n["self_reply"]),
            "skipped_hand_overs": [dict(s) for s in self.skipped],
            "hand_overs": max(len(self.notes) - 1, 0),
        }
