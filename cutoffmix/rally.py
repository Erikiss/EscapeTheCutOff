"""The rally preset's note parsing and bookkeeping (DESIGN_rally.md): which labelled lines a note has, the newest month
its writer says it knows, when the exchange has run dry, and the metrics of a run.

Everything here is pure: the relay (``relay.py``) feeds it the committed notes and asks it who writes next and whether
to stop. The dates are the models' own statements ("self-reported"); the project rule that self-reports do not count
is why these numbers describe what the exchange did and are no proof that it helped.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cutoffmix.config import RallyConfig

__all__ = ["RallyState", "month_number", "parse_month", "parse_note", "REPLY_LABELS"]

REPLY_LABELS = ("confirm", "update", "newer", "unverified", "nothing_newer")
_LABEL_NAMES = {
    "known": "known", "confirm": "confirm", "update": "update", "newer": "newer", "unverified": "unverified",
    "nothing newer": "nothing_newer", "newest i know": "newest",
}
_MARK = r"(?:\*\*|__)?"
# a label opens a line (after an optional bullet) or follows the end of a sentence in a run-together note
_LABEL_RE = re.compile(
    r"(?:^|(?<=[.!?;)\]])[ \t]+)[ \t]*(?:[-*•][ \t]+|\d{1,2}[.)][ \t]+)?" + _MARK + r"[ \t]*"
    r"(Known|Confirm|Update|Newer|Unverified|Nothing[ \t]+newer|Newest[ \t]+I[ \t]+know)[ \t]*" + _MARK + r"[ \t]*:",
    re.IGNORECASE | re.MULTILINE)
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
_ISO = re.compile(r"(?<![\d.])(20\d{2})[-/.](0[1-9]|1[0-2])(?!\d)")
_MY = re.compile(r"(?<![\d.\-/])(0?[1-9]|1[0-2])[/.](20\d{2})(?!\d)")
_NAMED = re.compile(r"\b([A-Za-zäöüÄÖÜ]{3,9})\.?[ \t]+(?:\d{1,2}(?:st|nd|rd|th)?\.?,?[ \t]+)?(20\d{2})(?!\d)")
_QUARTER = re.compile(r"\bQ([1-4])[ \t]*/?[ \t]*(20\d{2})(?!\d)", re.IGNORECASE)
_YEAR = re.compile(r"(?<![\d\-/.])(20\d{2})(?![\d\-/]|\.\d)")


def month_number(month: str | None) -> int | None:
    """``'2026-03'`` -> ``2026 * 12 + 2``; None for anything that is not ``YYYY-MM``."""
    if month and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        return int(month[:4]) * 12 + int(month[5:7]) - 1
    return None


def _label(number: int) -> str:
    return f"{number // 12:04d}-{number % 12 + 1:02d}"


def parse_month(text: str, today: _dt.date | str | None = None) -> tuple[str | None, bool, bool]:
    """The latest date in ``text`` as ``(YYYY-MM | None, approx, future)``.

    Understood: ``2026-03``, ``03/2026``, ``March 2026`` / ``Mar 2026`` / ``März 2026`` (also with a day), ``Q2 2026``
    (first month of the quarter, ``approx``) and a bare year (January, ``approx``). Dates after ``today``'s month are
    ``future``: they never become the result (a text with only such dates gives ``None, False, True``).
    """
    if isinstance(today, str):
        today = _dt.date.fromisoformat(today)
    today = today or _dt.date.today()
    limit = today.year * 12 + today.month - 1
    found: list[tuple[int, bool]] = []
    for m in _ISO.finditer(text):
        found.append((int(m[1]) * 12 + int(m[2]) - 1, False))
    for m in _MY.finditer(text):
        found.append((int(m[2]) * 12 + int(m[1]) - 1, False))
    for m in _NAMED.finditer(text):
        month = _MONTHS.get(m[1].lower())
        if month is not None:
            found.append((int(m[2]) * 12 + month - 1, False))
    for m in _QUARTER.finditer(text):
        found.append((int(m[2]) * 12 + 3 * (int(m[1]) - 1), True))
    for m in _YEAR.finditer(text):
        found.append((int(m[1]) * 12, True))
    future = any(n > limit for n, _ in found)
    usable = [(n, a) for n, a in found if n <= limit]
    if not usable:
        return None, False, future
    best = max(n for n, _ in usable)
    approx = all(a for n, a in usable if n == best)
    return _label(best), approx, future


def parse_note(text: str, today: _dt.date | str | None = None) -> dict[str, Any]:
    """Counts of the rally labels in ``text`` and what the note says about its writer's newest knowledge.

    ``labels``: counts of ``known``, ``confirm``, ``update``, ``newer``, ``unverified``, ``nothing_newer``.
    ``newest_raw``: the text after the last ``Newest I know:`` up to the end of its line (None without the line);
    ``newest`` / ``approx`` / ``future`` come from it (:func:`parse_month`). ``mentioned``: the latest non-future
    month anywhere in the note.
    """
    counts = {"known": 0, "confirm": 0, "update": 0, "newer": 0, "unverified": 0, "nothing_newer": 0}
    newest_raw: str | None = None
    for m in _LABEL_RE.finditer(text):
        name = _LABEL_NAMES[re.sub(r"[ \t]+", " ", m.group(1).lower())]
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
        self.stop_reason: str | None = None

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

    def add_note(self, index: int, model: str, text: str) -> dict[str, Any]:
        """Parse and record a committed think note; returns its record (the same dict as in ``metrics()["notes"]``)."""
        parsed = parse_note(text, self.today)
        labels = parsed["labels"]
        kind = self.kind()
        replied = any(labels[name] for name in REPLY_LABELS)
        note = {"index": index, "model": model, "kind": kind, "labels": labels, "newest": parsed["newest"],
                "newest_raw": parsed["newest_raw"], "approx": parsed["approx"], "future": parsed["future"],
                "mentioned": parsed["mentioned"],
                "quiet": kind == "reply" and replied and labels["update"] == 0 and labels["newer"] == 0}
        self.notes.append(note)
        return note

    def should_stop(self, partners_left: int | None = None) -> str | None:
        """``"max_turns"``, ``"converged"``, ``"no_partner"`` (fewer than two ring models can still write) or None."""
        n = len(self.notes)
        if n >= self.cfg.max_turns:
            return "max_turns"
        if n >= self.min_turns and n >= self.quiet_turns + 1 and all(x["quiet"] for x in self.notes[-self.quiet_turns:]):
            return "converged"
        if partners_left is not None and partners_left < 2:
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
        """``done.data["rally"]`` (DESIGN_rally.md section 6)."""
        by_turn: list[str | None] = []
        best: int | None = None
        first: dict[str, str | None] = {}
        adds: dict[str, dict[str, int]] = {}
        for note in self.notes:
            declared = month_number(note["newest"]) if not note["future"] or note["newest"] else None
            if declared is not None and (best is None or declared > best):
                best = declared
            by_turn.append(None if best is None else _label(best))
            first.setdefault(note["model"], note["newest"])
            row = adds.setdefault(note["model"], {name: 0 for name in REPLY_LABELS})
            for name in REPLY_LABELS:
                row[name] += note["labels"][name]
        opening = [month_number(v) for v in first.values() if month_number(v) is not None]
        gain = None if best is None or not opening else best - max(opening)
        return {
            "ring": list(self.ring), "preset": preset, "turns": len(self.notes), "stop_reason": self.stop_reason,
            "quiet_streak": self.quiet_streak(), "min_turns": self.min_turns, "max_turns": self.cfg.max_turns,
            "quiet_turns": self.quiet_turns,
            "notes": [dict(n) for n in self.notes],
            "frontier": {"self_reported": True, "by_turn": by_turn, "first_declared": first,
                         "final": by_turn[-1] if by_turn else None, "gain_months": gain},
            "adds_by_model": adds,
            "unlabelled_replies": sum(1 for n in self.notes if n["kind"] == "reply"
                                      and not any(n["labels"][k] for k in REPLY_LABELS)),
            "hand_overs": max(len(self.notes) - 1, 0),
        }
