"""Masking and name helpers of the key experiment (docs/DESIGN_keys.md section 5).

The masking helpers (:func:`mask_title`, :func:`mask_note`) work on the original text through an accent-free,
casefolded copy whose characters keep their positions in the original, so a name written "Pióro" or "Pioro",
"Jun-sol" or "Junsol" is found either way and replaced in place.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
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
    normalize_text,
)
from cutoffmix.bench.keys.common import PROBE, TITLE, WITHHELD
from cutoffmix.types import Event

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



# ------------------------------------------------------------------------------------------- adoption (contract)
# Interim implementations of the adoption rules of design section 6 (``offers`` and the three-way ``adoption_label``);
# their signatures are the contract the run and the report rely on.

ADOPTION_LABELS = ("endorsed", "reported", "absent")


def offers(response: str, name: str, event: Event) -> bool:
    """Whether ``response`` offers ``name`` as an answer (a full form, or its distinctive surname alone) in an
    asserted clause (design section 6)."""
    return _asserts(response, name, event)


def adoption_label(response: str, name: str, event: Event) -> str:
    """``"endorsed"`` (``name`` offered in the answer's own voice), ``"reported"`` (offered only in clauses that refer
    to the note, which the rules cannot decide) or ``"absent"`` (not offered) (design section 6)."""
    return "endorsed" if offers(response, name, event) else "absent"
