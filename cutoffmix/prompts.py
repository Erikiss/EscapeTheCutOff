"""Relay prompt templates and segment post-processing (DESIGN.md §3.2).

One neutral template ("template B") is used for every model: several analysts append the next intermediate result
plus its evidence to shared notes. The wording deliberately avoids the words thinking/reasoning/scratchpad and any
XML-like tags, which Anthropic lists as ``reasoning_extraction`` refusal triggers.

Request bodies must stay append-only so provider prompt caches keep hitting: the system prompt is static for a run,
the user message is a constant header block, one block per committed segment, and a constant tail block.
"""

from __future__ import annotations

import re

RELAY_SYSTEM_TEMPLATE = (
    "Several analysts are jointly working through the problem in the user message; their shared notes so far are "
    "included. Contribute the next intermediate result: state it and the evidence or calculation that supports it, "
    "in about {K} tokens, written so it can be appended to the notes as the next paragraph. Use the same language as "
    "the notes (or as the problem if there are no notes yet). If an earlier note is wrong, start with 'Correction:' "
    "and fix it. Prefer the most recent facts you know and say when a fact may have changed. Output only that "
    "paragraph, then {sentinel}. If the notes already reach a conclusion, write '{final_marker}' followed by the "
    "answer."
)
ANSWER_SYSTEM_TEMPLATE = (
    "Several analysts have worked through the problem in the user message; their shared notes are included. Write "
    "the final answer for the person who asked: answer the problem directly and concisely, in the language of the "
    "problem. Rely on the notes where they hold up and correct them where they are wrong or outdated. Prefer the "
    "most recent facts you know and say when a fact may have changed. Do not describe the notes or the analysts."
)
DATE_SENTENCE = " Today's date: {today}."

HEADER_TEMPLATE = "PROBLEM:\n{question}\n\nSHARED NOTES SO FAR:\n\n"
NO_NOTES = "(no notes yet)\n\n"
RELAY_TAIL = "END OF THE SHARED NOTES. Write the next paragraph."
ANSWER_TAIL = "END OF THE SHARED NOTES. Write the final answer to the problem."


def relay_system_prompt(target_tokens: int, today: str | None, sentinel: str, final_marker: str) -> str:
    """The static system prompt for think-phase segments; the date sentence is omitted when ``today`` is None."""
    text = RELAY_SYSTEM_TEMPLATE.format(K=target_tokens, sentinel=sentinel, final_marker=final_marker)
    return text + (DATE_SENTENCE.format(today=today) if today else "")


def answer_system_prompt(today: str | None) -> str:
    """The system prompt for the final-answer call; the date sentence is omitted when ``today`` is None."""
    return ANSWER_SYSTEM_TEMPLATE + (DATE_SENTENCE.format(today=today) if today else "")


def _note_blocks(question: str, segments: list[str]) -> list[dict]:
    blocks = [{"type": "text", "text": HEADER_TEMPLATE.format(question=question.strip())}]
    if segments:
        blocks += [{"type": "text", "text": s.rstrip() + "\n\n"} for s in segments]
    else:
        blocks.append({"type": "text", "text": NO_NOTES})
    return blocks


def relay_user_blocks(question: str, segments: list[str]) -> list[dict]:
    """User content for a think-phase request: header, one block per segment (or "(no notes yet)"), constant tail.

    Every block except the tail is identical across consecutive requests of a run, so the content is append-only.
    """
    return [*_note_blocks(question, segments), {"type": "text", "text": RELAY_TAIL}]


def answer_user_blocks(question: str, segments: list[str]) -> list[dict]:
    """User content for the final-answer request: same header and segment blocks, answer tail."""
    return [*_note_blocks(question, segments), {"type": "text", "text": ANSWER_TAIL}]


# --------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------

_OVERLAP_WINDOW = 2000  # longest echoed overlap we look for, in characters


def trim_overlap(notes_tail: str, new: str, min_len: int = 12) -> str:
    """Drop the longest prefix of ``new`` (at least ``min_len`` chars) that repeats the end of the notes.

    Models sometimes echo the last sentence of the notes before continuing. Leading whitespace of ``new`` and
    trailing whitespace of the notes are ignored when matching. Without such an overlap ``new`` is returned unchanged.
    """
    tail = notes_tail.rstrip()[-_OVERLAP_WINDOW:]
    body = new.lstrip()
    for n in range(min(len(tail), len(body)), min_len - 1, -1):
        if tail.endswith(body[:n]):
            return body[n:]
    return new


# Latin terminators need following whitespace ("3.5" is not an end); CJK full-width ones do not.
_SENTENCE_END = re.compile(r"[.!?…][\"'”’“»)\]]*(?=\s|$)|[。！？][」』”’）)]*")
_DOTTED_ABBREVIATION = re.compile(r"(?:[^\W\d_]\.)+[^\W\d_]")  # "u.s", "z.b" (word before the final dot)
# A standalone 1-2 digit number before the dot is a German ordinal ("zum 24. November", "im 3. Quartal") or a day
# range ("10./11.", "10.–12."); years ("2026.") and decimals ("3.5.") still end a sentence.
_ORDINAL = re.compile(r"\d{1,2}(?:\.[/–-]\d{1,2})?")
_ABBREVIATIONS = frozenset({
    # English
    "e.g", "i.e", "etc", "vs", "cf", "approx", "ca", "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "inc",
    "ltd", "co", "corp", "fig", "eq", "vol", "al", "gen", "gov", "sen", "rep", "lt", "col", "sgt", "capt", "mt",
    # German
    "nr", "z.b", "bzw", "usw", "vgl", "d.h", "u.a", "ggf", "evtl", "inkl", "bzgl", "zzgl", "bspw", "sog", "geb",
    "gest", "hrsg", "bd", "jh", "mio", "mrd", "tsd",
    # month abbreviations (English and German)
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "mär", "mrz", "okt", "dez",
})
# Abbreviations only when a number follows ("No. 5", "Art. 3", "pp. 4-7"): "no." also ends sentences.
_NUMBER_ABBREVIATIONS = frozenset({"no", "nos", "art", "abs", "pp"})
_NUMBER_NEXT = re.compile(r"\s+\d")


def _is_sentence_end(text: str, match: re.Match[str]) -> bool:
    if text[match.start()] != ".":
        return True
    words = text[: match.start()].split()
    word = (words[-1] if words else "").lstrip("([\"'“„«").lower()
    if len(word) == 1 and word.isalpha():  # initials such as "J. Smith", and "z. B.", "u. a.", "d. h."
        return False
    if _ORDINAL.fullmatch(word):
        return False
    if word in _NUMBER_ABBREVIATIONS and _NUMBER_NEXT.match(text, match.end()):
        return False
    return word not in _ABBREVIATIONS and not _DOTTED_ABBREVIATION.fullmatch(word)


def trim_to_sentence(text: str, max_chars: int) -> str:
    """Shorten ``text`` to at most ``max_chars``, ending at the last sentence end or line break inside the limit.

    A dot after an abbreviation (``z. B.``, ``bzw.``, ``e.g.``, ``Okt.``, ``No. 5``), an initial or a standalone
    1-2 digit ordinal (``zum 24. November``) is not a sentence end. Text already within the limit is returned
    (stripped) unchanged. If the first sentence alone is longer than the limit, the cut falls on the last whitespace,
    or exactly at ``max_chars`` if there is none.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text
    best = 0
    for m in _SENTENCE_END.finditer(text):
        if m.end() > max_chars:
            break
        if _is_sentence_end(text, m):
            best = m.end()
    newline = text.rfind("\n", 0, max_chars + 1)
    best = max(best, newline if newline > 0 else 0)
    if best <= 0:
        space = text.rfind(" ", 0, max_chars + 1)
        best = space if space > 0 else max_chars
    return text[:best].rstrip()


_FENCE_OPEN = re.compile(r"^```[\w+.-]*[ \t]*(?:\n|$)")
_FENCE_CLOSE = re.compile(r"(?:^|\n)[ \t]*```[ \t]*$")
_LABEL = re.compile(
    r"^(?:[*_#>]+[ \t]*)*"
    r"(?:here(?:'s| is)[ \t]+(?:the|my)[ \t]+)?"
    r"(?:next[ \t]+paragraph|next[ \t]+intermediate[ \t]+result|intermediate[ \t]+result|next[ \t]+result"
    r"|next[ \t]+step|next[ \t]+note|continuation|contribution|paragraph|analyst(?:[ \t]+\d+)?"
    r"|fortsetzung|n[äa]chster[ \t]+absatz|n[äa]chstes[ \t]+zwischenergebnis|zwischenergebnis"
    r"|n[äa]chster[ \t]+schritt|beitrag)"
    r"[ \t]*(?:[*_]+[ \t]*)?:[ \t]*(?:[*_]+[ \t]*)?\n?",
    re.IGNORECASE,
)
_QUOTE_PAIRS = {'"': ('"', "”"), "“": ("”", '"'), "„": ("“", '"'), "«": ("»",), "»": ("«",), "'": ("'", "’"),
                "‘": ("’", "'")}


def _strip_enclosing_quotes(text: str) -> str:
    """Remove quotes around the whole text, or an unbalanced leading quote; a quoted title at the start is kept."""
    if not text or text[0] not in _QUOTE_PAIRS:
        return text
    closings = _QUOTE_PAIRS[text[0]]
    inner = text[1:]
    if inner.endswith(closings) and not any(c in inner[:-1] for c in closings):
        return inner[:-1].strip()
    if not any(c in inner for c in closings):
        return inner.strip()
    return text


def strip_preamble(text: str) -> str:
    """Remove wrappers models put around a paragraph: labels ("Next paragraph:", "Continuation:", ...) including echoes
    of the template ("Next result:", "Zwischenergebnis:", "Nächster Schritt:", ...), code fences around the whole
    text, and enclosing quotes. Content-bearing openers ("Correction:", "Korrektur:") and the final-answer marker are
    kept."""
    out = text.strip()
    while True:
        before = out
        if _FENCE_OPEN.match(out):
            out = _FENCE_CLOSE.sub("", _FENCE_OPEN.sub("", out, count=1)).strip()
        out = _LABEL.sub("", out, count=1).strip()
        out = _strip_enclosing_quotes(out)
        if out == before:
            return out


def _sentinel_pattern(sentinel: str) -> re.Pattern[str] | None:
    """Tolerant pattern for mangled sentinels such as ``<END_STEP>`` or ``<<end step>>`` (needs an opening bracket)."""
    core = sentinel.strip("<>[]{}() ")
    if len(core) < 3:
        return None
    core_re = r"[_ ]".join(re.escape(part) for part in re.split(r"[_ ]", core))
    return re.compile(r"[<\[]{1,2}[ \t]*" + core_re + r"[ \t]*[>\]]{0,2}", re.IGNORECASE)


def _cut_at_sentinel(raw: str, sentinel: str) -> tuple[str, bool]:
    idx = raw.find(sentinel)
    if idx >= 0:
        return raw[:idx], True
    pattern = _sentinel_pattern(sentinel)
    m = pattern.search(raw) if pattern else None
    if m:
        return raw[: m.start()], True
    # a stream cut mid-sentinel leaves a prefix of it at the very end
    for n in range(min(len(sentinel) - 1, len(raw)), 1, -1):
        if raw.endswith(sentinel[:n]):
            return raw[:-n], False
    return raw, False


def clean_segment(raw: str, notes_tail: str, *, sentinel: str, hard_cut_chars: int) -> tuple[str, bool]:
    """Turn raw streamed text into a committable segment; returns ``(text, saw_sentinel)``.

    Steps: cut at the sentinel (also tolerating mangled forms and a truncated sentinel at the end), strip preambles,
    remove an echoed overlap with ``notes_tail``, hard-cut to ``hard_cut_chars`` and trim to a sentence end.
    """
    text, saw = _cut_at_sentinel(raw, sentinel)
    text = strip_preamble(text)
    text = trim_overlap(notes_tail, text)
    return trim_to_sentence(text, hard_cut_chars), saw


def _marker_pattern(final_marker: str) -> re.Pattern[str]:
    """Case-insensitive marker at the start of a line, tolerating markdown such as ``**Final answer:**``."""
    core = final_marker.strip().rstrip(":").strip()
    core_re = r"\s+".join(re.escape(w) for w in core.split())
    tail = r"[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(?:\*\*|__)?" if final_marker.strip().endswith(":") else r"\b(?:\*\*|__)?"
    return re.compile(r"(?im)^[ \t>#*_-]*" + core_re + tail)


def split_final(text: str, final_marker: str) -> tuple[str, str | None]:
    """Split a segment at the final-answer marker: ``(notes part, answer part)``; the answer is None without a marker
    and ``""`` when the marker is present but nothing follows it.

    The exact marker matches anywhere; a case/markdown variant (``**Final answer:**``) only at the start of a line.
    """
    idx = text.find(final_marker)
    if idx >= 0:
        start, end = idx, idx + len(final_marker)
    else:
        m = _marker_pattern(final_marker).search(text)
        if m is None:
            return text, None
        start, end = m.start(), m.end()
    notes = re.sub(r"[\s*_#>]+$", "", text[:start])
    answer = re.sub(r"^[\s*_]+", "", text[end:]).rstrip()
    return notes, answer
