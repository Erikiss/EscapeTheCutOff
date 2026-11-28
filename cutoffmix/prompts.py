"""Relay prompt templates and segment post-processing (DESIGN.md §3.2 and §7).

One neutral template ("template B") is used for every model: several analysts append the next intermediate result
plus its evidence to shared notes. The wording deliberately avoids the words thinking/reasoning/scratchpad and any
XML-like tags, which Anthropic lists as ``reasoning_extraction`` refusal triggers. The explore preset's texts
(labelled-items template, lens/check/continuation directives, record status, next-move menu) follow the same rule
and also avoid the stem "reason" altogether; their labels are plain words ending with a colon.

Request bodies must stay append-only so provider prompt caches keep hitting: the system prompt is static for a run,
the user message is a constant header block, one block per committed segment, and a tail block. In the default relay
the tail is constant; in the explore preset it is the *only* block that varies per segment (:func:`relay_tail`).
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

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


# --------------------------------------------------------------------------------------
# Explore preset (DESIGN.md §7): structured notes, lens, check, continuation, record status, next-move menu
# --------------------------------------------------------------------------------------

RELAY_SYSTEM_TEMPLATE_EXPLORE = (
    "Several analysts are jointly working through the problem in the user message and keep one shared set of "
    "working notes; the notes so far are included and are the only record that is kept. Contribute the next note in "
    "about {K} tokens, written so it can be appended as the next paragraph. A note is two to four short labelled "
    "items, each one or two sentences, chosen from: 'Claim:' (a definite statement); 'Evidence:' (a fact, number, "
    "observation or source that supports or weakens a claim; name where it comes from); 'Inference:' (what follows "
    "from which claims, as 'because A and B, therefore C'); 'Principle:' (a general rule from any field that applies "
    "here; name the field and apply it to the problem); 'Alternative set aside:' (a route or answer a careful analyst "
    "would propose, and the concrete grounds on which it is set aside); 'Open question:' (what is unsettled and what "
    "would settle it); 'Next:' (the single most useful thing the next analyst should do). Every note ends with "
    "'Where we stand:' followed by one sentence giving the current best answer, a confidence from low to high, and "
    "the weakest assumption. Keep the labels exactly as given, in English, even when the notes are in another "
    "language; write everything else in the language of the notes (or of the problem if there are no notes yet). Do "
    "not restate earlier items; extend, correct or contradict them and say which. If an earlier note is wrong, start "
    "with 'Correction:' and fix it. Prefer the most recent facts you know and say when a fact may have changed. The "
    "last paragraph of the user message says whether the moderator accepts a conclusion; only then may you write "
    "'{final_marker}' followed by the answer instead of a note. Output only the note, then {sentinel}."
)
ANSWER_LENS_SENTENCE = (
    " Where the notes mark a cross-field lens as kept, use only its concrete consequence; never present a lens, kept "
    "or dropped, as evidence."
)
ANSWER_BASIS_SENTENCE = (
    " After the answer add 'Basis:' naming the claims and evidence it rests on, and 'Not resolved:' listing the open "
    "questions that remain."
)
ANSWER_SYSTEM_EXPLORE_SUFFIX = ANSWER_LENS_SENTENCE + ANSWER_BASIS_SENTENCE

RELAY_TAIL_EXPLORE = "END OF THE SHARED NOTES. Write note {k}. {record_status} {directive}"
RECORD_CLOSED_TEXT = (
    "Items still missing from the record: {missing}. The moderator does not accept a conclusion yet; do not write "
    "'{final_marker}'."
)
RECORD_CLOSED_NO_ITEMS_TEXT = "The moderator does not accept a conclusion yet; do not write '{final_marker}'."
RECORD_OPEN_TEXT = (
    "The record is complete. If the notes settle the problem, the moderator accepts a conclusion: write "
    "'{final_marker}' followed by the answer instead of a note."
)
# The checklist phrase for the 'Where we stand:' line; also the reminder appended to the record status whenever the
# previous note has no such line (skipped by the model, or removed by the client-side hard cut).
WWS_ITEM = "a 'Where we stand:' line at the end of this note"
RECORD_OPEN_WWS_TEXT = " If you write a note, end it with a 'Where we stand:' line."

LENS_TEXT = (
    "Cross-field lens for this note: the field is {field}. Choose one specific, well-established concept from this "
    "field (for example {c1}, {c2} or {c3}) whose structure resembles the current intermediate result. Give two or "
    "three short mappings stating what in our problem plays the role of what in that concept; match by function and "
    "relations, not by surface features. Then give one 'Inference:' item with a concrete consequence or testable "
    "question this suggests for the problem, marked as a working analogy rather than evidence. Begin the note with "
    "'Lens ({field}):' and, before 'Where we stand:', write one sentence that returns to the problem. Use "
    "textbook-level knowledge of the field, not recent news."
)
CHECK_TEXT = (
    "The previous note applied a cross-field lens from {field}. Begin this note by testing it in one to three "
    "sentences: which mapping holds, where the comparison breaks (name at least one real difference), and whether its "
    "suggested consequence survives. Write 'Lens kept:' or 'Lens dropped:' followed by why. If kept, use the "
    "consequence concretely in the rest of the note; if dropped, continue the main line without it."
)
CONTINUE_TEXT = (
    "The moderator set aside the conclusion in the previous note because {why}. Begin this note with the words "
    "'{cue}' and add what is missing. If the previous note's 'Where we stand:' is wrong, say so and why; you may "
    "contradict it."
)
VS_MENU_SYSTEM = "You advise a group of analysts who keep shared working notes on a problem."
VS_MENU_USER = (
    "PROBLEM: {question}\n\nSHARED NOTES SO FAR: {notes}\n\n"
    "List five candidate directions for the next note. Write each on its own line in the form "
    "'n. direction | p=0.xx', where p is your estimate of the share of analysts who would take that direction next; "
    "the five values need not sum to one. At least three directions must have p below 0.10 and draw on fields of "
    "knowledge not yet used in the notes. One line each, at most 25 words per direction, no other text."
)
VS_DIRECTION_TEXT = (
    "Direction for this note: {direction}. Develop it with the usual labelled items and return to the problem before "
    "'Where we stand:'."
)
VS_MENU_P_MAX = 0.10  # a menu line counts as a tail direction when its p is below this
VS_MENU_MIN_LINES = 3  # fewer parsed lines than this: the menu is discarded

# Canonical labels (without the colon) in the order of the template; "Lens" stands for the 'Lens (field):' prefix.
LABELS: tuple[str, ...] = ("Claim", "Evidence", "Inference", "Principle", "Alternative set aside", "Open question",
                           "Next", "Where we stand", "Lens kept", "Lens dropped", "Correction", "Lens")
_LABEL_CORE = (
    r"(?P<label>Claim|Evidence|Inference|Principle|Alternative[ \t]+set[ \t]+aside|Open[ \t]+question|Next"
    r"|Where[ \t]+we[ \t]+stand|Lens[ \t]+kept|Lens[ \t]+dropped|Correction|Lens[ \t]*\((?P<field>[^()\n]+)\))"
)
_MARK = r"(?:\*\*|__)?"  # markdown bold around the label or the whole 'Label:'
_BULLET = r"(?:[-*•][ \t]+|\d{1,2}[.)][ \t]+)?"
_LABEL_TAIL = r"[ \t]*" + _MARK + r"[ \t]*:"
# Spec form: a label at the start of a line, tolerating markdown bold and a bullet.
LABEL_RE = re.compile(r"^[ \t]*" + _BULLET + _MARK + r"[ \t]*" + _LABEL_CORE + _LABEL_TAIL, re.IGNORECASE | re.MULTILINE)
# What the parser uses: also a label that follows a sentence end inside a paragraph ("... result. Evidence: ...").
_LABEL_ANYWHERE = re.compile(
    r"(?:^[ \t]*" + _BULLET + r"|(?<=[.!?…)\]\"”’])[ \t]+)" + _MARK + r"[ \t]*" + _LABEL_CORE
    + _LABEL_TAIL,
    re.IGNORECASE | re.MULTILINE,
)
# 'Lens (field):' as the first thing in a note (markdown bold and a bullet tolerated); group "field".
LENS_PREFIX_RE = re.compile(r"\A\s*" + _BULLET + _MARK + r"[ \t]*Lens[ \t]*\((?P<field>[^()\n]+)\)" + _LABEL_TAIL,
                            re.IGNORECASE)
VS_LINE_RE = re.compile(r"^\s*(\d)[.)]\s*(.+?)\s*\|\s*p\s*=\s*(0?\.\d+|1(?:\.0+)?)\s*$", re.MULTILINE)
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def relay_system_prompt_explore(target_tokens: int, today: str | None, sentinel: str, final_marker: str) -> str:
    """The static explore system prompt (structured notes); the date sentence is omitted when ``today`` is None."""
    text = RELAY_SYSTEM_TEMPLATE_EXPLORE.format(K=target_tokens, sentinel=sentinel, final_marker=final_marker)
    return text + (DATE_SENTENCE.format(today=today) if today else "")


def answer_system_prompt_explore(today: str | None, *, answer_basis: bool = True) -> str:
    """The explore answer prompt: the default one plus the lens sentence and, with ``answer_basis``, the
    'Basis:' / 'Not resolved:' sentence; the date sentence is omitted when ``today`` is None."""
    text = ANSWER_SYSTEM_TEMPLATE + ANSWER_LENS_SENTENCE + (ANSWER_BASIS_SENTENCE if answer_basis else "")
    return text + (DATE_SENTENCE.format(today=today) if today else "")


def _join_items(items: Sequence[str]) -> str:
    items = [s.strip() for s in items if s and s.strip()]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def record_status_text(missing: Sequence[str], gate_open: bool, final_marker: str = "FINAL ANSWER:", *,
                       remind_wws: bool = False) -> str:
    """The record-status sentence(s) of the explore tail: what is still missing while the gate is closed, or the
    permission to conclude once it is open. ``missing`` is ignored when ``gate_open`` is true.

    ``remind_wws`` (the previous note has no 'Where we stand:' line, or the hard cut removed it) adds
    :data:`WWS_ITEM` to the missing items while the gate is closed and :data:`RECORD_OPEN_WWS_TEXT` once it is open,
    so the reminder is in the tail even when the gate does not (or no longer) require the line.
    """
    if gate_open:
        return RECORD_OPEN_TEXT.format(final_marker=final_marker) + (RECORD_OPEN_WWS_TEXT if remind_wws else "")
    items = [s for s in missing if s and s.strip()]
    if remind_wws and WWS_ITEM not in items:
        items.append(WWS_ITEM)
    joined = _join_items(items)
    if not joined:
        return RECORD_CLOSED_NO_ITEMS_TEXT.format(final_marker=final_marker)
    return RECORD_CLOSED_TEXT.format(missing=joined, final_marker=final_marker)


def relay_tail(k: int, record_status: str, directive: str = "") -> str:
    """The explore tail block for note ``k`` (1-based): the only block that varies between segments. ``directive`` is
    empty, :data:`LENS_TEXT`, :data:`CHECK_TEXT`, :data:`CONTINUE_TEXT` or :data:`VS_DIRECTION_TEXT` (formatted)."""
    return RELAY_TAIL_EXPLORE.format(k=k, record_status=record_status.strip(), directive=directive.strip()).rstrip()


def lens_text(field: str, concepts: Sequence[str]) -> str:
    """:data:`LENS_TEXT` for a field (display name) and its three example concepts."""
    c = list(concepts) + [""] * 3
    return LENS_TEXT.format(field=field, c1=c[0], c2=c[1], c3=c[2])


def check_text(field: str) -> str:
    """:data:`CHECK_TEXT` for the field of the previous lens note."""
    return CHECK_TEXT.format(field=field)


def continue_text(why: str, cue: str) -> str:
    """:data:`CONTINUE_TEXT` with the moderator's grounds and the opening cue (surrounding whitespace dropped)."""
    return CONTINUE_TEXT.format(why=why.strip().rstrip("."), cue=cue.strip())


def vs_menu_messages(question: str, notes: Sequence[str]) -> list[dict[str, Any]]:
    """Chat messages of the next-move menu side call (verbalized sampling; the client picks from the tail)."""
    body = "\n\n".join(s.strip() for s in notes if s.strip()) or "(no notes yet)"
    return [{"role": "system", "content": VS_MENU_SYSTEM},
            {"role": "user", "content": VS_MENU_USER.format(question=question.strip(), notes=body)}]


def vs_direction_text(direction: str) -> str:
    """:data:`VS_DIRECTION_TEXT` for the chosen menu line."""
    return VS_DIRECTION_TEXT.format(direction=direction.strip().rstrip("."))


def _article(label: str) -> str:
    return "an" if label[:1].lower() in "aeiou" else "a"


def missing_record_items(*, main_line_count: int, min_segments: int, label_counts: Mapping[str, int],
                         require_labels: Iterable[str], wws_recent: bool, pending_check: bool) -> list[str]:
    """The phrases of the record checklist that still fail (DESIGN.md §7.4), for :func:`record_status_text`.

    ``label_counts`` is ``parse_labels(all notes)["counts"]`` (canonical labels without the colon);
    ``require_labels`` the configured labels with their colon; ``wws_recent`` whether a 'Where we stand:' line appears
    in one of the last ``where_we_stand_window`` notes (its phrase is :data:`WWS_ITEM`); ``pending_check`` whether a
    lens still awaits its check. An empty list means the record is complete.
    """
    out: list[str] = []
    short = min_segments - main_line_count
    if short > 0:
        n = _NUMBER_WORDS[short] if short < len(_NUMBER_WORDS) else str(short)
        out.append(f"{n} more main-line note{'s' if short != 1 else ''}")
    for label in require_labels:
        name = label.strip().rstrip(":").strip()
        if label_counts.get(name, 0) <= 0:
            detail = " that names its source" if name.lower() == "evidence" else ""
            out.append(f"{_article(name)} '{name}:' item{detail}")
    if not wws_recent:
        out.append(WWS_ITEM)
    if pending_check:
        out.append("the test of the previous cross-field lens ('Lens kept:' or 'Lens dropped:')")
    return out


def _canonical_label(raw: str) -> str:
    text = re.sub(r"\s+", " ", raw.strip()).lower()
    if text.startswith("lens ("):
        return "Lens"
    for label in LABELS:
        if label.lower() == text:
            return label
    return raw.strip()  # unreachable for the fixed alternatives; keeps the parser total


def parse_labels(text: str) -> dict[str, Any]:
    """Find the labelled items of a note (or of all notes joined).

    Returns ``{"counts": {label: n}, "items": n, "has_wws": bool, "lens_prefix": field | None,
    "lens_verdict": "kept" | "dropped" | None, "wws_sentence": str}``: ``counts`` uses the canonical labels of
    :data:`LABELS` (``"Lens"`` for a ``'Lens (field):'`` prefix, which also fills ``lens_prefix`` when it opens the
    text); ``items`` counts every labelled item except 'Where we stand:'; ``wws_sentence`` is the text after the
    last 'Where we stand:' up to the next label or the end (whitespace collapsed, ``""`` if absent). Labels are
    matched at line starts and after a sentence end inside a paragraph, case-insensitively, tolerating markdown
    bold and bullets.
    """
    counts: dict[str, int] = {}
    verdict: str | None = None
    wws_end: int | None = None
    matches = list(_LABEL_ANYWHERE.finditer(text))
    for m in matches:
        label = _canonical_label(m.group("label"))
        counts[label] = counts.get(label, 0) + 1
        if label in ("Lens kept", "Lens dropped") and verdict is None:
            verdict = "kept" if label == "Lens kept" else "dropped"
        if label == "Where we stand":
            wws_end = m.end()
    wws_sentence = ""
    if wws_end is not None:
        after = [m.start() for m in matches if m.start() >= wws_end]
        wws_sentence = re.sub(r"\s+", " ", text[wws_end: after[0] if after else len(text)]).strip()
    prefix = LENS_PREFIX_RE.match(text)
    return {
        "counts": counts,
        "items": sum(n for label, n in counts.items() if label != "Where we stand"),
        "has_wws": "Where we stand" in counts,
        "lens_prefix": prefix.group("field").strip() if prefix else None,
        "lens_verdict": verdict,
        "wws_sentence": wws_sentence,
    }


def parse_vs_menu(text: str) -> list[tuple[str, float]]:
    """The ``(direction, p)`` lines of a next-move menu, in order; lines not in the form ``n. direction | p=0.xx``
    are skipped (markdown bold around a direction is removed)."""
    out: list[tuple[str, float]] = []
    for m in VS_LINE_RE.finditer(text):
        direction = m.group(2).strip().strip("*_").strip()
        if direction:
            out.append((direction, float(m.group(3))))
    return out


def pick_vs_direction(menu: Sequence[tuple[str, float]], rng: random.Random) -> tuple[str, float] | None:
    """The client-side pick from a parsed menu: one of the lines with ``p`` below :data:`VS_MENU_P_MAX` (if fewer than
    two, the two lowest-``p`` lines); ``None`` when fewer than :data:`VS_MENU_MIN_LINES` lines parsed (the relay then
    writes a plain note)."""
    if len(menu) < VS_MENU_MIN_LINES:
        return None
    low = [line for line in menu if line[1] < VS_MENU_P_MAX]
    if len(low) < 2:
        low = sorted(menu, key=lambda line: line[1])[:2]
    return rng.choice(low)


def word3gram_jaccard(a: str, b: str) -> float:
    """Jaccard overlap of the word-3-gram sets of two texts (lower-cased word tokens); 0.0 when either text has fewer
    than three words. Used for loop detection between the last two notes."""
    def grams(text: str) -> set[tuple[str, ...]]:
        words = [w.lower() for w in _WORD.findall(text)]
        return {tuple(words[i: i + 3]) for i in range(len(words) - 2)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


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
# Rally preset (DESIGN_rally.md): models hand one record back and forth, each adding what it knows that is newer
# --------------------------------------------------------------------------------------

RELAY_SYSTEM_TEMPLATE_RALLY = (
    "Several analysts, each trained on different material, hand one set of working notes to each other. The "
    "problem and the notes so far are in the user message. Every analyst knows some recent facts the others do "
    "not, and none knows everything: your job is to add what you know that the notes lack or get wrong, and to "
    "say honestly where you know nothing more. Write the next note in about {K} tokens, as short labelled lines "
    "(labels in English, everything else in the language of the problem): 'Known:' (first note only), 'Confirm:', "
    "'Update:', 'Newer:', 'Unverified:', 'Nothing newer:'. Write each label exactly as given, directly followed by "
    "a colon, and put dates after the colon. Every claim carries the month and year it refers to. Never invent a "
    "date, number or event to have something to add: 'Nothing newer:' is a valuable answer, followed by what you "
    "checked. Do not agree to be polite and do not repeat earlier lines. End every note with 'Newest I know:' and "
    "the latest month and year relevant to the problem that you genuinely know about (not today's date). Do not "
    "write a final answer; the moderator ends the exchange. Output only the note, then {sentinel}."
)
ANSWER_RALLY_SENTENCE = (
    " The notes are an exchange between analysts with different knowledge. State plainly what a second analyst "
    "confirmed or corrected; mark a statement that only one analyst made with '(single source)'; where they "
    "disagreed, say so and give the better supported side; name the newest date the record supports."
)
RELAY_TAIL_RALLY = "END OF THE SHARED NOTES. Write note {k}. {directive}"
RALLY_OPEN_TEXT = (
    "This is the first note. List what you know about the problem as three to six 'Known:' items, each with the "
    "date or period it refers to, and mark the one you are least sure of. Then give 'Newest I know:'."
)
RALLY_REPLY_TEXT = (
    "The previous note was written by {writer}, an analyst whose knowledge differs from yours; their newest "
    "knowledge on this problem is {newest}. Go through its dated claims. Use 'Confirm:' for what you can confirm, "
    "'Update:' for what is outdated or wrong (say what you know instead and when), 'Newer:' for developments after "
    "{newest} that you know and the notes do not yet contain, including ones the previous note reminded you of, "
    "'Unverified:' for claims you doubt but cannot refute, and 'Nothing newer:' if you know nothing newer than "
    "{newest} that the notes lack. Then 'Newest I know:'."
)
RALLY_SELF_TEXT = (
    "The other analyst could not take its turn, so the previous note is your own. Do not confirm your own claims. "
    "Add with 'Update:' or 'Newer:' only what you know and the notes do not yet contain, with its date; otherwise "
    "write 'Nothing newer:' and what you checked. Then 'Newest I know:'."
)
# The light notes (RallyConfig.light_schema_models): the default system prompt and a prose tail with few quoted markers
RALLY_LIGHT_OPEN_TEXT = (
    "List what you know about the problem, each item with its month and year, and end with a line "
    "'Newest I know: <Month YYYY>'."
)
RALLY_LIGHT_REPLY_TEXT = (
    "Another analyst wrote the notes above; the newest thing they say they know is {newest}. Add what you know that "
    "is more recent and not yet in the notes, each item with month and year, or say plainly that you know nothing "
    "more recent. Correct anything that is wrong, starting with 'Correction:'. End with a line "
    "'Newest I know: <Month YYYY>'."
)
RALLY_LIGHT_SELF_TEXT = (
    "The other analyst could not take its turn, so the notes above end with your own note. Add only what you know "
    "that is more recent and not yet in the notes, each item with month and year, or say plainly that you know "
    "nothing more recent. End with a line 'Newest I know: <Month YYYY>'."
)
RALLY_ANSWER_SELF_SENTENCE = (
    " Note {n} (counting the notes from 1 in the order given) was written by the same analyst as the note before it, "
    "because the other analyst could not take its turn: do not count its 'Confirm:' lines as confirmation by a "
    "second analyst."
)
RALLY_ANSWER_SELF_SENTENCE_MANY = (
    " Notes {n} (counting the notes from 1 in the order given) were written by the same analyst as the note before "
    "them, because the other analyst could not take its turn: do not count their 'Confirm:' lines as confirmation "
    "by a second analyst."
)
RALLY_NEWEST_UNKNOWN = "not stated"
_MONTH_NAMES = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                "November", "December")


def relay_system_prompt_rally(target_tokens: int, today: str | None, sentinel: str) -> str:
    """The static system prompt for rally think notes; the date sentence is omitted when ``today`` is None."""
    return RELAY_SYSTEM_TEMPLATE_RALLY.format(K=target_tokens, sentinel=sentinel) + (
        DATE_SENTENCE.format(today=today) if today else "")


def _ordinals(numbers: Sequence[int]) -> str:
    items = [str(n) for n in numbers]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def answer_system_prompt_rally(today: str | None, *, answer_basis: bool = True,
                               self_reply_notes: Sequence[int] = ()) -> str:
    """The system prompt for the rally's answer call (default answer prompt, the exchange sentence, a warning about
    notes the same analyst wrote twice in a row (``self_reply_notes``: their 1-based positions), optionally the
    explore preset's 'Basis:' / 'Not resolved:' sentence)."""
    warning = ((RALLY_ANSWER_SELF_SENTENCE_MANY if len(self_reply_notes) > 1 else RALLY_ANSWER_SELF_SENTENCE)
               .format(n=_ordinals(self_reply_notes)) if self_reply_notes else "")
    return (ANSWER_SYSTEM_TEMPLATE + ANSWER_RALLY_SENTENCE + warning + (ANSWER_BASIS_SENTENCE if answer_basis else "")
            + (DATE_SENTENCE.format(today=today) if today else ""))


def month_display(month: str | None, *, raw: str | None = None, approx: bool = False) -> str:
    """``'2026-03'`` -> ``'March 2026'``; None or anything else -> :data:`RALLY_NEWEST_UNKNOWN`. An approximate date
    (a bare year, a quarter, a season) quotes the writer's own words when ``raw`` is given: the parsed month would
    claim more precision, and an earlier date, than the writer gave."""
    if approx and raw and raw.strip():
        words = " ".join(raw.replace('"', "'").split())
        return '"' + (words if len(words) <= 60 else words[:57].rstrip() + "...") + '"'
    if month and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        return f"{_MONTH_NAMES[int(month[5:7]) - 1]} {month[:4]}"
    return RALLY_NEWEST_UNKNOWN


def rally_directive(turn: int, writer: str | None = None, newest: str | None = None, *, newest_raw: str | None = None,
                    approx: bool = False, self_reply: bool = False, light: bool = False) -> str:
    """The tail directive of rally note ``turn`` (1-based): the opening text for note 1, else the reply text naming the
    previous writer and the month (``YYYY-MM`` or None; ``newest_raw`` and ``approx`` say how the writer put it) of
    its newest knowledge. ``self_reply``: the writer is the author of the previous note (its partner could not take
    the turn). ``light``: the prose variant for the models in ``RallyConfig.light_schema_models`` (no writer name)."""
    if turn <= 1:
        return RALLY_LIGHT_OPEN_TEXT if light else RALLY_OPEN_TEXT
    when = month_display(newest, raw=newest_raw, approx=approx)
    if self_reply:
        return RALLY_LIGHT_SELF_TEXT if light else RALLY_SELF_TEXT
    if light:
        return RALLY_LIGHT_REPLY_TEXT.format(newest=when)
    return RALLY_REPLY_TEXT.format(writer=writer or "another analyst", newest=when)


def relay_tail_rally_light(k: int, directive: str, final_marker: str = "FINAL ANSWER:") -> str:
    """The tail of a light rally note: the note number, the one-sentence ban on a conclusion that was verified to pass
    Fable's input filter in the explore preset (DESIGN.md 7.8), and the prose directive."""
    return RELAY_TAIL_RALLY.format(
        k=k, directive=RECORD_CLOSED_NO_ITEMS_TEXT.format(final_marker=final_marker) + " " + directive.strip()).rstrip()


def relay_tail_rally(k: int, directive: str) -> str:
    """The rally tail block for note ``k`` (1-based): the only block that varies between segments."""
    return RELAY_TAIL_RALLY.format(k=k, directive=directive.strip()).rstrip()


# --------------------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------------------

_OVERLAP_WINDOW = 2000  # longest echoed overlap we look for, in characters
# How much of the notes (their end) a model may repeat before its new paragraph. Real segments are 400-600 chars,
# and models sometimes echo the last two or three notes, or all of them, before continuing.
ECHO_WINDOW_CHARS = 8000


def notes_echo_window(segments: list[str]) -> str:
    """The end of the notes (at most :data:`ECHO_WINDOW_CHARS`) that a model may repeat before its new paragraph:
    the segments joined as the request shows them (blank line between), whitespace at both ends stripped."""
    return "\n\n".join(s.strip() for s in segments)[-ECHO_WINDOW_CHARS:]


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


def complete_sentences(text: str) -> str:
    """``text`` up to its last sentence end (same rules as :func:`trim_to_sentence`), or ``""`` if it has none.

    For a paragraph that was cut off (``finish_reason`` ``length``): the dangling half sentence is dropped.
    """
    text = text.strip()
    best = 0
    for m in _SENTENCE_END.finditer(text):
        if _is_sentence_end(text, m):
            best = m.end()
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


def cut_at_sentinel(raw: str, sentinel: str) -> tuple[str, bool]:
    """``(text before the first exact or mangled sentinel, whether one was found)``; a truncated sentinel at the very
    end (a stream cut inside it) is removed too, but does not count as found."""
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
    text, saw = cut_at_sentinel(raw, sentinel)
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
