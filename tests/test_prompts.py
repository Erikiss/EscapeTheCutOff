"""Tests for cutoffmix.prompts: relay templates and segment post-processing."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cutoffmix import prompts
from cutoffmix.prompts import (
    answer_system_prompt,
    answer_user_blocks,
    clean_segment,
    relay_system_prompt,
    relay_user_blocks,
    split_final,
    strip_preamble,
    trim_overlap,
    trim_to_sentence,
)

S = "<<END_STEP>>"
FM = "FINAL ANSWER:"

DESIGN_SYSTEM = (
    "Several analysts are jointly working through the problem in the user message; their shared notes so far are "
    "included. Contribute the next intermediate result: state it and the evidence or calculation that supports it, "
    "in about 128 tokens, written so it can be appended to the notes as the next paragraph. Use the same language as "
    "the notes (or as the problem if there are no notes yet). If an earlier note is wrong, start with 'Correction:' "
    "and fix it. Prefer the most recent facts you know and say when a fact may have changed. Output only that "
    "paragraph, then <<END_STEP>>. If the notes already reach a conclusion, write 'FINAL ANSWER:' followed by the "
    "answer. Today's date: 2026-11-24."
)


def _all_prompt_text() -> str:
    parts = [relay_system_prompt(128, "2026-11-24", S, FM), answer_system_prompt("2026-11-24")]
    for blocks in (relay_user_blocks("Q?", ["a", "b"]), answer_user_blocks("Q?", []), relay_user_blocks("Q?", [])):
        parts += [b["text"] for b in blocks]
    return "\n".join(parts)


def test_relay_system_prompt_is_design_template_b():
    assert relay_system_prompt(128, "2026-11-24", S, FM) == DESIGN_SYSTEM
    no_date = relay_system_prompt(64, None, S, FM)
    assert "Today's date" not in no_date and "about 64 tokens" in no_date
    assert "Today's date: 2026-11-24." in answer_system_prompt("2026-11-24")
    assert "Today's date" not in answer_system_prompt(None)


def test_prompts_avoid_refusal_triggers():
    text = _all_prompt_text()
    assert not re.search(r"(?i)thinking|reasoning|scratchpad|chain of thought", text)
    assert not re.search(r"<[A-Za-z/][^>]*>", text.replace(S, ""))  # no XML-like tags besides the sentinel


def test_relay_user_blocks_layout():
    blocks = relay_user_blocks("  Who leads X?  ", ["First note.", "Second note.\n"])
    assert all(set(b) == {"type", "text"} and b["type"] == "text" for b in blocks)
    assert blocks[0]["text"] == prompts.HEADER_TEMPLATE.format(question="Who leads X?")
    assert [b["text"] for b in blocks[1:-1]] == ["First note.\n\n", "Second note.\n\n"]
    assert blocks[-1]["text"] == prompts.RELAY_TAIL
    empty = relay_user_blocks("Who leads X?", [])
    assert [b["text"] for b in empty[1:]] == ["(no notes yet)\n\n", prompts.RELAY_TAIL]


def test_relay_user_blocks_are_append_only():
    b2 = relay_user_blocks("Q?", ["s1", "s2"])
    b3 = relay_user_blocks("Q?", ["s1", "s2", "s3"])
    assert b3[: len(b2) - 1] == b2[:-1]  # everything but the constant tail is a prefix
    assert b3[-1] == b2[-1]


def test_answer_user_blocks_share_notes_with_relay_blocks():
    relay = relay_user_blocks("Q?", ["s1"])
    answer = answer_user_blocks("Q?", ["s1"])
    assert answer[:-1] == relay[:-1]
    assert answer[-1]["text"] == prompts.ANSWER_TAIL != prompts.RELAY_TAIL


@pytest.mark.parametrize("tail, new, expected", [
    ("We know the treaty was signed in Mar", "ch 2026, so the next step", "ch 2026, so the next step"),
    ("We know the treaty was signed in Mar", "the treaty was signed in March 2026, so", "ch 2026, so"),
    ("The PM took office in May 2026.\n\n", "  took office in May 2026. Next, the cabinet.", " Next, the cabinet."),
    ("Short tail", "tail continues", "tail continues"),  # overlap "tail" is below min_len
    ("", "anything", "anything"),
])
def test_trim_overlap(tail, new, expected):
    assert trim_overlap(tail, new) == expected


def test_trim_overlap_min_len_parameter():
    assert trim_overlap("abc def", "def ghi", min_len=3) == " ghi"


def test_trim_to_sentence():
    text = "First sentence is here. Second one, e.g. with an abbreviation, continues. U.S. data rose 3.5 percent. Tail"
    assert trim_to_sentence("  Short.  ", 100) == "Short."
    assert trim_to_sentence(text, 30) == "First sentence is here."
    assert trim_to_sentence(text, 80) == "First sentence is here. Second one, e.g. with an abbreviation, continues."
    assert trim_to_sentence(text, 103) == ("First sentence is here. Second one, e.g. with an abbreviation, continues. "
                                           "U.S. data rose 3.5 percent.")
    assert trim_to_sentence("A list:\n- one item\n- two items that go on and on", 30) == "A list:\n- one item"
    assert trim_to_sentence("no sentence end in this long text", 12) == "no sentence"
    assert trim_to_sentence("abcdefghijklmnop", 5) == "abcde"
    assert trim_to_sentence("Er sagte: „Ja.“ Danach kam mehr Text", 20) == "Er sagte: „Ja.“"
    assert trim_to_sentence("第一句。第二句很长很长很长", 8) == "第一句。"
    assert trim_to_sentence("The rate rose to 3.5. It then fell again sharply", 30) == "The rate rose to 3.5."
    assert trim_to_sentence("So the answer is no. Then more text follows here", 30) == "So the answer is no."
    assert trim_to_sentence("Ask J. Smith about it. Then more text", 25) == "Ask J. Smith about it."


@pytest.mark.parametrize("raw, expected", [
    ("Next paragraph: The PM is X.", "The PM is X."),
    ("**Continuation:** The PM is X.", "The PM is X."),
    ("Here is the next paragraph:\nThe PM is X.", "The PM is X."),
    ("Nächster Absatz: Der Kanzler ist X.", "Der Kanzler ist X."),
    ("```text\nThe PM is X.\n```", "The PM is X."),
    ('"The PM is X."', "The PM is X."),
    ("„Der Kanzler ist X.“", "Der Kanzler ist X."),
    ('```\nNext paragraph: "The PM is X."\n```', "The PM is X."),
    ("Correction: the PM is Y, not X.", "Correction: the PM is Y, not X."),
    ("FINAL ANSWER: X", "FINAL ANSWER: X"),
    ("Step 3: divide by two.", "Step 3: divide by two."),
    ('"Jaws" was released in 1975.', '"Jaws" was released in 1975.'),
    ('"Jaws" and "Alien" are films.', '"Jaws" and "Alien" are films.'),
    ('"An unbalanced opening quote.', "An unbalanced opening quote."),
])
def test_strip_preamble(raw, expected):
    assert strip_preamble(raw) == expected


def test_clean_segment_cuts_at_sentinel_and_trims_echo():
    raw = "Continuation: signed in March 2026. Then the next fact follows.<<END_STEP>> leaked text"
    text, saw = clean_segment(raw, "We know the treaty was signed in March 2026.", sentinel=S, hard_cut_chars=600)
    assert (text, saw) == ("Then the next fact follows.", True)


@pytest.mark.parametrize("raw, expected, saw", [
    ("A complete note.", "A complete note.", False),
    ("A note. Cut mid-sentinel <<END_ST", "A note. Cut mid-sentinel", False),
    ("A note.<END STEP> extra", "A note.", True),
    ("A note.<<end_step>>", "A note.", True),
])
def test_clean_segment_sentinel_variants(raw, expected, saw):
    assert clean_segment(raw, "", sentinel=S, hard_cut_chars=600) == (expected, saw)


def test_clean_segment_hard_cut_trims_to_sentence():
    raw = "One. " + "Two is a longer sentence that keeps going. " * 20
    text, saw = clean_segment(raw, "", sentinel=S, hard_cut_chars=100)
    assert not saw and len(text) <= 100 and text.endswith("going.")


def test_split_final():
    assert split_final("So it is X. FINAL ANSWER: X", FM) == ("So it is X.", "X")
    assert split_final("So it is X.\n\n**Final answer:** X", FM) == ("So it is X.", "X")
    assert split_final("So it is X.\n## Final Answer: X\nmore", FM) == ("So it is X.", "X\nmore")
    assert split_final("No marker here.", FM) == ("No marker here.", None)
    assert split_final("Before a final answer: we check more.", FM) == ("Before a final answer: we check more.", None)
    assert split_final("Notes done. FINAL ANSWER:", FM) == ("Notes done.", "")


# --------------------------------------------------------------------------------------
# Regressions from the first live runs (tests/fixtures/live_sse)
# --------------------------------------------------------------------------------------

LIVE = Path(__file__).parent / "fixtures" / "live_sse"


def _live_segment_texts() -> dict[str, str]:
    trace = json.loads((LIVE / "relay_trace_round_robin.json").read_text(encoding="utf-8"))
    return {f"{s['index']}:{s['model']}": s["text"] for s in trace["segments"]}


def _live_sse_content(name: str) -> str:
    content = []
    for line in (LIVE / f"{name}.sse").read_text(encoding="utf-8").splitlines():
        if line.startswith("data: {"):
            for choice in json.loads(line[6:]).get("choices") or []:
                content.append((choice.get("delta") or {}).get("content") or "")
    return "".join(content)


# The live segments were cut right after a date ordinal; the raw text went on with "24. November 2026 ...", and the
# next sentence end lay beyond the 600-character hard cut.
_RUNON = (" November 2026 lassen sich aus den vorliegenden Notizen weder weitere bestätigte Starts noch belastbare "
          "Missionsergebnisse ableiten, solange kein aktuelles Startregister vorliegt, das die genannten Termine "
          "bestätigt oder widerlegt")


@pytest.mark.parametrize("key, last_sentence_end", [
    ("1:gpt-6.2-sol", "ist ein angekündigter Termin kein Startnachweis."),
    ("2:gemini-4-neon", "hatten Startfenster ab dem zweiten Quartal 2026 oder später."),
])
def test_live_segments_are_not_cut_after_a_date_ordinal(key, last_sentence_end):
    committed = _live_segment_texts()[key]
    assert committed.endswith(" zum 24.")  # what the live run committed
    raw = committed + _RUNON + ". Danach folgt der nächste Schritt."
    assert len(committed + _RUNON) > 600
    expected = committed[: committed.index(last_sentence_end) + len(last_sentence_end)]
    assert trim_to_sentence(raw, 600) == expected
    text, saw = clean_segment(raw + S + " leaked", "", sentinel=S, hard_cut_chars=600)
    assert (text, saw) == (expected, True)


def test_live_sse_content_ordinals_and_years():
    sol = _live_sse_content("gpt-6.2-sol")  # "... bis Oktober 2026. Für den Stichtag 24. November 2026 genügt ..."
    cut = sol.index("Stichtag 24.") + len("Stichtag 24.") + 3
    assert trim_to_sentence(sol, cut).endswith("reichte damit bis Oktober 2026.")  # a year still ends a sentence
    neon = _live_sse_content("gemini-4-neon")  # "... am 11. Oktober 2021 an. Da die reguläre ..."
    first = "Alar Karis trat sein Amt als Präsident von Estland am 11. Oktober 2021 an."
    # no sentence end inside the limit: the whitespace fallback is unchanged
    assert trim_to_sentence(neon, neon.index(" Oktober 2021") + 4) == first[: first.index(" Oktober")]
    assert trim_to_sentence(neon, len(first) + 10) == first


@pytest.mark.parametrize("abbreviation", [
    "z. B.", "z.B.", "u. a.", "u.a.", "d. h.", "d.h.", "bzw.", "ca.", "Nr. 4", "Dr. Kaja", "Prof. Tamm", "St. Petersburg",
    "vgl.", "usw.", "e.g.", "i.e.", "vs.", "Mr. Karis", "Mrs. Kallas", "Ms. Kallas", "Jan.", "Feb.", "Okt.", "Dez.",
    "Sept.", "No. 5", "am 3. Mal", "vom 10.–12. Oktober", "am 10./11. Oktober", "(24.",
])
def test_trim_to_sentence_skips_abbreviations_and_ordinals(abbreviation):
    head = "Der erste Satz endet hier."
    text = f"{head} Danach steht {abbreviation} mitten im Satz, der weit über die Grenze hinaus weitergeht und weitergeht"
    limit = text.index(abbreviation) + len(abbreviation) + 2
    assert trim_to_sentence(text, limit) == head


@pytest.mark.parametrize("text, limit, expected", [
    ("Die Wahl fand 2026. Danach kam der Rest des langen Textes", 30, "Die Wahl fand 2026."),
    ("The vote was in 1999. After that a long tail follows here", 30, "The vote was in 1999."),
    ("So the answer is no. Then more text follows here", 30, "So the answer is no."),
    ("Rates rose 3.5. Then they fell sharply again and again", 25, "Rates rose 3.5."),
])
def test_trim_to_sentence_keeps_real_sentence_ends(text, limit, expected):
    assert trim_to_sentence(text, limit) == expected


@pytest.mark.parametrize("raw, expected", [
    ("Next result: Karis took office in 2021.", "Karis took office in 2021."),
    ("next RESULT: Karis took office in 2021.", "Karis took office in 2021."),
    ("**Next result:** Karis took office in 2021.", "Karis took office in 2021."),
    ("**Next result**: Karis took office in 2021.", "Karis took office in 2021."),
    ("Next intermediate result: Karis took office.", "Karis took office."),
    ("Intermediate result:\nKaris took office.", "Karis took office."),
    ("Nächstes Zwischenergebnis: Karis trat 2021 an.", "Karis trat 2021 an."),
    ("**Zwischenergebnis:** Karis trat 2021 an.", "Karis trat 2021 an."),
    ("Nächster Schritt: Die Amtszeit endete 2026.", "Die Amtszeit endete 2026."),
    ("__Nächster Schritt__: Die Amtszeit endete 2026.", "Die Amtszeit endete 2026."),
    ("Correction: Karis took office in October 2021.", "Correction: Karis took office in October 2021."),
    ("Korrektur: Karis trat im Oktober 2021 an.", "Korrektur: Karis trat im Oktober 2021 an."),
    ("**Korrektur:** Karis trat im Oktober 2021 an.", "**Korrektur:** Karis trat im Oktober 2021 an."),
    ("Ergebnis: Karis ist Präsident.", "Ergebnis: Karis ist Präsident."),
    ("Result: Karis is president.", "Result: Karis is president."),
    ("Next results show that Karis won.", "Next results show that Karis won."),
])
def test_strip_preamble_removes_template_echoes_only(raw, expected):
    assert strip_preamble(raw) == expected
