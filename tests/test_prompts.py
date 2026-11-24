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


# --------------------------------------------------------------------------------------
# Explore preset (DESIGN.md §7): texts, labels, next-move menu, loop overlap
# --------------------------------------------------------------------------------------

import random

from cutoffmix.prompts import (
    LABEL_RE,
    LABELS,
    LENS_PREFIX_RE,
    answer_system_prompt_explore,
    check_text,
    continue_text,
    lens_text,
    missing_record_items,
    parse_labels,
    parse_vs_menu,
    pick_vs_direction,
    record_status_text,
    relay_system_prompt_explore,
    relay_tail,
    vs_direction_text,
    vs_menu_messages,
    word3gram_jaccard,
)

FORBIDDEN = re.compile(r"(?i)thinking|reason|scratchpad|chain of thought")  # incl. the stem "reason"
TAG = re.compile(r"<[A-Za-z/][^>]*>")


def _explore_texts() -> list[str]:
    return [
        relay_system_prompt_explore(192, "2026-11-24", S, FM),
        answer_system_prompt_explore("2026-11-24"),
        answer_system_prompt_explore(None, answer_basis=False),
        prompts.RELAY_SYSTEM_TEMPLATE_EXPLORE, prompts.ANSWER_SYSTEM_EXPLORE_SUFFIX, prompts.LENS_TEXT,
        prompts.CHECK_TEXT, prompts.CONTINUE_TEXT, prompts.VS_MENU_SYSTEM, prompts.VS_MENU_USER,
        prompts.VS_DIRECTION_TEXT, prompts.RECORD_CLOSED_TEXT, prompts.RECORD_CLOSED_NO_ITEMS_TEXT,
        prompts.RECORD_OPEN_TEXT, prompts.RECORD_OPEN_WWS_TEXT, prompts.WWS_ITEM, prompts.RELAY_TAIL_EXPLORE,
        record_status_text([], True, remind_wws=True),
        record_status_text(["one more main-line note"], False, remind_wws=True),
        lens_text("materials science", ("fatigue", "creep", "annealing")),
        check_text("materials science"),
        continue_text("the record has no 'Alternative set aside:' item yet", "Alternatively, "),
        vs_direction_text("Treat the market as a predator-prey system."),
        record_status_text(missing_record_items(main_line_count=1, min_segments=6, label_counts={},
                                                require_labels=["Evidence:", "Alternative set aside:", "Open question:"],
                                                wws_recent=False, pending_check=True), False),
        record_status_text([], True),
        relay_tail(3, record_status_text(["one more main-line note"], False), check_text("ecology")),
        *[m["content"] for m in vs_menu_messages("Q?", ["Claim: a.", "Evidence: b."])],
    ]


def test_explore_prompts_avoid_refusal_triggers_and_tags():
    for text in _explore_texts():
        assert not FORBIDDEN.search(text), text
        assert not TAG.search(text.replace(S, "")), text


def test_explore_system_prompt_matches_design_section_7():
    text = relay_system_prompt_explore(192, "2026-11-24", S, FM)
    assert text.startswith("Several analysts are jointly working through the problem in the user message and keep one "
                           "shared set of working notes; the notes so far are included and are the only record that "
                           "is kept. Contribute the next note in about 192 tokens, written so it can be appended as "
                           "the next paragraph. A note is two to four short labelled items, each one or two "
                           "sentences, chosen from: 'Claim:' (a definite statement); 'Evidence:' (a fact, number, "
                           "observation or source that supports or weakens a claim; name where it comes from); ")
    for label in ("'Claim:'", "'Evidence:'", "'Inference:'", "'Principle:'", "'Alternative set aside:'",
                  "'Open question:'", "'Next:'", "'Where we stand:'", "'Correction:'"):
        assert label in text
    assert "because A and B, therefore C" in text
    assert "Keep the labels exactly as given, in English, even when the notes are in another language" in text
    assert "Do not restate earlier items; extend, correct or contradict them and say which." in text
    assert ("The last paragraph of the user message says whether the moderator accepts a conclusion; only then may "
            "you write 'FINAL ANSWER:' followed by the answer instead of a note. Output only the note, then "
            "<<END_STEP>>. Today's date: 2026-11-24.") in text
    assert text.endswith(" Today's date: 2026-11-24.")
    no_date = relay_system_prompt_explore(128, None, "<<STOP>>", "ANSWER:")
    assert "Today's date" not in no_date and "about 128 tokens" in no_date
    assert "'ANSWER:' followed by the answer" in no_date and no_date.endswith("then <<STOP>>.")
    assert text != relay_system_prompt(192, "2026-11-24", S, FM)


def test_explore_answer_prompt():
    full = answer_system_prompt_explore("2026-11-24")
    assert full.startswith(prompts.ANSWER_SYSTEM_TEMPLATE)
    assert ("Where the notes mark a cross-field lens as kept, use only its concrete consequence; never present a "
            "lens, kept or dropped, as evidence.") in full
    assert ("After the answer add 'Basis:' naming the claims and evidence it rests on, and 'Not resolved:' listing "
            "the open questions that remain.") in full
    assert full.endswith(" Today's date: 2026-11-24.")
    assert full == prompts.ANSWER_SYSTEM_TEMPLATE + prompts.ANSWER_SYSTEM_EXPLORE_SUFFIX + " Today's date: 2026-11-24."
    plain = answer_system_prompt_explore(None, answer_basis=False)
    assert "Basis:" not in plain and "cross-field lens" in plain and "Today's date" not in plain


def test_directive_texts():
    lens = lens_text("materials science", ("fatigue", "creep", "annealing"))
    assert lens.startswith("Cross-field lens for this note: the field is materials science. ")
    assert "(for example fatigue, creep or annealing)" in lens
    assert "Begin the note with 'Lens (materials science):'" in lens
    assert lens.endswith("Use textbook-level knowledge of the field, not recent news.")
    check = check_text("materials science")
    assert check.startswith("The previous note applied a cross-field lens from materials science. ")
    assert "Write 'Lens kept:' or 'Lens dropped:' followed by why." in check
    cont = continue_text("the record has no 'Alternative set aside:' item yet.", "Alternatively, ")
    assert cont == ("The moderator set aside the conclusion in the previous note because the record has no "
                    "'Alternative set aside:' item yet. Begin this note with the words 'Alternatively,' and add what "
                    "is missing. If the previous note's 'Where we stand:' is wrong, say so and why; you may "
                    "contradict it.")
    assert vs_direction_text(" Treat the market as a predator-prey system. ") == (
        "Direction for this note: Treat the market as a predator-prey system. Develop it with the usual labelled "
        "items and return to the problem before 'Where we stand:'.")


def test_record_status_and_tail():
    closed = record_status_text(["three more main-line notes", "an 'Evidence:' item that names its source",
                                 "an 'Alternative set aside:' item"], False)
    assert closed == ("Items still missing from the record: three more main-line notes, an 'Evidence:' item that "
                      "names its source and an 'Alternative set aside:' item. The moderator does not accept a "
                      "conclusion yet; do not write 'FINAL ANSWER:'.")
    assert record_status_text(["one more main-line note"], False, "ANSWER:") == (
        "Items still missing from the record: one more main-line note. The moderator does not accept a conclusion "
        "yet; do not write 'ANSWER:'.")
    assert record_status_text([], False) == ("The moderator does not accept a conclusion yet; do not write "
                                             "'FINAL ANSWER:'.")
    assert record_status_text(["ignored"], True) == (
        "The record is complete. If the notes settle the problem, the moderator accepts a conclusion: write "
        "'FINAL ANSWER:' followed by the answer instead of a note.")
    assert relay_tail(1, record_status_text(["five more main-line notes"], False)) == (
        "END OF THE SHARED NOTES. Write note 1. Items still missing from the record: five more main-line notes. "
        "The moderator does not accept a conclusion yet; do not write 'FINAL ANSWER:'.")
    tail = relay_tail(7, record_status_text([], True), check_text("ecology"))
    assert tail.startswith("END OF THE SHARED NOTES. Write note 7. The record is complete. ")
    assert tail.endswith(check_text("ecology"))
    assert "  " not in tail and not tail.endswith(" ")


def test_record_status_reminds_of_the_where_we_stand_line():
    """``remind_wws`` (the previous note has no 'Where we stand:' line, or the cut removed it): the line joins the
    missing items while the gate is closed (once, even if the checklist lists it already) and a sentence follows the
    permission to conclude once the gate is open."""
    wws = prompts.WWS_ITEM
    assert wws == "a 'Where we stand:' line at the end of this note"
    assert record_status_text(["one more main-line note"], False, remind_wws=True) == (
        "Items still missing from the record: one more main-line note and a 'Where we stand:' line at the end of "
        "this note. The moderator does not accept a conclusion yet; do not write 'FINAL ANSWER:'.")
    assert record_status_text([], False, remind_wws=True) == (
        "Items still missing from the record: a 'Where we stand:' line at the end of this note. The moderator does "
        "not accept a conclusion yet; do not write 'FINAL ANSWER:'.")
    listed = record_status_text(["one more main-line note", wws], False, remind_wws=True)
    assert listed == record_status_text(["one more main-line note", wws], False) and listed.count(wws) == 1
    opened = record_status_text([], True, remind_wws=True)
    assert opened == record_status_text([], True) + " If you write a note, end it with a 'Where we stand:' line."
    assert opened.endswith(prompts.RECORD_OPEN_WWS_TEXT.strip())
    assert record_status_text([], True, remind_wws=False) == record_status_text([], True)
    tail = relay_tail(9, opened)
    assert "  " not in tail and tail.endswith("'Where we stand:' line.")


def test_missing_record_items():
    items = missing_record_items(main_line_count=3, min_segments=6, label_counts={"Claim": 2, "Open question": 1},
                                 require_labels=("Evidence:", "Alternative set aside:", "Open question:"),
                                 wws_recent=False, pending_check=True)
    assert items == ["three more main-line notes", "an 'Evidence:' item that names its source",
                     "an 'Alternative set aside:' item", "a 'Where we stand:' line at the end of this note",
                     "the test of the previous cross-field lens ('Lens kept:' or 'Lens dropped:')"]
    assert items[3] == prompts.WWS_ITEM
    assert missing_record_items(main_line_count=5, min_segments=6, label_counts={"Evidence": 1},
                                require_labels=["Evidence:"], wws_recent=True, pending_check=False) == [
        "one more main-line note"]
    assert missing_record_items(main_line_count=6, min_segments=6, label_counts={"Evidence": 1},
                                require_labels=["Evidence:"], wws_recent=True, pending_check=False) == []
    assert missing_record_items(main_line_count=0, min_segments=12, label_counts={}, require_labels=["Next:"],
                                wws_recent=True, pending_check=False) == ["12 more main-line notes", "a 'Next:' item"]


ALL_LABELS = ["Claim:", "Evidence:", "Inference:", "Principle:", "Alternative set aside:", "Open question:", "Next:",
              "Where we stand:", "Lens kept:", "Lens dropped:", "Correction:", "Lens (thermodynamics):",
              "Lens (materials science):"]


@pytest.mark.parametrize("label", ALL_LABELS)
def test_post_processing_keeps_every_explore_label(label):
    raw = f"{label} the mapping holds for the rate-limiting step. Where we stand: X, confidence low."
    assert strip_preamble(raw) == raw
    assert strip_preamble(f"**{label}** the mapping holds.") == f"**{label}** the mapping holds."
    text, saw = clean_segment(raw + S + " leaked", "Earlier note.", sentinel=S, hard_cut_chars=1000)
    assert (text, saw) == (raw, True)
    assert LABEL_RE.match(raw) and LABEL_RE.match(f"**{label}** x") and LABEL_RE.match(f"- {label} x")
    assert LABEL_RE.match(f"**{label[:-1]}**: x") and LABEL_RE.match(label.lower() + " x")
    canonical = "Lens" if label.startswith("Lens (") else label[:-1]
    assert canonical in LABELS
    assert parse_labels(raw)["counts"][canonical] >= 1


def test_label_regex_does_not_match_lookalikes():
    for line in ("Next steps: do X", "Claims: many", "The claim: X", "Evidence suggests: Y", "Lens: broad",
                 "Open questions: a, b", "Where we stand now: X", "Lens (unclosed: x", "Correctional: x"):
        assert not LABEL_RE.match(line), line
        assert parse_labels(line)["counts"] == {}, line


def test_parse_labels_on_a_structured_note():
    note = ("**Lens (thermodynamics):** The system is like a heat engine. Claim: X holds. Evidence: 3.5 percent "
            "(OECD 2026). Inference: because A and B, therefore C.\n- open question: what settles it?\n\n"
            "Where we stand: X, confidence medium; weakest assumption Y.")
    got = parse_labels(note)
    assert got["counts"] == {"Lens": 1, "Claim": 1, "Evidence": 1, "Inference": 1, "Open question": 1,
                             "Where we stand": 1}
    assert got["items"] == 5 and got["has_wws"] and got["lens_prefix"] == "thermodynamics"
    assert got["lens_verdict"] is None
    assert got["wws_sentence"] == "X, confidence medium; weakest assumption Y."
    assert LENS_PREFIX_RE.match(note).group("field") == "thermodynamics"
    check = ("Lens dropped: the engine has a fixed heat sink, the market has none. Alternative set aside: ignore "
             "the lens entirely; its question is still useful.\nNext: check Z.\nWhere we stand: still X. Lens kept: no")
    got = parse_labels(check)
    assert got["lens_verdict"] == "dropped" and got["counts"]["Lens dropped"] == 1 and got["counts"]["Lens kept"] == 1
    assert got["lens_prefix"] is None and got["wws_sentence"] == "still X."
    assert got["items"] == 4
    # labels only count after a sentence end or at a line start; the lens prefix only at the very start
    plain = parse_labels("The note mentions the claim: that is all; a later Lens (ecology): is not a prefix.")
    assert plain == {"counts": {}, "items": 0, "has_wws": False, "lens_prefix": None, "lens_verdict": None,
                     "wws_sentence": ""}
    assert parse_labels("")["has_wws"] is False
    # a note in another language keeps the English labels
    de = "Claim: Der Kanzler ist X. Evidence: Bundestag, Mai 2026. Where we stand: X, Konfidenz hoch."
    assert parse_labels(de)["counts"] == {"Claim": 1, "Evidence": 1, "Where we stand": 1}
    # the last 'Where we stand:' wins and its sentence stops at the next label
    two = "Where we stand: first. Claim: more. Where we stand: second one\nNext: go."
    assert parse_labels(two)["wws_sentence"] == "second one" and parse_labels(two)["counts"]["Where we stand"] == 2


def test_parse_vs_menu_and_pick():
    menu = ("Here you go:\n1. Look at supply chains | p=0.40\n2) **Thermodynamic analogy** | p=0.05\n"
            "3. Legal precedent angle | p=.08\n4. Music theory: tension and resolution | p=0.09\n"
            "5. Poll the analysts | p = 1.0\nthis line is noise | p=0.01\n6. no probability here\n")
    parsed = parse_vs_menu(menu)
    assert parsed == [("Look at supply chains", 0.4), ("Thermodynamic analogy", 0.05),
                      ("Legal precedent angle", 0.08), ("Music theory: tension and resolution", 0.09),
                      ("Poll the analysts", 1.0)]
    pick = pick_vs_direction(parsed, random.Random("7:3:vs"))
    assert pick in parsed[1:4] and pick[1] < 0.10
    assert pick == pick_vs_direction(parsed, random.Random("7:3:vs"))  # deterministic for a seed
    picks = {pick_vs_direction(parsed, random.Random(i))[0] for i in range(50)}
    assert picks == {"Thermodynamic analogy", "Legal precedent angle", "Music theory: tension and resolution"}
    # fewer than two lines below 0.10: the two lowest p are the candidates
    high = [("a", 0.5), ("b", 0.3), ("c", 0.2), ("d", 0.05)]
    assert {pick_vs_direction(high, random.Random(i))[0] for i in range(50)} == {"c", "d"}
    # fewer than three parsed lines: no menu
    assert pick_vs_direction(parsed[:2], random.Random(1)) is None
    assert parse_vs_menu("no menu at all") == [] and pick_vs_direction([], random.Random(1)) is None
    assert parse_vs_menu("1. x | p=1.5\n2. y | p=0.5") == [("y", 0.5)]  # out-of-range p is not a menu line


def test_vs_menu_messages():
    msgs = vs_menu_messages("  Who leads the lab?  ", ["Claim: A.\n", "Evidence: B."])
    assert msgs[0] == {"role": "system", "content": prompts.VS_MENU_SYSTEM}
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"].startswith("PROBLEM: Who leads the lab?\n\nSHARED NOTES SO FAR: Claim: A.\n\nEvidence: B.\n\n")
    assert "'n. direction | p=0.xx'" in msgs[1]["content"]
    assert "SHARED NOTES SO FAR: (no notes yet)" in vs_menu_messages("Q", [])[1]["content"]


def test_word3gram_jaccard():
    a = "The cat sat on the mat and looked at the dog."
    assert word3gram_jaccard(a, a) == 1.0
    assert word3gram_jaccard(a, "Completely different words appear here in this sentence.") == 0.0
    assert word3gram_jaccard("a b", "a b") == 0.0  # fewer than three words: no 3-grams
    assert word3gram_jaccard("", a) == 0.0
    assert round(word3gram_jaccard("the cat sat on the mat", "the dog sat on the mat"), 3) == 0.333
    assert word3gram_jaccard("The Cat SAT on the mat", "the cat sat on the mat") == 1.0  # case-insensitive
    near_loop = word3gram_jaccard(a, a + " Also, one more remark.")
    assert 0.5 < near_loop < 1.0
