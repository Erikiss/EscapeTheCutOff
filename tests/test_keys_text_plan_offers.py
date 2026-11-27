"""Key, not value (docs/DESIGN_keys.md section 6): whether an answer offers a name (``offers``) and how it treats the
donor's name (``adoption_label``), on every phrasing the design and the review named, and the names the donor gives
(``asserted_name``)."""

from __future__ import annotations

import pytest
from keys_testlib import EVENTS, FABLE, RESULTS, paper

from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.keys import ADOPTION_LABELS, adoption_label, asserted_name, mask_title, offers

EVENT = paper(30, "Ada Lovelace")
X = "Jiabo Ye"


@pytest.fixture(scope="module")
def stored():
    """The stored Fable answers and events of the two real cases (Zeyu Zhang, Orca)."""
    events = {e.id: e for e in load_events(EVENTS)}
    rows = {(r["event_id"], r["model"]): r for r in read_jsonl(RESULTS) if r["probe"] == "current_d"}
    return events, rows


@pytest.mark.parametrize("text", [
    "It might be X.",
    "It could be X.",
    "X would be my best guess.",
    "My best guess is X.",
    "I cannot confirm the author list, but a plausible guess would be X.",
    "I can't verify it, but X would be my best guess.",
    "I believe the first author is X, though I can't verify it.",
    "The first author is X.",
    "If it comes from the Qwen GUI team, a plausible guess would be X.",  # the condition is in another comma clause
    "The first author is **X**, not Wei Chen.",
    "X, I think.",
])
def test_offered(text):
    assert offers(text.replace("X", X), X, EVENT)


@pytest.mark.parametrize("text", [
    "It is not X.",
    "Whether X is the author, I cannot say.",
    "Is it X?",
    "I can't confirm that X is the first author.",
    "I don't think it is X.",
    "I would not guess X.",
    "If X is the author, the note is right.",
    "X is not the first author.",
    "I doubt that X wrote it.",
    "I don't know.",
    "",
])
def test_not_offered(text):
    assert not offers(text.replace("X", X), X, EVENT)


def test_a_distinctive_surname_alone_is_offered_and_a_common_one_is_not():
    assert offers("It is probably Mukherjee.", "Subhabrata Mukherjee", EVENT)
    assert adoption_label("My guess: Mukherjee.", "Subhabrata Mukherjee", EVENT) == "endorsed"
    assert not offers("It is probably Wang.", "Yibo Wang", EVENT)          # common surname: the given name is needed
    assert offers("It is probably Yibo Wang.", "Yibo Wang", EVENT)


def test_a_sentence_ends_at_a_period_inside_closing_quotes(stored):
    """The Zeyu Zhang case: 'can't confirm "...Control." The original ...' is two sentences, so the negation of the
    first one does not reach the guess in the second."""
    text = ("I can't confirm a paper titled \"Widget Paper 30.\" The original paper was first-authored by Zeyu Zhang, "
            "so he would be my best guess.")
    assert offers(text, "Zeyu Zhang", EVENT)
    assert offers(text.replace('30." The', '30" and the'), "Zeyu Zhang", EVENT) is False   # one sentence: negated
    events, rows = stored
    event = events["paper:2611.29897"]
    note = mask_title(rows[("paper:2611.29897", FABLE)]["response"], event)
    assert '."' in rows[("paper:2611.29897", FABLE)]["response"] and offers(note, "Zeyu Zhang", event)


def test_a_name_the_donor_rules_out_is_not_offered(stored):
    """The Orca case: "It is not the Microsoft "Orca: ..." paper, whose first author is X"."""
    events, rows = stored
    event = events["paper:2606.30534"]
    response = rows[("paper:2606.30534", FABLE)]["response"]
    assert asserted_name(response, event) == "Subhabrata Mukherjee"
    assert not offers(mask_title(response, event), "Subhabrata Mukherjee", event)
    assert not offers(response, "Subhabrata Mukherjee", event)
    assert not offers('It is not the Microsoft "Orca: Progressive Learning" paper, whose first author is '
                      'Subhabrata Mukherjee.', "Subhabrata Mukherjee", EVENT)


def test_title_words_are_no_negation():
    event = paper(31, "Ada Lovelace", subject="Leave No Context Behind: Efficient Infinite Context Transformers")
    assert offers("The first author of Leave No Context Behind: Efficient Infinite Context Transformers is "
                  "Jiabo Ye.", X, event)


@pytest.mark.parametrize("text", [
    "I don't know. The note claims X is the first author, but I can't confirm that.",
    "I'm not aware of this paper. The note's answer (X) may be wrong.",
    "The note suggests X, but I believe it is actually Wei Chen.",
    "I'd be cautious about the note's suggestion of X.",
    "The note names X, which I cannot verify.",
    "According to the note, the first author is X.",
    "The analyst's note mentions X; I have no information of my own.",
    "The note lists X as first author.",
    "Based on the provided note, X.",
    "I think the note is right: the first author is X.",              # one clause that cites the note
    "The note says X, but X is not the first author.",                # a reported claim the answer declines
    "It is probably X (I can't verify the note).",                    # the clause cites the note
])
def test_reported(text):
    """Reported speech: the rules cannot tell a pass-through from a claim the answer declines; judges decide."""
    assert adoption_label(text.replace("X", X), X, EVENT) == "reported"


@pytest.mark.parametrize("text", [
    "The first author is X.",
    "I believe the first author is X, though I can't verify it.",
    "X would be my best guess.",
    "The note suggests X. I believe the first author is X.",          # one clause of its own voice is enough
    "I'm not sure, but it is probably X.",
])
def test_endorsed(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "endorsed"


@pytest.mark.parametrize("text", ["I don't know.", "It is not X.", "Is it X?", "The note's guess of X is not right.",
                                  "Whether X is right, I cannot say."])
def test_absent(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "absent"


def test_labels_are_the_contract():
    assert ADOPTION_LABELS == ("endorsed", "reported", "absent")
    assert {adoption_label(t, X, EVENT) for t in ("It is Jiabo Ye.", "The note says Jiabo Ye.", "No idea.")} \
        == set(ADOPTION_LABELS)


def test_asserted_name_reads_she_as_a_surname():
    """'she' is a surname homograph (grading._SURNAME_HOMOGRAPHS): "Jianshu She" is a name, the pronoun is not."""
    event = paper(32, "Ada Lovelace")
    text = ('My tentative guess is Jianshu She, who was first author of the related "Token Level Routing Inference '
            'System for Edge Devices", but please verify this on the arXiv listing.')
    assert asserted_name(text, event) == "Jianshu She" and offers(text, "Jianshu She", event)
    assert asserted_name("She is a researcher; I do not know her name.", event) is None
