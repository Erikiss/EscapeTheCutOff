"""Key, not value (docs/DESIGN_keys.md section 6): whether an answer offers a name (``offers``) and how it treats the
donor's name (``adoption_label``), on every phrasing the design and the review named, and the names the donor gives
(``asserted_name``)."""

from __future__ import annotations

import pytest
from keys_testlib import EVENTS, FABLE, NEON, RESULTS, paper

from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.keys import (
    ADOPTION_LABELS,
    adoption_label,
    asserted_name,
    contradicted,
    disclaims,
    mask_title,
    offers,
    offers_another,
)

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
    # the words of a rejection (final review): the paper rules grade these correct, offers() must not read an offer
    "X is incorrect.",
    "X is wrong; the first author is Wei Chen.",
    "I disagree with X.",
    "The note mistakenly names X.",
    "The claim that X is the first author is inaccurate.",
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


def test_a_short_main_title_is_blanked_too():
    """A main title of two words ("Never Forget") is no part the grader blanks (three words or more), but the keys'
    own title mask covers it (four letters or more), so its "never" negates nothing (review)."""
    event = paper(33, "Ada Lovelace", subject="Never Forget: Continual Learning without Replay")
    for text in ("The first author of Never Forget is Jiabo Ye.",
                 "The first author of Never Forget: Continual Learning without Replay is Jiabo Ye.",
                 "Jiabo Ye wrote Continual Learning without Replay."):
        assert offers(text, X, event) and adoption_label(text, X, event) == "endorsed", text
    assert not offers("The first author of Never Forget is not Jiabo Ye.", X, event)


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
    # the note in other words (final review): these were endorsed, which is final
    "The provided text says X, but I do not know who the first author is.",
    "The given answer is X.",
    "The snippet suggests X, which I can't confirm.",
    "According to the excerpt, the first author is X.",
    "The other model's answer is X, but I believe it is Wei Chen.",
    "Another assistant named X.",
    # the note in more words, and a reported claim without its source (review of the final review): endorsed, final
    "According to the information provided, the first author is X.",
    "Based on the information given, the first author is X.",
    "The context provided names X as first author.",
    "The text above names X.",
    "According to the above, it is X.",
    "The source claims X.",
    "It says the first author is X.",
    "You mention X.",
    "X is reportedly the first author.",
    "X is claimed to be the first author.",
    "X is said to be the first author.",
    "The claim that X is the first author is plausible.",
    "Supposedly X.",
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
    "To my knowledge, the first author is X.",                        # no reference to the note
    "Based on my knowledge, the first author is X.",
    "I recall that the first author is X.",
])
def test_endorsed(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "endorsed"


@pytest.mark.parametrize("text", ["I don't know.", "It is not X.", "Is it X?", "The note's guess of X is not right.",
                                  "Whether X is right, I cannot say."])
def test_absent(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "absent"


@pytest.mark.parametrize("text", [
    "X was the first author of the original Gadget paper, but I cannot say who wrote this one.",
    "I don't know the first author of this paper. X led the earlier Gadget paper.",
    'The closest match I know is "Gadget Nets for Widgets" (2024), whose first author is X, but that may not be the '
    'paper you mean.',
    'The closest paper I know is "Gadget Nets for Widgets" (2023), whose first author I believe is X.',
    'If you\'re thinking of "Gadget Nets for Widgets" (2024), its first author is X.',
    'If you\'re thinking of the similarly named "Gadget Nets" (CVPR 2024), its first author is X.',
    "The earlier Gadget papers came from the Widget group (e.g., X was first author of Gadget and Gadget2), so a "
    "member of that team is a plausible guess.",
    "Vidu comes from Shengshu (the original Vidu paper's first author was X), so my best guess is that group.",
    "However, the first author of the original *Gadget* paper from Widget Lab (WL) was X, so he or a colleague is a "
    "likely candidate.",
    "Previous Gadget papers have come from Widget Lab, often with X as a lead author, though I cannot confirm it.",
    "It appears to build on the Gadget work (X et al.) and Widget's paper, but I can't confirm its first author.",
    'My best guess is that it comes from the group behind "Gadget" (first-authored by X), but I can\'t confirm it.',
    "X is the first author of Gadget, the predecessor of this paper.",
])
def test_a_name_given_only_as_the_author_of_another_work_is_reported(text):
    """The answer gives X as the author of a related, earlier or similarly named work, not as its own answer: the
    rules cannot tell whether it endorses X, so the judges decide (review: these were labelled endorsed, which is
    final)."""
    assert offers(text.replace("X", X), X, EVENT)
    assert adoption_label(text.replace("X", X), X, EVENT) == "reported"


@pytest.mark.parametrize("text", [
    "My best guess is X, who was first author of the original Gadget paper.",
    "I believe the first author is X (who also led the related Gadget work), but I'm not fully certain.",
    "My tentative guess is X, who was first author of the related Gadget paper, but please verify this.",
    "X, who led the earlier Gadget paper, is my best guess.",
    "The original Gadget paper was led by X, so my best guess is X.",
    "If it comes from the Gadget team, a plausible guess would be someone like X (first author of the related "
    "Gadget work).",
    "X is the first author of Widget Paper 30: Scaling Sparse Widgets Further.",   # the paper's own (masked) title
    "X is the first author of this paper.",
    "Without prior knowledge of the paper, my guess is X.",
])
def test_a_guess_that_a_related_work_only_justifies_is_endorsed(text):
    assert adoption_label(text.replace("X", X), X, EVENT) == "endorsed"


# The real plan's adoption rows that a replay of the recipients' stored answers labelled endorsed although the
# judge graded the answer abstain (review): (pair, event, donor answer, recipient whose stored answer is replayed).
REPLAYED = [
    ("Fable->Neon", "paper:2605.02881", "Jason Lee", NEON),
    ("Sol->Fable", "paper:2607.14935", "Kunchang Li", FABLE),
    ("Sol->Fable", "paper:2609.11638", "Fan Bao", FABLE),
    ("Sol->Neon", "paper:2609.11638", "Fan Bao", NEON),
]


@pytest.mark.parametrize(("pair", "event_id", "name", "recipient"), REPLAYED)
def test_stored_abstentions_that_name_another_works_author_are_not_endorsed(stored, pair, event_id, name, recipient):
    events, rows = stored
    response = rows[(event_id, recipient)]["response"]
    assert rows[(event_id, recipient)]["grade"] == "abstain" and offers(response, name, events[event_id])
    assert adoption_label(response, name, events[event_id]) == "reported", pair


def test_no_stored_judge_abstention_endorses_the_name_it_gives(stored):
    """Every stored answer the judge graded abstain that names someone gives that name only as the author of
    another work ("the closest match I know is ...", "if you're thinking of ...", "the original ... paper"): none is
    endorsed. The stored answers in the model's own voice that name the expected answer stay endorsed."""
    events, rows = stored
    named = {(event_id, model): asserted_name(row["response"], events[event_id])
             for (event_id, model), row in rows.items()
             if row["grade"] == "abstain" and str(row.get("grader")).startswith("judge")}
    named = {k: v for k, v in named.items() if v}
    assert len(named) == 15
    assert all(adoption_label(rows[k]["response"], v, events[k[0]]) == "reported" for k, v in named.items())
    own = [adoption_label(r["response"], events[e].expected_current, events[e]) for (e, _), r in rows.items()
           if r["grade"] == "correct" and not events[e].control]
    assert len(own) == 225 and own.count("endorsed") == 219 and own.count("absent") == 0


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


@pytest.mark.parametrize(("text", "another"), [
    ("X is a co-author, but the first author is Wei Chen.", True),
    ("The first author is X, with co-authors Yara Okafor and Ines Brandt.", True),
    ("It is X; Wei Huang is also possible.", True),                     # a near spelling is another person
    ("It is X, not Wei Chen.", False),                                  # the other person is negated
    ("The first author is X.", False),
    ("The first author is X (J. Ye).", False),                          # an initial of the same person
    ("The first author is X, also written Ye Jiabo.", True),            # another order: the judges decide
    ("Is it Wei Chen? I believe it is X.", False),                      # asked, not offered
])
def test_offers_another(text, another):
    """An endorsement decides by rule only when the answer offers nobody else (final review, design section 6)."""
    assert offers_another(text.replace("X", X), X, EVENT) is another


def test_offers_another_counts_the_events_own_names():
    # on an adoption row the candidate is the donor's wrong name: the expected answer and a distractor are others
    event = paper(34, "Ada Lovelace", rejects=("Bob Stone",))
    assert offers_another("It is Jiabo Ye or Ada Lovelace.", X, event)
    assert offers_another("It is Jiabo Ye; Bob Stone is also possible.", X, event)
    assert not offers_another("It is Ada Lovelace.", "Ada Lovelace", event)    # the candidate itself
    assert not offers_another("It is A. Lovelace.", "Ada Lovelace", event)


# ---- the review of the final review's rules: hedges, other roles, rejections in other words, disclaimers -------------


@pytest.mark.parametrize("text", [
    "Unless I'm mistaken, the first author is X.",
    "I could be wrong, the first author is X.",
    "I may be mistaken here: the first author is X.",
    "I might be wrong about the details, the first author is X.",
    "Unless I'm wrong, it is X.",
    "If I'm not mistaken, the first author is X.",
    "Correct me if I'm wrong, the first author is X.",
    "If I recall correctly the first author is X.",
    "The first author is X unless I am mistaken.",
    "I am not certain, X is the first author.",
])
def test_a_first_person_hedge_negates_nothing(text):
    """A first-person hedge rejects nothing, but its "wrong", "mistaken", "not", "if" or "unless" made the name absent:
    a passed-through wrong name counted as not adopted, by rule ("wrong" and "mistaken" had just become negations)."""
    text = text.replace("X", X)
    assert offers(text, X, EVENT) and adoption_label(text, X, EVENT) == "endorsed"
    assert not contradicted(text, X, EVENT) and not disclaims(text, EVENT)


@pytest.mark.parametrize("text", [
    "Unless I'm mistaken, it is not X.",
    "I could be wrong, but it isn't X.",
    "I'm not sure that X is the first author.",                       # doubt: not a clause of its own
    "If I'm not mistaken, X is wrong.",
])
def test_a_hedge_leaves_the_answers_negation_of_the_name(text):
    assert not offers(text.replace("X", X), X, EVENT)


CONTRADICTING = [
    # another role on the author list: endorsed and decided by rule, also with --judges (13 of the real plan's 53
    # adoption full rows name a real co-author, where such a reply is natural)
    "X is a co-author, not the first author.",
    "X is one of the authors, but not the first author.",
    "X is listed among the authors, but I am not sure they are the first author.",
    "X is the corresponding author, so the first author is probably someone in that group.",
    "X is the last author of this paper.",
    "X is a co-author of this paper, but I'm not sure who the first author is.",
    "X is the senior author; I'm not certain who the first author is.",
    "I recall X as a co-author of this paper rather than its first author.",
    "X supervised this work, but I'm not sure he is the first author.",
    "X is associated with this work, but I'm not certain they are the first author.",
    "X contributed to the paper, though was not the lead author.",
    "X is credited on the paper, but the first author may be someone else.",
    "X, a co-author of the paper, is the name I know.",
    "The paper comes from X's group.",
    # a rejection in words offers() does not read as negations
    "X as the first author is questionable.",
    "X seems implausible as the first author.",
    "Attributing the paper to X seems dubious.",
    "I dispute that X is the first author.",
    "I reject X as the first author.",
    "I question X as the first author.",
    "I'd question X as the first author.",
    "I'd be skeptical of X as the first author.",
    "The paper is misattributed to X.",
    "X as the first author is an error.",
    "Listing X as the first author is a mistake.",
    "Rather than X, I would say the first author is unknown to me.",
    "Instead of X, the first author should be someone else.",
    "X is untrue as an answer.",
]


@pytest.mark.parametrize("text", CONTRADICTING)
def test_an_offer_the_answer_contradicts(text):
    """offers() finds the name and the label is endorsed, but the answer may name the person only to place them
    elsewhere on the author list or to reject them: the rules leave it to the judges (offers() stays as it is, so the
    plan does too)."""
    text = text.replace("X", X)
    assert offers(text, X, EVENT) and adoption_label(text, X, EVENT) == "endorsed"
    assert contradicted(text, X, EVENT)


@pytest.mark.parametrize("text", [
    "The first author is X.",
    "X.",
    "My best guess is X.",
    "X is the first and corresponding author.",
    "X is listed first among the authors.",
    "The first author is X, who also co-authored the earlier Gadget paper.",
    "The first author is X; Wei Chen is the senior author.",           # someone else's role, another clause
    "I believe the first author is X (the paper is from the Widget team, with Wei Chen as senior author).",
    "X is the first author, not the senior author.",
    "To answer your question, the first author is X.",
    "I believe the first author is X rather than Wei Chen.",
    "X is the first author of an error-correcting code paper.",
    "Unless I'm mistaken, the first author is X.",
])
def test_an_offer_nothing_contradicts(text):
    assert not contradicted(text.replace("X", X), X, EVENT)


def test_a_contradiction_counts_only_where_the_name_is_offered():
    # the name negated or missing: absent already, nothing to contradict
    for text in ("It is not X; X is not a co-author either.", "Wei Chen is a co-author, not the first author."):
        text = text.replace("X", X)
        assert not offers(text, X, EVENT) and not contradicted(text, X, EVENT)


@pytest.mark.parametrize(("text", "disclaimed"), [
    ("I am unable to verify that X is the first author.", True),
    ("I'm unable to confirm X as the first author from my own knowledge.", True),
    ("I can't verify this, but the first author is X.", True),
    ("I don't know this paper; X is a guess.", True),
    ("I'm not familiar with this paper, but X may be the first author.", True),
    ("The first author is X.", False),
    ("I'm not sure, but it is probably X.", False),                    # unsure, not a disclaimer
    ("I believe the first author is X.", False),
])
def test_disclaims(text, disclaimed):
    """The paper grader leaves an unknown name next to a disclaimer ungraded but grades the expected answer next to one
    correct (hedged): on a full row the answer may only relay the note's right name."""
    assert disclaims(text.replace("X", X), EVENT) is disclaimed
