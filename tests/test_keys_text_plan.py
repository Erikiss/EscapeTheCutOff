"""Key, not value (docs/DESIGN_keys.md): masking and the leak check, cue tokens, names in answers, the plan on the
real paper data and on fixtures, and the cost estimate."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

import pytest
from keys_testlib import (
    EVENTS,
    FABLE,
    NEON,
    RESULTS,
    SOL,
    TODAY,
    fixture_data,
    paper,
    result,
)

from cutoffmix.bench.grading import _compact
from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.keys import (
    ARMS_BY_SET,
    BLANK_NOTE,
    KEY_USER_TEMPLATE,
    SETS,
    TITLE,
    WITHHELD,
    asserted_name,
    build_plan,
    cue_tokens,
    estimate_cost,
    mask_note,
    mask_title,
    offers,
    row_cost,
)
from cutoffmix.bench.keys.text import _contains_answer, _joined_to_name, _title_spans
from cutoffmix.bench.prompts import probe_messages

# ---- masking ----------------------------------------------------------------------------------------------------

PIORO = paper(1, "Maciej Pióro", subject="MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts")


@pytest.mark.parametrize("text", [
    "The first author is Maciej Pióro.",
    "The first author is Maciej Pioro.",                     # accent missing in the note
    "The first author is MACIEJ PIÓRO.",
    "The first author is Pióro, Maciej.",
    "The first author is M. Pióro.",
    "The first author is M Pioro.",                          # the dot of an initial is optional inside a name
    "The first author is Pióro M.",
])
def test_every_form_of_the_answer_is_withheld(text):
    masked, leaked = mask_note(text, PIORO)
    assert masked == "The first author is [withheld]." and not leaked


def test_surname_and_given_name_alone_are_withheld_and_the_rest_stays():
    masked, leaked = mask_note("Pióro's group (with Kamil Ciebiera) wrote it; Maciej led the work.", PIORO)
    assert masked == "[withheld]'s group (with Kamil Ciebiera) wrote it; [withheld] led the work." and not leaked
    accented = paper(2, "Andres Marafioti")
    assert mask_note("Andrés Marafioti (of Hugging Face)", accented)[0] == "[withheld] (of Hugging Face)"


def test_hyphen_and_space_variants_of_a_name():
    kim = paper(3, "Jun-sol Kim")
    for text in ("Junsol Kim wrote it.", "Jun Sol Kim wrote it.", "Jun-sol Kim wrote it.", "JUN-SOL KIM wrote it."):
        assert mask_note(text, kim) == ("[withheld] wrote it.", False), text
    gur = paper(4, "Yoav Gur-Arieh")
    assert mask_note("Yoav Gur Arieh and Mor Geva; Gur-Arieh led it.", gur)[0] == "[withheld] and Mor Geva; [withheld] led it."


def test_short_name_tokens_keep_their_case_so_pronouns_stay():
    he = paper(5, "Feng He")
    masked, leaked = mask_note("Feng He led it; he also wrote the code. He is at Tsinghua.", he)
    assert masked == "[withheld] led it; he also wrote the code. [withheld] is at Tsinghua." and not leaked
    lu = paper(6, "Wei Lu")
    assert mask_note("Wei Lu, with Luo and Lu's students.", lu)[0] == "[withheld], with Luo and [withheld]'s students."


def test_an_initial_must_be_upper_case_in_the_text():
    """The optional dot of an initial counts only for an upper-case initial (design section 5): "University's Yuan"
    is not "S. Yuan", "a Huang" is not "A. Huang"."""
    yuan = paper(40, "Shenghai Yuan")
    assert "S. Yuan" in yuan.aliases
    masked, leaked = mask_note("He is of Peking University's Yuan Group.", yuan)
    assert masked == "He is of Peking University's [withheld] Group." and leaked      # the lab's name: a leak
    assert mask_note("The first author is S. Yuan.", yuan) == ("The first author is [withheld].", False)
    assert mask_note("The first author is S Yuan.", yuan) == ("The first author is [withheld].", False)
    huang = paper(41, "Ann Huang")
    assert mask_note("She cited a Huang et al. paper.", huang)[0] == "She cited a [withheld] et al. paper."
    assert mask_note("She cited A. Huang et al.", huang)[0] == "She cited [withheld] et al."


@pytest.mark.parametrize(("name", "text", "masked"), [
    ("Yibo Wang", "The first author is Yibo Wang, with co-authors including Lei Wang and Lidong Bing.",
     "The first author is [withheld], with co-authors including Lei [withheld] and Lidong Bing."),
    ("Yinjie Wang", "The first author is Yinjie Wang (associated with Ling Yang and Mengdi Wang at Princeton).",
     "The first author is [withheld] (associated with Ling Yang and Mengdi [withheld] at Princeton)."),
    ("Nikita Koriagin", "It is Nikita Koriagin (with Yaroslav Aksenov, Nikita Balagansky, and Daniil Gavrilov).",
     "It is [withheld] (with Yaroslav Aksenov, [withheld] Balagansky, and Daniil Gavrilov)."),
    ("Xinhao Li", "Earlier papers were first-authored by researchers such as KunChang Li and Xinhao Li.",
     "Earlier papers were first-authored by researchers such as KunChang [withheld] and [withheld]."),
    ("Wei Lu", "The first author is Wei Lu of the **Lu** Lab.",
     "The first author is [withheld] of the **[withheld]** Lab."),
])
def test_a_placeholder_joined_to_a_capitalised_word_leaks(name, text, masked):
    """The masked token was part of another person's or a lab's name (design section 5): a recipient who knows that
    co-author or lab learns part of the answer, so the note leaks although no answer token is left."""
    event = paper(42, name)
    got, leaked = mask_note(text, event)
    assert got == masked and leaked and not _contains_answer(got, event)


@pytest.mark.parametrize("text", [
    "By Yibo Wang and colleagues.",                      # a capitalised function word at the start of a sentence
    "Probably Yibo Wang. The paper is from Tsinghua.",
    "I am not sure. Possibly Yibo Wang, from Tsinghua.",
    "- By Yibo Wang et al.",
    "The first author is Yibo Wang (Tsinghua).",
    "It is Meta's Yibo Wang.",                           # a possessive before the name is an affiliation
    "The first author is Yibo Wang\nThe paper appeared in 2025.",   # a line break is no join
    "The first author, I believe, is Yibo Wang; Lidong Bing is the senior author.",
])
def test_a_placeholder_next_to_no_name_does_not_leak(text):
    masked, leaked = mask_note(text, paper(43, "Yibo Wang"))
    assert WITHHELD in masked and not leaked, masked


def test_the_join_rule_reads_the_masked_text():
    assert _joined_to_name("Kamil [withheld] is the first author.")
    assert _joined_to_name("[withheld]-Gu and others")
    assert _joined_to_name("with Q. [withheld] and others")                    # an initial with its dot
    assert not _joined_to_name("The first author is [withheld] et al.")
    assert not _joined_to_name("**Answer:** [withheld]")
    assert not _joined_to_name("Tsinghua. [withheld] led it.")                 # a full stop is no join
    assert not _joined_to_name("The first author is [withheld] I believe.")    # the pronoun "I" is no name


def test_title_is_masked_whole_and_by_its_parts():
    assert mask_title("On MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts, see", PIORO) \
        == "On [title], see"
    assert mask_title("the moe mamba paper", PIORO) == "the [title] paper"
    assert mask_title("about efficient selective state space models with mixture of experts.", PIORO) \
        == "about [title]."
    assert mask_title("no title here", PIORO) == "no title here"
    short = paper(7, "Ann Bee", subject="Go: A Tiny Game")
    assert mask_title("I like Go and a tiny game", short) == "I like Go and [title]"   # 'Go' is too short alone


def test_adjacent_placeholders_collapse():
    masked, _ = mask_note("Maciej - Pióro and M. Pióro Maciej", PIORO)
    assert WITHHELD + " " + WITHHELD not in masked and masked.count(WITHHELD) <= 2


def test_a_form_the_mask_misses_is_reported_as_a_leak():
    masked, leaked = mask_note("The first author is Maciej **Pió**ro.", PIORO)   # markdown inside the name
    assert leaked and "**Pió**ro" in masked
    assert _contains_answer("The first author is [withheld].", PIORO) is False
    assert _contains_answer("The first author is Pioro.", PIORO) is True


def test_a_control_has_no_answer_to_mask():
    fake = paper(8, control=True, subject="Quantum Llamas: Teaching Alpacas to Tunnel")
    assert mask_note("The first author is Sho Maeoki.", fake) == ("The first author is Sho Maeoki.", False)


def test_asserted_name():
    event = paper(9, "Tamar Rott Shaham", rejects=("Sarah Schwettmann",))
    assert asserted_name("The first author is Sarah Schwettmann.", event) == "Sarah Schwettmann"
    assert asserted_name("My best guess is Jan Chorowski, but I can't verify it.", event) == "Jan Chorowski"
    quoted = ('I am not sure. The original "Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator" '
              'paper had Fan Bao as first author.')
    assert asserted_name(quoted, event) == "Fan Bao"
    assert asserted_name("I'm not sure and won't guess a name.", event) is None
    assert asserted_name("", event) is None


# ---- cue tokens ---------------------------------------------------------------------------------------------------

MAIA = paper(10, "Tamar Rott Shaham", subject="A Multimodal Automated Interpretability Agent")


@pytest.mark.parametrize("note", [
    BLANK_NOTE,
    'The first author of the arXiv paper "[title]" is **[withheld]**.',
    "[withheld] is the first author of *[title]*.",
    "I believe the first author is [withheld], though I'm not fully certain of this from memory, so it's worth "
    "verifying against the arXiv listing.",
    "I'm not certain, but my best recollection is that the first author is [withheld]; please verify this against "
    "the arXiv listing, as I can't confirm it.",
    'The first author of the arXiv paper *"[title]"* is **[withheld]** (though I am not 100% certain, as I cannot '
    'verify the arXiv listing directly).',                              # a percentage is no cue
    'The first author of the arXiv paper titled "[title]" (MAIA) is [withheld].',   # the paper's own acronym
    "The first author of the Multimodal Interpretability Agent paper is [withheld].",  # words of the question
])
def test_a_note_without_cue_tokens(note):
    assert cue_tokens(note, MAIA) == []


def test_cue_tokens_are_what_the_note_says_beyond_the_question():
    note = ('The first author of "[title]" (MAIA, 2024) is [withheld], of MIT CSAIL — listed as an equal contributor '
            'alongside Sarah Schwettmann.')
    assert cue_tokens(note, MAIA) == ["2024", "mit", "csail", "equal", "contributor", "sarah", "schwettmann"]
    assert cue_tokens("The first author is [withheld] (arXiv 2401.13627, CVPR 2024), with **Jinjin Gu**.", MAIA) \
        == ["2401", "13627", "cvpr", "2024", "jinjin"]                 # "Gu": under three letters
    assert cue_tokens("The first author is [withheld] of Peking University’s lab.", MAIA) == ["peking", "university"]
    assert cue_tokens("It is [withheld] (Pióro's group, about 95.5% sure).", MAIA) == ["pioro"]
    assert cue_tokens("It is [withheld] (of Google DeepMind).", paper(11, "Ann Bee",
                      subject="Mixture-of-Depths: Dynamically allocating compute")) == ["google", "deepmind"]
    supir = paper(12, "Fanghua Yu", subject="Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic "
                                            "Image Restoration In the Wild")
    assert cue_tokens('The first author of the arXiv paper "[title]" (SUPIR) is [withheld].', supir) == []
    assert cue_tokens("The first author is [withheld]; this is the SUPIR paper.", supir) == ["supir"]   # no parens
    video = paper(13, "Ann Bee", subject="Watching: A Video Deep Research Benchmark on Open Web")
    assert cue_tokens("It is [withheld] (the paper introduces the VideoDR benchmark).", video) == []
    vbvr = paper(14, "Ann Bee", subject="A Very Big Video Reasoning Suite")
    assert cue_tokens("The first author is [withheld] (VBVR).", vbvr) == []


@pytest.mark.parametrize(("subject", "affiliation"), [
    ("Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild", "MIT"),
    ("Scaling Up to Excellence: Practicing Model Scaling for Photo-Realistic Image Restoration In the Wild", "MSR"),
    ("A Multimodal Automated Interpretability Agent", "AI2"),
])
def test_an_affiliation_in_parentheses_is_no_acronym_of_the_title(subject, affiliation):
    """Its capitals occur in order among the title's initials ("MIT", "MSR") or as consecutive initials ("AI" of
    "Automated Interpretability"), but it does not start at the title's first word and has fewer than three capitals
    in a row of initials, so it is a cue (review: these keys were dropped as no_cue)."""
    assert cue_tokens(f"The first author is [withheld] ({affiliation}).", paper(15, "Ann Bee", subject=subject)) \
        == [affiliation.casefold()]


@pytest.mark.parametrize("note", [
    # Neon's real key for paper:2507.14683 ("MiroMind-M1: ..."): a time hedge, the title's lab and generic words
    "I am not completely certain, as I cannot verify the exact author list right now, but my best recollection is "
    "that the first author of the paper is **[withheld]** (with the work produced by the MiroMind team).",
    "The first author is [withheld] (I am about 90-95% sure).",           # the lower bound of a percentage range
    "The first author is [withheld] (70–80% confident).",
    "The first author is [withheld] (70 to 80 percent sure), though I currently cannot check.",
])
def test_hedges_ranges_and_generic_words_are_no_cue(note):
    event = paper(16, "Ann Bee", subject="MiroMind-M1: An Open-Source Advancement in Mathematical Reasoning")
    assert cue_tokens(note, event) == []
    assert cue_tokens(note.replace("(", "(2025, "), event) == ["2025"]   # a year is still a cue


# ---- plan on the real paper data ------------------------------------------------------------------------------------

PAIRS = [f"{a}->{b}" for a in (FABLE, SOL, NEON) for b in (FABLE, SOL, NEON) if a != b]
FIELDS = ["id", "set", "donor", "recipient", "event_id", "arm", "recognised", "donor_grade", "donor_answer", "note",
          "note_event", "today", "cues", "pointer", "related_known", "messages"]
# the dark items whose key ties [withheld] to another work, the same eight events in both Fable pairs (review V6)
POINTER_EVENTS = {"paper:2603.02138", "paper:2603.03283", "paper:2603.13398", "paper:2604.03144",
                  "paper:2605.12500", "paper:2605.29801", "paper:2606.16140", "paper:2609.17488"}


def _flat(note: str) -> str:
    """A note's letters and digits, accents folded and casefolded (an independent copy of the plan's rule, design
    section 3): notes that differ only in punctuation, markdown, quotes or whitespace are one note."""
    folded = unicodedata.normalize("NFKD", note)
    return "".join(ch for ch in folded if ch.isalnum() and not unicodedata.combining(ch)).casefold()


# the reviewer's adjacency pattern (review masking-3): a placeholder joined to a capitalised word
_JOINED = re.compile(r"[A-Z][\w'’]*[\s\-]+\[withheld\]|\[withheld\](?:['’]s)?[\s\-]+[A-Z]")


@pytest.fixture(scope="module")
def real_plan():
    events = load_events(EVENTS)
    results = read_jsonl(RESULTS)
    rows, summary = build_plan(events, results, today=TODAY)
    return events, results, rows, summary


def test_real_plan_counts(real_plan):
    _, _, rows, summary = real_plan
    pairs = summary["pairs"]
    assert list(pairs) == PAIRS
    fs, fn, ns = pairs[f"{FABLE}->{SOL}"], pairs[f"{FABLE}->{NEON}"], pairs[f"{NEON}->{SOL}"]
    assert (fs["dark_candidates"], fs["funnel"]["dark"]) == (96, {"candidates": 96, "leaks": 4, "no_cue": 13,
                                                                  "no_wrong_key": 0, "kept": 79})
    assert (fn["dark_candidates"], fn["funnel"]["dark"]) == (50, {"candidates": 50, "leaks": 2, "no_cue": 11,
                                                                  "no_wrong_key": 0, "kept": 37})
    assert (fs["dark"], fn["dark"], fs["keyed"], fn["keyed"]) == (79, 37, True, True)
    assert fn["funnel"]["both_right"] == {"candidates": 73, "leaks": 2, "no_cue": 2, "usable": 69, "sampled": 10}
    assert fs["both_right"] == fn["both_right"] == 10 and (fs["pointers"], fn["pointers"]) == (8, 9)
    # Neon answers in one sentence: its keys are the bare template, so Neon->Sol has no usable key (review V5). The
    # one key left adds a date ("(October 2025)"); the MiroMind key only hedges ("right now") and names the title's
    # own lab ("the work produced by the MiroMind team"), and the VideoChat key leaks ("KunChang [withheld]").
    assert ns["dark_candidates"] == 47 and ns["funnel"]["dark"]["kept"] == 1 and ns["keyed"] is False
    assert ns["dark"] == ns["both_right"] == 0 and ns["no_cue"] == 45 and ns["donor_wrong"] == 2
    for pair in (f"{SOL}->{FABLE}", f"{SOL}->{NEON}", f"{NEON}->{FABLE}"):
        assert pairs[pair]["dark_candidates"] == 1 and pairs[pair]["dark"] == 0 and pairs[pair]["keyed"] is False
    assert sorted((e["donor"], e["recipient"], e["dark"], e["dark_candidates"]) for e in summary["excluded"]) == sorted(
        [(SOL, FABLE, 0, 1), (SOL, NEON, 0, 1), (NEON, FABLE, 0, 1), (NEON, SOL, 1, 47)])
    assert pairs[f"{SOL}->{FABLE}"]["control"] == 4 and pairs[f"{SOL}->{NEON}"]["control"] == 4
    assert sum(p["donor_wrong"] for p in pairs.values()) == 45 and sum(p["control"] for p in pairs.values()) == 8
    assert {k: summary[k] for k in ("leaks", "no_cue", "no_wrong_key", "pointers", "no_name", "unasserted")} == {
        "leaks": 10, "no_cue": 73, "no_wrong_key": 0, "pointers": 17, "no_name": 4, "unasserted": 2}
    assert summary["calls_by_set_arm"] == {"dark": {arm: 116 for arm in ARMS_BY_SET["dark"]},
                                           "donor_wrong": {"alone": 45, "full": 45},
                                           "both_right": {"alone": 20, "key": 20, "full": 20},
                                           "control": {"alone": 8, "full": 8}}
    assert summary["calls"] == len(rows) == 116 * 5 + 45 * 2 + 20 * 3 + 8 * 2 == 746
    assert summary["calls_by_recipient"] == {SOL: 443, NEON: 275, FABLE: 28}
    assert summary["today"] == TODAY and summary["seed"] == 0


def test_real_plan_summary_counts_add_up(real_plan):
    _, _, rows, summary = real_plan
    calls = Counter(f"{r['donor']}->{r['recipient']}" for r in rows)
    for name, p in summary["pairs"].items():
        f = p["funnel"]
        dark, br = f["dark"], f["both_right"]
        assert dark["candidates"] == p["dark_candidates"] == (dark["leaks"] + dark["no_cue"] + dark["no_wrong_key"]
                                                              + dark["kept"])
        assert p["dark"] == (dark["kept"] if p["keyed"] else 0) and (p["keyed"] == (dark["kept"] >= 5))
        assert br["candidates"] == br["leaks"] + br["no_cue"] + br["usable"]
        assert p["both_right"] == br["sampled"] == min(10, br["usable"]) and (p["keyed"] or br["candidates"] == 0)
        for set_ in ("donor_wrong", "control"):
            assert f[set_]["candidates"] == f[set_]["no_name"] + f[set_]["unasserted"] + f[set_]["kept"]
            assert p[set_] == f[set_]["kept"]
        assert p["leaks"] == dark["leaks"] + br["leaks"] and p["no_cue"] == dark["no_cue"] + br["no_cue"]
        assert p["no_wrong_key"] == dark["no_wrong_key"]
        for drop in ("no_name", "unasserted"):
            assert p[drop] == f["donor_wrong"][drop] + f["control"][drop]
        assert p["calls"] == calls[name] == sum(p[s] * len(ARMS_BY_SET[s]) for s in SETS)
        assert p["pointers"] == len({r["event_id"] + r["set"] for r in rows if r["pointer"]
                                     and f"{r['donor']}->{r['recipient']}" == name})
    for drop in ("leaks", "no_cue", "no_wrong_key", "pointers", "no_name", "unasserted"):
        assert summary[drop] == sum(p[drop] for p in summary["pairs"].values())
    for entry in summary["excluded"]:
        dark = summary["pairs"][f"{entry['donor']}->{entry['recipient']}"]["funnel"]["dark"]
        assert entry["dark"] == dark["kept"] and all(entry[k] == dark[k] for k in ("leaks", "no_cue", "no_wrong_key"))


def test_real_plan_rows_are_well_formed(real_plan):
    events, _, rows, _ = real_plan
    by_id = {e.id: e for e in events}
    assert all(list(r) == FIELDS for r in rows)
    assert len({r["id"] for r in rows}) == len(rows) and all(re.fullmatch(r"[0-9a-f]{16}", r["id"]) for r in rows)
    items: dict[tuple, set] = {}
    for r in rows:
        items.setdefault((r["set"], r["donor"], r["recipient"], r["event_id"]), set()).add(r["arm"])
    assert all(arms == set(ARMS_BY_SET[key[0]]) for key, arms in items.items())
    for r in rows:
        event = by_id[r["event_id"]]
        system, user = r["messages"]
        assert system["role"] == "system" and f"Today's date: {TODAY}." in system["content"] and r["today"] == TODAY
        if r["arm"] == "alone":
            assert r["note"] is None and r["messages"] == probe_messages(event, "current_d", today=TODAY)
        else:
            assert user["content"] == KEY_USER_TEMPLATE.format(question=event.question_current, note=r["note"])
        if r["arm"] in ("key", "wrong_key"):
            assert not _contains_answer(r["note"], event), r["id"]
            assert not _title_spans(r["note"], event), r["id"]                 # neither its answer nor its title
        if r["arm"] == "wrong_key":
            other = by_id[r["note_event"]]
            assert other.id != event.id and _compact(other.expected_current) != _compact(event.expected_current)
            assert not _contains_answer(r["note"], other) and WITHHELD in r["note"]
        else:
            assert r["note_event"] is None
        if r["arm"] == "full" and r["set"] in ("dark", "both_right"):
            assert event.expected_current in r["note"] or _contains_answer(r["note"], event)
        if r["arm"] == "blank":
            assert r["note"] == BLANK_NOTE
        if r["set"] in ("dark", "both_right"):
            assert isinstance(r["cues"], list) and isinstance(r["pointer"], bool)
            assert isinstance(r["related_known"], list) and isinstance(r["recognised"], bool)
        else:
            assert r["cues"] is None and r["pointer"] is None and r["related_known"] is None
    dark = [r for r in rows if r["set"] == "dark"]
    assert all(r["donor_grade"] == "correct" and r["donor_answer"] == by_id[r["event_id"]].expected_current
               for r in dark)
    assert all(r["donor_grade"] == "incorrect" and r["donor_answer"] for r in rows if r["set"] in ("donor_wrong", "control"))


def test_real_plan_invariants(real_plan):
    """Every key carries a cue, no key is its own wrong key, no key or wrong key leaks through another name, and
    every donor answer is offered by its own full note (design sections 3 and 5)."""
    events, _, rows, _ = real_plan
    by_id = {e.id: e for e in events}
    items: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        items.setdefault((r["set"], r["donor"], r["recipient"], r["event_id"]), {})[r["arm"]] = r
    for (set_, _, _, event_id), arms in items.items():
        event = by_id[event_id]
        if set_ in ("dark", "both_right"):
            key = arms["key"]
            assert key["cues"] and key["cues"] == cue_tokens(key["note"], event)
            assert all(a["cues"] == key["cues"] and a["pointer"] == key["pointer"]
                       and a["related_known"] == key["related_known"] for a in arms.values())
            assert not _JOINED.search(key["note"].replace("*", "")) and not _joined_to_name(key["note"])
        if set_ == "dark":
            wrong = arms["wrong_key"]
            assert _flat(wrong["note"]) != _flat(arms["key"]["note"])
            assert cue_tokens(wrong["note"], by_id[wrong["note_event"]])
            assert not _JOINED.search(wrong["note"].replace("*", "")) and not _joined_to_name(wrong["note"])
        if set_ in ("donor_wrong", "control"):
            assert offers(arms["full"]["note"], arms["full"]["donor_answer"], event)
    adoption = {(r["event_id"], r["donor_answer"]) for r in rows if r["set"] == "donor_wrong"}
    assert ("paper:2610.12242", "Jianshu She") in adoption                    # 'she' is a surname homograph now
    assert not any(e == "paper:2606.30534" for e, _ in adoption)              # Orca: the donor rules the name out


def test_real_plan_leaks_pointers_and_related_known(real_plan):
    _, results, rows, _ = real_plan
    dark_events = {(r["donor"], r["recipient"]): set() for r in rows}
    for r in rows:
        if r["set"] == "dark":
            dark_events[(r["donor"], r["recipient"])].add(r["event_id"])
    # the four Fable->Sol keys that leak through a co-author's or a lab's name (review masking-3)
    leaking = {"paper:2507.12990", "paper:2601.09688", "paper:2603.04379", "paper:2603.10165"}
    assert not leaking & dark_events[(FABLE, SOL)]
    pointers = {(r["donor"], r["recipient"], r["event_id"]) for r in rows if r["set"] == "dark" and r["pointer"]}
    assert pointers == {(FABLE, rec, e) for rec in (SOL, NEON) for e in POINTER_EVENTS}
    stored = {(r["event_id"], r["model"], r["probe"]): r["grade"] for r in results}
    utonia = next(r for r in rows
                  if r["event_id"] == "paper:2603.03283" and r["recipient"] == NEON and r["arm"] == "key")
    # "the Pointcept group behind Sonata and Concerto": Concerto has the same first author (review V6)
    assert utonia["related_known"] == [{"event_id": "paper:2510.23607",
                                        "current_d": stored[("paper:2510.23607", NEON, "current_d")],
                                        "mcq": stored[("paper:2510.23607", NEON, "mcq")]}]
    assert utonia["related_known"][0]["current_d"] == "correct"
    # V6's false positive, a both_right key of Fable->Sol: "who led the paper (arXiv:2407.03320 ...)" is this paper
    itself = [r for r in rows if r["event_id"] == "paper:2407.03320" and r["arm"] == "key"]
    assert itself and all(r["set"] == "both_right" and r["pointer"] is False for r in itself)
    assert "who led the paper" in itself[0]["note"]


def test_real_plan_is_deterministic_and_seeded(real_plan):
    events, results, rows, summary = real_plan
    again, summary2 = build_plan(events, results, today=TODAY)
    assert again == rows and summary2 == summary
    other, _ = build_plan(events, results, today=TODAY, seed=1)
    wrong = {r["id"]: r["note_event"] for r in rows if r["arm"] == "wrong_key"}
    wrong1 = {r["id"]: r["note_event"] for r in other if r["arm"] == "wrong_key"}
    assert wrong.keys() == wrong1.keys() and wrong != wrong1                      # another seed draws other wrong keys


def test_real_plan_cost_estimate(real_plan):
    _, results, rows, _ = real_plan
    cost = estimate_cost(rows, results)
    assert cost["calls"] == len(rows) and set(cost["by_model"]) == {FABLE, SOL, NEON}
    assert 1.0 < cost["total_usd"] < 5.0 and cost["total_usd"] == pytest.approx(2.17, abs=0.05)
    assert math.isclose(sum(m["estimate_usd"] for m in cost["by_model"].values()), cost["total_usd"], abs_tol=1e-3)


# ---- plan on fixtures -------------------------------------------------------------------------------------------


def _stored(results: list[dict], event_id: str, model: str, grade: str, response: str) -> list[dict]:
    """The fixture rows with the ``current_d`` row of (``event_id``, ``model``) replaced."""
    out = [r for r in results if not (r["event_id"] == event_id and r["model"] == model and r["probe"] == "current_d")]
    assert len(out) == len(results) - 1
    return [*out, result(event_id, model, grade, response)]


def test_fixture_plan_sets_pairs_and_sampling():
    events, results = fixture_data()
    rows, summary = build_plan(events, results, pairs=[("a", "b"), ("b", "a")], both_right=1, min_items=3,
                               today=TODAY)
    ab = summary["pairs"]["a->b"]
    assert (ab["dark"], ab["donor_wrong"], ab["both_right"], ab["control"]) == (4, 1, 1, 1) and ab["keyed"]
    assert summary["pairs"]["b->a"]["keyed"] is False and summary["excluded"] == [
        {"donor": "b", "recipient": "a", "dark": 0, "dark_candidates": 0, "leaks": 0, "no_cue": 0, "no_wrong_key": 0}]
    assert len(rows) == 4 * 5 + 2 + 3 + 2 and {r["donor"] for r in rows} == {"a"}
    dark = [r for r in rows if r["set"] == "dark"]
    assert {r["event_id"]: r["recognised"] for r in dark} == {"paper:1": True, "paper:2": False, "paper:3": True,
                                                               "paper:4": False}
    assert all(r["cues"] and r["pointer"] is False and r["related_known"] == [] for r in dark)
    wrong = [r for r in dark if r["arm"] == "wrong_key"]
    assert all(r["note_event"] != r["event_id"] and r["note_event"] in {f"paper:{i}" for i in range(1, 7)}
               for r in wrong)
    keys = {r["event_id"]: r["note"] for r in dark if r["arm"] == "key"}
    assert all(r["note"] != keys[r["event_id"]] for r in wrong)
    assert [r["donor_answer"] for r in rows if r["set"] == "donor_wrong"] == ["Wrong Person"] * 2
    assert [r["donor_answer"] for r in rows if r["set"] == "control"] == ["Made Upname"] * 2
    picks = {build_plan(events, results, pairs=[("a", "b")], both_right=1, min_items=3, seed=s,
                        today=TODAY)[0][-3]["event_id"] for s in range(12)}
    assert picks == {"paper:5", "paper:6"}                                     # the seed decides which pair is sampled
    every, _ = build_plan(events, results, pairs=[("a", "b")], both_right=10, min_items=3, today=TODAY)
    assert sum(1 for r in every if r["set"] == "both_right") == 6            # all two candidates, three arms each
    none, summary_none = build_plan(events, results, pairs=[("a", "b")], min_items=5, today=TODAY)
    assert {r["set"] for r in none} == {"donor_wrong", "control"} and summary_none["excluded"][0]["dark"] == 4


def test_fixture_plan_excludes_a_leaking_key_and_validates_input():
    events, results = fixture_data()
    results[0] = result("paper:1", "a", "correct", "The first author is Alba **Qu**ist.")      # the mask misses it
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    assert summary["pairs"]["a->b"]["leaks"] == 1 and summary["pairs"]["a->b"]["dark"] == 3
    assert "paper:1" not in {r["event_id"] for r in rows if r["set"] == "dark"}
    assert all(r["note_event"] != "paper:1" for r in rows if r["arm"] == "wrong_key")    # nor as a wrong key
    with pytest.raises(ValueError, match="without stored results"):
        build_plan(events, results, pairs=[("a", "zzz")])
    with pytest.raises(ValueError, match="two different"):
        build_plan(events, results, pairs=[("a", "a")])
    with pytest.raises(ValueError):
        build_plan(events, results, today="27.11.2026")
    with pytest.raises(ValueError, match="duplicate models"):
        build_plan(events, results, models=["a", "b", "a"])


def test_a_repeated_pair_is_refused():
    events, results = fixture_data()
    with pytest.raises(ValueError, match=r"^duplicate pairs: a:b$"):
        build_plan(events, results, pairs=[("a", "b"), ("b", "a"), ("a", "b")], today=TODAY)


def test_a_key_joined_to_another_name_leaks_in_the_plan():
    events, results = fixture_data()
    results[1] = result("paper:2", "a", "correct", "The first author is Bodo Rinne, with Lena Rinne.")
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    assert summary["pairs"]["a->b"]["funnel"]["dark"]["leaks"] == 1
    assert "paper:2" not in {r["event_id"] for r in rows if r["set"] == "dark"}


def test_cue_free_keys_are_dropped_and_min_items_counts_what_survives():
    """A bare-sentence key is no key (``no_cue``); ``min_items`` applies to the items left after the filters, and
    the pair keeps its ``donor_wrong`` and ``control`` sets (design section 3)."""
    events, results = fixture_data()
    for i in (0, 1):                                     # papers 1 and 2: the donor answers in the bare template
        results[i] = result(f"paper:{i + 1}", "a", "correct", f"The first author is {events[i].expected_current}.")
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    ab = summary["pairs"]["a->b"]
    assert ab["dark_candidates"] == 4 and ab["funnel"]["dark"] == {"candidates": 4, "leaks": 0, "no_cue": 2,
                                                                   "no_wrong_key": 0, "kept": 2}
    assert ab["keyed"] is False and ab["dark"] == ab["both_right"] == 0
    assert summary["excluded"] == [{"donor": "a", "recipient": "b", "dark": 2, "dark_candidates": 4, "leaks": 0,
                                    "no_cue": 2, "no_wrong_key": 0}]
    assert {r["set"] for r in rows} == {"donor_wrong", "control"}
    kept, summary2 = build_plan(events, results, pairs=[("a", "b")], min_items=2, today=TODAY)
    assert summary2["pairs"]["a->b"]["dark"] == 2
    assert {r["note_event"] for r in kept if r["arm"] == "wrong_key"} <= {"paper:3", "paper:4", "paper:5", "paper:6"}


def test_a_wrong_key_is_never_the_key_itself():
    """Two donor notes that differ only in markdown, quotes, spacing or case are one note: neither serves as the
    other's wrong key; a note without a cue serves none (``no_wrong_key`` when nothing else is left)."""
    events, results = fixture_data()
    events, results = events[:2] + events[6:], [r for r in results if r["event_id"] not in
                                                 {f"paper:{i}" for i in range(3, 7)}]
    results[0] = result("paper:1", "a", "correct", 'The first author is Alba Quist, with "Zed Quillon".')
    results[1] = result("paper:2", "a", "correct", "the FIRST author is **Bodo Rinne**,   with Zed Quillon.")
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=1, today=TODAY)
    assert summary["pairs"]["a->b"]["funnel"]["dark"] == {"candidates": 2, "leaks": 0, "no_cue": 0,
                                                          "no_wrong_key": 2, "kept": 0}
    results[1] = result("paper:2", "a", "correct", "The first author is Bodo Rinne.")   # cue-free: no wrong key
    _, summary = build_plan(events, results, pairs=[("a", "b")], min_items=1, today=TODAY)
    assert summary["pairs"]["a->b"]["funnel"]["dark"] == {"candidates": 2, "leaks": 0, "no_cue": 1,
                                                          "no_wrong_key": 1, "kept": 0}


def test_a_wrong_key_never_points_at_the_same_person_in_another_spelling():
    """'Jun-sol Kim' and 'Junsol Kim' are one person (the masking and the pointer rule compare names letters-only), so
    the note about one paper is never the wrong key of the other: that control would point at the right person."""
    events = [paper(1, "Jun-sol Kim"), paper(2, "Junsol Kim", subject="Gadget Paper 2: Dense Gadgets at Scale"),
              paper(3, "Ann Bee", subject="Third Paper: Something Else Entirely")]
    results = [result("paper:1", "a", "correct", "The first author is Jun-sol Kim, with Yara Okafor."),
               result("paper:2", "a", "correct", "The first author is Junsol Kim, with Ines Brandt."),
               result("paper:3", "a", "correct", "The first author is Ann Bee, with Ugo Ferrante."),
               result("paper:1", "b", "abstain", "I don't know."), result("paper:2", "b", "correct", "Junsol Kim."),
               result("paper:3", "b", "correct", "Ann Bee.")]
    for seed in range(8):
        rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=1, both_right=0, seed=seed, today=TODAY)
        assert [r["note_event"] for r in rows if r["arm"] == "wrong_key"] == ["paper:3"]
    _, summary = build_plan(events[:2], results[:2] + results[3:5], pairs=[("a", "b")], min_items=1, both_right=0,
                            today=TODAY)
    assert summary["pairs"]["a->b"]["funnel"]["dark"] == {"candidates": 1, "leaks": 0, "no_cue": 0,
                                                          "no_wrong_key": 1, "kept": 0}


def test_pointers_are_flagged_with_the_recipients_grades_on_the_related_event():
    events, results = fixture_data()
    # paper:8 has the same first author as paper:1 and a short handle ("WidgetNet")
    events.append(paper(8, "Alba Quist", subject="WidgetNet: Dense Widgets at Scale"))
    results += [result("paper:8", "b", "incorrect", "Bo Ek."), result("paper:8", "b", "correct", "A", probe="mcq")]
    results[0] = result("paper:1", "a", "correct", "The first author is Alba Quist, of the WidgetNet team.")
    results[1] = result("paper:2", "a", "correct", "I guess Bodo Rinne, who also led the earlier Gadget paper.")
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    key = {r["event_id"]: r for r in rows if r["set"] == "dark" and r["arm"] == "key"}
    assert key["paper:1"]["pointer"] is True and key["paper:1"]["related_known"] == [
        {"event_id": "paper:8", "current_d": "incorrect", "mcq": "correct"}]
    assert key["paper:2"]["pointer"] is True and key["paper:2"]["related_known"] == []   # a relation pattern only
    assert key["paper:3"]["pointer"] is False and key["paper:3"]["related_known"] == []
    assert summary["pairs"]["a->b"]["pointers"] == 2
    assert all(r["pointer"] is key[r["event_id"]]["pointer"] for r in rows if r["set"] == "dark")


@pytest.mark.parametrize(("note", "pointer"), [
    ("My best guess is [withheld], who was first author of the earlier Gadget-1.5B paper.", True),
    ("The original Gadget paper (2024) had [withheld] as first author.", True),
    ("Previous papers in the Gadget series have been first-authored by researchers such as [withheld].", True),
    ("As a best guess based on the original *Gadget* paper, it may be [withheld].", True),
    ("I believe it is [withheld], who has served as the first author on previous Gadget reports.", True),
    ("I believe it is [withheld], who also led the related GadgetNet work.", True),
    ("I believe it is [withheld], who also led GadgetNet.", True),
    ("I believe it is [withheld], who also led the Gadget paper.", True),
    ("The first author is [withheld], who led the paper (published on arXiv in January 2024).", False),
    ("The first author is [withheld] (Tsinghua); the paper lists [withheld] first.", False),
    ("The first author is [withheld] (Tsinghua), with co-authors from related institutions.", False),
    # they speak of this paper only (review): no other work after the relation word or after "also" + verb
    ("The first author is [withheld], who also wrote the code release.", False),
    ("The first author is [withheld], who also led the project.", False),
    ("The original arXiv listing has [withheld] as the first author.", False),
    ("The original submission lists [withheld] first, with Lidong Bing as senior author.", False),
    ("Based on the original arXiv abstract page, it is [withheld].", False),
    ("I have no prior knowledge of this paper, but the listing I recall has [withheld] first.", False),
    ("The first author is [withheld], who is also the first author on the v1 preprint of this paper.", False),
    ("The paper was first-authored by [withheld] at Tsinghua.", False),
    ("The earlier version lists [withheld] first.", False),
])
def test_pointer_relation_patterns(note, pointer):
    events, results = fixture_data()
    results = _stored(results, "paper:3", "a", "correct", note.replace(WITHHELD, "Cora Sell"))
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    key = {r["event_id"]: r for r in rows if r["arm"] == "key"}["paper:3"]
    assert key["pointer"] is pointer and key["note"] == note, key["note"]


def test_a_donor_name_its_note_does_not_offer_is_unasserted():
    events, results = fixture_data()
    results = _stored(results, "paper:7", "a", "incorrect",
                      'I don\'t know. It is not the "Gadget" paper, whose first author is Wrong Person.')
    results = _stored(results, "paper:20", "a", "incorrect", "The Quantum Team wrote it.")
    rows, summary = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    ab = summary["pairs"]["a->b"]
    assert (ab["unasserted"], ab["no_name"], ab["donor_wrong"], ab["control"]) == (1, 1, 0, 0)
    assert ab["funnel"]["donor_wrong"] == {"candidates": 1, "no_name": 0, "unasserted": 1, "kept": 0}
    assert ab["funnel"]["control"] == {"candidates": 1, "no_name": 1, "unasserted": 0, "kept": 0}
    assert not {r["set"] for r in rows} & {"donor_wrong", "control"}
    results = _stored(results, "paper:7", "a", "incorrect",
                      "I can't verify it, but Wrong Person would be my best guess.")
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    assert [r["donor_answer"] for r in rows if r["set"] == "donor_wrong"] == ["Wrong Person"] * 2


def test_both_right_is_sampled_among_keys_that_pass_the_leak_and_cue_rules():
    events, results = fixture_data()
    results[4] = result("paper:5", "a", "correct", "The first author is Edda Ulm.")       # cue-free
    rows, summary = build_plan(events, results, pairs=[("a", "b")], both_right=2, min_items=3, today=TODAY)
    assert summary["pairs"]["a->b"]["funnel"]["both_right"] == {"candidates": 2, "leaks": 0, "no_cue": 1,
                                                                "usable": 1, "sampled": 1}
    both = [r for r in rows if r["set"] == "both_right"]
    assert {r["event_id"] for r in both} == {"paper:6"} and all(r["cues"] == ["omar", "saleh"] for r in both)
    assert {r["arm"] for r in both} == set(ARMS_BY_SET["both_right"])


def test_estimate_cost_scales_the_prompt_part():
    events, results = fixture_data()
    usage = {"cost": 0.01, "cost_details": {"upstream_inference_cost": 0.01, "upstream_inference_prompt_cost": 0.002}}
    byok = {"cost": 0, "is_byok": True, "cost_details": {"upstream_inference_cost": 0.004,
                                                           "upstream_inference_prompt_cost": 0.001}}
    results = [dict(r, usage=usage if r["model"] == "b" else byok) for r in results]
    rows, _ = build_plan(events, results, pairs=[("a", "b")], min_items=3, today=TODAY)
    cost = estimate_cost(rows, results)
    entry = cost["by_model"]["b"]
    assert entry["calls"] == len(rows) and entry["mean_stored_cost_usd"] == 0.01
    alone = [r for r in rows if r["arm"] == "alone"]
    keyed = [r for r in rows if r["arm"] != "alone"]
    assert estimate_cost(alone, results)["total_usd"] == pytest.approx(0.01 * len(alone), abs=1e-4)
    assert estimate_cost(keyed, results)["total_usd"] > 0.01 * len(keyed)       # the notes make the prompt longer
    assert row_cost({"usage": byok}) == 0.004 and row_cost({"usage": None}) == 0.0 and row_cost(None) == 0.0


def test_sets_and_arms_constants():
    assert SETS == ("dark", "donor_wrong", "both_right", "control")
    assert ARMS_BY_SET["dark"] == ("alone", "key", "full", "wrong_key", "blank")
    assert ARMS_BY_SET["both_right"] == ("alone", "key", "full")
    assert "{question}" in KEY_USER_TEMPLATE and "{note}" in KEY_USER_TEMPLATE and TITLE == "[title]"
