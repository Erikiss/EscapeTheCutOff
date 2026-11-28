"""Key, not value (docs/DESIGN_keys.md sections 6, 7 and 10): the check of 8739803, another person's name.

``text.names_answer`` blanks the names of other people before it looks for the expected answer X, and the open offers
(``offers_expected``) read what is left. At 8739803 ``text._another_person`` took two kinds of names that write X for
another person's, by ``_same_person``'s rule that a near spelling is another person: a title or a group word the paper
grader keeps inside a name ("Professor Guo", "Doctor Guo", "the Guo Group"), and a near spelling that
``grading.grade_paper`` itself leaves to the judges as a possible misspelling ("Daya Guoo", "Matthew Kowall", "Day
Guo"). A lost grading vote on such an answer, or without judges a near spelling the rules leave open, counted as a miss
with no check of the open offers. Now both write X, on the 107 events of the real plan with an expected answer and in
every template; "Yutong Mei", "Zeyu Hu", "Zichen Liu", "Zhaoyang Li", "KunChang Li" and every distractor stay another
person, and the check's scenarios read not testable, through the real run_plan and build_report.

The review of 4fd7637 (last section): any other word the grader keeps in a name still made it another person's (a word
before a listed title, a title off the list, a word that opens the sentence), and a near spelling of a form with an
initial ("Y. Zhang") took a different short given name for X ("Yu Zhang" for Yujia Zhang). Now a title drops the words
before it, the list holds more titles, an opening word is no name, and such a form counts only for a name without a
given name; the limits section 6 states (a word on none of the lists, a slip that is no near spelling) are pinned too.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import string
from types import SimpleNamespace

import pytest
from keys_testlib import EVENTS, RESULTS, TODAY, paper
from test_keys_named_offers import (
    CORRECT,
    DAYA,
    RECIPIENTS,
    UNDECIDED,
    Y,
    answer_of,
    parts,
    run_real,
    verdicts,
)
from test_keys_run import ScriptedBackend

from cutoffmix.backends.base import BackendError
from cutoffmix.bench.grading import (
    _NAME_STOP,
    _NEAR_NAME_RATIO,
    _compact,
    _near_name,
    _unknown_names,
    grade_rule,
)
from cutoffmix.bench.io import load_events, load_results, read_jsonl, write_jsonl
from cutoffmix.bench.keys import build_plan, build_report, run_plan
from cutoffmix.bench.keys import run as run_mod
from cutoffmix.bench.keys import text as text_mod

# every title, honorific and group word of the check (as an answer writes it), the stop words of the paper grader
# among them ("Dr.", "Prof.", "Lab", "Team", ...: it drops those at either end of a name itself)
WORDS = ["Professor", "Prof", "Prof.", "Doctor", "Dr", "Dr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Miss", "Mx",
         "Mx.", "Sir", "Dame", "Lady", "Lord", "Madam", "Group", "Lab", "Labs", "Laboratory", "Team", "Et", "Al",
         "Colleagues", "Co-Authors", "Coworkers", "Co-Workers", "Herr", "Frau", "Doktor", "Professorin", "Gruppe",
         "Arbeitsgruppe"]
# the words the paper grader drops at either end of a name itself (grading._NAME_STOP): they never reach
# text._another_person; the others it keeps inside the name
DROPPED = [w for w in WORDS if re.sub(r"[^a-z]", "", w.lower()) in _NAME_STOP]
KEPT = [w for w in WORDS if w not in DROPPED]
TEMPLATES = ["The first author is {N}.", "I think it is {N}, but I'm not sure.", "It might be {N}.", "Is it {N}?",
             "I'm not sure. {N}?", "It is not {Y}, it is {N}.", "It is not {Y}; the first author is {N}.",
             "Dr. {N}, with {Y} as senior author.", "The first author is {N}, with {Y} as senior author.",
             "My best guess is {N} (Tsinghua University).", "**{N}** et al., I think.",
             "Unless I'm mistaken, the first author is {N}.", "Der Erstautor ist {N}."]
# the stored answers' mentions inside another person's name (sections 6 and 9), which must stay so
OTHERS = [("paper:2507.13334", "Yutong Mei"), ("paper:2512.13564", "Zeyu Hu"), ("paper:2601.05242", "Zichen Liu"),
          ("paper:2507.03724", "Zhaoyang Li"), ("paper:2607.14935", "KunChang Li"), ("paper:2607.14935", "Kunchang Li")]


@pytest.fixture(scope="module")
def real():
    events = load_events(EVENTS)
    plan, _ = build_plan(events, load_results(RESULTS), today=TODAY)
    named = [e for e in (e for e in events if e.id in {p["event_id"] for p in plan})
             if not e.control and (e.expected_current or "").strip()]
    return SimpleNamespace(events={e.id: e for e in events}, plan=plan, named=sorted(named, key=lambda e: e.id))


def named(text: str, event) -> bool:
    """``names_answer``; where it holds, the run's ``offers_expected`` holds too (it is ``offers_answer`` or it)."""
    return text_mod.names_answer(text, event)


def slips(name: str) -> list[str]:
    """Every name a slip away from ``name``: a letter (a-z) added, dropped or replaced (a capital by a capital), or two
    neighbouring letters swapped."""
    out = set()
    for i, ch in enumerate(name):
        if ch.isalpha():
            out.add(name[:i] + name[i + 1:])
            out.update(name[:i] + (c.upper() if ch.isupper() else c) + name[i + 1:] for c in string.ascii_lowercase)
            if i + 1 < len(name) and name[i + 1].isalpha():
                out.add(name[:i] + name[i + 1] + ch + name[i + 2:])
    for i in range(len(name) + 1):
        if (name[i - 1: i] + name[i: i + 1]).strip(" -.,"):
            out.update(name[:i] + c + name[i:] for c in string.ascii_lowercase)
    out.discard(name)
    return sorted(out)


def ratio(a: str, b: str) -> float:
    """The paper grader's letters-only similarity of two names (``grading._near_name``)."""
    return difflib.SequenceMatcher(None, _compact(a), _compact(b)).ratio()


def misspelt(event) -> str:
    """X with the last letter of its surname doubled: a near spelling the paper rules leave to the judges."""
    _, s, g = parts(event)
    return f"{g} {s}{s[-1]}"


# ---- a title, an honorific or a group word -----------------------------------------------------------------------


def test_the_words_the_paper_grader_drops_itself_keep_naming_the_expected_answer(real):
    assert len(DROPPED) == 18 and len(KEPT) == 19
    for i, event in enumerate(real.named):
        x, s, _ = parts(event)
        word = DROPPED[i % len(DROPPED)]
        form = [f"{word} {s}", f"{s} {word}", f"{word} {x}", f"{x} {word}"][i % 4]
        text = TEMPLATES[i % len(TEMPLATES)].format(N=form, Y=Y)
        assert named(text, event), (event.id, text)


@pytest.mark.parametrize("word", KEPT)
def test_a_title_or_a_group_word_around_the_name_writes_the_expected_answer(real, word):
    # before and after the surname: every name the paper grader cannot place in such an answer writes X; and the
    # answer names X, also with the full name (one of the four forms per event, in one template, both taken in turn)
    assert len(real.named) == 107
    for i, event in enumerate(real.named):
        x, s, _ = parts(event)
        forms = [f"{word} {s}", f"{s} {word}", f"{word} {x}", f"{x} {word}"]
        for form in forms[:2]:
            text = f"The first author is {form}."
            assert not any(text_mod._another_person(u, event) for u in _unknown_names(text, event)), (event.id, form)
        text = TEMPLATES[(i + KEPT.index(word)) % len(TEMPLATES)].format(N=forms[i % len(forms)], Y=Y)
        assert named(text, event), (event.id, text)
        if i % 10 == 0:
            assert run_mod._offers_expected(event, text) is True, (event.id, text)


@pytest.mark.parametrize("template", TEMPLATES)
def test_every_template_reads_a_title_a_group_word_and_a_near_spelling(real, template):
    for i, event in enumerate(real.named):
        x, s, _ = parts(event)
        form = [f"Professor {s}", f"the {s} Group", f"Doctor {x}", f"{s}'s Group", misspelt(event)][i % 5]
        text = template.format(N=form, Y=Y)
        assert named(text, event), (event.id, text)


def test_the_check_of_8739803_reads_its_forms(real):
    daya, kowal = real.events[DAYA], real.events["paper:2401.10831"]
    for text in ("The first author is Professor Guo.", "It is not Sanjay Ramaswamy, it is Professor Guo.",
                 "I believe it is from the Guo Group.", "The first author is Doctor Guo.", "Daya Guoo.",
                 "It is not Sanjay Ramaswamy, it is Daya Guoo.", "The first author is Day Guo.",
                 "It came out of Professor Guo's Group.", "Herr Professor Guo, glaube ich."):
        assert named(text, daya), text
    assert named("The first author is Matthew Kowall.", kowal)
    # 8739803's reading: these names were another person's, so these answers named nobody
    assert text_mod._another_person("Professor Guo", daya) is False and text_mod._another_person("Daya Guoo",
                                                                                                daya) is False


# ---- a near spelling the paper rules leave to the judges ----------------------------------------------------------


def test_the_slips_the_paper_grader_takes_for_a_possible_misspelling_write_the_expected_answer(real):
    # of the slips of X that grading._near_name accepts and that the grader reads as a name it cannot place, every
    # tenth one its similarity accepts and every one that only its other tests accept (on a short name one letter
    # falls under the similarity: "Ziyo Ma" for Ziyu Ma); one in twenty-five of them through names_answer, in the
    # templates in turn
    seen, initials = 0, 0
    for event in real.named:
        x = event.expected_current
        for k, slip in enumerate(e for e in slips(x) if len(e.split()) >= 2):
            similar = ratio(slip, x) >= _NEAR_NAME_RATIO                  # the similarity: a possible misspelling
            if (similar and k % 10) or not (similar or _near_name(slip, x)):
                continue
            initials += not similar
            if not _unknown_names(f"The first author is {slip}.", event):
                continue
            seen += 1
            assert not text_mod._another_person(slip, event), (event.id, slip)
            if seen % 25 == 0:
                text = TEMPLATES[seen // 25 % len(TEMPLATES)].format(N=slip, Y=Y)
                assert named(text, event), (event.id, text)
    assert seen > 4000 and initials > 300


def test_a_near_spelling_of_an_accepted_form_writes_the_expected_answer(real):
    # surname first ("Mei Lingrul" for Lingrui Mei) and the same given name with four surname letters
    for event in real.named:
        _, s, g = parts(event)
        first = f"{s} {g}"
        if first in event.aliases:
            slip = first[:-1] + ("x" if first[-1] != "x" else "y")
            if _near_name(slip, first) and _unknown_names(f"It is {slip}.", event):
                assert named(f"It is not {Y}, it is {slip}.", event), (event.id, slip)
    kowal = real.events["paper:2401.10831"]           # only the test of the given name and four surname letters
    assert ratio("Matthew Kowalewski", "Matthew Kowal") < _NEAR_NAME_RATIO and _near_name("Matthew Kowalewski",
                                                                                           "Matthew Kowal")
    assert not text_mod._another_person("Matthew Kowalewski", kowal)
    assert named("I think it is Matthew Kowalewski.", kowal)


def test_a_different_given_name_of_the_same_surname_stays_another_person(real):
    # grading._near_name's test of initials reads any given name as initials, so it accepts "Zhaoyang Li" for Zhiyu Li
    # (the paper rules leave that stored answer open); more than a slip away it stays another person
    zhiyu = real.events["paper:2507.03724"]
    assert _near_name("Zhaoyang Li", "Zhiyu Li") and _near_name("Zhaoyang Li", "Z. Li")
    assert grade_rule(zhiyu, "current_d", "The first author is Zhaoyang Li.") is None
    assert text_mod._another_person("Zhaoyang Li", zhiyu) and not named("The first author is Zhaoyang Li.", zhiyu)
    assert not text_mod._another_person("Zhiyo Li", zhiyu) and named("The first author is Zhiyo Li.", zhiyu)
    ziyu = next(e for e in real.named if e.expected_current == "Ziyu Ma")
    assert ratio("Ziyo Ma", "Ziyu Ma") < _NEAR_NAME_RATIO and named("The first author is Ziyo Ma.", ziyu)
    assert not named("The first author is Zhaoyu Ma.", ziyu)                     # two slips: another person


# ---- another person -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("event_id, other", OTHERS)
def test_the_stored_answers_other_people_stay_another_person(real, event_id, other):
    event = real.events[event_id]
    assert text_mod._another_person(other, event)
    for template in TEMPLATES:
        text = template.format(N=other, Y=Y)
        assert text_mod.mentions_answer(text, event) and not named(text, event), text
    # with a title or a group word too, where the paper grader reads the person as a name it cannot place and the
    # rest of the answer names nobody ("Group" is a word of the title of paper:2601.05242: no name)
    checked = 0
    for i, word in enumerate(WORDS):
        for form in (f"{word} {other}", f"{other} {word}"):
            text = TEMPLATES[i % len(TEMPLATES)].format(N=form, Y=Y)
            placed = any(_compact(other) in _compact(u) for u in _unknown_names(text, event))
            if placed and not text_mod.mentions_answer(text.replace(other, "someone"), event):
                checked += 1
                assert not named(text, event), text
    assert checked >= 60


def test_every_distractor_stays_another_person(real):
    for i, event in enumerate(real.named):
        for d in event.reject_aliases:
            assert text_mod._another_person(d, event), (event.id, d)
            text = TEMPLATES[i % len(TEMPLATES)].format(N=d, Y=Y)
            assert not named(text, event), (event.id, text)
    # also a distractor that is a near spelling of the expected answer, or carries a title: a listed wrong answer
    event = paper(1, "Wei Zhang", rejects=("Wei Zhanng",))
    assert named("The first author is Wei Zhangg.", event)
    for text in ("The first author is Wei Zhanng.", "The first author is Professor Wei Zhanng."):
        assert text_mod.mentions_answer(text, event) and not named(text, event), text


# ---- the scenarios of the check, through run_plan and build_report ------------------------------------------------


@pytest.mark.parametrize("form", ["The first author is Professor {S}.", "It is not {Y}, it is Professor {S}.",
                                  "It is not {Y}, it is {N}.",
                                  # review of 4fd7637: a word before a title, a title off the old list, an opening word
                                  "The first author is Visiting Professor {S}.", "Die Arbeit stammt von Herrn {S}.",
                                  "Either {S} or {Y}, I think."])
def test_two_lost_votes_on_a_title_or_a_near_spelling_cannot_decide_prediction_4(real, tmp_path, form):
    rows = [p for p in real.plan if p["set"] == "both_right"]

    def answer(meta, event):
        if meta["arm"] == "key" and event.id == DAYA:
            return form.format(S=parts(event)[1], Y=Y, N=misspelt(event))
        return f"The first author is {event.expected_current}."
    lost = form.format(S="Guo", Y=Y, N="Daya Guoo")

    def grading(req):
        return BackendError("upstream 500", status=500) if answer_of(req) == lost else CORRECT
    got, _ = run_real(real, rows, answer, tmp_path, grading=grading)
    daya = [r for r in got.rows if r["event_id"] == DAYA and r["arm"] == "key"]
    assert {(r["grade"], r["grader"], r["offers_expected"]) for r in daya} == {("ungraded", "judge:j1", True)}
    assert len(daya) == got.judges_undecided == 2
    report = build_report(rows, got.rows)
    four = report["predictions"][4]
    assert four["verdict"] == UNDECIDED and report["totals"]["open_offers"] == 2
    assert (four["comparison"]["b"], four["comparison"]["c"], four["items"]) == (2, 0, 20)
    old = [{**r, "offers_expected": False} if r in daya else r for r in got.rows]   # 8739803: no open offer
    assert verdicts(rows, old)["4"] == "not supported"


def test_every_lost_vote_on_titled_key_rows_cannot_support_3a(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "dark"]

    def answer(meta, event):
        x, s, _ = parts(event)
        return {"key": f"The first author is Professor {s}.", "full": f"The first author is {x}."}.get(
            meta["arm"], "I don't know.")
    got, _ = run_real(real, rows, answer, tmp_path, grading=lambda req: BackendError("down", status=500))
    key = [r for r in got.rows if r["arm"] == "key"]
    tally = {(g, o): sum(1 for r in key if (r["grade"], r["offers_expected"]) == (g, o))
             for g, o in {(r["grade"], r["offers_expected"]) for r in key}}
    assert tally == {("correct", True): 24, ("ungraded", True): 92}          # a distinctive surname: correct by rule
    report = build_report(rows, got.rows)
    half = report["predictions"][3]["halves"]["3a"]
    assert half["verdict"] == UNDECIDED and report["totals"]["open_offers"] == 92
    assert (half["comparison"]["b"], half["comparison"]["c"]) == (92, 0)
    old = [{**r, "offers_expected": False} if r["arm"] == "key" and r["grade"] == "ungraded" else r
           for r in got.rows]
    assert verdicts(rows, old)["3a"] == "supported"                           # 8739803: b = 92, c = 0, no check


def test_without_judges_near_spellings_on_dark_key_rows_cannot_support_3a(real, tmp_path):
    rows = [p for p in real.plan if p["set"] == "dark"]

    def answer(meta, event):
        return {"key": f"The first author is {misspelt(event)}.",
                "full": f"The first author is {event.expected_current}."}.get(meta["arm"], "I don't know.")
    got, _ = run_real(real, rows, answer, tmp_path, judges=False)
    key = [r for r in got.rows if r["arm"] == "key"]
    assert len(key) == 116 and {(r["grade"], r["offers_expected"]) for r in key} == {("ungraded", True)}
    report = build_report(rows, got.rows)
    assert verdicts(rows, got.rows)["3a"] == UNDECIDED and report["totals"]["open_offers"] == 116
    old = [{**r, "offers_expected": False} if r["arm"] == "key" else r for r in got.rows]
    assert verdicts(rows, old)["3a"] == "supported"                           # 8739803: b = 116, c = 0, no check


def test_rows_stored_with_8739803s_reading_are_refreshed_without_a_call(real, tmp_path):
    # a near spelling the paper rules leave open: ungraded without judges; 8739803 stored it as naming nobody
    rows = [p for p in real.plan if p["set"] == "both_right"]

    def answer(meta, event):
        return f"The first author is {misspelt(event)}." if meta["arm"] == "key" else "I don't know."
    run_real(real, rows, answer, tmp_path, judges=False)
    out = tmp_path / "keys.jsonl"
    stored = read_jsonl(out)
    near = [r for r in stored if r["arm"] == "key"]
    assert len(near) == 20 and {(r["grade"], r["offers_expected"]) for r in near} == {("ungraded", True)}
    write_jsonl(out, [{**r, "offers_expected": False} if r in near else r for r in stored])
    backend = ScriptedBackend(lambda model, req: pytest.fail("no call"))
    again = asyncio.run(run_plan(backend, RECIPIENTS, rows, out, events=real.events, backend_name="scripted"))
    assert backend.calls == [] and again.start.offer_flag == len(near) == again.regraded
    assert {r["offers_expected"] for r in again.rows if r["arm"] == "key"} == {True}


# ---- review of 4fd7637: words that are no name, and an accepted form with an initial ------------------------------

# a word before a listed title, a title the list of 4fd7637 lacked, and a word that opens the sentence: the paper grader
# keeps each in the name it cannot place, and 4fd7637 read that name as another person's (on 81 of the 107 events the
# rules then leave the answer to the judges or open)
UNLISTED = ["The first author is Visiting Professor {S}.", "It is not {Y}, it is Distinguished Professor {S}.",
            "I believe it is Adjunct Professor {S}.", "The first author is Full Professor {S}.",
            "The first author is Lecturer {S}.", "Die Arbeit stammt von Herrn {S}.",
            "The first author is Monsieur {S}.", "The first author is Professeur {S}.",
            "The first author is Director {S}.", "Either {S} or {Y}, I think.",
            "Surely {S} wrote it.", "Clearly {S} is the first author.", "Honestly {S}, but I am not sure.",
            "Obviously {S}.", "Hmm {S}, maybe?", "Answer {S}, I believe.", "Namely {S}, with {Y} as senior author."]


@pytest.mark.parametrize("form", UNLISTED)
def test_a_word_before_a_title_a_new_title_and_an_opening_word_write_the_expected_answer(real, form):
    placed = 0
    for event in real.named:
        _, s, _ = parts(event)
        text = form.format(S=s, Y=Y)
        placed += any(s in u for u in _unknown_names(text, event))     # the paper grader keeps the word in the name
        assert named(text, event) and run_mod._offers_expected(event, text) is True, (event.id, text)
    assert placed == len(real.named) == 107


def test_every_word_of_the_lists_writes_the_expected_answer(real):
    # each honorific, title or group word the paper grader keeps in a name, before and after the surname, and each
    # opening word it keeps at the start of one; every fifth event in turn
    kept = sorted(w for w in text_mod._HONORIFICS if w not in _NAME_STOP)
    opening = sorted(w for w in text_mod._OPENING_WORDS if w not in _NAME_STOP)
    assert (len(kept), len(opening)) == (62, 35)
    for k, word in enumerate(kept + opening):
        for event in real.named[k % 5::5]:
            _, s, _ = parts(event)
            forms = [f"{word.title()} {s}"] if word in opening else [f"{word.title()} {s}", f"{s} {word.title()}"]
            for form in forms:
                text = f"The first author is {form}." if word not in opening else f"{form} is the first author."
                assert named(text, event), (event.id, text)


def test_other_people_and_the_listed_limits_stay_another_person(real):
    lingrui, daya, ziyu = real.events["paper:2507.13334"], real.events[DAYA], real.events["paper:2604.08377"]
    for text in ("The first author is Visiting Professor Yutong Mei.", "Either Yutong Mei or Sanjay Ramaswamy.",
                 "Die Arbeit stammt von Herrn Yutong Mei.", "Clearly Yutong Mei wrote it.", "Hmm Yutong Mei, maybe?"):
        assert text_mod.mentions_answer(text, lingrui) and not named(text, lingrui), text
    for i, event in enumerate(real.named):
        for d in event.reject_aliases:
            text = UNLISTED[i % len(UNLISTED)].format(S=d, Y=Y)
            assert not named(text, event), (event.id, text)
    # a word on none of the lists, also one that is a given name too, and a slip that is no near spelling: another
    # person, also where the rules leave the answer open ("Dean Guo": "Dean" begins like "Daya", the test of initials)
    for text, rule in (("The first author is Researcher Guo.", "incorrect"), ("The first author is Dean Guo.", None)):
        assert text_mod.mentions_answer(text, daya) and not named(text, daya), text
        assert (grade_rule(daya, "current_d", text) or (None,))[0] == rule, text
    assert ziyu.expected_current == "Ziyu Ma" and not _near_name("Ziyu Mi", "Ziyu Ma")
    assert grade_rule(ziyu, "current_d", "The first author is Ziyu Mi.")[0] == "incorrect"
    assert not named("The first author is Ziyu Mi.", ziyu)
    # section 6's other examples: a slip of a short surname, and another romanisation
    for x, other in (("Kuan Li", "Kuan La"), ("Zhiyu Li", "Zhiyu Lee"), ("Sen Xu", "Sen Hsu")):
        event = next(e for e in real.named if e.expected_current == x)
        text = f"The first author is {other}."
        assert grade_rule(event, "current_d", text)[0] == "incorrect" and not _near_name(other, x), other
        assert text_mod.mentions_answer(text, event) and not named(text, event), other


def test_a_short_given_name_with_the_expected_initial_stays_another_person(real):
    # 4fd7637 compared a name also with the forms that write the given name only as an initial ("Y. Zhang"), and
    # against such a short form the paper grader's similarity accepts a two-letter given name of the expected initial;
    # on every event one such name that is no near spelling of the expected answer or of a form written out is now
    # another person, as section 6 says (where the expected given name has two letters itself, every such name is one
    # slip away from it: "Yi Zhao" for Yu Zhao)
    checked, short = 0, [e for e in real.named if len(parts(e)[2]) == 2]
    assert [e.expected_current for e in short] == ["Yu Zhao", "Ji Zeng", "Lu Dai"]
    for event in real.named:
        x, s, g = parts(event)
        if event in short:
            assert not text_mod._another_person(f"{g[0]}a {s}", event), x
            continue
        forms = [f for f in dict.fromkeys([x, *event.aliases]) if len(f.split()) >= 2]
        spelt = [f for f in forms if text_mod._writes_given_name(f)]
        initial = [f for f in forms if not text_mod._writes_given_name(f)]
        for vowel in "aeiouy":
            name = f"{g[0]}{vowel} {s}"
            text = f"The first author is {name}."
            if name.split()[0] == g or not _unknown_names(text, event) or any(text_mod._near_spelling(name, f)
                                                                              for f in spelt):
                continue
            assert any(text_mod._near_spelling(name, f) for f in initial), (event.id, name)   # 4fd7637: the answer
            assert text_mod._another_person(name, event) and not named(text, event), (event.id, name)
            checked += 1
            break
    assert checked == len(real.named) - len(short) == 104
    for x, other, rule in (("Yujia Zhang", "Yu Zhang", None), ("Bin Wang", "Bo Wang", None),
                           ("Daya Guo", "Di Guo", None), ("Lingrui Mei", "Lu Mei", None),
                           ("Ailin Huang", "Gao Huang", "incorrect")):
        event = next(e for e in real.named if e.expected_current == x)
        text = f"The first author is {other}."
        assert text_mod._another_person(other, event) and not named(text, event), other
        assert (grade_rule(event, "current_d", text) or (None,))[0] == rule, other
    assert _near_name("Gao Huang", "A. Huang") and not _near_name("Gao Huang", "Ailin Huang")


def test_an_accepted_form_with_an_initial_still_reads_a_name_without_a_given_name(real):
    daya, yoav, lingrui = real.events[DAYA], real.events["paper:2501.08319"], real.events["paper:2507.13334"]
    assert yoav.expected_current == "Yoav Gur-Arieh"
    # "Gur-Ariehh" is near "Y. Gur-Arieh" only; "D. Guoo" is near "D. Guo", though no word of it mentions Daya Guo
    assert text_mod._near_spelling("gur-ariehh", "Y. Gur-Arieh") and not text_mod._near_spelling("gur-ariehh",
                                                                                                 "Yoav Gur-Arieh")
    assert not text_mod._another_person("Professor Gur-Ariehh", yoav)
    assert not text_mod._another_person("Professor D. Guoo", daya)
    assert not named("The first author is Professor D. Guoo.", daya)            # mentions_answer reads no misspelling
    for event, text in ((yoav, "The first author is Professor Gur-Ariehh."),
                        (lingrui, "The first author is Mei Lingrul."), (daya, "The first author is Guo Dayaa.")):
        assert named(text, event), text
    # a near spelling of a form written surname first, which the paper grader does not compare: graded incorrect
    for event, text in ((lingrui, "The first author is Mei Lingrul."), (daya, "The first author is Guo Dayaa.")):
        assert grade_rule(event, "current_d", text)[0] == "incorrect", text


def test_the_test_of_initials_reads_a_given_name_without_a_hyphen_as_initials(real):
    # section 6: grading._near_name's test of initials accepts a given name of one unhyphenated word with the expected
    # initial, not one of two words whose initials do not begin the expected given name, and nothing for a hyphenated
    # expected given name
    zhiyu, shih = real.events["paper:2507.03724"], real.events["paper:2601.05242"]
    assert (zhiyu.expected_current, shih.expected_current) == ("Zhiyu Li", "Shih-Yang Liu")
    assert _near_name("Zhaoyang Li", "Zhiyu Li") and _near_name("Zhao-Yang Li", "Zhiyu Li") is False
    assert not _near_name("Zhao Yang Li", "Zhiyu Li") and not _near_name("Sheng Liu", "Shih-Yang Liu")
    assert grade_rule(zhiyu, "current_d", "The first author is Zhao Yang Li.")[0] == "incorrect"
    assert grade_rule(shih, "current_d", "The first author is Sheng Liu.")[0] == "incorrect"
    assert _near_name("Shiyu Liu", "S. Liu") and not _near_name("Shiyu Liu", "Shih-Yang Liu")
    # a different given name the tests accept is read as the expected answer; more than a slip away it is not
    ziyu, yuxue = real.events["paper:2604.08377"], next(e for e in real.named if e.expected_current == "Yuxue Yang")
    assert ratio("Ziyi Ma", "Ziyu Ma") < _NEAR_NAME_RATIO <= ratio("Zhiyuan Li", "Zhiyu Li")
    assert f"{ratio('Zhiyuan Li', 'Zhiyu Li'):.3f}" == "0.875"
    for event, name in ((ziyu, "Ziyi Ma"), (zhiyu, "Zhiyuan Li"), (yuxue, "Yang Yue")):
        assert not text_mod._another_person(name, event), name
    assert text_mod._near_spelling("Yang Yue", "Yang Yuxue") and not _near_name("Yang Yue", "Yuxue Yang")
    assert text_mod._another_person("Zhaoyang Li", zhiyu) and text_mod._another_person("Zi Ma", ziyu)


@pytest.mark.parametrize("form", ["The first author is Visiting Professor {S}.", "I believe it is Lecturer {S}."])
def test_every_lost_vote_on_key_rows_with_a_word_before_the_surname_cannot_support_3a(real, tmp_path, form):
    rows = [p for p in real.plan if p["set"] == "dark"]

    def answer(meta, event):
        x, s, _ = parts(event)
        return {"key": form.format(S=s), "full": f"The first author is {x}."}.get(meta["arm"], "I don't know.")
    got, _ = run_real(real, rows, answer, tmp_path, grading=lambda req: BackendError("down", status=500))
    key = [r for r in got.rows if r["arm"] == "key"]
    tally = {(g, o): sum(1 for r in key if (r["grade"], r["offers_expected"]) == (g, o))
             for g, o in {(r["grade"], r["offers_expected"]) for r in key}}
    assert tally == {("correct", True): 24, ("ungraded", True): 92}
    report = build_report(rows, got.rows)
    half = report["predictions"][3]["halves"]["3a"]
    assert half["verdict"] == UNDECIDED and report["totals"]["open_offers"] == 92
    old = [{**r, "offers_expected": False} if r["arm"] == "key" and r["grade"] == "ungraded" else r
           for r in got.rows]
    assert verdicts(rows, old)["3a"] == "supported"                           # 4fd7637: b = 92, c = 0, no check


def test_a_slip_that_is_no_near_spelling_is_a_listed_limit(real, tmp_path):
    # section 6 and its row "words that are no name": a one-letter slip of a short surname is no near spelling, so
    # with every grading vote lost on it 3a can still be supported with no open offer; the undecided votes are counted
    rows = [p for p in real.plan if p["set"] == "dark"]
    slipped = {}
    for event in (real.events[e] for e in sorted({p["event_id"] for p in rows})):
        x, s, g = parts(event)
        for i in range(len(s) - 1, 0, -1):
            cands = [f"{g} {s[:i]}{c}{s[i + 1:]}" for c in "aeiou" if c != s[i]]
            cand = next((c for c in cands if _unknown_names(f"The first author is {c}.", event)
                         and not _near_name(c, x) and not run_mod._offers_expected(event, f"The first author is {c}.")),
                        None)
            if cand:
                slipped[event.id] = cand
                break
    assert len(slipped) == 7 and slipped["paper:2604.08377"] == "Ziyu Mi"
    lost = {f"The first author is {s}." for s in slipped.values()}

    def answer(meta, event):
        if meta["arm"] == "key" and event.id in slipped:
            return f"The first author is {slipped[event.id]}."
        return f"The first author is {event.expected_current}." if meta["arm"] in ("key", "full") else "I don't know."
    got, _ = run_real(real, rows, answer, tmp_path,
                      grading=lambda req: BackendError("down", status=500) if answer_of(req) in lost else CORRECT)
    slips = [r for r in got.rows if r["arm"] == "key" and r["event_id"] in slipped]
    assert len(slips) == got.judges_undecided == 11
    assert {(r["grade"], r["offers_expected"]) for r in slips} == {("ungraded", False)}
    report = build_report(rows, got.rows)
    half = report["predictions"][3]["halves"]["3a"]
    assert (half["verdict"], half["comparison"]["b"], half["comparison"]["c"]) == ("supported", 11, 0)
    assert report["totals"]["open_offers"] == 0 and report["totals"]["judges_undecided"] == 11
