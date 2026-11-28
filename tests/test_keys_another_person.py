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
                                  "It is not {Y}, it is {N}."])
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
