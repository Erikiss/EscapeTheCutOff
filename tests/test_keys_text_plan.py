"""Key, not value (docs/DESIGN_keys.md): masking, names in answers, the plan on the real paper data and on
fixtures, and the cost estimate."""

from __future__ import annotations

import math
import re

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

from cutoffmix.bench.io import load_events, read_jsonl
from cutoffmix.bench.keys import (
    ARMS_BY_SET,
    KEY_USER_TEMPLATE,
    SETS,
    TITLE,
    WITHHELD,
    asserted_name,
    build_plan,
    estimate_cost,
    mask_note,
    mask_title,
    row_cost,
)
from cutoffmix.bench.keys.text import _contains_answer, _title_spans
from cutoffmix.bench.prompts import probe_messages

# ---- masking ----------------------------------------------------------------------------------------------------

PIORO = paper(1, "Maciej Pióro", subject="MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts")


@pytest.mark.parametrize("text", [
    "The first author is Maciej Pióro.",
    "The first author is Maciej Pioro.",                     # accent missing in the note
    "The first author is MACIEJ PIÓRO.",
    "The first author is Pióro, Maciej.",
    "The first author is M. Pióro.",
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


# ---- plan on the real paper data ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_plan():
    events = load_events(EVENTS)
    results = read_jsonl(RESULTS)
    rows, summary = build_plan(events, results, today=TODAY)
    return events, results, rows, summary


def test_real_plan_counts(real_plan):
    _, _, rows, summary = real_plan
    pairs = summary["pairs"]
    assert pairs[f"{FABLE}->{SOL}"]["dark_candidates"] == 96 and pairs[f"{FABLE}->{SOL}"]["dark"] == 96
    assert pairs[f"{FABLE}->{NEON}"]["dark"] == 50 and pairs[f"{NEON}->{SOL}"]["dark"] == 47
    for pair in (f"{SOL}->{FABLE}", f"{SOL}->{NEON}", f"{NEON}->{FABLE}"):
        assert pairs[pair]["dark_candidates"] == 1 and pairs[pair]["dark"] == 0 and pairs[pair]["keyed"] is False
    assert sorted((e["donor"], e["recipient"], e["dark"]) for e in summary["excluded"]) == sorted(
        [(SOL, FABLE, 1), (SOL, NEON, 1), (NEON, FABLE, 1)])
    assert all(pairs[p]["both_right"] == 10 for p in (f"{FABLE}->{SOL}", f"{FABLE}->{NEON}", f"{NEON}->{SOL}"))
    assert pairs[f"{SOL}->{FABLE}"]["control"] == 4 and pairs[f"{SOL}->{NEON}"]["control"] == 4
    assert sum(p["donor_wrong"] for p in pairs.values()) == 45 and summary["leaks"] == 0
    assert summary["calls_by_set_arm"]["dark"] == {arm: 193 for arm in ARMS_BY_SET["dark"]}
    assert summary["calls"] == len(rows) == 193 * 5 + 45 * 2 + 30 * 3 + 8 * 2
    assert summary["today"] == TODAY and summary["seed"] == 0


def test_real_plan_rows_are_well_formed(real_plan):
    events, _, rows, _ = real_plan
    by_id = {e.id: e for e in events}
    fields = {"id", "set", "donor", "recipient", "event_id", "arm", "recognised", "donor_grade", "donor_answer", "note",
              "note_event", "messages"}
    assert all(set(r) == fields for r in rows)
    assert len({r["id"] for r in rows}) == len(rows) and all(re.fullmatch(r"[0-9a-f]{16}", r["id"]) for r in rows)
    items: dict[tuple, set] = {}
    for r in rows:
        items.setdefault((r["set"], r["donor"], r["recipient"], r["event_id"]), set()).add(r["arm"])
    assert all(arms == set(ARMS_BY_SET[key[0]]) for key, arms in items.items())
    for r in rows:
        event = by_id[r["event_id"]]
        system, user = r["messages"]
        assert system["role"] == "system" and f"Today's date: {TODAY}." in system["content"]
        if r["arm"] == "alone":
            assert r["note"] is None and r["messages"] == probe_messages(event, "current_d", today=TODAY)
        else:
            assert user["content"] == KEY_USER_TEMPLATE.format(question=event.question_current, note=r["note"])
        if r["arm"] in ("key", "wrong_key"):
            assert not _contains_answer(r["note"], event), r["id"]
            assert not _title_spans(r["note"], event), r["id"]                 # neither its answer nor its title
        if r["arm"] == "wrong_key":
            other = by_id[r["note_event"]]
            assert other.id != event.id and other.expected_current != event.expected_current
            assert not _contains_answer(r["note"], other) and WITHHELD in r["note"]
        else:
            assert r["note_event"] is None
        if r["arm"] == "full" and r["set"] in ("dark", "both_right"):
            assert event.expected_current in r["note"] or _contains_answer(r["note"], event)
    dark = [r for r in rows if r["set"] == "dark"]
    assert all(r["donor_grade"] == "correct" and r["donor_answer"] for r in dark)
    assert all(isinstance(r["recognised"], bool) for r in dark)
    assert all(r["donor_grade"] == "incorrect" and r["donor_answer"] for r in rows if r["set"] in ("donor_wrong", "control"))


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
    assert 1.0 < cost["total_usd"] < 5.0
    assert math.isclose(sum(m["estimate_usd"] for m in cost["by_model"].values()), cost["total_usd"], abs_tol=1e-3)


# ---- plan on fixtures -------------------------------------------------------------------------------------------



def test_fixture_plan_sets_pairs_and_sampling():
    events, results = fixture_data()
    rows, summary = build_plan(events, results, pairs=[("a", "b"), ("b", "a")], both_right=1, min_items=3,
                               today=TODAY)
    ab = summary["pairs"]["a->b"]
    assert (ab["dark"], ab["donor_wrong"], ab["both_right"], ab["control"]) == (4, 1, 1, 1) and ab["keyed"]
    assert summary["pairs"]["b->a"]["keyed"] is False and summary["excluded"] == [{"donor": "b", "recipient": "a",
                                                                                   "dark": 0}]
    assert len(rows) == 4 * 5 + 2 + 3 + 2 and {r["donor"] for r in rows} == {"a"}
    dark = [r for r in rows if r["set"] == "dark"]
    assert {r["event_id"]: r["recognised"] for r in dark} == {"paper:1": True, "paper:2": False, "paper:3": True,
                                                               "paper:4": False}
    wrong = [r for r in dark if r["arm"] == "wrong_key"]
    assert all(r["note_event"] != r["event_id"] and r["note_event"] in {f"paper:{i}" for i in range(1, 7)}
               for r in wrong)
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
