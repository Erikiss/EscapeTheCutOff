"""Key, not value (docs/DESIGN_keys.md version 2): the statistics, the join of plan and results with provenance, the
rates, the paired comparisons, the costs and the Markdown of the report (section 6). The decision rules of section 7
are tested in test_keys_report_rules.py, which reuses the builder below."""

from __future__ import annotations

import itertools
import json
import math
from fractions import Fraction
from typing import Any

import pytest

from cutoffmix.bench.keys import (
    ARMS_BY_SET,
    build_report,
    fisher_exact,
    items_needed,
    mcnemar_exact,
    prompt_sha,
    render_report,
    stratified_difference,
    stratified_permutation_test,
    wilson,
)

TODAY = "2026-11-27"
PAIR = ("fable", "sol")


# ---- a builder of plan and result rows ------------------------------------------------------------------------------


def plan_row(set_: str, pair: tuple[str, str], event: str, arm: str, *, recognised: bool | None = None,
             pointer: bool | None = None, related: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    donor, recipient = pair
    keyed = set_ in ("dark", "both_right")
    return {"id": f"{set_}|{donor}|{recipient}|{event}|{arm}", "set": set_, "donor": donor, "recipient": recipient,
            "event_id": event, "arm": arm, "recognised": recognised,
            "donor_grade": "correct" if keyed else "incorrect", "donor_answer": "Ada Quist",
            "note": None if arm == "alone" else f"a {arm} note", "note_event": "other" if arm == "wrong_key" else None,
            "today": TODAY, "cues": ["cue"] if keyed else None, "pointer": pointer if keyed else None,
            "related_known": related if keyed else None,
            "messages": [{"role": "system", "content": f"Today is {TODAY}."},
                         {"role": "user", "content": f"{set_} {donor} {recipient} {event} {arm}"}]}


def result_row(p: dict[str, Any], grade: str = "abstain", **fields: Any) -> dict[str, Any]:
    """A result row of ``run`` for plan row ``p``, with provenance, so it joins."""
    row = {"id": p["id"], "set": p["set"], "donor": p["donor"], "recipient": p["recipient"],
           "event_id": p["event_id"], "arm": p["arm"], "response": "some answer", "grade": grade, "hedged": False,
           "grader": "rule", "adopted": None, "adoption": None, "adoption_grader": None, "latency_s": 1.0,
           "usage": {"cost": 0.001}, "judge_usage": [], "backend": "openrouter",
           "prompt_sha": prompt_sha(p["messages"]), "request_sha": "0" * 40, "note_event": p["note_event"],
           "today": TODAY, "error": None}
    row.update(fields)
    return row


ADOPTION = {  # the result fields of an adoption outcome (design section 6)
    "endorsed": {"grade": "incorrect", "adoption": "endorsed", "adopted": True, "adoption_grader": "rule"},
    "undecided": {"grade": "ungraded", "adoption": "reported", "adopted": None, "adoption_grader": "rule"},
    "reported": {"grade": "abstain", "adoption": "reported", "adopted": False, "adoption_grader": "judge:j"},
    "absent": {"grade": "abstain", "adoption": "absent", "adopted": False, "adoption_grader": "rule"},
}


class Study:
    """Plan and result rows built item by item. The outcome of an arm is True (correct), False (abstain), another
    grade, a dict of result fields, or None (no result row: missing)."""

    def __init__(self) -> None:
        self.plan: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []

    def item(self, set_: str, event: str, outcomes: dict[str, Any] | None = None, *, pair: tuple[str, str] = PAIR,
             recognised: bool | None = None, pointer: bool | None = None,
             related: list[dict[str, Any]] | None = None, **common: Any) -> Study:
        outcomes = outcomes or {}
        for arm in ARMS_BY_SET[set_]:
            p = plan_row(set_, pair, event, arm, recognised=recognised, pointer=pointer, related=related)
            self.plan.append(p)
            out = outcomes.get(arm, False)
            if out is None:
                continue
            fields = dict(common)
            if isinstance(out, dict):
                fields.update(out)
                grade = fields.pop("grade", "abstain")
            else:
                grade = {True: "correct", False: "abstain"}.get(out, out) if isinstance(out, bool) else out
            self.results.append(result_row(p, grade, **fields))
        return self

    def dark(self, n: int, *, pair: tuple[str, str] = PAIR, prefix: str = "d", start: int = 0,
             correct: dict[str, Any] | None = None, recognised: Any = None, pointer: Any = (),
             **common: Any) -> Study:
        """``n`` dark items; ``correct[arm]`` holds the item indices where that arm is correct (``full`` defaults to
        every item); ``recognised`` is a value or a function of the index; ``pointer`` holds indices."""
        correct = correct or {}
        for i in range(start, start + n):
            outcomes = {arm: i in correct.get(arm, ()) for arm in ARMS_BY_SET["dark"]}
            if "full" not in correct:
                outcomes["full"] = True
            rec = recognised(i) if callable(recognised) else recognised
            self.item("dark", f"{prefix}{i}", outcomes, pair=pair, recognised=rec, pointer=i in pointer, **common)
        return self

    def adoption(self, n: int, *, pair: tuple[str, str] = PAIR, set_: str = "donor_wrong", prefix: str = "w",
                 full: Any = "absent", alone: Any = "absent") -> Study:
        """``n`` adoption items; ``full`` and ``alone`` are an ADOPTION label, a dict of result fields, or a function
        of the index giving one."""
        for i in range(n):
            labels = {arm: (f(i) if callable(f) else f) for arm, f in (("full", full), ("alone", alone))}
            self.item(set_, f"{prefix}{i}", {arm: (None if lab is None else dict(ADOPTION[lab] if isinstance(lab, str)
                                                                                    else lab))
                                              for arm, lab in labels.items()}, pair=pair)
        return self

    def both_right(self, n: int, *, pair: tuple[str, str] = PAIR, prefix: str = "r",
                   wrong: dict[str, Any] | None = None) -> Study:
        """``n`` both_right items, correct under every arm except the indices in ``wrong[arm]``."""
        wrong = wrong or {}
        for i in range(n):
            self.item("both_right", f"{prefix}{i}", {arm: i not in wrong.get(arm, ())
                                                     for arm in ARMS_BY_SET["both_right"]}, pair=pair, recognised=True)
        return self

    def report(self) -> dict[str, Any]:
        return build_report(self.plan, self.results)

    def verdicts(self) -> dict[Any, str]:
        return {p["id"]: p["verdict"] for p in self.report()["predictions"]}


def name(pair: tuple[str, str] = PAIR) -> str:
    return f"{pair[0]}->{pair[1]}"


# ---- statistics ---------------------------------------------------------------------------------------------------


def test_wilson_known_values():
    assert wilson(8, 10) == pytest.approx((0.4902, 0.9433), abs=1e-4)
    assert wilson(0, 10) == pytest.approx((0.0, 0.2775), abs=1e-4)
    assert wilson(10, 10) == pytest.approx((0.7225, 1.0), abs=1e-4)
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(3, 2)


def test_exact_tests_known_values():
    assert mcnemar_exact(10, 2) == pytest.approx(0.03857, abs=1e-5)
    assert mcnemar_exact(2, 10) == mcnemar_exact(10, 2)
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)
    assert fisher_exact(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_exact(10, 0, 0, 10) == pytest.approx(2 / math.comb(20, 10))
    assert fisher_exact(0, 0, 0, 0) == 1.0


def test_items_needed_for_a_gain():
    # the smallest n at which b = ceil(m n), c = 0 reaches p < 0.05 (design section 7)
    assert items_needed(0.10) == 51 and items_needed(0.20) == 26
    for margin in (0.10, 0.20, 0.30):
        n = items_needed(margin)
        exact = Fraction(str(margin))
        assert mcnemar_exact(math.ceil(exact * n), 0) < 0.05 <= mcnemar_exact(math.ceil(exact * (n - 1)), 0)
    assert mcnemar_exact(5, 0) == 0.0625 and mcnemar_exact(6, 0) == 0.03125     # the floor of 6 items
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError):
            items_needed(bad)


def test_items_needed_at_the_observed_discordance():
    # a gain of m among discordant pairs at the rate the data show: c = round(n (d - m) / 2), b = c + ceil(m n)
    def counts(n: int, d: Fraction, m: Fraction = Fraction(1, 10)) -> tuple[int, int]:
        c = math.floor(n * (max(d, m) - m) / 2 + Fraction(1, 2))
        return c + math.ceil(m * n), c

    d = Fraction(11, 71)                                  # b = 8, c = 3 on the decisive pair's 71 items
    n = items_needed(0.10, discordance=d)
    assert n == 71 and counts(71, d) == (10, 2) and mcnemar_exact(10, 2) < 0.05
    assert all(mcnemar_exact(*counts(k, d)) >= 0.05 for k in range(1, n))
    # no discordance, or less than the margin: the number without it (a gain of m disagrees on m n items at least)
    assert items_needed(0.10, discordance=0) == items_needed(0.10, discordance=0.05) == items_needed(0.10) == 51
    assert items_needed(0.20, discordance=Fraction(11, 53)) == items_needed(0.20) == 26
    # more disagreement needs more items
    needed = [items_needed(0.10, discordance=Fraction(k, 20)) for k in range(21)]
    assert needed == sorted(needed) and needed[-1] == 391 and items_needed(0.10, discordance=0.25) == 111
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            items_needed(0.10, discordance=bad)


def _exact_permutation_p(units):
    """Every assignment of the flags within each stratum, enumerated."""
    strata: dict[str, list[tuple[bool, int]]] = {}
    for stratum, flag, value in units:
        strata.setdefault(stratum, []).append((flag, value))

    def statistic(groups):
        total = Fraction(0)
        for items in groups:
            f = [v for fl, v in items if fl]
            o = [v for fl, v in items if not fl]
            if f and o:
                total += Fraction(len(f) * len(o), len(items)) * (Fraction(sum(f), len(f)) - Fraction(sum(o), len(o)))
        return total

    observed = abs(statistic(strata.values()))
    options = []
    for items in strata.values():
        k = sum(1 for fl, _ in items if fl)
        options.append([[(i in chosen, v) for i, (_, v) in enumerate(items)]
                        for chosen in map(set, itertools.combinations(range(len(items)), k))])
    hits = total = 0
    for combo in itertools.product(*options):
        total += 1
        hits += abs(statistic(combo)) >= observed
    return hits / total


def test_stratified_difference_by_hand():
    # stratum a: flagged 1, 0 against 0, 0, 0 (weight 2*3/5, difference 0.5); stratum b: 1 against 1, 0 (2/3, 0.5)
    units = [("a", True, 1), ("a", True, 0), ("a", False, 0), ("a", False, 0), ("a", False, 0),
             ("b", True, 1), ("b", False, 1), ("b", False, 0)]
    got = stratified_difference(units)
    assert got["statistic"] == pytest.approx(0.6 + 1 / 3) and got["weight"] == pytest.approx(1.2 + 2 / 3)
    assert got["difference"] == pytest.approx(0.5)
    # a stratum with one group only adds nothing; no stratum with both groups: no difference
    assert stratified_difference(units + [("c", True, 1), ("c", True, 1)]) == got
    assert stratified_difference([("a", True, 1), ("b", False, 0)]) == {"statistic": 0.0, "weight": 0.0,
                                                                         "difference": None}


def test_stratified_permutation_test_matches_exact_enumeration():
    units = [("a", i < 3, v) for i, v in enumerate([1, 1, 0, 0, -1, 0, 0])] + \
            [("b", i < 2, v) for i, v in enumerate([1, 0, 0, 1, 0, -1])] + \
            [("c", i < 4, v) for i, v in enumerate([1, 1, 1, 0, 0])]
    exact = _exact_permutation_p(units)
    got = stratified_permutation_test(units, permutations=20_000, seed=0)
    assert got["p"] == pytest.approx(exact, abs=0.015)
    assert got["statistic"] == pytest.approx(stratified_difference(units)["statistic"])
    assert got["permutations"] == 20_000 and got["seed"] == 0
    assert stratified_permutation_test(units, seed=0) == got                         # fixed seed: deterministic
    halves = [(s, f, Fraction(v, 2)) for s, f, v in units]                            # non-integer values
    assert stratified_permutation_test(halves, permutations=4000)["p"] == pytest.approx(exact, abs=0.03)


def test_stratified_permutation_test_permutes_within_strata():
    # pooled, the flagged units look better (4/5 against 1/5); within each stratum they are equal: statistic 0, p 1
    units = [("a", True, 1)] * 4 + [("a", False, 1)] + [("b", True, 0)] + [("b", False, 0)] * 4
    got = stratified_permutation_test(units, permutations=500)
    assert got["statistic"] == 0 and got["p"] == 1.0
    strong = [("a", i < 20, int(i < 20)) for i in range(40)]
    got = stratified_permutation_test(strong)
    assert got["difference"] == 1.0 and got["p"] == pytest.approx(1 / 20_001)        # (1 + 0) / (1 + 20 000)
    assert stratified_permutation_test([], permutations=3)["p"] == 1.0
    with pytest.raises(ValueError):
        stratified_permutation_test(strong, permutations=0)


# ---- join and provenance (robust-1) ---------------------------------------------------------------------------------


def test_stale_foreign_and_unprovenanced_rows_are_left_out_and_counted():
    study = Study().dark(10, correct={"key": range(10)})
    by_id = {r["id"]: r for r in study.results}
    key_ids = [p["id"] for p in study.plan if p["arm"] == "key"]
    by_id[key_ids[0]]["prompt_sha"] = "f" * 40                                   # stale: other messages
    by_id[key_ids[1]].pop("backend")                                              # no provenance
    by_id[key_ids[2]]["prompt_sha"] = None                                        # no provenance
    stale_then_fresh = dict(by_id[key_ids[3]], prompt_sha="e" * 40)
    study.results.insert(0, stale_then_fresh)                                     # an older stale row: last row counts
    fresh_then_stale = dict(by_id[key_ids[4]], prompt_sha="d" * 40)
    study.results.append(fresh_then_stale)                                        # a newer stale row: stale
    study.results.append({"id": "not-planned", "usage": {"cost": 1.5}, "backend": "openrouter", "prompt_sha": "x",
                          "judge_usage": [{"judge": "j", "purpose": "judge", "cost": 0.5, "usage": {"cost": 0.5}}]})
    study.results.append({"id": "not-planned", "usage": {"cost": 0.25}})          # foreign, a second line
    report = study.report()
    t = report["totals"]
    assert (t["stale"], t["no_provenance"], t["foreign_rows"], t["foreign_ids"]) == (2, 2, 2, 1)
    assert t["missing"] == 0 and t["joined"] == len(study.plan) - 4 and t["answered"] == t["joined"]
    assert t["cost"]["foreign"] == {"rows": 2, "recipients_usd": 1.75, "judges_usd": 0.5, "total_usd": 2.25}
    key = report["pairs"][name()]["dark"]["arms"]["key"]
    assert (key["planned"], key["answered"], key["stale"], key["no_provenance"], key["k"]) == (10, 6, 2, 2, 6)
    flags = " | ".join(report["flags"])
    assert "2 stale rows" in flags and "2 rows without provenance" in flags
    assert "2 rows of 1 ids not in the plan" in flags
    assert "interim" in flags                                                      # stale and unprovenanced rows
    assert report["predictions"][0]["verdict"] == "interim (6 of 10 planned items answered)"


def test_rows_from_several_backends_give_no_verdicts():
    study = Study().dark(40, correct={"key": range(20)})
    for row in study.results[: len(study.results) // 2]:
        row["backend"] = "mock"
    report = study.report()
    assert report["mixed_backends"] and not report["simulated"]
    assert report["backends"] == {"mock": 100, "openrouter": 100}
    assert {p["verdict"] for p in report["predictions"]} == {"not testable (mixed backends: mock, openrouter)"}
    md = render_report(report)
    assert md.startswith("**MIXED BACKENDS** (mock 100, openrouter 100)") and "SIMULATED" not in md


def test_a_report_of_mock_rows_only_is_simulated():
    study = Study().dark(40, correct={"key": range(20)}, backend="mock")
    study.results[0] = dict(study.results[0], backend="openrouter", prompt_sha="f" * 40)  # stale rows do not count
    report = study.report()
    assert report["simulated"] and not report["mixed_backends"] and report["backends"] == {"mock": 199}
    assert all(p["verdict"].startswith("simulated: ") for p in report["predictions"])
    assert report["predictions"][0]["verdict"] == "simulated: interim (39 of 40 planned items answered)"
    md = render_report(report)
    assert md.splitlines()[0].startswith("**SIMULATED**") and "**simulated: interim" in md
    assert "SIMULATED: every joined row comes from the mock backend" in report["flags"][0]
    real = Study().dark(40, correct={"key": range(20)}).report()
    assert not real["simulated"] and not real["mixed_backends"]
    assert not render_report(real).startswith("**")


def test_plan_and_result_rows_without_optional_fields_read_as_their_defaults():
    study = Study().dark(8, correct={"key": range(8)})
    for p in study.plan:
        for k in ("pointer", "recognised", "cues", "related_known", "today", "note_event"):
            p.pop(k)
    for r in study.results:
        for k in ("adoption", "adopted", "adoption_grader", "judge_usage", "request_sha", "hedged"):
            r.pop(k)
    report = study.report()
    assert report["paired"][name()]["items"] == {"all": 8, "no_pointer": 8, "pointer": 0}
    assert report["recognised"][name()]["all"]["unknown"] == 8
    assert report["predictions"][0]["verdict"] == "supported"
    json.dumps(report)


# ---- the primary analysis (V3) --------------------------------------------------------------------------------------


def test_ungraded_counts_as_not_correct_refusals_are_left_out():
    study = Study()
    grades = ["correct"] * 3 + ["ungraded"] * 2 + ["abstain"] * 5
    for i, grade in enumerate(grades):
        study.item("dark", f"d{i}", {"key": grade})
    study.item("dark", "refused", {"key": {"grader": "refusal", "grade": "ungraded"}})
    study.item("dark", "failed", {"key": {"error": "Timeout: boom", "grade": "ungraded", "grader": "none"}})
    study.item("dark", "missing", {"key": None})
    report = study.report()
    key = report["pairs"][name()]["dark"]["arms"]["key"]
    assert (key["planned"], key["answered"], key["refusals"], key["errors"], key["missing"]) == (13, 10, 1, 1, 1)
    assert (key["k"], key["n"], key["ungraded"]) == (3, 10, 2)                     # primary: ungraded not correct
    assert (key["sensitivity"]["k"], key["sensitivity"]["n"]) == (3, 8)            # sensitivity: ungraded left out
    assert key["ci"] == [round(v, 4) for v in wilson(3, 10)] and key["ungraded_flag"]
    assert any(f.startswith("fable->sol dark key: 2 of 10 answered rows are ungraded") for f in report["flags"])
    t = report["totals"]
    assert (t["refusals"], t["errors"], t["missing"], t["ungraded"]) == (1, 1, 1, 2)
    # exactly 10 % ungraded is not flagged
    tenth = Study()
    for i, grade in enumerate(["ungraded"] + ["abstain"] * 9):
        tenth.item("dark", f"d{i}", {"key": grade})
    assert not tenth.report()["pairs"][name()]["dark"]["arms"]["key"]["ungraded_flag"]
    # the share is of the answered rows: 1 of 5 answered (5 rows missing) is flagged, although it is 1 of 10 planned
    half = Study()
    for i, grade in enumerate(["ungraded"] + ["abstain"] * 4 + [None] * 5):
        half.item("dark", f"d{i}", {"key": grade})
    key = half.report()["pairs"][name()]["dark"]["arms"]["key"]
    assert (key["planned"], key["answered"], key["ungraded"], key["ungraded_flag"]) == (10, 5, 1, True)


def test_paired_comparisons_use_the_items_with_both_arms_answered():
    study = Study()
    study.item("dark", "a", {"key": True, "alone": False, "wrong_key": False, "blank": True, "full": True})
    study.item("dark", "b", {"key": True, "alone": True, "wrong_key": "ungraded", "blank": False, "full": True})
    study.item("dark", "c", {"key": False, "alone": True, "wrong_key": None, "blank": False, "full": True})
    study.item("dark", "d", {"key": "ungraded", "alone": False, "wrong_key": True, "blank": False, "full": True})
    study.item("dark", "e", {"key": True, "alone": {"error": "x"}, "wrong_key": False}, pointer=True)
    paired = study.report()["paired"][name()]
    plain, every, pointed = paired["no_pointer"], paired["all"], paired["pointer"]
    assert paired["items"] == {"all": 5, "no_pointer": 4, "pointer": 1}
    assert {k: plain["key_vs_alone"][k] for k in ("n_items", "b", "c")} == {"n_items": 4, "b": 1, "c": 1}
    assert {k: plain["key_vs_wrong_key"][k] for k in ("n_items", "b", "c")} == {"n_items": 3, "b": 2, "c": 1}
    assert plain["key_vs_wrong_key"]["sensitivity"] == {"n_items": 1, "b": 1, "c": 0, "p": 1.0}  # ungraded left out
    assert {k: plain["blank_vs_wrong_key"][k] for k in ("n_items", "b", "c")} == {"n_items": 3, "b": 1, "c": 1}
    assert {k: plain["full_vs_key"][k] for k in ("n_items", "b", "c")} == {"n_items": 4, "b": 2, "c": 0}
    assert every["key_vs_wrong_key"]["n_items"] == 4 and every["key_vs_alone"]["n_items"] == 4  # e's alone failed
    assert pointed["key_vs_wrong_key"] == {"n_items": 1, "b": 1, "c": 0, "p": 1.0,
                                           "sensitivity": {"n_items": 1, "b": 1, "c": 0, "p": 1.0}}


def test_recognition_split_and_guess_corrected_share():
    study = Study().dark(20, correct={"key": range(0, 20, 2)}, recognised=lambda i: i < 10, pointer=(0, 1))
    study.dark(2, prefix="u", recognised=None)
    rec = study.report()["recognised"][name()]
    assert (rec["all"]["recognised"], rec["all"]["unrecognised"], rec["all"]["unknown"]) == (10, 10, 2)
    assert rec["all"]["share"] == 0.5 and rec["all"]["guess_corrected"] == pytest.approx((0.5 - 0.25) / 0.75)
    assert rec["all"]["groups"]["recognised"]["key"]["k"] == 5 and rec["no_pointer"]["recognised"] == 8
    assert rec["no_pointer"]["groups"]["recognised"]["key"]["k"] == 4


def test_adoption_counts_per_arm():
    labels = ["endorsed"] * 4 + ["undecided"] * 3 + ["reported"] * 2 + ["absent"] * 6
    study = Study().adoption(len(labels), full=lambda i: labels[i], alone=lambda i: "endorsed" if i == 14 else "absent")
    study.adoption(2, set_="control", prefix="c", full="endorsed")
    report = study.report()
    full = report["adoption"][name()]["donor_wrong"]["arms"]["full"]
    assert (full["answered"], full["endorsed"], full["reported"], full["reported_undecided"], full["absent"]) == \
        (15, 4, 5, 3, 6)
    assert (full["k"], full["n"], full["undecided"]) == (4, 15, 3)                 # undecided counted as not adopted
    assert (full["sensitivity"]["k"], full["sensitivity"]["n"]) == (4, 12)
    test = report["adoption"][name()]["donor_wrong"]["full_vs_alone"]
    assert {k: test[k] for k in ("n_items", "b", "c")} == {"n_items": 15, "b": 4, "c": 1}
    pooled = report["pooled"]
    assert pooled["adoption_full_vs_alone"]["b"] == 6 and pooled["adoption_full_vs_alone"]["n_items"] == 17
    assert pooled["adoption_full_undecided"] == {"undecided": 3, "full_rows": 17, "share": 3 / 17}


def test_both_right_not_correct_rates_and_key_vs_alone():
    study = Study().both_right(12, wrong={"alone": (0, 1), "key": (1, 2, 3), "full": (4,)})
    harm = study.report()["harm"][name()]
    assert (harm["arms"]["alone"]["k"], harm["arms"]["key"]["k"], harm["arms"]["full"]["k"]) == (2, 3, 1)
    test = harm["key_vs_alone"]                                   # b = alone correct and key not, c the reverse
    assert (test["n_items"], test["b"], test["c"]) == (12, 2, 1) and test["excess_harm"] == pytest.approx(1 / 12)


# ---- costs (robust-5, robust-4) -------------------------------------------------------------------------------------


def test_cost_is_summed_over_every_row_of_the_results_file():
    study = Study().dark(2, correct={"key": range(2)})
    for row in study.results:
        row["usage"] = {"cost": 0.01}
    a, b, c, d, e = study.results[:5]
    regrade = {"usage": None, "error": None, "grader": "judge:j"}  # run --judges: the stored answer, judges only
    study.results.append(dict(a, **regrade, grade="correct",
                              judge_usage=[{"judge": "j", "purpose": "judge", "cost": 0.02, "usage": {"cost": 0.02}},
                                           {"cost": 0.03}]))                     # a bare usage mapping
    study.results.append(dict(b, usage={"cost": 0, "is_byok": True,                # the same request sent twice (BYOK)
                                        "cost_details": {"upstream_inference_cost": 0.04}}))
    study.results.insert(0, dict(c, usage={"cost": 0.05}, request_sha="1" * 40))  # an older answer to other settings
    study.results.insert(0, dict(d, usage={"cost": 0.06}, prompt_sha="e" * 40,    # an older answer to other messages
                                 request_sha="2" * 40))
    e.update(error="grading failed: ValueError: boom", grade="ungraded", grader="none")  # paid, grading raised ...
    study.results.append(dict(e, **regrade, grade="abstain", judge_usage=[{"cost": 0.01}]))  # ... graded again
    study.results.append({"id": "not-planned", "usage": {"cost": 0.5}})
    report = study.report()
    cost = report["totals"]["cost"]
    assert cost["all_rows"] == {"rows": 16, "recipients_usd": 0.75, "judges_usd": 0.06, "total_usd": 0.81}
    assert report["totals"]["cost_usd"] == 0.81 and report["totals"]["superseded_rows"] == 5
    assert cost["foreign"] == {"rows": 1, "recipients_usd": 0.5, "judges_usd": 0.0, "total_usd": 0.5}
    # the joined answers: per id every row of the request its last row answered. The re-graded answers keep the
    # recipient call that gave them (on an older line, the failed grading's included); the answers to other settings
    # or messages and the foreign row do not count
    assert cost["joined"] == {"ids": 10, "rows": 13, "recipients_usd": 0.14, "judges_usd": 0.06, "total_usd": 0.2}
    md = render_report(report)
    assert "recipients 0.7500 USD, judges 0.0600 USD, total 0.8100 USD" in md
    assert ("The joined answers, each with every row of the request it answered (the call, a failed grading, the "
            "re-grades): 0.2000 USD (recipients 0.1400, judges 0.0600; 13 rows of 10 ids).") in md


def test_the_joined_cost_keeps_the_recipient_calls_of_re_graded_answers():
    # 50 answers at 0.01; run --judges grades 20 of them again (usage None, 0.002 of judges each): the 50 joined answers
    # cost their 50 recipient calls and 20 judge calls, the whole file (the last row per id alone says 0.30 + 0.04)
    study = Study().dark(10, correct={"key": range(5)})
    for row in study.results:
        row["usage"] = {"cost": 0.01}
    for row in study.results[:20]:
        study.results.append(dict(row, usage=None, grade="abstain", grader="judge:j",
                                  judge_usage=[{"judge": "j", "purpose": "judge", "cost": 0.002,
                                                "usage": {"cost": 0.002}}]))
    cost = study.report()["totals"]["cost"]
    assert cost["joined"] == {"ids": 50, "rows": 70, "recipients_usd": 0.5, "judges_usd": 0.04, "total_usd": 0.54}
    assert cost["all_rows"]["total_usd"] == 0.54


# ---- rendering ---------------------------------------------------------------------------------------------------


def test_render_report_sections_and_json():
    study = Study().dark(30, correct={"key": range(12), "blank": (0,)}, recognised=lambda i: i % 2 == 0,
                         pointer=(29,), related=[{"event_id": "paper:9", "current_d": "correct", "mcq": "correct"}])
    study.adoption(10, full=lambda i: "endorsed" if i < 6 else "absent").both_right(10, wrong={"key": (0,)})
    report = study.report()
    json.loads(json.dumps(report))
    assert set(report) == {"totals", "backends", "simulated", "mixed_backends", "decisive_pair", "keyed_pairs",
                           "planned_dark", "pairs", "paired", "pointer_items", "recognised", "units", "adoption",
                           "harm", "pooled", "rules", "flags", "predictions"}
    assert [p["id"] for p in report["predictions"]] == [1, "1b", 2, 3, 4]
    assert all({"id", "text", "verdict", "evidence"} <= set(p) for p in report["predictions"])
    md = render_report(report)
    for heading in ("# Key, not value: report", "## Answers per pair, set and arm",
                    "## Paired comparisons on dark items", "## Pointer items", "## Dark items by recognition",
                    "## Units of prediction 2", "## Adoption of the donor's wrong name",
                    "## Items both answered correctly (both_right)",
                    "## Pre-registered predictions (docs/DESIGN_keys.md section 7)"):
        assert heading in md
    rules = report["rules"]                                         # the pre-registered constants (section 7)
    assert (rules["key_margin"], rules["adoption_margin"], rules["undecided_limit"], rules["floor_items"]) == \
        (0.10, 0.20, 0.20, 6)
    assert (rules["min_units"], rules["permutations"], rules["permutation_seed"], rules["harm_limit"]) == \
        (10, 20_000, 0, 0.10)
    assert "Decisive pair (section 7): fable->sol, 29 planned dark items without pointer (30 in all)" in md
    assert "| fable->sol | dark | key | 30 | 30 | 0 | 0 | 0 | 0 | 0 | 0 | 12/30 (40.0 %, 24.6-57.7) |" in md
    assert "| fable->sol | d29 | False | abstain | abstain | abstain | abstain | event_id paper:9, current_d " \
        "correct" in md
    assert "- **1** On dark items without pointer" in md and "- **1b** key beats blank" in md
    assert md == render_report(json.loads(json.dumps(report)))                   # renders from the JSON alone


def test_an_empty_plan_reports_nothing_testable():
    report = build_report([], [])
    assert report["decisive_pair"] is None and report["totals"]["planned"] == 0
    verdicts = {p["id"]: p["verdict"] for p in report["predictions"]}
    assert verdicts == {1: "not testable (n too small)", "1b": "not testable (n too small)",
                        2: "not testable (prediction 1 is not testable)", 3: "not testable (n too small)",
                        4: "not testable (n too small)"}
    md = render_report(report)
    assert "none (no keyed pair in the plan)" in md and "No pointer items in the plan." in md
