"""Key, not value (docs/DESIGN_keys.md version 2, section 7): the pre-registered decision rules of the report, on the
scenarios of the review of version 1 (V1 to V7, robust-2). The rows are built directly (no backend)."""

from __future__ import annotations

from typing import Any

from test_keys_report import Study, name

from cutoffmix.bench.keys import fisher_exact, render_report, wilson

SOL_NEON = ("neon", "sol")


def verdicts_of(report: dict[str, Any]) -> dict[Any, str]:
    return {p["id"]: p["verdict"] for p in report["predictions"]}


def keyed_dark(study: Study, n: int = 40) -> Study:
    """Dark items where the key recovers half of them and full all (predictions 1 and 3a supported)."""
    return study.dark(n, correct={"key": range(n // 2)})


def gain_study(rec_alone: int, rec_gain: int, unrec_alone: int, unrec_gain: int, n_rec: int = 40,
               n_unrec: int = 40, **correct: Any) -> Study:
    """Dark items, the recognised ones first: ``alone`` (and the wrong key) correct on the first ``*_alone`` items of
    each group, the key on those and the next ``*_gain`` (where d = 1)."""
    alone = set(range(rec_alone)) | set(range(n_rec, n_rec + unrec_alone))
    key = set(range(rec_alone + rec_gain)) | set(range(n_rec, n_rec + unrec_alone + unrec_gain))
    return Study().dark(n_rec + n_unrec, correct={"alone": alone, "wrong_key": alone, "key": key, **correct},
                        recognised=lambda i: i < n_rec)


def test_a_complete_study_supports_every_prediction():
    alone = {0, 1, 30}
    study = Study().dark(60, correct={"key": alone | set(range(2, 20)) | {31}, "alone": alone, "blank": {0}},
                         recognised=lambda i: i < 30)
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent")
    study.both_right(20, wrong={"alone": (3,), "key": (7,)})
    report = study.report()
    assert verdicts_of(report) == {1: "supported", "1b": "supported", 2: "supported", 3: "supported",
                                   4: "supported"}
    assert report["decisive_pair"] == name() and report["keyed_pairs"] == [name()]
    dark = report["pairs"][name()]["dark"]["arms"]
    assert (dark["key"]["k"], dark["key"]["n"], dark["alone"]["k"]) == (22, 60, 3)
    two = report["predictions"][2]
    assert two["tests"]["d"]["p"] < 0.05 < two["tests"]["d"]["difference"] + 0.05
    assert two["groups"]["recognised"]["b"] == 18 and two["groups"]["unrecognised"]["b"] == 1
    assert "**supported**" in render_report(report)


# ---- interim, floor, three-way verdict (robust-2) -------------------------------------------------------------------


def test_a_partial_run_is_interim():
    # 96 dark items planned, 12 answered (a --limit or a cost cap): every one recovered, and still no verdict
    study = Study().dark(96, correct={"key": range(96)})
    answered = {p["id"] for p in study.plan if int(p["event_id"][1:]) < 12}
    study.results = [r for r in study.results if r["id"] in answered]
    v = verdicts_of(study.report())
    assert v[1] == v["1b"] == v[2] == v[3] == "interim (12 of 96 planned items answered)"


def test_a_partial_adoption_run_is_interim():
    study = keyed_dark(Study(), 20).adoption(20, full="endorsed")
    kept = {p["id"] for p in study.plan if p["set"] == "donor_wrong" and int(p["event_id"][1:]) < 7}
    study.results = [r for r in study.results if r["set"] != "donor_wrong" or r["id"] in kept]
    report = study.report()
    v = verdicts_of(report)
    assert v[3] == "interim (27 of 40 planned items answered)" and v[1] == "supported"
    assert report["predictions"][3]["halves"]["3b"]["verdict"] == "interim (7 of 20 planned items answered)"


def test_a_complete_plan_without_recoveries_is_not_supported():
    v = verdicts_of(Study().dark(96).report())              # no arm is ever correct (full aside)
    assert wilson(0, 96)[1] < 0.10                          # the data rule out a gain of 10 %
    assert v[1] == v["1b"] == "not supported" and v[2] == "not testable (prediction 1 is not supported)"


def test_fewer_than_six_items_are_not_testable():
    v = verdicts_of(Study().dark(5, correct={"key": range(5)}).report())     # b = 5, c = 0: p = 0.0625 at best
    assert v[1] == v["1b"] == v[3] == "not testable (n too small)"


def test_between_support_and_refutation_is_inconclusive():
    v = verdicts_of(Study().dark(20, correct={"key": (0, 1, 2)}).report())    # b = 3: p = 0.25, upper bound 36 %
    assert v[1] == v["1b"] == "inconclusive (20 items; about 51 would be needed for a gain of 0.10)"


def test_prediction_1_needs_both_comparisons():
    # key beats alone, but the wrong key is right almost as often: inconclusive; equally often: not supported
    v = verdicts_of(Study().dark(40, correct={"key": range(20), "wrong_key": range(18)}).report())
    assert v[1] == "inconclusive (40 items; about 51 would be needed for a gain of 0.10)"
    v = verdicts_of(Study().dark(40, correct={"key": range(20), "wrong_key": range(20)}).report())
    assert v[1] == "not supported"


def test_errors_stale_and_unprovenanced_rows_keep_a_prediction_interim_refusals_do_not():
    def first_verdict(change: dict[str, Any]) -> str:
        study = Study().dark(30, correct={"key": range(12)})
        next(r for r in study.results if r["arm"] == "alone").update(change)
        return verdicts_of(study.report())[1]

    for change in ({"error": "Timeout: boom", "grade": "ungraded"}, {"prompt_sha": "f" * 40}, {"backend": None}):
        assert first_verdict(change) == "interim (29 of 30 planned items answered)"
    assert first_verdict({"grader": "refusal", "grade": "ungraded"}) == "supported"   # a final outcome, left out


# ---- the decisive pair (V3, V6) -------------------------------------------------------------------------------------


def test_ungraded_rows_count_as_not_correct_and_the_plan_fixes_the_decisive_pair():
    # fable->sol: 30 planned items, half of its wrong keys ungraded (version 1 dropped those items, 15 complete ones
    # were left); neon->sol: 20 items, all graded, where the key does nothing (version 1 decided on it)
    study = Study().dark(30, correct={"key": range(12)})
    for row in study.results:
        if row["arm"] == "wrong_key" and int(row["event_id"][1:]) >= 15:
            row["grade"] = "ungraded"
    study.dark(20, pair=SOL_NEON, prefix="n")
    report = study.report()
    one = report["predictions"][0]
    assert report["decisive_pair"] == one["pair"] == name() and one["verdict"] == "supported"
    assert one["comparisons"]["key_vs_wrong_key"]["n_items"] == 30                # the ungraded wrong keys stay in
    wrong = report["pairs"][name()]["dark"]["arms"]["wrong_key"]
    assert (wrong["k"], wrong["n"], wrong["ungraded"], wrong["sensitivity"]["n"], wrong["ungraded_flag"]) == \
        (0, 30, 15, 15, True)
    # rows missing on the decisive pair make the verdict interim; it does not move to the complete pair
    study.results = [r for r in study.results if not (r["donor"] == "fable" and r["arm"] == "alone"
                                                      and r["event_id"] in ("d0", "d1", "d2"))]
    assert verdicts_of(study.report())[1] == "interim (27 of 30 planned items answered)"


def test_the_decisive_pair_counts_planned_dark_items_without_pointer():
    study = Study().dark(40, pair=("fable", "neon"), prefix="f", pointer=range(20)).dark(30, pair=SOL_NEON)
    assert study.report()["decisive_pair"] == "neon->sol"                      # 30 against 20 without pointer
    tie = Study().dark(10, pair=("b", "x"), prefix="b").dark(12, pair=("c", "x"), prefix="c", pointer=(0, 1))
    assert tie.report()["decisive_pair"] == "c->x"                             # a tie: more planned dark items
    tie = Study().dark(10, pair=("b", "x"), prefix="b").dark(10, pair=("a", "x"), prefix="a")
    assert tie.report()["decisive_pair"] == "a->x"                             # then the first name


def test_pointer_items_carrying_every_success_do_not_support_prediction_1():
    related = [{"event_id": "paper:other", "current_d": "correct", "mcq": "correct"}]
    study = Study().dark(40, correct={"key": range(8)}, pointer=range(8), related=related)
    report = study.report()
    one = report["predictions"][0]
    assert one["all_items"]["key_vs_alone"]["b"] == 8 and one["all_items"]["key_vs_alone"]["p"] < 0.05
    assert one["verdict"] == "inconclusive (32 items; about 51 would be needed for a gain of 0.10)"
    assert one["pointer_items"]["items"] == 8 and one["pointer_items"]["key"]["k"] == 8
    pointer = report["pointer_items"][name()]
    assert pointer["items"] == 8 and pointer["list"][0]["related_known"] == related
    assert pointer["list"][0]["outcomes"]["key"] == "correct"
    assert report["paired"][name()]["pointer"]["key_vs_wrong_key"]["b"] == 8
    assert verdicts_of(report)[2] == "not testable (prediction 1 is inconclusive)"
    # prediction 2's units leave the pointer items out; its numbers over all items keep them
    assert report["units"]["units"] == 32 and report["units"]["all_items"]["units"] == 40
    assert "all items (pointer included): 40 units" in report["predictions"][2]["evidence"]
    assert "With the pointer items: 40 units" in render_report(report)


# ---- prediction 1b and the confound (V4) ----------------------------------------------------------------------------


def test_blank_beating_the_wrong_key_marks_prediction_1_confounded():
    # any note helps as much as the key (blank = key) and the wrong key misleads (never correct)
    report = Study().dark(40, correct={"key": range(20), "blank": range(20)}).report()
    v = verdicts_of(report)
    assert v[1] == "supported (confounded: the wrong key misleads)"
    assert v["1b"] == "not supported"                       # b = c = 0 on 40 items: a gain of 10 % is ruled out
    one = report["predictions"][0]
    assert one["confounded"] and one["comparisons"]["blank_vs_wrong_key"]["b"] == 20


def test_prediction_1b_verdicts():
    v = verdicts_of(Study().dark(40, correct={"key": range(20)}).report())
    assert v[1] == v["1b"] == "supported"                   # blank never right: no confound, the content works
    v = verdicts_of(Study().dark(20, correct={"key": (0, 1), "blank": (5,)}).report())
    assert v["1b"] == "inconclusive (20 items; about 51 would be needed for a gain of 0.10)"
    v = verdicts_of(Study().dark(5, correct={"key": range(5)}).report())
    assert v["1b"] == "not testable (n too small)"


# ---- prediction 2: the gain, not the level (V2) -------------------------------------------------------------------


def test_a_key_equal_to_alone_never_supports_prediction_2():
    # key == alone, re-ask success 30 % on recognised and 4 % on unrecognised items: version 1 compared the key's
    # level (12/40 against 2/50, Fisher p < 0.05) and said "supported"
    assert fisher_exact(12, 28, 2, 48) < 0.05
    v = verdicts_of(gain_study(12, 0, 2, 0, n_unrec=50).report())
    assert v[1] == "not supported" and v[2] == "not testable (prediction 1 is not supported)"


def test_a_gain_that_is_equal_in_both_groups_does_not_support_prediction_2():
    # the key adds 0.2 in both groups on top of re-ask success of 30 % and 4 %: prediction 1 holds, the levels differ
    # (20/40 against 12/50), the gains do not
    assert fisher_exact(20, 20, 12, 38) < 0.05
    report = gain_study(12, 8, 2, 10, n_unrec=50).report()
    two = report["predictions"][2]
    assert verdicts_of(report)[1] == "supported"
    assert two["groups"]["recognised"]["b"] == 8 and two["groups"]["recognised"]["p"] < 0.05     # (a) holds
    assert two["tests"]["d"]["statistic"] == 0 and two["tests"]["d"]["p"] == 1.0                 # (b) fails
    assert two["verdict"] == "inconclusive (stratified difference of d +0.000, p=1; of d_w +0.000, p=1)"


def test_a_gain_concentrated_on_recognised_units_supports_prediction_2():
    report = gain_study(4, 16, 2, 0).report()
    two = report["predictions"][2]
    assert verdicts_of(report)[1] == "supported" and two["verdict"] == "supported"
    assert two["tests"]["d"]["difference"] == 0.4 and two["tests"]["d"]["p"] < 0.001
    assert two["tests"]["d"]["permutations"] == 20_000 and two["tests"]["d"]["seed"] == 0
    assert report["units"]["d"]["p"] == two["tests"]["d"]["p"]
    # also behind a confounded prediction 1 (the blank beats the misleading wrong key: b = 12, c = 0)
    report = gain_study(4, 16, 2, 0, blank=set(range(16)) | {40, 41}).report()
    assert report["predictions"][0]["comparisons"]["blank_vs_wrong_key"]["b"] == 12
    assert verdicts_of(report)[1] == "supported (confounded: the wrong key misleads)"
    assert verdicts_of(report)[2] == "supported"


def test_prediction_2_is_never_refuted_and_needs_ten_units_per_group():
    v = verdicts_of(gain_study(4, 0, 2, 16).report())          # the gain sits on unrecognised units
    assert v[1] == "supported" and v[2].startswith("inconclusive (stratified difference of d -0.400")
    v = verdicts_of(gain_study(4, 16, 2, 0, n_unrec=9).report())
    assert v[1] == "supported" and v[2] == "not testable (n too small)"


def test_prediction_2_units_are_distinct_recipient_events():
    a, b, c, d = ("fable", "sol"), ("neon", "sol"), ("fable", "neon"), ("gemini", "sol")
    study = Study().dark(20, pair=a, prefix="e", pointer=(5,), correct={"key": range(10)},
                         recognised=lambda i: i % 2 == 0)
    study.dark(10, pair=b, prefix="e", recognised=True)          # e0..e9: the same Sol events as fable->sol
    study.dark(5, pair=b, prefix="f", recognised=False)
    study.dark(5, pair=c, prefix="e", recognised=True)           # Neon's e0..e4: other units
    study.dark(2, pair=d, prefix="f", recognised=False)          # f0, f1: also neon->sol's
    next(r for r in study.results if r["donor"] == "fable" and r["recipient"] == "sol" and r["event_id"] == "e7"
         and r["arm"] == "alone").update(grader="refusal", grade="ungraded")
    report = study.report()
    units = report["units"]
    kept = {(u["recipient"], u["event_id"]): u["pair"] for u in units["list"]}
    assert report["decisive_pair"] == "fable->sol"
    assert kept[("sol", "e0")] == "fable->sol" and kept[("sol", "e19")] == "fable->sol"
    assert kept[("sol", "e5")] == "neon->sol"                    # fable->sol's e5 is a pointer item
    assert kept[("sol", "e7")] == "neon->sol"                    # fable->sol's e7 has no answered alone row
    assert kept[("sol", "f0")] == "neon->sol"                    # 15 planned dark items against 2
    assert kept[("neon", "e0")] == "fable->neon"
    assert units["units"] == len(kept) == 18 + 7 + 5 and units["duplicates"] == 8 + 2
    assert set(units["by_pair"]) == {"fable->sol", "neon->sol", "fable->neon"}


# ---- prediction 3: copying and endorsing (V1) -----------------------------------------------------------------------


def test_reported_speech_alone_cannot_support_3b():
    # every full answer only reports the note's name and the judges said REPORT (adopted false)
    study = keyed_dark(Study()).adoption(25, full="reported").adoption(15, set_="control", prefix="c",
                                                                        full="reported")
    three = study.report()["predictions"][3]
    assert three["halves"]["3a"]["verdict"] == "supported"
    assert three["halves"]["3b"]["comparison"]["n_items"] == 40 and three["halves"]["3b"]["comparison"]["b"] == 0
    assert three["halves"]["3b"]["verdict"] == three["verdict"] == "not supported"
    # undecided reported speech (no judges) on 5 of 40 full rows counts as not adopted, never as adoption
    three = keyed_dark(Study()).adoption(40, full=lambda i: "undecided" if i < 5 else "absent") \
        .report()["predictions"][3]
    assert three["halves"]["3b"]["undecided"] == {"undecided": 5, "full_rows": 40, "share": 0.125}
    assert three["verdict"] == "not supported"


def test_more_than_a_fifth_of_the_full_rows_undecided_makes_3b_not_testable():
    report = keyed_dark(Study()).adoption(40, full=lambda i: "undecided" if i < 9 else "endorsed").report()
    three = report["predictions"][3]
    assert three["halves"]["3b"]["verdict"] == three["verdict"] == \
        "not testable (reported speech undecided; grade with --judges)"
    at_limit = keyed_dark(Study()).adoption(40, full=lambda i: "undecided" if i < 8 else "endorsed")  # 20 %
    assert verdicts_of(at_limit.report())[3] == "supported"


def test_prediction_3_combines_its_halves():
    def endorsed(k: int) -> Any:
        return lambda i: "endorsed" if i < k else "absent"

    assert verdicts_of(keyed_dark(Study()).adoption(40, full=endorsed(15)).report())[3] == "supported"
    v = verdicts_of(keyed_dark(Study()).adoption(20, full=endorsed(3)).report())[3]       # upper bound 36 %
    assert v == "inconclusive (20 items; about 26 would be needed for a gain of 0.20)"
    assert verdicts_of(keyed_dark(Study()).adoption(5, full="endorsed").report())[3] == "not testable (n too small)"
    no_copy = Study().dark(40, correct={"key": range(40), "full": range(40)})            # 3a refuted
    assert verdicts_of(no_copy.adoption(5, full="endorsed").report())[3] == "not supported"


# ---- prediction 4: excess harm over re-asking (V7) ------------------------------------------------------------------


def test_excess_harm_with_equal_flip_rates_under_alone_and_key_is_supported():
    # re-asking alone misses 5 of 30 correct recalls and the key 5 others: version 1 counted 5/30 = 17 % as harm
    four = Study().both_right(30, wrong={"alone": range(5), "key": range(5, 10)}).report()["predictions"][4]
    assert four["verdict"] == "supported" and four["excess_harm"] == 0
    assert (four["comparison"]["b"], four["comparison"]["c"]) == (5, 5)
    assert "alone 5/30" in four["evidence"] and "key 5/30" in four["evidence"] and "full 0/30" in four["evidence"]


def test_excess_harm_of_ten_percent_or_more_is_not_supported():
    assert verdicts_of(Study().both_right(30, wrong={"key": range(3)}).report())[4] == "not supported"   # 10 %
    assert verdicts_of(Study().both_right(30, wrong={"key": range(2)}).report())[4] == "supported"       # 6.7 %
    assert verdicts_of(Study().both_right(9).report())[4] == "not testable (n too small)"
    pooled = Study().both_right(6, wrong={"key": (0,)}).both_right(6, pair=SOL_NEON, prefix="q").report()
    assert pooled["predictions"][4]["items"] == 12 and verdicts_of(pooled)[4] == "supported"          # 1/12
    # an ungraded key answer counts as not correct: harm is not hidden
    study = Study().both_right(20)
    for row in [r for r in study.results if r["arm"] == "key"][:2]:
        row["grade"] = "ungraded"
    assert verdicts_of(study.report())[4] == "not supported"                                         # 2/20
    study.results = [r for r in study.results if r["arm"] != "full" or r["event_id"] != "r0"]
    assert verdicts_of(study.report())[4] == "interim (19 of 20 planned items answered)"
