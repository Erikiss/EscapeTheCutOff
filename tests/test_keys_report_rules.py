"""Key, not value (docs/DESIGN_keys.md version 2, section 7): the pre-registered decision rules of the report, on the
scenarios of the review of version 1 (V1 to V7, robust-2). The rows are built directly (no backend)."""

from __future__ import annotations

import json
from typing import Any

from test_keys_report import Study, name

from cutoffmix.bench.keys import build_report, fisher_exact, render_report, wilson

SOL_NEON = ("neon", "sol")
FABLE_NEON = ("fable", "neon")


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
    three = report["predictions"][3]
    halves = three["halves"]
    # each half on its own rows: 3a is complete and keeps its verdict; 3b (7 of 7 adopted) shows no verdict at all
    interim = "interim (7 of 20 planned items answered)"
    assert halves["3b"]["verdict"] == halves["3b"]["comparison"]["verdict"] == interim
    assert halves["3a"]["verdict"] == halves["3a"]["comparison"]["verdict"] == "supported"
    assert "n=7, b=7, c=0, p=0.01562 [Wilson upper bound of b/n 1.000] -> interim (7 of 20 planned items answered)" \
        in three["evidence"]


def all_verdicts(node: Any) -> list[str]:
    """Every ``verdict`` in a part of a report: the predictions', their comparisons' and their halves'."""
    if isinstance(node, dict):
        return [v for k, v in node.items() if k == "verdict" and isinstance(v, str)] + \
            [x for k, v in node.items() if k != "verdict" for x in all_verdicts(v)]
    if isinstance(node, list):
        return [x for v in node for x in all_verdicts(v)]
    return []


VERDICT_WORDS = ("supported", "refuted", "inconclusive", "confounded")


def verdict_words(report: dict[str, Any]) -> list[str]:
    """The verdict words in the JSON (the key of prediction 1's withheld confound, null, aside) and the Markdown."""
    texts = (json.dumps(report).replace('"confounded": null', ""), render_report(report))
    return [w for text in texts for w in VERDICT_WORDS if w in text]


def test_an_interim_report_states_no_verdict_anywhere():
    # 12 of 96 dark items answered, every one recovered; parts of the adoption and both_right sets: no "supported" in
    # the parts either, in the JSON or the Markdown (no optional stopping: robust-2)
    study = Study().dark(96, correct={"key": range(96)}).adoption(30, full="endorsed").both_right(20)
    study.results = [r for r in study.results if int(r["event_id"][1:]) < {"dark": 12}.get(r["set"], 10)]
    report = study.report()
    assert {p["id"]: p["verdict"] for p in report["predictions"]} == {
        1: "interim (12 of 96 planned items answered)", "1b": "interim (12 of 96 planned items answered)",
        2: "interim (12 of 96 planned items answered)", 3: "interim (22 of 126 planned items answered)",
        4: "interim (10 of 20 planned items answered)"}
    parts = all_verdicts(report["predictions"])
    assert len(parts) == 12 and all(v.startswith("interim (") for v in parts)
    one = report["predictions"][0]
    assert one["comparisons"]["key_vs_alone"]["verdict"] == one["verdict"] and one["confounded"] is None
    assert "key vs alone n=12, b=12, c=0, p=0.0004883 [Wilson upper bound of b/n 1.000]" in one["evidence"]
    assert verdict_words(report) == []


def test_a_mixed_backend_report_states_no_verdict_anywhere():
    study = Study().dark(40, correct={"key": range(20), "blank": range(20)})          # confounded, were it stated
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent").both_right(20)
    for row in study.results[: len(study.results) // 2]:
        row["backend"] = "mock"
    report = study.report()
    mixed = "not testable (mixed backends: mock, openrouter)"
    parts = all_verdicts(report["predictions"])
    assert len(parts) == 12 and set(parts) == {mixed}
    assert report["predictions"][0]["confounded"] is None
    three = report["predictions"][3]
    assert f"[Wilson upper bound of b/n 0.648] -> {mixed}; 3b" in three["evidence"]
    assert verdict_words(report) == []


def test_a_simulated_report_marks_every_verdict_and_its_parts():
    study = Study().dark(40, correct={"key": range(20)})
    study.adoption(30, full=lambda i: "endorsed" if i < 15 else "absent").both_right(20)
    for row in study.results:
        row["backend"] = "mock"
    report = study.report()
    parts = all_verdicts(report["predictions"])
    assert len(parts) == 12 and all(v.startswith("simulated: ") for v in parts)
    three = report["predictions"][3]
    assert three["halves"]["3a"]["verdict"] == three["halves"]["3a"]["comparison"]["verdict"] == "simulated: supported"
    assert "[simulated: supported; Wilson upper bound of b/n 0.648] -> simulated: supported; 3b" in three["evidence"]
    assert "key vs alone n=40, b=20, c=0, p=1.907e-06 [simulated: supported; Wilson" in \
        report["predictions"][0]["evidence"]
    assert "gate: prediction 1 simulated: supported" in report["predictions"][2]["evidence"]


def test_a_complete_plan_without_recoveries_is_not_supported():
    v = verdicts_of(Study().dark(96).report())              # no arm is ever correct (full aside)
    assert wilson(0, 96)[1] < 0.10                          # the data rule out a gain of 10 %
    assert v[1] == v["1b"] == "not supported" and v[2] == "not testable (prediction 1 is not supported)"


def test_fewer_than_six_items_are_not_testable():
    v = verdicts_of(Study().dark(5, correct={"key": range(5)}).report())     # b = 5, c = 0: p = 0.0625 at best
    assert v[1] == v["1b"] == v[3] == "not testable (n too small)"


def test_between_support_and_refutation_is_inconclusive():
    # b = 3, c = 0: p = 0.25, upper bound 36 %; at that discordance (3 of 20) a gain of 0.10 needs about 71 items
    report = Study().dark(20, correct={"key": (0, 1, 2)}).report()
    v = verdicts_of(report)
    assert v[1] == v["1b"] == "inconclusive (20 items; about 71 would be needed for a gain of 0.10)"
    assert report["predictions"][0]["comparisons"]["key_vs_alone"]["items_needed"] == 71
    # no discordant pair: the fewest items at which a gain of 0.10 can be significant at all
    none_discordant = verdicts_of(Study().dark(20).report())[1]
    assert none_discordant == "inconclusive (20 items; about 51 would be needed for a gain of 0.10)"


def test_an_inconclusive_verdict_on_enough_items_says_so():
    # the decisive pair's 71 items: key vs alone b = 8, c = 3 (p = 0.23, upper bound 21 %). A gain of 0.10 at this
    # discordance needs 71 items, which were there: the data neither show nor rule it out (version 1 said "about 51
    # would be needed", fewer than were used)
    rest = range(60, 63)
    report = Study().dark(71, correct={"key": range(8), "alone": rest, "wrong_key": rest, "blank": rest}).report()
    one = report["predictions"][0]
    ka = one["comparisons"]["key_vs_alone"]
    assert (ka["n_items"], ka["b"], ka["c"], ka["items_needed"]) == (71, 8, 3, 71)
    assert one["verdict"] == verdicts_of(report)["1b"] == \
        "inconclusive (71 items; a gain of 0.10 is neither shown nor ruled out)"
    # 3b on the plan's 53 adoption items: 8 adoptions under full against 3 under alone
    three = keyed_dark(Study(), 116).adoption(53, full=lambda i: "endorsed" if i < 8 else "absent",
                                              alone=lambda i: "endorsed" if i >= 50 else "absent").report()
    assert three["predictions"][3]["halves"]["3b"]["verdict"] == \
        "inconclusive (53 items; a gain of 0.20 is neither shown nor ruled out)"


def test_a_key_that_hurts_is_not_supported():
    # alone (and the controls) right on 20 of 40 dark items, the key on none: c = 20, p < 0.05, but b > c fails
    report = Study().dark(40, correct={"alone": range(20), "wrong_key": range(20), "blank": range(20)}).report()
    ka = report["predictions"][0]["comparisons"]["key_vs_alone"]
    assert (ka["b"], ka["c"]) == (0, 20) and ka["p"] < 0.05
    v = verdicts_of(report)
    assert v[1] == v["1b"] == "not supported"


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


def test_each_prediction_reads_its_preregistered_scope():
    # two keyed pairs: fable->sol decides (40 dark items, 4 of them pointer items), fable->neon has 20; the key
    # recovers half of the items on both
    study = Study().dark(40, correct={"key": range(20)}, pointer=range(36, 40))
    study.dark(20, pair=FABLE_NEON, prefix="n", correct={"key": range(10)})
    report = study.report()
    v = verdicts_of(report)
    assert report["decisive_pair"] == name() and v[1] == v["1b"] == "supported"
    half = report["predictions"][3]["halves"]["3a"]
    assert half["verdict"] == "supported" and half["comparison"]["n_items"] == 60       # 3a pools the keyed pairs

    def without(**match: str) -> dict[Any, str]:
        kept = [r for r in study.results if any(r[k] != value for k, value in match.items())]
        return verdicts_of(build_report(study.plan, kept))

    # a row missing on the other pair: 1 and 1b (the decisive pair's dark set) keep their verdicts; 2 and 3a read the
    # dark sets of every keyed pair
    v = without(recipient="neon", event_id="n0", arm="alone")
    assert v[1] == v["1b"] == "supported"
    assert v[2] == v[3] == "interim (59 of 60 planned items answered)"
    # a missing alone row of a pointer item of the decisive pair: 1 and 1b read its whole dark set, pointer included
    v = without(recipient="sol", event_id="d36", arm="alone")
    assert v[1] == v["1b"] == "interim (39 of 40 planned items answered)"


def test_the_confound_is_read_on_the_items_without_pointer():
    # the blank beats the wrong key on the 8 pointer items only (b = 8, p = 0.008 over all items)
    report = Study().dark(40, correct={"key": range(20), "blank": range(32, 40)}, pointer=range(32, 40)).report()
    every = report["paired"][name()]["all"]["blank_vs_wrong_key"]
    assert (every["b"], every["c"]) == (8, 0) and every["p"] < 0.05
    one = report["predictions"][0]
    assert one["comparisons"]["blank_vs_wrong_key"]["b"] == 0
    assert one["verdict"] == "supported" and one["confounded"] is False


def test_3a_uses_the_margin_of_a_tenth():
    # full beats key on 5 of 60 dark items: the Wilson upper bound of b/n (18 %) rules out a gain of 0.20, not of 0.10
    half = Study().dark(60, correct={"full": range(5)}).report()["predictions"][3]["halves"]["3a"]
    assert (half["comparison"]["b"], half["comparison"]["c"]) == (5, 0) and 0.10 < half["comparison"]["upper"] < 0.20
    assert half["verdict"] == "inconclusive (60 items; a gain of 0.10 is neither shown nor ruled out)"


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


def test_predictions_1_1b_and_2_print_the_pointer_items_on_their_own():
    # design section 7: next to 1, 1b and 2, their numbers over all items and on the pointer items alone; here the 8
    # pointer items carry every key success (B resolves A's pointer)
    report = Study().dark(40, correct={"key": range(8)}, pointer=range(8), recognised=lambda i: i % 2 == 0).report()
    one, one_b, two = report["predictions"][:3]
    assert one["pointer_items"]["key_vs_alone"]["b"] == one["pointer_items"]["key_vs_wrong_key"]["b"] == 8
    assert "pointer items (B resolves A's pointer; 8 items): key vs alone n=8, b=8, c=0" in one["evidence"]
    pointed_b = one_b["pointer_items"]
    assert pointed_b["items"] == 8 and (pointed_b["key_vs_blank"]["b"], pointed_b["key_vs_blank"]["c"]) == (8, 0)
    assert (pointed_b["key"]["k"], pointed_b["blank"]["k"]) == (8, 0)
    assert "pointer items (B resolves A's pointer; 8 items): key vs blank n=8, b=8, c=0" in one_b["evidence"]
    assert one_b["all_items"]["key_vs_blank"]["b"] == 8 and one_b["comparisons"]["key_vs_blank"]["b"] == 0
    pointed = report["units"]["pointer_items"]
    assert pointed["units"] == two["pointer_items"]["units"] == 8 and "list" not in pointed
    assert (pointed["groups"]["recognised"]["b"], pointed["groups"]["unrecognised"]["b"]) == (4, 4)
    assert two["all_items"]["units"] == 40 and two["units"] == 32
    assert "pointer items on their own (B resolves A's pointer): 8 units (recognised 4, unrecognised 4), key vs " \
        "alone recognised b=4, c=0, unrecognised b=4, c=0" in two["evidence"]
    assert "The pointer items on their own (B resolves A's pointer): 8 units" in render_report(report)


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
    assert v["1b"] == "inconclusive (20 items; about 71 would be needed for a gain of 0.10)"
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
    # the gain sits on unrecognised units: (a) fails (recognised b = c = 0) and the statistic is negative
    v = verdicts_of(gain_study(4, 0, 2, 16).report())
    assert v[1] == "supported" and v[2].startswith("inconclusive (stratified difference of d -0.400")
    v = verdicts_of(gain_study(4, 16, 2, 0, n_unrec=9).report())
    assert v[1] == "supported" and v[2] == "not testable (n too small)"


def _two(report: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    two = report["predictions"][2]
    return two, two["groups"]["recognised"], two["tests"]


def test_prediction_2_needs_a_positive_statistic():
    # (a) holds (recognised b = 12, c = 0) and both statistics are significant, but negative: the key gains more on
    # unrecognised units (0.85 against 0.30)
    n = 40
    report = Study().dark(2 * n, correct={"alone": range(2), "key": set(range(14)) | set(range(n, n + 34))},
                          recognised=lambda i: i < n).report()
    two, rec, tests = _two(report)
    assert verdicts_of(report)[1] == "supported" and (rec["b"], rec["c"]) == (12, 0) and rec["p"] < 0.05
    assert tests["d"]["statistic"] < 0 and tests["d"]["p"] < 0.05
    assert tests["d_w"]["statistic"] < 0 and tests["d_w"]["p"] < 0.05
    assert two["verdict"] == "inconclusive (stratified difference of d -0.550, p=5e-05; of d_w -0.500, p=5e-05)"


def test_prediction_2_needs_the_same_on_d_w():
    # (a) and (b) hold, but against the wrong key the key gains 0.3 in both groups: an effect of any note is not
    # recovery (V2, BETTER_FIX point 6)
    n = 40
    report = Study().dark(2 * n, correct={"alone": set(range(4)) | set(range(n, n + 14)),
                                          "key": set(range(20)) | set(range(n, n + 14)),
                                          "wrong_key": set(range(8)) | {n, n + 1}}, recognised=lambda i: i < n).report()
    two, rec, tests = _two(report)
    assert verdicts_of(report)[1] == "supported" and (rec["b"], rec["c"]) == (16, 0)
    assert tests["d"]["difference"] == 0.4 and tests["d"]["p"] < 0.05
    assert (rec["mean_d_w"], two["groups"]["unrecognised"]["mean_d_w"]) == (0.3, 0.3)
    assert two["verdict"] == "inconclusive (stratified difference of d +0.400, p=5e-05; of d_w +0.000, p=1)"


def test_prediction_2_needs_recognised_units_to_gain():
    # two keyed pairs. fable->sol (decisive, 40 items) recovers 4 recognised and 4 unrecognised items: prediction 1
    # holds (b = 8, c = 0) and its stratum adds nothing. On fable->neon (30) the key loses 12 unrecognised items that
    # alone and the wrong key get right: (b) and (c) hold, but on recognised units the key gains on 5 only (p = 0.0625)
    study = Study().dark(40, correct={"key": {0, 1, 2, 3, 20, 21, 22, 23}}, recognised=lambda i: i < 20)
    study.dark(30, pair=FABLE_NEON, prefix="n", recognised=lambda i: i < 10,
               correct={"key": {0}, "alone": range(10, 22), "wrong_key": range(10, 22)})
    report = study.report()
    two, rec, tests = _two(report)
    assert report["decisive_pair"] == name() and verdicts_of(report)[1] == "supported"
    unrec = two["groups"]["unrecognised"]
    assert (rec["units"], unrec["units"], rec["b"], rec["c"], rec["p"]) == (30, 40, 5, 0, 0.0625)
    for statistic in ("d", "d_w"):
        assert tests[statistic]["statistic"] > 0 and tests[statistic]["p"] < 0.05
    assert two["verdict"] == "inconclusive (stratified difference of d +0.280, p=0.01805; of d_w +0.280, p=0.01805)"


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


def test_3b_counts_adopted_not_the_rule_label():
    # the judges decide reported speech (ENDORSE: adopted true), and a correct answer is never an adoption (V1)
    judged = {"grade": "incorrect", "adoption": "reported", "adopted": True, "adoption_grader": "judge:j"}
    three = keyed_dark(Study()).adoption(30, full=judged).report()["predictions"][3]
    assert three["halves"]["3b"]["comparison"]["b"] == 30 and three["halves"]["3b"]["verdict"] == "supported"
    named = {"grade": "correct", "adoption": "endorsed", "adopted": False, "adoption_grader": "rule"}
    three = keyed_dark(Study()).adoption(30, full=named).report()["predictions"][3]
    assert three["halves"]["3b"]["comparison"]["b"] == 0 and three["halves"]["3b"]["verdict"] == "not supported"


def test_the_undecided_share_counts_answered_full_rows():
    # 8 undecided of 35 answered full rows (5 refusals): 22.9 %, more than a fifth (of all 40 planned rows: 20 %)
    def full(i: int) -> Any:
        if i < 5:
            return {"grade": "ungraded", "grader": "refusal"}
        return "undecided" if i < 13 else "endorsed"

    three = keyed_dark(Study()).adoption(40, full=full).report()["predictions"][3]
    assert three["halves"]["3b"]["undecided"] == {"undecided": 8, "full_rows": 35, "share": 8 / 35}
    assert three["halves"]["3b"]["verdict"] == "not testable (reported speech undecided; grade with --judges)"


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
