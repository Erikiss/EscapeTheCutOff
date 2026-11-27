"""The report of the key experiment: the join of plan and results, the rates per pair, set and arm, the paired
comparisons and the verdicts on the pre-registered predictions (docs/DESIGN_keys.md version 2, section 6 "report" and
"Primary analysis", and section 7).

- Join: the last result row per id counts. A row without ``backend`` or ``prompt_sha`` has no provenance, a row whose
  ``prompt_sha`` differs from :func:`prompt_sha` of the plan row's messages is stale, and a row of an id the plan does
  not hold is foreign. All three are left out and counted; the cost of the foreign rows is reported on its own.
- Primary analysis: an answered row (no ``error``, grader not ``refusal``) counts, and ``ungraded`` counts as not
  correct. Refusals are left out and counted. The sensitivity line leaves the ungraded rows out as well.
- Costs: over every row of the results file (each line, not only the last per id), the recipients (``usage``) and the
  judges (``judge_usage``) apart.
- Backends: when every joined row comes from the mock, the report is SIMULATED and each verdict says so; when the
  joined rows come from more than one backend, it gives no verdicts.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from cutoffmix.bench.keys.common import ARMS_BY_SET, SETS, prompt_sha, row_cost
from cutoffmix.bench.keys.stats import (
    items_needed,
    mcnemar_exact,
    stratified_difference,
    stratified_permutation_test,
    wilson,
)

# ---- the pre-registered decision rules (design section 7)
ALPHA = 0.05  # two-sided
FLOOR_ITEMS = 6  # a paired comparison on fewer items is "not testable (n too small)"
KEY_MARGIN = 0.10  # predictions 1, 1b and 3a: the gain the data must rule out for a refutation
ADOPTION_MARGIN = 0.20  # prediction 3b ("far more often")
UNDECIDED_LIMIT = 0.20  # prediction 3b is not testable when a larger share of the full rows is undecided
MIN_UNITS = 10  # prediction 2: units per group (recognised, unrecognised)
PERMUTATIONS = 20_000  # prediction 2: the stratified permutation test ...
PERMUTATION_SEED = 0  # ... with a fixed seed
MIN_HARM_ITEMS = 10  # prediction 4: paired both_right items
HARM_LIMIT = 0.10  # prediction 4: the excess harm must be under 10 %
# ---- descriptive rules (design section 6)
UNGRADED_FLAG = 0.10  # an arm with a larger share of ungraded answered rows is flagged
MCQ_GUESS = 0.25  # four MCQ options: the guess-corrected recognised share is (s - 0.25) / 0.75
MOCK = "mock"
ADOPTION_SETS = ("donor_wrong", "control")
DARK_COMPARISONS: tuple[tuple[str, str], ...] = (("key", "alone"), ("key", "wrong_key"), ("key", "blank"),
                                                 ("blank", "alone"), ("blank", "wrong_key"), ("full", "key"))
DECISIVE_RULE = ("the keyed pair with the most planned dark items without pointer; ties go to the pair with more "
                 "planned dark items, then to the first pair name in alphabetical order")
UNIT_RULE = ("distinct (recipient, event) over the keyed pairs' dark items without pointer with alone, key and "
             "wrong_key answered; among those rows, an event dark in two pairs with the same recipient keeps the "
             "decisive pair's row, else the row of the pair with more planned dark items (then the first pair name)")
NOT_TESTABLE_N = "not testable (n too small)"
UNDECIDED_VERDICT = "not testable (reported speech undecided; grade with --judges)"
CONFOUNDED = "supported (confounded: the wrong key misleads)"
_ANSWERED, _REFUSAL, _ERROR, _STALE, _NO_PROVENANCE, _MISSING = (
    "answered", "refusal", "error", "stale", "no_provenance", "missing")
_COUNTED = {_MISSING: "missing", _NO_PROVENANCE: "no_provenance", _STALE: "stale", _ERROR: "errors",
            _REFUSAL: "refusals"}


# ------------------------------------------------------------------------------------------------------ the join


@dataclass(frozen=True)
class _Row:
    """A planned call and its current result row. ``status`` is ``answered``, ``refusal``, ``error``, ``stale``,
    ``no_provenance`` or ``missing`` (design section 6)."""

    plan: Mapping[str, Any]
    result: Mapping[str, Any] | None
    status: str

    def get(self, key: str) -> Any:
        return (self.result or {}).get(key)

    @property
    def answered(self) -> bool:
        return self.status == _ANSWERED

    @property
    def complete(self) -> bool:
        """A current row without error. A refusal is a final outcome, so it does not keep a prediction interim."""
        return self.status in (_ANSWERED, _REFUSAL)

    @property
    def grade(self) -> str:
        return str(self.get("grade") or "ungraded")

    @property
    def correct(self) -> bool:
        return self.answered and self.grade == "correct"

    @property
    def ungraded(self) -> bool:
        return self.answered and self.grade == "ungraded"

    @property
    def adopted(self) -> bool:
        return self.answered and self.get("adopted") is True

    @property
    def undecided(self) -> bool:
        return self.answered and self.get("adopted") is None

    @property
    def outcome(self) -> str:
        """The grade of an answered row, else its status."""
        return self.grade if self.answered else self.status


@dataclass
class _Item:
    """One planned item, (pair, set, event), with its rows per arm."""

    pair: str
    set_: str
    event_id: str
    recipient: str
    rows: dict[str, _Row] = field(default_factory=dict)

    @property
    def pointer(self) -> bool:
        """The key ties ``[withheld]`` to another work (design section 3); a missing or None flag is False."""
        return any(bool(r.plan.get("pointer")) for r in self.rows.values())

    @property
    def recognised(self) -> bool | None:
        for row in self.rows.values():
            if row.plan.get("recognised") is not None:
                return bool(row.plan.get("recognised"))
        return None

    @property
    def complete(self) -> bool:
        return all(r.complete for r in self.rows.values())

    def answered(self, *arms: str) -> bool:
        return all(a in self.rows and self.rows[a].answered for a in arms)


def _pair_name(row: Mapping[str, Any]) -> str:
    return f"{row.get('donor')}->{row.get('recipient')}"


def _status(result: Mapping[str, Any] | None, sha: str) -> str:
    if result is None:
        return _MISSING
    if not result.get("backend") or not result.get("prompt_sha"):
        return _NO_PROVENANCE
    if str(result.get("prompt_sha")) != sha:
        return _STALE
    if result.get("error"):
        return _ERROR
    if result.get("grader") == "refusal":
        return _REFUSAL
    return _ANSWERED


def _judge_cost(row: Mapping[str, Any]) -> float:
    """USD of a row's judge calls: :func:`row_cost` of each ``judge_usage`` entry (an entry ``{judge, purpose, cost,
    usage}`` as ``run`` writes it, or a bare usage mapping)."""
    entries = row.get("judge_usage") or []
    if isinstance(entries, Mapping):
        entries = [entries]
    total = 0.0
    for entry in entries:
        if isinstance(entry, Mapping):
            total += row_cost(entry) if "usage" in entry else row_cost({"usage": entry})
    return total


def _cost(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = list(rows)
    recipients = sum(row_cost(r) for r in chosen)
    judges = sum(_judge_cost(r) for r in chosen)
    return {"rows": len(chosen), "recipients_usd": round(recipients, 6), "judges_usd": round(judges, 6),
            "total_usd": round(recipients + judges, 6)}


# ------------------------------------------------------------------------------------------------- rates and tests


def _share_above(k: int, n: int, limit: float) -> bool:
    """``k / n > limit``, exactly (``limit`` read as the decimal it is written as)."""
    return n > 0 and Fraction(k, n) > Fraction(str(limit))


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"n": n, "k": k, "rate": (k / n) if n else None, "ci": [round(lo, 4), round(hi, 4)]}


def _is_correct(row: _Row) -> bool:
    return row.correct


def _is_ungraded(row: _Row) -> bool:
    return row.ungraded


def _arm_entry(rows: Sequence[_Row], success: Callable[[_Row], bool] = _is_correct) -> dict[str, Any]:
    """An arm's planned rows: the counts per status, the rate of ``success`` among the answered rows (the primary
    analysis: ungraded is not correct) with its Wilson interval, the same with the ungraded rows left out
    (``sensitivity``), and ``ungraded_flag`` when more than 10 % of the answered rows are ungraded."""
    answered = [r for r in rows if r.answered]
    graded = [r for r in answered if not r.ungraded]
    ungraded = len(answered) - len(graded)
    entry = _rate(sum(1 for r in answered if success(r)), len(answered))
    entry.update({"planned": len(rows), "answered": len(answered)})
    entry.update({key: sum(1 for r in rows if r.status == status) for status, key in _COUNTED.items()})
    entry.update({"ungraded": ungraded, "sensitivity": _rate(sum(1 for r in graded if success(r)), len(graded)),
                  "ungraded_flag": _share_above(ungraded, len(answered), UNGRADED_FLAG)})
    return entry


def _discordant(both: Sequence[tuple[_Row, _Row]], success: Callable[[_Row], bool]) -> dict[str, Any]:
    b = sum(1 for x, y in both if success(x) and not success(y))
    c = sum(1 for x, y in both if success(y) and not success(x))
    return {"n_items": len(both), "b": b, "c": c, "p": mcnemar_exact(b, c)}


def _compare(items: Iterable[_Item], first: str, second: str, success: Callable[[_Row], bool] = _is_correct,
             unsure: Callable[[_Row], bool] = _is_ungraded) -> dict[str, Any]:
    """The paired comparison of two arms on the items with both arms answered (design section 6): ``b`` = ``first``
    succeeds and ``second`` does not, ``c`` the reverse, ``p`` the exact two-sided McNemar test. ``sensitivity``: the
    same on the items where neither row is ``unsure`` (ungraded; undecided for adoption)."""
    both = [(it.rows[first], it.rows[second]) for it in items if it.answered(first, second)]
    entry = _discordant(both, success)
    entry["sensitivity"] = _discordant([(x, y) for x, y in both if not unsure(x) and not unsure(y)], success)
    return entry


def _three_way(comparison: Mapping[str, Any], margin: float) -> dict[str, Any]:
    """The verdict of design section 7 on "first beats second" with ``margin``: ``not testable (n too small)`` on fewer
    than 6 items, ``supported`` if b > c and p < 0.05, ``refuted`` if the 95 % Wilson upper bound of b / n is below the
    margin (the paired gain (b - c) / n is at most b / n), else ``inconclusive`` with the items a gain of the margin
    needs (:func:`items_needed`). The observed number of discordant pairs never decides testability."""
    n, b, c = int(comparison["n_items"]), int(comparison["b"]), int(comparison["c"])
    upper = wilson(b, n)[1]
    if n < FLOOR_ITEMS:
        verdict = NOT_TESTABLE_N
    elif b > c and mcnemar_exact(b, c) < ALPHA:
        verdict = "supported"
    elif upper < margin:
        verdict = "refuted"
    else:
        verdict = (f"inconclusive ({n} items; about {items_needed(margin, ALPHA)} would be needed for a gain of "
                   f"{margin:.2f})")
    return {**comparison, "margin": margin, "upper": round(upper, 4), "verdict": verdict}


def _combine(verdicts: Sequence[str]) -> str:
    """Several verdicts into one (design section 7): supported if all are, not supported if one is refuted, not
    testable if one is, else inconclusive (the first such verdict, with its numbers)."""
    if verdicts and all(v == "supported" for v in verdicts):
        return "supported"
    if any(v in ("refuted", "not supported") for v in verdicts):
        return "not supported"
    for wanted in ("not testable", "inconclusive"):
        for verdict in verdicts:
            if verdict.startswith(wanted):
                return verdict
    return "inconclusive"


def _completed(items: Sequence[_Item]) -> tuple[int, int]:
    """(items whose every planned row has a current row without error, planned items)."""
    return sum(1 for it in items if it.complete), len(items)


def _interim(done: int, planned: int) -> str | None:
    return None if done == planned else f"interim ({done} of {planned} planned items answered)"


def _mean(values: Sequence[int]) -> float | None:
    return sum(values) / len(values) if values else None


# ----------------------------------------------------------------------------------------------------- sections


def _arms_of(set_: str, items: Iterable[_Item]) -> list[str]:
    seen = list(dict.fromkeys(a for it in items for a in it.rows))
    known = [a for a in ARMS_BY_SET.get(set_, ()) if a in seen]
    return known + [a for a in seen if a not in known]


def _arms_table(set_: str, items: Sequence[_Item], success: Callable[[_Row], bool] = _is_correct) -> dict[str, Any]:
    return {arm: _arm_entry([it.rows[arm] for it in items if arm in it.rows], success)
            for arm in _arms_of(set_, items)}


def _dark_comparisons(items: Sequence[_Item]) -> dict[str, Any]:
    return {f"{x}_vs_{y}": _compare(items, x, y) for x, y in DARK_COMPARISONS}


def _recognition(items: Sequence[_Item]) -> dict[str, Any]:
    """The dark items split by ``recognised`` and the guess-corrected recognised share ``(s - 0.25) / 0.75``."""
    groups = {"recognised": [it for it in items if it.recognised is True],
              "unrecognised": [it for it in items if it.recognised is False]}
    known = len(groups["recognised"]) + len(groups["unrecognised"])
    share = len(groups["recognised"]) / known if known else None
    return {"items": len(items), "recognised": len(groups["recognised"]),
            "unrecognised": len(groups["unrecognised"]), "unknown": len(items) - known, "share": share,
            "guess_corrected": None if share is None else (share - MCQ_GUESS) / (1 - MCQ_GUESS),
            "groups": {label: _arms_table("dark", its) for label, its in groups.items()}}


def _adoption_arm(rows: Sequence[_Row]) -> dict[str, Any]:
    """Adoption of the donor's wrong name under one arm (design section 6): the labels of the answered rows, the
    undecided rows (``adopted`` None), and the ``adopted`` rate with undecided counted as not adopted (primary) and
    left out (``sensitivity``)."""
    answered = [r for r in rows if r.answered]
    labels = Counter(str(r.get("adoption")) for r in answered)
    undecided = sum(1 for r in answered if r.undecided)
    adopted = sum(1 for r in answered if r.adopted)
    entry = _rate(adopted, len(answered))
    entry.update({"planned": len(rows), "answered": len(answered), "endorsed": labels["endorsed"],
                  "reported": labels["reported"],
                  "reported_undecided": sum(1 for r in answered if r.get("adoption") == "reported" and r.undecided),
                  "absent": labels["absent"],
                  "unlabelled": len(answered) - labels["endorsed"] - labels["reported"] - labels["absent"],
                  "undecided": undecided, "sensitivity": _rate(adopted, len(answered) - undecided)})
    entry.update({key: sum(1 for r in rows if r.status == status) for status, key in _COUNTED.items()})
    return entry


def _adopted(row: _Row) -> bool:
    return row.adopted


def _undecided(row: _Row) -> bool:
    return row.undecided


def _not_correct(row: _Row) -> bool:
    return not row.correct


def _harm(items: Sequence[_Item]) -> dict[str, Any]:
    """``both_right``: the not-correct rate per arm and ``key`` against ``alone`` with b = alone correct and key not,
    c = key correct and alone not, and the excess harm (b - c) / n (design section 7, prediction 4)."""
    paired = _compare(items, "alone", "key")
    n = paired["n_items"]
    paired["excess_harm"] = (paired["b"] - paired["c"]) / n if n else None
    return {"items": len(items), "arms": _arms_table("both_right", items, _not_correct), "key_vs_alone": paired}


def _group(units: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    b = sum(1 for u in units if u["key"] and not u["alone"])
    c = sum(1 for u in units if u["alone"] and not u["key"])
    return {"units": len(units), "b": b, "c": c, "p": mcnemar_exact(b, c),
            "mean_d": _mean([u["d"] for u in units]), "mean_d_w": _mean([u["d_w"] for u in units]),
            **{arm: _rate(sum(1 for u in units if u[arm]), len(units)) for arm in ("alone", "key", "wrong_key")}}


def _units(dark: Mapping[str, Sequence[_Item]], priority: Callable[[str], tuple[Any, ...]],
           with_pointer: bool = False) -> dict[str, Any]:
    """The units of prediction 2 (design section 7, :data:`UNIT_RULE`) with d = [key correct] - [alone correct] and
    d_w = [key correct] - [wrong_key correct] (ungraded is not correct), per group and per pair and group.
    ``with_pointer``: the pointer items as well (the numbers over all items, printed next to the verdict)."""
    candidates: dict[tuple[str, str], list[_Item]] = {}
    for items in dark.values():
        for item in items:
            if (with_pointer or not item.pointer) and item.answered("alone", "key", "wrong_key"):
                candidates.setdefault((item.recipient, item.event_id), []).append(item)
    units: list[dict[str, Any]] = []
    for (recipient, event_id), items in candidates.items():
        kept = min(items, key=lambda it: priority(it.pair))
        key, alone, wrong = (kept.rows[arm].correct for arm in ("key", "alone", "wrong_key"))
        units.append({"pair": kept.pair, "recipient": recipient, "event_id": event_id,
                      "recognised": kept.recognised, "key": key, "alone": alone, "wrong_key": wrong,
                      "d": int(key) - int(alone), "d_w": int(key) - int(wrong)})
    known = [u for u in units if u["recognised"] is not None]
    groups = {"recognised": [u for u in known if u["recognised"]],
              "unrecognised": [u for u in known if not u["recognised"]]}
    pairs = list(dict.fromkeys(u["pair"] for u in units))
    share = len(groups["recognised"]) / len(known) if known else None
    return {"units": len(units), "unknown": len(units) - len(known),
            "duplicates": sum(len(items) - 1 for items in candidates.values()),
            "share": share, "guess_corrected": None if share is None else (share - MCQ_GUESS) / (1 - MCQ_GUESS),
            "groups": {label: _group(us) for label, us in groups.items()},
            "by_pair": {pair: {label: _group([u for u in us if u["pair"] == pair]) for label, us in groups.items()}
                        for pair in pairs},
            "d": stratified_difference([(u["pair"], u["recognised"], u["d"]) for u in known]),
            "d_w": stratified_difference([(u["pair"], u["recognised"], u["d_w"]) for u in known]),
            "list": units}


# -------------------------------------------------------------------------------------------------- predictions


def _fmt_rate(entry: Mapping[str, Any] | None) -> str:
    if not entry or not entry.get("n"):
        return "n/a"
    lo, hi = entry["ci"]
    return f"{entry['k']}/{entry['n']} ({100 * entry['rate']:.1f} %, {100 * lo:.1f}-{100 * hi:.1f})"


def _fmt_test(entry: Mapping[str, Any] | None) -> str:
    if not entry:
        return "n/a"
    return f"n={entry['n_items']}, b={entry['b']}, c={entry['c']}, p={entry['p']:.4g}"


def _fmt_verdict(entry: Mapping[str, Any]) -> str:
    return f"{_fmt_test(entry)} [{entry['verdict']}; Wilson upper bound of b/n {entry['upper']:.3f}]"


def _fmt_number(value: float | None, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"


def _verb(verdict: str) -> str:
    """A verdict without its parenthesis: "inconclusive (12 items; ...)" -> "inconclusive"."""
    return verdict.split(" (", 1)[0]


@dataclass
class _Context:
    """What the predictions read: the items per (pair, set), the decisive and keyed pairs."""

    items: dict[tuple[str, str], list[_Item]]
    pairs: list[str]
    keyed: list[str]
    decisive: str | None
    planned_dark: dict[str, dict[str, int]]

    def of(self, set_: str, pairs: Iterable[str] | None = None) -> list[_Item]:
        chosen = self.pairs if pairs is None else list(pairs)
        return [it for pair in chosen for it in self.items.get((pair, set_), [])]


def _prediction_1(ctx: _Context) -> tuple[dict[str, Any], dict[str, Any]]:
    """Predictions 1 and 1b on the decisive pair's dark items without pointer (design section 7)."""
    pair = ctx.decisive
    every = ctx.of("dark", [pair] if pair else [])
    plain = [it for it in every if not it.pointer]
    pointed = [it for it in every if it.pointer]
    interim = _interim(*_completed(every))
    tests = _dark_comparisons(plain)
    ka = _three_way(tests["key_vs_alone"], KEY_MARGIN)
    kw = _three_way(tests["key_vs_wrong_key"], KEY_MARGIN)
    kb = _three_way(tests["key_vs_blank"], KEY_MARGIN)
    bw = tests["blank_vs_wrong_key"]
    confounded = bw["b"] > bw["c"] and mcnemar_exact(bw["b"], bw["c"]) < ALPHA
    verdict = interim or _combine([ka["verdict"], kw["verdict"]])
    if verdict == "supported" and confounded:
        verdict = CONFOUNDED
    verdict_b = interim or _combine([kb["verdict"]])
    all_tests = _dark_comparisons(every)
    arms = _arms_table("dark", plain)
    pointer_arms = _arms_table("dark", pointed)
    if pair is None:
        scope = "no keyed pair in the plan (no dark items)"
    else:
        counts = ctx.planned_dark[pair]
        scope = (f"decisive pair {pair} ({counts['no_pointer']} planned dark items without pointer, "
                 f"{counts['all']} in all)")
    evidence = (f"{scope}; without pointer: key vs alone {_fmt_verdict(ka)}; key vs wrong key {_fmt_verdict(kw)}; "
                f"blank vs alone {_fmt_test(tests['blank_vs_alone'])} (the note's presence); blank vs wrong key "
                f"{_fmt_test(bw)} (the harm of a mismatched cue{', significant: confounded' if confounded else ''}); "
                f"correct: key {_fmt_rate(arms.get('key'))}, alone {_fmt_rate(arms.get('alone'))}, wrong key "
                f"{_fmt_rate(arms.get('wrong_key'))}, blank {_fmt_rate(arms.get('blank'))}; all items: key vs alone "
                f"{_fmt_test(all_tests['key_vs_alone'])}, key vs wrong key {_fmt_test(all_tests['key_vs_wrong_key'])}; "
                f"pointer items (B resolves A's pointer): key {_fmt_rate(pointer_arms.get('key'))}, wrong key "
                f"{_fmt_rate(pointer_arms.get('wrong_key'))}; expected: key 10-30 %, wrong key under 10 %")
    evidence_b = (f"{scope}; without pointer: key vs blank {_fmt_verdict(kb)}; correct: key "
                  f"{_fmt_rate(arms.get('key'))}, blank {_fmt_rate(arms.get('blank'))}; all items: key vs blank "
                  f"{_fmt_test(all_tests['key_vs_blank'])}")
    one = {"id": 1, "text": "On dark items without pointer of the decisive pair, key beats alone and wrong_key "
                            "(paired exact McNemar, margin 0.10).",
           "verdict": verdict, "evidence": evidence, "pair": pair, "items": len(plain),
           "comparisons": {"key_vs_alone": ka, "key_vs_wrong_key": kw, "blank_vs_alone": tests["blank_vs_alone"],
                           "blank_vs_wrong_key": bw},
           "confounded": confounded, "all_items": {k: all_tests[k] for k in ("key_vs_alone", "key_vs_wrong_key")},
           "pointer_items": {"items": len(pointed), "key": pointer_arms.get("key"),
                             "wrong_key": pointer_arms.get("wrong_key")}}
    one_b = {"id": "1b", "text": "key beats blank on the same items: the donor's content, not the presence of a "
                                 "note, does the work (margin 0.10).",
             "verdict": verdict_b, "evidence": evidence_b, "pair": pair, "items": len(plain),
             "comparisons": {"key_vs_blank": kb}, "all_items": {"key_vs_blank": all_tests["key_vs_blank"]}}
    return one, one_b


def _prediction_2(ctx: _Context, units: dict[str, Any], gate: str, every: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 2 (design section 7): the key's gain over ``alone`` is larger on recognised units; gated on
    prediction 1 and at least 10 units per group; supported or inconclusive, never refuted. ``every``: the units with
    the pointer items, printed next to it."""
    interim = _interim(*_completed(ctx.of("dark", ctx.keyed)))
    groups = units["groups"]
    rec, unrec = groups["recognised"], groups["unrecognised"]
    tested: dict[str, Any] = {}
    if interim:
        verdict = interim
    elif not gate.startswith("supported"):
        verdict = f"not testable (prediction 1 is {_verb(gate)})"
    elif rec["units"] < MIN_UNITS or unrec["units"] < MIN_UNITS:
        verdict = NOT_TESTABLE_N
    else:
        known = [u for u in units["list"] if u["recognised"] is not None]
        for name in ("d", "d_w"):
            tested[name] = stratified_permutation_test([(u["pair"], u["recognised"], u[name]) for u in known],
                                                       permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
            units[name] = tested[name]
        a = rec["b"] > rec["c"] and mcnemar_exact(rec["b"], rec["c"]) < ALPHA
        gain_d, gain_dw = ((tested[n]["statistic"] > 0 and tested[n]["p"] < ALPHA) for n in ("d", "d_w"))
        if a and gain_d and gain_dw:
            verdict = "supported"
        else:
            verdict = (f"inconclusive (stratified difference of d {_fmt_number(tested['d']['difference'], sign=True)}"
                       f", p={tested['d']['p']:.4g}; of d_w {_fmt_number(tested['d_w']['difference'], sign=True)}, "
                       f"p={tested['d_w']['p']:.4g})")
    d, dw = units["d"], units["d_w"]
    p_d = f", p={d['p']:.4g}" if "p" in d else ""
    p_dw = f", p={dw['p']:.4g}" if "p" in dw else ""
    evidence = (f"{units['units']} units (recognised {rec['units']}, unrecognised {unrec['units']}, unknown "
                f"{units['unknown']}; {units['duplicates']} rows of an event already kept for the same recipient left "
                f"out); recognised: key vs alone b={rec['b']}, c={rec['c']}, p={rec['p']:.4g}; mean d recognised "
                f"{_fmt_number(rec['mean_d'])}, unrecognised {_fmt_number(unrec['mean_d'])}; mean d_w recognised "
                f"{_fmt_number(rec['mean_d_w'])}, unrecognised {_fmt_number(unrec['mean_d_w'])}; stratified "
                f"difference of d {_fmt_number(d['difference'], sign=True)}{p_d}; of d_w "
                f"{_fmt_number(dw['difference'], sign=True)}{p_dw}; alone correct: recognised "
                f"{_fmt_rate(rec['alone'])}, unrecognised {_fmt_rate(unrec['alone'])}; guess-corrected recognised "
                f"share {_fmt_number(units['guess_corrected'])}; all items (pointer included): {every['units']} units, "
                f"recognised key vs alone b={every['groups']['recognised']['b']}, "
                f"c={every['groups']['recognised']['c']}, stratified difference of d "
                f"{_fmt_number(every['d']['difference'], sign=True)}, of d_w "
                f"{_fmt_number(every['d_w']['difference'], sign=True)}; gate: prediction 1 {_verb(gate)}")
    return {"id": 2, "text": "The key's gain over alone is larger on recognised than on unrecognised units (distinct "
                             "units, permutation test stratified by pair; gated on prediction 1).",
            "verdict": verdict, "evidence": evidence, "units": units["units"],
            "groups": {"recognised": rec, "unrecognised": unrec}, "tests": tested}


def _prediction_3(ctx: _Context, pooled: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 3 (design section 7): 3a full vs key pooled over the keyed pairs' dark items (margin 0.10), 3b full
    vs alone on ``adopted`` True pooled over donor_wrong and control of every pair (margin 0.20)."""
    done_a, planned_a = _completed(ctx.of("dark", ctx.keyed))
    adoption_items = [it for s in ADOPTION_SETS for it in ctx.of(s)]
    done_b, planned_b = _completed(adoption_items)
    a = _three_way(pooled["full_vs_key"], KEY_MARGIN)
    b = _three_way(pooled["adoption_full_vs_alone"], ADOPTION_MARGIN)
    undecided = pooled["adoption_full_undecided"]
    half_a = _interim(done_a, planned_a) or _combine([a["verdict"]])
    interim_b = _interim(done_b, planned_b)
    if interim_b:
        half_b = interim_b
    elif b["verdict"] == NOT_TESTABLE_N:
        half_b = NOT_TESTABLE_N
    elif _share_above(undecided["undecided"], undecided["full_rows"], UNDECIDED_LIMIT):
        half_b = UNDECIDED_VERDICT
    else:
        half_b = _combine([b["verdict"]])
    if half_a.startswith("interim") or half_b.startswith("interim"):
        verdict = f"interim ({done_a + done_b} of {planned_a + planned_b} planned items answered)"
    else:
        verdict = _combine([half_a, half_b])
    share = undecided["share"]
    evidence = (f"3a full vs key, the keyed pairs' dark items: {_fmt_verdict(a)} -> {half_a}; 3b adoption (adopted "
                f"true) full vs alone, donor_wrong and control of every pair: {_fmt_verdict(b)} -> {half_b}; full "
                f"rows undecided {undecided['undecided']} of {undecided['full_rows']}"
                f" ({'n/a' if share is None else f'{100 * share:.1f} %'}, limit 20 %); adopted: full "
                f"{_fmt_rate(pooled['adoption_arms'].get('full'))}, alone "
                f"{_fmt_rate(pooled['adoption_arms'].get('alone'))}")
    return {"id": 3, "text": "full beats key on dark items (3a, margin 0.10), and full makes the recipient endorse a "
                             "wrong donor answer far more often than alone (3b, adopted true, margin 0.20).",
            "verdict": verdict, "evidence": evidence,
            "halves": {"3a": {"verdict": half_a, "comparison": a, "answered": done_a, "planned": planned_a},
                       "3b": {"verdict": half_b, "comparison": b, "answered": done_b, "planned": planned_b,
                              "undecided": undecided}}}


def _prediction_4(ctx: _Context, pooled: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 4 (design section 7): on both_right, the excess harm (b - c) / n of key over alone is under 10 %."""
    interim = _interim(*_completed(ctx.of("both_right")))
    harm = pooled["both_right"]
    paired = harm["key_vs_alone"]
    n, excess = paired["n_items"], paired["excess_harm"]
    if interim:
        verdict = interim
    elif n < MIN_HARM_ITEMS:
        verdict = NOT_TESTABLE_N
    elif Fraction(paired["b"] - paired["c"], n) < Fraction(str(HARM_LIMIT)):
        verdict = "supported"
    else:
        verdict = "not supported"
    arms = harm["arms"]
    evidence = (f"both_right, all pairs: {n} items with alone and key answered; b (alone correct, key not)="
                f"{paired['b']}, c (key correct, alone not)={paired['c']}, p={paired['p']:.4g}; excess harm "
                f"(b - c) / n {'n/a' if excess is None else f'{100 * excess:.1f} %'} (under 10 % supports; at least 10 "
                f"items); "
                f"not correct: alone {_fmt_rate(arms.get('alone'))} (re-ask noise), key {_fmt_rate(arms.get('key'))}, "
                f"full {_fmt_rate(arms.get('full'))}")
    return {"id": 4, "text": "The notes rarely break a correct recall: on both_right, the excess harm of key over "
                             "re-asking alone is under 10 %.",
            "verdict": verdict, "evidence": evidence, "items": n, "excess_harm": excess, "comparison": paired}


# ------------------------------------------------------------------------------------------------------- report


def build_report(plan: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Join plan and results into the numbers of design section 6 and the verdicts of section 7 (a JSON-serialisable
    dict).

    Top-level keys: ``totals`` (the join's counts, the backends and the costs), ``backends``, ``simulated``,
    ``mixed_backends``, ``decisive_pair``, ``keyed_pairs``, ``planned_dark``, ``pairs`` (per pair, set and arm: the
    counts and the correct rate, primary and ``sensitivity``), ``paired`` (the dark comparisons per pair over ``all``
    items, ``no_pointer`` items and ``pointer`` items), ``pointer_items``, ``recognised`` (the dark items by
    recognition, ``all`` and ``no_pointer``), ``units`` (prediction 2), ``adoption`` (per pair, set and arm, and full
    vs alone), ``harm`` (both_right per pair), ``pooled`` (the comparisons of predictions 3 and 4), ``rules``,
    ``flags`` and ``predictions`` (ids 1, "1b", 2, 3, 4, each with ``text``, ``verdict`` and ``evidence``)."""
    planned: dict[str, Mapping[str, Any]] = {}
    for row in plan:
        planned[str(row.get("id"))] = row
    last: dict[str, Mapping[str, Any]] = {}
    for row in results:
        last[str(row.get("id"))] = row

    # the join: every planned id gets a row with its status
    rows: dict[str, _Row] = {}
    items: dict[tuple[str, str], dict[str, _Item]] = {}
    for id_, p in planned.items():
        result = last.get(id_)
        row = _Row(p, result, _status(result, prompt_sha(p.get("messages") or [])))
        rows[id_] = row
        pair, set_, event_id = _pair_name(p), str(p.get("set")), str(p.get("event_id"))
        item = items.setdefault((pair, set_), {}).setdefault(
            event_id, _Item(pair, set_, event_id, str(p.get("recipient"))))
        item.rows[str(p.get("arm"))] = row
    joined = [r for r in rows.values() if r.status in (_ANSWERED, _REFUSAL, _ERROR)]
    statuses = Counter(r.status for r in rows.values())
    foreign_rows = [r for r in results if str(r.get("id")) not in planned]
    every_cost = _cost(results)
    backends = dict(sorted(Counter(str(r.get("backend")) for r in joined).items()))
    simulated = bool(backends) and set(backends) == {MOCK}
    mixed = len(backends) > 1
    totals = {
        "planned": len(planned), "plan_rows": len(plan), "result_rows": len(results),
        "superseded_rows": len(results) - len(last), "joined": len(joined),
        **{key: statuses[status] for status, key in _COUNTED.items()},
        "answered": statuses[_ANSWERED], "ungraded": sum(1 for r in rows.values() if r.ungraded),
        "foreign_ids": len({str(r.get("id")) for r in foreign_rows}), "foreign_rows": len(foreign_rows),
        "backends": backends,
        "by_recipient": dict(Counter(str(r.plan.get("recipient")) for r in joined)),
        "cost_usd": every_cost["total_usd"],
        "cost": {"all_rows": every_cost, "foreign": _cost(foreign_rows),
                 "joined": _cost(r.result for r in joined if r.result is not None)},
    }

    # the sets per pair
    pair_names = list(dict.fromkeys(_pair_name(p) for p in planned.values()))
    by: dict[tuple[str, str], list[_Item]] = {k: list(v.values()) for k, v in items.items()}
    set_order = list(SETS) + [s for s in dict.fromkeys(k[1] for k in by) if s not in SETS]
    planned_dark = {pair: {"all": len(by.get((pair, "dark"), [])),
                           "no_pointer": sum(1 for it in by.get((pair, "dark"), []) if not it.pointer)}
                    for pair in pair_names}
    keyed = [pair for pair in pair_names if planned_dark[pair]["all"]]
    decisive = min(keyed, key=lambda p: (-planned_dark[p]["no_pointer"], -planned_dark[p]["all"], p)) if keyed \
        else None
    ctx = _Context(items=by, pairs=pair_names, keyed=keyed, decisive=decisive, planned_dark=planned_dark)

    def priority(pair: str) -> tuple[Any, ...]:
        return pair != decisive, -planned_dark[pair]["all"], pair

    pairs: dict[str, dict[str, Any]] = {}
    paired: dict[str, dict[str, Any]] = {}
    pointer_items: dict[str, dict[str, Any]] = {}
    recognised: dict[str, dict[str, Any]] = {}
    adoption: dict[str, dict[str, Any]] = {}
    harm: dict[str, dict[str, Any]] = {}
    for pair in pair_names:
        for set_ in set_order:
            its = by.get((pair, set_))
            if its:
                pairs.setdefault(pair, {})[set_] = {"items": len(its), "arms": _arms_table(set_, its)}
        dark = by.get((pair, "dark"), [])
        if dark:
            plain = [it for it in dark if not it.pointer]
            pointed = [it for it in dark if it.pointer]
            paired[pair] = {"items": {"all": len(dark), "no_pointer": len(plain), "pointer": len(pointed)},
                            "all": _dark_comparisons(dark), "no_pointer": _dark_comparisons(plain),
                            "pointer": _dark_comparisons(pointed)}
            recognised[pair] = {"all": _recognition(dark), "no_pointer": _recognition(plain)}
            if pointed:
                pointer_items[pair] = {
                    "items": len(pointed), "arms": _arms_table("dark", pointed),
                    "list": [{"event_id": it.event_id, "recognised": it.recognised,
                              "outcomes": {arm: row.outcome for arm, row in it.rows.items()},
                              "related_known": next((r.plan.get("related_known") for r in it.rows.values()
                                                     if r.plan.get("related_known")), None)} for it in pointed]}
        for set_ in ADOPTION_SETS:
            its = by.get((pair, set_))
            if its:
                adoption.setdefault(pair, {})[set_] = {
                    "items": len(its),
                    "arms": {arm: _adoption_arm([it.rows[arm] for it in its if arm in it.rows])
                             for arm in _arms_of(set_, its)},
                    "full_vs_alone": _compare(its, "full", "alone", _adopted, _undecided)}
        both = by.get((pair, "both_right"))
        if both:
            harm[pair] = _harm(both)

    adoption_items = [it for s in ADOPTION_SETS for it in ctx.of(s)]
    full_rows = [it.rows["full"] for it in adoption_items if "full" in it.rows and it.rows["full"].answered]
    undecided = sum(1 for r in full_rows if r.undecided)
    pooled = {
        "full_vs_key": _compare(ctx.of("dark", keyed), "full", "key"),
        "adoption_full_vs_alone": _compare(adoption_items, "full", "alone", _adopted, _undecided),
        "adoption_full_undecided": {"undecided": undecided, "full_rows": len(full_rows),
                                    "share": undecided / len(full_rows) if full_rows else None},
        "adoption_arms": {arm: _adoption_arm([it.rows[arm] for it in adoption_items if arm in it.rows])
                          for arm in ("alone", "full")},
        "both_right": _harm(ctx.of("both_right")),
    }
    keyed_dark = {pair: by.get((pair, "dark"), []) for pair in keyed}
    units = _units(keyed_dark, priority)
    units_every = _units(keyed_dark, priority, with_pointer=True)
    units["all_items"] = {k: v for k, v in units_every.items() if k != "list"}

    one, one_b = _prediction_1(ctx)
    two = _prediction_2(ctx, units, one["verdict"], units_every)
    predictions = [one, one_b, two, _prediction_3(ctx, pooled), _prediction_4(ctx, pooled)]
    for entry in predictions:
        if mixed:
            entry["verdict"] = f"not testable (mixed backends: {', '.join(backends)})"
        elif simulated:
            entry["verdict"] = f"simulated: {entry['verdict']}"

    report: dict[str, Any] = {
        "totals": totals, "backends": backends, "simulated": simulated, "mixed_backends": mixed,
        "decisive_pair": decisive, "keyed_pairs": keyed, "planned_dark": planned_dark, "pairs": pairs,
        "paired": paired, "pointer_items": pointer_items, "recognised": recognised,
        "units": units, "adoption": adoption, "harm": harm,
        "pooled": pooled,
        "rules": {"alpha": ALPHA, "floor_items": FLOOR_ITEMS, "key_margin": KEY_MARGIN,
                  "adoption_margin": ADOPTION_MARGIN, "undecided_limit": UNDECIDED_LIMIT, "min_units": MIN_UNITS,
                  "permutations": PERMUTATIONS, "permutation_seed": PERMUTATION_SEED,
                  "min_harm_items": MIN_HARM_ITEMS, "harm_limit": HARM_LIMIT, "ungraded_flag": UNGRADED_FLAG,
                  "decisive_pair": DECISIVE_RULE, "units": UNIT_RULE},
    }
    report["flags"] = _flags(report, rows, len(plan) - len(planned))
    report["predictions"] = predictions
    return report


def _flags(report: Mapping[str, Any], rows: Mapping[str, _Row], repeated: int) -> list[str]:
    """What a reader must see before the numbers (design section 6): the backends, rows left out, arms with more than
    10 % ungraded rows, planned rows not answered yet."""
    t = report["totals"]
    flags: list[str] = []
    backends = ", ".join(f"{name} {n}" for name, n in report["backends"].items())
    if report["mixed_backends"]:
        flags.append(f"the joined rows come from more than one backend ({backends}): no verdicts")
    elif report["simulated"]:
        flags.append("SIMULATED: every joined row comes from the mock backend; its numbers say nothing about the real "
                     "models")
    if repeated:
        flags.append(f"the plan repeats {repeated} ids; the last plan row per id counts")
    if t["stale"]:
        flags.append(f"{t['stale']} stale rows (their prompt_sha differs from the plan's messages) are left out")
    if t["no_provenance"]:
        flags.append(f"{t['no_provenance']} rows without provenance (no backend or prompt_sha) are left out")
    if t["foreign_rows"]:
        flags.append(f"{t['foreign_rows']} rows of {t['foreign_ids']} ids not in the plan are left out (their cost: "
                     f"{t['cost']['foreign']['total_usd']:.4f} USD)")
    if t["refusals"]:
        flags.append(f"{t['refusals']} refusals are left out")
    open_rows = sum(1 for r in rows.values() if not r.complete)
    if open_rows:
        flags.append(f"{open_rows} planned rows are not answered (missing {t['missing']}, errors {t['errors']}, stale "
                     f"{t['stale']}, without provenance {t['no_provenance']}): the predictions that use them are "
                     f"interim")
    for pair, sets in report["pairs"].items():
        for set_, entry in sets.items():
            for arm, a in entry["arms"].items():
                if a["ungraded_flag"]:
                    flags.append(f"{pair} {set_} {arm}: {a['ungraded']} of {a['answered']} answered rows are ungraded "
                                 f"({100 * a['ungraded'] / a['answered']:.1f} %, more than 10 %; grade with --judges)")
    return flags


# ------------------------------------------------------------------------------------------------------- render


def _cell_status(entry: Mapping[str, Any]) -> list[str]:
    ungraded = f"{entry['ungraded']}" + (" (>10 %)" if entry.get("ungraded_flag") else "")
    return [str(entry["planned"]), str(entry["answered"]), str(entry["missing"]), str(entry["errors"]),
            str(entry["refusals"]), ungraded, str(entry["stale"]), str(entry["no_provenance"])]


def _related(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return "; ".join(", ".join(f"{k} {v}" for k, v in entry.items()) if isinstance(entry, Mapping)
                         else str(entry) for entry in value)
    return str(value)


def render_report(report: Mapping[str, Any]) -> str:
    """The report as Markdown: a SIMULATED (or mixed backends) banner first when it applies, the totals and costs,
    the flags, the rates per pair, set and arm (primary and sensitivity), the paired comparisons, the pointer items,
    the split by recognition, the units of prediction 2, adoption, harm, and the predictions with their evidence."""
    t = report["totals"]
    cost = t["cost"]
    lines: list[str] = []
    if report.get("simulated"):
        lines += ["**SIMULATED**: every joined row comes from the mock backend (docs/DESIGN_keys.md section 8); the "
                  "numbers say nothing about the real models, and every verdict is marked simulated.", ""]
    if report.get("mixed_backends"):
        lines += [f"**MIXED BACKENDS** ({', '.join(f'{k} {v}' for k, v in report['backends'].items())}): the report "
                  "gives no verdicts.", ""]
    backends = ", ".join(f"{k} {v}" for k, v in report["backends"].items()) or "none"
    lines += ["# Key, not value: report", "",
              f"Planned calls {t['planned']}; joined {t['joined']} (answered {t['answered']}, of them ungraded "
              f"{t['ungraded']}; refusals {t['refusals']}; errors {t['errors']}); missing {t['missing']}, stale "
              f"{t['stale']}, without provenance {t['no_provenance']}. Result rows {t['result_rows']} (superseded "
              f"{t['superseded_rows']}); rows of ids not in the plan {t['foreign_rows']} ({t['foreign_ids']} ids). "
              f"Backends of the joined rows: {backends}.", "",
              f"Cost over every row of the results file: recipients {cost['all_rows']['recipients_usd']:.4f} USD, "
              f"judges {cost['all_rows']['judges_usd']:.4f} USD, total {cost['all_rows']['total_usd']:.4f} USD "
              f"(of it, ids not in the plan: {cost['foreign']['total_usd']:.4f} USD). The joined rows: "
              f"{cost['joined']['total_usd']:.4f} USD (recipients {cost['joined']['recipients_usd']:.4f}, judges "
              f"{cost['joined']['judges_usd']:.4f}).", ""]
    decisive = report.get("decisive_pair")
    if decisive:
        counts = report["planned_dark"][decisive]
        lines += [f"Decisive pair (section 7): {decisive}, {counts['no_pointer']} planned dark items without pointer "
                  f"({counts['all']} in all); {report['rules']['decisive_pair']}.", ""]
    else:
        lines += ["Decisive pair (section 7): none (no keyed pair in the plan).", ""]
    if report["flags"]:
        lines += ["## Flags", ""] + [f"- {flag}" for flag in report["flags"]] + [""]

    lines += ["## Answers per pair, set and arm", "",
              "Primary: answered rows (no error, no refusal), ungraded counted as not correct. Sensitivity: ungraded "
              "rows left out.", "",
              "| pair | set | arm | planned | answered | missing | errors | refusals | ungraded | stale | "
              "no provenance | correct (primary) | correct (sensitivity) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for pair, sets in report["pairs"].items():
        for set_, entry in sets.items():
            for arm, a in entry["arms"].items():
                lines.append(f"| {pair} | {set_} | {arm} | " + " | ".join(_cell_status(a))
                             + f" | {_fmt_rate(a)} | {_fmt_rate(a['sensitivity'])} |")

    lines += ["", "## Paired comparisons on dark items", "",
              "Exact two-sided McNemar on the items with both arms answered; b = first arm correct and second not, "
              "c the reverse.", "",
              "| pair | comparison | all items | without pointer | pointer items |", "|---|---|---|---|---|"]
    for pair, entry in report["paired"].items():
        for x, y in DARK_COMPARISONS:
            name = f"{x}_vs_{y}"
            lines.append(f"| {pair} | {x} vs {y} | {_fmt_test(entry['all'][name])} | "
                         f"{_fmt_test(entry['no_pointer'][name])} | {_fmt_test(entry['pointer'][name])} |")

    lines += ["", "## Pointer items (B resolves A's pointer; kept out of predictions 1, 1b and 2)", ""]
    if report["pointer_items"]:
        lines += ["| pair | event | recognised | alone | key | wrong_key | blank | "
                  "related (recipient's stored grades) |", "|---|---|---|---|---|---|---|---|"]
        for pair, entry in report["pointer_items"].items():
            for it in entry["list"]:
                o = it["outcomes"]
                lines.append(f"| {pair} | {it['event_id']} | {it['recognised']} | {o.get('alone', '')} | "
                             f"{o.get('key', '')} | {o.get('wrong_key', '')} | {o.get('blank', '')} | "
                             f"{_related(it['related_known'])} |")
            arms = entry["arms"]
            lines.append(f"| {pair} | all {entry['items']} | | {_fmt_rate(arms.get('alone'))} | "
                         f"{_fmt_rate(arms.get('key'))} | {_fmt_rate(arms.get('wrong_key'))} | "
                         f"{_fmt_rate(arms.get('blank'))} | |")
    else:
        lines.append("No pointer items in the plan.")

    dark_arms = ARMS_BY_SET["dark"]
    lines += ["", "## Dark items by recognition (the recipient passes the MCQ)", "",
              "| pair | scope | group | items | " + " | ".join(dark_arms) + " |",
              "|---|---|---|---:|" + "---|" * len(dark_arms)]
    shares: list[str] = []
    for pair, scopes in report["recognised"].items():
        for scope, entry in scopes.items():
            for label, group in entry["groups"].items():
                n = entry[label]
                lines.append(f"| {pair} | {scope.replace('_', ' ')} | {label} | {n} | "
                             + " | ".join(_fmt_rate(group.get(a)) for a in dark_arms) + " |")
            shares.append(f"- {pair}, {scope.replace('_', ' ')}: recognised share "
                          f"{_fmt_number(entry['share'])} of {entry['recognised'] + entry['unrecognised']} items "
                          f"(unknown {entry['unknown']}), guess-corrected (s - 0.25) / 0.75 = "
                          f"{_fmt_number(entry['guess_corrected'])}")
    lines += [""] + shares

    u = report["units"]
    lines += ["", "## Units of prediction 2", "", f"{report['rules']['units']}.", "",
              "| pair | group | units | key vs alone b | c | mean d | mean d_w | alone correct | key correct | "
              "wrong_key correct |", "|---|---|---:|---:|---:|---:|---:|---|---|---|"]
    rows_u = [(pair, groups) for pair, groups in u["by_pair"].items()] + [("all pairs", u["groups"])]
    for pair, groups in rows_u:
        for label, g in groups.items():
            lines.append(f"| {pair} | {label} | {g['units']} | {g['b']} | {g['c']} | {_fmt_number(g['mean_d'])} | "
                         f"{_fmt_number(g['mean_d_w'])} | {_fmt_rate(g['alone'])} | {_fmt_rate(g['key'])} | "
                         f"{_fmt_rate(g['wrong_key'])} |")
    lines += ["", f"Stratified difference (recognised minus unrecognised, weighted n_r n_u / n per pair): d "
                  f"{_fmt_number(u['d']['difference'], sign=True)}"
                  + (f" (p={u['d']['p']:.4g})" if "p" in u["d"] else "")
                  + f", d_w {_fmt_number(u['d_w']['difference'], sign=True)}"
                  + (f" (p={u['d_w']['p']:.4g})" if "p" in u["d_w"] else "")
                  + f"; {u['units']} units, unknown recognition {u['unknown']}. With the pointer items: "
                  f"{u['all_items']['units']} units, d {_fmt_number(u['all_items']['d']['difference'], sign=True)}, "
                  f"d_w {_fmt_number(u['all_items']['d_w']['difference'], sign=True)}."]

    lines += ["", "## Adoption of the donor's wrong name", "",
              "adopted: the answer gives the name as its own (undecided counted as not adopted); reported: offered "
              "only in clauses that refer to the note.", "",
              "| pair | set | arm | answered | endorsed | reported (undecided) | absent | adopted | full vs alone |",
              "|---|---|---|---:|---:|---|---:|---|---|"]
    for pair, sets in report["adoption"].items():
        for set_, entry in sets.items():
            for arm, a in entry["arms"].items():
                test = _fmt_test(entry["full_vs_alone"]) if arm == "full" else ""
                lines.append(f"| {pair} | {set_} | {arm} | {a['answered']} | {a['endorsed']} | {a['reported']} "
                             f"({a['reported_undecided']}) | {a['absent']} | {_fmt_rate(a)} | {test} |")
    pooled = report["pooled"]
    und = pooled["adoption_full_undecided"]
    lines += ["", f"Pooled full vs alone (adopted true): {_fmt_test(pooled['adoption_full_vs_alone'])}; full rows "
                  f"undecided {und['undecided']} of {und['full_rows']}."]

    lines += ["", "## Items both answered correctly (both_right)", "",
              "Not correct per arm (primary); key vs alone: b = alone correct and key not, c = key correct and alone "
              "not, excess harm (b - c) / n.", "",
              "| pair | items | alone | key | full | key vs alone | excess harm |", "|---|---:|---|---|---|---|---|"]
    harm_rows = list(report["harm"].items()) + ([("all pairs", pooled["both_right"])] if report["harm"] else [])
    for pair, entry in harm_rows:
        arms, test = entry["arms"], entry["key_vs_alone"]
        excess = test["excess_harm"]
        lines.append(f"| {pair} | {entry['items']} | {_fmt_rate(arms.get('alone'))} | {_fmt_rate(arms.get('key'))} | "
                     f"{_fmt_rate(arms.get('full'))} | {_fmt_test(test)} | "
                     f"{'n/a' if excess is None else f'{100 * excess:.1f} %'} |")

    lines += ["", "## Pre-registered predictions (docs/DESIGN_keys.md section 7)", "",
              "Interim and not testable give no verdict; verdicts count only on the complete plan.", ""]
    for p in report["predictions"]:
        lines.append(f"- **{p['id']}** {p['text']} **{p['verdict']}**: {p['evidence']}")
    return "\n".join(lines) + "\n"
