"""The report of the key experiment: the join of plan and results, the rates per pair, set and arm, the paired
comparisons and the verdicts on the pre-registered predictions (docs/DESIGN_keys.md version 2, section 6 "report" and
"Primary analysis", and section 7).

- Join: the last result row per id counts. A row without ``backend`` or ``prompt_sha`` has no provenance, a row whose
  ``prompt_sha`` differs from :func:`prompt_sha` of the plan row's messages is stale, and a row of an id the plan does
  not hold is foreign. All three are left out and counted; the cost of the foreign rows is reported on its own.
- Primary analysis: an answered row (no ``error``, grader not ``refusal``) counts, and ``ungraded`` counts as not
  correct. Refusals are left out and counted. The sensitivity line leaves the ungraded rows out as well.
- Costs: over every row of the results file (each line, not only the last per id), the recipients (``usage``) and the
  judges (``judge_usage``) apart. The joined answers' cost counts, per joined id, every row of the request its last
  row answered (the call, a failed grading, the re-grades of ``run --judges``, whose ``usage`` is None).
- Backends: when every joined row comes from the mock, the report is SIMULATED and each verdict says so; when the
  joined rows come from more than one backend, it gives no verdicts.
- Verdicts are stated by :class:`_Voice`: what holds for a prediction's verdict holds for its parts (the comparisons,
  the halves of prediction 3). Interim and mixed-backend parts carry the interim or mixed text, and the evidence prints
  their numbers without a verdict (no verdict on partial data: robust-2); simulated parts are prefixed as well.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

from cutoffmix.bench.keys.common import ARMS_BY_SET, SETS, judge_cost, prompt_sha, row_cost
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
UNDECIDED_LIMIT = 0.20  # 3a and 3b are not testable when a larger share of their full rows is undecided
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
UNDECIDED_VERDICT = "not testable (undecided answers; grade with --judges)"
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


def _request(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    """The request a result row answered: its ``backend``, ``prompt_sha`` and ``request_sha``. ``run`` copies them
    from the answer it grades again (a re-grade of ``run --judges`` or after a failed grading: ``usage`` None, the new
    judges' calls only), so the answering call and its re-grades share them; a stale answer or one given to other
    settings does not."""
    return row.get("backend"), row.get("prompt_sha"), row.get("request_sha")


def _cost(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    chosen = list(rows)
    recipients = sum(row_cost(r) for r in chosen)
    judges = sum(judge_cost(r) for r in chosen)
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


def _discordant(both: Sequence[tuple[_Row, _Row]], success: Callable[[_Row], bool],
                second: Callable[[_Row], bool] | None = None) -> dict[str, Any]:
    """b, c and the exact McNemar p of the pairs ``(x, y)``: ``success`` decides x, and y as well unless ``second``
    is given."""
    succeeds = second or success
    b = sum(1 for x, y in both if success(x) and not succeeds(y))
    c = sum(1 for x, y in both if succeeds(y) and not success(x))
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


def _against(items: Iterable[_Item], first: str, second: str, success: Callable[[_Row], bool],
             unsure_first: Callable[[_Row], bool], unsure_second: Callable[[_Row], bool]) -> dict[str, Any]:
    """The paired comparison of :func:`_compare` twice more, with the undecided rows resolved against a verdict
    (design section 7): ``refute`` counts every undecided ``first`` row as a success (deciding them could only raise
    b and lower c), ``support`` every undecided ``second`` row (that could only lower b and raise c)."""
    both = [(it.rows[first], it.rows[second]) for it in items if it.answered(first, second)]
    return {"refute": _discordant(both, lambda r: success(r) or unsure_first(r), success),
            "support": _discordant(both, success, lambda r: success(r) or unsure_second(r))}


def _three_way(comparison: Mapping[str, Any], margin: float) -> dict[str, Any]:
    """The verdict of design section 7 on "first beats second" with ``margin``: ``not testable (n too small)`` on fewer
    than 6 items, ``supported`` if b > c and p < 0.05, ``refuted`` if the 95 % Wilson upper bound of b / n is below the
    margin (the paired gain (b - c) / n is at most b / n), else ``inconclusive``. The observed number of discordant
    pairs never decides testability.

    ``items_needed``: the items a gain of the margin would need at the observed share of discordant pairs
    (:func:`items_needed` with ``discordance = (b + c) / n``). An inconclusive verdict names it when it exceeds the
    items at hand ("k items; about N would be needed for a gain of m"); when it does not, the items sufficed for such a
    gain and the data show none: "k items; a gain of m is neither shown nor ruled out"."""
    n, b, c = int(comparison["n_items"]), int(comparison["b"]), int(comparison["c"])
    upper = wilson(b, n)[1]
    needed = items_needed(margin, ALPHA, discordance=Fraction(b + c, n) if n else None)
    if n < FLOOR_ITEMS:
        verdict = NOT_TESTABLE_N
    elif b > c and mcnemar_exact(b, c) < ALPHA:
        verdict = "supported"
    elif upper < margin:
        verdict = "refuted"
    elif needed > n:
        verdict = f"inconclusive ({n} items; about {needed} would be needed for a gain of {margin:.2f})"
    else:
        verdict = f"inconclusive ({n} items; a gain of {margin:.2f} is neither shown nor ruled out)"
    return {**comparison, "margin": margin, "upper": round(upper, 4), "items_needed": needed, "verdict": verdict}


@dataclass(frozen=True)
class _Voice:
    """How the report states verdicts (design sections 6 and 7). ``withheld`` stands for every verdict when the joined
    rows come from more than one backend (the report gives none); ``prefix`` is ``"simulated: "`` when every joined
    row comes from the mock. A part of a prediction (a comparison, a half of prediction 3) whose planned rows are not
    all answered carries the interim text instead of its verdict, and the evidence prints its numbers without one:
    a verdict on partial data is not shown, "supported" included (robust-2)."""

    withheld: str | None = None
    prefix: str = ""

    def verdict(self, raw: str, interim: str | None = None) -> str:
        """The verdict as stated: the withheld text, else the interim text or ``raw``, with the prefix."""
        return self.withheld or self.prefix + (interim or raw)

    def label(self, raw: str, interim: str | None = None) -> str | None:
        """The verdict printed next to a comparison's numbers in the evidence: None when withheld or interim."""
        return None if self.withheld or interim else self.prefix + raw


def _stated(entry: dict[str, Any], voice: _Voice, interim: str | None) -> str | None:
    """Replace a :func:`_three_way` entry's ``verdict`` by the one :class:`_Voice` states; return its evidence label."""
    raw = str(entry["verdict"])
    entry["verdict"] = voice.verdict(raw, interim)
    return voice.label(raw, interim)


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


def _never(row: _Row) -> bool:
    """No row of this arm is undecided: in 3a an ungraded ``key`` row counts as not correct (the key never shows the
    name, so the rules leave almost only misses ungraded; design section 6)."""
    return False


# the labelled rows (design section 6) in the order the diagnostic table lists them
LABELLED: tuple[tuple[str, str], ...] = (("dark", "full"), ("both_right", "full"), ("donor_wrong", "alone"),
                                         ("donor_wrong", "full"), ("control", "alone"), ("control", "full"))
_VERSUS = ("judged", "undecided", "agree", "rule_yes_judges_no", "rule_no_judges_yes", "open_yes", "open_no",
           "no_rule")


def _rules_vs_judges(rows: Iterable[_Row]) -> dict[str, dict[str, Any]]:
    """The diagnostic of design section 6 ("rules versus judges"; no verdict uses it): for the answered labelled rows
    whose label the adoption judges decided (``adoption_grader`` ``judge:<keys>``), per set and arm pooled over the
    pairs, how the judges' decision (``adopted``) compares with the rules' own (``rule_adopted``): ``agree``, the rules
    adopted and the judges did not (``rule_yes_judges_no``) or the reverse (``rule_no_judges_yes``), the rules left it
    open and the judges adopted (``open_yes``) or did not (``open_no``); ``undecided`` the rows the judges' votes left
    open, ``no_rule`` those written before ``rule_adopted`` was stored. On the ``full`` rows of ``dark`` and
    ``both_right`` "adopted" is the right name given as the answer's own (correct)."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if (not row.answered or row.get("adoption") is None
                or not str(row.get("adoption_grader") or "").startswith("judge")):
            continue
        set_, arm = str(row.plan.get("set")), str(row.plan.get("arm"))
        entry = out.setdefault(f"{set_} {arm}", {"set": set_, "arm": arm, **{k: 0 for k in _VERSUS}})
        entry["judged"] += 1
        adopted = row.get("adopted")
        if adopted is None:
            entry["undecided"] += 1
        elif "rule_adopted" not in (row.result or {}):
            entry["no_rule"] += 1
        elif row.get("rule_adopted") is None:
            entry["open_yes" if adopted else "open_no"] += 1
        elif bool(row.get("rule_adopted")) == bool(adopted):
            entry["agree"] += 1
        else:
            entry["rule_yes_judges_no" if row.get("rule_adopted") else "rule_no_judges_yes"] += 1
    order = {f"{s} {a}": i for i, (s, a) in enumerate(LABELLED)}
    return dict(sorted(out.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0])))


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


_UNIT_SCOPES: dict[str, Callable[[_Item], bool]] = {
    "no_pointer": lambda item: not item.pointer, "all": lambda item: True, "pointer": lambda item: item.pointer}


def _units(dark: Mapping[str, Sequence[_Item]], priority: Callable[[str], tuple[Any, ...]],
           scope: str = "no_pointer") -> dict[str, Any]:
    """The units of prediction 2 (design section 7, :data:`UNIT_RULE`) with d = [key correct] - [alone correct] and
    d_w = [key correct] - [wrong_key correct] (ungraded is not correct), per group and per pair and group. ``scope``:
    the items without pointer (the units the verdict reads), ``all`` items or the ``pointer`` items on their own (the
    numbers printed next to the verdict)."""
    keep = _UNIT_SCOPES[scope]
    candidates: dict[tuple[str, str], list[_Item]] = {}
    for items in dark.values():
        for item in items:
            if keep(item) and item.answered("alone", "key", "wrong_key"):
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


def _fmt_verdict(entry: Mapping[str, Any], label: str | None) -> str:
    """A comparison's numbers with its verdict ``label`` (left out when None: see :class:`_Voice`)."""
    bound = f"Wilson upper bound of b/n {entry['upper']:.3f}"
    return f"{_fmt_test(entry)} [{label}; {bound}]" if label else f"{_fmt_test(entry)} [{bound}]"


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


def _prediction_1(ctx: _Context, voice: _Voice) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Predictions 1 and 1b on the decisive pair's dark items without pointer (design section 7), with their numbers
    over all items and on the pointer items on their own; also prediction 1's verdict before :class:`_Voice` states
    it (the gate of prediction 2)."""
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
    raw = _combine([ka["verdict"], kw["verdict"]])
    if raw == "supported" and confounded:
        raw = CONFOUNDED
    raw_b = _combine([kb["verdict"]])
    withheld = bool(voice.withheld or interim)  # no verdict, so no qualifier of one either
    label_ka, label_kw, label_kb = (_stated(entry, voice, interim) for entry in (ka, kw, kb))
    all_tests = _dark_comparisons(every)
    pointer_tests = _dark_comparisons(pointed)
    arms = _arms_table("dark", plain)
    pointer_arms = _arms_table("dark", pointed)
    if pair is None:
        scope = "no keyed pair in the plan (no dark items)"
    else:
        counts = ctx.planned_dark[pair]
        scope = (f"decisive pair {pair} ({counts['no_pointer']} planned dark items without pointer, "
                 f"{counts['all']} in all)")
    pointer_scope = f"pointer items (B resolves A's pointer; {len(pointed)} items)"
    evidence = (f"{scope}; without pointer: key vs alone {_fmt_verdict(ka, label_ka)}; key vs wrong key "
                f"{_fmt_verdict(kw, label_kw)}; blank vs alone {_fmt_test(tests['blank_vs_alone'])} (the note's "
                f"presence); blank vs wrong key {_fmt_test(bw)} (the harm of a mismatched cue"
                f"{', significant: confounded' if confounded and not withheld else ''}); correct: key "
                f"{_fmt_rate(arms.get('key'))}, alone {_fmt_rate(arms.get('alone'))}, wrong key "
                f"{_fmt_rate(arms.get('wrong_key'))}, blank {_fmt_rate(arms.get('blank'))}; all items: key vs alone "
                f"{_fmt_test(all_tests['key_vs_alone'])}, key vs wrong key {_fmt_test(all_tests['key_vs_wrong_key'])}; "
                f"{pointer_scope}: key vs alone {_fmt_test(pointer_tests['key_vs_alone'])}, key vs wrong key "
                f"{_fmt_test(pointer_tests['key_vs_wrong_key'])}, correct: key {_fmt_rate(pointer_arms.get('key'))}, "
                f"alone {_fmt_rate(pointer_arms.get('alone'))}, wrong key {_fmt_rate(pointer_arms.get('wrong_key'))}; "
                f"expected: key 10-30 %, wrong key under 10 %")
    evidence_b = (f"{scope}; without pointer: key vs blank {_fmt_verdict(kb, label_kb)}; correct: key "
                  f"{_fmt_rate(arms.get('key'))}, blank {_fmt_rate(arms.get('blank'))}; all items: key vs blank "
                  f"{_fmt_test(all_tests['key_vs_blank'])}; {pointer_scope}: key vs blank "
                  f"{_fmt_test(pointer_tests['key_vs_blank'])}, correct: key {_fmt_rate(pointer_arms.get('key'))}, "
                  f"blank {_fmt_rate(pointer_arms.get('blank'))}")
    one = {"id": 1, "text": "On dark items without pointer of the decisive pair, key beats alone and wrong_key "
                            "(paired exact McNemar, margin 0.10).",
           "verdict": voice.verdict(raw, interim), "evidence": evidence, "pair": pair, "items": len(plain),
           "comparisons": {"key_vs_alone": ka, "key_vs_wrong_key": kw, "blank_vs_alone": tests["blank_vs_alone"],
                           "blank_vs_wrong_key": bw},
           "confounded": None if withheld else confounded,
           "all_items": {k: all_tests[k] for k in ("key_vs_alone", "key_vs_wrong_key")},
           "pointer_items": {"items": len(pointed), "key_vs_alone": pointer_tests["key_vs_alone"],
                             "key_vs_wrong_key": pointer_tests["key_vs_wrong_key"], "key": pointer_arms.get("key"),
                             "alone": pointer_arms.get("alone"), "wrong_key": pointer_arms.get("wrong_key")}}
    one_b = {"id": "1b", "text": "key beats blank on the same items: the donor's content, not the presence of a "
                                 "note, does the work (margin 0.10).",
             "verdict": voice.verdict(raw_b, interim), "evidence": evidence_b, "pair": pair, "items": len(plain),
             "comparisons": {"key_vs_blank": kb}, "all_items": {"key_vs_blank": all_tests["key_vs_blank"]},
             "pointer_items": {"items": len(pointed), "key_vs_blank": pointer_tests["key_vs_blank"],
                               "key": pointer_arms.get("key"), "blank": pointer_arms.get("blank")}}
    return one, one_b, interim or raw


def _fmt_units(units: Mapping[str, Any]) -> str:
    """Units of another scope, briefly: their count per group, key vs alone per group, the stratified differences."""
    rec, unrec = units["groups"]["recognised"], units["groups"]["unrecognised"]
    return (f"{units['units']} units (recognised {rec['units']}, unrecognised {unrec['units']}), key vs alone "
            f"recognised b={rec['b']}, c={rec['c']}, unrecognised b={unrec['b']}, c={unrec['c']}, mean d recognised "
            f"{_fmt_number(rec['mean_d'])}, unrecognised {_fmt_number(unrec['mean_d'])}, stratified difference of d "
            f"{_fmt_number(units['d']['difference'], sign=True)}, of d_w "
            f"{_fmt_number(units['d_w']['difference'], sign=True)}")


def _brief(units: Mapping[str, Any]) -> dict[str, Any]:
    return {k: units[k] for k in ("units", "groups", "d", "d_w")}


def _prediction_2(ctx: _Context, voice: _Voice, units: dict[str, Any], gate: str, gate_stated: str,
                  every: Mapping[str, Any], pointed: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 2 (design section 7): the key's gain over ``alone`` is larger on recognised units; gated on
    prediction 1 (``gate``: its verdict before :class:`_Voice`, ``gate_stated``: as stated) and at least 10 units per
    group; supported or inconclusive, never refuted. ``every`` and ``pointed``: the units over all items and of the
    pointer items on their own, printed next to it."""
    interim = _interim(*_completed(ctx.of("dark", ctx.keyed)))
    groups = units["groups"]
    rec, unrec = groups["recognised"], groups["unrecognised"]
    tested: dict[str, Any] = {}
    if interim:
        raw = interim
    elif not gate.startswith("supported"):
        raw = f"not testable (prediction 1 is {_verb(gate)})"
    elif rec["units"] < MIN_UNITS or unrec["units"] < MIN_UNITS:
        raw = NOT_TESTABLE_N
    else:
        known = [u for u in units["list"] if u["recognised"] is not None]
        for name in ("d", "d_w"):
            tested[name] = stratified_permutation_test([(u["pair"], u["recognised"], u[name]) for u in known],
                                                       permutations=PERMUTATIONS, seed=PERMUTATION_SEED)
            units[name] = tested[name]
        a = rec["b"] > rec["c"] and mcnemar_exact(rec["b"], rec["c"]) < ALPHA
        gain_d, gain_dw = ((tested[n]["statistic"] > 0 and tested[n]["p"] < ALPHA) for n in ("d", "d_w"))
        if a and gain_d and gain_dw:
            raw = "supported"
        else:
            raw = (f"inconclusive (stratified difference of d {_fmt_number(tested['d']['difference'], sign=True)}"
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
                f"share {_fmt_number(units['guess_corrected'])}; all items (pointer included): {_fmt_units(every)}; "
                f"pointer items on their own (B resolves A's pointer): {_fmt_units(pointed)}; gate: prediction 1 "
                f"{_verb(gate_stated)}")
    return {"id": 2, "text": "The key's gain over alone is larger on recognised than on unrecognised units (distinct "
                             "units, permutation test stratified by pair; gated on prediction 1).",
            "verdict": voice.verdict(raw, interim), "evidence": evidence, "units": units["units"],
            "groups": {"recognised": rec, "unrecognised": unrec}, "tests": tested,
            "all_items": _brief(every), "pointer_items": _brief(pointed)}


def _gated(comparison: Mapping[str, Any], undecided: Mapping[str, Any], against: Mapping[str, Any],
           margin: float) -> str:
    """A half of prediction 3 (design section 7): the floor, then the undecided share of its full rows (more than
    :data:`UNDECIDED_LIMIT`: :data:`UNDECIDED_VERDICT`), then the three-way verdict, which stands only if the undecided
    rows resolved against it (``against``, :func:`_against`) leave it as it is: a refutation that the undecided first
    rows counted as successes would overturn, or a support that the undecided second rows counted as successes would,
    is :data:`UNDECIDED_VERDICT` as well."""
    raw = str(comparison["verdict"])
    if raw == NOT_TESTABLE_N:
        return NOT_TESTABLE_N
    if _share_above(undecided["undecided"], undecided["full_rows"], UNDECIDED_LIMIT):
        return UNDECIDED_VERDICT
    for verdict, case in (("refuted", "refute"), ("supported", "support")):
        if raw == verdict and _three_way(against[case], margin)["verdict"] != verdict:
            return UNDECIDED_VERDICT
    return _combine([raw])


def _fmt_undecided(undecided: Mapping[str, Any]) -> str:
    share = undecided["share"]
    return (f"{undecided['undecided']} of {undecided['full_rows']} "
            f"({'n/a' if share is None else f'{100 * share:.1f} %'}, limit 20 %)")


def _fmt_against(against: Mapping[str, Any], first: str, second: str | None, what: str) -> str:
    """The comparisons with the undecided rows resolved against the verdict, numbers only (``second`` None: that arm
    has no undecided rows, so only the refutation's check is printed)."""
    r, s = against["refute"], against["support"]
    text = f"undecided {first} rows counted as {what} b={r['b']}, c={r['c']}"
    return text if second is None else f"{text}; undecided {second} rows counted as {what} b={s['b']}, c={s['c']}"


def _prediction_3(ctx: _Context, voice: _Voice, pooled: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 3 (design section 7): 3a full vs key pooled over the keyed pairs' dark items (margin 0.10), 3b full
    vs alone on ``adopted`` True pooled over donor_wrong and control of every pair (margin 0.20). Each half checks the
    floor, then the undecided share of its full rows (3a: the dark full rows left ungraded, 3b: adoption undecided;
    more than 20 % undecided is not testable, since an undecided row counts as a miss), then the three-way verdict,
    which must survive the undecided rows resolved against it (3a: the ungraded full rows counted as correct; 3b: the
    undecided full rows, or the undecided alone rows, counted as adopted), else the half is not testable either.
    Each half is interim on its own rows; the prediction is interim when one half is (k and N summed over both)."""
    done_a, planned_a = _completed(ctx.of("dark", ctx.keyed))
    adoption_items = [it for s in ADOPTION_SETS for it in ctx.of(s)]
    done_b, planned_b = _completed(adoption_items)
    interim_a, interim_b = _interim(done_a, planned_a), _interim(done_b, planned_b)
    a = _three_way(pooled["full_vs_key"], KEY_MARGIN)
    b = _three_way(pooled["adoption_full_vs_alone"], ADOPTION_MARGIN)
    undecided_a, undecided = pooled["full_undecided"], pooled["adoption_full_undecided"]
    against_a, against_b = pooled["full_vs_key_against"], pooled["adoption_full_vs_alone_against"]
    raw_a = _gated(a, undecided_a, against_a, KEY_MARGIN)
    raw_b = _gated(b, undecided, against_b, ADOPTION_MARGIN)
    interim = (f"interim ({done_a + done_b} of {planned_a + planned_b} planned items answered)"
               if interim_a or interim_b else None)
    half_a, half_b = voice.verdict(raw_a, interim_a), voice.verdict(raw_b, interim_b)
    label_a, label_b = _stated(a, voice, interim_a), _stated(b, voice, interim_b)
    evidence = (f"3a full vs key, the keyed pairs' dark items: {_fmt_verdict(a, label_a)} -> {half_a}; dark full "
                f"rows undecided (ungraded) {_fmt_undecided(undecided_a)}; "
                f"{_fmt_against(against_a, 'full', None, 'correct')}; 3b adoption (adopted true) full vs alone, "
                f"donor_wrong and control of every pair: {_fmt_verdict(b, label_b)} -> {half_b}; full rows undecided "
                f"{_fmt_undecided(undecided)}; {_fmt_against(against_b, 'full', 'alone', 'adopted')}; adopted: full "
                f"{_fmt_rate(pooled['adoption_arms'].get('full'))}, alone "
                f"{_fmt_rate(pooled['adoption_arms'].get('alone'))}")
    return {"id": 3, "text": "full beats key on dark items (3a, margin 0.10), and full makes the recipient endorse a "
                             "wrong donor answer far more often than alone (3b, adopted true, margin 0.20).",
            "verdict": voice.verdict(_combine([raw_a, raw_b]), interim), "evidence": evidence,
            "halves": {"3a": {"verdict": half_a, "comparison": a, "answered": done_a, "planned": planned_a,
                              "undecided": undecided_a, "against": against_a},
                       "3b": {"verdict": half_b, "comparison": b, "answered": done_b, "planned": planned_b,
                              "undecided": undecided, "against": against_b}}}


def _prediction_4(ctx: _Context, voice: _Voice, pooled: Mapping[str, Any]) -> dict[str, Any]:
    """Prediction 4 (design section 7): on both_right, the excess harm (b - c) / n of key over alone is under 10 %."""
    interim = _interim(*_completed(ctx.of("both_right")))
    harm = pooled["both_right"]
    paired = harm["key_vs_alone"]
    n, excess = paired["n_items"], paired["excess_harm"]
    if n < MIN_HARM_ITEMS:
        raw = NOT_TESTABLE_N
    elif Fraction(paired["b"] - paired["c"], n) < Fraction(str(HARM_LIMIT)):
        raw = "supported"
    else:
        raw = "not supported"
    verdict = voice.verdict(raw, interim)
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
    recognition, ``all`` and ``no_pointer``), ``units`` (prediction 2; its numbers over ``all_items`` and on the
    ``pointer_items`` on their own), ``adoption`` (per pair, set and arm, and full vs alone), ``harm`` (both_right per
    pair), ``pooled`` (the comparisons of predictions 3 and 4), ``rules_vs_judges`` (the diagnostic of the labelled rows
    the judges decided, :func:`_rules_vs_judges`), ``rules``, ``flags`` and ``predictions`` (ids 1, "1b",
    2, 3, 4, each with ``text``, ``verdict`` and ``evidence``; 1, 1b and 2 also with ``all_items`` and
    ``pointer_items``). Verdicts are stated as :class:`_Voice` says, in the predictions' parts as well."""
    planned: dict[str, Mapping[str, Any]] = {}
    for row in plan:
        planned[str(row.get("id"))] = row
    last: dict[str, Mapping[str, Any]] = {}
    lines: dict[str, list[Mapping[str, Any]]] = {}  # every row per id, in file order
    for row in results:
        last[str(row.get("id"))] = row
        lines.setdefault(str(row.get("id")), []).append(row)

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
    # the joined answers' cost: per joined id, every row of the request its last row answered (design section 6)
    answer_rows = [line for r in joined if r.result is not None for line in lines[str(r.plan.get("id"))]
                   if _request(line) == _request(r.result)]
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
                 "joined": {"ids": len(joined), **_cost(answer_rows)}},
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
    # 3a's gate: the keyed pairs' answered dark full rows left ungraded (no judge decided the grade or the right name).
    # Their note shows the right name, so an ungraded one may be a pass-through the judges call correct, also when
    # the rules could not grade it at all ("It might be X.": "might" is a negation of the paper rules)
    dark_full = [it.rows["full"] for it in ctx.of("dark", keyed) if "full" in it.rows and it.rows["full"].answered]
    undecided_dark = sum(1 for r in dark_full if r.ungraded)
    pooled = {
        "full_vs_key": _compare(ctx.of("dark", keyed), "full", "key"),
        "full_undecided": {"undecided": undecided_dark, "full_rows": len(dark_full),
                           "share": undecided_dark / len(dark_full) if dark_full else None},
        "full_vs_key_against": _against(ctx.of("dark", keyed), "full", "key", _is_correct, _is_ungraded,
                                        _never),
        "adoption_full_vs_alone": _compare(adoption_items, "full", "alone", _adopted, _undecided),
        "adoption_full_undecided": {"undecided": undecided, "full_rows": len(full_rows),
                                    "share": undecided / len(full_rows) if full_rows else None},
        "adoption_full_vs_alone_against": _against(adoption_items, "full", "alone", _adopted, _undecided,
                                                   _undecided),
        "adoption_arms": {arm: _adoption_arm([it.rows[arm] for it in adoption_items if arm in it.rows])
                          for arm in ("alone", "full")},
        "both_right": _harm(ctx.of("both_right")),
    }
    keyed_dark = {pair: by.get((pair, "dark"), []) for pair in keyed}
    units = _units(keyed_dark, priority)
    units_every = _units(keyed_dark, priority, scope="all")
    units_pointer = _units(keyed_dark, priority, scope="pointer")
    units["all_items"] = {k: v for k, v in units_every.items() if k != "list"}
    units["pointer_items"] = {k: v for k, v in units_pointer.items() if k != "list"}

    voice = _Voice(withheld=f"not testable (mixed backends: {', '.join(backends)})" if mixed else None,
                   prefix="simulated: " if simulated else "")
    one, one_b, gate = _prediction_1(ctx, voice)
    two = _prediction_2(ctx, voice, units, gate, one["verdict"], units_every, units_pointer)
    predictions = [one, one_b, two, _prediction_3(ctx, voice, pooled), _prediction_4(ctx, voice, pooled)]

    report: dict[str, Any] = {
        "totals": totals, "backends": backends, "simulated": simulated, "mixed_backends": mixed,
        "decisive_pair": decisive, "keyed_pairs": keyed, "planned_dark": planned_dark, "pairs": pairs,
        "paired": paired, "pointer_items": pointer_items, "recognised": recognised,
        "units": units, "adoption": adoption, "harm": harm,
        "pooled": pooled, "rules_vs_judges": _rules_vs_judges(rows.values()),
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
    the split by recognition, the units of prediction 2, adoption, harm, the rules versus the judges (a diagnostic),
    and the predictions with their evidence."""
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
              f"(of it, ids not in the plan: {cost['foreign']['total_usd']:.4f} USD). The joined answers, each with "
              f"every row of the request it answered (the call, a failed grading, the re-grades): "
              f"{cost['joined']['total_usd']:.4f} USD (recipients {cost['joined']['recipients_usd']:.4f}, judges "
              f"{cost['joined']['judges_usd']:.4f}; {cost['joined']['rows']} rows of {cost['joined']['ids']} ids).",
              ""]
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
                  f"d_w {_fmt_number(u['all_items']['d_w']['difference'], sign=True)}. The pointer items on their own "
                  f"(B resolves A's pointer): {_fmt_units(u['pointer_items'])}."]

    lines += ["", "## Adoption of the donor's wrong name", "",
              "adopted: the answer gives the name as its own (undecided counted as not adopted); reported: offered "
              "only in clauses that refer to the note; undecided: left to the judges (reported speech, or an "
              "endorsement the answer contradicts) and not decided.", "",
              "| pair | set | arm | answered | endorsed | reported | absent | undecided | adopted | full vs alone |",
              "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for pair, sets in report["adoption"].items():
        for set_, entry in sets.items():
            for arm, a in entry["arms"].items():
                test = _fmt_test(entry["full_vs_alone"]) if arm == "full" else ""
                lines.append(f"| {pair} | {set_} | {arm} | {a['answered']} | {a['endorsed']} | {a['reported']} | "
                             f"{a['absent']} | {a['undecided']} | {_fmt_rate(a)} | {test} |")
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

    lines += ["", "## Rules versus judges (diagnostic; no verdict uses it)", "",
              ("The labelled rows the adoption judges were asked about, by set and arm over all pairs: the rules' own "
               "decision (rule_adopted) next to the judges'. Yes: the answer gives the name as its own (adopted; on "
               "the full rows of dark and both_right the right name, so correct). A judge that rejects plain answers "
               "shows as rules yes, judges no."), ""]
    versus = report.get("rules_vs_judges") or {}
    if versus:
        lines += [("| set | arm | judged | judges undecided | agree | rules yes, judges no | rules no, judges yes | "
                   "rules open, judges yes | rules open, judges no |"), "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for entry in versus.values():
            lines.append(f"| {entry['set']} | {entry['arm']} | {entry['judged']} | {entry['undecided']} | "
                         f"{entry['agree']} | {entry['rule_yes_judges_no']} | {entry['rule_no_judges_yes']} | "
                         f"{entry['open_yes']} | {entry['open_no']} |")
        no_rule = sum(entry["no_rule"] for entry in versus.values())
        if no_rule:
            lines += ["", (f"{no_rule} rows the judges decided carry no rule decision (written before rule_adopted was "
                           "stored; a later run stores it without a call).")]
    else:
        lines.append("No labelled row was decided by the adoption judges.")

    lines += ["", "## Pre-registered predictions (docs/DESIGN_keys.md section 7)", "",
              "Interim and not testable give no verdict; verdicts count only on the complete plan.", ""]
    for p in report["predictions"]:
        lines.append(f"- **{p['id']}** {p['text']} **{p['verdict']}**: {p['evidence']}")
    return "\n".join(lines) + "\n"
