"""Statistics of the key experiment: Wilson intervals, exact tests, the number of items a gain needs, and the
stratified permutation test of prediction 2 (docs/DESIGN_keys.md sections 6 and 7)."""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Iterable
from fractions import Fraction
from typing import Any

# ------------------------------------------------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of ``k`` successes in ``n`` trials (``(0.0, 1.0)`` for ``n == 0``)."""
    if n <= 0:
        return 0.0, 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar test: the binomial probability (p = 1/2) of a split of the ``b + c`` discordant pairs
    at least as uneven as the one observed; 1.0 without discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test for the table ``[[a, b], [c, d]]``: the summed probability of every table with the
    same margins that is no more likely than the observed one."""
    row1, row2, col1 = a + b, c + d, a + c
    n = row1 + row2
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = prob(a)
    lo, hi = max(0, col1 - row2), min(row1, col1)
    return min(1.0, sum(p for x in range(lo, hi + 1) if (p := prob(x)) <= observed * (1 + 1e-9)))


def items_needed(margin: float, alpha: float = 0.05) -> int:
    """The smallest number of paired items ``n`` at which a gain of ``margin`` can be significant: ``b = ceil(margin
    n)`` discordant pairs for the first arm and none for the second give ``mcnemar_exact(b, 0) < alpha`` (design
    section 7, the "inconclusive" verdict). 51 for a margin of 0.10, 26 for 0.20."""
    if not 0 < margin <= 1:
        raise ValueError(f"margin must be in (0, 1], got {margin}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    exact = Fraction(str(margin))  # the decimal as written: ceil(0.1 * 60) is 6, not 7
    n = 1
    while mcnemar_exact(math.ceil(exact * n), 0) >= alpha:
        n += 1
    return n


def _strata(units: Iterable[tuple[Hashable, Any, Any]]) -> dict[Hashable, list[tuple[bool, Fraction]]]:
    strata: dict[Hashable, list[tuple[bool, Fraction]]] = {}
    for stratum, flag, value in units:
        strata.setdefault(stratum, []).append((bool(flag), Fraction(value)))
    return strata


def _difference(strata: dict[Hashable, list[tuple[bool, Fraction]]]) -> tuple[Fraction, Fraction]:
    statistic, weight = Fraction(0), Fraction(0)
    for items in strata.values():
        flagged = [v for f, v in items if f]
        others = [v for f, v in items if not f]
        if flagged and others:
            w = Fraction(len(flagged) * len(others), len(items))
            statistic += w * (sum(flagged, Fraction(0)) / len(flagged) - sum(others, Fraction(0)) / len(others))
            weight += w
    return statistic, weight


def stratified_difference(units: Iterable[tuple[Hashable, Any, Any]]) -> dict[str, float | None]:
    """The stratified difference of means of ``units`` ``(stratum, flag, value)`` (design section 7, prediction 2):
    ``statistic`` is the sum over strata of ``n_f n_o / n`` times (the mean value of the flagged units minus that of
    the others), ``weight`` the sum of those factors, and ``difference = statistic / weight`` the weighted mean
    difference (None when no stratum holds both groups). A stratum with one group only adds nothing."""
    statistic, weight = _difference(_strata(units))
    return {"statistic": float(statistic), "weight": float(weight),
            "difference": float(statistic / weight) if weight else None}


def stratified_permutation_test(units: Iterable[tuple[Hashable, Any, Any]], *, permutations: int = 20_000,
                                seed: int = 0) -> dict[str, Any]:
    """Two-sided permutation test of :func:`stratified_difference` (design section 7, prediction 2): the flags are
    permuted within each stratum (each stratum keeps its number of flagged units), ``permutations`` times with
    ``random.Random(seed)``, and ``p = (1 + k) / (1 + permutations)`` with ``k`` the permutations whose statistic is
    at least as far from 0 as the observed one. Exact arithmetic: within a stratum the statistic is the sum of the
    flagged values minus its mean under permutation, so only that sum moves. Returns the fields of
    :func:`stratified_difference` and ``p``, ``permutations`` and ``seed``."""
    if permutations < 1:
        raise ValueError(f"permutations must be >= 1, got {permutations}")
    strata = _strata(units)
    statistic, weight = _difference(strata)
    # statistic = sum over strata of (S_f - n_f * S / n): S_f, the flagged units' sum, is all that a permutation moves
    centre, observed, fixed = Fraction(0), Fraction(0), Fraction(0)
    moving: list[tuple[list[Any], int, bool, Any]] = []
    integral = all(v.denominator == 1 for items in strata.values() for _, v in items)
    for items in strata.values():
        values = [v for _, v in items]
        flagged = sum(1 for f, _ in items if f)
        total = sum(values, Fraction(0))
        part = sum((v for f, v in items if f), Fraction(0))
        centre += flagged * total / len(items)
        observed += part
        if 0 < flagged < len(items) and len(set(values)) > 1:
            plain = [int(v) for v in values] if integral else values
            # draw the smaller side: the flagged sum is the total minus the others' sum
            if flagged <= len(items) - flagged:
                moving.append((plain, flagged, False, int(total) if integral else total))
            else:
                moving.append((plain, len(items) - flagged, True, int(total) if integral else total))
        else:
            fixed += part  # one group only, or equal values: the same sum under every permutation
    distance = abs(observed - centre)
    rng = random.Random(seed)
    hits = 0
    if integral:
        upper, lower = math.ceil(centre + distance), math.floor(centre - distance)
        base = int(fixed)
        for _ in range(permutations):
            x = base
            for values, k, complement, total in moving:
                drawn = sum(rng.sample(values, k))
                x += total - drawn if complement else drawn
            hits += x >= upper or x <= lower
    else:
        for _ in range(permutations):
            y = fixed
            for values, k, complement, total in moving:
                drawn = sum(rng.sample(values, k), Fraction(0))
                y += total - drawn if complement else drawn
            hits += abs(y - centre) >= distance
    return {"statistic": float(statistic), "weight": float(weight),
            "difference": float(statistic / weight) if weight else None,
            "p": (1 + hits) / (1 + permutations), "permutations": permutations, "seed": seed}
