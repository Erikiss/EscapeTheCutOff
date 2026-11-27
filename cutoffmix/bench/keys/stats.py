"""Statistics of the key experiment: Wilson intervals and exact tests (docs/DESIGN_keys.md section 6)."""

from __future__ import annotations

import math

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

