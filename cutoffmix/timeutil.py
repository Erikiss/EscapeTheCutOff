"""Month-index helpers.

All cutoff estimates in cutoffmix live on a continuous *month axis*:

    x = (year - 2024) * 12 + (month - 1)      # 2024-01 -> 0, 2026-11 -> 34

Fractional values are meaningful: ``x = 26.5`` is the middle of 2026-03.
The epoch (2024-01) matches the first month of the apoorvumang
knowledge-cutoff benchmark so imported data lines up without conversion.
"""

from __future__ import annotations

import datetime as _dt
import math

EPOCH_YEAR = 2024


def month_index(year: int, month: int) -> int:
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range: {month}")
    return (year - EPOCH_YEAR) * 12 + (month - 1)


def parse_month(value: str | _dt.date | _dt.datetime) -> int:
    """Parse ``'2026-03'``, ``'2026-03-17'``, ``'2026-03-17T00:00:00Z'`` or a date into a month index."""
    if isinstance(value, (_dt.date, _dt.datetime)):
        return month_index(value.year, value.month)
    s = str(value).strip()
    if len(s) < 7 or s[4] != "-":
        raise ValueError(f"cannot parse month from {value!r}")
    return month_index(int(s[0:4]), int(s[5:7]))


def parse_fractional_month(value: str | _dt.date | _dt.datetime) -> float:
    """Like :func:`parse_month` but adds the day as a fraction of the month (mid-month = +0.5)."""
    if isinstance(value, _dt.datetime):
        value = value.date()
    if isinstance(value, _dt.date):
        d = value
    else:
        s = str(value).strip()
        if len(s) >= 10 and s[7] == "-":
            d = _dt.date(int(s[0:4]), int(s[5:7]), int(s[8:10]))
        else:
            return parse_month(s) + 0.5
    days_in_month = (_dt.date(d.year + (d.month == 12), d.month % 12 + 1, 1) - _dt.date(d.year, d.month, 1)).days
    return month_index(d.year, d.month) + (d.day - 0.5) / days_in_month


def month_label(x: float, with_fraction: bool = False) -> str:
    """Inverse of :func:`month_index`; ``34 -> '2026-11'``. Non-finite values render as ``'n/a'``."""
    if x is None or not math.isfinite(x):
        return "n/a"
    base = math.floor(x)
    year = EPOCH_YEAR + base // 12
    month = base % 12 + 1
    label = f"{year:04d}-{month:02d}"
    if with_fraction:
        label += f" (+{x - base:.2f})"
    return label


def today_month(today: _dt.date | None = None) -> float:
    return parse_fractional_month(today or _dt.date.today())
