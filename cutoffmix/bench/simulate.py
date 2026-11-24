"""Synthetic benchmark for checking the estimator against known truths (DESIGN.md §3.5).

:func:`synthetic_events` builds dated events for every topic x month cell, :func:`simulate_results` answers them
for simulated models with the same 4PL response model as ``cutoffmix.backends.mock`` (``P(correct) = g + (a - g)
·σ((τ_mz - t)/s)``, otherwise stale with ``stale_rate``, else abstain / another wrong option), and
:func:`recovery_report` compares a fitted index with the truths (RMSE, rank agreement, P_best calibration).
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from cutoffmix.timeutil import EPOCH_YEAR, month_label, parse_fractional_month
from cutoffmix.types import Event, Grade, ProbeKind, ProbeResult

if TYPE_CHECKING:
    from cutoffmix.router.index import CutoffIndex

__all__ = ["TruthLike", "recovery_report", "simulate_results", "synthetic_events", "true_tau"]

_CATEGORIES: tuple[str, ...] = ("office_change", "death", "election", "release", "award", "sports_result")
_SYLLABLES: tuple[str, ...] = ("ka", "lo", "mi", "ren", "sa", "tor", "vel", "dan", "ri", "mo", "zen", "pa", "lu",
                               "ne", "ko", "fa", "gil", "ta", "bri", "on")
_LETTERS = ("A", "B", "C", "D")
_ABSTAIN_TEXT = "I don't know; I have no reliable information about that."
_CALIBRATION_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)


class TruthLike(Protocol):
    """What the simulator needs from a truth (``cutoffmix.backends.mock.MockModelTruth`` fits)."""

    key: str
    tau: dict[str, float]
    default_tau: float
    slope: float
    plateau: float
    stale_rate: float


def true_tau(truth: TruthLike, topic: str) -> float:
    """The truth's cutoff for ``topic`` (``default_tau`` when the topic has no entry)."""
    return truth.tau.get(topic, truth.default_tau)


def _name(rng: np.random.Generator) -> str:
    def word() -> str:
        return "".join(_SYLLABLES[i] for i in rng.integers(len(_SYLLABLES), size=int(rng.integers(2, 4))))
    return f"{word().capitalize()} {word().capitalize()}"


def synthetic_events(topics: Sequence[str], months: Iterable[int], per_cell: int, seed: int = 0) -> list[Event]:
    """``per_cell`` synthetic events for every topic x month (month indices on the timeutil axis, 0 = 2024-01).

    Each event has a random day in its month, a current/recall/MCQ question, a unique current answer
    (``aliases``) and a unique stale answer (``reject_aliases``, also one of the MCQ options), origin
    ``"synthetic"`` and low predictability. Deterministic given ``seed``.
    """
    if per_cell < 1:
        raise ValueError(f"per_cell must be >= 1, got {per_cell}")
    rng = np.random.default_rng(seed)
    events: list[Event] = []
    for topic in topics:
        for x in months:
            year, month = EPOCH_YEAR + x // 12, x % 12 + 1
            ym = f"{year:04d}-{month:02d}"
            month_name = f"{calendar.month_name[month]} {year}"
            for k in range(per_cell):
                day = int(rng.integers(1, calendar.monthrange(year, month)[1] + 1))
                subject = f"{topic.replace('_', ' ')} office {ym}-{k:03d}"
                current, stale, *others = (_name(rng) + f" {i}{k}{x}" for i in range(4))
                answer = _LETTERS[int(rng.integers(4))]
                wrong = [stale, *others[:2]]
                rng.shuffle(wrong)
                options = iter(wrong)
                choices = [f"{letter}) {current if letter == answer else next(options)}" for letter in _LETTERS]
                events.append(Event(
                    id=f"syn-{topic}-{ym}-{k:03d}", date=f"{ym}-{day:02d}", month=ym, topic=topic,
                    category=_CATEGORIES[k % len(_CATEGORIES)], subject=subject,
                    fact=f"{current} succeeded {stale} as holder of the {subject} on {ym}-{day:02d}.",
                    question_current=f"Who currently holds the {subject}?", expected_current=current,
                    question_recall=f"Who took over the {subject} in {month_name}?", expected_recall=current,
                    mcq_question=f"Who holds the {subject}?", mcq_choices=choices, mcq_answer=answer,
                    aliases=[current], reject_aliases=[stale], source="synthetic", origin="synthetic",
                ))
    return events


def simulate_results(
    truths: Mapping[str, TruthLike],
    events: Sequence[Event],
    probes: Sequence[str],
    seed: int = 0,
) -> list[ProbeResult]:
    """Simulated, already graded answers of every truth model to every probe of every event.

    Open probes: correct with ``P = a·σ((τ_mz - t)/s)``, else stale (``reject_aliases[0]``, grade ``outdated``) with
    ``stale_rate``, else ``abstain``. MCQ: ``g = 0.25`` floor; a miss picks the stale option with ``stale_rate``,
    else another wrong letter (``incorrect``). Grader ``"synthetic"``; deterministic given ``seed``.
    """
    rng = np.random.default_rng(seed)
    out: list[ProbeResult] = []
    times = np.array([parse_fractional_month(e.date) for e in events], dtype=np.float64)
    for key, truth in truths.items():
        taus = np.array([true_tau(truth, e.topic) for e in events])
        for probe in probes:
            mcq = probe == "mcq"
            g = 0.25 if mcq else 0.0
            p = g + (truth.plateau - g) / (1.0 + np.exp(-(taus - times) / truth.slope))
            u_correct, u_stale = rng.random(len(events)), rng.random(len(events))
            u_pick = rng.integers(0, 2, size=len(events))
            for i, event in enumerate(events):
                response, grade = _answer(event, mcq, u_correct[i] < p[i], u_stale[i] < truth.stale_rate,
                                          int(u_pick[i]))
                out.append(ProbeResult(event_id=event.id, model=key, probe=cast(ProbeKind, probe), response=response,
                                       grade=grade, grader="synthetic"))
    return out


def _answer(event: Event, mcq: bool, correct: bool, stale: bool, pick: int) -> tuple[str, Grade]:
    stale_text = event.reject_aliases[0] if event.reject_aliases else ""
    if mcq:
        answer = event.mcq_answer.strip().upper()[:1]
        if correct:
            return answer, "correct"
        stale_letter = next((c.strip()[:1] for c in event.mcq_choices if stale_text and stale_text in c), None)
        if stale and stale_letter:
            return stale_letter, "outdated"
        wrong = [c.strip()[:1] for c in event.mcq_choices if c.strip()[:1] not in (answer, stale_letter)]
        return wrong[pick % len(wrong)], "incorrect"
    if correct:
        return f"{event.expected_current}.", "correct"
    if stale and stale_text:
        return f"As far as I know, {stale_text}.", "outdated"
    return _ABSTAIN_TEXT, "abstain"


def recovery_report(
    index: CutoffIndex,
    truths: Mapping[str, TruthLike],
    probe: str,
    *,
    min_gap: float = 1.0,
    min_n: int = 0,
) -> dict[str, Any]:
    """Compare a fitted index with the truths for one probe.

    Cells: (model, topic) pairs present in both with at least ``min_n`` graded items. Returns RMSE / MAE / bias of
    τ, the coverage of the ±1.645 sd interval (nominal 90%), per-topic argmax agreement (all topics and topics
    whose true best-vs-second gap is >= ``min_gap``), mean Spearman rank correlation per topic, and P_best
    calibration (Brier score, mean P_best of the true best model, reliability bins). ``cells`` lists every cell.
    """
    p = index.probes[probe]
    models = [m for m in index.models if m in truths]
    if not models:
        raise ValueError("no model of the index has a truth")
    mpos = [index.models.index(m) for m in models]
    true = np.array([[true_tau(truths[m], z) for z in index.topics] for m in models])
    est, sd, n = p.tau[mpos], np.sqrt(p.var[mpos]), p.n[mpos]
    mask = (n >= min_n) & np.isfinite(est)
    err = (est - true)[mask]
    report: dict[str, Any] = {
        "probe": probe, "n_cells": int(mask.sum()),
        "rmse": float(np.sqrt(np.mean(err ** 2))) if err.size else math.nan,
        "mae": float(np.mean(np.abs(err))) if err.size else math.nan,
        "bias": float(np.mean(err)) if err.size else math.nan,
        "coverage90": float(np.mean(np.abs(err) <= 1.645 * sd[mask])) if err.size else math.nan,
        "rmse_by_model": {m: float(np.sqrt(np.mean((est[i] - true[i])[mask[i]] ** 2))) if mask[i].any() else math.nan
                          for i, m in enumerate(models)},
    }
    agree: list[bool] = []
    agree_gap: list[bool] = []
    rhos: list[float] = []
    brier: list[float] = []
    p_true_best: list[float] = []
    pairs: list[tuple[float, float]] = []
    for z, topic in enumerate(index.topics):
        t_col, e_col = true[:, z], est[:, z]
        if len(models) < 2 or not np.isfinite(e_col).all():
            continue
        order = np.sort(t_col)[::-1]
        best_true, best_est = int(np.argmax(t_col)), int(np.argmax(e_col))
        agree.append(best_true == best_est)
        if order[0] - order[1] >= min_gap:
            agree_gap.append(best_true == best_est)
        rhos.append(_spearman(t_col, e_col))
        pb = index.p_best({topic: 1.0}, probe=probe, models=models)
        probs = np.array([pb[m] for m in models])
        onehot = (np.arange(len(models)) == best_true).astype(float)
        brier.append(float(np.sum((probs - onehot) ** 2)))
        p_true_best.append(float(probs[best_true]))
        pairs.extend(zip(probs.tolist(), onehot.tolist(), strict=True))
    report.update({
        "argmax_agreement": float(np.mean(agree)) if agree else math.nan,
        "argmax_agreement_gap": float(np.mean(agree_gap)) if agree_gap else math.nan,
        "n_topics_gap": len(agree_gap), "min_gap": min_gap,
        "spearman_mean": float(np.nanmean(rhos)) if rhos else math.nan,
        "p_best_brier": float(np.mean(brier)) if brier else math.nan,
        "p_best_true_mean": float(np.mean(p_true_best)) if p_true_best else math.nan,
        "p_best_calibration": _calibration(pairs),
        "cells": [{"model": m, "topic": z, "true": float(true[i, j]), "true_label": month_label(true[i, j]),
                   "est": float(est[i, j]), "sd": float(sd[i, j]), "n": int(n[i, j]),
                   "err": float(est[i, j] - true[i, j])}
                  for i, m in enumerate(models) for j, z in enumerate(index.topics)],
    })
    return report


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return math.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _calibration(pairs: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    """Reliability bins of (predicted P_best, observed best indicator)."""
    if not pairs:
        return []
    probs, hits = np.array(pairs, dtype=np.float64).T
    edges = np.asarray(_CALIBRATION_BINS)
    which = np.clip(np.searchsorted(edges, probs, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        sel = which == b
        if sel.any():
            bins.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum()),
                         "mean_p": float(probs[sel].mean()), "freq": float(hits[sel].mean())})
    return bins
