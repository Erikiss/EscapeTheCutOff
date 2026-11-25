"""Cutoff index estimator (DESIGN.md §4): graded probe results -> :class:`~cutoffmix.router.index.CutoffIndex`.

Per probe kind, on graded results of non-control, low/medium-predictability events (``y = 1`` iff correct,
``t`` = fractional event month on the :mod:`cutoffmix.timeutil` axis):

1. **Model-level 4PL** ``P(y=1) = g + (a - g)·σ((τ - t)/s)`` (``g = 0.25`` for MCQ, else 0), grid MAP over
   τ ∈ [-6, 40], s ∈ {0.25 … 4}, a ∈ {0.55 … 0.99} with priors τ ~ N(prior or the median event month, prior_sd²),
   log s ~ N(0, 0.5²), a ~ Beta(9, 1) -> ``global_tau``, ``slope``, ``plateau``.
2. **Per-cell posterior** of τ for every (model, topic) on the τ grid with ``s``/``a`` fixed at the model's values
   and prior N(global_tau_m + γ_z, σ_ε²): γ_z is a topic offset shared by all models (precision-weighted mean of
   the cell MLE deviations of cells whose likelihood pins τ (standard error below 3 months), shrunk to 0 with
   σ_γ = 2) and σ_ε comes from the method of moments on such cells with at least 8 items, clipped to [0.5, 3] (1.5
   if fewer than 3 such cells). γ and σ_ε are iterated to a fixed point. Censored cells (all correct / all wrong) get a
   finite posterior in the prior-driven upper / lower region; empty cells get the prior.
3. **Draws**: paired cluster bootstrap (events resampled within each topic, one resample shared by all models),
   recomputing the step-2 posterior means incl. γ (σ_ε, s, a and global τ fixed). Cells without data, and topics
   with fewer than 5 distinct events (whose resamples barely vary), are drawn from Normal(tau, var).
   ``bootstrap=0`` stores no draws (consumers then use Normal(tau, var)).
4. **Hazard / abstain**: among graded items with ``t > tau + 2·s``, (outdated+incorrect) resp. abstain rates with
   two-level Beta shrinkage, 5 pseudo-observations each: cell rate toward the model-level rate, model-level rate
   toward the rate pooled over all models (a model with few items after its cutoff gets about the pooled rate).

With ``global_origins`` (CLI ``--global-origins kc``) step 1 sees only rows whose event origin is in that set, so the
model-level anchor τ_m (and s, a) comes from one kind of knowledge; steps 2-4 (cells, γ/σ_ε, bootstrap, rates) use all
rows, so a topic with its own data from another origin still gets its own cutoff while topics without data inherit the
anchor of the chosen origins. A model without rows from those origins falls back to the prior as if it had no data
(``meta.hyper[probe].model_level`` records, per model, the rows and origins behind step 1 and that fallback).

Everything is vectorised numpy; log-likelihoods use stable log-sigmoid forms and posteriors are normalised with
log-sum-exp, so no cell can overflow, underflow to an empty posterior or run off to infinity.
"""

from __future__ import annotations

import datetime as _dt
import math
import zlib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from cutoffmix.bench.grading import CONTROL_CATEGORIES
from cutoffmix.bench.io import ORIGIN_ATTRIBUTION
from cutoffmix.router.index import CutoffIndex, ProbeIndex
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.timeutil import parse_fractional_month
from cutoffmix.types import Event, ProbeResult

__all__ = [
    "A_GRID", "MCQ_FLOOR", "PROBE_ORDER", "S_GRID", "TAU_MAX", "TAU_MIN", "eligible_event", "fit_index",
    "kc_style_cutoffs",
]

TAU_MIN, TAU_MAX = -6.0, 40.0
S_GRID: tuple[float, ...] = (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0)
A_GRID: tuple[float, ...] = tuple(round(0.55 + 0.01 * i, 2) for i in range(45))  # 0.55 .. 0.99
MCQ_FLOOR = 0.25
LOG_S_SD = 0.5
PLATEAU_BETA = (9.0, 1.0)
SIGMA_GAMMA = 2.0
SIGMA_EPS_DEFAULT = 1.5
SIGMA_EPS_BOUNDS = (0.5, 3.0)
MIN_CELL_N = 8
INFORMATIVE_SE = 3.0  # a cell whose likelihood leaves τ more uncertain than this (months) is censored
MIN_BOOTSTRAP_EVENTS = 5  # topics with fewer distinct events get Normal(tau, var) draws instead of a bootstrap
MIN_MOM_CELLS = 3
HAZARD_PSEUDO = 5.0
PROBE_ORDER: tuple[str, ...] = ("current_d", "current", "recall", "mcq")
_GRADE_CODES = {"correct": 0, "outdated": 1, "incorrect": 2, "abstain": 3}
_COARSE_STEP = 0.5  # model-level search: coarse τ grid, then refined at grid_step within ±_REFINE_HALF_WIDTH
_REFINE_HALF_WIDTH = 1.0
_COARSE_A_EVERY = 4  # coarse pass uses every 4th plateau value (0.55, 0.59, ..., 0.99), refined ±4 steps
_EB_MAX_ITER = 20
_METHOD = ("4PL grid MAP per model (g=0.25 for mcq) + per-cell grid posterior of tau with empirical-Bayes topic "
           "offsets (sigma_gamma=2, sigma_eps by moments); draws: paired cluster bootstrap of events within topic; "
           "hazard/abstain: two-level Beta shrinkage cell->model->pooled (5 pseudo-obs each) over items with "
           "t > tau + 2s")
MONTH_AXIS = "x = (year - 2024) * 12 + (month - 1); 2024-01 -> 0; day adds (day - 0.5) / days_in_month"


def eligible_event(event: Event) -> bool:
    """Events that count for cutoff curves: not a control, predictability low or medium."""
    return not event.control and event.category not in CONTROL_CATEGORIES and event.predictability != "high"


# ------------------------------------------------------------------------------------------------- data


@dataclass
class _Rows:
    """Graded results of one probe as parallel arrays."""

    model: np.ndarray  # [N] model position
    topic: np.ndarray  # [N] topic position, -1 for topics outside the index
    event: np.ndarray  # [N] position in the event list
    t: np.ndarray  # [N] fractional event month
    code: np.ndarray  # [N] 0 correct, 1 outdated, 2 incorrect, 3 abstain
    origin: np.ndarray  # [N] event origin (``Event.origin``: kc, arxiv_hf, synthetic, ...)

    @property
    def y(self) -> np.ndarray:
        return self.code == 0

    def step1_mask(self, global_origins: Sequence[str] | None) -> np.ndarray:
        """Rows that feed the model-level fit: all of them, or those whose event origin is in ``global_origins``."""
        if global_origins is None:
            return np.ones(len(self.t), dtype=bool)
        return np.isin(self.origin, np.asarray(list(global_origins), dtype=str))


def _latest(results: Sequence[ProbeResult]) -> list[ProbeResult]:
    """One result per (event, model, probe): the last one wins (resumed runs append retries)."""
    latest: dict[tuple[str, str, str], ProbeResult] = {}
    for r in results:
        latest[(r.event_id, r.model, r.probe)] = r
    return list(latest.values())


def _usable(events: Sequence[Event], results: Sequence[ProbeResult]) -> tuple[dict[str, Event], list[ProbeResult]]:
    by_id = {e.id: e for e in events}
    usable = [r for r in _latest(results)
              if r.grade in _GRADE_CODES and r.event_id in by_id and eligible_event(by_id[r.event_id])]
    return by_id, usable


def _event_time(event: Event) -> float:
    try:
        return parse_fractional_month(event.date)
    except ValueError as e:
        raise ValueError(f"event {event.id!r} has an unparseable date {event.date!r}") from e


def _prior_value(value: float | str) -> float:
    """A prior centre as a month index; strings like ``"2026-02"`` or ``"2026-02-20"`` are parsed."""
    return parse_fractional_month(value) if isinstance(value, str) else float(value)


# ------------------------------------------------------------------------------------------------- 4PL pieces


def _expit(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # overflow-free logistic


def _log_p(x: np.ndarray, a: float, g: float) -> np.ndarray:
    """log P(correct) for x = (τ - t)/s."""
    if g == 0.0:
        return math.log(a) - np.logaddexp(0.0, -x)  # log a + log σ(x), exact even for very negative x
    return np.log(g + (a - g) * _expit(x))  # >= log g


def _log_1mp(x: np.ndarray, a: float, g: float) -> np.ndarray:
    """log P(not correct); 1 - P >= 1 - a >= 0.01 on the plateau grid, so log1p is exact."""
    return np.log1p(-(g + (a - g) * _expit(x)))


def _row_loglik(t: np.ndarray, y: np.ndarray, grid: np.ndarray, s: float, a: float, g: float) -> np.ndarray:
    """Per-row log-likelihood over the τ grid: [n, G]."""
    out = np.empty((len(t), len(grid)))
    for mask, fn in ((y, _log_p), (~y, _log_1mp)):
        if mask.any():
            out[mask] = fn((grid[None, :] - t[mask, None]) / s, a, g)
    return out


def _fit_global(t: np.ndarray, y: np.ndarray, g: float, mu0: float, prior_sd: float,
                grid: np.ndarray) -> tuple[float, float, float]:
    """Model-level 4PL grid MAP ``(tau, s, a)`` on sufficient statistics per distinct event time."""
    u, inv = np.unique(t, return_inverse=True)
    k = np.bincount(inv, weights=y.astype(np.float64), minlength=len(u))
    n = np.bincount(inv, minlength=len(u)).astype(np.float64)
    s_grid, a_all = np.asarray(S_GRID), np.asarray(A_GRID)
    alpha, beta = PLATEAU_BETA
    log_prior_s = -0.5 * (np.log(s_grid) / LOG_S_SD) ** 2

    def objective(taus: np.ndarray, a_grid: np.ndarray) -> np.ndarray:  # log posterior [S, A, len(taus)]
        out = np.empty((len(s_grid), len(a_grid), len(taus)))
        for i, s in enumerate(s_grid):
            x = (taus[:, None] - u[None, :]) / s
            p = g + (a_grid - g)[:, None, None] * _expit(x)[None]  # [A, tau, U]
            wrong = np.log1p(-p) @ (n - k)
            if g == 0.0:  # log P = log a + log σ(x) keeps the far tail exact
                right = (-np.logaddexp(0.0, -x) @ k)[None, :] + np.log(a_grid)[:, None] * k.sum()
            else:
                right = np.log(p) @ k
            out[i] = right + wrong
        log_prior_a = (alpha - 1) * np.log(a_grid) + (beta - 1) * np.log1p(-a_grid)
        return (out + log_prior_s[:, None, None] + log_prior_a[None, :, None]
                - 0.5 * ((taus - mu0) / prior_sd)[None, None, :] ** 2)

    # coarse pass (τ step 0.5, every 4th plateau value), then the full resolution around its optimum
    coarse_tau = np.arange(TAU_MIN, TAU_MAX + 1e-9, _COARSE_STEP)
    coarse_a = np.arange(0, len(a_all), _COARSE_A_EVERY)
    obj = objective(coarse_tau, a_all[coarse_a])
    _, j, k_tau = np.unravel_index(np.argmax(obj), obj.shape)
    fine_tau = grid[np.abs(grid - coarse_tau[k_tau]) <= _REFINE_HALF_WIDTH + 1e-9]
    fine_a = a_all[max(coarse_a[j] - _COARSE_A_EVERY, 0): coarse_a[j] + _COARSE_A_EVERY + 1]
    obj = objective(fine_tau, fine_a)
    i, j, k_tau = np.unravel_index(np.argmax(obj), obj.shape)
    return float(fine_tau[k_tau]), float(s_grid[i]), float(fine_a[j])


# ------------------------------------------------------------------------------------------------- per-cell stage


def _softmax_moments(ll: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance of the grid distribution ∝ exp(ll) along the last axis (log-sum-exp normalised)."""
    w = np.exp(ll - ll.max(axis=-1, keepdims=True))
    w /= w.sum(axis=-1, keepdims=True)
    mean = w @ grid
    var = np.einsum("...g,...g->...", w, (grid - mean[..., None]) ** 2)
    return mean, var


def _cell_summaries(ll: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell MLE (grid argmax) and its squared standard error (variance of the normalised likelihood)."""
    _, var = _softmax_moments(ll, grid)
    return grid[np.argmax(ll, axis=-1)], var


def _gamma(mle: np.ndarray, se2: np.ndarray, mu: np.ndarray, has: np.ndarray, sigma_eps: float) -> np.ndarray:
    """Topic offsets: precision-weighted mean over models (axis 0) of ``mle - mu``, shrunk to 0 (σ_γ).

    Censored cells (likelihood SE of at least :data:`INFORMATIVE_SE`, e.g. a few items all answered correctly) are
    left out: their grid argmax sits at an edge of the τ grid, so ``mle - mu`` would measure the grid, not the data.
    """
    use = has & (se2 < INFORMATIVE_SE ** 2)
    w = np.where(use, 1.0 / (se2 + sigma_eps ** 2), 0.0)
    d = np.where(use, mle - mu, 0.0)
    return (w * d).sum(axis=0) / (w.sum(axis=0) + 1.0 / SIGMA_GAMMA ** 2)


def _posterior(ll: np.ndarray, prior_mean: np.ndarray, prior_var: np.ndarray,
               grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and variance of τ on the grid: likelihood × N(prior_mean, prior_var)."""
    lp = ll - 0.5 * (grid - prior_mean[..., None]) ** 2 / prior_var[..., None]
    return _softmax_moments(lp, grid)


@dataclass
class _Hyper:
    sigma_eps: float
    gamma: np.ndarray  # [T]
    mom_cells: int


def _empirical_bayes(mle: np.ndarray, se2: np.ndarray, mu: np.ndarray, has: np.ndarray, n: np.ndarray) -> _Hyper:
    """Fixed point of (γ_z, σ_ε) on the full data; arrays are [M, T], ``mu`` is [M, 1]."""
    informative = has & (n >= MIN_CELL_N) & (se2 < INFORMATIVE_SE ** 2)
    count = int(informative.sum())
    sigma = SIGMA_EPS_DEFAULT
    gamma = _gamma(mle, se2, mu, has, sigma)
    if count < MIN_MOM_CELLS:
        return _Hyper(sigma, gamma, count)
    for _ in range(_EB_MAX_ITER):
        dev = (mle - mu - gamma[None, :])[informative]
        var = float(np.mean(dev ** 2 - se2[informative]))
        new = float(np.clip(math.sqrt(max(var, 0.0)), *SIGMA_EPS_BOUNDS))
        gamma = _gamma(mle, se2, mu, has, new)
        if abs(new - sigma) < 1e-4:
            sigma = new
            break
        sigma = new
    return _Hyper(sigma, gamma, count)


# ------------------------------------------------------------------------------------------------- one probe


@dataclass
class _ProbeFit:
    tau: np.ndarray
    var: np.ndarray
    hazard: np.ndarray
    abstain: np.ndarray
    n: np.ndarray
    draws: np.ndarray | None
    global_tau: np.ndarray
    slope: np.ndarray
    plateau: np.ndarray
    hyper: _Hyper
    step1: np.ndarray  # [N] rows that fed the model-level fit (all rows unless global_origins restricts them)


def _fit_probe(rows: _Rows, g: float, n_models: int, n_topics: int, prior_mu: np.ndarray, prior_sd: float,
               bootstrap: int, rng: np.random.Generator, grid: np.ndarray,
               global_origins: Sequence[str] | None = None) -> _ProbeFit:
    M, T, G = n_models, n_topics, len(grid)
    y = rows.y

    # step 1: model-level 4PL (on the rows of the chosen origins only, when global_origins is given)
    step1 = rows.step1_mask(global_origins)
    global_tau = prior_mu.copy()
    slope = np.full(M, np.nan)
    plateau = np.full(M, np.nan)
    for m in range(M):
        sel = step1 & (rows.model == m)
        if sel.any():
            mu0 = prior_mu[m] if np.isfinite(prior_mu[m]) else float(np.median(rows.t[sel]))
            global_tau[m], slope[m], plateau[m] = _fit_global(rows.t[sel], y[sel], g, mu0, prior_sd, grid)
    fitted = np.isfinite(slope)  # models with step-1 data (without global_origins: models with data)
    known = np.isfinite(global_tau)  # models with data or a prior

    # step 2: cell likelihoods (rows are grouped by topic for the bootstrap). A model without step-1 rows has no
    # slope/plateau to evaluate cell likelihoods with: it is treated as having no data (its rows count nowhere).
    in_index = rows.topic >= 0
    n = np.zeros((M, T), dtype=np.int64)
    np.add.at(n, (rows.model[in_index], rows.topic[in_index]), 1)
    n[~fitted] = 0
    cell_rows = {(m, z): np.flatnonzero(in_index & (rows.model == m) & (rows.topic == z))
                 for m in range(M) for z in range(T) if n[m, z] > 0}

    def cell_loglik(m: int, z: int) -> np.ndarray:
        idx = cell_rows[(m, z)]
        return _row_loglik(rows.t[idx], y[idx], grid, slope[m], plateau[m], g)

    ll_full = np.zeros((M, T, G))
    for (m, z) in cell_rows:
        ll_full[m, z] = cell_loglik(m, z).sum(axis=0)
    has = n > 0
    mu = np.where(known, global_tau, 0.0)[:, None]
    mle, se2 = _cell_summaries(ll_full, grid)
    hyper = _empirical_bayes(mle, se2, mu, has, n)
    prior_var = np.where(fitted, hyper.sigma_eps ** 2, hyper.sigma_eps ** 2 + prior_sd ** 2)[:, None]
    prior_mean = mu + hyper.gamma[None, :]
    tau, var = _posterior(ll_full, prior_mean, np.broadcast_to(prior_var, (M, T)), grid)
    tau = np.where(known[:, None], tau, np.nan)
    var = np.where(known[:, None], var, np.nan)

    # step 3: paired cluster bootstrap within topics
    draws = None
    if bootstrap > 0:
        draws = np.empty((bootstrap, M, T), dtype=np.float64)
        thin = np.zeros(T, dtype=bool)
        for z in range(T):
            members = np.unique(rows.event[in_index & (rows.topic == z)])
            thin[z] = len(members) < MIN_BOOTSTRAP_EVENTS
            ll_b = np.zeros((M, bootstrap, G))
            weight = np.zeros((M, bootstrap))
            if len(members):
                counts = rng.multinomial(len(members), np.full(len(members), 1.0 / len(members)),
                                         size=bootstrap).astype(np.float64)  # [B, events of topic z]
                for m in range(M):
                    if (m, z) in cell_rows:
                        w = counts[:, np.searchsorted(members, rows.event[cell_rows[(m, z)]])]
                        ll_b[m] = w @ cell_loglik(m, z)
                        weight[m] = w.sum(axis=1)
            has_b = weight > 0
            mle_b, se2_b = _cell_summaries(ll_b, grid)
            gamma_b = _gamma(mle_b, se2_b, mu, has_b, hyper.sigma_eps)  # [B]
            mean_b, _ = _posterior(ll_b, mu + gamma_b[None, :], np.broadcast_to(prior_var, (M, bootstrap)), grid)
            draws[:, :, z] = mean_b.T
        # no data, or too few events for resampling to show the uncertainty (one event: every resample is the same):
        # draw from the cell's posterior instead
        empty = (~has | thin[None, :]) & known[:, None]
        noise = rng.standard_normal((bootstrap, M, T))
        draws = np.where(empty[None], tau[None] + np.sqrt(np.where(empty, var, 0.0))[None] * noise, draws)
        draws = np.where(known[None, :, None], draws, np.nan)

    hazard, abstain = _rates(rows, tau, slope, M, T)
    return _ProbeFit(tau, var, hazard, abstain, n, draws, global_tau, slope, plateau, hyper, step1)


def _rates(rows: _Rows, tau: np.ndarray, slope: np.ndarray, M: int, T: int) -> tuple[np.ndarray, np.ndarray]:
    """Shrunk stale-assertion hazard and abstain rate over items after each cell's cutoff (step 4).

    Two-level Beta shrinkage with ``HAZARD_PSEUDO`` pseudo-observations at each level: cell -> model -> pooled.
    """
    sel = rows.topic >= 0
    m, z = rows.model[sel], rows.topic[sel]
    threshold = tau[m, z] + 2.0 * slope[m]
    after = np.isfinite(threshold) & (rows.t[sel] > np.nan_to_num(threshold, nan=np.inf))
    code = rows.code[sel]
    flat = m * T + z
    n_after = np.bincount(flat, weights=after.astype(float), minlength=M * T).reshape(M, T)
    bad = np.bincount(flat, weights=(after & ((code == 1) | (code == 2))).astype(float),
                      minlength=M * T).reshape(M, T)
    abst = np.bincount(flat, weights=(after & (code == 3)).astype(float), minlength=M * T).reshape(M, T)
    out = []
    for k in (bad, abst):
        n_m, k_m = n_after.sum(axis=1), k.sum(axis=1)
        pooled = (k.sum() + 1.0) / (n_after.sum() + 2.0) if n_after.sum() > 0 else np.nan  # Beta(1, 1)
        base = (k_m + HAZARD_PSEUDO * pooled) / (n_m + HAZARD_PSEUDO)  # model-level rate, shrunk to the pooled one
        rate = (k + HAZARD_PSEUDO * base[:, None]) / (n_after + HAZARD_PSEUDO)
        out.append(np.where(np.isfinite(tau), rate, np.nan))
    return out[0], out[1]


# ------------------------------------------------------------------------------------------------- public API


def fit_index(
    events: Sequence[Event],
    results: Sequence[ProbeResult],
    *,
    models: Sequence[str] | None = None,
    topics: Sequence[str] | None = None,
    probes: Sequence[str] | None = None,
    priors: Mapping[str, float | str] | None = None,
    prior_sd: float = 3.0,
    bootstrap: int = 200,
    seed: int = 0,
    grid_step: float = 0.05,
    global_origins: Sequence[str] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> CutoffIndex:
    """Fit the per-(model, topic) cutoff index from graded probe results (see the module docstring, DESIGN.md §4).

    Excluded: controls (``control`` flag, ``control_alive``/``fake_event``), high-predictability events, ``ungraded``
    rows and results whose event is unknown. For repeated (event, model, probe) rows the last one counts.

    ``models`` default: models in the results (first-seen order); ``topics`` default: the 15 taxonomy topics plus
    any other topic present; ``probes`` default: probes present (``current_d, current, recall, mcq`` order).
    ``priors`` maps model key -> prior centre of the model-level τ (month index or ``"YYYY-MM[-DD]"``); without one
    the median event month of the model's rows is used. A model with neither rows nor a prior gets NaN cells
    (never chosen by the router). Probes without any usable row are left out; ``ValueError`` if none is left.
    Draws are seeded per probe, so the same ``seed`` and data give the same index.

    ``global_origins`` restricts the *model-level* fit (τ_m, slope, plateau; step 1) to rows whose event origin is in
    that set, e.g. ``["kc"]`` to anchor every model on the world-event evidence while AI-paper probes (origin
    ``arxiv_hf``, topic ``ai_ml`` only, a different plateau and knowledge structure) still shape their own topic
    cell: the per-cell posteriors, γ_z/σ_ε, the bootstrap and the hazard/abstain rates use all rows. A model with no
    rows from those origins for a probe falls back to the prior exactly as a model without data (its other rows
    count nowhere for that probe). ``None`` (the default) uses all rows everywhere. ``ValueError`` if no usable row
    at all comes from the chosen origins (a typo, most likely).

    ``index.meta`` records provenance: ``sources`` (the attribution of every event origin used, most events first),
    ``events_by_origin``, ``data_range`` (first and last event date), ``global_origins`` and the fit settings;
    ``hyper[probe].model_level`` lists per model the rows (``n``, ``by_origin``) behind the model-level fit, the
    rows left out of it (``excluded``) and a ``fallback`` (``prior``/``none``) when it had none; entries of ``meta``
    (e.g. a ``note`` on what the index may be used for) are added last.
    """
    if prior_sd <= 0:
        raise ValueError(f"prior_sd must be > 0, got {prior_sd}")
    if bootstrap < 0:
        raise ValueError(f"bootstrap must be >= 0, got {bootstrap}")
    if not 0 < grid_step <= 1:
        raise ValueError(f"grid_step must be in (0, 1], got {grid_step}")
    origin_list: list[str] | None = None
    if global_origins is not None:
        origin_list = list(dict.fromkeys(global_origins))
        if not origin_list or not all(isinstance(o, str) and o for o in origin_list):
            raise ValueError(f"global_origins must name at least one event origin, got {list(global_origins)!r}")
    by_id, usable = _usable(events, results)
    model_list = list(models) if models is not None else list(dict.fromkeys(r.model for r in usable))
    present_topics = sorted({by_id[r.event_id].topic for r in usable} - set(TOPIC_IDS))
    topic_list = list(topics) if topics is not None else [*TOPIC_IDS, *present_topics]
    present_probes = {r.probe for r in usable}
    probe_list = (list(probes) if probes is not None
                  else [p for p in PROBE_ORDER if p in present_probes] + sorted(present_probes - set(PROBE_ORDER)))
    if not model_list or not topic_list:
        raise ValueError("no models or topics to fit (no usable graded results?)")

    model_pos = {k: i for i, k in enumerate(model_list)}
    topic_pos = {k: i for i, k in enumerate(topic_list)}
    used = [r for r in usable if r.model in model_pos and r.probe in probe_list]
    event_ids = sorted({r.event_id for r in used})
    event_pos = {e: i for i, e in enumerate(event_ids)}
    times = {e: _event_time(by_id[e]) for e in event_ids}
    if origin_list is not None:
        present = {by_id[e].origin for e in event_ids}
        if not present & set(origin_list):
            raise ValueError(f"no usable result has an event of origin {origin_list} for the model-level fit; "
                             f"origins present: {sorted(present)}")
    prior_map = {k: _prior_value(v) for k, v in (priors or {}).items()}
    prior_mu = np.array([prior_map.get(m, np.nan) for m in model_list], dtype=np.float64)
    grid = np.round(np.arange(TAU_MIN, TAU_MAX + grid_step / 2, grid_step), 10)

    fitted: dict[str, Any] = {}
    hyper_meta: dict[str, Any] = {}
    for probe in probe_list:
        sel = [r for r in used if r.probe == probe]
        if not sel:
            continue
        rows = _Rows(
            model=np.array([model_pos[r.model] for r in sel], dtype=np.int64),
            topic=np.array([topic_pos.get(by_id[r.event_id].topic, -1) for r in sel], dtype=np.int64),
            event=np.array([event_pos[r.event_id] for r in sel], dtype=np.int64),
            t=np.array([times[r.event_id] for r in sel], dtype=np.float64),
            code=np.array([_GRADE_CODES[r.grade] for r in sel], dtype=np.int8),
            origin=np.array([by_id[r.event_id].origin for r in sel], dtype=str),
        )
        rng = np.random.default_rng([seed, zlib.crc32(probe.encode("utf-8"))])
        g = MCQ_FLOOR if probe == "mcq" else 0.0
        fit = _fit_probe(rows, g, len(model_list), len(topic_list), prior_mu, prior_sd, bootstrap, rng, grid,
                         origin_list)
        fitted[probe] = ProbeIndex(probe=probe, tau=fit.tau, var=fit.var, hazard=fit.hazard, abstain=fit.abstain,
                                   n=fit.n, draws=fit.draws, global_tau=fit.global_tau, slope=fit.slope,
                                   plateau=fit.plateau)
        hyper_meta[probe] = {"sigma_eps": round(fit.hyper.sigma_eps, 4), "mom_cells": fit.hyper.mom_cells,
                             "gamma": {t: round(float(v), 4) for t, v in zip(topic_list, fit.hyper.gamma, strict=True)},
                             "n_results": len(sel), "floor": g,
                             "model_level": _model_level_meta(rows, fit.step1, model_list, prior_mu)}
    if not fitted:
        raise ValueError("no usable graded results for the requested models/probes")
    used_in_fit = [r for r in used if r.probe in fitted]
    used_events = {r.event_id for r in used_in_fit}
    by_origin = Counter(by_id[e].origin for e in used_events).most_common()
    dates = sorted(by_id[e].date for e in used_events)
    info: dict[str, Any] = {
        "created": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "sources": [ORIGIN_ATTRIBUTION.get(origin, origin) for origin, _ in by_origin],
        "events_by_origin": dict(by_origin),
        "data_range": {"first": dates[0], "last": dates[-1]},
        "n_events": len(used_events),
        "n_results": len(used_in_fit),
        "probes": list(fitted),
        "method": _METHOD,
        "priors": {m: prior_map[m] for m in model_list if m in prior_map},
        "prior_sd": prior_sd,
        "bootstrap": bootstrap,
        "seed": seed,
        "grid_step": grid_step,
        "global_origins": origin_list,
        "month_axis": MONTH_AXIS,
        "hyper": hyper_meta,
    }
    info.update(meta or {})
    return CutoffIndex(model_list, topic_list, fitted, info)


def _model_level_meta(rows: _Rows, step1: np.ndarray, models: Sequence[str],
                      prior_mu: np.ndarray) -> dict[str, dict[str, Any]]:
    """Per model: the rows behind the model-level fit (``n``, ``by_origin``), the model's rows left out of it
    (``excluded``) and, without any, the ``fallback`` (``prior`` or ``none`` = NaN cells)."""
    out: dict[str, dict[str, Any]] = {}
    for m, model in enumerate(models):
        mine = rows.model == m
        sel = step1 & mine
        entry: dict[str, Any] = {"n": int(sel.sum()),
                                 "by_origin": dict(Counter(rows.origin[sel].tolist()).most_common()),
                                 "excluded": int((mine & ~step1).sum())}
        if not entry["n"]:
            entry["fallback"] = "prior" if np.isfinite(prior_mu[m]) else "none"
        out[model] = entry
    return out


def kc_style_cutoffs(
    events: Sequence[Event],
    results: Sequence[ProbeResult],
    *,
    threshold: float = 0.5,
    min_tail: int = 2,
) -> dict[str, dict[str, dict[str, Any]]]:
    """apoorvumang/knowledge-cutoff style cutoffs ``{model: {probe: summary}}`` for comparison with the 4PL fit.

    Monthly ``known_rate = correct / n`` over eligible events (ungraded rows dropped). ``last_above`` is the latest
    month with known_rate >= threshold; ``crossover`` the latest such month followed by at least ``min_tail``
    months that all stay below it. ``cutoff_status``: ``no_data``, ``before_window`` (never above),
    ``never_below`` (crossover = last month + "+"), ``no_tail`` (dips and recovers) or ``pinned``.
    """
    by_id, usable = _usable(events, results)
    tallies: dict[tuple[str, str], dict[str, Counter[str]]] = {}
    for r in usable:
        event = by_id[r.event_id]
        month = event.month or event.date[:7]
        tallies.setdefault((r.model, r.probe), {}).setdefault(month, Counter())[r.grade] += 1
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for (model, probe), months in tallies.items():
        curve = []
        for month in sorted(months):
            c = months[month]
            total = sum(c.values())
            curve.append({"month": month, "n": total, "known_rate": c["correct"] / total,
                          "wrong_rate": (c["outdated"] + c["incorrect"]) / total,
                          "abstain_rate": c["abstain"] / total})
        out.setdefault(model, {})[probe] = _kc_summary(curve, threshold, min_tail)
    return out


def _kc_summary(curve: list[dict[str, Any]], threshold: float, min_tail: int) -> dict[str, Any]:
    above = [c["month"] for c in curve if c["known_rate"] >= threshold]
    crossover: str | None = None
    status = "pinned"
    if not curve:
        status = "no_data"
    elif not above:
        status = "before_window"
    elif len(above) == len(curve):
        status, crossover = "never_below", curve[-1]["month"] + "+"
    else:
        for i, c in enumerate(curve):
            tail = curve[i + 1:]
            if c["known_rate"] >= threshold and len(tail) >= min_tail and all(d["known_rate"] < threshold
                                                                               for d in tail):
                crossover = c["month"]
        if crossover is None:
            status = "no_tail"
    return {"curve": curve, "threshold": threshold, "last_above": above[-1] if above else None,
            "crossover": crossover, "cutoff_status": status, "n_events": sum(c["n"] for c in curve)}
