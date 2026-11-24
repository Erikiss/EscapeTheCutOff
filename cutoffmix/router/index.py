"""The cutoff index: per (model, topic) virtual knowledge cutoff estimates and the routing score built on them.

Month axis as in :mod:`cutoffmix.timeutil` (0 = 2024-01). For each probe kind the index holds, per model ``m`` and
topic ``z``, the posterior mean cutoff ``tau[m, z]``, its variance, the stale-assertion hazard, the abstain rate, the
number of graded items and optionally posterior/bootstrap draws ``draws[b, m, z]`` (fitted by
``cutoffmix.bench.estimate.fit_index``).

Window score for a topic posterior ``π``::

    U_m = Σ_z π_z τ_mz − κ·sqrt(Σ_z π_z² Var_mz) − λ·Σ_z π_z H_mz

Topics of ``π`` that the index lacks use the model's ``global_tau`` with the model's mean variance and mean hazard;
models the index lacks score ``-inf``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cutoffmix.timeutil import month_label

__all__ = ["DEFAULT_PROBE_PREFERENCE", "PACKAGED_INDEX_PATH", "CutoffIndex", "ProbeIndex", "load_index"]

DEFAULT_PROBE_PREFERENCE: tuple[str, ...] = ("current_d", "current", "recall", "mcq")
PACKAGED_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index_prior.json"

_FORMAT = "cutoffmix.index"
_VERSION = 1
_DRAW_DECIMALS = 3
_VALUE_DECIMALS = 5
_FALLBACK_VAR = 9.0  # (3 months)^2, the estimator's prior when a model has no finite variance at all
_P_BEST_SAMPLES = 4000  # Monte Carlo size for p_best when the index has no draws
_MATRIX_FIELDS = ("tau", "var", "hazard", "abstain", "n")
_VECTOR_FIELDS = ("global_tau", "slope", "plateau")


@dataclass
class ProbeIndex:
    """Index arrays for one probe kind. ``M`` models x ``T`` topics, ``B`` draws."""

    probe: str
    tau: np.ndarray  # [M,T] posterior mean virtual cutoff (month index)
    var: np.ndarray  # [M,T]
    hazard: np.ndarray  # [M,T] P(outdated|incorrect | item after cutoff)
    abstain: np.ndarray  # [M,T]
    n: np.ndarray  # [M,T] graded items used
    draws: np.ndarray | None  # [B,M,T] posterior/bootstrap draws of tau
    global_tau: np.ndarray  # [M]
    slope: np.ndarray  # [M]
    plateau: np.ndarray  # [M]

    def __post_init__(self) -> None:
        # copies: the index owns its arrays, so later edits to the caller's arrays cannot change routing
        for name in ("tau", "var", "hazard", "abstain", "global_tau", "slope", "plateau"):
            setattr(self, name, np.array(getattr(self, name), dtype=np.float64))
        self.n = np.array(self.n, dtype=np.int64)
        if self.draws is not None:
            self.draws = np.array(self.draws, dtype=np.float32)
        if self.tau.ndim != 2:
            raise ValueError(f"probe {self.probe!r}: tau must be [M,T], got shape {self.tau.shape}")
        shape = self.tau.shape
        for name in _MATRIX_FIELDS:
            if getattr(self, name).shape != shape:
                raise ValueError(f"probe {self.probe!r}: {name} has shape {getattr(self, name).shape}, expected {shape}")
        for name in _VECTOR_FIELDS:
            if getattr(self, name).shape != (shape[0],):
                raise ValueError(f"probe {self.probe!r}: {name} has shape {getattr(self, name).shape}, "
                                 f"expected ({shape[0]},)")
        if self.draws is not None and (self.draws.ndim != 3 or self.draws.shape[1:] != shape or not len(self.draws)):
            raise ValueError(f"probe {self.probe!r}: draws must be [B>0,{shape[0]},{shape[1]}], "
                             f"got {self.draws.shape}")

    @property
    def shape(self) -> tuple[int, int]:
        """``(M, T)``."""
        return self.tau.shape  # type: ignore[return-value]

    def filled(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(tau, var, hazard, mean_var[M], mean_hazard[M])`` with non-finite cells replaced by model-level values.

        A missing ``tau`` becomes ``global_tau``, a missing variance the model's mean variance (or 9 = 3² months²
        when the model has none), a missing hazard the model's mean hazard (or 0).
        """
        mean_var = _row_nanmean(self.var, _FALLBACK_VAR)
        mean_hazard = _row_nanmean(self.hazard, 0.0)
        tau = np.where(np.isfinite(self.tau), self.tau, self.global_tau[:, None])
        var = np.where(np.isfinite(self.var), self.var, mean_var[:, None])
        hazard = np.where(np.isfinite(self.hazard), self.hazard, mean_hazard[:, None])
        return tau, np.maximum(var, 0.0), hazard, mean_var, mean_hazard

    def to_json(self) -> dict[str, Any]:
        """JSON-ready dict; draws are rounded to 3 decimals, other floats to 5; non-finite values become null."""
        d: dict[str, Any] = {"probe": self.probe}
        for name in ("tau", "var", "hazard", "abstain", "global_tau", "slope", "plateau"):
            d[name] = _float_list(getattr(self, name), _VALUE_DECIMALS)
        d["n"] = self.n.tolist()
        d["draws"] = None if self.draws is None else _float_list(self.draws, _DRAW_DECIMALS)
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ProbeIndex":
        """Inverse of :meth:`to_json`."""
        try:
            arrays = {name: np.asarray(d[name], dtype=np.float64) for name in
                      ("tau", "var", "hazard", "abstain", "global_tau", "slope", "plateau")}
            n = np.asarray(d["n"], dtype=np.int64)
            draws = None if d.get("draws") is None else np.asarray(d["draws"], dtype=np.float32)
            probe = str(d["probe"])
        except KeyError as e:
            raise ValueError(f"probe index entry is missing field {e}") from e
        return cls(probe=probe, n=n, draws=draws, **arrays)


def _row_nanmean(a: np.ndarray, default: float) -> np.ndarray:
    finite = np.isfinite(a)
    counts = finite.sum(axis=1)
    sums = np.where(finite, a, 0.0).sum(axis=1)
    return np.where(counts > 0, sums / np.maximum(counts, 1), default)


def _float_list(a: np.ndarray, decimals: int) -> list[Any]:
    rounded = np.round(np.asarray(a, dtype=np.float64), decimals)
    if np.isfinite(rounded).all():
        return rounded.tolist()
    return np.where(np.isfinite(rounded), rounded, None).tolist()  # object array: NaN/inf -> None


class CutoffIndex:
    """Per-probe cutoff estimates for ``models`` x ``topics`` and the routing scores derived from them."""

    def __init__(self, models: list[str], topics: list[str], probes: dict[str, ProbeIndex], meta: dict | None = None):
        self.models: list[str] = list(models)
        self.topics: list[str] = list(topics)
        self.probes: dict[str, ProbeIndex] = dict(probes)
        self.meta: dict[str, Any] = dict(meta or {})
        if not self.models or not self.topics:
            raise ValueError("an index needs at least one model and one topic")
        if len(set(self.models)) != len(self.models) or len(set(self.topics)) != len(self.topics):
            raise ValueError("model and topic names must be unique")
        if not self.probes:
            raise ValueError("an index needs at least one probe")
        for key, p in self.probes.items():
            if p.probe != key:
                raise ValueError(f"probe stored under {key!r} is named {p.probe!r}")
            if p.shape != (len(self.models), len(self.topics)):
                raise ValueError(f"probe {key!r} has shape {p.shape}, expected {(len(self.models), len(self.topics))}")
        self._model_pos = {m: i for i, m in enumerate(self.models)}
        self._topic_pos = {t: i for i, t in enumerate(self.topics)}

    # ------------------------------------------------------------------------------------------------- lookup

    def primary(self, preference: Sequence[str] = DEFAULT_PROBE_PREFERENCE) -> ProbeIndex:
        """The first probe of ``preference`` that the index has, else the first stored probe."""
        for name in preference:
            if name in self.probes:
                return self.probes[name]
        return next(iter(self.probes.values()))

    def _probe(self, probe: str | None) -> ProbeIndex:
        if probe is None:
            return self.primary()
        try:
            return self.probes[probe]
        except KeyError:
            raise KeyError(f"probe {probe!r} not in index; available: {', '.join(self.probes)}") from None

    def _weights(self, posterior: Mapping[str, float]) -> tuple[np.ndarray, float, float]:
        """``(π over index topics [T], mass of absent topics, Σ π² over absent topics)``, normalised to sum 1.

        Negative and non-finite entries count as 0; an all-zero posterior becomes uniform over the index topics.
        """
        pi = np.zeros(len(self.topics))
        absent: list[float] = []
        for topic, p in posterior.items():
            v = float(p)
            if not math.isfinite(v) or v <= 0:
                continue
            pos = self._topic_pos.get(topic)
            if pos is None:
                absent.append(v)
            else:
                pi[pos] += v
        total = pi.sum() + sum(absent)
        if total <= 0:
            return np.full(len(self.topics), 1.0 / len(self.topics)), 0.0, 0.0
        return pi / total, sum(absent) / total, sum((a / total) ** 2 for a in absent)

    def _select(self, models: Iterable[str] | None) -> list[str]:
        return list(self.models if models is None else models)

    # ------------------------------------------------------------------------------------------------- scores

    def scores(
        self,
        posterior: Mapping[str, float],
        *,
        probe: str | None = None,
        kappa: float = 1.0,
        lam: float = 2.0,
        models: list[str] | None = None,
    ) -> dict[str, float]:
        """``U_m`` for each requested model (``-inf`` for models the index lacks)."""
        p = self._probe(probe)
        pi, a_mass, a_sq = self._weights(posterior)
        tau, var, hazard, mean_var, mean_hazard = p.filled()
        mean = tau @ pi + a_mass * p.global_tau
        sd = np.sqrt(var @ (pi * pi) + a_sq * mean_var)
        haz = hazard @ pi + a_mass * mean_hazard
        u = mean - kappa * sd - lam * haz
        return {m: self._finite_or_ninf(u, m) for m in self._select(models)}

    def sample_scores(
        self,
        posterior: Mapping[str, float],
        rng: np.random.Generator,
        *,
        probe: str | None = None,
        models: list[str] | None = None,
    ) -> dict[str, float]:
        """One Thompson draw of ``Σ_z π_z τ^(b)_mz`` per requested model.

        With draws, one draw index ``b`` is shared by all models (keeps their correlation); without, the sum is
        drawn from its Normal(tau, var) distribution. Mass on topics the index lacks is drawn from
        Normal(global_tau, mean var).
        """
        p = self._probe(probe)
        pi, a_mass, a_sq = self._weights(posterior)
        tau, var, _, mean_var, _ = p.filled()
        absent = a_mass * p.global_tau + np.sqrt(a_sq * mean_var) * rng.standard_normal(len(self.models))
        if p.draws is not None:
            draw = p.draws[int(rng.integers(len(p.draws)))].astype(np.float64)
            draw = np.where(np.isfinite(draw), draw, tau)
            s = draw @ pi + absent
        else:
            s = tau @ pi + np.sqrt(var @ (pi * pi)) * rng.standard_normal(len(self.models)) + absent
        return {m: self._finite_or_ninf(s, m) for m in self._select(models)}

    def p_best(
        self, posterior: Mapping[str, float], *, probe: str | None = None, models: list[str] | None = None
    ) -> dict[str, float]:
        """``P(m is best | π)``: share of draws in which ``m`` has the largest ``Σ_z π_z τ^(b)_mz``.

        Only the requested models compete; models the index lacks get 0. Without draws, a fixed-seed Monte Carlo
        sample from Normal(tau, var) stands in (deterministic).
        """
        selected = self._select(models)
        cols = [self._model_pos[m] for m in selected if m in self._model_pos]
        out = {m: 0.0 for m in selected}
        if not cols:
            return out
        p = self._probe(probe)
        pi, a_mass, a_sq = self._weights(posterior)
        tau, var, _, mean_var, _ = p.filled()
        if p.draws is not None:
            draws = p.draws.astype(np.float64)
            draws = np.where(np.isfinite(draws), draws, tau[None])
            s = draws @ pi + a_mass * p.global_tau[None, :]
        else:
            rng = np.random.default_rng(0)
            sd = np.sqrt(var @ (pi * pi) + a_sq * mean_var)
            s = tau @ pi + a_mass * p.global_tau + sd * rng.standard_normal((_P_BEST_SAMPLES, len(self.models)))
        s = np.where(np.isfinite(s), s, -np.inf)[:, cols]
        winners = np.bincount(np.argmax(s, axis=1), minlength=len(cols)) / len(s)
        for col, share in zip(cols, winners):
            out[self.models[col]] = float(share)
        return out

    def _finite_or_ninf(self, values: np.ndarray, model: str) -> float:
        """``values[model]`` as a float; ``-inf`` for models the index lacks or non-finite values."""
        pos = self._model_pos.get(model)
        if pos is None:
            return -math.inf
        v = float(values[pos])
        return v if math.isfinite(v) else -math.inf

    def table(self, probe: str | None = None) -> list[dict[str, Any]]:
        """Rows ``{topic, model, tau, tau_label, sd, n, hazard, abstain, p_best}`` for UI/CLI, topic-major order.

        ``p_best`` is the probability that the model is best on that topic alone.
        """
        p = self._probe(probe)
        tau, var, hazard, _, _ = p.filled()
        rows = []
        for ti, topic in enumerate(self.topics):
            best = self.p_best({topic: 1.0}, probe=p.probe)
            for mi, model in enumerate(self.models):
                t = float(tau[mi, ti])
                rows.append({
                    "topic": topic,
                    "model": model,
                    "tau": t,
                    "tau_label": month_label(t),
                    "sd": float(math.sqrt(var[mi, ti])),
                    "n": int(p.n[mi, ti]),
                    "hazard": float(hazard[mi, ti]),
                    "abstain": float(p.abstain[mi, ti]) if math.isfinite(p.abstain[mi, ti]) else None,
                    "p_best": best[model],
                })
        return rows

    # ------------------------------------------------------------------------------------------------- I/O

    def to_json(self) -> dict[str, Any]:
        """JSON-ready dict (see :meth:`ProbeIndex.to_json` for the rounding)."""
        return {
            "format": _FORMAT,
            "version": _VERSION,
            "models": self.models,
            "topics": self.topics,
            "meta": self.meta,
            "probes": {k: p.to_json() for k, p in self.probes.items()},
        }

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "CutoffIndex":
        """Inverse of :meth:`to_json`."""
        if d.get("format", _FORMAT) != _FORMAT:
            raise ValueError(f"not a cutoffmix index (format={d.get('format')!r})")
        if int(d.get("version", _VERSION)) > _VERSION:
            raise ValueError(f"index version {d.get('version')} is newer than supported version {_VERSION}")
        try:
            probes = {k: ProbeIndex.from_json(v) for k, v in d["probes"].items()}
            return cls(list(d["models"]), list(d["topics"]), probes, dict(d.get("meta") or {}))
        except KeyError as e:
            raise ValueError(f"index JSON is missing field {e}") from e

    def save(self, path: str | Path) -> None:
        """Write compact JSON to ``path`` (parent directories are created)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CutoffIndex":
        """Read an index written by :meth:`save`."""
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_json(json.load(f))


def load_index(path: str | None = None) -> CutoffIndex | None:
    """Load the index at ``path``; with ``None``, the packaged ``cutoffmix/data/index_prior.json`` if it exists.

    An explicit ``path`` that does not exist raises ``FileNotFoundError``.
    """
    if path is None:
        return CutoffIndex.load(PACKAGED_INDEX_PATH) if PACKAGED_INDEX_PATH.is_file() else None
    return CutoffIndex.load(path)
