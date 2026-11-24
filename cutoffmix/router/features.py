"""Text features shared by the router's learned components.

* :func:`tokenize` -- lowercase, unicode-aware word tokens (German umlauts and ß stay inside words).
* :func:`hashed_features` -- word uni+bigrams and per-word character 3-5grams, hashed into a fixed number of
  buckets with a process-independent hash (CRC32), sublinear tf, l2-normalised.
* :class:`SparseSoftmaxRegression` -- multinomial logistic regression over such sparse feature dicts, trained with
  minibatch Adagrad in pure numpy. It backs :class:`~cutoffmix.router.topics.LinearTopicClassifier` and the learned
  part of :class:`~cutoffmix.router.gate.TimeSensitivityGate`.
"""

from __future__ import annotations

import math
import re
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

__all__ = ["DEFAULT_DIM", "WORD_RE", "SparseSoftmaxRegression", "hashed_features", "softmax", "tokenize"]

DEFAULT_DIM = 2**18

# letters and digits of any script; "_" is a \w character but never part of a word here
WORD_RE = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into lowercase word tokens.

    Unicode-aware: ``"Außenminister Müller"`` -> ``["außenminister", "müller"]``; punctuation and hyphens separate
    tokens, so ``"GPT-6.2"`` -> ``["gpt", "6", "2"]``.
    """
    return WORD_RE.findall(text.lower())


def _bucket(feature: str, dim: int) -> int:
    return zlib.crc32(feature.encode("utf-8")) % dim


def hashed_features(text: str, dim: int = DEFAULT_DIM) -> dict[int, float]:
    """Hash ``text`` into a sparse, l2-normalised feature vector ``{bucket: value}``.

    Features: word unigrams, word bigrams and character 3-5grams of every word padded with ``<``/``>`` boundary
    markers (fastText style, robust to German compounds and inflection). Counts are damped to ``1 + ln(count)``.
    The hash is CRC32, so buckets are identical across processes and Python versions. Empty text gives ``{}``.
    """
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")
    tokens = tokenize(text)
    counts: dict[int, int] = {}

    def add(feature: str) -> None:
        h = _bucket(feature, dim)
        counts[h] = counts.get(h, 0) + 1

    for tok in tokens:
        add("w|" + tok)
        padded = "<" + tok + ">"
        for n in (3, 4, 5):
            for i in range(len(padded) - n + 1):
                add("c|" + padded[i : i + n])
    for a, b in zip(tokens, tokens[1:]):
        add("b|" + a + " " + b)
    if not counts:
        return {}
    values = {h: 1.0 + math.log(c) for h, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in values.values()))
    return {h: v / norm for h, v in values.items()}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = logits - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=-1, keepdims=True)


class _CSR:
    """Compressed sparse rows built from feature dicts (only what minibatch training needs)."""

    def __init__(self, rows: Sequence[Mapping[int, float]]):
        lengths = np.fromiter((len(r) for r in rows), dtype=np.int64, count=len(rows))
        self.indptr = np.zeros(len(rows) + 1, dtype=np.int64)
        np.cumsum(lengths, out=self.indptr[1:])
        nnz = int(self.indptr[-1])
        self.indices = np.empty(nnz, dtype=np.int64)
        self.data = np.empty(nnz, dtype=np.float32)
        for i, r in enumerate(rows):
            a, b = self.indptr[i], self.indptr[i + 1]
            self.indices[a:b] = np.fromiter(r.keys(), dtype=np.int64, count=len(r))
            self.data[a:b] = np.fromiter(r.values(), dtype=np.float32, count=len(r))

    def batch(self, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(indices, data, segment_starts)`` for the given rows (each row must be non-empty)."""
        starts = self.indptr[row_ids]
        ends = self.indptr[row_ids + 1]
        lengths = ends - starts
        seg = np.zeros(len(row_ids), dtype=np.int64)
        np.cumsum(lengths[:-1], out=seg[1:])
        gather = np.repeat(starts - seg, lengths) + np.arange(int(lengths.sum()), dtype=np.int64)
        return self.indices[gather], self.data[gather], seg


class SparseSoftmaxRegression:
    """Multinomial logistic regression on hashed sparse features (numpy, CPU).

    Weights ``W[dim, K]`` (float32) and bias ``b[K]``. Training is minibatch Adagrad on the mean cross-entropy with
    a lazy L2 penalty applied to the rows touched by each batch. Prediction for one text costs one gather of the
    rows of its active features.
    """

    def __init__(self, dim: int, n_classes: int):
        if dim <= 0 or n_classes < 2:
            raise ValueError(f"need dim > 0 and n_classes >= 2, got dim={dim}, n_classes={n_classes}")
        self.dim = dim
        self.n_classes = n_classes
        self.W = np.zeros((dim, n_classes), dtype=np.float32)
        self.b = np.zeros(n_classes, dtype=np.float64)

    def fit(
        self,
        X: Sequence[Mapping[int, float]],
        y: Sequence[int] | np.ndarray,
        *,
        epochs: int = 8,
        lr: float = 0.5,
        l2: float = 1e-6,
        batch_size: int = 32,
        seed: int = 0,
    ) -> SparseSoftmaxRegression:
        """Train in place on feature dicts ``X`` and integer class labels ``y`` and return ``self``.

        Rows without any feature carry no information for ``W`` and are skipped (the bias still learns the class
        balance from the others).
        """
        y_arr = np.asarray(y, dtype=np.int64)
        if len(X) != len(y_arr):
            raise ValueError(f"X and y differ in length: {len(X)} != {len(y_arr)}")
        if np.any((y_arr < 0) | (y_arr >= self.n_classes)):
            raise ValueError(f"labels must lie in [0, {self.n_classes})")
        keep = np.fromiter((len(r) > 0 for r in X), dtype=bool, count=len(X))
        rows = [r for r, k in zip(X, keep) if k]
        y_arr = y_arr[keep]
        if not rows:
            raise ValueError("no training example has any feature")
        for r in rows:
            for h in r:
                if not 0 <= h < self.dim:
                    raise ValueError(f"feature index {h} outside [0, {self.dim})")
        csr = _CSR(rows)
        rng = np.random.default_rng(seed)
        acc_w = np.zeros_like(self.W)
        acc_b = np.zeros_like(self.b)
        eps = 1e-8
        n = len(rows)
        for _ in range(epochs):
            order = rng.permutation(n)
            for lo in range(0, n, batch_size):
                ids = order[lo : lo + batch_size]
                idx, val, seg = csr.batch(ids)
                contrib = val[:, None] * self.W[idx]
                logits = np.add.reduceat(contrib, seg, axis=0).astype(np.float64) + self.b
                grad_out = softmax(logits)
                grad_out[np.arange(len(ids)), y_arr[ids]] -= 1.0
                grad_out /= len(ids)
                # d loss / d W[j] = sum over occurrences of feature j of value * grad_out[row]
                row_of = np.repeat(np.arange(len(ids)), np.diff(np.append(seg, len(idx))))
                per_nnz = val[:, None] * grad_out[row_of]
                sort = np.argsort(idx, kind="stable")
                sidx = idx[sort]
                first = np.flatnonzero(np.concatenate(([True], sidx[1:] != sidx[:-1])))
                uniq = sidx[first]
                g = np.add.reduceat(per_nnz[sort], first, axis=0)
                g += l2 * self.W[uniq]
                acc_w[uniq] += (g * g).astype(np.float32)
                self.W[uniq] -= (lr * g / (np.sqrt(acc_w[uniq]) + eps)).astype(np.float32)
                gb = grad_out.sum(axis=0)
                acc_b += gb * gb
                self.b -= lr * gb / (np.sqrt(acc_b) + eps)
        return self

    def logits(self, features: Mapping[int, float]) -> np.ndarray:
        """Class logits ``[K]`` for one feature dict."""
        if not features:
            return self.b.copy()
        idx = np.fromiter(features.keys(), dtype=np.int64, count=len(features))
        val = np.fromiter(features.values(), dtype=np.float64, count=len(features))
        return val @ self.W[idx] + self.b

    def predict_proba(self, features: Mapping[int, float]) -> np.ndarray:
        """Class probabilities ``[K]`` for one feature dict."""
        return softmax(self.logits(features))

    def to_arrays(self, prefix: str = "") -> dict[str, Any]:
        """Compact arrays for ``np.savez``: only the non-zero rows of ``W`` are stored."""
        rows = np.flatnonzero(np.any(self.W != 0, axis=1)).astype(np.int64)
        return {
            prefix + "dim": np.asarray(self.dim, dtype=np.int64),
            prefix + "rows": rows,
            prefix + "W_rows": self.W[rows],
            prefix + "b": self.b,
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str = "") -> SparseSoftmaxRegression:
        """Inverse of :meth:`to_arrays`."""
        try:
            dim = int(arrays[prefix + "dim"])
            rows = np.asarray(arrays[prefix + "rows"], dtype=np.int64)
            w_rows = np.asarray(arrays[prefix + "W_rows"], dtype=np.float32)
            b = np.asarray(arrays[prefix + "b"], dtype=np.float64)
        except KeyError as e:
            raise ValueError(f"not a saved SparseSoftmaxRegression (missing array {e})") from e
        model = cls(dim, len(b))
        if w_rows.shape != (len(rows), len(b)):
            raise ValueError(f"inconsistent saved weights: rows {rows.shape}, W_rows {w_rows.shape}, b {b.shape}")
        model.W[rows] = w_rows
        model.b[:] = b
        return model
