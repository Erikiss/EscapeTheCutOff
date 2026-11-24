import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cutoffmix.router.features import SparseSoftmaxRegression, hashed_features, softmax, tokenize


def test_tokenize_is_lowercase_and_unicode_aware():
    assert tokenize("Außenminister Müller trifft GRÖßE") == ["außenminister", "müller", "trifft", "größe"]
    assert tokenize("GPT-6.2 Sol, snake_case!") == ["gpt", "6", "2", "sol", "snake", "case"]
    assert tokenize("") == []


def test_hashed_features_are_l2_normalised_and_bounded():
    f = hashed_features("The war in Ukraine; der Krieg in der Ukraine.", dim=1024)
    assert f and all(0 <= k < 1024 for k in f)
    assert math.isclose(math.sqrt(sum(v * v for v in f.values())), 1.0, rel_tol=1e-9)
    assert hashed_features("", dim=1024) == {}
    assert hashed_features("abc") != hashed_features("xyz")


def test_hashed_features_include_char_ngrams():
    # inflected forms share character n-grams, so their vectors overlap
    a, b = hashed_features("Landtagswahl"), hashed_features("Landtagswahlen")
    assert sum(a[k] * b.get(k, 0.0) for k in a) > 0.5


def test_hashed_features_are_stable_across_processes():
    code = (
        "from cutoffmix.router.features import hashed_features\n"
        "print(sorted(hashed_features('Wer ist Kanzler?', 4096).items()))"
    )
    outs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True,
                             cwd=Path(__file__).resolve().parents[1])
        outs.add(res.stdout)
    assert len(outs) == 1
    assert outs.pop().strip() == str(sorted(hashed_features("Wer ist Kanzler?", 4096).items()))


def test_hashed_features_rejects_bad_dim():
    with pytest.raises(ValueError):
        hashed_features("x", dim=0)


def test_softmax_is_stable():
    p = softmax(np.array([1e6, 1e6 - 1.0, -1e6]))
    assert np.isfinite(p).all() and math.isclose(p.sum(), 1.0)
    assert p[0] > p[1] > p[2]


def test_sparse_softmax_regression_learns_and_round_trips():
    texts = ["red apple", "green apple", "red cherry", "blue sky", "blue sea", "grey sky"] * 5
    labels = [0, 0, 0, 1, 1, 1] * 5
    X = [hashed_features(t, 512) for t in texts]
    model = SparseSoftmaxRegression(512, 2).fit(X, labels, epochs=20, lr=0.5, seed=1)
    preds = [int(np.argmax(model.predict_proba(x))) for x in X]
    assert preds == labels
    copy = SparseSoftmaxRegression.from_arrays(model.to_arrays())
    assert np.allclose(copy.predict_proba(X[0]), model.predict_proba(X[0]))
    assert math.isclose(float(model.predict_proba({}).sum()), 1.0)


def test_sparse_softmax_regression_validates_input():
    m = SparseSoftmaxRegression(16, 2)
    with pytest.raises(ValueError):
        m.fit([{1: 1.0}], [2])
    with pytest.raises(ValueError):
        m.fit([{1: 1.0}], [0, 1])
    with pytest.raises(ValueError):
        m.fit([{}], [0])
    with pytest.raises(ValueError):
        m.fit([{99: 1.0}], [0])
    with pytest.raises(ValueError):
        SparseSoftmaxRegression(16, 1)
    with pytest.raises(ValueError):
        SparseSoftmaxRegression.from_arrays({"dim": np.asarray(16)})
