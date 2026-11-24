import json
import math

import numpy as np
import pytest

from cutoffmix.router import index as index_mod
from cutoffmix.router.index import CutoffIndex, ProbeIndex, load_index

MODELS = ["fable", "sol", "neon"]
TOPICS = ["ai_ml", "sports", "government"]
TAU = np.array([[30.0, 26.0, 27.0], [28.0, 29.0, 28.5], [27.0, 27.5, 29.5]])
VAR = np.array([[0.25, 1.0, 0.5], [0.5, 0.25, 0.25], [1.0, 0.5, 0.25]])
HAZ = np.array([[0.1, 0.2, 0.0], [0.3, 0.1, 0.2], [0.0, 0.0, 0.4]])


def make_probe(name="current_d", draws=True, B=400, seed=0):
    rng = np.random.default_rng(seed)
    d = TAU[None] + rng.normal(0.0, np.sqrt(VAR)[None], size=(B, 3, 3)) if draws else None
    return ProbeIndex(probe=name, tau=TAU, var=VAR, hazard=HAZ, abstain=np.full((3, 3), 0.05), n=np.full((3, 3), 12),
                      draws=d, global_tau=np.array([27.8, 28.75, 28.0]), slope=np.ones(3), plateau=np.full(3, 0.9))


def make_index(draws=True, probes=("current_d",)):
    return CutoffIndex(MODELS, TOPICS, {p: make_probe(p, draws) for p in probes}, {"source": "test"})


def test_scores_match_formula():
    idx = make_index()
    post = {"ai_ml": 0.6, "sports": 0.4}
    pi = np.array([0.6, 0.4, 0.0])
    s = idx.scores(post, kappa=1.0, lam=2.0)
    for m, i in zip(MODELS, range(3)):
        expected = TAU[i] @ pi - math.sqrt(VAR[i] @ pi**2) - 2.0 * HAZ[i] @ pi
        assert s[m] == pytest.approx(expected)
    assert max(s, key=s.get) == "fable"
    s0 = idx.scores(post, kappa=0.0, lam=0.0)
    assert s0["sol"] == pytest.approx(TAU[1] @ pi)


def test_scores_normalise_and_handle_absent_topics_and_models():
    idx = make_index()
    a = idx.scores({"ai_ml": 3.0, "sports": 2.0})
    b = idx.scores({"ai_ml": 0.6, "sports": 0.4})
    assert a == pytest.approx(b)
    absent = idx.scores({"space_astronomy": 1.0}, kappa=1.0, lam=2.0)
    mean_var = VAR.mean(axis=1)
    mean_haz = HAZ.mean(axis=1)
    for i, m in enumerate(MODELS):
        assert absent[m] == pytest.approx([27.8, 28.75, 28.0][i] - math.sqrt(mean_var[i]) - 2 * mean_haz[i])
    s = idx.scores({"ai_ml": 1.0}, models=["sol", "gemini-4-hydrogen"])
    assert list(s) == ["sol", "gemini-4-hydrogen"] and s["gemini-4-hydrogen"] == -math.inf
    uniform = idx.scores({})
    assert uniform == pytest.approx(idx.scores({t: 1.0 for t in TOPICS}))


def test_nan_cells_fall_back_to_model_level_values():
    p = make_probe(draws=False)
    p.tau[0, 1] = np.nan
    p.var[0, 1] = np.nan
    idx = CutoffIndex(MODELS, TOPICS, {"current_d": p})
    s = idx.scores({"sports": 1.0}, kappa=1.0, lam=0.0)
    assert s["fable"] == pytest.approx(27.8 - math.sqrt((0.25 + 0.5) / 2))


def test_sample_scores_with_draws_uses_one_shared_draw():
    idx = make_index()
    rng = np.random.default_rng(3)
    draws = idx.probes["current_d"].draws.astype(np.float64)
    post = {"government": 1.0}
    s = idx.sample_scores(post, rng)
    matches = [b for b in range(len(draws)) if np.allclose(draws[b, :, 2], [s[m] for m in MODELS], atol=1e-5)]
    assert matches
    assert idx.sample_scores(post, rng, models=["neon", "x"])["x"] == -math.inf


def test_sample_scores_without_draws_follow_normal():
    idx = make_index(draws=False)
    rng = np.random.default_rng(0)
    samples = np.array([[d[m] for m in MODELS] for d in (idx.sample_scores({"sports": 1.0}, rng) for _ in range(4000))])
    assert samples.mean(axis=0) == pytest.approx(TAU[:, 1], abs=0.05)
    assert samples.std(axis=0) == pytest.approx(np.sqrt(VAR[:, 1]), abs=0.05)


def test_p_best_from_draws_matches_definition():
    idx = make_index()
    draws = idx.probes["current_d"].draws.astype(np.float64)
    post = {"ai_ml": 0.3, "government": 0.7}
    pi = np.array([0.3, 0.0, 0.7])
    winners = np.argmax(draws @ pi, axis=1)
    expected = {m: float(np.mean(winners == i)) for i, m in enumerate(MODELS)}
    assert idx.p_best(post) == pytest.approx(expected)
    assert sum(idx.p_best(post).values()) == pytest.approx(1.0)
    restricted = idx.p_best(post, models=["fable", "sol", "unknown"])
    assert restricted["unknown"] == 0.0 and sum(restricted.values()) == pytest.approx(1.0)
    assert idx.p_best({"ai_ml": 1.0})["fable"] > 0.95


def test_p_best_without_draws_is_deterministic():
    idx = make_index(draws=False)
    a = idx.p_best({"sports": 0.5, "government": 0.5})
    assert a == idx.p_best({"sports": 0.5, "government": 0.5})
    assert sum(a.values()) == pytest.approx(1.0)


def test_primary_probe_and_lookup():
    idx = make_index(probes=("mcq", "current", "recall"))
    assert idx.primary().probe == "current"
    assert idx.primary(("recall", "current")).probe == "recall"
    assert idx.primary(("nothing",)).probe == "mcq"
    assert idx.scores({"ai_ml": 1.0}, probe="mcq")
    with pytest.raises(KeyError):
        idx.scores({"ai_ml": 1.0}, probe="current_d")


def test_table_rows():
    rows = make_index().table()
    assert len(rows) == 9
    first = rows[0]
    assert set(first) >= {"topic", "model", "tau", "tau_label", "sd", "n", "hazard", "p_best"}
    assert first["topic"] == "ai_ml" and first["model"] == "fable" and first["tau_label"] == "2026-07"
    assert first["sd"] == pytest.approx(0.5) and first["n"] == 12
    per_topic = sum(r["p_best"] for r in rows if r["topic"] == "sports")
    assert per_topic == pytest.approx(1.0)


def test_json_round_trip_preserves_arrays(tmp_path):
    p = make_probe(B=1000)
    p.var[1, 2] = np.nan
    idx = CutoffIndex(MODELS, TOPICS, {"current_d": p, "mcq": make_probe("mcq", draws=False)}, {"n_events": 5})
    text = json.dumps(idx.to_json())
    assert "NaN" not in text
    back = CutoffIndex.from_json(json.loads(text))
    assert back.models == MODELS and back.topics == TOPICS and back.meta == {"n_events": 5}
    q = back.probes["current_d"]
    assert q.draws.dtype == np.float32 and q.draws.shape == (1000, 3, 3)
    assert np.allclose(q.draws, p.draws, atol=5e-4)
    assert np.allclose(q.tau, p.tau) and np.array_equal(q.n, p.n)
    assert np.isnan(q.var[1, 2]) and np.allclose(np.nan_to_num(q.var), np.nan_to_num(p.var))
    assert back.probes["mcq"].draws is None
    # draws are stored rounded to 3 decimals, so the JSON stays compact
    raw = np.array(json.loads(text)["probes"]["current_d"]["draws"], dtype=np.float64)
    assert np.allclose(raw, np.round(raw, 3), equal_nan=True)

    path = tmp_path / "deep" / "index.json"
    idx.save(path)
    loaded = CutoffIndex.load(path)
    assert loaded.scores({"ai_ml": 1.0}) == pytest.approx(back.scores({"ai_ml": 1.0}))


def test_validation_errors():
    with pytest.raises(ValueError):
        CutoffIndex(MODELS, TOPICS[:2], {"current_d": make_probe()})
    with pytest.raises(ValueError):
        CutoffIndex(MODELS, TOPICS, {"current": make_probe("current_d")})
    with pytest.raises(ValueError):
        CutoffIndex(MODELS, TOPICS, {})
    with pytest.raises(ValueError):
        ProbeIndex("x", TAU, VAR[:2], HAZ, HAZ, HAZ, None, np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        ProbeIndex("x", TAU, VAR, HAZ, HAZ, HAZ, np.zeros((2, 3, 2)), np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        CutoffIndex.from_json({"format": "other"})
    with pytest.raises(ValueError):
        CutoffIndex.from_json({"models": MODELS})


def test_load_index(tmp_path, monkeypatch):
    monkeypatch.setattr(index_mod, "PACKAGED_INDEX_PATH", tmp_path / "missing.json")
    assert load_index() is None
    packaged = tmp_path / "index_prior.json"
    make_index().save(packaged)
    monkeypatch.setattr(index_mod, "PACKAGED_INDEX_PATH", packaged)
    assert isinstance(load_index(), CutoffIndex)
    assert load_index(str(packaged)).models == MODELS
    with pytest.raises(FileNotFoundError):
        load_index(str(tmp_path / "nope.json"))
