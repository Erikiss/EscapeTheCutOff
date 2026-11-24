from dataclasses import replace

import numpy as np
import pytest

from cutoffmix.config import RouterConfig, default_config
from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.index import CutoffIndex, ProbeIndex
from cutoffmix.router.policy import FixedPolicy, SoftmaxPolicy
from cutoffmix.router.router import RouteDecision, Router
from cutoffmix.router.topics import KeywordTopicClassifier
from cutoffmix.taxonomy import TOPIC_IDS

MODELS = ["claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"]


def make_index(draws: bool = True) -> CutoffIndex:
    """Sol freshest overall (28.75); Fable freshest on ai_ml (31); Neon freshest on space (31)."""
    T = len(TOPIC_IDS)
    tau = np.tile(np.array([[26.8], [28.75], [27.0]]), (1, T))
    tau[0, TOPIC_IDS.index("ai_ml")] = 31.0
    tau[2, TOPIC_IDS.index("space_astronomy")] = 31.0
    var = np.full((3, T), 0.2)
    rng = np.random.default_rng(0)
    d = tau[None] + rng.normal(0, np.sqrt(var)[None], (200, 3, T)) if draws else None
    probe = ProbeIndex("current_d", tau, var, np.zeros((3, T)), np.zeros((3, T)), np.full((3, T), 20), d,
                       np.array([26.8, 28.75, 27.0]), np.ones(3), np.full(3, 0.9))
    return CutoffIndex(MODELS, list(TOPIC_IDS), {"current_d": probe}, {})


def make_router(**kw) -> Router:
    cfg = kw.pop("cfg", RouterConfig())
    return Router(MODELS, index=kw.pop("index", make_index()), cfg=cfg, seed=kw.pop("seed", 7), **kw)


AI_WINDOW = "Who is the current CEO of Anthropic? The latest Claude model and GPT-6.2 benchmarks matter here."
MATH_WINDOW = "If 3x + 5 = 20, what is x? We solve the equation: 3x = 15, so x = 5. Prove the remainder is 0."


def test_decide_returns_complete_decision():
    router = make_router()
    d = router.decide(AI_WINDOW, current=None)
    assert isinstance(d, RouteDecision)
    assert d.model in MODELS and d.decision in ("sample", "hold")
    assert set(d.probs) == set(MODELS) and sum(d.probs.values()) == pytest.approx(1.0)
    assert set(d.scores) == set(MODELS)
    assert set(d.posterior) == set(TOPIC_IDS) and sum(d.posterior.values()) == pytest.approx(1.0)
    assert 0.0 <= d.r <= 1.0 and d.r > 0.6


def test_time_sensitive_ai_window_prefers_the_ai_freshest_model():
    d = make_router().decide(AI_WINDOW, current=None)
    assert max(d.scores, key=d.scores.get) == "claude-fable-5.2"
    assert d.probs["claude-fable-5.2"] > 0.8


def test_time_insensitive_window_mixes_almost_uniformly():
    d = make_router().decide(MATH_WINDOW, current=None)
    assert d.r < 0.2
    assert max(d.probs.values()) - min(d.probs.values()) < 0.25


def test_without_index_scores_are_zero_and_choice_is_uniform():
    router = make_router(index=None)
    assert router.probe is None
    d = router.decide(AI_WINDOW, current=None)
    assert d.scores == {m: 0.0 for m in MODELS}
    assert d.probs == pytest.approx({m: 1 / 3 for m in MODELS})
    assert router.score_draft(AI_WINDOW, "draft", "gpt-6.2-sol") == 0.0


def test_posterior_ema_and_update_flag():
    router = make_router(cfg=RouterConfig(ema=0.5))
    clf = router.classifier
    first = router.decide(AI_WINDOW, current=None).posterior
    assert first == pytest.approx(dict(zip(clf.topics, clf.posterior(AI_WINDOW))))
    sports = "The club won the championship final; the coach praised the league title."
    peek = router.decide(sports, current=None, update=False).posterior
    expected = {t: 0.5 * p + 0.5 * first[t] for t, p in zip(clf.topics, clf.posterior(sports))}
    assert peek == pytest.approx(expected)
    again = router.decide(sports, current=None).posterior  # previous call did not store its posterior
    assert again == pytest.approx(expected)
    router.reset()
    assert router.decide(sports, current=None).posterior == pytest.approx(dict(zip(clf.topics, clf.posterior(sports))))


def test_available_restricts_choice():
    router = make_router()
    for _ in range(20):
        d = router.decide(AI_WINDOW, current="gpt-6.2-sol", available=["gpt-6.2-sol", "gemini-4-neon"])
        assert d.model in ("gpt-6.2-sol", "gemini-4-neon") and set(d.probs) == {"gpt-6.2-sol", "gemini-4-neon"}
    forced = router.decide(AI_WINDOW, current=None, available=["gemini-4-neon"])
    assert (forced.model, forced.decision) == ("gemini-4-neon", "forced")
    with pytest.raises(ValueError):
        router.decide(AI_WINDOW, current=None, available=[])


def test_models_missing_from_index_are_never_chosen():
    router = Router([*MODELS, "gemini-4-hydrogen"], index=make_index(), cfg=RouterConfig(), seed=1)
    for _ in range(30):
        d = router.decide(AI_WINDOW, current=None)
        assert d.model != "gemini-4-hydrogen" and d.probs["gemini-4-hydrogen"] == 0.0


def test_reset_reseeds_for_reproducible_runs():
    router = make_router(seed=3)
    windows = [AI_WINDOW, MATH_WINDOW, "Der Bundeskanzler sprach im Bundestag.", MATH_WINDOW] * 3
    first = [router.decide(w, current=None).model for w in windows]
    router.reset()
    assert [router.decide(w, current=None).model for w in windows] == first


def test_thompson_policy_through_router():
    router = make_router(cfg=RouterConfig(policy="thompson"))
    d = router.decide("Which space telescope launched most recently? The lander and the rocket matter.", current=None)
    assert d.model == "gemini-4-neon" and d.decision == "sample"
    router_no_draws = make_router(cfg=RouterConfig(policy="thompson"), index=make_index(draws=False))
    assert router_no_draws.decide(AI_WINDOW, current=None).model == "claude-fable-5.2"


def test_explicit_components_are_used():
    gate = TimeSensitivityGate()
    clf = KeywordTopicClassifier()
    router = make_router(classifier=clf, gate=gate, policy=FixedPolicy("gemini-4-neon"))
    assert router.classifier is clf and router.gate is gate
    d = router.decide(AI_WINDOW, current=None)
    assert (d.model, d.decision) == ("gemini-4-neon", "forced")
    assert isinstance(make_router().policy, SoftmaxPolicy)


def test_window():
    assert Router.window("Q?", ["a", "b", "c", "d"], 2) == "Q?\n\nc\n\nd"
    assert Router.window("Q?", ["a"], 3) == "Q?\n\na"
    assert Router.window("Q?", ["a", "b"], 0) == "Q?"
    assert Router.window("  Q?  ", [" a ", ""], 5) == "Q?\n\na"


def test_score_draft_and_speculative_choice():
    router = make_router()
    window = "We need the most recent facts for this question."
    ai_draft = "The newest Claude model from Anthropic tops the LLM benchmark."
    space_draft = "The lander touched down on the Moon after the rocket launch from the space station."
    def draft_score(draft, model):
        return router.score_draft(window, draft, model)

    assert draft_score(ai_draft, "claude-fable-5.2") > draft_score(ai_draft, "gpt-6.2-sol")
    assert draft_score(space_draft, "gemini-4-neon") > draft_score(space_draft, "gpt-6.2-sol")
    before = router._posterior
    d = router.decide_speculative(window, {"claude-fable-5.2": ai_draft, "gpt-6.2-sol": "Nothing specific."},
                                  current="gpt-6.2-sol")
    assert d.decision == "speculative" and d.model in ("claude-fable-5.2", "gpt-6.2-sol")
    assert set(d.scores) == {"claude-fable-5.2", "gpt-6.2-sol"}
    assert router._posterior is before
    with pytest.raises(ValueError):
        router.decide_speculative(window, {}, current=None)


def test_router_needs_models():
    with pytest.raises(ValueError):
        Router([], index=None, cfg=RouterConfig())


def test_from_config_uses_the_packaged_prior_index_by_default():
    router = Router.from_config(default_config())
    assert router.index is not None and router.probe == "current"
    assert router.index.models == default_config().model_keys()
    assert router.index.meta["sources"] == ["apoorvumang/knowledge-cutoff (CC BY 4.0)"]
    assert router.index.meta["data_range"]["last"] <= "2026-06-30"  # a prior: the data cannot see later cutoffs


def test_default_config_and_from_config(tmp_path):
    router = Router(MODELS, index=None)  # cfg=None -> RouterConfig()
    assert router.cfg.policy == "softmax" and router.cfg.ema == 0.5

    cfg = default_config()
    index_path = tmp_path / "index.json"
    make_index().save(index_path)
    cfg.router = replace(cfg.router, index_path=str(index_path), classifier="keyword", seed=11, policy="argmax")
    router = Router.from_config(cfg)
    assert router.models == cfg.model_keys()
    assert router.index is not None and router.probe == "current_d" and router.seed == 11
    assert isinstance(router.classifier, KeywordTopicClassifier)
    d = router.decide(AI_WINDOW, current=None)
    assert d.decision == "argmax"
