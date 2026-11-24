"""Tests for cutoffmix.config and the packaged default.yaml."""

from __future__ import annotations

import datetime as dt

import pytest

from cutoffmix.config import (
    AppConfig,
    ConfigError,
    RelayConfig,
    RouterConfig,
    config_from_dict,
    default_config,
    load_config,
)
from cutoffmix.types import ModelSpec

KEYS = ["claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"]


def test_default_models_match_design():
    cfg = default_config()
    assert cfg.model_keys() == KEYS
    fable, sol, neon = (cfg.model(k) for k in KEYS)
    assert (fable.provider_id, sol.provider_id, neon.provider_id) == (
        "anthropic/claude-fable-5.2", "openai/gpt-6.2-sol", "google/gemini-4-neon")
    assert (fable.color, sol.color, neon.color) == ("#d97757", "#10a37f", "#4285f4")
    assert fable.provider_routing == {"only": ["anthropic", "azure"], "allow_fallbacks": False}
    assert sol.provider_routing == {"only": ["openai", "azure"], "allow_fallbacks": False}
    assert neon.provider_routing == {"only": ["google-vertex"], "allow_fallbacks": False}
    assert [m.supports_stop for m in (fable, sol, neon)] == [True, False, True]
    assert [m.supports_temperature for m in (fable, sol, neon)] == [False, False, True]
    assert [m.cancellable for m in (fable, sol, neon)] == [True, True, False]
    assert [m.cache_style for m in (fable, sol, neon)] == ["anthropic", "openai", "implicit"]
    assert neon.max_tokens_slack == 1536
    assert all(m.reasoning == {"effort": "low", "exclude": True} for m in (fable, sol, neon))
    assert sol.extra_body == {"verbosity": "low"}
    assert "verbosity" not in fable.extra_body and "verbosity" not in neon.extra_body


def test_yaml_defaults_agree_with_dataclass_defaults():
    cfg = default_config()
    assert cfg.relay == RelayConfig()
    assert cfg.router == RouterConfig()
    assert (cfg.today, cfg.base_url, cfg.api_key_env, cfg.app_title) == (
        None, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "cutoffmix")


def test_today_str_and_model_lookup():
    cfg = default_config()
    assert cfg.today_str() == dt.date.today().isoformat()
    assert load_config(overrides={"today": "2026-11-24"}).today_str() == "2026-11-24"
    with pytest.raises(KeyError, match=r"known: claude-fable-5\.2"):
        cfg.model("nope")


def test_overrides_deep_merge_and_model_patch():
    cfg = load_config(overrides={
        "relay": {"segment_tokens": 64, "strategy": "speculative"},
        "router": {"policy": "fixed:gpt-6.2-sol", "probe_preference": ["mcq", "current"]},
        "models": {"gemini-4-neon": {"max_tokens_slack": 2048, "provider_routing": {"allow_fallbacks": True}}},
    })
    assert cfg.relay.segment_tokens == 64 and cfg.relay.strategy == "speculative"
    assert cfg.relay.hard_cut_chars == 600  # untouched defaults survive
    assert cfg.router.fixed_model == "gpt-6.2-sol"
    assert cfg.router.probe_preference == ("mcq", "current")
    neon = cfg.model("gemini-4-neon")
    assert neon.max_tokens_slack == 2048
    assert neon.provider_routing == {"only": ["google-vertex"], "allow_fallbacks": True}
    assert cfg.model("claude-fable-5.2").max_tokens_slack == 2048


def test_models_list_override_replaces_all():
    cfg = load_config(overrides={"models": [{"key": "solo", "provider_id": "x/solo"}]})
    assert cfg.model_keys() == ["solo"]
    assert cfg.model("solo").reasoning == {"effort": "low", "exclude": True}  # ModelSpec default


def test_load_from_partial_yaml_file(tmp_path):
    path = tmp_path / "my.yaml"
    path.write_text("today: '2026-11-01'\nrelay:\n  max_segments: 10\nrouter:\n  T0: 1\n", encoding="utf-8")
    cfg = load_config(path, overrides={"relay": {"max_segments": 12}})
    assert cfg.today == "2026-11-01"
    assert cfg.relay.max_segments == 12  # overrides win over the file
    assert cfg.router.T0 == 1.0 and isinstance(cfg.router.T0, float)
    assert cfg.model_keys() == KEYS


def test_to_dict_round_trip():
    cfg = load_config(overrides={"relay": {"answer_mode": "inline"}, "router": {"seed": 7}})
    d = cfg.to_dict()
    assert d["router"]["probe_preference"] == ["current_d", "current", "recall", "mcq"]
    assert config_from_dict(d) == cfg


@pytest.mark.parametrize("overrides, match", [
    ({"relay": {"strategy": "parallel"}}, "relay.strategy"),
    ({"relay": {"answer_mode": "x"}}, "relay.answer_mode"),
    ({"relay": {"segment_tokens": "128"}}, "must be an integer"),
    ({"relay": {"segment_tokens": 0}}, "segment_tokens"),
    ({"relay": {"inject_date": "yes"}}, "true or false"),
    ({"relay": {"speculative_gate": 2}}, "speculative_gate"),
    ({"relay": {"sentinel": "  "}}, "sentinel"),
    ({"relay": {"bogus": 1}}, "unknown key"),
    ({"bogus": 1}, "unknown key"),
    ({"router": {"policy": "greedy"}}, "router.policy"),
    ({"router": {"policy": "fixed:nope"}}, "unknown model"),
    ({"router": {"classifier": "bert"}}, "router.classifier"),
    ({"router": {"probe_preference": ["direct"]}}, "probe_preference"),
    ({"router": {"T0": 0}}, "T0"),
    ({"router": {"kappa": True}}, "must be a number"),
    ({"today": "24.11.2026"}, "ISO date"),
    ({"models": {"nope": {"color": "#000"}}}, "unknown model"),
    ({"models": {"gpt-6.2-sol": {"cache_style": "redis"}}}, "cache_style"),
    ({"models": {"gpt-6.2-sol": {"colour": "#000"}}}, "unknown key"),
    ({"models": {"claude-fable-5.2": {"extra_body": {"verbosity": "low"}}}}, "verbosity"),
    ({"models": []}, "non-empty"),
])
def test_invalid_settings_raise_config_error(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(overrides=overrides)


def test_file_errors(tmp_path):
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("relay: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(bad)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("42\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(scalar)


def test_direct_construction_validates():
    m = ModelSpec(key="a", provider_id="x/a")
    with pytest.raises(ConfigError, match="duplicate"):
        AppConfig(models=[m, m])
    with pytest.raises(ConfigError, match="at least one model"):
        AppConfig(models=[])
    cfg = AppConfig(models=[m])
    assert cfg.relay == RelayConfig() and cfg.router == RouterConfig()
