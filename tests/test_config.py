"""Tests for cutoffmix.config, the packaged default.yaml and the packaged presets (explore)."""

from __future__ import annotations

import datetime as dt
from importlib import resources

import pytest
import yaml

from cutoffmix.config import (
    AppConfig,
    ConfigError,
    ExploreConfig,
    LensConfig,
    RelayConfig,
    RouterConfig,
    config_from_dict,
    default_config,
    list_presets,
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
    assert fable.provider_routing == {"only": ["anthropic", "azure"]}
    assert sol.provider_routing == {"only": ["openai", "azure"]}
    assert neon.provider_routing == {"only": ["google-vertex"]}
    assert [m.supports_stop for m in (fable, sol, neon)] == [True, False, True]
    # no target model honours a sampling parameter: Google ignores Neon's in-range temperature (DESIGN.md §7)
    assert [m.supports_temperature for m in (fable, sol, neon)] == [False, False, False]
    assert [m.cancellable for m in (fable, sol, neon)] == [True, True, False]
    assert [m.cache_style for m in (fable, sol, neon)] == ["anthropic", "openai", "implicit"]
    assert neon.max_tokens_slack == 1536
    assert all(m.reasoning == {"effort": "low", "exclude": True} for m in (fable, sol, neon))
    assert sol.extra_body == {"verbosity": "low"}
    assert "verbosity" not in fable.extra_body and "verbosity" not in neon.extra_body
    # the explore-only model settings are off in the default
    assert all(m.answer_extra_body == {} and not m.supports_seed for m in (fable, sol, neon))


def test_yaml_defaults_agree_with_dataclass_defaults():
    cfg = default_config()
    assert cfg.relay == RelayConfig()
    assert cfg.router == RouterConfig()
    assert cfg.explore == ExploreConfig() and not cfg.explore.enabled
    assert cfg.explore.lens == LensConfig()
    assert (cfg.today, cfg.base_url, cfg.api_key_env, cfg.app_title) == (
        None, "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "cutoffmix")


# --------------------------------------------------------------------------------------
# explore preset (DESIGN.md §7)
# --------------------------------------------------------------------------------------


def _preset_yaml(name: str) -> dict:
    text = resources.files("cutoffmix").joinpath("data").joinpath("presets").joinpath(f"{name}.yaml").read_text()
    return yaml.safe_load(text)


def test_list_presets_and_unknown_preset():
    assert list_presets() == ["explore", "rally"]
    with pytest.raises(ConfigError, match=r"unknown preset 'nope'; available: explore, rally"):
        load_config(preset="nope")
    with pytest.raises(ConfigError, match="unknown preset"):
        load_config(preset="../default")  # never a path


def test_explore_preset_matches_design_section_2():
    cfg = load_config(preset="explore")
    ex = cfg.explore
    assert ex.enabled is True and ex.seed is None
    assert ex.min_segments == 6
    assert ex.require_labels == ("Evidence:", "Alternative set aside:", "Open question:")
    assert ex.where_we_stand_window == 2 and ex.max_forced_continuations == 3 and ex.loop_overlap_jaccard == 0.5
    assert ex.cues == ("Alternatively, ", "One thing not yet checked: ",
                       "The strongest objection to the current conclusion is ", "Perhaps the missing piece is ",
                       "Before closing, the weakest assumption above is ")
    assert ex.lens == LensConfig(p=0.25, p_stagnation=0.5, min=1, max=2, first_index=2, cooldown=2,
                                 tiers={"mid": 0.55, "far": 0.35, "near": 0.10}, exclude_from_window=True,
                                 check_other_model=True)
    assert ex.vs_menu == "off" and ex.answer_basis is True
    assert (cfg.relay.segment_tokens, cfg.relay.hard_cut_chars, cfg.relay.max_segments, cfg.relay.answer_mode) == (
        192, 1500, 16, "routed")
    assert cfg.relay.window_segments == 3
    # routing by cutoff stays the product: the preset does not touch the policy
    assert cfg.router == RouterConfig()
    assert "policy" not in (_preset_yaml("explore").get("router") or {})
    sol, neon, fable = cfg.model("gpt-6.2-sol"), cfg.model("gemini-4-neon"), cfg.model("claude-fable-5.2")
    assert sol.extra_body == {"verbosity": "medium"} and sol.answer_extra_body == {"verbosity": "low"}
    assert sol.supports_seed is True
    assert neon.supports_temperature is False and not neon.supports_seed
    assert fable.extra_body == {} and fable.answer_extra_body == {} and not fable.supports_seed
    # everything else is the default
    assert (cfg.relay.sentinel, cfg.relay.max_retries, cfg.today, cfg.base_url) == (
        "<<END_STEP>>", 2, None, "https://openrouter.ai/api/v1")
    assert config_from_dict(cfg.to_dict()) == cfg


def test_preset_merges_before_user_file_and_overrides(tmp_path):
    path = tmp_path / "my.yaml"
    path.write_text("explore:\n  min_segments: 5\n  lens:\n    max: 1\nrelay:\n  max_segments: 12\n", encoding="utf-8")
    cfg = load_config(path, preset="explore")
    assert cfg.explore.enabled and cfg.explore.min_segments == 5 and cfg.explore.lens.max == 1
    assert cfg.explore.lens.p == 0.25  # the rest of the preset's lens block survives the partial file
    assert cfg.relay.max_segments == 12 and cfg.relay.segment_tokens == 192
    cfg2 = load_config(path, overrides={"explore": {"enabled": False, "lens": {"max": 2}}}, preset="explore")
    assert not cfg2.explore.enabled and cfg2.explore.lens.max == 2 and cfg2.explore.min_segments == 5
    # a user file may also turn the preset's model patches back
    path.write_text("models:\n  gpt-6.2-sol:\n    extra_body: {verbosity: low}\n", encoding="utf-8")
    assert load_config(path, preset="explore").model("gpt-6.2-sol").extra_body == {"verbosity": "low"}


def test_preset_echo_is_set_by_the_loader_only(tmp_path):
    assert load_config().preset is None and default_config().preset is None
    cfg = load_config(preset="explore")
    assert cfg.preset == "explore" and cfg.to_dict()["preset"] == "explore"
    assert config_from_dict(cfg.to_dict()) == cfg and config_from_dict(cfg.to_dict()).preset == "explore"
    # a `preset:` key in a config file (or in overrides) is an echo, never a loader: it is overwritten
    path = tmp_path / "labelled.yaml"
    path.write_text("preset: explore\n", encoding="utf-8")
    plain = load_config(path)
    assert plain.preset is None and plain.explore.enabled is False and plain.relay.segment_tokens == 128
    assert load_config(path, overrides={"preset": "explore"}).preset is None
    assert load_config(path, preset="explore").preset == "explore"
    with pytest.raises(ConfigError, match="preset must be a packaged preset name"):
        AppConfig(models=[ModelSpec(key="a", provider_id="x/a")], preset="Not A Name")


def test_explore_without_preset_is_off_and_fully_overridable():
    cfg = load_config(overrides={"explore": {"enabled": True, "seed": 7, "vs_menu": "stagnation",
                                             "lens": {"tiers": {"far": 1.0}}}})
    assert cfg.explore.enabled and cfg.explore.seed == 7 and cfg.explore.vs_menu == "stagnation"
    # a partial `tiers` mapping patches the default weights (deep merge, like every nested mapping)
    assert cfg.explore.lens.tiers == {"mid": 0.55, "far": 1.0, "near": 0.10} and cfg.explore.lens.p == 0.25
    only_far = load_config(overrides={"explore": {"lens": {"tiers": {"mid": 0, "far": 1.0, "near": 0}}}})
    assert only_far.explore.lens.tiers == {"mid": 0, "far": 1.0, "near": 0}
    assert cfg.relay.segment_tokens == 128  # the preset's relay changes are not implied by explore.enabled
    assert load_config(overrides={"explore": {"vs_menu": False}}).explore.vs_menu == "off"  # YAML's bare `off`


def test_explore_to_dict_round_trip_and_plain_types():
    cfg = load_config(overrides={"explore": {"cues": ["Alternatively, "], "require_labels": ["Next:"]}})
    d = cfg.to_dict()
    assert d["explore"]["cues"] == ["Alternatively, "] and d["explore"]["require_labels"] == ["Next:"]
    assert isinstance(d["explore"]["lens"]["tiers"], dict)
    assert config_from_dict(d) == cfg


@pytest.mark.parametrize("overrides, match", [
    ({"explore": {"enabled": "yes"}}, "true or false"),
    ({"explore": {"min_segments": 0}}, "explore.min_segments"),
    ({"explore": {"where_we_stand_window": 0}}, "where_we_stand_window"),
    ({"explore": {"max_forced_continuations": -1}}, "max_forced_continuations"),
    ({"explore": {"loop_overlap_jaccard": 1.5}}, "loop_overlap_jaccard"),
    ({"explore": {"require_labels": ["Evidence"]}}, "ending with ':'"),
    ({"explore": {"require_labels": "Evidence:"}}, "must be a list"),
    ({"explore": {"cues": []}}, "explore.cues"),
    ({"explore": {"vs_menu": "always"}}, "explore.vs_menu"),
    ({"explore": {"seed": 1.5}}, "must be an integer"),
    ({"explore": {"bogus": 1}}, "unknown key"),
    ({"explore": {"lens": "x"}}, "explore.lens must be a mapping"),
    ({"explore": {"lens": {"bogus": 1}}}, "unknown key"),
    ({"explore": {"lens": {"p": 1.5}}}, "explore.lens.p"),
    ({"explore": {"lens": {"p_stagnation": -0.1}}}, "explore.lens.p_stagnation"),
    ({"explore": {"lens": {"min": 2, "max": 1}}}, "explore.lens.max"),
    ({"explore": {"lens": {"first_index": -1}}}, "first_index"),
    ({"explore": {"lens": {"cooldown": -1}}}, "cooldown"),
    ({"explore": {"lens": {"tiers": {"close": 1}}}}, "unknown tier"),
    ({"explore": {"lens": {"tiers": {"mid": -1}}}}, "explore.lens.tiers.mid"),
    ({"explore": {"lens": {"tiers": {"mid": 0, "far": 0, "near": 0}}}}, "positive total"),
    ({"explore": {"lens": {"tiers": [1, 2]}}}, "must be a mapping"),
    ({"models": {"claude-fable-5.2": {"answer_extra_body": {"verbosity": "low"}}}}, "verbosity"),
    ({"models": {"gpt-6.2-sol": {"supports_seed": "yes"}}}, "true or false"),
])
def test_invalid_explore_settings_raise_config_error(overrides, match):
    with pytest.raises(ConfigError, match=match):
        load_config(overrides=overrides)


def test_explore_dataclass_direct_construction_validates():
    with pytest.raises(ConfigError, match="explore.lens.max"):
        LensConfig(min=3, max=2)
    with pytest.raises(ConfigError, match="positive total"):
        LensConfig(tiers={"mid": 0})
    with pytest.raises(ConfigError, match="unknown tier"):
        LensConfig(tiers={"mid": 1, "close": 1})
    with pytest.raises(ConfigError, match="explore.vs_menu"):
        ExploreConfig(vs_menu="sometimes")
    ex = ExploreConfig(cues=["a "], require_labels=["Claim:"])  # lists are normalised to tuples
    assert ex.cues == ("a ",) and ex.require_labels == ("Claim:",)
    cfg = AppConfig(models=[ModelSpec(key="a", provider_id="x/a")])
    assert cfg.explore == ExploreConfig()


def test_today_str_and_model_lookup():
    cfg = default_config()
    assert cfg.today_str() == dt.date.today().isoformat()
    assert load_config(overrides={"today": "2026-11-24"}).today_str() == "2026-11-24"
    with pytest.raises(KeyError, match=r"known: claude-fable-5\.2"):
        cfg.model("nope")


def test_default_routing_keeps_failover_within_the_only_list():
    # OpenRouter: allow_fallbacks false without `order` serves only the top-ranked provider, so a second provider
    # in `only` would never take over during an outage
    for model in default_config().models:
        routing = model.provider_routing
        assert routing.get("only"), model.key
        if len(routing["only"]) > 1 and routing.get("allow_fallbacks") is False:
            assert routing.get("order"), f"{model.key}: fallbacks disabled without order"


def test_unquoted_iso_date_for_today_in_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text("today: 2026-11-24\n", encoding="utf-8")  # YAML parses this as a date, not a string
    cfg = load_config(path)
    assert cfg.today == "2026-11-24" and cfg.today_str() == "2026-11-24"
    assert load_config(overrides={"today": dt.date(2026, 11, 2)}).today == "2026-11-02"


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
