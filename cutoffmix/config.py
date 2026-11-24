"""Application configuration: models, relay, router and explore settings (DESIGN.md §3.1 and §7).

The packaged ``cutoffmix/data/default.yaml`` holds the three relay models and every default. An optional packaged
preset (``cutoffmix/data/presets/<name>.yaml``, e.g. ``explore``), a user YAML file and an ``overrides`` dict are
deep-merged on top of it in that order, so partial files work::

    cfg = load_config(overrides={"relay": {"segment_tokens": 64},
                                 "models": {"gemini-4-neon": {"max_tokens_slack": 2048}}})
    cfg = load_config(preset="explore")

In ``models`` an override may be a list (it replaces every model) or a mapping ``{model key: {field: value}}`` that
patches individual models. Unknown keys and invalid values raise :class:`ConfigError` naming the offending setting.
"""

from __future__ import annotations

import collections.abc
import copy
import dataclasses
import datetime as _dt
import re
import types as _pytypes
import typing
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml

from cutoffmix.types import ModelSpec

STRATEGIES: tuple[str, ...] = ("sequential", "speculative")
ANSWER_MODES: tuple[str, ...] = ("routed", "inline")
POLICIES: tuple[str, ...] = ("softmax", "thompson", "argmax", "round_robin", "random")  # plus "fixed:<model key>"
CLASSIFIERS: tuple[str, ...] = ("auto", "keyword", "linear", "embedding")
PROBES: tuple[str, ...] = ("current", "current_d", "recall", "mcq")
VS_MENU_MODES: tuple[str, ...] = ("off", "stagnation", "every_lens")
LENS_TIERS: tuple[str, ...] = ("mid", "far", "near")
_PRESET_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")


class ConfigError(ValueError):
    """Raised for unknown keys, wrong types or out-of-range values in a configuration."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ConfigError(message)


@dataclass
class RelayConfig:
    """How the shared notes are produced segment by segment (see DESIGN.md §5)."""

    segment_tokens: int = 128  # K, visible tokens asked per segment
    hard_cut_chars: int = 600  # client-side cap on visible characters per segment
    max_segments: int = 60
    window_segments: int = 3  # routing window = question + last N segments
    strategy: str = "sequential"  # "sequential" | "speculative"
    speculative_gate: float = 0.5  # draft with all models only if r(W) >= gate
    answer_mode: str = "routed"  # "routed" (policy picks the answer model) | "inline" (text after FINAL ANSWER:)
    answer_max_tokens: int = 1024
    sentinel: str = "<<END_STEP>>"
    final_marker: str = "FINAL ANSWER:"
    inject_date: bool = True
    max_retries: int = 2
    disable_after_refusals: int = 2  # a model that refuses this often is excluded for the rest of the run
    # wall-clock budget of one attempt, which keep-alive comments cannot reset (then a retryable error)
    first_content_timeout_s: float = 60.0  # until the first visible text (hidden reasoning comes first)
    segment_timeout_s: float = 120.0  # the whole stream of a think segment, draft or answer

    def __post_init__(self) -> None:
        _require(self.segment_tokens >= 1, f"relay.segment_tokens must be >= 1, got {self.segment_tokens}")
        _require(self.hard_cut_chars >= 1, f"relay.hard_cut_chars must be >= 1, got {self.hard_cut_chars}")
        _require(self.max_segments >= 1, f"relay.max_segments must be >= 1, got {self.max_segments}")
        _require(self.window_segments >= 0, f"relay.window_segments must be >= 0, got {self.window_segments}")
        _require(self.strategy in STRATEGIES, f"relay.strategy must be one of {STRATEGIES}, got {self.strategy!r}")
        _require(0.0 <= self.speculative_gate <= 1.0,
                 f"relay.speculative_gate must be in [0, 1], got {self.speculative_gate}")
        _require(self.answer_mode in ANSWER_MODES,
                 f"relay.answer_mode must be one of {ANSWER_MODES}, got {self.answer_mode!r}")
        _require(self.answer_max_tokens >= 1, f"relay.answer_max_tokens must be >= 1, got {self.answer_max_tokens}")
        _require(bool(self.sentinel.strip()), "relay.sentinel must be a non-whitespace string")
        _require(bool(self.final_marker.strip()), "relay.final_marker must be a non-whitespace string")
        _require(self.max_retries >= 0, f"relay.max_retries must be >= 0, got {self.max_retries}")
        _require(self.disable_after_refusals >= 1,
                 f"relay.disable_after_refusals must be >= 1, got {self.disable_after_refusals}")
        for name in ("first_content_timeout_s", "segment_timeout_s"):
            _require(getattr(self, name) > 0, f"relay.{name} must be > 0, got {getattr(self, name)}")


@dataclass
class RouterConfig:
    """Routing policy and index settings (see DESIGN.md §3.4)."""

    policy: str = "softmax"  # softmax | thompson | argmax | fixed:<key> | round_robin | random
    T0: float = 0.5  # months
    beta: float = 0.25  # stickiness bonus for the current model, months
    kappa: float = 1.0  # uncertainty penalty
    lam: float = 0.5  # stale-assertion hazard penalty, months per unit hazard (a tie-breaker: at most T0)
    r_min: float = 0.05
    delta: float = 0.5  # argmax hysteresis margin, months
    ema: float = 0.5  # topic-posterior EMA weight on the newest window
    index_path: str | None = None  # None -> packaged cutoffmix/data/index_prior.json (if present)
    probe_preference: tuple[str, ...] = ("current_d", "current", "recall", "mcq")
    classifier: str = "auto"  # auto | keyword | linear | embedding
    classifier_path: str | None = None
    gate_path: str | None = None  # trained time-sensitivity gate (router train: gate_linear.npz); None -> lexical
    seed: int | None = None

    def __post_init__(self) -> None:
        self.probe_preference = tuple(self.probe_preference)
        fixed = self.policy.startswith("fixed:")
        _require(self.policy in POLICIES or (fixed and len(self.policy) > len("fixed:")),
                 f"router.policy must be one of {POLICIES} or 'fixed:<model key>', got {self.policy!r}")
        _require(self.T0 > 0, f"router.T0 must be > 0, got {self.T0}")
        _require(0 < self.r_min <= 1, f"router.r_min must be in (0, 1], got {self.r_min}")
        _require(0 < self.ema <= 1, f"router.ema must be in (0, 1], got {self.ema}")
        for name in ("beta", "kappa", "lam", "delta"):
            _require(getattr(self, name) >= 0, f"router.{name} must be >= 0, got {getattr(self, name)}")
        unknown = [p for p in self.probe_preference if p not in PROBES]
        _require(bool(self.probe_preference) and not unknown,
                 f"router.probe_preference must be a non-empty list drawn from {PROBES}, got {self.probe_preference}")
        _require(self.classifier in CLASSIFIERS,
                 f"router.classifier must be one of {CLASSIFIERS}, got {self.classifier!r}")

    @property
    def fixed_model(self) -> str | None:
        """The model key of a ``fixed:<key>`` policy, else None."""
        return self.policy.split(":", 1)[1] if self.policy.startswith("fixed:") else None


@dataclass
class LensConfig:
    """Schedule of the client-side domain lens (DESIGN.md §7.3): when a lens is drawn and from which distance tier."""

    p: float = 0.25  # draw probability per eligible segment
    p_stagnation: float = 0.50  # draw probability while the main line stagnates
    min: int = 1  # force one lens at the last eligible slot if none was drawn
    max: int = 2  # lenses per run
    first_index: int = 2  # 0-based segment index; never in the first two segments
    cooldown: int = 2  # segments after a committed check before the next lens may be drawn
    tiers: Mapping[str, float] = field(default_factory=lambda: {"mid": 0.55, "far": 0.35, "near": 0.10})
    exclude_from_window: bool = True  # lens notes are not part of the routing window
    check_other_model: bool = True  # the check is routed away from the lens writer

    def __post_init__(self) -> None:
        self.tiers = dict(self.tiers)
        for name in ("p", "p_stagnation"):
            _require(0.0 <= getattr(self, name) <= 1.0, f"explore.lens.{name} must be in [0, 1], got {getattr(self, name)}")
        _require(self.min >= 0, f"explore.lens.min must be >= 0, got {self.min}")
        _require(self.max >= self.min, f"explore.lens.max must be >= explore.lens.min, got {self.max} < {self.min}")
        _require(self.first_index >= 0, f"explore.lens.first_index must be >= 0, got {self.first_index}")
        _require(self.cooldown >= 0, f"explore.lens.cooldown must be >= 0, got {self.cooldown}")
        unknown = sorted(set(self.tiers) - set(LENS_TIERS))
        _require(not unknown, f"explore.lens.tiers has unknown tier(s) {', '.join(map(str, unknown))}; "
                              f"allowed: {', '.join(LENS_TIERS)}")
        for tier, weight in self.tiers.items():
            _require(isinstance(weight, (int, float)) and not isinstance(weight, bool) and weight >= 0,
                     f"explore.lens.tiers.{tier} must be a number >= 0, got {weight!r}")
        _require(sum(self.tiers.values()) > 0, "explore.lens.tiers must have a positive total weight")


@dataclass
class ExploreConfig:
    """The opt-in explore preset (DESIGN.md §7): structured notes, record gate, domain lens, forced continuations.

    ``enabled`` is False in the packaged default; ``load_config(preset="explore")`` turns it on with the values of
    ``cutoffmix/data/presets/explore.yaml``. The relay ignores every other field while ``enabled`` is False.
    """

    enabled: bool = False
    seed: int | None = None  # overrides the run seed (router.seed / --seed) for every client-side draw when set
    min_segments: int = 6  # main-line notes (lens and check notes not counted) before a conclusion is accepted
    require_labels: tuple[str, ...] = ("Evidence:", "Alternative set aside:", "Open question:")
    where_we_stand_window: int = 2  # a 'Where we stand:' line must appear in one of the last N notes
    max_forced_continuations: int = 3
    loop_overlap_jaccard: float = 0.5  # word-3-gram Jaccard of the last two notes above this -> stop forcing
    cues: tuple[str, ...] = (
        "Alternatively, ",
        "One thing not yet checked: ",
        "The strongest objection to the current conclusion is ",
        "Perhaps the missing piece is ",
        "Before closing, the weakest assumption above is ",
    )
    lens: LensConfig = field(default_factory=LensConfig)
    vs_menu: str = "off"  # off | stagnation | every_lens (the side call always runs on gpt-6.2-sol)
    answer_basis: bool = True  # the answer ends with 'Basis:' and 'Not resolved:'
    # Models that get the LIGHT schema: the default system prompt and a tail without the labelled-items checklist
    # (directives, note number and the conclusion permission stay). Live finding 2026-11-24: Anthropic's input
    # filter blocks the full explore schema for Fable 5.2 in most calls ('duplicating model outputs' category).
    light_schema_models: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.require_labels = tuple(self.require_labels)
        self.cues = tuple(self.cues)
        self.light_schema_models = tuple(self.light_schema_models)
        for i, key in enumerate(self.light_schema_models):
            _require(isinstance(key, str) and bool(key.strip()),
                     f"explore.light_schema_models[{i}] must be a non-empty model key, got {key!r}")
        _require(self.min_segments >= 1, f"explore.min_segments must be >= 1, got {self.min_segments}")
        _require(self.where_we_stand_window >= 1,
                 f"explore.where_we_stand_window must be >= 1, got {self.where_we_stand_window}")
        _require(self.max_forced_continuations >= 0,
                 f"explore.max_forced_continuations must be >= 0, got {self.max_forced_continuations}")
        _require(0.0 <= self.loop_overlap_jaccard <= 1.0,
                 f"explore.loop_overlap_jaccard must be in [0, 1], got {self.loop_overlap_jaccard}")
        for i, label in enumerate(self.require_labels):
            _require(bool(label.strip()) and label.rstrip().endswith(":"),
                     f"explore.require_labels[{i}] must be a label ending with ':', got {label!r}")
        _require(bool(self.cues) and all(c.strip() for c in self.cues),
                 "explore.cues must be a non-empty list of non-blank opening phrases")
        _require(self.vs_menu in VS_MENU_MODES,
                 f"explore.vs_menu must be one of {VS_MENU_MODES}, got {self.vs_menu!r}")
        _require(isinstance(self.lens, LensConfig), "explore.lens must be a mapping")


@dataclass
class AppConfig:
    """Complete runtime configuration: the relay models plus relay, router, explore and gateway settings."""

    models: list[ModelSpec]
    relay: RelayConfig = field(default_factory=RelayConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    explore: ExploreConfig = field(default_factory=ExploreConfig)
    today: str | None = None  # ISO date; None -> date.today()
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_title: str = "cutoffmix"
    # The packaged preset :func:`load_config` merged into this configuration (``"explore"``), else None. An echo for
    # the relay's metrics and the server, always set by the loader: a ``preset:`` key in a config file loads nothing.
    preset: str | None = None

    def __post_init__(self) -> None:
        _require(bool(self.models), "config needs at least one model")
        _require(isinstance(self.explore, ExploreConfig), "config.explore must be a mapping")
        _require(self.preset is None or bool(_PRESET_NAME.match(self.preset)),
                 f"preset must be a packaged preset name (lowercase letters, digits, _ or -), got {self.preset!r}")
        keys = [m.key for m in self.models]
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        _require(not dupes, f"duplicate model keys: {', '.join(dupes)}")
        for m in self.models:
            _validate_model(m)
        fixed = self.router.fixed_model
        _require(fixed is None or fixed in keys,
                 f"router.policy {self.router.policy!r} names an unknown model; known: {', '.join(keys)}")
        if self.today is not None:
            try:
                _dt.date.fromisoformat(self.today)
            except ValueError as e:
                raise ConfigError(f"today must be an ISO date (YYYY-MM-DD), got {self.today!r}") from e

    def model(self, key: str) -> ModelSpec:
        """Return the model with this key; raises KeyError listing the known keys."""
        for m in self.models:
            if m.key == key:
                return m
        raise KeyError(f"unknown model {key!r}; known: {', '.join(self.model_keys())}")

    def model_keys(self) -> list[str]:
        """Model keys in configuration order."""
        return [m.key for m in self.models]

    def today_str(self) -> str:
        """The configured date, or today's date, as ``YYYY-MM-DD``."""
        return self.today or _dt.date.today().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON/YAML-serialisable form; ``config_from_dict(cfg.to_dict())`` reproduces the config."""
        d = dataclasses.asdict(self)
        d["router"]["probe_preference"] = list(self.router.probe_preference)
        d["explore"]["require_labels"] = list(self.explore.require_labels)
        d["explore"]["cues"] = list(self.explore.cues)
        d["explore"]["light_schema_models"] = list(self.explore.light_schema_models)
        d["explore"]["lens"]["tiers"] = dict(self.explore.lens.tiers)
        return d


def _validate_model(m: ModelSpec) -> None:
    _require(bool(m.key) and bool(m.provider_id), f"model needs a non-empty key and provider_id: {m!r}")
    _require(m.max_tokens_slack >= 0, f"model {m.key}: max_tokens_slack must be >= 0")
    _require(m.cache_style in typing.get_args(typing.get_type_hints(ModelSpec)["cache_style"]),
             f"model {m.key}: unknown cache_style {m.cache_style!r}")
    # OpenRouter maps `verbosity` onto Claude's effort setting, which would silently change the model's behaviour.
    _require(not (m.provider_id.startswith("anthropic/")
                  and ("verbosity" in m.extra_body or "verbosity" in m.answer_extra_body)),
             f"model {m.key}: never send `verbosity` to an Anthropic model (OpenRouter maps it to effort)")


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _packaged_default() -> dict[str, Any]:
    text = resources.files("cutoffmix").joinpath("data").joinpath("default.yaml").read_text(encoding="utf-8")
    return _parse_yaml(text, "packaged default.yaml")


def _presets_dir() -> Any:
    return resources.files("cutoffmix").joinpath("data").joinpath("presets")


def list_presets() -> list[str]:
    """Names of the packaged presets (``cutoffmix/data/presets/<name>.yaml``), sorted."""
    folder = _presets_dir()
    if not folder.is_dir():
        return []
    return sorted(p.name[: -len(".yaml")] for p in folder.iterdir() if p.name.endswith(".yaml"))


def _packaged_preset(name: str) -> dict[str, Any]:
    known = list_presets()
    _require(bool(_PRESET_NAME.match(name)) and name in known,
             f"unknown preset {name!r}; available: {', '.join(known) or 'none'}")
    text = _presets_dir().joinpath(f"{name}.yaml").read_text(encoding="utf-8")
    return _parse_yaml(text, f"packaged preset {name}.yaml")


def _parse_yaml(text: str, origin: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {origin}: {e}") from e
    if data is None:
        return {}
    _require(isinstance(data, dict), f"{origin} must contain a mapping at the top level")
    return data


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _merge_config(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Deep merge, except that a mapping under ``models`` patches the base model list by key."""
    over = dict(over)
    patches = over.pop("models", None)
    out = _deep_merge(base, over)
    if isinstance(patches, Mapping):
        models = copy.deepcopy(list(out.get("models") or []))
        index = {m.get("key"): i for i, m in enumerate(models) if isinstance(m, Mapping)}
        for key, patch in patches.items():
            _require(key in index, f"models override names unknown model {key!r}; known: {', '.join(map(str, index))}"
                                   " (pass a full `models` list to add models)")
            _require(isinstance(patch, Mapping), f"models override for {key!r} must be a mapping")
            models[index[key]] = _deep_merge(models[index[key]], patch)
        out["models"] = models
    elif patches is not None:
        out["models"] = copy.deepcopy(patches)
    return out


def _check_value(where: str, hint: Any, value: Any) -> Any:
    """Check ``value`` against a (simple) type hint and return it normalised; raise ConfigError otherwise."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, _pytypes.UnionType):
        options = typing.get_args(hint)
        if value is None and type(None) in options:
            return None
        errors = []
        for opt in options:
            if opt is type(None):
                continue
            try:
                return _check_value(where, opt, value)
            except ConfigError as e:
                errors.append(str(e))
        raise ConfigError(errors[0] if errors else f"{where}: invalid value {value!r}")
    if origin is typing.Literal:
        _require(value in typing.get_args(hint), f"{where} must be one of {typing.get_args(hint)}, got {value!r}")
        return value
    if origin is tuple:
        _require(isinstance(value, (list, tuple)), f"{where} must be a list, got {value!r}")
        (item_hint, *_) = typing.get_args(hint)
        return tuple(_check_value(f"{where}[{i}]", item_hint, v) for i, v in enumerate(value))
    if origin is collections.abc.Mapping:
        _require(isinstance(value, Mapping), f"{where} must be a mapping, got {value!r}")
        return copy.deepcopy(dict(value))
    if hint is bool:
        _require(isinstance(value, bool), f"{where} must be true or false, got {value!r}")
        return value
    if hint is int:
        _require(isinstance(value, int) and not isinstance(value, bool), f"{where} must be an integer, got {value!r}")
        return value
    if hint is float:
        _require(isinstance(value, (int, float)) and not isinstance(value, bool),
                 f"{where} must be a number, got {value!r}")
        return float(value)
    if hint is str:
        _require(isinstance(value, str), f"{where} must be a string, got {value!r}")
        return value
    return value


def _build(cls: type, data: Any, where: str, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    """Type-check the mapping ``data`` against the dataclass ``cls``; return constructor kwargs."""
    _require(isinstance(data, Mapping), f"{where} must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls) if f.name not in skip]
    unknown = sorted(set(data) - set(names) - set(skip))
    _require(not unknown, f"unknown key(s) in {where}: {', '.join(map(str, unknown))}; "
                          f"allowed: {', '.join(f.name for f in dataclasses.fields(cls))}")
    return {k: _check_value(f"{where}.{k}", hints[k], v) for k, v in data.items() if k in names}


def config_from_dict(data: Mapping[str, Any]) -> AppConfig:
    """Build a validated :class:`AppConfig` from a complete plain mapping (the YAML structure)."""
    if isinstance(data.get("today"), _dt.date):  # YAML reads an unquoted ISO date as a date
        data = {**data, "today": data["today"].isoformat()}
    top = _build(AppConfig, data, "config", skip=("models", "relay", "router", "explore"))
    raw_models = data.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise ConfigError("config.models must be a non-empty list")
    models = [ModelSpec(**_build(ModelSpec, m, f"models[{i}]")) for i, m in enumerate(raw_models)]
    relay = RelayConfig(**_build(RelayConfig, data.get("relay") or {}, "relay"))
    router = RouterConfig(**_build(RouterConfig, data.get("router") or {}, "router"))
    explore = _explore_from_dict(data.get("explore") or {})
    return AppConfig(models=models, relay=relay, router=router, explore=explore, **top)


def _explore_from_dict(data: Any) -> ExploreConfig:
    if isinstance(data, Mapping) and data.get("vs_menu") is False:  # YAML 1.1 reads a bare `off` as false
        data = {**data, "vs_menu": "off"}
    kwargs = _build(ExploreConfig, data, "explore", skip=("lens",))
    lens = data.get("lens") if isinstance(data, Mapping) else None
    kwargs["lens"] = LensConfig(**_build(LensConfig, lens if lens is not None else {}, "explore.lens"))
    return ExploreConfig(**kwargs)


def load_config(path: str | Path | None = None, overrides: Mapping[str, Any] | None = None, *,
                preset: str | None = None) -> AppConfig:
    """Load the packaged defaults, deep-merge the packaged ``preset`` (if given), then the YAML file at ``path`` (if
    given) and then ``overrides``.

    ``preset`` names a file ``cutoffmix/data/presets/<preset>.yaml`` (see :func:`list_presets`); an unknown name
    raises :class:`ConfigError`. ``overrides`` is a nested dict such as ``{"relay": {"strategy": "speculative"}}``.
    The result's ``preset`` field echoes the ``preset`` argument (None when none was merged), whatever the file or
    the overrides say.
    """
    data = _packaged_default()
    if preset is not None:
        data = _merge_config(data, _packaged_preset(preset))
    if path is not None:
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ConfigError(f"cannot read config file {p}: {e}") from e
        data = _merge_config(data, _parse_yaml(text, str(p)))
    if overrides:
        data = _merge_config(data, overrides)
    data["preset"] = preset
    return config_from_dict(data)


def default_config() -> AppConfig:
    """The packaged default configuration."""
    return load_config()
