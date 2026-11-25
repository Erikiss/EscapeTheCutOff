"""The light schema of the explore preset (DESIGN.md §7.8): models listed in ``explore.light_schema_models`` get the
default system prompt and a tail without the labelled-items checklist, because Anthropic's input filter blocks the
full schema for Fable 5.2 in most calls (live finding, 2026-11-24)."""

from __future__ import annotations

import pytest

from cutoffmix.config import ConfigError, config_from_dict, load_config
from cutoffmix.prompts import (
    RECORD_CLOSED_NO_ITEMS_TEXT,
    RECORD_OPEN_TEXT,
    relay_system_prompt,
    relay_system_prompt_explore,
)
from cutoffmix.relay import Relay, _Plan, _RunState


def test_preset_lists_fable_and_default_lists_nobody() -> None:
    assert load_config().explore.light_schema_models == ()
    assert load_config(preset="explore").explore.light_schema_models == ("claude-fable-5.2",)


def test_round_trip_and_validation() -> None:
    cfg = load_config(preset="explore", overrides={"explore": {"light_schema_models": ["gpt-6.2-sol"]}})
    assert cfg.explore.light_schema_models == ("gpt-6.2-sol",)
    assert config_from_dict(cfg.to_dict()).explore.light_schema_models == ("gpt-6.2-sol",)
    with pytest.raises(ConfigError, match="light_schema_models"):
        load_config(preset="explore", overrides={"explore": {"light_schema_models": [""]}})


@pytest.fixture
def relay_and_state() -> tuple[Relay, _RunState]:
    from tests.test_explore_relay import make_explore_relay

    cfg = load_config(preset="explore")
    relay, _backend, _router = make_explore_relay(lambda model, req, n: [], cfg)
    st = relay._new_state("Warum verlieren Lithium-Ionen-Akkus bei Kälte an Kapazität?", run_id="t")
    return relay, st


def _requests(relay: Relay, st: _RunState) -> tuple:
    plan = _Plan(kind="main")
    fable = relay._request(st, "think", None, plan, relay.cfg.model("claude-fable-5.2"))
    sol = relay._request(st, "think", None, plan, relay.cfg.model("gpt-6.2-sol"))
    neon = relay._request(st, "think", None, plan, relay.cfg.model("gemini-4-neon"))
    return fable, sol, neon


def test_light_model_gets_the_default_system_prompt_and_a_bare_tail(relay_and_state) -> None:
    relay, st = relay_and_state
    rc = relay.cfg.relay
    fable, sol, neon = _requests(relay, st)
    assert fable.system_override == relay_system_prompt(rc.segment_tokens, st.today, rc.sentinel, rc.final_marker)
    assert sol.system_override == relay_system_prompt_explore(rc.segment_tokens, st.today, rc.sentinel, rc.final_marker)
    assert neon.system_override == sol.system_override
    assert fable.tail is not None and sol.tail is not None
    # the light tail: note number, no checklist, the plain "do not conclude yet" sentence (live-verified to pass)
    assert "Write note 1." in fable.tail
    assert RECORD_CLOSED_NO_ITEMS_TEXT.format(final_marker=rc.final_marker) in fable.tail
    for forbidden in ("Items still missing", "'Evidence:'", "'Alternative set aside:'", "'Open question:'", "Where we stand"):
        assert forbidden not in fable.tail
    assert "Items still missing from the record" in sol.tail
    assert fable.metadata["light"] is True and sol.metadata["light"] is False


def test_light_tail_keeps_the_directive_and_the_conclusion_permission(relay_and_state) -> None:
    relay, st = relay_and_state
    rc = relay.cfg.relay
    plan = _Plan(kind="continue", directive="Begin this note with the words 'Alternatively, '.", cue="Alternatively, ")
    req = relay._request(st, "think", None, plan, relay.cfg.model("claude-fable-5.2"))
    assert "Begin this note with the words 'Alternatively, '." in (req.tail or "")
    # gate open: the same permission sentence as the full schema, still without the checklist
    ex = st.explore
    assert ex is not None
    ex.record_missing = lambda **kw: []  # type: ignore[method-assign]
    req2 = relay._request(st, "think", None, _Plan(kind="main"), relay.cfg.model("claude-fable-5.2"))
    assert RECORD_OPEN_TEXT.format(final_marker=rc.final_marker) in (req2.tail or "")
    assert "Items still missing" not in (req2.tail or "")


def test_answer_phase_is_unchanged_for_light_models(relay_and_state) -> None:
    relay, st = relay_and_state
    a = relay._request(st, "answer", 256, None, relay.cfg.model("claude-fable-5.2"))
    b = relay._request(st, "answer", 256, None, relay.cfg.model("gpt-6.2-sol"))
    assert a.system_override == b.system_override and a.tail is None and b.tail is None
