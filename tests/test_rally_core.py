"""The rally preset's configuration, note parsing, state machine and prompts (DESIGN_rally.md sections 2 to 6)."""

from __future__ import annotations

import datetime as dt

import pytest

from cutoffmix.config import AppConfig, ConfigError, ExploreConfig, RallyConfig, config_from_dict, load_config
from cutoffmix.prompts import (
    answer_system_prompt_rally,
    month_display,
    rally_directive,
    relay_system_prompt_rally,
    relay_tail_rally,
)
from cutoffmix.rally import RallyState, month_number, parse_month, parse_note
from cutoffmix.types import ModelSpec

TODAY = "2026-11-26"
MODELS = [ModelSpec(key=k, provider_id=f"x/{k}") for k in ("a", "b", "c")]


# ---- config -----------------------------------------------------------------------------------------------------


def test_the_packaged_default_has_no_rally_and_the_preset_turns_it_on():
    assert load_config().rally.enabled is False
    cfg = load_config(preset="rally")
    assert cfg.rally.enabled and cfg.preset == "rally"
    assert cfg.rally_ring() == ["claude-fable-5.2", "gemini-4-neon"]
    assert (cfg.relay.segment_tokens, cfg.relay.max_segments, cfg.relay.answer_mode) == (160, 12, "routed")
    assert cfg.explore.enabled is False


def test_ring_override_and_round_trip():
    cfg = load_config(preset="rally", overrides={"rally": {"ring": ["gemini-4-neon", "gpt-6.2-sol", "claude-fable-5.2"]}})
    assert cfg.rally_ring() == ["gemini-4-neon", "gpt-6.2-sol", "claude-fable-5.2"]
    again = config_from_dict(cfg.to_dict())
    assert again.rally == cfg.rally and again.to_dict()["rally"]["ring"] == list(cfg.rally.ring)
    assert load_config().rally_ring() == ["claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"]  # default: config order


@pytest.mark.parametrize("rally, message", [
    ({"max_turns": 1}, "max_turns must be >= 2"),
    ({"max_turns": 4, "min_turns": 5}, "min_turns must be between"),
    ({"quiet_turns": 0}, "quiet_turns must be >= 1"),
    ({"ring": ["a", "a"]}, "must not repeat"),
    ({"ring": ["a", ""]}, "model key"),
])
def test_rally_config_fields_are_validated(rally, message):
    with pytest.raises(ConfigError, match=message):
        RallyConfig(**rally)


@pytest.mark.parametrize("rally, message", [
    ({"enabled": True, "ring": ("a",)}, "at least two models"),
    ({"ring": ("a", "nope")}, "unknown model"),
])
def test_rally_ring_is_checked_against_the_models(rally, message):
    with pytest.raises(ConfigError, match=message):
        AppConfig(models=list(MODELS), rally=RallyConfig(**rally))
    ok = AppConfig(models=list(MODELS), rally=RallyConfig(enabled=True, ring=("c", "a")))
    assert ok.rally_ring() == ["c", "a"]


def test_rally_and_explore_exclude_each_other():
    with pytest.raises(ConfigError, match="mutually exclusive"):
        AppConfig(models=list(MODELS), rally=RallyConfig(enabled=True), explore=ExploreConfig(enabled=True))
    with pytest.raises(ConfigError, match="mutually exclusive"):
        load_config(preset="rally", overrides={"explore": {"enabled": True}})
    with pytest.raises(ConfigError, match="unknown key"):
        config_from_dict({**load_config().to_dict(), "rally": {"nope": 1}})


# ---- parsing ----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("March 2026", ("2026-03", False, False)),
    ("Mar 2026 (X); Dec 2025", ("2026-03", False, False)),
    ("17 March 2026", ("2026-03", False, False)),
    ("March 17, 2026", ("2026-03", False, False)),
    ("2026-03-17", ("2026-03", False, False)),
    ("03/2026", ("2026-03", False, False)),
    ("März 2026", ("2026-03", False, False)),
    ("Oktober 2025", ("2025-10", False, False)),
    ("Q2 2026", ("2026-04", True, False)),
    ("2025", ("2025-01", True, False)),
    ("Nov 2026", ("2026-11", False, False)),
    ("December 2026 and 2026-02", ("2026-02", False, True)),   # a future date never wins, but is flagged
    ("Jan 2027", (None, False, True)),
    ("release 2024.5 of v2.0", (None, False, False)),            # version numbers are not dates
    ("no date at all", (None, False, False)),
])
def test_parse_month(text, expected):
    assert parse_month(text, TODAY) == expected
    assert parse_month(text, dt.date.fromisoformat(TODAY)) == expected


def test_month_numbers_and_display():
    assert month_number("2026-03") == 2026 * 12 + 2 and month_number("2026-13") is None and month_number(None) is None
    assert month_display("2026-03") == "March 2026" and month_display(None) == "not stated" and month_display("x") == "not stated"


def test_parse_note_counts_labels_and_reads_the_newest_month():
    note = ("Known: A happened (Feb 2026).\n- **Update:** X was replaced in March 2026.\n"
            "Newer: Y shipped in 2026-04. Nothing newer: after that.\n**Newest I know:** April 2026 (Y).")
    got = parse_note(note, TODAY)
    assert got["labels"] == {"known": 1, "confirm": 0, "update": 1, "newer": 1, "unverified": 0, "nothing_newer": 1}
    assert got["newest_raw"] == "April 2026 (Y)." and got["newest"] == "2026-04" and not got["approx"]
    assert got["mentioned"] == "2026-04"


def test_parse_note_is_tolerant_but_not_greedy():
    got = parse_note("confirm: ok. UPDATE: changed in May 2026\n2) Unverified: maybe\nNewest I know: not sure", TODAY)
    assert got["labels"]["confirm"] == 1 and got["labels"]["update"] == 1 and got["labels"]["unverified"] == 1
    assert got["newest_raw"] == "not sure" and got["newest"] is None
    plain = parse_note("The newer model is an update of the older one; nothing newer is known to me.", TODAY)
    assert sum(plain["labels"].values()) == 0 and plain["newest_raw"] is None
    future = parse_note("Newest I know: June 2027", TODAY)
    assert future["newest"] is None and future["future"] is True


# ---- state ------------------------------------------------------------------------------------------------------


def _state(ring=("a", "b"), **cfg) -> RallyState:
    return RallyState(RallyConfig(enabled=True, ring=tuple(ring), **cfg), list(ring), TODAY)


def _note(state, model, update=0, newer=0, quiet=True, newest="April 2026"):
    parts = ["Update: x." for _ in range(update)] + ["Newer: y." for _ in range(newer)]
    parts.append("Nothing newer:" if quiet else "Confirm: z.")
    state.add_note(len(state.notes), model, "\n".join(parts) + f"\nNewest I know: {newest}")


def test_hand_over_follows_the_ring_and_skips_unavailable_models():
    s = _state(("a", "b", "c"))
    assert s.next_model(["a", "b", "c"]) == "a" and s.next_model(["b", "c"]) == "b" and s.next_model([]) is None
    s.add_note(0, "a", "Known: x (Jan 2026).\nNewest I know: January 2026")
    assert s.next_model(["a", "b", "c"]) == "b" and s.next_model(["a", "c"]) == "c"
    s.add_note(1, "b", "Confirm: x.")
    assert s.next_model(["a", "b", "c"]) == "c" and s.next_model(["a", "b"]) == "a"   # wraps around
    assert s.next_model(["b"]) == "b"                                                  # last resort: the same model again
    assert s.kind() == "reply" and s.turns == 2 and s.previous()["model"] == "b"


def test_kinds_and_quiet_notes():
    s = _state()
    s.add_note(0, "a", "Known: x (Jan 2026).\nNewest I know: January 2026")
    assert s.notes[0]["kind"] == "open" and s.notes[0]["quiet"] is False
    _note(s, "b", newer=1, quiet=False)
    assert s.notes[1]["kind"] == "reply" and s.notes[1]["quiet"] is False                 # it added something
    s.add_note(2, "a", "Just prose without any label.")
    assert s.notes[2]["quiet"] is False                                                    # silence is not 'nothing newer'
    _note(s, "b")
    assert s.notes[3]["quiet"] is True and s.quiet_streak() == 1


def test_stop_rule_converged_max_turns_and_no_partner():
    s = _state()
    assert (s.min_turns, s.quiet_turns) == (4, 2)
    s.add_note(0, "a", "Known: x (Jan 2026).\nNewest I know: January 2026")
    for model in ("b", "a"):
        _note(s, model)
    assert s.should_stop() is None                          # three notes: below min_turns although two are quiet
    _note(s, "b")
    assert s.should_stop() == "converged" and s.quiet_streak() == 3

    loud = _state(max_turns=5)
    loud.add_note(0, "a", "Known: x.")
    for model in ("b", "a", "b"):
        _note(loud, model, update=1, quiet=False)
    assert loud.should_stop() is None
    _note(loud, "a", update=1, quiet=False)
    assert loud.should_stop() == "max_turns"

    lone = _state()
    lone.add_note(0, "a", "Known: x.")
    assert lone.should_stop(partners_left=1) == "no_partner" and lone.should_stop(partners_left=2) is None
    three = _state(("a", "b", "c"))
    assert (three.min_turns, three.quiet_turns) == (6, 3)
    custom = _state(min_turns=3, quiet_turns=1)
    custom.add_note(0, "a", "Known: x.")
    _note(custom, "b")
    _note(custom, "a")
    assert custom.should_stop() == "converged"


def test_metrics_frontier_and_adds():
    s = _state()
    s.add_note(0, "a", "Known: x (Feb 2026).\nNewest I know: February 2026")
    s.add_note(1, "b", "Update: p.\nNewer: q (May 2026).\nNewer: r (Apr 2026).\nNewest I know: May 2026")
    s.add_note(2, "a", "Confirm: q.\nNewer: s (Jul 2026).\nNewest I know: July 2026")
    s.add_note(3, "b", "Confirm: s.\nNothing newer:\nNewest I know: Dec 2026")                    # a future date: ignored
    s.stop_reason = "max_turns"
    m = s.metrics(preset="rally")
    assert m["ring"] == ["a", "b"] and m["preset"] == "rally" and m["turns"] == 4 and m["hand_overs"] == 3
    assert m["stop_reason"] == "max_turns" and m["quiet_streak"] == 1 and m["unlabelled_replies"] == 0
    assert m["frontier"] == {"self_reported": True, "by_turn": ["2026-02", "2026-05", "2026-07", "2026-07"],
                             "first_declared": {"a": "2026-02", "b": "2026-05"}, "final": "2026-07", "gain_months": 2}
    assert m["adds_by_model"]["b"] == {"confirm": 1, "update": 1, "newer": 2, "unverified": 0, "nothing_newer": 1}
    assert m["adds_by_model"]["a"]["newer"] == 1 and len(m["notes"]) == 4 and m["notes"][3]["future"] is True
    empty = _state().metrics()
    assert empty["turns"] == 0 and empty["frontier"]["final"] is None and empty["frontier"]["gain_months"] is None


def test_a_ring_of_one_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        RallyState(RallyConfig(enabled=True), ["a"], TODAY)


# ---- prompts ----------------------------------------------------------------------------------------------------


def test_rally_prompts():
    system = relay_system_prompt_rally(160, "2026-11-26", "<<END_STEP>>")
    assert "about 160 tokens" in system and system.endswith("Today's date: 2026-11-26.") and "<<END_STEP>>" in system
    assert "FINAL ANSWER" not in system and "Nothing newer:" in system and "Do not write a final answer" in system
    assert "Today's date" not in relay_system_prompt_rally(160, None, "<<END_STEP>>")
    opening = relay_tail_rally(1, rally_directive(1))
    assert opening.startswith("END OF THE SHARED NOTES. Write note 1. This is the first note.")
    reply = relay_tail_rally(3, rally_directive(3, "Gemini 4 Neon", "2026-03"))
    assert "Write note 3." in reply and "Gemini 4 Neon" in reply and reply.count("March 2026") == 3
    assert "not stated" in rally_directive(2, None, None) and "another analyst" in rally_directive(2, None, None)
    answer = answer_system_prompt_rally("2026-11-26")
    assert "single source" in answer and "Basis:" in answer and answer.endswith("Today's date: 2026-11-26.")
    assert "Basis:" not in answer_system_prompt_rally(None, answer_basis=False)
