"""The rally preset's configuration, note parsing, state machine and prompts (DESIGN_rally.md sections 2 to 6)."""

from __future__ import annotations

import datetime as dt
import random
import re
import time

import pytest

from cutoffmix import rally
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


def test_partner_failures_and_light_models_are_validated_and_round_trip():
    with pytest.raises(ConfigError, match="partner_failures must be >= 1"):
        RallyConfig(partner_failures=0)
    with pytest.raises(ConfigError, match="light_schema_models"):
        RallyConfig(light_schema_models=("a", ""))
    with pytest.raises(ConfigError, match="unknown model"):
        AppConfig(models=list(MODELS), rally=RallyConfig(light_schema_models=("nope",)))
    cfg = AppConfig(models=list(MODELS), rally=RallyConfig(enabled=True, ring=("a", "b"), light_schema_models=("a",),
                                                            partner_failures=3))
    again = config_from_dict(cfg.to_dict())
    assert again.rally == cfg.rally and again.to_dict()["rally"]["light_schema_models"] == ["a"]
    assert RallyConfig().partner_failures == 2 and RallyConfig().light_schema_models == ()


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
    # a future month whose year is not future must not leave a bare year behind (review P2)
    ("December 2026", (None, False, True)),
    ("Dec 2026", (None, False, True)),
    ("December 15, 2026", (None, False, True)),
    ("December 2026 and March 2025", ("2025-03", False, True)),
    ("since 2026", ("2026-01", True, False)),                    # a year that no month token has used is still a year
    ("Version 2026", ("2026-01", True, False)),
    ("the 2026 model", ("2026-01", True, False)),
    # separators between the month and the year, and numeric day-first dates (review P8)
    ("March, 2026", ("2026-03", False, False)),
    ("February, 2026", ("2026-02", False, False)),
    ("Feb-2026", ("2026-02", False, False)),
    ("Feb.2026", ("2026-02", False, False)),
    ("Feb. 2026", ("2026-02", False, False)),
    ("Nov. 12, 2026", ("2026-11", False, False)),
    ("12 Nov 2026", ("2026-11", False, False)),
    ("3. März 2026", ("2026-03", False, False)),
    ("mid-2026", ("2026-01", True, False)),
    ("late 2025", ("2025-01", True, False)),
    ("Q4-2025", ("2025-10", True, False)),
    ("Q4/2025", ("2025-10", True, False)),
    ("17.03.2026", ("2026-03", False, False)),
    ("1.3.2026", ("2026-03", False, False)),
    ("17/03/2026", ("2026-03", False, False)),
    ("03-2026", ("2026-03", False, False)),
    ("03.2026", ("2026-03", False, False)),
    ("2026/03", ("2026-03", False, False)),
    # what must stay unread: month-first US dates, versions, addresses, ranges
    ("03/17/2026", (None, False, False)),
    ("1.2.3.2026", (None, False, False)),
    ("192.168.1.2026", (None, False, False)),
    ("v1.3.2026", (None, False, False)),
    ("2025-2026", (None, False, False)),
])
def test_parse_month(text, expected):
    assert parse_month(text, TODAY) == expected
    assert parse_month(text, dt.date.fromisoformat(TODAY)) == expected


@pytest.mark.parametrize("text, today, expected", [
    ("Q4 2026", "2026-07-01", (None, False, True)),              # a future quarter leaves no bare year behind
    ("Q4 2026", "2026-11-26", ("2026-10", True, False)),
    ("April 2026", "2026-02-10", (None, False, True)),
    ("April 2026", "2026-04-01", ("2026-04", False, False)),     # the current month is not the future
    ("May 2026 and 2026", "2026-06-30", ("2026-05", False, False)),
])
def test_parse_month_against_today(text, today, expected):
    assert parse_month(text, today) == expected


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


def _labels(text: str) -> dict[str, int]:
    return {name: n for name, n in parse_note(text, TODAY)["labels"].items() if n}


@pytest.mark.parametrize("text, expected", [
    # qualifiers between the label and the colon (review P1)
    ("Confirm: x\nUpdate (March 2026): y\nNewer (April 2026): z", {"confirm": 1, "update": 1, "newer": 1}),
    ("**Update (Mar 2026):** y", {"update": 1}),
    ("Update 1: y\nUpdate 2: z", {"update": 2}),
    ("Confirm: x\nUpdates: y", {"confirm": 1, "update": 1}),
    ("Correction: the budget passed in Nov 2025.", {"update": 1}),            # a correction is an update
    ("**Update**: y\n__Newer__: z", {"update": 1, "newer": 1}),
    ("### Update: y\n## Newer: z", {"update": 1, "newer": 1}),
    ("Update:\u00a0y\n\u00a0Newer: z", {"update": 1, "newer": 1}),            # no-break spaces
    # 'Nothing newer than <date>' with or without the colon, as the reply directive itself words it
    ("Nothing newer than March 2026:\nNewest I know: March 2026", {"nothing_newer": 1}),
    ("Nothing newer than March 2026.\nNewest I know: March 2026", {"nothing_newer": 1}),
    ("**Nothing newer** than March 2026.", {"nothing_newer": 1}),
    ("Confirm: x. Nothing newer than Feb 2026 is known to me.", {"confirm": 1, "nothing_newer": 1}),
    ("Nothing newer: I know nothing after March 2026.", {"nothing_newer": 1}),
    # what is prose and no label
    ("Newer than the 2025 report: nothing.", {}),
    ("Confirm: a\nNewer than March 2026: B shipped.", {"confirm": 1}),
    ("I know nothing newer than March 2026.", {}),
    ("The Update: was wrong.", {}),
    ("Updated: y", {}),
    ("Nothing newer thanks", {}),
    ("Update (", {}),
])
def test_label_variants(text, expected):
    assert _labels(text) == expected


def test_a_dated_label_does_not_hide_an_update_from_the_stop_rule():
    note = ("Confirm: A launched Feb 2026.\nUpdate (March 2026): A was renamed B.\nNewer (April 2026): B2 shipped.\n"
            "Newest I know: April 2026")
    labels = parse_note(note, TODAY)["labels"]
    assert (labels["confirm"], labels["update"], labels["newer"]) == (1, 1, 1)


_OLD_MARK = r"(?:\*\*|__)?"
_OLD_LABEL_RE = re.compile(  # the first version of the label pattern: the rewrite must find the same labels
    r"(?:^|(?<=[.!?;)\]])[ \t]+)[ \t]*(?:[-*•][ \t]+|\d{1,2}[.)][ \t]+)?" + _OLD_MARK + r"[ \t]*"
    r"(Known|Confirm|Update|Newer|Unverified|Nothing[ \t]+newer|Newest[ \t]+I[ \t]+know)[ \t]*"
    + _OLD_MARK + r"[ \t]*:",
    re.IGNORECASE | re.MULTILINE)


def test_the_linear_label_pattern_finds_what_the_first_version_found():
    atoms = ["Known", "Confirm", "Update", "Newer", "Nothing newer", "Newest I know", "Nothing", "newer", ":", ".", "!",
             ";", ")", "]", " ", "  ", "\t", "\n", "-", "*", "**", "__", "•", "1.", "12)", "x", "foo", "_", "123."]
    rng = random.Random(1)
    for _ in range(30000):
        text = "".join(rng.choice(atoms) for _ in range(rng.randint(1, 14)))
        old = [(" ".join(m.group(1).lower().split()), m.end()) for m in _OLD_LABEL_RE.finditer(text)]
        new = [(" ".join((m.group("label") or m.group("than")).lower().split()), m.end())
               for m in rally._LABEL_RE.finditer(text)]
        assert old == new, text


@pytest.mark.parametrize("prefix", [".", ";", ")", "\n", "\n- ", ".\n", "** ", "Nothing", "Update", "Nothing newer"])
@pytest.mark.parametrize("space", [" ", "\t", "\u00a0"])
def test_parsing_is_linear_in_the_length_of_a_run_of_whitespace(prefix, space):
    text = prefix + space * 200_000 + "x"
    start = time.perf_counter()
    parse_note(text, TODAY)
    assert time.perf_counter() - start < 1.0, f"{prefix!r} + 200000 x {space!r}"


@pytest.mark.parametrize("text", [
    ". " * 100_000, "- " * 100_000, "** " * 100_000, "Known " * 50_000, "1. " * 50_000, ". **" * 50_000,
    "March 2026 " * 20_000, "x" * 200_000, "0" * 200_000, " " * 200_000 + "2026", "a-" * 100_000, "Q4" * 100_000,
])
def test_parsing_is_linear_for_other_shapes(text):
    start = time.perf_counter()
    parse_note(text, TODAY)
    assert time.perf_counter() - start < 2.0


# ---- state ------------------------------------------------------------------------------------------------------

OPENING = "Known: x (Jan 2026).\nNewest I know: January 2026"


def _state(ring=("a", "b"), **cfg) -> RallyState:
    return RallyState(RallyConfig(enabled=True, ring=tuple(ring), **cfg), list(ring), TODAY)


def _note(state, model, update=0, newer=0, quiet=True, newest="January 2026", **kwargs):
    parts = ["Update: x." for _ in range(update)] + ["Newer: y." for _ in range(newer)]
    parts.append("Nothing newer:" if quiet else "Confirm: z.")
    return state.add_note(len(state.notes), model, "\n".join(parts) + f"\nNewest I know: {newest}", **kwargs)


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
    s.add_note(0, "a", OPENING)
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
    s.add_note(0, "a", OPENING)
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
                             "by_turn_approx": [False, False, False, False],
                             "first_declared": {"a": "2026-02", "b": "2026-05"}, "final": "2026-07", "gain_months": 2,
                             "gain_months_approx": None}
    assert m["self_replies"] == 0 and m["skipped_hand_overs"] == []
    assert m["adds_by_model"]["b"] == {"confirm": 1, "update": 1, "newer": 2, "unverified": 0, "nothing_newer": 1}
    assert m["adds_by_model"]["a"]["newer"] == 1 and len(m["notes"]) == 4 and m["notes"][3]["future"] is True
    empty = _state().metrics()
    assert empty["turns"] == 0 and empty["frontier"]["final"] is None and empty["frontier"]["gain_months"] is None


def test_a_ring_of_one_is_refused():
    with pytest.raises(ValueError, match="at least two"):
        RallyState(RallyConfig(enabled=True), ["a"], TODAY)


@pytest.mark.parametrize("note, quiet", [
    ("Confirm: ok.\nNewest I know: January 2026", True),
    ("Unverified: maybe.\nNewest I know: January 2026", True),
    ("Nothing newer:\nNewest I know: January 2026", True),
    ("Nothing newer:", True),
    ("Nothing newer than January 2026.\nNewest I know: January 2026", True),
    ("Confirm: ok.\nNewest I know: June 2026", False),                 # a later newest month, no 'Newer:' behind it
    ("Nothing newer:\nNewest I know: June 2026", False),               # a note that contradicts itself is not quiet
    ("Confirm: ok.\nUpdate: x.\nNewest I know: January 2026", False),
    ("Confirm: ok.\nUpdate (March 2026): x.\nNewest I know: January 2026", False),
    ("Confirm: ok.\nNewer: y.\nNewest I know: January 2026", False),
    ("Correction: x was wrong.\nNewest I know: January 2026", False),
    ("Known: z (Dec 2025).\nNewest I know: January 2026", False),      # restating facts is no 'nothing newer'
    ("Just prose.", False),
    ("Just prose.\nNewest I know: not sure", False),
    # light notes carry no labels: a readable closing line that is not above the record is the whole signal
    ("Nothing more recent that I know of.\nNewest I know: January 2026", True),
    ("Das stimmt so.\nNewest I know: Dezember 2025", True),
    ("Das stimmt so.\nNewest I know: June 2026", False),
])
def test_a_reply_is_quiet_only_when_it_adds_nothing_and_says_so(note, quiet):
    s = _state()
    s.add_note(0, "a", OPENING)
    assert s.add_note(1, "b", note)["quiet"] is quiet


def test_label_free_light_replies_can_close_the_exchange():
    s = _state(("fable", "neon"))
    s.add_note(0, "fable", "Known: A (May 2025).\nNewest I know: May 2025", light=True)
    s.add_note(1, "neon", "Confirm: A.\nNewer: B (Nov 2025).\nNewest I know: November 2025")
    note = s.add_note(2, "fable", "Ich kenne nichts Neueres als November 2025; die Angaben stimmen.\n"
                                  "Newest I know: November 2025", light=True)
    assert note["light"] is True and note["quiet"] is True and sum(note["labels"].values()) == 0
    s.add_note(3, "neon", "Confirm: ok.\nNothing newer:\nNewest I know: November 2025")
    assert s.should_stop() == "converged"
    assert s.metrics()["unlabelled_replies"] == 1                    # the light note, counted as before


def test_a_hard_cut_note_is_read_from_the_whole_text_and_never_quiet_when_its_end_is_unknown():
    s = _state()
    s.add_note(0, "a", OPENING)
    whole = "Confirm: x.\nNothing newer:\nNewest I know: January 2026"
    note = s.add_note(1, "b", "Confirm: x.", full=whole, cut=True)          # the cut removed the lines after the first
    assert note["cut"] is True and note["newest"] == "2026-01" and note["labels"]["nothing_newer"] == 1
    assert note["quiet"] is True and note["light"] is False
    lost = s.add_note(2, "a", "Confirm: x.\nConfirm: y.", cut=True)         # truncated stream: the end is unknown
    assert lost["cut"] is True and lost["newest"] is None and lost["quiet"] is False
    whole_newer = "Confirm: x.\nNewer: y shipped (Oct 2026).\nNewest I know: October 2026"
    hidden = s.add_note(3, "b", "Confirm: x.", full=whole_newer, cut=True)   # the cut removed a 'Newer:' line
    assert hidden["labels"]["newer"] == 1 and hidden["newest"] == "2026-10" and hidden["quiet"] is False
    assert s.frontier() == "2026-10"
    kept = _state()
    kept.add_note(0, "a", OPENING)
    assert kept.add_note(1, "b", "Confirm: x.\nConfirm: y.")["quiet"] is True    # uncut, no closing line: its choice


def test_a_self_reply_is_never_quiet_and_two_in_a_row_end_the_exchange():
    s = _state()
    s.add_note(0, "a", OPENING)
    _note(s, "b", newer=1, quiet=False)
    _note(s, "a")                                                # a reads b's note and finds nothing newer
    assert [n["quiet"] for n in s.notes] == [False, False, True] and s.self_reply_streak() == 0
    third = _note(s, "a")                                        # b could not take its turn: a answers its own note
    assert third["self_reply"] is True and third["quiet"] is False
    assert s.should_stop() is None                               # one fallback note is allowed (a hand-over can recover)
    _note(s, "a")
    assert s.self_reply_streak() == 2 and s.should_stop() == "no_partner"
    m = s.metrics()
    assert m["self_replies"] == 2 and [n["self_reply"] for n in m["notes"]] == [False, False, False, True, True]
    _note(s, "b")                                                # the partner is back: the streak is over
    assert s.self_reply_streak() == 0


def test_a_monologue_of_one_model_never_looks_like_convergence():
    s = _state()
    s.add_note(0, "a", OPENING)
    for _ in range(3):
        _note(s, "a")                                            # the partner never answers (dead provider)
    assert [n["quiet"] for n in s.notes] == [False, False, False, False]
    assert s.quiet_streak() == 0 and s.should_stop() == "no_partner"


def test_converged_needs_a_revolution_of_different_writers():
    s = _state(("a", "b", "c"))
    s.add_note(0, "a", OPENING)
    for model in ("b", "c", "a", "b"):
        _note(s, model)
    assert len(s.notes) == 5 and s.should_stop() is None         # below min_turns (6)
    _note(s, "c")
    assert s.should_stop() == "converged"                        # the last three notes: a, b, c

    shrunk = _state(("a", "b", "c"))
    shrunk.add_note(0, "a", OPENING)
    for model in ("b", "a", "b", "a", "b"):
        _note(shrunk, model)                                     # c is out; a and b keep exchanging
    assert shrunk.should_stop() is None                          # the whole ring is expected to have taken part ...
    assert shrunk.should_stop(partners_left=2) == "converged"    # ... unless it has shrunk to the two that can write
    assert shrunk.should_stop(partners_left=1) == "converged"    # converged wins over no_partner when both hold
    assert _state().should_stop(partners_left=0) == "no_partner"


def test_frontier_is_the_running_maximum_of_the_declared_months():
    s = _state()
    assert s.frontier() is None
    s.add_note(0, "a", "Known: x.\nNewest I know: not sure")
    assert s.frontier() is None
    s.add_note(1, "b", "Newer: y (May 2026).\nNewest I know: May 2026")
    s.add_note(2, "a", "Nothing newer:\nNewest I know: March 2026")
    assert s.frontier() == "2026-05"
    s.add_note(3, "b", "Newer: z.\nNewest I know: December 2026")          # after today (2026-11-26): never enters
    assert s.frontier() == "2026-05" and s.notes[3]["future"] is True and s.notes[3]["newest"] is None


def test_failed_hand_overs_count_up_to_the_models_exit():
    s = _state(partner_failures=3)
    assert s.hand_over_failed("b") is False and s.hand_over_failed("b") is False
    assert s.hand_over_failed("b") is True                       # the third in a row
    s.hand_over_ok("b")
    assert s.hand_over_failed("b") is False                      # the streak starts again
    assert s.hand_over_failed("a", fatal=True) is True           # not retryable: out at once

    t = _state()                                                 # partner_failures defaults to 2
    assert t.hand_over_failed("b") is False
    t.add_note(0, "b", OPENING)                                  # a committed note ends the streak, too
    assert t.hand_over_failed("b") is False and t.hand_over_failed("b") is True
    assert t.hand_over_failed("a") is False                      # the counts are per model


def test_skipped_hand_overs_are_reported():
    s = _state()
    s.add_note(0, "a", OPENING)
    s.note_skipped(1, "b", "error: HTTP 402")
    s.note_skipped(1, "b", "refused")
    m = s.metrics()
    assert m["skipped_hand_overs"] == [{"index": 1, "model": "b", "reason": "error: HTTP 402"},
                                       {"index": 1, "model": "b", "reason": "refused"}]
    m["skipped_hand_overs"].clear()
    assert len(s.skipped) == 2                                   # the metrics hold copies


def _frontier(notes, ring=("a", "b")):
    s = _state(ring)
    for i, (model, text) in enumerate(notes):
        s.add_note(i, model, text)
    return s.metrics()["frontier"]


def test_gain_is_unknown_when_a_writers_first_note_is_undated():
    opening = "Known: A launched (Feb 2026).\nNewest I know: February 2026"
    f = _frontier([("a", opening), ("b", "Confirm: A.\nNewest I know: not sure"),
                   ("a", "Nothing newer:\nNewest I know: February 2026"),
                   ("b", "Newer: C shipped (Oct 2026).\nNewest I know: October 2026")])
    assert f["first_declared"] == {"a": "2026-02", "b": None} and f["final"] == "2026-10"
    assert f["gain_months"] is None and f["gain_months_approx"] is None      # not 8: b's own knowledge is no gain
    g = _frontier([("a", opening), ("b", "Confirm: A. Newer: C shipped (Oct 2026).\nNewest I know: October 2026")])
    assert g["gain_months"] == 0                                              # dated at once: nothing the exchange added
    h = _frontier([("a", opening), ("b", "Confirm: A.\nUnverified: x"), ("a", "Newer: D.\nNewest I know: May 2026")])
    assert h["first_declared"]["b"] is None and h["gain_months"] is None      # no closing line at all


def test_approximate_dates_never_look_like_a_lift():
    exact = _frontier([("a", "Known: x.\nNewest I know: March 2025"), ("b", "Newer: y.\nNewest I know: June 2025")])
    assert exact["gain_months"] == 0 and exact["by_turn_approx"] == [False, False]
    # a writer who opens with a bare year and later names a month of that year has added precision (P5)
    precise = _frontier([("a", "Known: x.\nNewest I know: 2025"), ("b", "Confirm: x.\nNewest I know: March 2025"),
                         ("a", "Confirm: x again.\nNewest I know: October 2025")])
    assert precise["by_turn"] == ["2025-01", "2025-03", "2025-10"] and precise["by_turn_approx"] == [True, False, False]
    assert precise["gain_months"] is None and precise["gain_months_approx"] == 7
    # an approximate final month is a lift only roughly
    rough = _frontier([("a", "Known: x.\nNewest I know: March 2025"),
                       ("b", "Nothing newer:\nNewest I know: March 2025"), ("a", "Newer: y.\nNewest I know: Q4 2025")])
    assert rough["by_turn_approx"] == [False, False, True] and rough["gain_months"] is None
    assert rough["gain_months_approx"] == 7
    # an approximate opening far below the baseline cannot hide anything: the gain stays exact
    far = _frontier([("a", "Known: x.\nNewest I know: 2024"), ("b", "Known: y.\nNewest I know: March 2025"),
                     ("a", "Newer: z.\nNewest I know: June 2025")])
    assert far["gain_months"] == 3 and far["gain_months_approx"] is None
    assert far["by_turn_approx"] == [True, False, False]
    # an exact date that reaches the month of an approximate maximum takes over the maximum
    tie = _frontier([("a", "Known: x.\nNewest I know: Q4 2025"), ("b", "Confirm: x.\nNewest I know: October 2025")])
    assert tie["by_turn"] == ["2025-10", "2025-10"] and tie["by_turn_approx"] == [True, False]
    # nothing declared: no maximum, nothing approximate
    none = _frontier([("a", "Known: x."), ("b", "Confirm: x.")])
    assert none["by_turn"] == [None, None] and none["by_turn_approx"] == [False, False] and none["gain_months"] is None


def test_a_future_month_in_the_current_year_is_not_a_date_the_next_writer_hears_about():
    s = _state()
    s.add_note(0, "a", "Known: x (Mar 2025).\nNewest I know: March 2025")
    note = s.add_note(1, "b", "Newer: y (Dec 2026).\nNewest I know: December 2026")
    assert note["newest"] is None and note["future"] is True and note["approx"] is False
    f = s.metrics()["frontier"]
    assert f["by_turn"] == ["2025-03", "2025-03"] and f["first_declared"] == {"a": "2025-03", "b": None}


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
