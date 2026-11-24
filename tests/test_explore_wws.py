"""The 'Where we stand:' line under the explore preset (DESIGN_explore §6.1 and §6.3): the reminder in the tail when
the previous note has none, the client-side hard cut as the cause the relay detects (``ExploreFlags.cut``), the gate
waiver after two notes whose only missing item is the line, the ``explore.wws`` metrics, and the mock run end to end.

Background: in the first live run (tests/fixtures/live_sse/relay_trace_explore_round_robin.json) German notes of
192 tokens ran to 800-996 characters after a 1000-character hard cut, every stream ended normally, and only one of
ten notes still had its trailing 'Where we stand:' line: the cut removed it silently, and the gate then waited for
something the cut had taken away.
"""

from __future__ import annotations

import functools
import json

from test_explore_relay import (
    WWS,
    analyst,
    explore_cfg,
    kinds_of,
    make_explore_relay,
    think_requests,
)
from test_relay import SENTINEL, collect, committed, text_events

from cutoffmix import cli
from cutoffmix.cli import main
from cutoffmix.config import load_config
from cutoffmix.prompts import (
    RECORD_OPEN_WWS_TEXT,
    WWS_ITEM,
    record_status_text,
    relay_tail,
)
from cutoffmix.relay import WWS_WAIVER_NOTES, RelayRuntime

QUESTION_DE = "Warum verlieren Lithium-Ionen-Akkus bei Kälte an Kapazität?"
TODAY = "2026-11-24"


def fixed_note(k: int, *, wws: bool = True, pad: int = 0) -> str:
    """A note with the whole checklist whose length does not depend on ``k`` (single digits); ``pad`` extra
    'Inference:' sentences go before the 'Where we stand:' line, so a hard cut set to the plain length removes
    exactly that line (the sentence trim ends the note after the padding)."""
    parts = [f"Claim: item {k} holds.", f"Evidence: record {k} shows it (source S{k}).",
             f"Alternative set aside: route R{k}, set aside because Q{k} fails.",
             f"Open question: what settles item {k}?"]
    parts += [f"Inference: because record {k} holds, therefore item {k} stands."] * pad
    if wws:
        parts.append(WWS)
    return " ".join(parts)


PLAIN_LEN = len(fixed_note(0))
assert all(len(fixed_note(k)) == PLAIN_LEN for k in range(10))


def tails_of(backend) -> list[str]:
    return [req.tail for req in think_requests(backend)]


# --------------------------------------------------------------------------------------
# The hard cut removes the line: the next tail reminds of it, the flags say why, the metrics count it
# --------------------------------------------------------------------------------------


async def test_a_note_cut_by_the_hard_cut_loses_its_line_and_the_next_tail_reminds_of_it():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if "The record is complete" in (req.tail or ""):
            return text_events("FINAL ANSWER: 42" + SENTINEL)
        k = len(req.segments)
        return text_events(fixed_note(k, pad=1 if k == 1 else 0) + SENTINEL)  # note 1 overshoots the cut

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=3, relay={"hard_cut_chars": PLAIN_LEN})
    relay, backend, _ = make_explore_relay(script, cfg)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok" and kinds_of(events) == ["main"] * 4
    ends = committed(events)
    # note 1: the stream ended normally (sentinel seen, finish "stop"), the client-side cut took the trailing line
    assert ends[1].text == fixed_note(1, wws=False, pad=1) and "Where we stand" not in ends[1].text
    assert ends[1].data["finish_reason"] == "stop" and ends[1].data["saw_sentinel"] is True
    assert ends[1].data["explore"]["cut"] is True and ends[1].data["explore"]["has_wws"] is False
    assert [e.data["explore"]["cut"] for e in ends] == [False, True, False, False]
    assert ends[0].data["explore"]["has_wws"] is True and ends[2].data["explore"]["has_wws"] is True
    tails = tails_of(backend)
    assert WWS_ITEM in tails[0] and "Where we stand" not in tails[1]  # note 0 ended with the line
    # the gate (a line in one of the last two notes) is satisfied by note 0, the reminder is in the tail anyway
    assert tails[2] == relay_tail(3, record_status_text(["one more main-line note"], False, remind_wws=True))
    assert tails[2].count("Where we stand") == 1 and "at the end of this note" in tails[2]
    assert tails[3] == relay_tail(4, record_status_text([], True))  # note 2 has the line: no reminder, gate open
    assert ends[3].data["final"] is True and "premature_final" not in ends[3].data
    ex = events[-1].data["explore"]
    assert ex["wws"] == {"missing": 0, "cut": 1, "waived": 0, "waived_at": []}
    assert ex["labels"]["coverage"]["where_we_stand"] == 0.6667 and ex["gate_open"] is True
    assert [s["explore"]["cut"] for s in events[-1].data["segments"] if s["phase"] == "think"] == [False, True,
                                                                                                    False, False]
    assert len(ex["wws_lines"]) == 2


async def test_a_truncated_stream_counts_as_cut_and_a_skipped_line_as_missing():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        if "The record is complete" in (req.tail or ""):
            return text_events("FINAL ANSWER: 42" + SENTINEL)
        k = len(req.segments)
        if k == 1:  # max_tokens reached before the line: no sentinel, finish "length", complete sentences only
            return text_events(fixed_note(k, wws=False), finish="length")
        if k == 3:  # the model simply skipped the line
            return text_events(fixed_note(k, wws=False) + SENTINEL)
        return text_events(fixed_note(k) + SENTINEL)

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=4)
    relay, backend, _ = make_explore_relay(script, cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main"] * 5 and events[-1].data["status"] == "ok"
    assert ends[1].data["finish_reason"] == "length" and ends[1].text == fixed_note(1, wws=False)
    assert [(e.data["explore"]["cut"], e.data["explore"]["has_wws"]) for e in ends] == [
        (False, True), (True, False), (False, True), (False, False), (False, False)]  # the last one is the marker
    tails = tails_of(backend)
    assert "at the end of this note" in tails[2]  # gate closed (two more notes): the line joins the missing items
    assert all("Where we stand" not in t for t in (tails[1], tails[3]))
    assert tails[4] == relay_tail(5, record_status_text([], True, remind_wws=True))  # gate open, reminder kept
    assert tails[4].endswith(RECORD_OPEN_WWS_TEXT.strip())
    ex = events[-1].data["explore"]
    assert ex["wws"] == {"missing": 1, "cut": 1, "waived": 0, "waived_at": []}


# --------------------------------------------------------------------------------------
# The waiver: a cut line must not hold the gate closed for good
# --------------------------------------------------------------------------------------


async def test_the_gate_waives_the_line_after_two_notes_where_it_is_the_only_missing_item():
    # every note overshoots the cut by one character: the trailing line never survives
    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=2, relay={"hard_cut_chars": PLAIN_LEN - 1})
    relay, backend, _ = make_explore_relay(analyst(note=fixed_note), cfg)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok" and events[-1].data["answer"].startswith("The answer is 42.")
    ends = committed(events)
    assert kinds_of(events) == ["main"] * 4
    assert all(e.data["explore"]["cut"] and not e.data["explore"]["has_wws"] for e in ends)
    assert all("Where we stand" not in e.text for e in ends)
    tails = tails_of(backend)
    # notes 0-1: the floor and the line; after note 1 the line is the only missing item (streak 1)
    assert tails[1] == relay_tail(2, record_status_text(["one more main-line note", WWS_ITEM], False))
    assert tails[2] == relay_tail(3, record_status_text([WWS_ITEM], False))
    assert "does not accept a conclusion" in tails[2]
    # after note 2 the streak reaches WWS_WAIVER_NOTES: the gate opens, the tail still asks for the line
    assert WWS_WAIVER_NOTES == 2
    assert tails[3] == relay_tail(4, record_status_text([], True, remind_wws=True))
    assert "The record is complete" in tails[3] and tails[3].endswith("end it with a 'Where we stand:' line.")
    assert ends[3].data["final"] is True and "premature_final" not in ends[3].data  # accepted, not set aside
    ex = events[-1].data["explore"]
    assert ex["wws"] == {"missing": 0, "cut": 4, "waived": 1, "waived_at": [2]}
    assert ex["gate_open"] is True and ex["gate_missing_at_final"] is None and ex["premature_finals"] == []
    assert ex["forced_continuations"] == 0 and ex["first_final_index"] == 3
    assert ex["labels"]["coverage"]["where_we_stand"] == 0.0 and ex["wws_lines"] == []
    assert ex["think_segments"] == 4  # without the waiver the run would go on to max_segments


async def test_the_waiver_lapses_when_the_line_returns_and_is_granted_again_later():
    def note(k: int) -> str:
        return fixed_note(k, wws=k == 3)  # only note 3 ends with the line; nothing is cut

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=2)
    relay, backend, _ = make_explore_relay(analyst(conclude_at=8, note=note), cfg)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok" and kinds_of(events) == ["main"] * 9
    ends = committed(events)
    assert not any(e.data["explore"]["cut"] for e in ends)
    tails = tails_of(backend)
    open_, closed = record_status_text([], True), record_status_text([WWS_ITEM], False)
    reminded = record_status_text([], True, remind_wws=True)
    assert tails[2] == relay_tail(3, closed)  # streak 1 after note 1
    assert tails[3] == relay_tail(4, reminded)  # waiver after note 2
    assert tails[4] == relay_tail(5, open_)  # note 3 has the line: the record is complete on its own merits
    assert tails[5] == relay_tail(6, reminded)  # note 4 lacks it, note 3 still in the window: open, reminded
    assert tails[6] == relay_tail(7, closed)  # notes 4 and 5 lack it: the rule applies again (streak 1)
    assert tails[7] == relay_tail(8, reminded) and tails[8] == relay_tail(9, reminded)  # second waiver after note 6
    assert ends[8].data["final"] is True
    ex = events[-1].data["explore"]
    assert ex["wws"] == {"missing": 8, "cut": 0, "waived": 2, "waived_at": [2, 6]}
    assert ex["labels"]["coverage"]["where_we_stand"] == round(1 / 9, 4) and len(ex["wws_lines"]) == 1


# --------------------------------------------------------------------------------------
# The mock run end to end (the command of the task): the gate opens at the expected segment, no cut, no waiver
# --------------------------------------------------------------------------------------

EXPECTED_KINDS = ["main", "main", "main", "continue", "main", "lens", "check", "main", "main"]


def test_cli_mock_run_with_the_preset_opens_the_gate_at_the_expected_segment(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli, "RelayRuntime", functools.partial(RelayRuntime, mock_latency_s=0.0))
    out = tmp_path / "explore.json"
    code = main(["run", "--mock", "--preset", "explore", "--seed", "3", "--today", TODAY, "--json", str(out),
                 QUESTION_DE])
    printed = capsys.readouterr().out
    assert code == 0
    trace = json.loads(out.read_text(encoding="utf-8"))
    ex = trace["explore"]
    assert trace["status"] == "ok" and ex["kinds"] == EXPECTED_KINDS and ex["main_line_segments"] == 6 + 1
    thinks = [s for s in trace["segments"] if s["phase"] == "think"]
    # six main-line notes are in after segment 7; segment 8 is written with the gate open and is the conclusion
    assert thinks[8]["text"] == "" and ex["first_final_index"] == 2 and ex["premature_finals"][0]["index"] == 2
    assert ex["gate_open"] is True and ex["gate_missing_at_final"] is None
    assert all(s["explore"]["cut"] is False for s in thinks) and max(len(s["text"]) for s in thinks) < 1500
    assert all(s["explore"]["has_wws"] for s in thinks[:8])  # the mock always ends a note with the line
    assert ex["wws"] == {"missing": 0, "cut": 0, "waived": 0, "waived_at": []}
    summary = [line for line in printed.splitlines() if line.startswith("segments ")][-1]
    assert "lenses 1 (0 kept, 1 dropped) · labels 100% · topics 1 · forced 1" in summary and "wws" not in summary


async def test_mock_tails_carry_the_line_only_in_the_first_request_and_open_the_gate_for_segment_8():
    cfg = load_config(preset="explore", overrides={"router": {"seed": 3}, "today": TODAY})
    assert cfg.relay.hard_cut_chars == 1500 and cfg.relay.segment_tokens == 192
    runtime = RelayRuntime(cfg, mock=True, seed=3, mock_latency_s=0.0)
    seen: list[tuple[int, str]] = []
    stream = runtime.backend.stream_continuation

    def spy(model, req):
        if req.phase == "think":
            seen.append((req.metadata["segment_index"], req.tail or ""))
        return stream(model, req)

    runtime.backend.stream_continuation = spy  # type: ignore[method-assign]
    trace = await runtime.relay().run_to_trace(QUESTION_DE)
    assert trace.to_dict()["explore"]["kinds"] == EXPECTED_KINDS
    assert [i for i, t in seen if "The record is complete" in t] == [8]
    assert [i for i, t in seen if WWS_ITEM in t] == [0]  # listed with the checklist of the first note, never reminded
    assert not any(RECORD_OPEN_WWS_TEXT.strip() in t for _, t in seen)
    await runtime.aclose()
