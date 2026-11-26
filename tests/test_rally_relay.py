"""The rally preset in the relay (DESIGN_rally.md sections 3 and 4): ring hand-over, requests, stopping, metrics."""

from __future__ import annotations

from typing import Any

from test_relay import MODELS, SENTINEL, FakeBackend, FakeRouter, collect, text_events

from cutoffmix.config import AppConfig, RallyConfig, RelayConfig, RouterConfig
from cutoffmix.relay import RallyTrace, Relay

# models a, b, c carry the display names "Model A", "Model B", "Model C"


def rally_cfg(*, ring=("a", "b"), relay: dict[str, Any] | None = None, **rally: Any) -> AppConfig:
    return AppConfig(models=list(MODELS), relay=RelayConfig(segment_tokens=64, **(relay or {})), router=RouterConfig(),
                     rally=RallyConfig(enabled=True, ring=tuple(ring), **rally), today="2026-11-24", preset="rally")


def make(script, cfg: AppConfig | None = None, *, pick=None):
    backend = FakeBackend(script)
    router = FakeRouter([m.key for m in MODELS], pick=pick)
    return Relay(cfg or rally_cfg(), backend, router, retry_base_delay=0.0), backend, router


def dry_script(model: str, req, n: int):
    """a opens (newest Feb 2026); b adds one newer item (May 2026); everything after that is quiet."""
    if req.phase == "answer":
        return text_events("The answer. (single source)")
    turn = len(req.segments) + 1
    if turn == 1:
        text = "Known: A happened (Feb 2026).\nNewest I know: February 2026"
    elif turn == 2:
        text = "Update: A was replaced (Apr 2026).\nNewer: B shipped (May 2026).\nNewest I know: May 2026"
    else:
        text = f"Confirm: B shipped, checked on pass {turn}.\nNothing newer:\nNewest I know: May 2026"
    return text_events(text + SENTINEL)


def loud_script(model: str, req, n: int):
    if req.phase == "answer":
        return text_events("The answer.")
    return text_events(f"Update: change number {len(req.segments)} (Mar 2026).\nNewest I know: March 2026{SENTINEL}")


def notes(events):
    return [e for e in events if e.type == "segment_end" and e.phase == "think"]


def done(events):
    return events[-1].data


async def test_models_write_in_ring_order_whatever_the_router_prefers():
    relay, backend, router = make(dry_script, pick=lambda avail, cur, n: avail[-1])
    events = await collect(relay)
    assert [m for m, r in backend.calls if r.phase == "think"] == ["a", "b", "a", "b"]
    assert all(e.decision == "forced" for e in notes(events))
    assert [e.data["probs"] for e in notes(events)] == [{"a": 1.0}, {"b": 1.0}, {"a": 1.0}, {"b": 1.0}]
    assert all(call["available"] == ["a", "b"] for call in router.decide_calls[:4])      # c is never offered
    starts = [e for e in events if e.type == "segment_start"]
    assert starts[0].data["rally"] == {"kind": "open", "turn": 1, "ring": ["a", "b"]}
    assert starts[3].data["rally"]["turn"] == 4
    assert starts[1].topic_posterior and starts[1].scores                       # the router still reports topic and scores


async def test_requests_carry_the_rally_prompts_and_name_the_previous_writer():
    relay, backend, _ = make(dry_script)
    await collect(relay)
    think = [r for m, r in backend.calls if r.phase == "think"]
    assert all("hand one set of working notes" in r.system_override for r in think)
    assert [r.metadata["kind"] for r in think] == ["open", "reply", "reply", "reply"]
    assert "This is the first note" in think[0].tail and think[0].tail.startswith("END OF THE SHARED NOTES. Write note 1.")
    assert "written by Model A" in think[1].tail and "February 2026" in think[1].tail
    assert "written by Model B" in think[2].tail and "May 2026" in think[2].tail
    answer = [r for m, r in backend.calls if r.phase == "answer"][0]
    assert "exchange between analysts" in answer.system_override and "Basis:" in answer.system_override


async def test_the_exchange_ends_when_a_full_revolution_adds_nothing():
    relay, backend, _ = make(dry_script)
    events = await collect(relay)
    d = done(events)
    assert d["status"] == "ok" and len(notes(events)) == 4 and d["answer"] == "The answer. (single source)"
    rally = d["rally"]
    assert rally["stop_reason"] == "converged" and rally["turns"] == 4 and rally["quiet_streak"] == 2
    assert [n["kind"] for n in rally["notes"]] == ["open", "reply", "reply", "reply"]
    assert [n["quiet"] for n in rally["notes"]] == [False, False, True, True]
    assert rally["frontier"]["by_turn"] == ["2026-02", "2026-05", "2026-05", "2026-05"]
    assert rally["frontier"]["first_declared"] == {"a": "2026-02", "b": "2026-05"} and rally["frontier"]["gain_months"] == 0
    assert rally["adds_by_model"]["b"]["newer"] == 1 and rally["adds_by_model"]["a"]["nothing_newer"] == 1
    assert rally["ring"] == ["a", "b"] and rally["preset"] == "rally" and rally["frontier"]["self_reported"] is True
    # per note: segment_end carries its record, the trace holds the same dicts
    assert notes(events)[1].data["rally"] == rally["notes"][1] == d["segments"][1]["rally"]
    assert "rally" not in d["segments"][-1] or d["segments"][-1]["phase"] == "answer"   # the answer is no note


async def test_max_turns_ends_an_exchange_that_never_goes_quiet():
    relay, backend, _ = make(loud_script, rally_cfg(max_turns=5))
    events = await collect(relay)
    rally = done(events)["rally"]
    assert len(notes(events)) == 5 and rally["stop_reason"] == "max_turns" and rally["quiet_streak"] == 0
    assert [m for m, r in backend.calls if r.phase == "think"] == ["a", "b", "a", "b", "a"]
    assert done(events)["answer"] == "The answer."


async def test_three_models_form_a_ring_and_the_answer_comes_from_it():
    def script(model, req, n):
        return dry_script(model, req, n) if req.phase == "think" else text_events("ok")
    cfg = rally_cfg(ring=("c", "a", "b"))
    relay, backend, router = make(script, cfg, pick=lambda avail, cur, n: avail[0])
    events = await collect(relay)
    assert [m for m, r in backend.calls if r.phase == "think"] == ["c", "a", "b", "c", "a", "b"]       # min_turns = 6
    assert done(events)["rally"]["turns"] == 6 and done(events)["rally"]["stop_reason"] == "converged"
    answer_call = router.decide_calls[-1]
    assert sorted(answer_call["available"]) == ["a", "b", "c"] and backend.calls[-1][1].phase == "answer"


async def test_the_answer_model_is_chosen_among_the_ring_only():
    relay, backend, router = make(dry_script, pick=lambda avail, cur, n: avail[0])
    await collect(relay)
    assert router.decide_calls[-1]["available"] == ["a", "b"] and backend.calls[-1][0] in ("a", "b")
    cfg = rally_cfg()
    cfg.rally.answer_from_ring = False
    relay, backend, router = make(dry_script, cfg, pick=lambda avail, cur, n: avail[-1])
    await collect(relay)
    assert router.decide_calls[-1]["available"] == ["a", "b", "c"] and backend.calls[-1][0] == "c"


async def test_a_conclusion_inside_a_note_is_set_aside():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("final")
        turn = len(req.segments) + 1
        if turn == 2 and model == "b" and n < 3:
            return text_events("Newer: B shipped (May 2026).\nFINAL ANSWER: B is out." + SENTINEL)
        if turn == 2:
            return text_events("Nothing newer:\nNewest I know: May 2026" + SENTINEL)
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    events = await collect(relay)
    second = notes(events)[1]
    assert second.model == "b" and "FINAL ANSWER" not in second.text and second.text.startswith("Newer: B shipped")
    assert second.data["premature_final"] is True and second.data["final"] is False
    assert done(events)["status"] == "ok" and "B is out" not in done(events)["answer"]


async def test_an_empty_note_that_only_concluded_is_dropped_and_rerouted():
    state = {"first": True}

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("final")
        turn = len(req.segments) + 1
        if turn == 2 and model == "b" and state["first"]:
            state["first"] = False
            return text_events("FINAL ANSWER: done" + SENTINEL)
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    events = await collect(relay)
    d = done(events)
    assert d["status"] == "ok" and d["dropped"][0]["reason"] == "empty" and d["dropped"][0]["premature_final"] is True
    assert [n.model for n in notes(events)][:2] == ["a", "b"]           # b is asked again for the same turn


async def test_a_refusing_partner_ends_the_exchange_without_failing_the_run():
    def script(model, req, n):
        if model == "b" and req.phase == "think":
            return [__import__("cutoffmix.types", fromlist=["StreamEvent"]).StreamEvent(
                kind="done", finish_reason="content_filter", error="refused: policy")]
        return dry_script(model, req, n)
    cfg = rally_cfg(relay={"disable_after_refusals": 2})
    relay, backend, _ = make(script, cfg)
    events = await collect(relay)
    d = done(events)
    assert d["status"] == "ok" and d["disabled"] == ["b"] and d["refusals"] == {"b": 2}
    assert d["rally"]["stop_reason"] == "no_partner" and d["rally"]["turns"] >= 1
    assert d["answer"] == "The answer. (single source)"


async def test_the_speculative_strategy_is_ignored_while_the_rally_runs():
    router = FakeRouter([m.key for m in MODELS], r=0.9)
    backend = FakeBackend(dry_script)
    relay = Relay(rally_cfg(relay={"strategy": "speculative"}), backend, router, retry_base_delay=0.0)
    events = await collect(relay)
    assert backend.max_open == 1 and done(events)["rally"]["stop_reason"] == "converged"
    assert not any(e.data and "drafts" in e.data for e in events if e.type == "segment_end")


async def test_without_the_rally_nothing_changes():
    cfg = AppConfig(models=list(MODELS), relay=RelayConfig(segment_tokens=64), router=RouterConfig(), today="2026-11-24")
    backend = FakeBackend(dry_script)
    relay = Relay(cfg, backend, FakeRouter([m.key for m in MODELS]), retry_base_delay=0.0)
    events = await collect(relay)
    assert "rally" not in done(events) and all("rally" not in s for s in done(events)["segments"])
    assert all(r.system_override is None and r.tail is None for _, r in backend.calls)
    trace = await Relay(cfg, FakeBackend(dry_script), FakeRouter([m.key for m in MODELS]),
                        retry_base_delay=0.0).run_to_trace("q")
    assert not isinstance(trace, RallyTrace) and "rally" not in trace.to_dict()
    ring_trace = await make(dry_script)[0].run_to_trace("q")
    assert isinstance(ring_trace, RallyTrace) and ring_trace.to_dict()["rally"]["turns"] == 4
