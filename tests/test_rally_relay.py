"""The rally preset in the relay (DESIGN_rally.md sections 3 and 4): ring hand-over, requests, stopping, metrics."""

from __future__ import annotations

from typing import Any

from test_relay import MODELS, SENTINEL, FakeBackend, FakeRouter, FlakyError, collect, text_events

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
    assert starts[0].data["rally"] == {"kind": "open", "turn": 1, "ring": ["a", "b"], "self_reply": False}
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


# ---- review fixes: failed hand-overs, self-replies, cut notes, echo guard, light notes -----------------------------


def think_calls(backend):
    return [(m, r) for m, r in backend.calls if r.phase == "think"]


async def test_a_dead_partner_ends_the_exchange_instead_of_turning_it_into_a_monologue():
    def script(model, req, n):
        if model == "b" and req.phase == "think":
            return [FlakyError("HTTP 402", retryable=False)]
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    events = await collect(relay)
    d = done(events)
    rally = d["rally"]
    assert d["status"] == "ok" and d["disabled"] == ["b"] and d["answer"] == "The answer. (single source)"
    assert rally["stop_reason"] == "no_partner" and rally["turns"] == 2
    assert [n["model"] for n in rally["notes"]] == ["a", "a"]
    assert [n["self_reply"] for n in rally["notes"]] == [False, True] and rally["self_replies"] == 1
    assert rally["notes"][1]["quiet"] is False
    assert rally["skipped_hand_overs"] == [{"index": 1, "model": "b", "reason": "error"}]
    # attempts: a (note 1), b (failed hand-over), a (note 2, replying to its own note)
    assert [(e.model, e.data["rally"]["self_reply"]) for e in events if e.type == "segment_start"] == [
        ("a", False), ("b", False), ("a", True)]
    self_reply = think_calls(backend)[2][1]                                    # calls: a, b (failed), a
    assert self_reply.metadata["self_reply"] is True and "could not take its turn" in self_reply.tail
    assert "knowledge differs" not in self_reply.tail
    answer = backend.calls[-1][1]
    assert "Note 2 (counting the notes from 1" in answer.system_override


async def test_a_partner_that_recovers_keeps_the_exchange_going():
    state = {"b_calls": 0}

    def script(model, req, n):
        if model == "b" and req.phase == "think":
            state["b_calls"] += 1
            if state["b_calls"] <= 3:                                          # one hand-over: first call plus two retries
                return [FlakyError("HTTP 503")]
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    d = done(await collect(relay))
    rally = d["rally"]
    assert d["disabled"] == [] and rally["stop_reason"] == "converged" and rally["turns"] == 4
    assert [n["model"] for n in rally["notes"]] == ["a", "a", "b", "a"]
    assert [n["self_reply"] for n in rally["notes"]] == [False, True, False, False] and rally["self_replies"] == 1
    assert rally["skipped_hand_overs"] == [{"index": 1, "model": "b", "reason": "error"}]


async def test_a_self_reply_never_closes_the_exchange():
    def script(model, req, n):
        if model == "b" and req.phase == "think" and len(req.segments) == 3:
            return [FlakyError("HTTP 503")] * 1                                # b cannot take turn 4, a answers its own note 3
        return dry_script(model, req, n)
    relay, backend, _ = make(script, rally_cfg(max_turns=6))
    d = done(await collect(relay))
    rally = d["rally"]
    assert [n["model"] for n in rally["notes"]][:4] == ["a", "b", "a", "a"]
    assert rally["notes"][3]["self_reply"] is True and rally["notes"][3]["quiet"] is False
    assert rally["stop_reason"] != "converged" or len({n["model"] for n in rally["notes"][-2:]}) == 2


async def test_when_no_ring_model_can_write_the_answer_comes_from_the_record():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("answer from the record")
        if req.segments:
            return [FlakyError("HTTP 404", retryable=False)]
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    d = done(await collect(relay))
    assert d["status"] == "ok" and d["rally"]["stop_reason"] == "no_partner" and d["rally"]["turns"] == 1
    assert sorted(d["disabled"]) == ["a", "b"] and d["answer"] == "answer from the record"
    assert backend.calls[-1][0] == "c"                                          # the ring is out: any model answers


async def test_a_note_cut_by_the_hard_cut_is_read_from_its_whole_text():
    body = "\n".join(f"Confirm: claim {i} holds (Jan 2026)." for i in range(8))
    reply = body + "\nNewer: B shipped (May 2026).\nNewest I know: May 2026"      # ~340 characters, the cut is at 300

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("ok")
        turn = len(req.segments) + 1
        if turn == 1:
            return text_events("Known: A (Feb 2026).\nNewest I know: February 2026" + SENTINEL)
        if turn == 2:
            return text_events(reply + SENTINEL)
        return dry_script(model, req, n)
    relay, backend, _ = make(script, rally_cfg(relay={"hard_cut_chars": 300}))
    events = await collect(relay)
    note = done(events)["rally"]["notes"][1]
    assert note["cut"] is True and note["labels"]["newer"] == 1 and note["quiet"] is False
    assert note["newest"] == "2026-05"
    committed_text = notes(events)[1].text
    assert len(committed_text) <= 300 and "Newest I know" not in committed_text
    assert "May 2026" in think_calls(backend)[2][1].tail                        # the next writer is told what the note said


async def test_a_quiet_note_may_repeat_the_wording_of_an_earlier_quiet_note():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("ok")
        turn = len(req.segments) + 1
        if turn == 1:
            return text_events("Known: A happened (Feb 2026).\nNewest I know: February 2026" + SENTINEL)
        if turn == 2:
            return text_events("Update: A was replaced (Apr 2026).\nNewer: B shipped (May 2026).\n"
                               "Newest I know: May 2026" + SENTINEL)
        return text_events("Nothing newer: nothing after May 2026 on this.\nNewest I know: May 2026" + SENTINEL)
    relay, backend, _ = make(script)
    events = await collect(relay)
    d = done(events)
    assert d["dropped"] == [] and d["rally"]["stop_reason"] == "converged" and d["rally"]["turns"] == 4
    assert notes(events)[2].text == notes(events)[3].text


async def test_a_loud_repeat_of_an_earlier_note_is_still_dropped():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("ok")
        turn = len(req.segments) + 1
        if turn == 1:
            return text_events("Known: A happened (Feb 2026).\nNewest I know: February 2026" + SENTINEL)
        return text_events("Update: A was replaced (Apr 2026).\nNewer: B shipped (May 2026).\n"
                           "Newest I know: May 2026" + SENTINEL)
    relay, backend, _ = make(script)
    d = done(await collect(relay))
    assert [row["reason"] for row in d["dropped"]] == ["empty", "empty"] and d["rally"]["stop_reason"] == "empty"
    assert d["rally"]["turns"] == 2


async def test_light_notes_for_a_model_carry_no_label_schema():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("ok")
        turn = len(req.segments) + 1
        if model == "a":
            if turn == 1:
                return text_events("A happened in February 2026.\nNewest I know: February 2026" + SENTINEL)
            return text_events(f"I know nothing more recent than May 2026 (pass {turn}).\n"
                               "Newest I know: May 2026" + SENTINEL)
        return dry_script(model, req, n)
    relay, backend, _ = make(script, rally_cfg(light_schema_models=("a",)))
    d = done(await collect(relay))
    rally = d["rally"]
    assert rally["stop_reason"] == "converged" and rally["turns"] == 4          # a's label-free replies can be quiet
    assert [n["light"] for n in rally["notes"]] == [True, False, True, False]
    light = [r for m, r in think_calls(backend) if m == "a"]
    full = [r for m, r in think_calls(backend) if m == "b"]
    assert all("jointly working" in r.system_override and "hand one set" not in r.system_override for r in light)
    assert all("'Confirm:'" not in r.tail and "does not accept a conclusion" in r.tail for r in light)
    assert all(r.metadata["light"] is True for r in light) and all(r.metadata["light"] is False for r in full)
    assert all("hand one set of working notes" in r.system_override and "'Confirm:'" in r.tail for r in full)
    assert "Gemini" not in light[1].tail and "Model" not in light[1].tail      # the other analyst is not named


async def test_an_approximate_newest_date_is_quoted_not_converted():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("ok")
        if len(req.segments) == 0:
            return text_events("Known: A happened (Feb 2025).\nNewest I know: late 2025" + SENTINEL)
        return dry_script(model, req, n)
    relay, backend, _ = make(script)
    await collect(relay)
    tail = think_calls(backend)[1][1].tail
    assert '"late 2025"' in tail and "January 2025" not in tail
    assert "that the notes lack" in tail and "do not yet contain" in tail
