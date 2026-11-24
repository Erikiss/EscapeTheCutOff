"""Tests for the explore preset in cutoffmix.relay (DESIGN.md §7.6): a scripted fake backend whose notes follow the
labelled template and react to the per-segment tail, plus the recorded expectation that the default relay (preset
off) is byte-identical to before."""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import pytest
from test_relay import (
    MODELS,
    SENTINEL,
    FakeBackend,
    FakeRouter,
    collect,
    committed,
    drafting_script,
    finishing_script,
    make_relay,
    para,
    text_events,
)

from cutoffmix.config import AppConfig, ExploreConfig, LensConfig, RelayConfig, RouterConfig
from cutoffmix.lenses import field_by_name, tier_of
from cutoffmix.prompts import (
    answer_system_prompt_explore,
    check_text,
    continue_text,
    parse_labels,
    record_status_text,
    relay_system_prompt_explore,
    relay_tail,
    vs_direction_text,
)
from cutoffmix.relay import VS_MENU_MAX_TOKENS, VS_MENU_MODEL, ExploreTrace, Relay, to_jsonable
from cutoffmix.types import ChatResult, ModelSpec, RelayEvent, RelayTrace, StreamEvent

FIXTURE = Path(__file__).parent / "fixtures" / "explore" / "default_relay_expectation.json"
CUES = ExploreConfig().cues
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "cost": 0.002}
WWS = "Where we stand: answer A, confidence low, weakest assumption W."
MODELS_WITH_SOL = [
    ModelSpec(key=VS_MENU_MODEL, provider_id=f"openai/{VS_MENU_MODEL}", supports_stop=False, supports_seed=True,
              max_tokens_slack=100),
    *MODELS[:2],
]


# --------------------------------------------------------------------------------------
# Fakes: a config with the preset on, an analyst that follows the template and reacts to the tail
# --------------------------------------------------------------------------------------


def explore_cfg(*, seed: int | None = 7, lens: dict[str, Any] | None = None, relay: dict[str, Any] | None = None,
                models: list[ModelSpec] | None = None, **explore: Any) -> AppConfig:
    lens_cfg = LensConfig(**(lens or {}))
    ex = ExploreConfig(enabled=True, lens=lens_cfg, **explore)
    return AppConfig(models=list(models or MODELS), relay=RelayConfig(segment_tokens=64, **(relay or {})),
                     router=RouterConfig(seed=seed), explore=ex, today="2026-11-24")


def make_explore_relay(script, cfg: AppConfig | None = None, *, router: FakeRouter | None = None,
                       backend: FakeBackend | None = None, **kw: Any) -> tuple[Relay, FakeBackend, FakeRouter]:
    cfg = cfg or explore_cfg(**kw)
    backend = backend or FakeBackend(script)
    router = router or FakeRouter(cfg.model_keys())
    return Relay(cfg, backend, router, retry_base_delay=0.0), backend, router


_WORDS = ["steady", "rising", "measured", "uneven", "confirmed", "partial", "disputed", "late", "early", "flat",
          "stable", "mixed", "sparse", "dense", "brief", "long"]


def plain_note(k: int, *, alternative: bool = True, wws: bool = True) -> str:
    """A main-line note with the whole checklist; wording varies with ``k`` so consecutive notes do not loop."""
    w = _WORDS[k % len(_WORDS)]
    parts = [f"Claim: the {w} rate of item {k} holds.", f"Evidence: the {w} record {k} shows it (source S{k})."]
    if alternative:
        parts.append(f"Alternative set aside: the {w} route R{k}, set aside because Q{k} fails.")
    parts.append(f"Open question: what settles the {w} item {k}?")
    if wws:
        parts.append(WWS)
    return " ".join(parts)


def lens_field_of(tail: str) -> str:
    return re.search(r"the field is ([^.]+)\.", tail).group(1)


def cue_of(tail: str) -> str:
    return re.search(r"Begin this note with the words '([^']+)'", tail).group(1)


def analyst(*, conclude_at: int | None = None, lens_prefix: bool = True, verdict: str | None = "kept",
            answer: str = "The answer is 42. Basis: items 0-5. Not resolved: nothing.", note=plain_note):
    """A writer that follows the labelled template and reacts to the tail: a lens note for a lens directive, a check
    with a verdict after a lens, the cue first in a forced continuation, and a conclusion once the moderator accepts
    one. With ``conclude_at`` it instead concludes in every think call from that one on, whatever the tail says
    (premature while the gate is closed, also inside a forced continuation), and never before.
    """

    def script(model: str, req, n: int) -> list[Any]:
        if req.phase == "answer":
            return text_events(answer, usage=USAGE)
        tail = req.tail or ""
        k = len(req.segments)
        concluding = ("The record is complete" in tail) if conclude_at is None else k >= conclude_at
        if "Cross-field lens" in tail:
            field = lens_field_of(tail)
            prefix = f"Lens ({field}): " if lens_prefix else ""
            body = (f"{prefix}the concept of feedback gain from {field} resembles item {k}: our rate plays the role "
                    f"of the gain. Inference: as a working analogy this suggests testing item {k} twice. Returning "
                    f"to the problem, item {k} stands. Where we stand: answer A{k}, confidence medium, weakest "
                    "assumption W.")
        elif "applied a cross-field lens" in tail:
            line = f"Lens {verdict}: the gain mapping {'holds' if verdict == 'kept' else 'breaks on the sink'}. " \
                if verdict else "The comparison is unclear. "
            body = line + f"Next: use item {k} concretely. " + WWS
        elif "set aside the conclusion" in tail:
            body = f"{cue_of(tail)} the {_WORDS[k % len(_WORDS)]} item {k} still needs a source. {note(k)}"
        else:
            body = note(k)
        if concluding and "Cross-field lens" not in tail and "applied a cross-field lens" not in tail:
            body += " FINAL ANSWER: 42"
        return text_events(body + SENTINEL, usage=USAGE)

    return script


def kinds_of(events: list[RelayEvent]) -> list[str]:
    return [ev.data["explore"]["kind"] for ev in committed(events)]


def starts(events: list[RelayEvent]) -> list[RelayEvent]:
    return [ev for ev in events if ev.type == "segment_start"]


def think_requests(backend: FakeBackend) -> list[Any]:
    return [req for _, req in backend.calls if req.phase == "think"]


# --------------------------------------------------------------------------------------
# The default stays byte-identical (preset off)
# --------------------------------------------------------------------------------------

TIME_KEYS = {"ttft_s", "duration_s", "started_at", "ended_at", "latency", "total_s", "first_delta_s"}


def scrub(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: ("<time>" if k in TIME_KEYS else scrub(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    return obj


def req_dict(model: str, req) -> dict:
    return {"model": model, **scrub(dataclasses.asdict(req))}


def refusing_script(model, req, n):
    if model == "a":
        return [StreamEvent(kind="done", finish_reason="content_filter", error="refused: policy")]
    if req.phase == "answer":
        return text_events("Answer by someone else.")
    return text_events(("FINAL ANSWER: ok" if len(req.segments) >= 2 else para(len(req.segments))) + SENTINEL)


async def test_default_relay_is_byte_identical_to_the_recorded_expectation():
    """Events, requests and router calls of three default runs equal the expectation recorded before the preset
    existed (tests/fixtures/explore/default_relay_expectation.json; time fields scrubbed)."""
    expected = json.loads(FIXTURE.read_text("utf-8"))
    got: dict[str, Any] = {}

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: avail[n % len(avail)])
    relay, backend, router = make_relay(finishing_script(), router=router)
    events = await collect(relay)
    got["finishing_round_robin"] = {"events": [scrub(ev.to_dict()) for ev in events],
                                    "requests": [req_dict(m, r) for m, r in backend.calls],
                                    "decide_calls": router.decide_calls}

    router = FakeRouter([m.key for m in MODELS], pick=lambda avail, cur, n: "a" if "a" in avail else avail[0])
    relay, backend, router = make_relay(refusing_script, router=router, disable_after_refusals=2)
    events = await collect(relay)
    got["refusal_reroute"] = {"events": [scrub(ev.to_dict()) for ev in events],
                              "requests": [req_dict(m, r) for m, r in backend.calls],
                              "decide_calls": router.decide_calls}

    router = FakeRouter([m.key for m in MODELS], r=0.9, draft_scores={"a": 1.0, "b": 5.0, "c": 2.0})
    relay, backend, router = make_relay(drafting_script(0.0), router=router, strategy="speculative")
    events = await collect(relay)
    got["speculative"] = {"events": [scrub(ev.to_dict()) for ev in events],
                          "requests": sorted((req_dict(m, r) for m, r in backend.calls),
                                             key=lambda d: (d["phase"], d["model"])),
                          "decide_calls": router.decide_calls}

    assert json.loads(json.dumps(got, sort_keys=True)) == expected
    # and the trace stays a plain RelayTrace without an "explore" key
    trace = await make_relay(finishing_script())[0].run_to_trace("Who is the chancellor right now?")
    assert type(trace) is RelayTrace and "explore" not in trace.to_dict()


async def test_default_config_leaves_the_preset_off():
    from cutoffmix.config import load_config

    assert load_config().explore.enabled is False and load_config(preset="explore").explore.enabled is True


# --------------------------------------------------------------------------------------
# Requests under the preset: system override, tail, seed
# --------------------------------------------------------------------------------------


async def test_requests_carry_the_explore_prompts_tail_and_seed():
    cfg = explore_cfg(seed=11, lens={"p": 0.0, "min": 0}, min_segments=2)
    relay, backend, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok"
    reqs = think_requests(backend)
    system = relay_system_prompt_explore(64, "2026-11-24", SENTINEL, "FINAL ANSWER:")
    assert all(r.system_override == system for r in reqs)
    assert [r.metadata for r in reqs] == [
        {"segment": i, "phase": "think", "segment_index": i, "seed": 11, "kind": "main", "light": False}
        for i in range(3)]
    # note 1: the whole checklist is missing; note 2: one more main-line note; note 3: the record is complete
    missing_all = ["two more main-line notes", "an 'Evidence:' item that names its source",
                   "an 'Alternative set aside:' item", "an 'Open question:' item",
                   "a 'Where we stand:' line at the end of this note"]
    assert reqs[0].tail == relay_tail(1, record_status_text(missing_all, False))
    assert reqs[1].tail == relay_tail(2, record_status_text(["one more main-line note"], False))
    assert reqs[2].tail == relay_tail(3, record_status_text([], True))
    assert "do not write 'FINAL ANSWER:'" in reqs[0].tail and "The record is complete" in reqs[2].tail
    answer_req = backend.calls[-1][1]
    assert answer_req.phase == "answer" and answer_req.tail is None
    assert answer_req.system_override == answer_system_prompt_explore("2026-11-24", answer_basis=True)
    assert answer_req.metadata == {"segment": 3, "phase": "answer", "segment_index": 3, "seed": 11}
    assert events[-1].data["answer"].startswith("The answer is 42.")
    assert events[-1].data["explore"]["seed"] == 11 and events[-1].data["explore"]["seed_source"] == "run"


async def test_answer_basis_off_and_explore_seed_override_and_random_seed():
    cfg = explore_cfg(seed=3, lens={"p": 0.0, "min": 0}, min_segments=1, answer_basis=False)
    cfg = dataclasses.replace(cfg, explore=dataclasses.replace(cfg.explore, seed=99))  # explore.seed wins
    relay, backend, _ = make_explore_relay(analyst(), cfg)
    done = (await collect(relay))[-1].data
    assert backend.calls[-1][1].system_override == answer_system_prompt_explore("2026-11-24", answer_basis=False)
    assert "Basis:" not in backend.calls[-1][1].system_override
    assert done["explore"]["seed"] == 99 and done["explore"]["seed_source"] == "explore"
    assert all(req.metadata["seed"] == 99 for _, req in backend.calls)
    # neither seed set: a random 31-bit seed, logged
    cfg = explore_cfg(seed=None, lens={"p": 0.0, "min": 0}, min_segments=1)
    relay, backend, _ = make_explore_relay(analyst(), cfg)
    done = (await collect(relay))[-1].data
    seed = done["explore"]["seed"]
    assert isinstance(seed, int) and 0 <= seed < 2 ** 31 and done["explore"]["seed_source"] == "random"
    assert all(req.metadata["seed"] == seed for _, req in backend.calls)


# --------------------------------------------------------------------------------------
# Lens schedule
# --------------------------------------------------------------------------------------


async def test_lens_never_in_the_first_two_segments_nor_after_the_gate_opens():
    # p = 1: a lens whenever eligible (up to 5). Lens and check notes do not count as main line, so with cooldown 2
    # the gate opens after note 4 (main-line notes 0, 1, 4); the analyst only concludes from think call 10 on, so
    # segments 5-9 are written with the gate open and four lenses still unused: none may be drawn.
    cfg = explore_cfg(lens={"p": 1.0, "max": 5, "min": 0, "cooldown": 2}, min_segments=3)
    relay, backend, router = make_explore_relay(analyst(conclude_at=10), cfg)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok"
    kinds = kinds_of(events)
    assert kinds == ["main", "main", "lens", "check", "main", "main", "main", "main", "main", "main", "main"]
    assert committed(events)[-1].data["final"] is True  # the conclusion at index 10 was accepted
    tails = [req.tail for req in think_requests(backend)]
    assert all("The record is complete" in t for t in tails[5:]) and not any("Cross-field" in t for t in tails[5:])
    assert all("The record is complete" not in t for t in tails[:5])
    ex = events[-1].data["explore"]
    assert ex["lens"]["offered"] == 1 and ex["gate_open"] is True and ex["premature_finals"] == []
    # without a cooldown the lenses are used up before the gate can open (main-line notes 0, 1, 6)
    cfg = explore_cfg(lens={"p": 1.0, "max": 2, "min": 0, "cooldown": 0}, min_segments=3)
    events = await collect(make_explore_relay(analyst(conclude_at=10), cfg)[0])
    assert kinds_of(events) == ["main", "main", "lens", "check", "lens", "check", "main", "main", "main", "main",
                                "main"]


async def test_exactly_one_check_after_each_lens_written_by_another_model():
    cfg = explore_cfg(lens={"p": 1.0, "max": 2, "min": 0, "cooldown": 0}, min_segments=3)
    relay, backend, router = make_explore_relay(analyst(), cfg)  # FakeRouter picks avail[0]
    events = await collect(relay)
    ends = committed(events)
    kinds = kinds_of(events)
    assert kinds == ["main", "main", "lens", "check", "lens", "check", "main", "main"]
    for i, kind in enumerate(kinds):
        if kind == "lens":
            assert kinds[i + 1] == "check" and ends[i + 1].model != ends[i].model
            assert ends[i].model == "a" and ends[i + 1].model == "b"  # a (first available) wrote the lens
            # the check's routing decision never saw the lens writer; the lens decision kept the EMA untouched
            assert router.decide_calls[i + 1]["available"] == ["b", "c"]
            assert router.decide_calls[i]["update"] is False and router.decide_calls[i + 1]["update"] is True
            # the lens note is not in any later routing window, the check note is
            lens_text_, check_text_ = ends[i].text, ends[i + 1].text
            assert all(lens_text_ not in c["window"] for c in router.decide_calls[i + 1:])
            assert check_text_ in router.decide_calls[i + 2]["window"]
            # tails: the lens directive names the field and its concepts, the check names the same field
            lens_req, check_req = think_requests(backend)[i], think_requests(backend)[i + 1]
            field = lens_field_of(lens_req.tail)
            assert ends[i].data["explore"]["field"] == field and ends[i].data["explore"]["tier"] in ("near", "mid",
                                                                                                     "far")
            assert check_req.tail.endswith(check_text(field)) and ends[i + 1].data["explore"]["field"] == field
            assert ends[i + 1].data["explore"]["lens_verdict"] == "kept"
            assert ends[i].data["explore"]["lens_prefix_ok"] is True and ends[i].text.startswith(f"Lens ({field}):")
    assert kinds.count("check") == kinds.count("lens") == 2
    # the two lenses used different fields (drawn without replacement) and the tier fits the router's topic
    fields = events[-1].data["explore"]["lens"]["fields"]
    assert len(set(fields)) == 2
    for f, tier in zip(fields, events[-1].data["explore"]["lens"]["tiers"]):
        assert tier_of(field_by_name(f).home_topic, "government") == tier  # FakeRouter's argmax topic
    # the answer decision's window has no lens note either
    assert all(f"Lens ({f}):" not in router.decide_calls[-1]["window"] for f in fields)


async def test_cooldown_keeps_two_segments_between_a_check_and_the_next_lens():
    cfg = explore_cfg(lens={"p": 1.0, "max": 5, "min": 0, "cooldown": 2}, min_segments=8)
    relay, _, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    kinds = kinds_of(events)
    assert kinds == ["main", "main", "lens", "check", "main", "main", "lens", "check", "main", "main", "lens",
                     "check", "main", "main", "main"]
    assert events[-1].data["explore"]["main_line_segments"] == 9  # 8 notes opened the gate, then the conclusion


async def test_min_lens_is_forced_at_the_last_eligible_slot():
    # p = 0: never drawn by chance; min = 1 forces it when the main line is one note short of the gate
    cfg = explore_cfg(lens={"p": 0.0, "min": 1, "max": 2}, min_segments=4)
    relay, _, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    assert kinds_of(events) == ["main", "main", "main", "lens", "check", "main", "main"]
    assert events[-1].data["explore"]["lens"]["offered"] == 1 and events[-1].data["explore"]["lens"]["kept"] == 1
    # min = 0: no lens at all
    cfg = explore_cfg(lens={"p": 0.0, "min": 0, "max": 2}, min_segments=4)
    events = await collect(make_explore_relay(analyst(), cfg)[0])
    assert kinds_of(events) == ["main"] * 5


async def test_lens_draw_is_deterministic_for_the_seed():
    runs = []
    for _ in range(2):
        cfg = explore_cfg(seed=21, lens={"p": 1.0, "max": 2, "min": 0, "cooldown": 0}, min_segments=3)
        done = (await collect(make_explore_relay(analyst(), cfg)[0]))[-1].data
        runs.append((done["explore"]["kinds"], done["explore"]["lens"]["fields"], done["explore"]["lens"]["tiers"]))
    assert runs[0] == runs[1]
    # the draw depends on the seed: over a handful of seeds the first field is not always the same one
    firsts = set()
    for seed in range(22, 30):
        cfg = explore_cfg(seed=seed, lens={"p": 1.0, "max": 1, "min": 0, "cooldown": 0}, min_segments=3)
        firsts.add((await collect(make_explore_relay(analyst(), cfg)[0]))[-1].data["explore"]["lens"]["fields"][0])
    assert len(firsts) > 1


async def test_ignored_lens_prefix_and_missing_verdict_are_logged_not_retried():
    cfg = explore_cfg(lens={"p": 1.0, "max": 1, "min": 0}, min_segments=3)
    relay, backend, _ = make_explore_relay(analyst(lens_prefix=False, verdict=None), cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events)[2:4] == ["lens", "check"]
    assert ends[2].data["explore"]["lens_prefix_ok"] is False and ends[3].data["explore"]["lens_verdict"] is None
    lens = events[-1].data["explore"]["lens"]
    assert lens == {"offered": 1, "written": 0, "ignored": 1, "kept": 0, "dropped": 0, "unverified": 1,
                    "fields": lens["fields"], "tiers": lens["tiers"], "keep_rate": 0.0}
    assert len([m for m, r in backend.calls if r.phase == "think"]) == len(ends)  # one call per segment, no retry


# --------------------------------------------------------------------------------------
# Record gate, premature conclusions, forced continuations
# --------------------------------------------------------------------------------------


async def test_premature_final_strips_the_answer_and_forces_at_most_three_continuations_with_rotating_cues():
    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=6)
    relay, backend, router = make_explore_relay(analyst(conclude_at=0), cfg)  # concludes in every note
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main", "continue", "continue", "continue"]
    assert [e.data["explore"]["cue"] for e in ends] == [None, *CUES[:3]]
    assert [e.data.get("premature_final") for e in ends] == [True, True, True, None]
    assert [e.data["final"] for e in ends] == [False, False, False, True]  # the fourth conclusion is accepted
    assert all("FINAL ANSWER" not in e.text and "42" not in e.text for e in ends)
    assert all("FINAL" not in ev.text for ev in events if ev.type == "delta")
    # the continuation begins with the cue and was written by another model than the one that concluded
    for i in (1, 2, 3):
        assert ends[i].text.startswith(CUES[i - 1].strip()) and ends[i].model != ends[i - 1].model
        assert ends[i - 1].model not in router.decide_calls[i]["available"]
        tail = think_requests(backend)[i].tail
        assert tail.endswith(continue_text(f"the record still lacks {['five', 'four', 'three'][i - 1]} more "
                                           "main-line notes", CUES[i - 1]))
        assert "do not write 'FINAL ANSWER:'" in tail  # the gate text still states the checklist
    ex = events[-1].data["explore"]
    assert ex["forced_continuations"] == 3 and ex["first_final_index"] == 0 and ex["loop_stop"] is False
    assert [(r["index"], r["forced"], r["cue"]) for r in ex["premature_finals"]] == [
        (0, True, CUES[0]), (1, True, CUES[1]), (2, True, CUES[2])]
    assert ex["premature_finals"][0]["missing"] == ["five more main-line notes"]
    assert ex["gate_open"] is False and ex["gate_missing_at_final"] == ["two more main-line notes"]
    assert events[-1].data["answer"].startswith("The answer is 42.")
    assert [s["explore"]["kind"] for s in events[-1].data["segments"] if s["phase"] == "think"] == kinds_of(events)


async def test_loop_detection_stops_forcing_and_accepts_the_conclusion():
    same = "Claim: the rate of the item holds. Evidence: the record shows it (source S). " \
           "Alternative set aside: the route, set aside because it fails. Open question: what settles it? " + WWS

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Looped answer.")
        tail = req.tail or ""
        prefix = cue_of(tail) if "set aside the conclusion" in tail else ""
        return text_events(f"{prefix}{same} FINAL ANSWER: 42" + SENTINEL)

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=6, loop_overlap_jaccard=0.5)
    relay, _, _ = make_explore_relay(script, cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main", "continue"]
    assert ends[0].data.get("premature_final") is True and ends[1].data["final"] is True
    ex = events[-1].data["explore"]
    assert ex["loop_stop"] is True and ex["forced_continuations"] == 1
    assert [(r["forced"], r["loop"]) for r in ex["premature_finals"]] == [(True, False), (False, True)]
    assert ex["premature_finals"][1]["overlap_last2"] > 0.5 and ex["ngram_overlap_last2"] > 0.5
    assert events[-1].data["answer"] == "Looped answer."


async def test_gate_opens_only_when_the_checklist_passes():
    # notes without 'Alternative set aside:' (and notes 2 and 3 without 'Where we stand:'); a conclusion at call 6
    # is premature; the continuation adds the missing item, then the moderator accepts the next conclusion
    def note(k: int) -> str:
        return plain_note(k, alternative=k >= 7, wws=k not in (2, 3))

    def script(model, req, n):  # concludes once in think call 6, otherwise as the tail allows
        return analyst(conclude_at=6 if len(req.segments) == 6 else None, note=note)(model, req, n)

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=6)
    relay, backend, _ = make_explore_relay(script, cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main"] * 7 + ["continue", "main"]
    tails = [req.tail for req in think_requests(backend)]
    wws_item = "a 'Where we stand:' line at the end of this note"
    assert wws_item in tails[0] and all(wws_item not in t for t in tails[1:3])
    # note 2 had none, but note 1 did (window of two notes): the gate does not require the line yet, but the tail
    # reminds of it because the previous note lacks it (listed once, with the gate's own items)
    assert tails[3] == relay_tail(4, record_status_text(["three more main-line notes",
                                                         "an 'Alternative set aside:' item"], False, remind_wws=True))
    assert tails[3].count("Where we stand") == 1
    assert wws_item in tails[4] and tails[4].count(wws_item) == 1  # neither note 2 nor note 3 had one: required
    assert wws_item not in tails[5]  # note 4 has one again
    assert all("an 'Alternative set aside:' item" in t for t in tails[:8])
    assert "more main-line note" not in tails[6]  # six main-line notes are in: only the alternative is missing
    assert tails[7] == relay_tail(8, record_status_text(["an 'Alternative set aside:' item"], False),
                                  continue_text("the record still lacks an 'Alternative set aside:' item", CUES[0]))
    assert tails[8] == relay_tail(9, record_status_text([], True))
    assert ends[6].data.get("premature_final") is True and ends[8].data["final"] is True
    assert ends[7].data["explore"]["kind"] == "continue" and ends[7].text.startswith("Alternatively, the")
    ex = events[-1].data["explore"]
    assert ex["gate_open"] is True and ex["gate_missing_at_final"] is None and len(ex["premature_finals"]) == 1
    assert ex["premature_finals"][0]["missing"] == ["an 'Alternative set aside:' item"]
    assert ex["labels"]["counts"]["Alternative set aside"] == 2 and ex["labels"]["coverage"]["where_we_stand"] < 1
    # the two notes without the line were not cut: skipped by the writer; the line was never the only missing item
    assert ex["wws"] == {"missing": 2, "cut": 0, "waived": 0, "waived_at": []}
    assert [e.data["explore"]["cut"] for e in ends] == [False] * 9


async def test_premature_final_with_an_empty_note_is_dropped_and_the_retry_is_the_continuation():
    def script(model, req, n):
        if req.phase == "answer":
            return text_events("A.")
        tail = req.tail or ""
        if model == "a" and not req.segments:
            return text_events("FINAL ANSWER: 42" + SENTINEL)  # nothing but a conclusion
        if "set aside the conclusion" in tail:
            return text_events(cue_of(tail) + plain_note(len(req.segments)) + SENTINEL)
        if "The record is complete" in tail:
            return text_events("FINAL ANSWER: 42" + SENTINEL)
        return text_events(plain_note(len(req.segments)) + SENTINEL)

    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=2)
    relay, _, router = make_explore_relay(script, cfg)
    events = await collect(relay)
    drops = [ev for ev in events if ev.type == "segment_end" and (ev.data or {}).get("dropped")]
    assert [(d.model, d.data["reason"], d.data.get("premature_final")) for d in drops] == [("a", "empty", True)]
    ends = committed(events)
    assert ends[0].segment == 0 and ends[0].model == "b" and ends[0].data["explore"]["kind"] == "continue"
    assert ends[0].text.startswith("Alternatively,") and router.decide_calls[1]["available"] == ["b", "c"]
    ex = events[-1].data["explore"]
    assert ex["forced_continuations"] == 1 and ex["premature_finals"][0]["missing"][0] == "two more main-line notes"
    assert events[-1].data["dropped"][0]["premature_final"] is True
    assert events[-1].data["status"] == "ok"


async def test_zero_forced_continuations_accepts_the_first_conclusion_and_logs_the_gap():
    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=6, max_forced_continuations=0)
    relay, backend, _ = make_explore_relay(analyst(conclude_at=1), cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main", "main"] and ends[1].data["final"] is True
    assert "do not write 'FINAL ANSWER:'" in think_requests(backend)[1].tail
    ex = events[-1].data["explore"]
    assert ex["premature_finals"] == [] and ex["gate_missing_at_final"] == ["four more main-line notes"]


async def test_no_lens_while_a_continuation_is_forced_and_the_check_absorbs_a_pending_forcing():
    # a lens whenever eligible; the analyst concludes inside the lens note (call 2): the check that follows is routed
    # away from the lens writer (who is also the concluding model) and consumes the forcing
    cfg = explore_cfg(lens={"p": 1.0, "max": 1, "min": 0, "cooldown": 0}, min_segments=3)

    def note(k):
        return plain_note(k)

    base = analyst(note=note)

    def script(model, req, n):
        out = base(model, req, n)
        if req.phase == "think" and "Cross-field lens" in (req.tail or ""):
            text = "".join(ev.text for ev in out if ev.kind == "text").replace(SENTINEL, "")
            return text_events(text + " FINAL ANSWER: 42" + SENTINEL, usage=USAGE)
        return out

    relay, backend, router = make_explore_relay(script, cfg)
    events = await collect(relay)
    ends = committed(events)
    assert kinds_of(events) == ["main", "main", "lens", "check", "main", "main"]
    assert ends[2].data.get("premature_final") is True and ends[3].model != ends[2].model
    assert ends[2].model not in router.decide_calls[3]["available"]
    assert think_requests(backend)[3].tail.endswith(check_text(ends[2].data["explore"]["field"]))
    ex = events[-1].data["explore"]
    assert ex["forced_continuations"] == 0 and len(ex["premature_finals"]) == 1
    assert ex["premature_finals"][0]["missing"][-1] == \
        "the test of the previous cross-field lens ('Lens kept:' or 'Lens dropped:')"


# --------------------------------------------------------------------------------------
# Metrics and trace shape
# --------------------------------------------------------------------------------------


async def test_metrics_on_a_known_script():
    cfg = explore_cfg(seed=5, lens={"p": 1.0, "max": 1, "min": 0, "cooldown": 0}, min_segments=3)
    relay, backend, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    done = events[-1].data
    ex = done["explore"]
    think = [s for s in done["segments"] if s["phase"] == "think"]
    assert ex["kinds"] == ["main", "main", "lens", "check", "main", "main"]
    assert (ex["think_segments"], ex["main_line_segments"], ex["lens_segments"], ex["check_segments"],
            ex["continue_segments"]) == (6, 4, 1, 1, 0)
    assert ex["topics"] == ["government"] * 6 and ex["topics_visited"] == {"all": 1, "main_line": 1}
    assert ex["topic_entropy_bits"] == {"mean_posterior": 0.8813, "argmax_histogram": 0.0}
    assert ex["topic_switches"] == 0
    field = ex["lens"]["fields"][0]
    assert ex["lens"] == {"offered": 1, "written": 1, "ignored": 0, "kept": 1, "dropped": 0, "unverified": 0,
                          "fields": [field], "tiers": [ex["lens"]["tiers"][0]], "keep_rate": 1.0}
    expected_counts: dict[str, int] = {}
    for s in think:
        for label, n in parse_labels(s["text"])["counts"].items():
            expected_counts[label] = expected_counts.get(label, 0) + n
    assert ex["labels"]["counts"] == expected_counts
    assert expected_counts["Lens"] == 1 and expected_counts["Lens kept"] == 1 and expected_counts["Claim"] == 4
    assert expected_counts["Where we stand"] == 6 and expected_counts["Evidence"] == 4
    assert ex["labels"]["coverage"] == {"two_or_more_items": 1.0, "where_we_stand": 1.0}
    assert len(ex["wws_lines"]) == 6 and ex["wws_lines"][2].startswith("answer A2,")
    assert ex["answer_flips"] == 2  # A -> A2 (lens note) -> A
    assert ex["premature_finals"] == [] and ex["forced_continuations"] == 0 and ex["first_final_index"] == 5
    assert ex["gate_open"] is True and ex["gate_missing_at_final"] is None and ex["loop_stop"] is False
    assert 0.0 <= ex["ngram_overlap_last2"] < 0.5
    assert ex["notes_chars"] == sum(len(s["text"]) for s in think) and ex["notes_tokens_est"] == ex["notes_chars"] // 4
    assert ex["garble"] == 0
    assert ex["vs_menu"] == {"mode": "off", "calls": 0, "parsed": 0, "chosen_p": [], "menus": [], "errors": [],
                             "skipped": None}
    assert ex["seed"] == 5 and ex["seed_source"] == "run"
    assert ex["preset"] is None  # a hand-built AppConfig: no packaged preset was loaded
    assert ex["config"]["min_segments"] == 3 and ex["config"]["lens"]["p"] == 1.0
    assert ex["config"]["cues"] == list(CUES) and ex["config"]["require_labels"] == list(cfg.explore.require_labels)
    # the done payload is JSON-safe as it is
    assert json.loads(json.dumps(to_jsonable(done)))["explore"] == to_jsonable(ex)


async def test_metrics_echo_the_preset_of_the_config():
    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=2)
    cfg.preset = "explore"  # what load_config(preset="explore") sets
    relay, _, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    assert events[-1].data["explore"]["preset"] == "explore"
    assert json.loads(json.dumps(to_jsonable(events[-1].data)))["explore"]["preset"] == "explore"


async def test_trace_and_events_carry_the_per_segment_flags():
    cfg = explore_cfg(lens={"p": 1.0, "max": 1, "min": 0, "cooldown": 0}, min_segments=3)
    relay, _, _ = make_explore_relay(analyst(), cfg)
    trace = await relay.run_to_trace("Who leads the lab right now?", run_id="r")
    assert isinstance(trace, ExploreTrace) and isinstance(trace, RelayTrace)
    d = trace.to_dict()
    assert d["explore"] is trace.explore and d["explore"]["kinds"] == ["main", "main", "lens", "check", "main", "main"]
    segs = [s for s in trace.segments if s.phase == "think"]
    lens, check = segs[2].explore, segs[3].explore
    assert lens.kind == "lens" and lens.tier in ("near", "mid", "far") and lens.lens_prefix_ok is True
    assert lens.labels["Lens"] == 1 and lens.labels["Inference"] == 1 and lens.has_wws is True
    assert check.kind == "check" and check.field == lens.field and check.lens_verdict == "kept"
    assert check.tier is None and check.cue is None
    assert trace.segments[-1].phase == "answer" and trace.segments[-1].explore is None
    assert d["segments"][2]["explore"]["kind"] == "lens" and d["segments"][-1]["explore"] is None
    # segment_start badges
    relay, _, _ = make_explore_relay(analyst(), cfg)
    events = await collect(relay)
    badges = [ev.data["explore"] for ev in starts(events)]
    assert badges[0] == {"kind": "main", "field": None, "tier": None, "cue": None}
    assert badges[2]["kind"] == "lens" and badges[2]["field"] == lens.field and badges[2]["tier"] == lens.tier
    assert badges[3] == {"kind": "check", "field": lens.field, "tier": None, "cue": None}
    assert all(ev.data["strategy"] == "sequential" for ev in starts(events))
    end = committed(events)[2]
    assert end.data["explore"] == dataclasses.asdict(lens) and "premature_final" not in end.data


async def test_hard_cut_and_max_segments_still_apply_and_log_the_gate_gap():
    cfg = explore_cfg(lens={"p": 0.0, "min": 0}, min_segments=6, relay={"max_segments": 3})

    def script(model, req, n):
        if req.phase == "answer":
            return text_events("Capped.")
        return text_events(plain_note(len(req.segments), alternative=False) + SENTINEL)

    relay, backend, _ = make_explore_relay(script, cfg)
    events = await collect(relay)
    assert kinds_of(events) == ["main"] * 3 and events[-1].data["answer"] == "Capped."
    ex = events[-1].data["explore"]
    assert ex["gate_missing_at_final"] == ["three more main-line notes", "an 'Alternative set aside:' item"]
    assert ex["first_final_index"] is None


# --------------------------------------------------------------------------------------
# Next-move menu side call (vs_menu)
# --------------------------------------------------------------------------------------

MENU = ("1. Compare the two records directly | p=0.45\n2. Treat the lab as a predator-prey system | p=0.05\n"
        "3. Ask what a court would accept as proof | p=0.07\n4. Model the handover as a phase transition | p=0.04\n"
        "5. Summarise the notes | p=0.30\n")


class MenuBackend(FakeBackend):
    def __init__(self, script, menu: str | Exception = MENU) -> None:
        super().__init__(script)
        self.menu = menu
        self.chats: list[tuple[str, Any]] = []

    async def chat(self, model, req):
        self.chats.append((model.key, req))
        if isinstance(self.menu, Exception):
            raise self.menu
        return ChatResult(text=self.menu, usage={"prompt_tokens": 40, "completion_tokens": 60, "cost": 0.0005})


def stagnant_note(k: int) -> str:
    """Notes whose numbers and capitalised tokens never change (the stagnation proxy), worded differently."""
    w = _WORDS[k % len(_WORDS)]
    return (f"Claim: the {w} rate holds. Evidence: the {w} record shows it (source S). Alternative set aside: the "
            f"{w} route, set aside because it fails. Open question: what settles the {w} item? " + WWS)


async def test_vs_menu_on_stagnation_adds_the_chosen_direction_to_the_tail():
    cfg = explore_cfg(seed=4, lens={"p": 0.0, "p_stagnation": 0.0, "min": 0}, min_segments=6, vs_menu="stagnation",
                      models=MODELS_WITH_SOL)
    backend = MenuBackend(analyst(note=stagnant_note))
    relay, backend, router = make_explore_relay(None, cfg, backend=backend)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok"
    # stagnation needs four committed segments with the same argmax topic and no new salient token: indexes 4
    # and 5; the gate opens after note 5 (six main-line notes), and no menu runs with the gate open
    assert [m for m, _ in backend.chats] == [VS_MENU_MODEL, VS_MENU_MODEL]
    chat_req = backend.chats[0][1]
    assert chat_req.max_tokens == VS_MENU_MAX_TOKENS
    assert chat_req.metadata == {"seed": 4, "segment_index": 4, "purpose": "vs_menu"}
    assert chat_req.messages[0]["role"] == "system" and "PROBLEM:" in chat_req.messages[1]["content"]
    tails = [req.tail for req in think_requests(backend)]
    assert not any("Direction for this note" in t for t in tails[:4])
    chosen = {"Treat the lab as a predator-prey system", "Ask what a court would accept as proof",
              "Model the handover as a phase transition"}
    for t in tails[4:6]:
        direction = re.search(r"Direction for this note: ([^.]+)\.", t).group(1)
        assert direction in chosen and t.endswith(vs_direction_text(direction))
    ex = events[-1].data["explore"]["vs_menu"]
    assert ex["mode"] == "stagnation" and ex["calls"] == 2 and ex["parsed"] == 2
    assert all(p < 0.10 for p in ex["chosen_p"]) and [m["index"] for m in ex["menus"]] == [4, 5]
    assert ex["menus"][0]["chosen"] in chosen and ex["menus"][0]["p"] == ex["chosen_p"][0]
    assert ex["errors"] == [] and ex["skipped"] is None
    # side calls are billed and counted with the model's own calls (the concluding segment is closed at the
    # marker, so its usage never arrives: 7 think calls + 1 answer + 2 menus, 6 + 1 + 2 with a cost)
    assert events[-1].data["usage"][VS_MENU_MODEL]["calls"] == 7 + 1 + 2
    assert events[-1].data["usage"][VS_MENU_MODEL]["cost"] == pytest.approx(7 * USAGE["cost"] + 2 * 0.0005)
    assert kinds_of(events) == ["main"] * 7  # a direction does not change the kind


async def test_vs_menu_is_skipped_without_sol_and_survives_a_failing_side_call():
    cfg = explore_cfg(seed=4, lens={"p": 0.0, "p_stagnation": 0.0, "min": 0}, min_segments=6, vs_menu="stagnation")  # no Sol
    backend = MenuBackend(analyst(note=stagnant_note))
    events = await collect(make_explore_relay(None, cfg, backend=backend)[0])
    assert backend.chats == [] and events[-1].data["status"] == "ok"
    ex = events[-1].data["explore"]["vs_menu"]
    assert ex["calls"] == 0 and ex["skipped"] == f"{VS_MENU_MODEL} is not configured"

    cfg = explore_cfg(seed=4, lens={"p": 0.0, "p_stagnation": 0.0, "min": 0}, min_segments=6, vs_menu="stagnation",
                      models=MODELS_WITH_SOL)
    backend = MenuBackend(analyst(note=stagnant_note), menu=RuntimeError("menu down"))
    events = await collect(make_explore_relay(None, cfg, backend=backend)[0])
    assert events[-1].data["status"] == "ok" and len(backend.chats) == 2
    ex = events[-1].data["explore"]["vs_menu"]
    assert ex["calls"] == 2 and ex["parsed"] == 0 and [e["index"] for e in ex["errors"]] == [4, 5]
    assert ex["errors"][0]["error"] == "RuntimeError: menu down"
    assert not any("Direction for this note" in req.tail for req in think_requests(backend))

    # a menu with fewer than three parsable lines is discarded: a plain note, logged as a call that was not parsed
    backend = MenuBackend(analyst(note=stagnant_note), menu="1. only one | p=0.05\nnothing else")
    events = await collect(make_explore_relay(None, cfg, backend=backend)[0])
    ex = events[-1].data["explore"]["vs_menu"]
    assert ex["calls"] == 2 and ex["parsed"] == 0 and ex["menus"][0] == {"index": 4, "menu": [["only one", 0.05]],
                                                                          "chosen": None, "p": None}


async def test_vs_menu_every_lens_runs_at_eligible_slots_without_a_lens():
    cfg = explore_cfg(seed=4, lens={"p": 0.0, "min": 0}, min_segments=4, vs_menu="every_lens", models=MODELS_WITH_SOL)
    backend = MenuBackend(analyst())
    events = await collect(make_explore_relay(None, cfg, backend=backend)[0])
    # eligible from index 2 while the gate is closed: indexes 2 and 3 (the gate opens after note 4)
    assert [req.metadata["segment_index"] for _, req in backend.chats] == [2, 3]
    assert kinds_of(events) == ["main"] * 5


# --------------------------------------------------------------------------------------
# Speculative strategy under the preset
# --------------------------------------------------------------------------------------


async def test_speculative_drafts_follow_the_plan_and_strip_premature_conclusions():
    cfg = explore_cfg(lens={"p": 1.0, "max": 1, "min": 0, "cooldown": 0}, min_segments=3,
                      relay={"strategy": "speculative", "speculative_gate": 0.0})
    router = FakeRouter([m.key for m in MODELS], r=0.9, draft_scores={"a": 1.0, "b": 5.0, "c": 2.0})
    relay, backend, router = make_explore_relay(analyst(conclude_at=0), cfg, router=router)
    events = await collect(relay)
    assert events[-1].data["status"] == "ok"
    kinds = kinds_of(events)
    ends = committed(events)
    assert kinds[0] == "main" and ends[0].model == "b" and ends[0].data.get("premature_final") is True
    assert ends[0].decision == "speculative" and starts(events)[0].data["explore"]["kind"] == "main"
    assert kinds[1] == "continue" and ends[1].model != "b" and ends[1].text.startswith(CUES[0].strip())
    assert "b" not in starts(events)[1].data["available"]  # the concluding model drafted no continuation
    assert all("FINAL ANSWER" not in e.text for e in ends)
    assert all(req.tail is not None and req.system_override is not None for req in think_requests(backend))
    ex = events[-1].data["explore"]
    assert 1 <= ex["forced_continuations"] <= 3 and ex["premature_finals"][0]["model"] == "b"
    assert ex["forced_continuations"] == 3  # it concludes in every note: the limit is reached, then accepted
