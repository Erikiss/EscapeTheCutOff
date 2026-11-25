"""Tests for cutoffmix.backends.mock (and the backends package factory)."""

from __future__ import annotations

import re

import pytest

import hashlib

from cutoffmix.backends import MockBackend, MockModelTruth, OpenRouterBackend, default_mock_truths, make_backend
from cutoffmix.backends.base import BackendError
from cutoffmix.config import default_config, load_config
from cutoffmix.prompts import (
    answer_system_prompt_explore,
    check_text,
    clean_segment,
    continue_text,
    lens_text,
    missing_record_items,
    parse_labels,
    record_status_text,
    relay_system_prompt_explore,
    relay_tail,
    split_final,
    vs_direction_text,
)
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.timeutil import month_index, month_label
from cutoffmix.types import Backend, ChatRequest, ContinuationRequest, Event, ModelSpec, StreamEvent

CFG = default_config()
FABLE, SOL, NEON = (CFG.model(k) for k in ("claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"))
S = "<<END_STEP>>"
FINAL = "FINAL ANSWER:"


class FakeClassifier:
    """Deterministic one-topic classifier, so text expectations do not depend on the keyword lexicon."""

    topics = TOPIC_IDS
    rules = (("model", "ai_ml"), ("moon", "space_astronomy"), ("minister", "government"))

    def posterior(self, text: str) -> list[float]:
        low = text.lower()
        hot = next((topic for word, topic in self.rules if word in low), "sports")
        return [0.86 if t == hot else 0.01 for t in self.topics]


def mock(**kw) -> MockBackend:
    kw.setdefault("classifier", FakeClassifier())
    kw.setdefault("today", "2026-11-24")
    truths = kw.pop("truths", None) or default_mock_truths(latency_s=0)
    return MockBackend(truths, **kw)


async def stream(b: MockBackend, model: ModelSpec, question: str, segments: list[str] | None = None,
                 **kw) -> list[StreamEvent]:
    r = ContinuationRequest(question=question, segments=list(segments or []), **kw)
    return [e async for e in b.stream_continuation(model, r)]


def text_of(events: list[StreamEvent]) -> str:
    return "".join(e.text for e in events if e.kind == "text")


# --------------------------------------------------------------------------------------
# truths
# --------------------------------------------------------------------------------------


def test_default_truths_have_config_keys_and_reversals():
    truths = default_mock_truths()
    assert list(truths) == CFG.model_keys()
    best = {z: max(truths.values(), key=lambda t: t.tau_for(z)).key for z in TOPIC_IDS}
    assert best["ai_ml"] == best["computing_software"] == "claude-fable-5.2"
    assert best["science_research"] == best["space_astronomy"] == "gemini-4-neon"
    others = set(TOPIC_IDS) - {"ai_ml", "computing_software", "science_research", "space_astronomy"}
    assert {best[z] for z in others} == {"gpt-6.2-sol"}
    for z in TOPIC_IDS:  # every winner leads by at least one month, so routing differences are visible
        taus = sorted((t.tau_for(z) for t in truths.values()), reverse=True)
        assert taus[0] - taus[1] >= 1.0


def test_p_correct_curve():
    t = MockModelTruth(key="m", tau={"sports": 30.0}, default_tau=20.0, slope=0.5, plateau=0.9)
    assert t.tau_for("sports") == 30.0 and t.tau_for("elections") == 20.0
    assert t.p_correct("sports", 30.0) == pytest.approx(0.45)
    assert t.p_correct("sports", 10.0) == pytest.approx(0.9)
    assert t.p_correct("sports", 1e6, guess=0.25) == pytest.approx(0.25)  # no overflow far past the cutoff
    assert t.p_correct("sports", 29.0) > t.p_correct("sports", 31.0)


@pytest.mark.parametrize("field, value", [("slope", 0.0), ("plateau", 1.5), ("stale_rate", -0.1),
                                          ("refusal_rate", 2.0), ("latency_s", -1.0)])
def test_truth_validation(field, value):
    with pytest.raises(ValueError, match=field):
        MockModelTruth(key="m", tau={}, default_tau=20.0, **{field: value})


# --------------------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------------------


def test_mock_backend_satisfies_protocol():
    assert isinstance(mock(), Backend)


async def test_stream_is_deterministic_and_chunked():
    a = await stream(mock(seed=3), SOL, "Which language model is newest?", ["Note one."])
    b = await stream(mock(seed=3), SOL, "Which language model is newest?", ["Note one."])
    assert a == b
    assert all(0 < len(e.text) <= 8 for e in a if e.kind == "text")
    assert [e.kind for e in a].count("done") == 1 and a[-1].kind == "done"
    assert a[-1].usage["completion_tokens"] > 0


async def test_routing_relevant_text_is_topic_dependent():
    b = mock()
    q_ai, q_space = "Which language model did each lab release last?", "When is the next crewed Moon landing?"
    truths = default_mock_truths()
    for model in (FABLE, SOL, NEON):
        for question, topic in ((q_ai, "ai_ml"), (q_space, "space_astronomy")):
            text = text_of(await stream(b, model, question, today="2026-11-24"))
            assert f"known as of {month_label(truths[model.key].tau_for(topic))}" in text
    fable_ai = text_of(await stream(b, FABLE, q_ai))
    sol_ai = text_of(await stream(b, SOL, q_ai))
    assert "2026-08" in fable_ai and "2026-04" in sol_ai  # Fable knows AI news four months longer


async def test_a_question_about_two_topics_is_covered_topic_by_topic():
    class TwoTopics:
        topics = TOPIC_IDS

        def posterior(self, text: str) -> list[float]:
            hot = {"government", "ai_ml"} if "president" in text.lower() else {"sports"}
            return [0.45 if t in hot else 0.1 / (len(TOPIC_IDS) - len(hot)) for t in TOPIC_IDS]

    b = mock(classifier=TwoTopics(), segments_to_answer=6)
    q = "Who is the president, and which AI models are new?"
    assert b.question_topics(q) == ["government", "ai_ml"]
    notes: list[str] = []
    for _ in range(6):
        notes.append(clean_segment(text_of(await stream(b, FABLE, q, notes, sentinel=S)), "", sentinel=S,
                                   hard_cut_chars=10_000)[0])
    assert ["office" in n for n in notes] == [True] * 3 + [False] * 3  # three segments per topic
    assert all("AI model landscape" in n for n in notes[3:])
    _, answer = split_final(text_of(await stream(b, FABLE, q, notes, sentinel=S)), "FINAL ANSWER:")
    assert answer is not None and "who holds the office known as of 2026-03" in answer
    assert "AI model landscape known as of 2026-08" in answer  # Fable's own per-topic cutoffs


async def test_sentinel_and_finish_reason_follow_stop_support():
    b = mock()
    sol = await stream(b, SOL, "Who is the prime minister?", sentinel=S)
    assert text_of(sol).endswith(S) and sol[-1].finish_reason == "sentinel"
    fable = await stream(b, FABLE, "Who is the prime minister?", sentinel=S)
    assert S not in text_of(fable) and fable[-1].finish_reason == "stop"


async def test_mock_notes_survive_clean_segment():
    b = mock()
    segments: list[str] = []
    for i in range(5):
        model = (FABLE, SOL, NEON)[i % 3]
        raw = text_of(await stream(b, model, "Who is the prime minister of Japan?", segments))
        text, _ = clean_segment(raw, "".join(segments), sentinel=S, hard_cut_chars=600)
        assert text and S not in text and len(text) <= 600
        assert text not in segments  # the notes progress
        segments.append(text)


async def test_short_target_gives_short_notes():
    long = text_of(await stream(mock(), FABLE, "Who is the minister?", target_tokens=128))
    short = text_of(await stream(mock(), FABLE, "Who is the minister?", target_tokens=20))
    assert len(short) < len(long)


async def test_final_marker_after_segments_to_answer():
    b = mock(segments_to_answer=2)
    early = text_of(await stream(b, NEON, "Who is the minister?", ["a.", "b."][:1]))
    assert "FINAL ANSWER:" not in early
    final = text_of(await stream(b, NEON, "Who is the minister?", ["a.", "b."]))
    notes, answer = split_final(final, "FINAL ANSWER:")
    assert notes and answer and "known as of" in answer


async def test_answer_phase():
    events = await stream(mock(), SOL, "Who is the minister?", ["a.", "b."], phase="answer")
    text = text_of(events)
    assert text.startswith("Based on the shared notes") and S not in text and "FINAL ANSWER" not in text
    assert events[-1].finish_reason == "stop"


async def test_refusal_rate():
    truths = default_mock_truths(latency_s=0)
    truths["claude-fable-5.2"].refusal_rate = 1.0
    events = await stream(mock(truths=truths), FABLE, "Who is the minister?")
    assert [e.kind for e in events] == ["done"] and events[0].finish_reason == "content_filter"
    answer = await stream(mock(truths=truths), FABLE, "Who is the minister?", phase="answer")
    assert answer[-1].finish_reason == "stop"  # refusals are simulated for think segments only


async def test_unknown_model_raises():
    with pytest.raises(BackendError, match="no truth"):
        await stream(mock(), ModelSpec(key="other", provider_id="x/other"), "Q?")
    with pytest.raises(BackendError):
        await mock().chat(ModelSpec(key="other", provider_id="x/other"), ChatRequest(messages=[]))


async def test_classifier_topic_outside_taxonomy():
    class Custom:
        topics = ("quantum_cooking", "other")

        def posterior(self, text: str) -> list[float]:
            return [0.9, 0.1]

    text = text_of(await stream(mock(classifier=Custom()), SOL, "Q?"))
    assert "quantum cooking" in text and f"known as of {month_label(28.75)}" in text


async def test_default_keyword_classifier_is_loaded_lazily():
    b = MockBackend(default_mock_truths(latency_s=0), today="2026-11-24")
    text = text_of(await stream(b, SOL, "Which large language model did OpenAI release last?"))
    assert "known as of" in text


async def test_default_output_is_pinned():
    """The explore preset added a second note style; without ``system_override`` the output stays byte-identical
    (hashes recorded before the explore preset existed)."""
    b = mock(seed=3, segments_to_answer=3)
    q = "Who is the minister of finance?"
    notes: list[str] = []
    hashes = []
    for model in (FABLE, SOL, NEON, SOL):
        text = text_of(await stream(b, model, q, notes, target_tokens=128))
        hashes.append(hashlib.sha256(text.encode()).hexdigest()[:16])
        notes.append(text.replace(S, ""))
    answer = text_of(await stream(b, FABLE, q, notes, phase="answer"))
    hashes.append(hashlib.sha256(answer.encode()).hexdigest()[:16])
    assert hashes == ["09d5634921a64a7c", "fb87dfae67b20bf3", "5a96790d347095e2", "06eade36f5ed5e22",
                      "11258190941d000a"]
    short = text_of(await stream(b, FABLE, "Who is the minister?", target_tokens=20))
    assert short == ('First, the question ("Who is the minister?") is about who holds the office, so the answer '
                     "depends on recent cabinet changes, resignations and appointments rather than on stable "
                     "background knowledge.")


# --------------------------------------------------------------------------------------
# explore preset (DESIGN.md §7): structured notes under the per-segment directive
# --------------------------------------------------------------------------------------

EXPLORE_CFG = load_config(preset="explore")
EXPLORE_SYSTEM = relay_system_prompt_explore(192, "2026-11-24", S, FINAL)
Q = "Who is the minister of finance?"


def closed_tail(k: int, directive: str = "", missing: list[str] | None = None) -> str:
    items = missing if missing is not None else [f"{6 - k + 1} more main-line notes"]
    return relay_tail(k, record_status_text(items, False), directive)


def open_tail(k: int, directive: str = "") -> str:
    return relay_tail(k, record_status_text([], True), directive)


async def explore_note(b: MockBackend, model: ModelSpec, notes: list[str], tail: str, **kw) -> str:
    raw = text_of(await stream(b, model, Q, notes, target_tokens=192, hard_cut_chars=1000,
                               system_override=EXPLORE_SYSTEM, tail=tail, sentinel=S, **kw))
    return clean_segment(raw, "", sentinel=S, hard_cut_chars=1000)[0]


async def test_explore_notes_are_labelled_items_ending_with_where_we_stand():
    b = mock()
    for n, model in enumerate((FABLE, SOL)):  # n >= 2 would be the one early conclusion (tested below)
        notes = ["a note." * 3] * n
        text = await explore_note(b, model, notes, closed_tail(n + 1))
        parsed = parse_labels(text)
        assert 2 <= parsed["items"] <= 4 and parsed["has_wws"], text
        assert parsed["counts"].get("Principle", 0) <= 1
        assert text.rstrip().startswith(tuple(f"{label}:" for label in parsed["counts"]))
        assert text.rstrip().endswith(".") and "Where we stand:" in text.splitlines()[-1]
        assert "known as of" in text or "as of 2026" in text  # still topic- and cutoff-specific
        assert FINAL not in text  # n < 2: no early conclusion, and the gate is closed


async def test_explore_notes_cover_the_labels_the_tail_says_are_missing():
    b = mock()
    missing = missing_record_items(main_line_count=1, min_segments=6, label_counts={},
                                   require_labels=EXPLORE_CFG.explore.require_labels, wws_recent=True,
                                   pending_check=False)
    assert len(missing) == 4  # five more notes + three labels
    text = await explore_note(b, SOL, ["a first note."], closed_tail(2, missing=missing))
    counts = parse_labels(text)["counts"]
    assert counts.get("Evidence") == 1 and counts.get("Alternative set aside") == 1 and counts.get("Open question") == 1
    assert "(source:" in text  # the Evidence item names its source
    custom = closed_tail(2, missing=["an 'Observation:' item"])
    assert "Observation:" in await explore_note(b, SOL, ["a first note."], custom)  # unknown labels get a generic item


async def test_explore_lens_note_opens_with_the_field_and_carries_an_inference():
    b = mock()
    directive = lens_text("music theory", ("counterpoint", "polyrhythm", "harmonic series"))
    text = await explore_note(b, NEON, ["one.", "two."], closed_tail(3, directive))
    parsed = parse_labels(text)
    assert text.startswith("Lens (music theory):") and parsed["lens_prefix"] == "music theory"
    assert "counterpoint" in text and "polyrhythm" in text and "harmonic series" not in text.split("\n")[0][-30:]
    assert parsed["counts"].get("Inference") == 1 and "working analogy" in text
    assert parsed["has_wws"] and "Back to the problem" in text and len(text) < 1000
    assert text.index("Back to the problem") < text.index("Where we stand:")


async def test_explore_check_verdict_is_deterministic_and_mostly_kept():
    directive = check_text("music theory")
    a = await explore_note(mock(seed=1), FABLE, ["one.", "two.", "Lens (music theory): x."], closed_tail(4, directive))
    b = await explore_note(mock(seed=1), FABLE, ["one.", "two.", "Lens (music theory): x."], closed_tail(4, directive))
    assert a == b and parse_labels(a)["lens_verdict"] in ("kept", "dropped")
    assert a.startswith(("Lens kept:", "Lens dropped:")) and parse_labels(a)["has_wws"]
    verdicts = []
    backend = mock(seed=7)
    for i in range(120):
        notes = ["one."] * (i % 5 + 1)
        text = await explore_note(backend, (FABLE, SOL, NEON)[i % 3], notes, closed_tail(len(notes) + 1,
                                                                                       check_text(f"field {i}")))
        verdicts.append(parse_labels(text)["lens_verdict"])
    assert None not in verdicts
    assert 0.45 < verdicts.count("kept") / len(verdicts) < 0.75  # about 60 % kept


async def test_explore_continuation_starts_with_the_cue():
    b = mock()
    directive = continue_text("the record has no 'Open question:' item yet", "Alternatively, ")
    text = await explore_note(b, SOL, ["one.", "two.", "three."], closed_tail(4, directive))
    assert text.startswith("Alternatively, ") and FINAL not in text
    assert parse_labels(text)["counts"].get("Open question") == 1 and parse_labels(text)["has_wws"]
    colon = continue_text("why", "One thing not yet checked: ")
    text2 = await explore_note(b, SOL, ["one.", "two.", "three."], closed_tail(4, colon))
    assert text2.startswith("One thing not yet checked: ")


async def test_explore_direction_from_the_next_move_menu_shapes_the_first_claim():
    text = await explore_note(mock(), FABLE, ["one."], closed_tail(2, vs_direction_text("compare two sources")))
    assert text.startswith("Claim: following the direction (compare two sources)")


async def test_explore_final_answer_only_when_the_moderator_accepts_plus_one_early_attempt():
    b = mock()
    for n in (0, 1):
        assert FINAL not in await explore_note(b, FABLE, ["x."] * n, closed_tail(n + 1))
    early_raw = text_of(await stream(b, FABLE, Q, ["one.", "two."], system_override=EXPLORE_SYSTEM,
                                     tail=closed_tail(3), sentinel=S, hard_cut_chars=1000, target_tokens=192))
    note, answer = split_final(early_raw, FINAL)
    assert answer and "known as of" in answer
    assert note.startswith("Claim: the record already settles the problem") and "Where we stand:" in note
    # the relay commits the note part; from then on the mock concludes only when the gate is open
    later = await explore_note(b, SOL, ["one.", "two.", note], closed_tail(4))
    assert FINAL not in later and FINAL not in await explore_note(b, NEON, ["one.", "two.", note, later],
                                                                  closed_tail(5))
    accepted = text_of(await stream(b, SOL, Q, ["one.", "two.", note, later], system_override=EXPLORE_SYSTEM,
                                    tail=open_tail(5), sentinel=S))
    assert accepted.startswith(FINAL) and accepted.endswith(S)  # the answer instead of a note
    assert "should be checked against a current source" in accepted


async def test_explore_notes_fit_the_hard_cut_for_every_topic():
    b = mock(classifier=None)  # the keyword classifier: whichever topic phrases a question hits
    questions = ["What happened on the front line near the city?", "Which bank changed its policy rate?",
                 "What is the mission status of the Moon lander?", "Who won the championship?"]
    directive = lens_text("narrative and visual arts", ("narrative arc", "negative space", "foreshadowing"))
    for q in questions:
        for tail in (closed_tail(3, directive), closed_tail(3, check_text("narrative and visual arts")),
                     closed_tail(3)):
            raw = text_of(await stream(b, SOL, q, ["one.", "two."], target_tokens=192, hard_cut_chars=1000,
                                       system_override=EXPLORE_SYSTEM, tail=tail, sentinel=S))
            text = raw.replace(S, "")
            assert len(text) < 1000 and text.rstrip().endswith("."), text
            assert "Where we stand:" in text.splitlines()[-1]


async def test_explore_answer_names_its_basis_when_asked():
    b = mock()
    with_basis = text_of(await stream(b, FABLE, Q, ["a."], phase="answer",
                                      system_override=answer_system_prompt_explore("2026-11-24")))
    assert with_basis.startswith("Based on the shared notes") and "\n\nBasis: " in with_basis
    assert with_basis.rstrip().split("\n")[-1].startswith("Not resolved: ")
    without = text_of(await stream(b, FABLE, Q, ["a."], phase="answer",
                                   system_override=answer_system_prompt_explore("2026-11-24", answer_basis=False)))
    assert "Basis:" not in without and without == with_basis.split("\n\nBasis:")[0]
    plain = text_of(await stream(b, FABLE, Q, ["a."], phase="answer"))
    assert plain == without  # the default answer is unchanged


async def test_explore_usage_counts_the_tail_and_stream_shape_is_unchanged():
    b = mock()
    events = await stream(b, SOL, Q, [], system_override=EXPLORE_SYSTEM, tail=closed_tail(1), sentinel=S)
    assert events[-1].kind == "done" and events[-1].finish_reason == "sentinel"
    assert text_of(events).endswith(S) and all(0 < len(e.text) <= 8 for e in events if e.kind == "text")
    plain = await stream(b, SOL, Q, [], sentinel=S)
    assert events[-1].usage["prompt_tokens"] > plain[-1].usage["prompt_tokens"]


# --------------------------------------------------------------------------------------
# chat probes
# --------------------------------------------------------------------------------------


def make_event(i: int, month: int, topic: str = "government", **kw) -> Event:
    year, mon = 2024 + month // 12, month % 12 + 1
    base = dict(
        id=f"ev{i}", date=f"{year:04d}-{mon:02d}-15", month=f"{year:04d}-{mon:02d}", topic=topic,
        category="office_change", subject=f"Person {i}", fact=f"Person {i} became minister",
        question_current="Who is the minister?", expected_current=f"Person {i}",
        question_recall="Who became minister?", expected_recall=f"Person {i} (recall)",
        mcq_question="Who is the minister?",
        mcq_choices=[f"A) Person {i}", f"B) Old Minister {i}", "C) Someone", "D) Nobody"], mcq_answer="A",
        aliases=[f"Person {i}"], reject_aliases=[f"Old Minister {i}"],
    )
    base.update(kw)
    return Event(**base)


async def ask(b: MockBackend, model: ModelSpec, event: Event, probe: str):
    meta = {"event_id": event.id, "probe": probe}
    return await b.chat(model, ChatRequest(messages=[{"role": "user", "content": "Q"}], metadata=meta))


async def test_probe_accuracy_follows_true_cutoff():
    sol_tau = default_mock_truths()["gpt-6.2-sol"].tau_for("government")  # 28.75
    old = [make_event(i, month_index(2025, 3)) for i in range(150)]
    new = [make_event(1000 + i, month_index(2026, 11)) for i in range(150)]
    assert month_index(2025, 3) < sol_tau - 5 and month_index(2026, 11) > sol_tau + 5
    b = mock(events={e.id: e for e in old + new})

    async def accuracy(events):
        results = [await ask(b, SOL, e, "current") for e in events]
        return sum(r.raw["outcome"] == "correct" for r in results) / len(results), results

    acc_old, _ = await accuracy(old)
    acc_new, res_new = await accuracy(new)
    assert acc_old > 0.85 and acc_new < 0.1
    stale = [r for r in res_new if r.raw["outcome"] == "outdated"]
    assert stale and all(re.fullmatch(r"Old Minister \d+\.", r.text) for r in stale)  # reject_aliases[0]
    assert {r.raw["outcome"] for r in res_new} <= {"outdated", "abstain", "correct"}


async def test_probe_answers_are_deterministic_and_seed_dependent():
    events = [make_event(i, month_index(2026, 5)) for i in range(60)]
    b1, b2, b3 = (mock(seed=s, events={e.id: e for e in events}) for s in (1, 1, 2))
    r1 = [(await ask(b1, NEON, e, "current")).text for e in events]
    r2 = [(await ask(b2, NEON, e, "current")).text for e in events]
    r3 = [(await ask(b3, NEON, e, "current")).text for e in events]
    assert r1 == r2 and r1 != r3


async def test_mcq_returns_letters_with_guess_floor_and_stale_option():
    events = [make_event(i, month_index(2026, 11)) for i in range(200)]  # far past every cutoff
    b = mock(events={e.id: e for e in events})
    res = [await ask(b, SOL, e, "mcq") for e in events]
    assert all(r.text in {"A", "B", "C", "D"} for r in res)
    correct = sum(r.text == "A" for r in res) / len(res)
    assert 0.15 < correct < 0.4  # 25% guessing floor
    wrong = [r.text for r in res if r.text != "A"]
    assert wrong.count("B") / len(wrong) > 0.8  # Sol (stale_rate 0.95) picks the predecessor


async def test_recall_probe_and_generic_stale_texts():
    truths = default_mock_truths(latency_s=0)
    truths["gemini-4-neon"].stale_rate = 1.0
    truths["gemini-4-neon"].plateau = 1.0
    death = make_event(1, month_index(2026, 11), category="death", subject="Jane Doe", reject_aliases=[],
                       mcq_choices=[])
    office = make_event(2, month_index(2026, 11), reject_aliases=[])
    past = make_event(3, month_index(2024, 3))
    b = mock(truths=truths, events={e.id: e for e in (death, office, past)})
    assert (await ask(b, NEON, death, "current")).text == "As far as I know, Jane Doe is alive."
    assert "still in office" in (await ask(b, NEON, office, "current")).text
    assert (await ask(b, NEON, past, "recall")).text == "Person 3 (recall)."
    truths["gemini-4-neon"].stale_rate = 0.0
    assert "don't know" in (await ask(b, NEON, office, "current")).text


async def test_unscripted_chat():
    res = await mock().chat(FABLE, ChatRequest(messages=[{"role": "user", "content": "Hello"}]))
    assert "no scripted answer" in res.text and res.raw["outcome"] == "unscripted" and res.finish_reason == "stop"
    res2 = await mock().chat(FABLE, ChatRequest(messages=[], metadata={"event_id": "missing", "probe": "current"}))
    assert res2.raw["outcome"] == "unscripted"


# --------------------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------------------


def test_make_backend():
    real = make_backend(load_config(overrides={"api_key_env": "MY_KEY", "app_title": "demo"}))
    assert isinstance(real, OpenRouterBackend) and real.api_key_env == "MY_KEY" and real.app_title == "demo"
    cfg = load_config(overrides={"today": "2026-11-24", "models": [
        {"key": "gpt-6.2-sol", "provider_id": "openai/gpt-6.2-sol"}, {"key": "extra", "provider_id": "x/extra"}]})
    fake = make_backend(cfg, mock=True, seed=5)
    assert isinstance(fake, MockBackend) and fake.seed == 5 and fake.today == "2026-11-24"
    assert set(cfg.model_keys()) <= set(fake.truths)
