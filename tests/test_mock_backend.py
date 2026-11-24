"""Tests for cutoffmix.backends.mock (and the backends package factory)."""

from __future__ import annotations

import re

import pytest

from cutoffmix.backends import MockBackend, MockModelTruth, OpenRouterBackend, default_mock_truths, make_backend
from cutoffmix.backends.base import BackendError
from cutoffmix.config import default_config, load_config
from cutoffmix.prompts import clean_segment, split_final
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.timeutil import month_index, month_label
from cutoffmix.types import Backend, ChatRequest, ContinuationRequest, Event, ModelSpec, StreamEvent

CFG = default_config()
FABLE, SOL, NEON = (CFG.model(k) for k in ("claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"))
S = "<<END_STEP>>"


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
