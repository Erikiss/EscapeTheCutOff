"""Tests for cutoffmix.bench.probegen (validation gates, MCQ assembly, LLM probe generation with a fake backend)."""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import json
from typing import Any

import pytest

from cutoffmix.bench import probegen as pg
from cutoffmix.bench.prompts import PROBEGEN_SYSTEM
from cutoffmix.types import ChatRequest, ChatResult, Event, ModelSpec

MODEL = ModelSpec(key="generator", provider_id="vendor/generator")
ITEM = {
    "date": "2026-03-03", "ce_category": "Law and crime", "story_path": [], "topic": "law_crime",
    "text": "Purported yakuza leader Takeshi Ebisawa is convicted by a New York federal court and sentenced to "
            "20 years in prison for trafficking nuclear material, drugs, and weapons.",
    "links": ["Takeshi Ebisawa", "yakuza"], "sources": [{"url": "https://example.org/a", "outlet": "Japan Times"}],
}
GOOD = {
    "usable": True, "reason": "A verdict with a clear outcome.", "category": "other", "subject": "Takeshi Ebisawa",
    "q_current": "What prison sentence did a New York federal court give yakuza leader Takeshi Ebisawa?",
    "a_current": "20 years", "q_dated": "What sentence did Takeshi Ebisawa receive in March 2026?",
    "a_dated": "20 years in prison", "stale_answer": "He had not been sentenced",
    "mcq_question": "What prison sentence did a New York federal court give yakuza leader Takeshi Ebisawa?",
    "mcq_choices": ["A) 20 years", "B) He had not been sentenced", "C) Life imprisonment", "D) 5 years"],
    "mcq_answer": "A", "aliases": ["twenty years"], "distractor_sources": "typical sentences",
    "predictability": "low",
}


def _valid_event(**kw: Any) -> Event:
    choices, letter = pg.assemble_mcq("e1", "Ülle Madise", ["Alar Karis", "Kersti Kaljulaid", "Kaja Kallas"])
    base = dict(id="e1", date="2026-10-12", month="2026-10", topic="government", category="office_change",
                subject="Ülle Madise", fact="f", question_current="Who is the President of Estonia?",
                expected_current="Ülle Madise", question_recall="Who became President of Estonia in October 2026?",
                expected_recall="Ülle Madise", mcq_question="Who is the President of Estonia?", mcq_choices=choices,
                mcq_answer=letter, aliases=["Ülle Madise", "Ulle Madise"], reject_aliases=["Alar Karis"])
    base.update(kw)
    return Event(**base)


def test_stable_hash_is_deterministic():
    assert pg.stable_hash("abc") == pg.stable_hash("abc") != pg.stable_hash("abd")
    assert pg.stable_hash("abc") == 13436514500253700074  # independent of PYTHONHASHSEED


def test_assemble_mcq_balanced_and_consistent():
    letters = collections.Counter()
    for i in range(400):
        choices, letter = pg.assemble_mcq(f"ev-{i}", "Right", ["Stale", "Other 1", "Other 2"])
        letters[letter] += 1
        assert [c[:3] for c in choices] == ["A) ", "B) ", "C) ", "D) "]
        assert choices["ABCD".index(letter)] == f"{letter}) Right"
        assert sorted(c[3:] for c in choices) == ["Other 1", "Other 2", "Right", "Stale"]
        assert pg.assemble_mcq(f"ev-{i}", "Right", ["Stale", "Other 1", "Other 2"]) == (choices, letter)
    assert all(70 <= n <= 130 for n in letters.values()), letters


def test_assemble_mcq_rejects_bad_options():
    with pytest.raises(ValueError, match="exactly 3"):
        pg.assemble_mcq("e", "a", ["b", "c"])
    with pytest.raises(ValueError, match="unique"):
        pg.assemble_mcq("e", "a", ["b", "c", "A"])


def test_name_variants():
    assert pg.name_variants("Ülle Madise") == ["Ülle Madise", "Ulle Madise"]
    assert pg.name_variants("Louis Gossett Jr.") == ["Louis Gossett Jr.", "Louis Gossett"]
    assert pg.name_variants("Democratic Party (United States)") == ["Democratic Party (United States)",
                                                                     "Democratic Party"]


def test_date_tokens():
    assert pg.date_tokens("Who won in 2026 and the 1990s?") == ["2026", "1990s"]
    assert pg.date_tokens("What happened in March?") == ["March"]
    assert pg.date_tokens("Who died on 3 May?") == ["3 May"]
    assert pg.date_tokens("Who may succeed the current pope?") == []
    assert pg.date_tokens("Who won the most recent Marche regional election?") == []  # region, not month
    assert pg.date_tokens("What changed on Oct. 3?") == ["Oct."]
    assert pg.date_tokens("Who scored 120 points with 12026 fans?") == []
    assert pg.date_tokens("Which deal closed on 3.12.2026?") == ["2026"]
    assert pg.month_year("2026-03-17") == "March 2026" and pg.long_date("2026-03-07") == "March 7, 2026"


def test_validate_event_accepts_valid_event():
    assert pg.validate_event(_valid_event()) == []


@pytest.mark.parametrize("changes,fragment", [
    ({"date": "2026-13-01"}, "not an ISO date"),
    ({"month": "2026-09"}, "does not match date"),
    ({"topic": "politics"}, "unknown topic"),
    ({"category": "news"}, "unknown category"),
    ({"predictability": "certain"}, "bad predictability"),
    ({"question_current": "Who became President of Estonia in October?"}, "date tokens"),
    ({"question_current": "Who is the President of Estonia in 2026?"}, "date tokens"),
    ({"expected_current": ""}, "without expected_current"),
    ({"question_recall": "Did Ülle Madise become president?"}, "appears in a question"),
    ({"mcq_choices": ["A) x", "B) y", "C) z"]}, "4 choices"),
    ({"mcq_choices": ["A) x", "B) y", "C) x", "D) w"]}, "not unique"),
    ({"mcq_choices": ["A) x", "C) y", "B) z", "D) w"]}, "must start with 'B) '"),
    ({"mcq_answer": "E"}, "not one of A-D"),
    ({"reject_aliases": ["ulle madise"]}, "overlap"),
    ({"question_current": "", "question_recall": "", "mcq_question": "", "mcq_choices": [], "mcq_answer": ""},
     "no probe question"),
])
def test_validate_event_gates(changes, fragment):
    problems = pg.validate_event(dataclasses.replace(_valid_event(), **changes))
    assert any(fragment in p for p in problems), problems


def test_parse_json_object():
    assert pg.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ValueError, match="no JSON object"):
        pg.parse_json_object("sorry")
    with pytest.raises(ValueError, match="invalid JSON"):
        pg.parse_json_object("{not json}")


def test_event_from_probe_json():
    event = pg.event_from_probe_json(ITEM, GOOD)
    assert event is not None and pg.validate_event(event) == []
    assert event.id == pg.item_event_id(ITEM) and event.id.startswith("wp-2026-03-03-")
    assert event.origin == "wikipedia" and event.topic == "law_crime" and event.predictability == "low"
    assert event.question_recall == GOOD["q_dated"] and event.expected_recall == GOOD["a_dated"]
    assert event.mcq_choices["ABCD".index(event.mcq_answer)][3:] == "20 years"
    assert {c[3:] for c in event.mcq_choices} == {"20 years", "He had not been sentenced", "Life imprisonment",
                                                 "5 years"}
    assert event.aliases == ["twenty years", "20 years"]
    assert event.reject_aliases == ["He had not been sentenced"]
    assert event.source == "https://example.org/a"


def test_event_from_probe_json_edge_cases():
    assert pg.event_from_probe_json(ITEM, {"usable": False, "reason": "ongoing story"}) is None
    with pytest.raises(ValueError, match="mcq_answer"):
        pg.event_from_probe_json(ITEM, {**GOOD, "mcq_answer": "Z"})
    with pytest.raises(ValueError, match="4 mcq_choices"):
        pg.event_from_probe_json(ITEM, {**GOOD, "mcq_choices": ["a", "b"]})
    with pytest.raises(ValueError, match="q_current"):
        pg.event_from_probe_json(ITEM, {**GOOD, "q_current": ""})
    odd = pg.event_from_probe_json({**ITEM, "topic": None}, {**GOOD, "category": "fake_event",
                                                            "predictability": "?"})
    assert odd is not None and odd.category == "other" and odd.predictability == "medium"
    assert odd.topic == "law_crime"  # from the Current-events heading via ce_item_topic


class FakeBackend:
    """Returns scripted replies per item text; records requests and the peak number of concurrent calls."""

    def __init__(self, replies: dict[str, Any]):
        self.replies = replies
        self.requests: list[ChatRequest] = []
        self.active = 0
        self.peak = 0

    def stream_continuation(self, model, req):  # pragma: no cover - not used by probegen
        raise NotImplementedError

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        self.requests.append(req)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(0.01)
            text = req.messages[1]["content"].split("TEXT: ", 1)[1].split("\n", 1)[0]
            reply = self.replies[text]
            if isinstance(reply, Exception):
                raise reply
            return ChatResult(text=reply if isinstance(reply, str) else json.dumps(reply))
        finally:
            self.active -= 1


async def test_generate_probes_filters_and_limits_concurrency():
    items = [dict(ITEM, text=f"Item {i}: the court sentences the defendant.") for i in range(8)]
    replies: dict[str, Any] = {it["text"]: GOOD for it in items}
    replies[items[1]["text"]] = RuntimeError("HTTP 500")
    replies[items[2]["text"]] = "I cannot help with that."
    replies[items[3]["text"]] = {"usable": False}
    replies[items[4]["text"]] = {**GOOD, "q_current": "What sentence did he get in March 2026?"}
    backend = FakeBackend(replies)
    events = await pg.generate_probes(backend, MODEL, items + [items[0]], concurrency=3)
    assert [e.fact for e in events] == [items[i]["text"] for i in (0, 5, 6, 7)]
    assert backend.peak <= 3 and len(backend.requests) == 9
    req = backend.requests[0]
    assert req.response_format == {"type": "json_object"}
    assert req.reasoning == {"effort": "low", "exclude": True}
    assert req.messages[0]["content"] == PROBEGEN_SYSTEM
    assert req.metadata["purpose"] == "probegen"


async def test_generate_probes_rejects_bad_concurrency():
    with pytest.raises(ValueError):
        await pg.generate_probes(FakeBackend({}), MODEL, [], concurrency=0)
