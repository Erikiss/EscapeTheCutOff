"""Tests for cutoffmix.bench.kc_import on a 14-event sample of the kc benchmark (CC BY 4.0; no network)."""

from __future__ import annotations

import collections
import copy
import json
from pathlib import Path

import httpx
import pytest

from cutoffmix.bench import kc_import as kc
from cutoffmix.bench import sources
from cutoffmix.bench.io import load_events, load_results, read_jsonl
from cutoffmix.bench.sources.wikidata import USER_AGENT
from cutoffmix.taxonomy import TOPIC_IDS

FIX = Path(__file__).parent / "fixtures"
RAW = read_jsonl(FIX / "kc_events_sample.jsonl")
ANSWERS = json.loads((FIX / "kc_report_answers_sample.json").read_text(encoding="utf-8"))

EXPECTED_TOPICS = {
    "2024-01-death-beckenbauer": "sports",  # "German football legend"
    "2024-01-death-musser": "arts_entertainment",  # "American actor"
    "2024-03-death-kahneman": "science_research",  # "Nobel laureate psychologist"
    "2025-04-death-pope-francis": "government",  # popes are institutional leaders
    "2024-02-death-chao": "business_economy",  # "CEO of Foremost Group"
    "2024-02-office-finland": "elections",  # "Who won the most recent Finnish presidential election ..."
    "2024-09-office-hill": "business_economy",  # Nike CEO
    "control-alive-messi": "sports",  # "Is footballer Lionel Messi alive?"
    "fake-death-keanu-reeves": "arts_entertainment",
    "2024-06-office-nato": "international_relations",  # NATO Secretary General
    "2024-07-death-haniyeh": "conflict_security",  # Hamas leader killed
    "2024-02-death-weathers": "arts_entertainment",  # "Actor and former NFL player": earliest mention wins
    "2026-05-death-busch": "sports",  # NASCAR
    "2024-03-office-rnc": "government",  # party chair, not an election question
}


def test_tag_topic_on_sample():
    assert {d["id"]: kc.tag_topic(d) for d in RAW} == EXPECTED_TOPICS


@pytest.mark.parametrize("subject,text,topic", [
    ("Jim Lovell", "Jim Lovell, the NASA astronaut who commanded Apollo 13, died at 97.", "space_astronomy"),
    ("Suchir Balaji", "Suchir Balaji, a former OpenAI researcher, died.", "ai_ml"),
    ("Ozzy Osbourne", "Ozzy Osbourne, Black Sabbath frontman and heavy metal pioneer, died.", "arts_entertainment"),
    ("Dick Cheney", "Former U.S. Vice President Dick Cheney died.", "government"),
    ("Making Example", "Jane Doe, a screenwriter, died.", "arts_entertainment"),  # topic_for_occupation fallback
    ("Q", "Q, whose work nobody described, died.", "arts_entertainment"),  # default
])
def test_person_topic_rules(subject, text, topic):
    assert kc.person_topic(subject, text) == topic


@pytest.mark.parametrize("subject,question,topic", [
    ("Apple CEO succession", "Who did Apple name to become its next chief executive officer?", "business_economy"),
    ("Susan Monarez", "What became of Susan Monarez, director of the U.S. Centers for Disease Control?",
     "health_medicine"),
    ("Erik Siebert", "Who resigned as U.S. Attorney for the Eastern District of Virginia?", "law_crime"),
    ("Zohran Mamdani", "Who won the Democratic Party primary for Mayor of New York City?", "elections"),
    ("Friedrich Merz", "Who is the current Chancellor of Germany?", "government"),
])
def test_office_topic_rules(subject, question, topic):
    assert kc.office_topic(subject, question) == topic


def test_convert_events():
    events = {e.id: e for e in kc.convert_events(RAW)}
    assert len(events) == len(RAW)
    for d in RAW:
        e = events[d["id"]]
        assert e.origin == "kc" and e.topic in TOPIC_IDS
        assert e.question_current == d["question_direct"] and e.expected_current == d["expected_direct"]
        assert e.mcq_choices == d["mcq_choices"] and e.mcq_answer == d["mcq_answer"]
        assert e.control == (d["category"] in ("control_alive", "fake_event"))
        assert e.question_recall == ""
    stubb = events["2024-02-office-finland"]
    assert stubb.aliases == ["Alexander Stubb"] and stubb.reject_aliases == ["Sauli Niinisto"]
    assert events["2024-09-office-hill"].reject_aliases == ["John Donahoe"]
    assert events["2024-03-office-rnc"].reject_aliases == ["Ronna McDaniel"]
    assert events["2024-01-death-beckenbauer"].aliases == [] == events["2024-01-death-beckenbauer"].reject_aliases


def _results(answers=ANSWERS, models=None):
    events = {e.id: e for e in kc.convert_events(RAW)}
    return kc.convert_answers(answers, events, models), events


def test_convert_answers_mapping():
    results, _ = _results()
    assert len(results) == 3 * 2 * 14
    assert {r.grader for r in results} == {"kc"} and {r.probe for r in results} == {"current", "mcq"}
    grade = {(r.model, r.probe, r.event_id): r.grade for r in results}
    assert grade[("gpt-6.2-sol", "current", "2024-01-death-beckenbauer")] == "correct"
    assert grade[("gpt-4o", "current", "2024-01-death-beckenbauer")] == "outdated"  # w on a death
    assert grade[("gpt-4o", "current", "2024-01-death-musser")] == "abstain"
    assert grade[("gpt-4o", "mcq", "2024-01-death-beckenbauer")] == "outdated"  # chose "He is alive ..."
    assert grade[("gpt-4o", "mcq", "2024-02-death-chao")] == "outdated"  # "She is still leading ..."
    assert grade[("gpt-4o", "mcq", "2024-02-office-finland")] == "incorrect"  # "B) Pekka Haavisto": not stale
    assert grade[("gpt-4o", "mcq", "2024-09-office-hill")] == "incorrect"
    resp = {(r.model, r.probe, r.event_id): r.response for r in results}
    assert resp[("gpt-4o", "mcq", "2024-02-office-finland")] == "B) Pekka Haavisto"


def test_convert_answers_controls_ungraded_and_filters():
    answers = copy.deepcopy(ANSWERS)
    answers["gpt-4o"]["direct"]["control-alive-messi"] = {"l": "w", "r": "He died in 2025."}
    answers["gpt-4o"]["mcq"]["control-alive-messi"] = {"l": "u"}
    answers["gpt-4o"]["direct"]["not-in-events"] = {"l": "c", "r": "x"}
    results, _ = _results(answers, models=["gpt-4o"])
    grade = {(r.probe, r.event_id): r.grade for r in results}
    assert grade[("current", "control-alive-messi")] == "incorrect"  # w on a control is not "outdated"
    assert grade[("mcq", "control-alive-messi")] == "ungraded"
    assert {r.model for r in results} == {"gpt-4o"} and ("current", "not-in-events") not in grade
    counts = collections.Counter(r.grade for r in results)
    assert counts["ungraded"] == 1


def test_convert_answers_rejects_unknown_codes():
    with pytest.raises(ValueError, match="unknown kc label"):
        _results({"m": {"direct": {"2024-01-death-musser": {"l": "x"}}}})
    with pytest.raises(ValueError, match="unknown kc probe"):
        _results({"m": {"essay": {}}})


def _transport(calls: list[httpx.Request]) -> httpx.MockTransport:
    events_body = (FIX / "kc_events_sample.jsonl").read_bytes()
    answers_body = (FIX / "kc_report_answers_sample.json").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        url = str(request.url)
        if url == kc.KC_EVENTS_URL:  # Hugging Face redirects "resolve" URLs to its CDN
            return httpx.Response(302, headers={"Location": "https://cdn.example.org/events.jsonl"})
        if url == "https://cdn.example.org/events.jsonl":
            return httpx.Response(200, content=events_body)
        if url == kc.KC_ANSWERS_URL:
            return httpx.Response(200, content=answers_body)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_download_kc_follows_redirects_and_keeps_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "pause", lambda s: None)
    calls: list[httpx.Request] = []
    with httpx.Client(transport=_transport(calls), headers={"User-Agent": USER_AGENT}) as client:
        events_path, answers_path = kc.download_kc(tmp_path / "kc", client=client)
        assert events_path.read_bytes() == (FIX / "kc_events_sample.jsonl").read_bytes()
        assert json.loads(answers_path.read_text(encoding="utf-8")) == ANSWERS
        assert len(calls) == 3 and all(c.headers["User-Agent"] == USER_AGENT for c in calls)
        kc.download_kc(tmp_path / "kc", client=client)
        assert len(calls) == 3  # existing files are kept
        kc.download_kc(tmp_path / "kc", client=client, force=True)
        assert len(calls) == 6
    assert not list((tmp_path / "kc").glob("*.part"))


def test_import_kc_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(sources, "pause", lambda s: None)
    with httpx.Client(transport=_transport([])) as client:
        events_path, results_path = kc.import_kc(tmp_path, client=client, models=["claude-fable-5.2"])
    events = load_events(events_path)
    results = load_results(results_path)
    assert len(events) == 14 and len(results) == 28
    assert {r.model for r in results} == {"claude-fable-5.2"}
    assert {r.event_id for r in results} <= {e.id for e in events}
