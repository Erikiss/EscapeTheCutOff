"""Tests for cutoffmix.bench.grading (MCQ letters, rule grading, judges)."""

from __future__ import annotations

from typing import Any

import pytest

from cutoffmix.backends.base import BackendError
from cutoffmix.bench.grading import (
    extract_letter,
    grade_mcq,
    grade_response,
    grade_results,
    grade_rule,
    grade_with_judges,
    normalize_text,
    parse_judge_output,
)
from cutoffmix.types import ChatRequest, ChatResult, Event, ModelSpec, ProbeResult

OFFICE = Event(
    id="e-office", date="2026-03-10", month="2026-03", topic="government", category="office_change",
    subject="Prime Minister of Exampleland",
    fact="Alice Smith succeeded Bob Jones as Prime Minister of Exampleland on 10 March 2026.",
    question_current="Who is the prime minister of Exampleland?", expected_current="Alice Smith",
    question_recall="Who became prime minister of Exampleland in March 2026?", expected_recall="Alice Smith",
    mcq_question="Who is the prime minister of Exampleland?",
    mcq_choices=["A) Bob Jones", "B) Alice Smith", "C) Carol White", "D) Dan Brown"], mcq_answer="B",
    aliases=["Alice Smith"], reject_aliases=["Bob Jones"],
)
DEATH = Event(
    id="e-death", date="2026-02-14", month="2026-02", topic="arts_entertainment", category="death",
    subject="Jane Example", fact="Actress Jane Example died on 14 February 2026.",
    question_current="Is actress Jane Example alive, or has she died? If she died, when?",
    expected_current="She died on February 14, 2026, at age 80.",
    mcq_question="Which statement about Jane Example is correct?",
    mcq_choices=["A) She is alive and filming a new series", "B) She died in February 2026",
                 "C) She died in 2019", "D) She retired to Spain"], mcq_answer="B",
)
CONTROL = Event(
    id="e-control", date="2026-06-01", month="2026-06", topic="arts_entertainment", category="control_alive",
    subject="Paul Example", fact="Musician Paul Example is alive.", question_current="Is musician Paul Example alive?",
    expected_current="Yes, Paul Example is alive.", mcq_question="Which statement about Paul Example is correct?",
    mcq_choices=["A) He died in 2025", "B) He is alive", "C) He died in early 2026", "D) He died on tour"],
    mcq_answer="B", control=True,
)
ACCENTED = Event(
    id="e-accent", date="2024-01-20", month="2024-01", topic="government", category="office_change",
    subject="President of the DRC", fact="Félix Tshisekedi was inaugurated for a second term.",
    question_current="Who was sworn in as president of the DRC after the most recent election?",
    expected_current="Félix Tshisekedi, inaugurated for a second term on January 20, 2024.",
    aliases=["Félix Tshisekedi"], reject_aliases=["Ana Li"],
)
NO_ALIASES = Event(
    id="e-head", date="2025-10-21", month="2025-10", topic="government", category="office_change",
    subject="Prime Minister of Japan", fact="Sanae Takaichi became prime minister of Japan.",
    question_current="Who is the prime minister of Japan?",
    expected_current="Sanae Takaichi, Japan's first female prime minister.",
    mcq_question="Who is Japan's prime minister?",
    mcq_choices=["A) Sanae Takaichi, the first female prime minister", "B) Shigeru Ishiba, second term",
                 "C) Fumio Kishida", "D) Yoshihide Suga"], mcq_answer="A",
)


# ------------------------------------------------------------------------------------------------- MCQ


@pytest.mark.parametrize("response, expected", [
    ("B", "B"), ("(C)", "C"), ("D.", "D"), ("a) Bob Jones", "A"), ("  B: Alice Smith", "B"),
    ("The answer is C", "C"), ("answer: d", "D"), ("**B**", "B"), ("I would pick D here.", "D"),
    ("Based on the options, C.", "C"), ("None of these", None), ("", None), ("I cannot tell.", None),
    ("A person who...", "A"),  # kc's leading rule is case-insensitive: replicated on purpose
])
def test_extract_letter_follows_kc_regex_order(response: str, expected: str | None) -> None:
    assert extract_letter(response) == expected


def test_extract_letter_valid_set_only_restricts_isolated_letters() -> None:
    assert extract_letter("I choose D", valid="ABC") is None
    assert extract_letter("D", valid="ABC") == "D"  # leading rule ignores the valid set, as in kc


@pytest.mark.parametrize("event, response, grade", [
    (OFFICE, "B", "correct"), (OFFICE, "A", "outdated"), (OFFICE, "C) Carol White", "incorrect"),
    (OFFICE, "I cannot tell.", "abstain"), (OFFICE, "   ", "ungraded"),
    (DEATH, "A", "outdated"), (DEATH, "C", "incorrect"), (DEATH, "B", "correct"),
])
def test_grade_mcq(event: Event, response: str, grade: str) -> None:
    assert grade_mcq(event, response) == grade


def test_grade_mcq_requires_an_answer_letter() -> None:
    with pytest.raises(ValueError, match="mcq_answer"):
        grade_mcq(Event(id="x", date="2026-01-01", month="2026-01", topic="sports", category="other", subject="s",
                        fact="f", mcq_choices=["A) a", "B) b"]), "A")


# ------------------------------------------------------------------------------------------------- rules


def test_normalize_text_folds_accents_case_quotes() -> None:
    assert normalize_text("  FÉLIX  Tshisekedi\u2019s Straße ") == "felix tshisekedi's strasse"


@pytest.mark.parametrize("response, expected", [
    ("Alice Smith is the prime minister of Exampleland.", ("correct", False)),
    ("The prime minister is Alice Smith, who succeeded Bob Jones in March 2026.", ("correct", False)),
    ("Bob Jones was replaced by Alice Smith in March.", ("correct", False)),
    ("Alice Smith. Bob Jones resigned earlier this year.", ("correct", False)),
    ("Bob Jones is the prime minister.", ("outdated", False)),
    ("As of my last update, Bob Jones was prime minister.", ("outdated", True)),
    ("I'm not sure who replaced him. As of early 2026, Bob Jones was still in office.", ("outdated", True)),
    ("Stand meines Wissens ist Bob Jones Premierminister.", ("outdated", True)),
    ("Bob Jones is no longer prime minister.", None),  # stale name only in a negated clause, nothing asserted
    ("Alice Smith and Bob Jones are both prime ministers.", None),  # conflicting assertions
    ("I don't know.", ("abstain", False)),
    ("Ich habe keine Informationen darüber.", ("abstain", False)),
    ("I'm not sure, but my best guess is Carol White.", None),  # a guess at an unknown name
    ("I cannot verify this; Carol White may have taken over.", None),  # names someone we cannot place
    ("Is it Alice Smith? I really cannot say.", None),  # a question is no answer, but any mention goes to judges
    ("I really cannot say who holds the office now.", ("abstain", False)),
    ("Carol White became prime minister in 2026.", None),  # unknown name: judges decide
    ("I'm not aware of an announcement; Alice Smith was mentioned as a candidate.", None),  # speculation
    ("", ("ungraded", False)),
])
def test_grade_rule_office_change(response: str, expected: tuple[str, bool] | None) -> None:
    assert grade_rule(OFFICE, "current", response) == expected


@pytest.mark.parametrize("response, expected", [
    ("Jane Example died on February 14, 2026, at the age of 80.", ("correct", False)),
    ("No, she passed away in February 2026.", ("correct", False)),
    ("As far as I know, Jane Example is alive.", ("outdated", True)),
    ("Jane Example is alive. I'm not aware of any credible report that she has died.", ("outdated", True)),
    ("As of my last update in March 2026, Jane Example was still alive, but I cannot confirm her current status.",
     ("outdated", True)),
    ("Soweit ich weiß, lebt Jane Example noch.", ("outdated", True)),
    ("Jane Example ist im Februar 2026 gestorben.", ("correct", False)),
    ("I can't confirm whether Jane Example is alive or when she died.", None),  # all mentions negated
    ("If she died, it happened after my knowledge cutoff.", None),
    ("Jane Example died in 2019.", None),  # wrong year: not the event's death
    ("Jane Example died on July 3, 2026.", None),  # wrong month
    ("I don't have reliable information about Jane Example.", ("abstain", False)),
    ("M. Example? I cannot verify whether Jane Example, the actress, has died.", None),  # negation spans the comma
])
def test_grade_rule_death(response: str, expected: tuple[str, bool] | None) -> None:
    assert grade_rule(DEATH, "current", response) == expected


@pytest.mark.parametrize("response, expected", [
    ("Yes, Paul Example is alive.", ("correct", False)),
    ("Paul Example died in 2025.", ("incorrect", False)),
    ("No, Paul Example has not died; reports of his death were a hoax. He is alive.", ("correct", False)),
])
def test_grade_rule_controls(response: str, expected: tuple[str, bool] | None) -> None:
    assert grade_rule(CONTROL, "current", response) == expected


def test_grade_rule_aliases_are_accent_case_insensitive_on_word_boundaries() -> None:
    assert grade_rule(ACCENTED, "current", "FELIX TSHISEKEDI is president.") == ("correct", False)
    assert grade_rule(ACCENTED, "current", "Félix-Tshisekedi was sworn in.") == ("correct", False)
    assert grade_rule(ACCENTED, "current", "Diana Lima is president.") is None  # "Ana Li" must not match inside


def test_grade_rule_derives_the_name_head_of_expected_answers() -> None:
    assert grade_rule(NO_ALIASES, "current", "The prime minister of Japan is **Sanae Takaichi**.") == \
        ("correct", False)
    assert grade_rule(NO_ALIASES, "current", "Shigeru Ishiba is prime minister.") is None


def test_grade_rule_mcq_and_recall_probes() -> None:
    assert grade_rule(OFFICE, "mcq", "B") == ("correct", False)
    assert grade_rule(OFFICE, "recall", "Alice Smith took over in March 2026.") == ("correct", False)


# ------------------------------------------------------------------------------------------------- judges


@pytest.mark.parametrize("text, expected", [
    ('{"label": "correct", "hedged": false, "reason": "x"}', ("correct", False)),
    ('```json\n{"label": "outdated", "hedged": true}\n```', ("outdated", True)),
    ('Sure! {"label": "abstain", "hedged": "yes"} hope this helps', ("abstain", True)),
    ('{"note": 1} then {"label": "stale"}', ("outdated", False)),
    ("label: other_wrong, hedged: false", ("incorrect", False)),
    ("Incorrect.", ("incorrect", False)),
    ('{"label": "banana"}', None), ("", None), ("{not json", None),
])
def test_parse_judge_output(text: str, expected: tuple[str, bool] | None) -> None:
    assert parse_judge_output(text) == expected


class JudgeBackend:
    """Scripted judge replies per model key; an exception item is raised instead of answering."""

    def __init__(self, replies: dict[str, list[Any]]) -> None:
        self.replies = {k: list(v) for k, v in replies.items()}
        self.calls: list[tuple[str, ChatRequest]] = []

    def stream_continuation(self, model: ModelSpec, req: Any) -> Any:
        raise NotImplementedError

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        self.calls.append((model.key, req))
        item = self.replies[model.key].pop(0)
        if isinstance(item, Exception):
            raise item
        return ChatResult(text=item)


J1, J2, J3 = (ModelSpec(key=k, provider_id=f"judge/{k}") for k in ("j1", "j2", "j3"))
OK = '{"label": "correct", "hedged": false}'
OUT = '{"label": "outdated", "hedged": true}'


async def test_two_agreeing_judges_decide_without_the_third() -> None:
    backend = JudgeBackend({"j1": [OK], "j2": [OK], "j3": [OUT]})
    assert await grade_with_judges(backend, [J1, J2, J3], OFFICE, "current", "Carol?") == \
        ("correct", False, "judge:j1+j2")
    assert [k for k, _ in backend.calls] == ["j1", "j2"]
    req = backend.calls[0][1]
    assert req.response_format == {"type": "json_object"} and req.temperature == 0.0


async def test_third_judge_breaks_a_disagreement() -> None:
    backend = JudgeBackend({"j1": [OK], "j2": [OUT], "j3": [OUT]})
    assert await grade_with_judges(backend, [J1, J2, J3], OFFICE, "current", "Bob?") == \
        ("outdated", True, "judge:j2+j3")


async def test_no_majority_or_failures_give_ungraded() -> None:
    three_way = JudgeBackend({"j1": [OK], "j2": [OUT], "j3": ['{"label": "abstain"}']})
    grade, hedged, grader = await grade_with_judges(three_way, [J1, J2, J3], OFFICE, "current", "?")
    assert (grade, hedged, grader) == ("ungraded", False, "judge:j1+j2+j3")
    failing = JudgeBackend({"j1": [RuntimeError("boom")], "j2": [OK], "j3": ["no label here"]})
    assert (await grade_with_judges(failing, [J1, J2, J3], OFFICE, "current", "?"))[0] == "ungraded"
    two_only = JudgeBackend({"j1": [OK], "j2": [OUT]})
    assert (await grade_with_judges(two_only, [J1, J2], OFFICE, "current", "?"))[0] == "ungraded"


async def test_one_failed_judge_is_replaced_by_the_third() -> None:
    backend = JudgeBackend({"j1": [BackendError("down", status=503)], "j2": [OK], "j3": [OK]})
    assert await grade_with_judges(backend, [J1, J2, J3], OFFICE, "current", "?") == \
        ("correct", False, "judge:j2+j3")


async def test_judge_retries_without_json_mode_when_rejected() -> None:
    backend = JudgeBackend({"j1": [BackendError("response_format unsupported", status=400), OK]})
    assert await grade_with_judges(backend, [J1], OFFICE, "current", "?") == ("correct", False, "judge:j1")
    assert [req.response_format for _, req in backend.calls] == [{"type": "json_object"}, None]


async def test_grade_with_judges_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one judge"):
        await grade_with_judges(JudgeBackend({}), [], OFFICE, "current", "x")
    assert await grade_with_judges(JudgeBackend({}), [J1], OFFICE, "current", "  ") == ("ungraded", False, "judge")


async def test_grade_response_uses_judges_only_when_rules_are_unsure() -> None:
    backend = JudgeBackend({"j1": [OUT], "j2": [OUT]})
    assert await grade_response(OFFICE, "mcq", "B", backend=backend, judges=[J1, J2]) == ("correct", False, "mcq")
    assert await grade_response(OFFICE, "current", "Alice Smith.", backend=backend, judges=[J1, J2]) == \
        ("correct", False, "rule")
    assert backend.calls == []
    assert await grade_response(OFFICE, "current", "Carol White.") == ("ungraded", False, "rule")
    assert await grade_response(OFFICE, "current", "Carol White.", backend=backend, judges=[J1, J2]) == \
        ("outdated", True, "judge:j1+j2")


async def test_grade_results_regrades_only_unsure_rows() -> None:
    rows = [
        ProbeResult("e-office", "m", "current", "Carol White.", "ungraded"),
        ProbeResult("e-office", "m", "recall", "Alice Smith.", "incorrect", grader="judge:x"),
        ProbeResult("e-office", "m2", "current", "", "ungraded", error="timeout"),
        ProbeResult("unknown", "m", "current", "x", "ungraded"),
    ]
    backend = JudgeBackend({"j1": [OK], "j2": [OK]})
    out = await grade_results(rows, [OFFICE], backend=backend, judges=[J1, J2])
    assert [r.grade for r in out] == ["correct", "incorrect", "ungraded", "ungraded"]
    assert out[0].grader == "judge:j1+j2" and rows[0].grade == "ungraded"  # inputs untouched
    again = await grade_results(rows, {OFFICE.id: OFFICE}, regrade_all=True)
    assert [r.grade for r in again][:2] == ["ungraded", "correct"]
    with pytest.raises(ValueError):
        await grade_results(rows, [OFFICE], concurrency=0)
