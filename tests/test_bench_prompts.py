"""Tests for cutoffmix.bench.prompts (probe, judge and probe-generator prompts)."""

from __future__ import annotations

import datetime as dt

import pytest

from cutoffmix.bench import prompts as bp
from cutoffmix.types import Event


def _event(**kw) -> Event:
    base = dict(
        id="wd-office-Q890005", date="2026-10-12", month="2026-10", topic="government", category="office_change",
        subject="Ülle Madise", fact="Ülle Madise became President of Estonia on October 12, 2026.",
        question_current="Who is the President of Estonia?", expected_current="Ülle Madise",
        question_recall="Who became President of Estonia in October 2026?", expected_recall="Ülle Madise (Oct 2026)",
        mcq_question="Who is the President of Estonia?",
        mcq_choices=["A) Alar Karis", "B) Ülle Madise", "C) Kersti Kaljulaid", "D) Kaja Kallas"], mcq_answer="B",
        aliases=["Ülle Madise"], reject_aliases=["Alar Karis"],
    )
    base.update(kw)
    return Event(**base)


def test_kc_strings_are_verbatim():
    assert bp.PROBE_SYSTEM_NEUTRAL == ("Answer the question directly and factually based on what you know. "
                                       "If you are not sure, say so, but give your best answer.")
    assert bp.PROBE_SYSTEM_MCQ == ("Answer the multiple-choice question. Respond with ONLY the single letter "
                                   "(A, B, C, or D) of the best answer, nothing else.")


@pytest.mark.parametrize("probe,user", [
    ("current", "Who is the President of Estonia?"),
    ("recall", "Who became President of Estonia in October 2026?"),
])
def test_neutral_probes_have_no_date(probe, user):
    msgs = bp.probe_messages(_event(), probe, today="2026-11-24")
    assert msgs == [{"role": "system", "content": bp.PROBE_SYSTEM_NEUTRAL}, {"role": "user", "content": user}]
    assert "2026-11-24" not in msgs[0]["content"]


def test_mcq_probe_uses_kc_format():
    msgs = bp.probe_messages(_event(), "mcq")
    assert msgs[0]["content"] == bp.PROBE_SYSTEM_MCQ
    assert msgs[1]["content"] == ("Who is the President of Estonia?\n\nA) Alar Karis\nB) Ülle Madise\n"
                                  "C) Kersti Kaljulaid\nD) Kaja Kallas\n\nAnswer with a single letter:")


def test_current_d_has_date_line_and_question_current():
    msgs = bp.probe_messages(_event(), "current_d", today="2026-11-24")
    system = msgs[0]["content"]
    assert system.endswith(" Today's date: 2026-11-24.")
    assert system.startswith(bp.CURRENT_D_INSTRUCTION)
    assert msgs[1]["content"] == "Who is the President of Estonia?"


def test_current_d_defaults_to_local_date():
    assert bp.current_d_system().endswith(f"Today's date: {dt.date.today().isoformat()}.")


def test_current_d_with_deployment_preamble():
    with_date = bp.current_d_system("2026-11-24", "Relay preamble. Today's date: 2026-11-24.")
    assert with_date.count("2026-11-24") == 1
    assert with_date.startswith("Relay preamble.") and bp.CURRENT_D_INSTRUCTION in with_date
    without_date = bp.current_d_system("2026-11-24", "Relay preamble.")
    assert without_date.endswith("Today's date: 2026-11-24.")


def test_date_line_matches_relay_style():
    from cutoffmix.prompts import answer_system_prompt

    assert answer_system_prompt("2026-11-24").endswith(bp.DATE_LINE.format(today="2026-11-24"))
    assert bp.current_d_system("2026-11-24").endswith(bp.DATE_LINE.format(today="2026-11-24"))


def test_missing_question_and_unknown_probe():
    with pytest.raises(ValueError, match="no question for probe 'recall'"):
        bp.probe_messages(_event(question_recall=""), "recall")
    with pytest.raises(ValueError, match="no question for probe 'mcq'"):
        bp.probe_messages(_event(mcq_question="", mcq_choices=[]), "mcq")
    with pytest.raises(ValueError, match="unknown probe"):
        bp.probe_messages(_event(), "use")


def test_judge_system_contract():
    for word in ("correct", "outdated", "incorrect", "abstain", "hedged", "As of", "STRICT JSON"):
        assert word in bp.JUDGE_SYSTEM
    assert bp.JUDGE_LABELS == ("correct", "outdated", "incorrect", "abstain")


def test_judge_messages_content():
    msgs = bp.judge_messages(_event(), "current", "As of 2025, Alar Karis is the president.")
    assert msgs[0] == {"role": "system", "content": bp.JUDGE_SYSTEM}
    user = msgs[1]["content"]
    assert "QUESTION:\nWho is the President of Estonia?" in user
    assert "GROUND TRUTH:\nÜlle Madise" in user
    assert "Alar Karis; the previous holder is still in office" in user
    assert "Event date: 2026-10-12" in user
    assert user.endswith("MODEL ANSWER:\nAs of 2025, Alar Karis is the president.")


def test_judge_messages_recall_mcq_and_death_default():
    assert "GROUND TRUTH:\nÜlle Madise (Oct 2026)" in bp.judge_messages(_event(), "recall", "x")[1]["content"]
    assert "GROUND TRUTH:\nB) Ülle Madise" in bp.judge_messages(_event(), "mcq", "B")[1]["content"]
    death = _event(category="death", subject="Patrick Stewart", reject_aliases=[])
    assert "Patrick Stewart is alive or still active" in bp.judge_messages(death, "current", "x")[1]["content"]
    with pytest.raises(ValueError, match="matches no choice"):
        bp.judge_messages(_event(mcq_answer="E"), "mcq", "E")


def test_probegen_prompt():
    for field in bp.PROBEGEN_FIELDS:
        assert f'"{field}"' in bp.PROBEGEN_SYSTEM
    item = {"date": "2026-03-03", "ce_category": "Science and technology", "story_path": ["March 2026 lunar eclipse"],
            "text": "A total lunar eclipse occurs.", "links": ["Asia", "Oceania"],
            "sources": [{"url": "https://example.org/e", "outlet": "EclipseWise"}]}
    msgs = bp.probegen_messages(item)
    assert msgs[0]["content"] == bp.PROBEGEN_SYSTEM
    user = msgs[1]["content"]
    for part in ("EVENT DATE: 2026-03-03", "SECTION: Science and technology", "STORY: March 2026 lunar eclipse",
                 "TEXT: A total lunar eclipse occurs.", "LINKED ENTITIES: Asia; Oceania",
                 "SOURCE: https://example.org/e"):
        assert part in user
    bare = bp.probegen_messages({"date": "2026-03-03", "text": "x"})[1]["content"]
    assert "STORY: (none)" in bare and "SOURCE: (none)" in bare and "SECTION: (unknown)" in bare
