"""Prompts for probing models, judging their answers and generating probes (DESIGN.md §3.5).

* ``current`` / ``recall`` use :data:`PROBE_SYSTEM_NEUTRAL` and ``mcq`` uses :data:`PROBE_SYSTEM_MCQ` with the user
  format :data:`MCQ_USER_TEMPLATE`: the exact strings of the apoorvumang/knowledge-cutoff benchmark (its "direct" and
  "mcq" probes), so results stay comparable. They reveal neither the date nor the expected answer.
* ``current_d`` (the primary routing index) asks the same date-free question under deployment conditions: the
  system prompt carries the relay's date line (``Today's date: YYYY-MM-DD``, the style of
  :func:`cutoffmix.prompts.answer_system_prompt`) and the relay's recency instruction, and asks for a direct answer.
* :data:`JUDGE_SYSTEM` grades free-text answers against the ground truth into correct / outdated / incorrect /
  abstain plus a ``hedged`` flag, as strict JSON.
* :data:`PROBEGEN_SYSTEM` turns one Wikipedia Current-events item into probe questions, as strict JSON.

Messages are plain ``[{"role": ..., "content": str}]`` lists for :class:`cutoffmix.types.ChatRequest`.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Mapping

from cutoffmix.types import Event

# --------------------------------------------------------------------------------------
# Probe prompts
# --------------------------------------------------------------------------------------

# Exact benchmark strings (kc/prompts.py): facts about the benchmark, kept verbatim for comparability.
PROBE_SYSTEM_NEUTRAL = (
    "Answer the question directly and factually based on what you know. "
    "If you are not sure, say so, but give your best answer."
)
PROBE_SYSTEM_MCQ = (
    "Answer the multiple-choice question. Respond with ONLY the single "
    "letter (A, B, C, or D) of the best answer, nothing else."
)
MCQ_USER_TEMPLATE = "{question}\n\n{choices}\n\nAnswer with a single letter:"

# Same wording as the relay's date sentence in cutoffmix.prompts (" Today's date: {today}.").
DATE_LINE = " Today's date: {today}."
CURRENT_D_INSTRUCTION = (
    "Answer the question in the user message directly and factually, in one or two sentences. Prefer the most "
    "recent facts you know and say when a fact may have changed. If you are not sure, say so, but give your best "
    "answer."
)

PROBE_KINDS: tuple[str, ...] = ("current", "current_d", "recall", "mcq")


def current_d_system(today: str | None = None, deployment_system: str | None = None) -> str:
    """System prompt of the ``current_d`` probe.

    Without ``deployment_system``: :data:`CURRENT_D_INSTRUCTION` plus the date line. With it (e.g. the relay's
    answer-phase system prompt): that preamble, then the instruction; the date line is appended only if the
    preamble does not already contain ``today``. ``today`` defaults to the local date (ISO).
    """
    day = today or _dt.date.today().isoformat()
    if deployment_system is None:
        return CURRENT_D_INSTRUCTION + DATE_LINE.format(today=day)
    text = deployment_system.rstrip() + "\n\n" + CURRENT_D_INSTRUCTION
    return text if day in deployment_system else text + DATE_LINE.format(today=day)


def mcq_user_text(event: Event) -> str:
    """The benchmark's MCQ user message: question, one choice per line, then the single-letter request."""
    return MCQ_USER_TEMPLATE.format(question=event.mcq_question, choices="\n".join(event.mcq_choices))


def probe_question(event: Event, probe: str) -> str:
    """The user text a model sees for ``probe``; ``ValueError`` if the event has no question for it."""
    if probe in ("current", "current_d"):
        text = event.question_current
    elif probe == "recall":
        text = event.question_recall
    elif probe == "mcq":
        text = mcq_user_text(event) if event.mcq_question and event.mcq_choices else ""
    else:
        raise ValueError(f"unknown probe {probe!r}; expected one of {', '.join(PROBE_KINDS)}")
    if not text:
        raise ValueError(f"event {event.id!r} has no question for probe {probe!r}")
    return text


def probe_messages(event: Event, probe: str, *, today: str | None = None,
                   deployment_system: str | None = None) -> list[dict[str, Any]]:
    """Chat messages for one probe of one event.

    ``current``/``recall``: neutral benchmark system prompt, no date. ``mcq``: benchmark MCQ prompt. ``current_d``:
    :func:`current_d_system` (``today`` and ``deployment_system`` are used only here) with ``question_current``.
    """
    user = probe_question(event, probe)
    if probe == "current_d":
        system = current_d_system(today, deployment_system)
    elif probe == "mcq":
        system = PROBE_SYSTEM_MCQ
    else:
        system = PROBE_SYSTEM_NEUTRAL
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------------------
# Judge
# --------------------------------------------------------------------------------------

JUDGE_LABELS: tuple[str, ...] = ("correct", "outdated", "incorrect", "abstain")

JUDGE_SYSTEM = (
    "You are a strict grader. You are given a QUESTION that was put to a model, the GROUND TRUTH answer, the "
    "OUTDATED answers that were true before the event, and the MODEL ANSWER. Classify the MODEL ANSWER into "
    "exactly one label:\n"
    "  correct   - it asserts the ground-truth fact (minor wording or date-precision differences are fine; the key "
    "fact must match).\n"
    "  outdated  - it asserts the state of the world from before the event: one of the outdated answers, or for "
    "example that a person who has died is alive or still active, that the previous office holder is still in "
    "office, that the previous winner is the latest winner, or that the event has not happened yet.\n"
    "  incorrect - it asserts something else that contradicts the ground truth (a wrong person, outcome or date, or "
    "an invented event).\n"
    "  abstain   - it declines to answer, says it does not know, or only says its information may be out of date, "
    "without committing to the true or to a false claim.\n"
    "Set \"hedged\" to true when the answer qualifies its claim with its knowledge horizon or with uncertainty "
    "('As of my last update', 'As of March 2026', 'I believe', 'this may have changed'). A hedged claim keeps the "
    "label of the claim it makes: 'As of <date>, <pre-event state>' is outdated with hedged true, and a hedged "
    "ground-truth answer is correct with hedged true. A refusal without any claim is abstain with hedged false.\n"
    "Judge ONLY against the provided ground truth and outdated answers. Do not use your own knowledge of current "
    "events. Output STRICT JSON only, no other text: "
    "{\"label\": \"correct|outdated|incorrect|abstain\", \"hedged\": true|false, \"reason\": \"<one sentence>\"}"
)

_DEFAULT_OUTDATED: Mapping[str, str] = {
    "death": "{subject} is alive or still active",
    "office_change": "the previous holder is still in office",
    "election": "the winner of the previous election",
    "award": "the previous recipient",
    "sports_result": "the previous winner or champion",
    "release": "the previous release or version",
    "control_alive": "(none: nothing changed)",
    "fake_event": "(none: the event did not happen)",
}


def _ground_truth(event: Event, probe: str) -> str:
    if probe == "mcq":
        letter = event.mcq_answer.strip().upper()
        for choice in event.mcq_choices:
            if choice.strip()[:1].upper() == letter:
                return choice.strip()
        raise ValueError(f"event {event.id!r}: mcq_answer {event.mcq_answer!r} matches no choice")
    if probe == "recall" and event.expected_recall:
        return event.expected_recall
    return event.expected_current or event.fact


def _outdated(event: Event) -> str:
    parts = list(dict.fromkeys(a for a in event.reject_aliases if a.strip()))
    default = _DEFAULT_OUTDATED.get(event.category, "the state of affairs before the event")
    parts.append(default.format(subject=event.subject or "the person"))
    return "; ".join(parts)


def judge_messages(event: Event, probe: str, response: str) -> list[dict[str, Any]]:
    """Messages asking a judge to label ``response`` to ``probe`` of ``event`` (answer with :data:`JUDGE_SYSTEM`'s
    JSON). The judge sees the question, the ground truth with the fact and event date as context, the outdated
    answers (``reject_aliases`` plus a category default) and the model answer."""
    user = (
        f"QUESTION:\n{probe_question(event, probe)}\n\n"
        f"GROUND TRUTH:\n{_ground_truth(event, probe)}\n"
        f"(context: {event.fact} Event date: {event.date}.)\n\n"
        f"OUTDATED ANSWERS (true before the event):\n{_outdated(event)}\n\n"
        f"MODEL ANSWER:\n{response}"
    )
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------------------
# Probe generation (Wikipedia Current-events items -> probes)
# --------------------------------------------------------------------------------------

PROBEGEN_FIELDS: tuple[str, ...] = (
    "usable", "reason", "category", "subject", "q_current", "a_current", "q_dated", "a_dated", "stale_answer",
    "mcq_question", "mcq_choices", "mcq_answer", "aliases", "distractor_sources", "predictability",
)

PROBEGEN_SYSTEM = (
    "You write test questions that check whether someone knows about one news event. You get one dated news item "
    "from the Wikipedia Current events portal. Use only facts stated in the item; never add facts you are unsure "
    "about.\n"
    "Return STRICT JSON only (no markdown, no other text) with exactly these fields:\n"
    "{\"usable\": true|false, \"reason\": \"<one sentence>\", "
    "\"category\": \"death|office_change|election|release|award|sports_result|other\", "
    "\"subject\": \"<main person, organisation or thing>\", "
    "\"q_current\": \"<question>\", \"a_current\": \"<short answer>\", "
    "\"q_dated\": \"<question>\", \"a_dated\": \"<short answer>\", "
    "\"stale_answer\": \"<pre-event answer to q_current, or empty>\", "
    "\"mcq_question\": \"<question>\", \"mcq_choices\": [\"<A>\", \"<B>\", \"<C>\", \"<D>\"], "
    "\"mcq_answer\": \"A|B|C|D\", "
    "\"aliases\": [\"<accepted short forms of a_current>\"], "
    "\"distractor_sources\": \"<where the wrong options come from>\", \"predictability\": \"low|medium|high\"}\n"
    "Rules:\n"
    "1. q_current asks about the present state of the world as it is after the event, so that the answer is a "
    "short name, outcome or number. It must make sense without any date: no year, month, day or relative time "
    "('recently', 'this week', 'last year'). Phrases like 'current' or 'most recent' are fine. It must not contain "
    "or give away the answer.\n"
    "2. a_current is the correct short answer to q_current given the event.\n"
    "3. q_dated asks about the event itself and names its month and year explicitly (for example 'in March 2026'), "
    "without giving away the answer. a_dated is its short answer.\n"
    "4. stale_answer is what a well-informed person would have answered to q_current just before the event (the "
    "predecessor, the previous winner, 'alive', the old figure); empty if there is no meaningful earlier answer.\n"
    "5. mcq_question is q_current or a close variant. mcq_choices are 4 short options of the same type and format, "
    "without letter prefixes, in the order A, B, C, D: exactly one correct option, the stale answer as one option "
    "if it exists, and plausible real entities of the same type that are false for this question as the others. "
    "No 'all/none of the above'. mcq_answer is the letter of the correct option.\n"
    "6. aliases are short accepted surface forms of a_current (full name, common short name); no generic words.\n"
    "7. predictability: 'high' if a well-informed person could have predicted the outcome a month earlier (a "
    "scheduled event with an expected result, an incumbent re-elected, a routine version bump), 'low' for "
    "surprises (sudden deaths, upsets, resignations), else 'medium'.\n"
    "8. Set usable to false (other fields may be empty) when the item has no crisp, checkable new fact: an update "
    "to an ongoing story, casualty counts that keep changing, statements, opinions, or facts that only make sense "
    "with the date."
)


def probegen_messages(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Messages asking a generator model for the probe JSON of one Current-events item.

    ``item`` is a record from :func:`cutoffmix.bench.sources.wikipedia.parse_ce_wikitext` (``date``,
    ``ce_category``, ``story_path``, ``text``, ``links``, ``sources``).
    """
    story = " > ".join(str(s) for s in item.get("story_path") or []) or "(none)"
    links = "; ".join(str(s) for s in (item.get("links") or [])[:20]) or "(none)"
    sources = item.get("sources") or []
    source = sources[0].get("url", "") if sources and isinstance(sources[0], Mapping) else ""
    user = (
        f"EVENT DATE: {item['date']}\n"
        f"SECTION: {item.get('ce_category') or '(unknown)'}\n"
        f"STORY: {story}\n"
        f"TEXT: {item['text']}\n"
        f"LINKED ENTITIES: {links}\n"
        f"SOURCE: {source or '(none)'}\n\n"
        "Write the probe JSON for this event."
    )
    return [{"role": "system", "content": PROBEGEN_SYSTEM}, {"role": "user", "content": user}]
