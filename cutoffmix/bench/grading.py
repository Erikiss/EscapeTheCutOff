"""Grading of probe responses (DESIGN.md §3.5, evaluation side).

Three graders, cheapest first:

- **MCQ** (:func:`grade_mcq`): deterministic letter extraction with the apoorvumang/knowledge-cutoff regex order
  (:func:`extract_letter`). Choosing the stale (pre-event) option is ``outdated``.
- **Rules** (:func:`grade_rule`): accepted surface forms (``Event.aliases``, short expected answers, a short correct
  MCQ option) and stale forms (``Event.reject_aliases``), matched accent- and case-insensitively on word
  boundaries, plus EN/DE abstain and hedge phrases. Death events (and alive controls) also use generic died/alive
  patterns. A match inside a negated, conditional or interrogative clause ("I'm not aware that he has died") does
  not count as an assertion, and a stale name right after a succession cue ("succeeded Bob Jones") does not either.
  Whenever the evidence is ambiguous the rule grader returns ``None`` so that judges decide.
- **Judges** (:func:`grade_with_judges`): two judge models, a third one on disagreement; strict JSON
  ``{"label", "hedged"}`` parsed robustly; any failure -> ``ungraded``.

Label semantics follow ``types.Grade``: a hedged stale assertion ("As of March 2026, X is alive") is
``outdated`` with ``hedged=True``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from cutoffmix.backends.base import BackendError
from cutoffmix.bench.prompts import judge_messages
from cutoffmix.types import Backend, ChatRequest, Event, Grade, ModelSpec, ProbeResult

__all__ = [
    "CONTROL_CATEGORIES",
    "extract_letter",
    "grade_mcq",
    "grade_response",
    "grade_results",
    "grade_rule",
    "grade_with_judges",
    "normalize_text",
    "parse_judge_output",
]

CONTROL_CATEGORIES: frozenset[str] = frozenset({"control_alive", "fake_event"})

# ------------------------------------------------------------------------------------------------- MCQ

# apoorvumang/knowledge-cutoff kc/grade.py, in this order: a leading letter ("B", "(C)", "D."), then
# "answer is X" / "answer: X", then the first isolated capital letter that is a valid option.
_LETTER_LEADING = re.compile(r"\s*\(?([A-D])\)?[\.\):]?\b", re.IGNORECASE)
_LETTER_ANSWER = re.compile(r"answer\s*(?:is|:)?\s*\(?([A-D])\)?", re.IGNORECASE)
_LETTER_ISOLATED = re.compile(r"\b([A-D])\b")


def extract_letter(response: str, valid: Iterable[str] = "ABCD") -> str | None:
    """The option letter chosen in ``response`` (upper case), or ``None`` if none can be found.

    Replicates the knowledge-cutoff benchmark's regex order exactly; ``valid`` restricts only the last
    (isolated-letter) rule, as in the benchmark.
    """
    if not response:
        return None
    text = response.strip()
    m = _LETTER_LEADING.match(text)
    if m:
        return m.group(1).upper()
    m = _LETTER_ANSWER.search(text)
    if m:
        return m.group(1).upper()
    allowed = {v.strip().upper() for v in valid}
    for letter in _LETTER_ISOLATED.findall(text):
        if letter.upper() in allowed:
            return letter.upper()
    return None


def _choice_letter(choice: str) -> str:
    return choice.strip()[:1].upper()


def _choice_text(choice: str) -> str:
    """``"B) Tshering Tobgay"`` -> ``"Tshering Tobgay"``."""
    return re.sub(r"^\s*\(?[A-Za-z]\s*[\)\.:]\s*", "", choice).strip()


def _stale_letters(event: Event) -> set[str]:
    """Letters of the MCQ options that state the stale (pre-event) world state."""
    stale = _compile_forms(event.reject_aliases)
    if _is_death(event):
        stale.append(_ALIVE)
    out: set[str] = set()
    for choice in event.mcq_choices:
        text = normalize_text(_choice_text(choice))
        if any(p.search(text) for p in stale) and not _NEGATION.search(text):
            out.add(_choice_letter(choice))
    return out


def grade_mcq(event: Event, response: str) -> Grade:
    """Grade a multiple-choice response deterministically.

    Empty response -> ``ungraded`` (missing data, e.g. the reasoning budget ran out); no parseable letter ->
    ``abstain``; the correct letter -> ``correct``; the stale option -> ``outdated``; any other letter ->
    ``incorrect``.
    """
    answer = event.mcq_answer.strip().upper()[:1]
    if not answer:
        raise ValueError(f"event {event.id!r} has no mcq_answer")
    if not (response or "").strip():
        return "ungraded"
    letters = [_choice_letter(c) for c in event.mcq_choices] or ["A", "B", "C", "D"]
    letter = extract_letter(response, letters)
    if letter is None:
        return "abstain"
    if letter == answer:
        return "correct"
    if letter in _stale_letters(event):
        return "outdated"
    return "incorrect"


# ------------------------------------------------------------------------------------------------- text helpers

_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'", "\u00b4": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u00ab": '"', "\u00bb": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
})


def normalize_text(text: str) -> str:
    """Accent-, case- and punctuation-normalised text: NFKD without combining marks, casefolded (``ß`` -> ``ss``),
    typographic quotes and dashes mapped to ASCII, whitespace collapsed."""
    decomposed = unicodedata.normalize("NFKD", text.translate(_PUNCT_MAP))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(stripped.casefold().split())


def _phrases(*fragments: str) -> re.Pattern[str]:
    """One case-insensitive pattern for regex fragments written over *normalised* text, on word boundaries."""
    return re.compile(r"(?<!\w)(?:" + "|".join(fragments) + r")(?!\w)")


def _compile_forms(forms: Iterable[str]) -> list[re.Pattern[str]]:
    """Word-boundary patterns for literal surface forms; spaces and hyphens inside a form match either."""
    out = []
    for form in forms:
        tokens = [t for t in re.split(r"[\s\-]+", normalize_text(form or "")) if t]
        if not tokens or len("".join(tokens)) < 2:
            continue
        out.append(re.compile(r"(?<!\w)" + r"[\s\-]+".join(re.escape(t) for t in tokens) + r"(?!\w)"))
    return out


# All cue patterns are written over normalize_text() output: lower case, no accents, "ß" -> "ss", ASCII quotes.
_DIED = _phrases(r"died", r"has died", r"passed away", r"dead", r"death", r"deceased", r"killed",
                 r"gestorben", r"verstorben", r"starb", r"ist tot", r"tod", r"ums leben gekommen")
_ALIVE = _phrases(r"alive", r"still living", r"is living", r"lebt", r"lebt noch", r"am leben")
# A match in a clause with one of these is not an assertion ("not aware that he died", "if he died", "hoax").
_NEGATION = re.compile(
    r"(?<!\w)(?:not|no|never|neither|nor|none|nothing|without|cannot|if|whether|unless|would|could|might|"
    r"rumou?rs?|rumou?red|hoax\w*|false|falsely|fake|unconfirmed|unverified|denied|nicht|kein\w*|nie|niemals|ob|"
    r"falls|wenn|geruchte?|falschmeldung)(?!\w)|n't(?!\w)")
# A stale name right after these is history, not the current state ("Alice succeeded Bob Jones").
_SUCCESSION_BEFORE = re.compile(
    r"(?<!\w)(?:succeed\w*|replac\w*|after|following|former\w*|previous\w*|predecessor\w*|outgoing|ousted|ex|"
    r"nachfolge\w*|folgte|vorganger\w*|ehemalig\w*|abgelost|ersetzt\w*|loste)(?!\w)")
_SUCCESSION_AFTER = re.compile(
    r"^\W*(?:\w+\W+){0,3}?(?:stepped down|resigned|left|was replaced|was succeeded|was ousted|was removed|lost|"
    r"died|retired|trat zuruck|zurucktrat|abgelost|ersetzt)(?!\w)")
_HEDGE = _phrases(
    r"as of", r"as far as i know", r"as far as i'?m aware", r"as far as i am aware", r"to my knowledge",
    r"to the best of my knowledge", r"according to my (?:latest |last |most recent )?(?:information|knowledge|data)",
    r"based on my (?:latest |last |most recent )?(?:information|knowledge|training|data)",
    r"my (?:last|latest|most recent) (?:knowledge |training )?update", r"knowledge cut-?off",
    r"at the time of my (?:last )?(?:training|update)", r"last i (?:knew|heard|checked)",
    r"stand meines wissens", r"meines wissens", r"soweit ich weiss",
    r"nach meinem (?:letzten )?(?:wissensstand|kenntnisstand|stand)", r"nach meinen (?:letzten )?informationen",
    r"zum zeitpunkt meines (?:letzten )?(?:trainings|wissensstands)")
_ABSTAIN = _phrases(
    r"i (?:don'?t|do not) know", r"(?:i'?m|i am) not (?:sure|certain)", r"not sure", r"(?:i'?m|i am) unsure",
    r"(?:cannot|can'?t|can not|unable to) (?:confirm|verify|determine|say|tell|know|provide)",
    r"(?:i'?m|i am) not aware", r"not aware of",
    r"no (?:reliable |current |up-to-date |recent |confirmed |real-time )?information",
    r"(?:don'?t|do not) have (?:any )?(?:reliable |current |up-to-date |recent |real-time )?"
    r"(?:information|data|access|knowledge)",
    r"no access to (?:real-time|current|live)", r"(?:beyond|after|past) my (?:knowledge|training) (?:cut-?off|date)",
    r"as of my (?:knowledge|training) cut-?off,? i (?:cannot|can'?t|do not|don'?t)", r"no record of",
    r"have no record", r"ich weiss (?:es )?nicht", r"weiss ich nicht", r"nicht sicher",
    r"keine (?:aktuellen |verlasslichen |gesicherten |weiteren )?informationen",
    r"kann ich (?:leider )?nicht (?:sagen|bestatigen|beurteilen|verifizieren)", r"(?:ist|sind) mir nicht bekannt",
    r"liegen mir keine", r"habe ich keine (?:informationen|kenntnis)", r"keine kenntnis")
# A guess or a contrast after an abstain phrase means the response still asserts something: let judges decide.
_GUESS_OR_CONTRAST = _phrases(
    r"best guess", r"my guess", r"i (?:think|believe|suspect)", r"probably", r"likely", r"possibly", r"presumably",
    r"but", r"however", r"vermutlich", r"wahrscheinlich", r"moglicherweise", r"ich (?:glaube|vermute|denke)",
    r"aber", r"jedoch", r"allerdings")
_YEAR = re.compile(r"(?<!\d)(?:19|20)\d\d(?!\d)")
_MONTHS: dict[str, int] = {
    **{m: i for i, m in enumerate(("january", "february", "march", "april", "may", "june", "july", "august",
                                   "september", "october", "november", "december"), 1)},
    **{m: i for i, m in enumerate(("januar", "februar", "marz", "april", "mai", "juni", "juli", "august",
                                   "september", "oktober", "november", "dezember"), 1)},
}
# month names; "may" only when a day or year follows ("may 26", "may 2024"), since it is also a modal verb
_MONTH = re.compile(r"(?<!\w)(" + "|".join(m for m in _MONTHS if m != "may") + r"|may(?=\s+\d))(?!\w)")
# sentence ends, but not after initials ("m. emmet walsh", "u.s.") or common abbreviations ("dr.", "inc.")
_SENTENCE_SPLIT = re.compile(
    r"(?<=[.!?])(?<!\b[a-z]\.)(?<!\b(?:mr|dr|st|jr|sr|ms|vs|no)\.)(?<!\b(?:mrs|inc|ltd|gen|gov|sen|rep|col|etc)\.)"
    r"\s+|\n+")
# Negation scope ends at a semicolon or a contrastive conjunction ("I'm not sure, but he died in 2024").
_SEGMENT_SPLIT = re.compile(
    r";|\s(?=(?:but|however|although|though|whereas|aber|jedoch|obwohl|allerdings)\s)")
_LEADING_PARTICLE = re.compile(r"^\W*(?:yes|no|nope|ja|nein)\s*[,.!:;-]+\s*")  # "No, he died" asserts "he died"
_COMMA_CLAUSE_END = re.compile(r",\s*(?!\d)|\s-\s|[()]")
_GENERIC_ANSWERS = frozenset({"yes", "no", "none", "none of the above", "all of the above", "true", "false",
                              "unknown", "ja", "nein"})
_NAME_HEAD_SPLIT = re.compile(r",|;|\(|\s[-\u2013\u2014]\s|:")
_NAME_CONNECTORS = frozenset({"de", "del", "der", "van", "von", "da", "di", "du", "la", "le", "al", "bin", "ibn",
                              "of", "and", "y", "e"})
_MAX_NAME_WORDS = 5
_CAPITALISED_RUN = re.compile(r"\b[A-Z\u00c0-\u00de][\w'\u2019\-]+(?:\s+[A-Z\u00c0-\u00de][\w'\u2019\-]+)+")
_COMMON_CAPITALISED = frozenset({
    "i", "as", "the", "my", "it", "he", "she", "they", "this", "that", "however", "based", "according", "since",
    "after", "before", "in", "on", "at", "yes", "no", "if", "but", "and", "or", "there", "while",
    "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november",
    "december", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "ai", "ceo",
})


def _is_control(event: Event) -> bool:
    return event.control or event.category in CONTROL_CATEGORIES


def _is_death(event: Event) -> bool:
    return event.category == "death" and not _is_control(event)


def _name_head(text: str) -> str | None:
    """The leading name of an answer ("Sanae Takaichi, Japan's first female prime minister." -> "Sanae
    Takaichi"), or ``None`` if the head is not a short capitalised name or title."""
    head = _NAME_HEAD_SPLIT.split((text or "").strip(), maxsplit=1)[0].strip().rstrip(".!")
    words = head.split()
    if not words or len(words) > _MAX_NAME_WORDS or normalize_text(head) in _GENERIC_ANSWERS:
        return None
    if words[0].lower() in {"he", "she", "it", "they", "yes", "no", "the", "a", "an"}:
        return None
    if not all(w[:1].isupper() or w[:1].isdigit() or w.lower() in _NAME_CONNECTORS for w in words):
        return None
    return head if re.search(r"[^\W\d_]", head) else None


def _outcome_patterns(event: Event, probe: str) -> dict[str, list[re.Pattern[str]]]:
    """Patterns per outcome: ``correct`` (current state), ``outdated`` (stale state), ``incorrect`` (other wrong).

    Accepted forms: ``aliases`` plus, for events that are not about a death or a control, the leading name of the
    expected answer and of the correct MCQ option. Death events and alive controls add generic died/alive patterns.
    """
    accepted = list(event.aliases)
    if not _is_death(event) and not _is_control(event):
        expected = (event.expected_recall if probe == "recall" else "") or event.expected_current
        accepted += [h for h in (_name_head(expected), _name_head(_choice_text(_correct_choice(event)))) if h]
    patterns: dict[str, list[re.Pattern[str]]] = {
        "correct": _compile_forms(accepted), "outdated": _compile_forms(event.reject_aliases), "incorrect": []}
    if _is_death(event):
        patterns["correct"].append(_DIED)
        patterns["outdated"].append(_ALIVE)
    elif event.category in CONTROL_CATEGORIES:  # the person is alive; "died" confirms a fake or wrong report
        patterns["correct"].append(_ALIVE)
        patterns["incorrect"].append(_DIED)
    return patterns


def _correct_choice(event: Event) -> str:
    answer = event.mcq_answer.strip().upper()[:1]
    return next((c for c in event.mcq_choices if _choice_letter(c) == answer), "")


def _segments(text: str) -> list[tuple[str, bool]]:
    """``(segment, asserted)`` pairs; segments of questions are never assertions."""
    out = []
    for sentence in _SENTENCE_SPLIT.split(text):
        question = sentence.rstrip().endswith("?")
        for segment in _SEGMENT_SPLIT.split(sentence):
            segment = _LEADING_PARTICLE.sub("", segment.strip())
            if segment:
                out.append((segment, not question))
    return out


def _negated(segment: str, start: int, end: int) -> bool:
    """A negation/condition cue earlier in the segment ("cannot verify if X, the CEO, has died") or later in the
    same comma clause ("Bob Jones is no longer prime minister") turns a match into a non-assertion."""
    rest = segment[end:]
    clause_end = _COMMA_CLAUSE_END.search(rest)
    return bool(_NEGATION.search(segment[:start]) or _NEGATION.search(rest[: clause_end.start() if clause_end
                                                                            else len(rest)]))


def _hits(text: str, patterns: Mapping[str, Sequence[re.Pattern[str]]],
          event_date: str) -> tuple[set[str], set[str]]:
    """``(strong, weak)`` outcome sets: strong = asserted, weak = mentioned but negated, conditional,
    interrogative, (for stale forms) next to a succession cue, or (for the generic "died" pattern) dated to a
    different year or month than ``event_date`` (ISO)."""
    strong: set[str] = set()
    weak: set[str] = set()
    death_years: set[str] = set()  # years and months stated next to an asserted death
    death_months: set[int] = set()
    for segment, asserted in _segments(text):
        for outcome, pats in patterns.items():
            for pat in pats:
                for m in pat.finditer(segment):
                    plain = asserted and not _negated(segment, m.start(), m.end())
                    if plain and outcome == "outdated":
                        before = " ".join(segment[: m.start()].split()[-5:])
                        plain = not (_SUCCESSION_BEFORE.search(before) or _SUCCESSION_AFTER.search(segment[m.end():]))
                    if plain and pat is _DIED and outcome == "correct":
                        death_years.update(_YEAR.findall(segment))
                        death_months.update(_MONTHS[x] for x in _MONTH.findall(segment))
                    (strong if plain else weak).add(outcome)
    year, month = event_date[:4], int(event_date[5:7])
    wrong_year = bool(death_years) and year not in death_years
    wrong_month = bool(death_months) and month not in death_months
    alias_match = any(p is not _DIED and p.search(text) for p in patterns["correct"])
    if "correct" in strong and (wrong_year or wrong_month) and _DIED in patterns["correct"] and not alias_match:
        # the response dates the death differently: a different claim unless an alias matched
        strong.discard("correct")
        weak.add("correct")
    return strong, weak


def _names_someone_new(response: str, event: Event) -> bool:
    """True if the response contains a capitalised multi-word name whose words are not in the question/subject
    (e.g. it names an office holder we have no alias for), i.e. it is more than a plain refusal."""
    known = set(normalize_text(" ".join([event.question_current, event.question_recall, event.subject])).split())
    for run in _CAPITALISED_RUN.findall(response):
        words = [w for w in normalize_text(run).split() if w not in _COMMON_CAPITALISED and len(w) > 2]
        if any(w not in known for w in words):
            return True
    return False


def grade_rule(event: Event, probe: str, response: str) -> tuple[Grade, bool] | None:
    """Rule-based grade ``(grade, hedged)`` for one response, or ``None`` when the rules are unsure.

    - empty response -> ``("ungraded", False)``; ``probe == "mcq"`` -> :func:`grade_mcq`.
    - exactly one outcome asserted (current / stale / other wrong form outside a negated, conditional or
      interrogative context) -> that outcome; other outcomes may only be mentioned in such contexts or as history
      ("Alice, who succeeded Bob"). A death dated to another month or year than the event is not asserted, and a
      current *name* next to an abstain phrase ("not aware of an announcement ... names like X") is left to judges.
    - nothing asserted or mentioned, an abstain phrase, no guess/contrast and no unknown name -> ``("abstain",
      False)``.
    - anything else (conflicting assertions, guesses, names we cannot place) -> ``None``.

    ``hedged`` is set when an asserted answer comes with a hedge ("as of", "as far as I know", "Stand meines
    Wissens", ...) or an abstain phrase.
    """
    if not (response or "").strip():
        return "ungraded", False
    if probe == "mcq":
        return grade_mcq(event, response), False
    text = normalize_text(response)
    strong, weak = _hits(text, _outcome_patterns(event, probe), event.date)
    abstain = bool(_ABSTAIN.search(text))
    if len(strong) == 1:
        outcome = cast(Grade, next(iter(strong)))
        if outcome == "correct" and abstain and not _is_death(event) and not _is_control(event):
            return None  # a current name next to "I'm not aware of / cannot verify" is usually speculation
        return outcome, bool(_HEDGE.search(text)) or abstain
    if (not strong and not weak and abstain and not _GUESS_OR_CONTRAST.search(text)
            and not _names_someone_new(response, event)):
        return "abstain", False
    return None


# ------------------------------------------------------------------------------------------------- judges

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_LABEL_FIELD = re.compile(r"""["']?label["']?\s*[:=]\s*["']?([A-Za-z_\-]+)""", re.IGNORECASE)
_HEDGED_FIELD = re.compile(r"""["']?hedged["']?\s*[:=]\s*["']?(true|false|yes|no|1|0)""", re.IGNORECASE)
_LABEL_SYNONYMS: dict[str, str] = {
    "correct": "correct", "current": "correct",
    "outdated": "outdated", "stale": "outdated", "out_of_date": "outdated",
    "incorrect": "incorrect", "wrong": "incorrect", "other_wrong": "incorrect", "hallucinated": "incorrect",
    "abstain": "abstain", "abstained": "abstain", "abstention": "abstain", "refused": "abstain",
}
_JUDGE_RESPONSE_FORMAT: dict[str, Any] = {"type": "json_object"}
_UNSUPPORTED_FORMAT_STATUS = frozenset({400, 415, 422})


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _label(value: Any) -> Grade | None:
    key = re.sub(r"[\s\-]+", "_", str(value or "").strip().lower())
    return cast(Grade, _LABEL_SYNONYMS[key]) if key in _LABEL_SYNONYMS else None


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_judge_output(text: str) -> tuple[Grade, bool] | None:
    """Parse a judge reply into ``(label, hedged)``; ``None`` if no valid label can be found.

    Accepts strict JSON, JSON inside code fences or surrounding prose, ``label: x`` key/value text, or a bare label.
    Label synonyms (``current`` -> correct, ``stale`` -> outdated, ``other_wrong`` -> incorrect) are mapped.
    """
    if not text or not text.strip():
        return None
    candidates = [m.group(1) for m in _JSON_FENCE.finditer(text)] + [text]
    for candidate in candidates:
        for obj in _json_objects(candidate):
            label = _label(obj.get("label"))
            if label is not None:
                return label, _as_bool(obj.get("hedged", False))
    m = _LABEL_FIELD.search(text)
    if m and (label := _label(m.group(1))) is not None:
        h = _HEDGED_FIELD.search(text)
        return label, bool(h and _as_bool(h.group(1)))
    bare = _label(text.strip().strip(".\"'`*"))
    return (bare, False) if bare is not None else None


async def _ask_judge(backend: Backend, judge: ModelSpec, messages: list[dict[str, Any]],
                     max_tokens: int) -> tuple[Grade, bool] | None:
    """One judge vote, or ``None`` on any failure. Retries once without ``response_format`` if the request with it
    is rejected as invalid (the model or provider does not support JSON mode)."""
    req = ChatRequest(messages=messages, max_tokens=max_tokens, temperature=0.0,
                      response_format=dict(_JUDGE_RESPONSE_FORMAT), metadata={"purpose": "judge"})
    try:
        try:
            result = await backend.chat(judge, req)
        except BackendError as e:
            if e.status not in _UNSUPPORTED_FORMAT_STATUS:
                raise
            result = await backend.chat(judge, dataclasses.replace(req, response_format=None))
    except Exception:  # a failed judge is a missing vote, never a crash of the grading run
        return None
    return parse_judge_output(result.text)


def _majority(votes: Sequence[tuple[str, tuple[Grade, bool] | None]]) -> tuple[Grade, bool, list[str]] | None:
    """The label with at least two votes (or the only vote when exactly one judge was consulted)."""
    valid = [(key, v) for key, v in votes if v is not None]
    if len(votes) == 1 and valid:
        key, (label, hedged) = valid[0]
        return label, hedged, [key]
    counts: dict[Grade, list[tuple[str, bool]]] = {}
    for key, (label, hedged) in valid:
        counts.setdefault(label, []).append((key, hedged))
    for label, voters in counts.items():
        if len(voters) >= 2:
            return label, any(h for _, h in voters), [k for k, _ in voters]
    return None


async def grade_with_judges(
    backend: Backend,
    judges: Sequence[ModelSpec],
    event: Event,
    probe: str,
    response: str,
    *,
    max_tokens: int = 1024,
) -> tuple[Grade, bool, str]:
    """Grade with LLM judges: ``(grade, hedged, grader)``, grader = ``"judge:<key>[+<key>...]"``.

    The first two judges vote concurrently; if they disagree (or one fails) and a third judge is given, it breaks
    the tie. A label needs two votes (one if only one judge is given); otherwise, and on any judge failure that
    leaves no majority, the grade is ``ungraded``. ``hedged`` is true if any judge of the winning label said so.
    Prompts come from ``cutoffmix.bench.prompts.judge_messages``. ``max_tokens`` includes hidden reasoning.
    """
    if not judges:
        raise ValueError("grade_with_judges needs at least one judge model")
    if not (response or "").strip():
        return "ungraded", False, "judge"
    messages = judge_messages(event, probe, response)
    first = list(judges[:2])
    answers = await asyncio.gather(*(_ask_judge(backend, j, messages, max_tokens) for j in first))
    votes = list(zip((j.key for j in first), answers, strict=True))
    decided = _majority(votes)
    if decided is None and len(judges) >= 3:
        third = judges[2]
        votes.append((third.key, await _ask_judge(backend, third, messages, max_tokens)))
        decided = _majority(votes)
    if decided is None:
        return "ungraded", False, "judge:" + "+".join(k for k, _ in votes)
    label, hedged, voters = decided
    return label, hedged, "judge:" + "+".join(voters)


# ------------------------------------------------------------------------------------------------- pipeline


async def grade_response(
    event: Event,
    probe: str,
    response: str,
    *,
    backend: Backend | None = None,
    judges: Sequence[ModelSpec] | None = None,
) -> tuple[Grade, bool, str]:
    """Full grading pipeline for one response: ``(grade, hedged, grader)``.

    MCQ -> :func:`grade_mcq` (grader ``"mcq"``); open probes -> :func:`grade_rule` (``"rule"``); when the rules are
    unsure -> :func:`grade_with_judges` if ``backend`` and ``judges`` are given, else ``("ungraded", False,
    "rule")`` so that a later pass (:func:`grade_results`) can judge it.
    """
    if probe == "mcq":
        return grade_mcq(event, response), False, "mcq"
    ruled = grade_rule(event, probe, response)
    if ruled is not None:
        return ruled[0], ruled[1], "rule"
    if backend is not None and judges:
        return await grade_with_judges(backend, judges, event, probe, response)
    return "ungraded", False, "rule"


async def grade_results(
    results: Sequence[ProbeResult],
    events: Sequence[Event] | Mapping[str, Event],
    *,
    backend: Backend | None = None,
    judges: Sequence[ModelSpec] | None = None,
    concurrency: int = 8,
    regrade_all: bool = False,
) -> list[ProbeResult]:
    """Re-grade stored results with :func:`grade_response` and return new records (inputs are not modified).

    By default only rows that are ``ungraded`` although a response exists (rules were unsure, or no judge was
    available at run time) are re-graded; ``regrade_all=True`` re-grades every row with a response. Rows with an
    API ``error`` or an event missing from ``events`` are returned unchanged. ``concurrency`` bounds judge calls.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    by_id = dict(events) if isinstance(events, Mapping) else {e.id: e for e in events}
    sem = asyncio.Semaphore(concurrency)

    async def one(r: ProbeResult) -> ProbeResult:
        event = by_id.get(r.event_id)
        if r.error or event is None or not r.response.strip() or not (regrade_all or r.grade == "ungraded"):
            return r
        async with sem:
            grade, hedged, grader = await grade_response(event, r.probe, r.response, backend=backend, judges=judges)
        return dataclasses.replace(r, grade=grade, hedged=hedged, grader=grader)

    return list(await asyncio.gather(*(one(r) for r in results)))
