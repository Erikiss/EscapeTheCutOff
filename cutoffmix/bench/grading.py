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
import difflib
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
    "PAPER_CATEGORIES",
    "extract_letter",
    "grade_mcq",
    "grade_paper",
    "grade_response",
    "grade_results",
    "grade_rule",
    "grade_with_judges",
    "normalize_text",
    "parse_judge_output",
]

CONTROL_CATEGORIES: frozenset[str] = frozenset({"control_alive", "fake_event", "fake_paper"})
# arXiv first-author probes (bench/sources/papers.py): graded by :func:`grade_paper`
PAPER_CATEGORIES: frozenset[str] = frozenset({"paper", "fake_paper"})

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
    """Letters of the MCQ options that state the stale (pre-event) world state. Paper probes have no stale state:
    their ``reject_aliases`` are distractor authors, so a distractor pick is plainly ``incorrect``."""
    if event.category in PAPER_CATEGORIES:
        return set()
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


def _compile_forms(forms: Iterable[str], *, loose: bool = False) -> list[re.Pattern[str]]:
    """Word-boundary patterns for literal surface forms; spaces and hyphens inside a form match either.

    ``loose=True`` (person names) also lets a space or hyphen appear, or disappear, between any two characters of
    a token, so that ``Junsol Kim`` matches "Jun-sol Kim", "Jun Sol Kim" and "Junsol Kim" alike (romanisations
    of the same name differ in exactly this way) and ``Byung-Kwan Lee`` matches "Byungkwan Lee"."""
    out = []
    for form in forms:
        tokens = [t for t in re.split(r"[\s\-]+", normalize_text(form or "")) if t]
        if not tokens or len("".join(tokens)) < 2:
            continue
        if loose:
            parts = [r"[\s\-]?".join(re.escape(ch) for ch in t) for t in tokens]
            out.append(re.compile(r"(?<!\w)" + r"[\s\-]*".join(parts) + r"(?!\w)"))
        else:
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
# The same cues without "after"/"following", for current (correct) names: "her predecessor was X" or "Y replaced
# X" makes X history; "after the election, X became ..." does not.
_SUCCESSION_BEFORE_STRICT = re.compile(
    r"(?<!\w)(?:succeed\w*|replac\w*|former\w*|previous\w*|predecessor\w*|outgoing|ousted|ex|"
    r"nachfolge\w*|folgte|vorganger\w*|ehemalig\w*|abgelost|ersetzt\w*|loste)(?!\w)")
_BY_END = re.compile(r"(?<!\w)(?:by|durch|von)\W*$")
# Any succession cue anywhere in a response ("succeeding", "replaced", "predecessor", "took over from").
_SUCCESSION_ANY = re.compile(
    r"(?<!\w)(?:succeed\w*|replac\w*|predecessor\w*|took over from|nachfolge\w*|vorganger\w*|abgelost|"
    r"ersetzt\w*)(?!\w)")
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
# "I don't know", "no information", "cannot verify": the response disclaims knowledge.
_ABSTAIN_FRAGMENTS: tuple[str, ...] = (
    r"i (?:don'?t|do not) know",
    r"(?:cannot|can'?t|can not|unable to) (?:confirm|verify|determine|say|tell|know|provide)",
    r"(?:i'?m|i am) not aware", r"not aware of",
    r"no (?:reliable |current |up-to-date |recent |confirmed |real-time )?information",
    r"(?:don'?t|do not) have (?:any )?(?:reliable |current |up-to-date |recent |real-time )?"
    r"(?:information|data|access|knowledge)",
    r"no access to (?:real-time|current|live)", r"(?:beyond|after|past) my (?:knowledge|training) (?:cut-?off|date)",
    r"as of my (?:knowledge|training) cut-?off,? i (?:cannot|can'?t|do not|don'?t)", r"no record of",
    r"have no record", r"ich weiss (?:es )?nicht", r"weiss ich nicht",
    r"keine (?:aktuellen |verlasslichen |gesicherten |weiteren )?informationen",
    r"kann ich (?:leider )?nicht (?:sagen|bestatigen|beurteilen|verifizieren)", r"(?:ist|sind) mir nicht bekannt",
    r"liegen mir keine", r"habe ich keine (?:informationen|kenntnis)", r"keine kenntnis")
# "I'm not sure", "not certain": uncertainty about an answer the response may still give.
_UNSURE_FRAGMENTS: tuple[str, ...] = (
    r"(?:i'?m|i am) not (?:sure|certain)", r"not sure", r"(?:i'?m|i am) unsure", r"nicht sicher")
_ABSTAIN = _phrases(*_ABSTAIN_FRAGMENTS, *_UNSURE_FRAGMENTS)
_DISCLAIMS = _phrases(*_ABSTAIN_FRAGMENTS)
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
                    if plain and outcome in ("outdated", "correct") and pat not in (_DIED, _ALIVE):
                        # a name in succession context is history ("Y, who replaced X"; "her predecessor was X"),
                        # not the current state; for current names "after"/"following" do not count as cues
                        before = " ".join(segment[: m.start()].split()[-5:])
                        if outcome == "outdated":
                            history = bool(_SUCCESSION_BEFORE.search(before))
                        else:  # "Bob was replaced by Alice" names Alice as the successor, not as history
                            history = bool(_SUCCESSION_BEFORE_STRICT.search(before)) and not _BY_END.search(before)
                        plain = not (history or _SUCCESSION_AFTER.search(segment[m.end():]))
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


def _names_someone_new(response: str, event: Event, extra: Iterable[str] = ()) -> bool:
    """True if the response contains a capitalised multi-word name whose words are not in the question/subject
    (or in ``extra``, e.g. the known answer forms), e.g. it names an office holder we have no alias for."""
    known = set(normalize_text(" ".join([event.question_current, event.question_recall, event.subject,
                                         *extra])).split())
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
    if event.category in PAPER_CATEGORIES:
        return grade_paper(event, probe, response)
    text = normalize_text(response)
    strong, weak = _hits(text, _outcome_patterns(event, probe), event.date)
    abstain = bool(_ABSTAIN.search(text))
    if len(strong) == 1:
        outcome = cast(Grade, next(iter(strong)))
        if outcome == "correct" and abstain and not _is_death(event) and not _is_control(event):
            return None  # a current name next to "I'm not aware of / cannot verify" is usually speculation
        if (outcome == "correct" and _SUCCESSION_ANY.search(text) and not _is_death(event)
                and not _is_control(event)
                and _names_someone_new(response, event, [*event.aliases, *event.reject_aliases])):
            return None  # the expected name next to an unknown name and a succession cue: who succeeded whom?
        return outcome, bool(_HEDGE.search(text)) or abstain
    if (not strong and not weak and abstain and not _GUESS_OR_CONTRAST.search(text)
            and not _names_someone_new(response, event)):
        return "abstain", False
    return None


# ------------------------------------------------------------------------------------------------- papers

# Extra abstain cues for "who is the first author of <paper>" probes (over normalised text; _ABSTAIN also applies).
_PAPER_ABSTAIN = _phrases(
    r"(?:i'?m|i am) not aware of", r"(?:i'?m|i am) not familiar with", r"not familiar with",
    r"(?:don'?t|do not|cannot|can'?t) (?:recognize|recognise|recall|identify|place) (?:this|that|the|a|any|it)",
    r"(?:don'?t|do not) have (?:any )?(?:specific |reliable |detailed |verified )?(?:information|details|knowledge|data)",
    r"(?:cannot|can'?t|can not|could not|couldn'?t|unable to|not able to) (?:verify|confirm|find|locate|identify|"
    r"recall|determine|access)",
    r"no record", r"(?:not|never) (?:seen|encountered|come across)", r"(?:does not|doesn'?t|may not|might not) exist",
    r"(?:no|not) (?:such|any) (?:paper|record|publication|preprint)", r"not in my (?:training )?data",
    r"(?:don'?t|do not) (?:know|have) (?:this|that|the) (?:paper|publication|preprint)",
    r"(?:there is|there'?s|is|are|found|find|have|has) no (?:\w+ ){0,2}(?:paper|publication|preprint|record|arxiv|"
    r"entry|listing|match)",
    r"no idea", r"(?:can'?t|cannot|can not|unable to) help", r"^\W*unknown\W*$",
    r"ist mir nicht bekannt", r"kenne ich nicht", r"ich kenne (?:\w+ ){0,3}nicht", r"nicht bekannt", r"unbekannt",
    r"kann ich (?:leider )?nicht (?:verifizieren|bestatigen|uberprufen|nachprufen|finden|zuordnen|nennen)",
    r"ich kann (?:\w+ ){0,4}(?:leider )?nicht (?:sagen|nennen|bestatigen|verifizieren|beantworten|zuordnen|finden|"
    r"uberprufen|nachprufen)",
    r"liegen mir (?:keine|nicht) (?:informationen|daten|angaben)?", r"keine (?:informationen|angaben|kenntnis|daten) "
    r"(?:zu|uber|dazu|daruber)", r"(?:gibt es|existiert) (?:nicht|kein\w*)", r"nicht finden", r"kein(?:e|en)? (?:solche|"
    r"derartige)", r"keine ahnung",
)
# Uncertainty words that make an asserted name "hedged".
_UNCERTAIN = _phrases(
    r"i (?:think|believe|suspect|recall|guess|assume)", r"if i recall", r"(?:i'?m|i am) not (?:entirely |completely |"
    r"fully |100% )?(?:sure|certain|confident)", r"not (?:entirely |completely |fully |100% )?(?:sure|certain)",
    r"probably", r"likely", r"possibly", r"perhaps", r"maybe", r"might be", r"may be", r"could be", r"it seems",
    r"seems to be", r"appears to be", r"best guess", r"my guess", r"(?:i'?m|i am) uncertain", r"uncertain",
    r"(?:may|might|could) (?:have been|be) (?:mistaken|wrong|misremembering)", r"vermutlich", r"wahrscheinlich",
    r"moglicherweise", r"vielleicht", r"ich (?:glaube|vermute|denke|meine|schatze)", r"konnte", r"unsicher",
    r"nicht sicher", r"soweit ich (?:mich erinnere|weiss)",
)
# Surnames too common to accept alone (the given name's first token is required too).
_COMMON_SURNAMES = frozenset({
    "li", "wang", "zhang", "chen", "liu", "yang", "kim", "lee", "park", "singh", "kumar", "smith", "huang", "zhao",
    "zhou", "zhu", "yuan", "nguyen", "patel", "gupta", "sharma", "jones", "brown", "williams", "johnson",
    "garcia", "martin", "muller", "schmidt", "cheng", "zheng", "jiang", "tang", "wei", "shen", "song",
})
# Capitalised words that never start or make up a person's name in a first-author answer.
_NAME_STOP = frozenset({
    *_COMMON_CAPITALISED, "first", "author", "authors", "arxiv", "paper", "papers", "title", "titled", "unfortunately",
    "note", "sorry", "please", "unknown", "university", "institute", "research", "lab", "labs", "google", "deepmind",
    "openai", "microsoft", "meta", "anthropic", "nvidia", "stanford", "berkeley", "mit", "cmu", "hugging", "face",
    "team", "llm", "llms", "model", "models", "language", "learning", "transformer", "transformers", "neural",
    "network", "networks", "deep", "machine", "large", "data", "dataset", "benchmark", "new", "york", "san", "francisco",
    "china", "united", "states", "european", "kingdom", "dr", "prof", "mr", "ms", "mrs", "et", "al", "der", "die",
    "das", "ein", "eine", "mein", "meine", "meinem", "ich", "es", "leider", "dieses", "diese", "erstautor",
    "erstautorin", "autor", "autorin", "wissensstand", "kenntnisstand", "informationen", "kenntnis", "daten",
    "training", "trainingsdaten", "stand", "zeitpunkt", "veroffentlichung", "titel", "arbeit", "artikel", "modell",
    "sprachmodell", "forschung", "universitat", "institut", "antwort", "frage", "hinweis", "quelle", "quellen",
    "is", "an", "a", "of", "to", "by", "for", "with", "was", "are", "be", "not", "so", "we", "do", "did", "up",
    "using", "also", "indeed", "actually", "additionally", "specifically", "i'm", "i've", "i'd", "it's", "that's",
    "there's", "here", "then", "thus", "hence", "therefore", "though", "although", "because", "yet", "when", "where",
    "which", "who", "whose", "what", "why", "how", "can", "could", "would", "should", "will", "might", "must", "let",
    "see", "ok", "okay", "hi", "hello", "thanks", "thank", "you", "your", "our", "me", "him", "her", "them", "its",
    "their", "his", "one", "two", "three", "some", "any", "all", "none", "each", "every", "both", "such", "same",
    "other", "another", "more", "most", "less", "least", "many", "much", "few", "several", "very", "too", "quite",
    "rather", "only", "just", "even", "still", "already", "again", "once", "now", "today", "recently", "currently",
    "finally", "second", "third", "last", "next", "previous", "well", "perhaps", "maybe", "probably", "likely",
    "possibly", "certainly", "definitely", "apparently", "presumably", "supposedly", "reportedly", "allegedly",
    "generally", "typically", "usually", "often", "sometimes", "always", "never", "written", "led", "lead",
    "corresponding", "senior", "co-author", "coauthor", "co-authors", "coauthors", "colleagues", "collaborators",
    # services, venues, fields and knowledge-horizon words that an abstention mentions in capitals
    "scholar", "semantic", "openreview", "github", "neurips", "icml", "iclr", "cvpr", "emnlp", "acl", "aaai",
    "proceedings", "conference", "workshop", "journal", "preprint", "publication", "abstract", "version", "search",
    "bing", "internet", "online", "database", "dblp", "pubmed", "crossref", "doi", "citation", "citations",
    "references", "wikipedia", "interpretability", "mechanistic", "knowledge", "cutoff", "cut-off", "reasoning",
    "evaluation", "agent", "agents", "vision", "video", "image", "generation", "diffusion", "reinforcement",
    "attention", "memory", "context", "token", "tokens", "survey", "science", "scientific", "apologies", "apologize",
    "apologise", "laboratory", "department", "school", "college", "faculty", "center", "centre", "academy",
    "sciences", "national", "international", "technical", "technology", "polytechnic", "corporation", "company",
    "foundation", "inc", "ltd", "llc", "ibm", "amazon", "tencent", "alibaba", "baidu", "bytedance", "huawei",
    "salesforce", "adobe", "intel", "qualcomm", "samsung", "naver", "cohere", "mistral", "xai", "deepseek", "qwen",
    "kimi", "moonshot", "zhipu", "minimax", "tsinghua", "zhejiang", "peking", "shanghai", "beijing", "kaist", "epfl",
    "eth", "mila", "meines", "wissens", "wissen", "entschuldigung", "bedauere", "suche", "datenbank", "keine",
    "kein", "ahnung", "nichts", "sagt", "mir", "diesem", "dieser", "dazu", "davon", "hierzu",
})
_INITIAL = re.compile(r"^[A-Z]\.$")
_TOKEN_TRIM = "()[]{}\"'“”‘’*_,;:!?"
_NAME_PARTICLES = frozenset({"van", "von", "de", "der", "den", "da", "di", "du", "la", "le", "del", "bin", "ibn"})
# stop words that are also surnames: inside a run after a name-like token ("Feng He", "Long May") they are names
_SURNAME_HOMOGRAPHS = frozenset({"he", "she", "an", "will", "may", "so", "do", "no", "see", "well", "long", "young",
                                 "new"})
_POSSESSIVE = re.compile(r"['’]s$")
# Markdown emphasis around an answer ("**Sho Maeoki**", "*Sho Maeoki*", "`Sho Maeoki`", "__Sho Maeoki__"): the
# markers are not part of the name and must not stick to it ("Maeoki**." is "Maeoki.").
_MARKDOWN = re.compile(r"\*+|`+|(?<!\w)_+(?=\w)|(?<=\w)_+(?!\w)")
# A token that opens with one of these starts a new unit after a name: an affiliation "(University of Tokyo)", a
# bracketed note or a quoted title.
_RUN_OPENERS = frozenset('(["“„«')
_TOKEN_ENDS = ".,;:!?)\"'”’"


def _plain(response: str) -> str:
    """``response`` without markdown emphasis markers (bold, italics, inline code)."""
    return _MARKDOWN.sub("", response)


def _name_like(token: str) -> bool:
    """A capitalised word of letters (``Zhang``, ``He``, ``Shi-Yu``, ``McCarthy``) with at least one lower-case
    letter and no more capitals than lower-case letters, no digits: not ``LLM``, ``GPT-4``, ``SAEs``."""
    if len(token) < 2 or not token[0].isupper() or any(c.isdigit() for c in token):
        return False
    if not all(c.isalpha() or c in "'’-." for c in token):
        return False
    lower = sum(c.islower() for c in token)
    return lower >= 1 and sum(c.isupper() for c in token) <= lower


def _compact(name: str) -> str:
    """``'Jun-sol Kim'`` -> ``'junsolkim'``: letters only, accents folded; the same for every romanisation."""
    return re.sub(r"[^a-z0-9]", "", normalize_text(name))


def _unknown_names(response: str, event: Event) -> list[str]:
    """Capitalised runs of two to five tokens (``Wei Chen``, ``J. Smith``, ``Jan van der Berg``) that are not an
    accepted or rejected answer form (compared letters-only, so "Jun-sol Kim" is "Junsol Kim") and whose words occur
    neither in the question or title nor in :data:`_NAME_STOP`: people we have no alias for, i.e. wrong or invented
    authors. Leading and trailing stop words of a run ("Perhaps Wei Chen") are ignored, except surname homographs
    after a name ("Feng He"); a stop word inside the run ("Hugging Face") dismisses it. Markdown emphasis around a
    name is not part of it, a middle initial is ("Bardienus P. Duisterhof"), and an opening parenthesis, bracket or
    quote ends a run: "Sho Maeoki (Tokyo Tech)" names Sho Maeoki."""
    known = set(normalize_text(f"{event.question_current} {event.question_recall} {event.subject}").split())
    known |= {w.strip(".,;:()'\"") for w in known}
    names = {_compact(a) for a in [event.expected_current, event.expected_recall, *event.aliases,
                                   *event.reject_aliases] if a}
    found: list[str] = []

    def stop(words: list[str], i: int) -> bool:
        if i > 0 and words[i] in _SURNAME_HOMOGRAPHS:
            return False
        return words[i] in _NAME_STOP

    def person(run: list[str]) -> str | None:
        words = [_POSSESSIVE.sub("", normalize_text(t).strip(".,;:()'\"")) for t in run]
        while words and stop(words, 0):
            words, run = words[1:], run[1:]
        while words and stop(words, len(words) - 1):
            words, run = words[:-1], run[:-1]
        if not 2 <= len(run) <= 5 or sum(_name_like(t) for t in run) < 2 or _compact(" ".join(words)) in names:
            return None
        if any(w in known or stop(words, i) for i, w in enumerate(words) if w):
            return None
        return " ".join(run)

    def flush(run: list[str]) -> None:
        name = person(run)
        if name is not None:
            found.append(name)

    run: list[str] = []
    for raw in _plain(response).split():
        if run and raw[:1] in _RUN_OPENERS:  # "(University of Tokyo)" or a quoted title after a name
            flush(run)
            run = []
        token = raw.strip(_TOKEN_TRIM)
        ends = raw[-1:] in _TOKEN_ENDS
        if len(token) > 2:  # a trailing period ends a word ("Maeoki).", "Maeoki**."), not an initial ("P.")
            token = token.rstrip(".").strip(_TOKEN_TRIM)
        if _name_like(token) or _INITIAL.match(token) or (run and token.casefold() in _NAME_PARTICLES):
            run.append(token)
            if ends and not _INITIAL.match(token):
                flush(run)
                run = []
        else:
            flush(run)
            run = []
    flush(run)
    return found


def _other_person(response: str, event: Event) -> bool:
    """True if the response names a person we have no alias for (see :func:`_unknown_names`)."""
    return bool(_unknown_names(response, event))


def _surname_given(name: str) -> tuple[str, str]:
    """``'Jan van der Berg'`` -> ``('Berg', 'Jan')``: last token and first token of a name."""
    toks = name.split()
    return (toks[-1] if toks else ""), (toks[0] if len(toks) >= 2 else "")


_INITIALS_ONLY = re.compile(r"^(?:[^\W\d_]\.?\s*)+$")
_NEAR_NAME_RATIO = 0.85


def _near_name(name: str, expected: str) -> bool:
    """Could ``name`` be ``expected`` misspelt or written out? True for a letters-only similarity of at least
    :data:`_NEAR_NAME_RATIO` ("Maciej Piorro" ~ "Maciej Pióro"), for the same given name with the same first four
    surname letters, and, when the expected given name is initials only ("B. Y. Yan"), for the same surname with
    initials that extend or shorten the expected ones ("Bingyi Yan", "Bo Yan")."""
    a, b = _compact(name), _compact(expected)
    if not a or not b:
        return False
    if difflib.SequenceMatcher(None, a, b).ratio() >= _NEAR_NAME_RATIO:
        return True
    surname, given = _surname_given(expected)
    n_surname, n_given = _surname_given(name)
    s, n_s = _compact(surname), _compact(n_surname)
    if given and n_given and _compact(given) == _compact(n_given) and len(s) >= 4 and s[:4] == n_s[:4]:
        return True
    if s != n_s:
        return False
    expected_given = expected[: len(expected) - len(surname)].strip()
    if expected_given and _INITIALS_ONLY.match(expected_given) and n_given:
        wanted = _compact(expected_given)
        got = "".join(t[:1] for t in _compact_tokens(name[: len(name) - len(n_surname)]))
        return bool(got) and (wanted.startswith(got) or got.startswith(wanted))
    return False


def _compact_tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s\-]+", normalize_text(text)) if re.search(r"[a-z]", t)]


_TITLE_MIN_TOKENS = 3


def _title_patterns(subject: str) -> list[re.Pattern[str]]:
    """Patterns (over normalised text) for the paper title and, when it has a colon, for its main title and its
    subtitle alone, each of at least :data:`_TITLE_MIN_TOKENS` word tokens, with any punctuation between the words
    (a model may quote "Infini-attention" as "Infini attention", or the main title only)."""
    parts = [subject, *subject.split(":", 1)] if ":" in subject else [subject]
    out = []
    for part in parts:
        tokens = re.findall(r"\w+", normalize_text(part))
        if len(tokens) >= _TITLE_MIN_TOKENS:
            out.append(re.compile(r"(?<!\w)" + r"\W+".join(map(re.escape, tokens)) + r"(?!\w)"))
    return out


def _mask_title(text: str, event: Event) -> str:
    """``text`` (normalised) with the paper's own title replaced by a space, so that words of the title ("Leave
    **No** Context Behind ...", "... under Uncertainty") are never read as negation, hedge or abstain cues."""
    for pat in _title_patterns(event.subject):
        text = pat.sub(" ", text)
    return " ".join(text.split())


def grade_paper(event: Event, probe: str, response: str) -> tuple[Grade, bool] | None:
    """Rule grade for first-author probes (categories ``paper`` and ``fake_paper``), or ``None`` when unsure.

    Real papers: ``correct`` when the expected name is asserted in a full form (``Event.aliases``: full name,
    ``Surname, Given``, ``G. Surname``, diacritics folded; spaces and hyphens inside a name are free, so "Jun-sol
    Kim" is "Junsol Kim") or by surname alone, unless the surname is common (four letters or fewer, or in
    :data:`_COMMON_SURNAMES`), in which case the given name's first token must appear too; ``incorrect`` when a
    distractor (``reject_aliases``) or any other capitalised two-token name is asserted instead; ``abstain`` for
    an abstain phrase without any name; otherwise ``None`` (a name only in a negated clause, the answer next to a
    distractor, an unknown name next to a phrase that disclaims knowledge of the paper, a name that could be the
    expected one misspelt or written out, ...). Controls (fake papers): a distractor or any named person is
    ``incorrect`` (a confabulation), an abstention is ``correct``; a name next to a phrase that disclaims the
    paper ("I cannot find this paper; you may mean X by Y") goes to the judges, as does anything else. ``hedged``
    is set when the asserted name comes with uncertainty or knowledge-horizon words or an abstain phrase.

    Surface forms: markdown emphasis around a name (``**bold**``, ``*italics*``, backticks) is not part of it; a
    name may follow "is", "was", "by" or "first author:" and carry a middle initial ("Bardienus P. Duisterhof");
    an affiliation in parentheses or a quoted title after a name is not part of it; and the words of the paper's
    own title ("Leave No Context Behind ...") are never read as negation, hedge or abstain cues.
    """
    if not (response or "").strip():
        return "ungraded", False
    if probe == "mcq":
        return grade_mcq(event, response), False
    text = _mask_title(normalize_text(_plain(response)), event)
    abstain = bool(_ABSTAIN.search(text) or _PAPER_ABSTAIN.search(text))
    # "I don't know this paper" / "cannot find it", as opposed to a mere "not sure" about an answer still given
    disclaims = bool(_DISCLAIMS.search(text) or _PAPER_ABSTAIN.search(text))
    uncertain = bool(_HEDGE.search(text) or _UNCERTAIN.search(text))
    reject = _compile_forms(event.reject_aliases, loose=True)
    unknown = _unknown_names(response, event)
    if _is_control(event):
        strong, _ = _hits(text, {"correct": [], "outdated": [], "incorrect": reject}, event.date)
        if "incorrect" in strong:
            return "incorrect", uncertain or abstain
        if unknown:
            return None if disclaims else ("incorrect", uncertain or abstain)
        return ("correct", False) if abstain else None
    expected = (event.expected_recall if probe == "recall" else "") or event.expected_current
    full_forms = _compile_forms((a for a in [*event.aliases, expected] if len(a.split()) >= 2), loose=True)
    strong, weak = _hits(text, {"correct": full_forms, "outdated": [], "incorrect": reject}, event.date)
    if "correct" not in strong:
        surname, given = _surname_given(expected)
        s_strong, s_weak = _hits(text, {"correct": _compile_forms([surname], loose=True), "outdated": [],
                                        "incorrect": []}, event.date)
        common = len(normalize_text(surname)) <= 4 or normalize_text(surname) in _COMMON_SURNAMES
        given_seen = bool(given) and any(p.search(text) for p in _compile_forms([given], loose=True))
        if "correct" in s_strong and (not common or given_seen):
            strong.add("correct")
        elif "correct" in s_strong or "correct" in s_weak:
            weak.add("correct")
    if "correct" in strong:
        if "incorrect" in strong:
            return None  # the answer and a distractor are both asserted: judges decide
        return "correct", uncertain or abstain
    if "incorrect" in strong:
        return "incorrect", uncertain or abstain
    if unknown:
        if disclaims or any(_near_name(n, expected) for n in unknown):
            return None  # an abstention with a stray name, or a possible misspelling of the answer: judges decide
        return "incorrect", uncertain or abstain
    if abstain and not weak:
        return "abstain", False
    return None


# ------------------------------------------------------------------------------------------------- judges

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
# the captured label must end the value: "correct|outdated|..." (an echoed template) is not a label
_LABEL_FIELD = re.compile(r"""["']?label["']?\s*[:=]\s*["']?([A-Za-z_\-]+)(?=["'\s,;}]|$)([^,;}\n]*)""",
                          re.IGNORECASE)
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
        others = {_label(w) for w in re.findall(r"[A-Za-z_\-]+", m.group(2))} - {None, label}
        if others:  # "label: correct or outdated" names two labels: no decision
            return None
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
    A row keeps an earlier judge verdict when the new pass cannot decide (the rules are unsure and no judge is
    given, or no two judges agree), so an offline rule-only pass never erases judge grades; every other field
    (``usage``, ``latency_s``, ...) is kept as it was.
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
        if grade == "ungraded" and r.grade != "ungraded" and r.grader.startswith("judge"):
            return r  # the rules are unsure and no judge decided: the earlier judge verdict stands
        return dataclasses.replace(r, grade=grade, hedged=hedged, grader=grader)

    return list(await asyncio.gather(*(one(r) for r in results)))
