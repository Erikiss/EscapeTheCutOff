"""Simulated models with known per-topic knowledge cutoffs (DESIGN.md §3.3).

``MockBackend`` implements ``types.Backend`` without any network access. It is fully deterministic given its seed,
so tests, ``cutoffmix simulate`` and the ``--mock`` UI demo are reproducible:

- ``stream_continuation`` writes plausible analyst notes stating what the model knows "as of" its true cutoff for
  the segment's topic, and streams them in ~8-character chunks. A question about several topics ("Who is the
  president, and what is new in AI?") is covered topic by topic, three segments each, so the routing window moves
  between topics; otherwise the topic is the dominant one of the question and recent notes.
- ``chat`` answers benchmark probes for known events according to a 4PL curve around the model's true cutoff,
  so an index fitted from mock probe results should recover ``MockModelTruth.tau``.
- Under the explore preset (DESIGN.md §7; the request carries ``system_override``) the notes follow the
  structured-notes template instead: labelled items ending with ``Where we stand:``, a ``Lens (field):`` note when
  the tail asks for a cross-field lens, ``Lens kept:`` / ``Lens dropped:`` for the check that follows, the requested
  opening cue after a conclusion the moderator set aside, and ``FINAL ANSWER:`` only when the tail says the
  moderator accepts a conclusion (plus one deliberate early attempt per run, so the continuation path is exercised).
  Without ``system_override`` the output is exactly the default one.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import math
import random
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from cutoffmix.backends.base import BackendError
from cutoffmix.timeutil import month_label, parse_fractional_month, today_month
from cutoffmix.types import ChatRequest, ChatResult, ContinuationRequest, Event, ModelSpec, StreamEvent, TopicClassifier

CHUNK_CHARS = 8
_WINDOW_SEGMENTS = 3  # notes taken into account when picking the topic of a mock segment
_MAX_QUESTION_TOPICS = 3  # a question is covered topic by topic for at most this many topics
_SEGMENTS_PER_TOPIC = 3


@dataclass
class MockModelTruth:
    """Ground truth for one simulated model."""

    key: str
    tau: dict[str, float]  # true per-topic virtual cutoff (month index); missing topics -> default_tau
    default_tau: float
    slope: float = 0.6
    plateau: float = 0.95
    stale_rate: float = 0.7  # P(assert stale state | unknown); else abstain
    latency_s: float = 0.02  # per streamed chunk
    refusal_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.slope <= 0:
            raise ValueError(f"{self.key}: slope must be > 0, got {self.slope}")
        if not 0 < self.plateau <= 1:
            raise ValueError(f"{self.key}: plateau must be in (0, 1], got {self.plateau}")
        for name in ("stale_rate", "refusal_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{self.key}: {name} must be in [0, 1], got {getattr(self, name)}")
        if self.latency_s < 0:
            raise ValueError(f"{self.key}: latency_s must be >= 0, got {self.latency_s}")

    def tau_for(self, topic: str) -> float:
        """The true cutoff for ``topic`` (``default_tau`` if the topic has no entry)."""
        return self.tau.get(topic, self.default_tau)

    def p_correct(self, topic: str, t_event: float, guess: float = 0.0) -> float:
        """4PL probability of a correct answer about an event at month ``t_event``: g + (a - g)·σ((τ - t)/s)."""
        return guess + (self.plateau - guess) * _sigmoid((self.tau_for(topic) - t_event) / self.slope)


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def default_mock_truths(latency_s: float = 0.02) -> dict[str, MockModelTruth]:
    """Truths for the three configured models.

    GPT-6.2 Sol is freshest on most topics (as in the real benchmark data), with clear reversals: Claude Fable 5.2
    leads on ``ai_ml`` and ``computing_software``, Gemini 4 Neon on ``science_research`` and ``space_astronomy``.
    Sol never abstains (it asserts stale facts), Neon abstains most often.
    """
    return {
        "claude-fable-5.2": MockModelTruth(
            key="claude-fable-5.2", default_tau=26.8,
            tau={"ai_ml": 31.0, "computing_software": 30.25, "law_crime": 27.5},
            slope=0.6, plateau=0.95, stale_rate=0.6, latency_s=latency_s),
        "gpt-6.2-sol": MockModelTruth(
            key="gpt-6.2-sol", default_tau=28.75,
            tau={"ai_ml": 27.0, "computing_software": 27.25, "science_research": 27.5, "space_astronomy": 27.0,
                 "arts_entertainment": 29.5, "business_economy": 29.25, "sports": 29.0},
            slope=0.5, plateau=0.96, stale_rate=0.95, latency_s=latency_s),
        "gemini-4-neon": MockModelTruth(
            key="gemini-4-neon", default_tau=27.0,
            tau={"science_research": 30.5, "space_astronomy": 31.25, "natural_disasters_climate": 27.75},
            slope=0.7, plateau=0.93, stale_rate=0.45, latency_s=latency_s),
    }


# (what the topic is about, which facts go stale) -- used to write topic-specific mock notes
_TOPIC_PHRASES: dict[str, tuple[str, str]] = {
    "conflict_security": ("the situation on the ground", "front lines, ceasefire terms and casualty figures"),
    "international_relations": ("the diplomatic track", "sanctions, treaties and summit outcomes"),
    "government": ("who holds the office", "cabinet changes, resignations and appointments"),
    "elections": ("the latest election results", "certified results, runoffs and coalition talks"),
    "law_crime": ("the legal proceedings", "rulings, indictments and appeals"),
    "accidents": ("the incident reports", "casualty counts and investigation findings"),
    "natural_disasters_climate": ("the disaster and climate record", "damage estimates and climate-policy decisions"),
    "health_medicine": ("the medical evidence", "approvals, trial readouts and outbreak numbers"),
    "business_economy": ("the corporate and market picture", "leadership changes, deals and rate decisions"),
    "ai_ml": ("the AI model landscape", "model releases, benchmark results and lab announcements"),
    "computing_software": ("the software and hardware versions", "release numbers, deprecations and advisories"),
    "space_astronomy": ("the mission status", "launches, landings and mission schedules"),
    "science_research": ("the research findings", "published results, replications and prizes"),
    "sports": ("the sporting results", "champions, transfers and coaching changes"),
    "arts_entertainment": ("the cultural record", "releases, awards and box-office figures"),
}

_NOTE_TEMPLATES: tuple[tuple[str, ...], ...] = (
    ("First, the question ({q}) is about {subject}, so the answer depends on recent {facts} rather than on stable "
     "background knowledge.",
     "The record available here is known as of {asof}, {gap} before today ({today}).",
     "Any change after {asof} would not show up in these notes."),
    ("Evidence: the most recent {facts} accounted for here are known as of {asof}.",
     "Everything stated about {subject} so far is consistent with that record.",
     "Since {asof} ({gap} ago) the notes cannot rule out a change, so this part stays provisional."),
    ("Next, the notes are checked against the timeline known as of {asof}.",
     "No entry in that timeline conflicts with the summary of {subject} above.",
     "The open risk is a development after {asof} ({gap} ago), which is typical for {facts}."),
    ("Correction: an earlier note treated {subject} as settled.",
     "Known as of {asof}, it was still developing, so the intermediate result should be phrased as the latest "
     "state at that date.",
     "That keeps the claim accurate even if {facts} changed later."),
    ("Putting this together, the answer should name the latest known state of {subject}.",
     "It should say explicitly that this is known as of {asof} and flag that {facts} may have changed since.",
     "A reader can then check the period after {asof} ({gap}) against a current source."),
    ("One more check on {subject}: comparing the dates in the notes with the record known as of {asof} gives no "
     "conflicting entry.",
     "The intermediate result therefore stands, with the caveat that {facts} move quickly."),
)


# ---- explore preset (DESIGN.md §7) ----------------------------------------------------------------------------
# The per-segment directive is read from the tail block the way a model would read it (prompts.py: RELAY_TAIL_EXPLORE,
# RECORD_*_TEXT, LENS_TEXT, CHECK_TEXT, CONTINUE_TEXT, VS_DIRECTION_TEXT).
_GATE_OPEN_PHRASE = "the moderator accepts a conclusion"  # only in RECORD_OPEN_TEXT ("does not accept ... yet" otherwise)
_LENS_FIELD_RE = re.compile(r"Begin the note with 'Lens \((?P<field>[^()\n]+)\):'")
_LENS_CONCEPTS_RE = re.compile(r"for example (?P<c1>[^,]+?), (?P<c2>[^,]+?) or (?P<c3>[^)]+?)\)")
_CHECK_FIELD_RE = re.compile(r"applied a cross-field lens from (?P<field>.+?)\. Begin this note")
_CUE_RE = re.compile(r"Begin this note with the words '(?P<cue>[^'\n]+)'")
_DIRECTION_RE = re.compile(r"Direction for this note: (?P<direction>.+?)\. Develop it")
_MISSING_LABEL_RE = re.compile(r"'(?P<label>[A-Z][a-z]+(?: [a-z]+)*):' item")  # "an 'Evidence:' item that ..."
_PREMATURE_MARK = "the record already settles the problem"  # Claim of the one deliberate early conclusion per run
_KEPT_RATE = 0.6  # share of checks that keep the lens (deterministic per seed, model, question, field and note)
_MAX_ITEMS = 4  # labelled items per note (the template asks for two to four)
_EXPLORE_ITEMS: dict[str, tuple[str, ...]] = {
    "Claim": (
        "Claim: the answer turns on the latest state of {subject}, so it depends on recent {facts} rather than on "
        "stable background knowledge.",
        "Claim: nothing in the notes so far contradicts the state of {subject} known as of {asof}.",
        "Claim: the question ({q}) is settled by the record up to {asof} unless {facts} changed afterwards.",
    ),
    "Evidence": (
        "Evidence: the most recent {facts} accounted for here are known as of {asof} (source: the record available to "
        "this analyst, {gap} before today, {today}).",
        "Evidence: the timeline of {subject} known as of {asof} has no entry that conflicts with the notes (source: "
        "the same dated record).",
    ),
    "Inference": (
        "Inference: because the record ends at {asof} and {facts} move quickly, therefore any statement about "
        "{subject} must carry that date.",
        "Inference: because no later entry is known and the notes agree, therefore the latest known state stands as "
        "the working answer.",
    ),
    "Principle": (
        "Principle: in archival research a claim is only as current as its latest dated source; applied here, the "
        "answer about {subject} carries the date {asof}.",
    ),
    "Alternative set aside": (
        "Alternative set aside: treating {subject} as settled for good; set aside because {facts} may have changed "
        "after {asof} ({gap} ago).",
        "Alternative set aside: answering from stable background knowledge alone; set aside because the question "
        "depends on recent {facts}.",
    ),
    "Open question": (
        "Open question: whether {facts} changed after {asof}; a current source for the period since then would "
        "settle it.",
        "Open question: which analyst's record of {subject} is the most recent; comparing their dates would settle "
        "it.",
    ),
    "Next": (
        "Next: check the period after {asof} ({gap}) against a current source before concluding.",
        "Next: state the answer as the latest known state of {subject}, dated {asof}.",
    ),
}
_GENERIC_ITEM = "{label}: for {subject}, the record known as of {asof} is the only {lower} available here."
_EXPLORE_PLAN: tuple[tuple[str, ...], ...] = (  # labels of consecutive notes (cycled; required ones are added)
    ("Claim", "Evidence", "Open question"),
    ("Evidence", "Alternative set aside", "Next"),
    ("Inference", "Principle", "Open question"),
    ("Claim", "Alternative set aside", "Next"),
    ("Evidence", "Inference"),
    ("Principle", "Open question", "Next"),
)
_LENS_HEAD = (
    "Lens ({field}): {c1} resembles the current intermediate result: the dated record of {subject} plays the role of "
    "the initial state, {facts} play the role of the driving change, and the cutoff {asof} is the boundary at which "
    "{c2} applies."
)
_LENS_INFERENCE = (
    "Inference: because in {c1} what lies past the boundary is not visible from inside it, therefore the testable "
    "question is whether {facts} after {asof} changed {subject} (a working analogy, not evidence)."
)
_LENS_RETURN = "Back to the problem: the analogy only says what to check; the dated answer stands."
_CHECK_KEPT = (
    "Lens kept: the mapping of the dated record onto {field} holds for the boundary at {asof}; one real difference is "
    "that {facts} are discrete events rather than quantities, but the suggested consequence survives as a question "
    "about the period after {asof}."
)
_CHECK_DROPPED = (
    "Lens dropped: the comparison with {field} breaks because {facts} are discrete events rather than quantities, so "
    "its suggested consequence does not survive; the main line continues without it."
)
_CUE_SENTENCE = "{cue} the latest state of {subject} may have changed after {asof}, which the notes did not yet weigh."
_DIRECTION_CLAIM = "Claim: following the direction ({direction}), the dated answer about {subject} still rests on the record as of {asof}."
_PREMATURE_CLAIM = "Claim: {mark}, since the notes agree on the state of {subject} known as of {asof}."
_WHERE_WE_STAND = (
    "Where we stand: best answer, the latest known state of {subject} as of {asof}; confidence {confidence}; weakest "
    "assumption, that {facts} did not change after {asof}."
)
_ANSWER_BASIS = (
    "\n\nBasis: the dated record of {subject} known as of {asof} (the Evidence items) and the claim that no later "
    "entry conflicts with it.\n\nNot resolved: whether {facts} changed after {asof}."
)


def _month_gap(tau: float, today: str) -> str:
    months = round(today_month(_dt.date.fromisoformat(today)) - tau)
    if months <= 0:
        return "less than a month"
    return "about 1 month" if months == 1 else f"about {months} months"


def _short_question(question: str, limit: int = 80) -> str:
    q = " ".join(question.split())
    return f'"{q}"' if len(q) <= limit else f'"{q[: limit - 3].rstrip()}..."'


def _chunks(text: str, size: int = CHUNK_CHARS) -> list[str]:
    return [text[i: i + size] for i in range(0, len(text), size)]


class MockBackend:
    """Deterministic simulated backend implementing ``types.Backend`` (see the module docstring)."""

    def __init__(
        self,
        truths: dict[str, MockModelTruth] | None = None,
        *,
        seed: int = 0,
        classifier: TopicClassifier | None = None,
        segments_to_answer: int = 6,
        events: dict[str, Event] | None = None,
        today: str | None = None,
    ) -> None:
        self.truths = truths if truths is not None else default_mock_truths()
        self.seed = seed
        self._classifier = classifier
        self.segments_to_answer = segments_to_answer
        self.events: dict[str, Event] = dict(events or {})
        self.today = today

    # ---- helpers ------------------------------------------------------------------------

    @property
    def classifier(self) -> TopicClassifier:
        """The topic classifier (the keyword classifier from ``cutoffmix.router.topics`` unless one was given)."""
        if self._classifier is None:
            from cutoffmix.router.topics import KeywordTopicClassifier  # lazy: router is optional for the backend

            self._classifier = cast(TopicClassifier, KeywordTopicClassifier())
        return self._classifier

    def topic_of(self, text: str) -> str:
        """The most probable topic id of ``text``."""
        clf = self.classifier
        post = clf.posterior(text)
        return clf.topics[max(range(len(post)), key=post.__getitem__)]

    def question_topics(self, question: str) -> list[str]:
        """The topics a question asks about, most probable first.

        The most probable topic plus every topic with at least half its probability (and at least twice the
        uniform probability), at most three. A question without a clear topic gives one (arbitrary) topic.
        """
        clf = self.classifier
        post = clf.posterior(question)
        order = sorted(range(len(post)), key=lambda i: (-post[i], i))
        floor = max(0.5 * post[order[0]], 2.0 / len(post))
        picked = [clf.topics[i] for i in order[:_MAX_QUESTION_TOPICS] if post[i] >= floor]
        return picked or [clf.topics[order[0]]]

    def _segment_topics(self, req: ContinuationRequest) -> list[str]:
        """``[topic of this segment]``, or every question topic for the answer and the concluding segment."""
        asked = self.question_topics(req.question)
        concluding = req.phase == "answer" or len(req.segments) >= self.segments_to_answer
        if len(asked) > 1:
            return asked if concluding else [asked[(len(req.segments) // _SEGMENTS_PER_TOPIC) % len(asked)]]
        return [self.topic_of(" ".join([req.question, *req.segments[-_WINDOW_SEGMENTS:]]))]

    def _rng(self, *parts: object) -> random.Random:
        digest = hashlib.sha256("\x1f".join(map(str, (self.seed, *parts))).encode("utf-8")).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def _truth(self, model: ModelSpec) -> MockModelTruth:
        try:
            return self.truths[model.key]
        except KeyError:
            raise BackendError(f"MockBackend has no truth for model {model.key!r}; known: {', '.join(self.truths)}",
                               error_type="unknown_model") from None

    def _today(self, req_today: str | None) -> str:
        return req_today or self.today or _dt.date.today().isoformat()

    # ---- streaming ----------------------------------------------------------------------

    def _note_text(self, model: ModelSpec, req: ContinuationRequest, fields: dict[str, str]) -> str:
        n = len(req.segments)
        if n == 0:
            template = _NOTE_TEMPLATES[0]
        else:  # rotate through the follow-up templates with a per-model offset: consecutive notes never repeat
            k = len(_NOTE_TEMPLATES) - 1
            template = _NOTE_TEMPLATES[1 + (n - 1 + self._rng("offset", model.key).randrange(k)) % k]
        budget = max(req.target_tokens * 4, 1)  # ~4 characters per token
        sentences: list[str] = []
        for sentence in template:
            text = sentence.format(**fields)
            if sentences and sum(len(s) + 1 for s in sentences) + len(text) > budget:
                break
            sentences.append(text)
        return " ".join(sentences)

    def _segment_text(self, model: ModelSpec, req: ContinuationRequest, fields: dict[str, str],
                      known: list[tuple[str, str]]) -> str:
        """The segment (or answer) text; ``known`` lists ``(subject, as-of month)`` for every topic it covers."""
        states = ", and of ".join(f"{subject} known as of {asof}" for subject, asof in known)
        dates = known[0][1] if len(known) == 1 else "those dates"
        answer = f"The latest state of {states}; anything after {dates} should be checked against a current source."
        if req.system_override is not None:  # explore preset: structured notes under the per-segment directive
            return self._explore_text(model, req, fields, answer)
        if req.phase == "answer":
            return (f"Based on the shared notes: {answer[0].lower()}{answer[1:]} This is {fields['gap']} before "
                    f"today ({fields['today']}).")
        if len(req.segments) >= self.segments_to_answer:
            covered = " and ".join(subject for subject, _ in known)
            return (f"The notes now cover {covered} up to what is known as of {fields['asof']}, and no open "
                    f"point remains. {req.final_marker} {answer}")
        return self._note_text(model, req, fields)

    # ---- explore preset -----------------------------------------------------------------

    def _explore_items(self, model: ModelSpec, n: int, fields: dict[str, str], required: list[str],
                       room: int) -> list[str]:
        """Up to ``room`` labelled items for note ``n``: the ``required`` labels (named in the tail) first, then the
        plan's labels; every label at most once (so at most one 'Principle:'); consecutive notes differ."""
        offset = self._rng("explore_offset", model.key).randrange(len(_EXPLORE_PLAN))
        labels: list[str] = []
        for label in [*required, *_EXPLORE_PLAN[(n + offset) % len(_EXPLORE_PLAN)]]:
            if label not in labels and len(labels) < room:
                labels.append(label)
        items: list[str] = []
        for label in labels:
            variants = _EXPLORE_ITEMS.get(label)
            template = variants[(n + offset) % len(variants)] if variants else _GENERIC_ITEM
            items.append(template.format(label=label, lower=label.lower(), **fields))
        return items

    def _explore_text(self, model: ModelSpec, req: ContinuationRequest, fields: dict[str, str], answer: str) -> str:
        """A note (or answer) under the explore template; the directive is read from ``req.tail`` (DESIGN.md §7)."""
        if req.phase == "answer":
            text = (f"Based on the shared notes: {answer[0].lower()}{answer[1:]} This is {fields['gap']} before "
                    f"today ({fields['today']}).")
            if "'Basis:'" in (req.system_override or ""):  # answer_basis: the answer names its basis
                text += _ANSWER_BASIS.format(**fields)
            return text
        tail, n = req.tail or "", len(req.segments)
        if _GATE_OPEN_PHRASE in tail:  # the moderator accepts a conclusion: the answer instead of a note
            return f"{req.final_marker} {answer}"
        required = list(dict.fromkeys(m.group("label") for m in _MISSING_LABEL_RE.finditer(tail)))
        head: list[str] = []  # the opening the directive asks for; ``labelled`` of its lines count as items
        labelled = 0
        lens, check, cue, direction = (p.search(tail) for p in (_LENS_FIELD_RE, _CHECK_FIELD_RE, _CUE_RE,
                                                                _DIRECTION_RE))
        premature = False
        if lens:
            concepts = _LENS_CONCEPTS_RE.search(tail)
            c1, c2 = (concepts.group("c1"), concepts.group("c2")) if concepts else ("its central concept",
                                                                                   "its boundary condition")
            head += [_LENS_HEAD.format(field=lens.group("field"), c1=c1, c2=c2, **fields),
                     _LENS_INFERENCE.format(c1=c1, **fields)]
            labelled = 2
            required = [label for label in required if label != "Inference"]
        elif check:
            field = check.group("field")
            kept = self._rng("verdict", model.key, req.question, field, n).random() < _KEPT_RATE
            head.append((_CHECK_KEPT if kept else _CHECK_DROPPED).format(field=field, **fields))
            labelled = 1
        elif cue:
            head.append(_CUE_SENTENCE.format(cue=cue.group("cue"), **fields))
        elif direction:
            head.append(_DIRECTION_CLAIM.format(direction=direction.group("direction"), **fields))
            labelled = 1
        elif n >= 2 and not any(_PREMATURE_MARK in s for s in req.segments):
            premature = True  # once per run: conclude while the record is incomplete (the moderator sets it aside)
            head.append(_PREMATURE_CLAIM.format(mark=_PREMATURE_MARK, **fields))
            labelled = 1
        items = self._explore_items(model, n, fields, required, max(_MAX_ITEMS - labelled, 2))
        confidence = "low" if n < 2 else "medium" if n < 5 else "high"
        close = ([_LENS_RETURN] if lens else []) + [_WHERE_WE_STAND.format(confidence=confidence, **fields)]
        budget = min(max(req.target_tokens * 4, 1), req.hard_cut_chars - 40)  # ~4 chars/token; the cut must not reach WWS
        while len(items) > max(2 - labelled, 0) and sum(len(s) + 1 for s in [*head, *items, *close]) > budget:
            items.pop()
        text = "\n".join([*head, *items, *close])
        return f"{text} {req.final_marker} {answer}" if premature else text

    async def stream_continuation(self, model: ModelSpec, req: ContinuationRequest) -> AsyncIterator[StreamEvent]:
        """Stream a simulated segment (or answer): ``text`` chunks, then exactly one ``done`` event.

        Models with ``supports_stop`` end with finish_reason ``"stop"`` (the provider swallows the sentinel); the
        others stream the sentinel and end with ``"sentinel"``. A refusal (``refusal_rate``) yields only ``done``
        with finish_reason ``"content_filter"``.
        """
        truth = self._truth(model)
        rng = self._rng("stream", model.key, req.phase, req.question, len(req.segments))
        if req.phase == "think" and rng.random() < truth.refusal_rate:
            await asyncio.sleep(truth.latency_s)
            yield StreamEvent(kind="done", finish_reason="content_filter", error="refused: simulated refusal",
                              usage=self._usage(req, "", rng))
            return
        topics = self._segment_topics(req)
        phrases = [_TOPIC_PHRASES.get(z, (z.replace("_", " "), "recent developments")) for z in topics]
        tau = min(truth.tau_for(z) for z in topics)  # the oldest knowledge bounds the whole statement
        today = self._today(req.today)
        fields = {"q": _short_question(req.question), "subject": phrases[0][0], "facts": phrases[0][1],
                  "asof": month_label(tau), "gap": _month_gap(tau, today), "today": today}
        known = [(subject, month_label(truth.tau_for(z))) for z, (subject, _) in zip(topics, phrases, strict=True)]
        text = self._segment_text(model, req, fields, known)
        if req.phase == "think" and not model.supports_stop:
            streamed, finish = text + req.sentinel, "sentinel"
        else:
            streamed, finish = text, "stop"
        for piece in _chunks(streamed):
            await asyncio.sleep(truth.latency_s)
            yield StreamEvent(kind="text", text=piece)
        yield StreamEvent(kind="done", usage=self._usage(req, text, rng), finish_reason=finish)

    @staticmethod
    def _usage(req: ContinuationRequest, text: str, rng: random.Random) -> dict[str, Any]:
        prompt_chars = 900 + len(req.question) + sum(len(s) for s in req.segments)  # ~900: system prompt + framing
        prompt_chars += len(req.tail or "")
        reasoning = 64 + rng.randrange(192)
        completion = len(text) // 4 + reasoning
        prompt = prompt_chars // 4
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
                "completion_tokens_details": {"reasoning_tokens": reasoning}}

    # ---- probes -------------------------------------------------------------------------

    @staticmethod
    def _stale_letter(event: Event) -> str | None:
        if not event.reject_aliases:
            return None
        stale = event.reject_aliases[0].lower()
        for choice in event.mcq_choices:
            if stale in choice.lower():
                return choice.strip()[:1].upper()
        return None

    @staticmethod
    def _stale_text(event: Event) -> str:
        if event.reject_aliases:
            return f"{event.reject_aliases[0]}."
        if event.category == "death":
            return f"As far as I know, {event.subject} is alive."
        if event.category == "office_change":
            return "As far as I know, the previous office holder is still in office."
        return f"As far as I know, nothing has changed regarding {event.subject}."

    def _probe_answer(self, model: ModelSpec, truth: MockModelTruth, event: Event,
                      probe: str) -> tuple[str, str, float]:
        """Return (response text, simulated outcome, P(correct)) for one probe of one event."""
        mcq = probe == "mcq"
        p = truth.p_correct(event.topic, parse_fractional_month(event.date), guess=0.25 if mcq else 0.0)
        rng = self._rng("probe", model.key, event.id, probe)
        if mcq:
            letters = [c.strip()[:1].upper() for c in event.mcq_choices] or ["A", "B", "C", "D"]
            answer = event.mcq_answer.strip().upper()
            if rng.random() < p:
                return answer, "correct", p
            wrong = [x for x in letters if x != answer] or letters
            stale = self._stale_letter(event)
            if stale in wrong and rng.random() < truth.stale_rate:
                return stale, "outdated", p
            return rng.choice(wrong), "incorrect", p
        if rng.random() < p:
            correct = (event.expected_recall if probe == "recall" else "") or event.expected_current
            correct = correct or (event.aliases[0] if event.aliases else event.fact)
            return f"{correct.rstrip('.')}.", "correct", p
        if rng.random() < truth.stale_rate:
            return self._stale_text(event), "outdated", p
        return "I don't know; I have no reliable information about that.", "abstain", p

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult:
        """Answer a probe for a known event (``req.metadata`` has ``event_id`` and ``probe``) according to the
        model's truth; any other request gets a fixed placeholder reply. Deterministic per (seed, model, event,
        probe); ``ChatResult.raw`` records the simulated outcome and P(correct)."""
        truth = self._truth(model)
        meta = req.metadata
        event = self.events.get(str(meta["event_id"])) if "event_id" in meta and "probe" in meta else None
        if event is not None:
            text, outcome, p = self._probe_answer(model, truth, event, str(meta["probe"]))
            raw = {"mock": True, "model": model.key, "event_id": event.id, "probe": meta["probe"],
                   "outcome": outcome, "p_correct": p}
        else:
            text = f"Mock reply from {model.label()}: this request has no scripted answer."
            raw = {"mock": True, "model": model.key, "outcome": "unscripted"}
        await asyncio.sleep(truth.latency_s)
        prompt = sum(len(str(m.get("content", ""))) for m in req.messages) // 4
        completion = max(len(text) // 4, 1)
        return ChatResult(text=text, usage={"prompt_tokens": prompt, "completion_tokens": completion,
                                            "total_tokens": prompt + completion},
                          finish_reason="stop", latency_s=truth.latency_s, raw=raw)
