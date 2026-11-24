"""Simulated models with known per-topic knowledge cutoffs (DESIGN.md §3.3).

``MockBackend`` implements ``types.Backend`` without any network access. It is fully deterministic given its seed,
so tests, ``cutoffmix simulate`` and the ``--mock`` UI demo are reproducible:

- ``stream_continuation`` writes plausible analyst notes about the dominant topic of the question and recent notes,
  stating what the model knows "as of" its true cutoff for that topic, and streams them in ~8-character chunks.
- ``chat`` answers benchmark probes for known events according to a 4PL curve around the model's true cutoff,
  so an index fitted from mock probe results should recover ``MockModelTruth.tau``.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import math
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, cast

from cutoffmix.backends.base import BackendError
from cutoffmix.timeutil import month_label, parse_fractional_month, today_month
from cutoffmix.types import ChatRequest, ChatResult, ContinuationRequest, Event, ModelSpec, StreamEvent, TopicClassifier

CHUNK_CHARS = 8
_WINDOW_SEGMENTS = 3  # notes taken into account when picking the topic of a mock segment


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

    def _segment_text(self, model: ModelSpec, req: ContinuationRequest, fields: dict[str, str]) -> str:
        answer = (f"The latest state of {fields['subject']} known as of {fields['asof']}; anything after "
                  f"{fields['asof']} should be checked against a current source.")
        if req.phase == "answer":
            return (f"Based on the shared notes: {answer[0].lower()}{answer[1:]} This is {fields['gap']} before "
                    f"today ({fields['today']}).")
        if len(req.segments) >= self.segments_to_answer:
            return (f"The notes now cover {fields['subject']} up to what is known as of {fields['asof']}, and no open "
                    f"point remains. {req.final_marker} {answer}")
        return self._note_text(model, req, fields)

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
        topic = self.topic_of(" ".join([req.question, *req.segments[-_WINDOW_SEGMENTS:]]))
        tau = truth.tau_for(topic)
        subject, facts = _TOPIC_PHRASES.get(topic, (topic.replace("_", " "), "recent developments"))
        today = self._today(req.today)
        fields = {"q": _short_question(req.question), "subject": subject, "facts": facts, "asof": month_label(tau),
                  "gap": _month_gap(tau, today), "today": today}
        text = self._segment_text(model, req, fields)
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
