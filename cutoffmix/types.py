"""Core data types and interface contracts shared by every cutoffmix module.

Keep this module dependency-free (stdlib only) so that every other module can import it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Mapping, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """A model that can take a turn in the shared chain of thought.

    All three target models have *mandatory* hidden reasoning on OpenRouter (it cannot be disabled), expose no
    logprobs, and never return raw chain-of-thought. The relay therefore shares a *visible* working-notes stream;
    every call's own hidden reasoning is discarded (``reasoning.exclude``) and kept small (``effort: low``).
    """

    key: str  # short stable key used everywhere (index rows, traces, UI), e.g. "claude-fable-5.2"
    provider_id: str  # id at the gateway, e.g. "anthropic/claude-fable-5.2" on OpenRouter
    display: str = ""
    color: str = "#888888"  # UI colour of this model's segments
    supports_stop: bool = True  # gateway honours `stop` on the pinned providers (GPT-6.2 Sol: no)
    supports_temperature: bool = False  # Fable 5.2 / GPT-6.2 Sol: no; Gemini 4 Neon: yes
    cancellable: bool = True  # closing the stream stops billing (Anthropic/OpenAI/Azure yes; Google no)
    # OpenRouter `provider` routing object, e.g. {"only": ["anthropic", "azure"], "allow_fallbacks": False}
    provider_routing: Mapping[str, Any] = field(default_factory=dict)
    # OpenRouter `reasoning` object used for relay segments
    reasoning: Mapping[str, Any] = field(default_factory=lambda: {"effort": "low", "exclude": True})
    cache_style: Literal["anthropic", "openai", "implicit", "none"] = "none"
    max_tokens_slack: int = 2048  # hidden-reasoning allowance added to the visible target (reasoning shares max_tokens)
    extra_body: Mapping[str, Any] = field(default_factory=dict)  # merged into every request body

    def label(self) -> str:
        return self.display or self.key


# --------------------------------------------------------------------------------------
# Backend contract
# --------------------------------------------------------------------------------------

Phase = Literal["think", "answer"]


@dataclass
class ContinuationRequest:
    """Ask a model for the next segment of the shared notes (phase="think") or the final answer (phase="answer").

    ``segments`` are the notes so far, one string per committed segment, *never edited* after commit, so that
    request bodies stay append-only and provider prompt caches keep hitting.
    """

    question: str
    segments: list[str] = field(default_factory=list)
    phase: Phase = "think"
    target_tokens: int = 128  # visible length we ask for ("about K tokens")
    max_tokens: int | None = None  # total incl. hidden reasoning; None -> target_tokens + model.max_tokens_slack
    today: str | None = None  # ISO date injected into the (static) system prompt when not None
    sentinel: str = "<<END_STEP>>"
    final_marker: str = "FINAL ANSWER:"
    hard_cut_chars: int = 600  # client-side cap on visible characters per segment
    run_id: str = ""  # used for prompt_cache_key / session_id
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatRequest:
    """A plain single-turn chat call (used by the benchmark runner / probe generator / judge)."""

    messages: list[dict[str, Any]]
    max_tokens: int = 512
    temperature: float | None = 0.0
    reasoning: dict[str, Any] | None = None  # passed through as OpenRouter's `reasoning` object
    response_format: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    kind: Literal["text", "reasoning", "done", "error"]
    text: str = ""
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class ChatResult:
    text: str
    reasoning: str = ""
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    latency_s: float = 0.0
    raw: dict[str, Any] | None = None


@runtime_checkable
class Backend(Protocol):
    """Anything that can stream a continuation segment and answer a chat call.

    Implementations: ``backends.openrouter.OpenRouterBackend`` (real) and ``backends.mock.MockBackend``
    (simulated models with known per-topic cutoffs, used by tests and ``cutoffmix simulate``).
    Cancelling the async iterator (``aclose()`` / leaving the ``async for``) MUST abort the upstream request.
    """

    def stream_continuation(self, model: ModelSpec, req: ContinuationRequest) -> AsyncIterator[StreamEvent]: ...

    async def chat(self, model: ModelSpec, req: ChatRequest) -> ChatResult: ...


# --------------------------------------------------------------------------------------
# Topic classifier contract
# --------------------------------------------------------------------------------------


@runtime_checkable
class TopicClassifier(Protocol):
    topics: Sequence[str]  # topic ids, fixed order; must match the index's topic order (by id lookup)

    def posterior(self, text: str) -> list[float]:
        """Return p(topic | text) over ``topics`` (non-negative, sums to 1). Must be fast (<5 ms typical)."""
        ...


# --------------------------------------------------------------------------------------
# Relay trace
# --------------------------------------------------------------------------------------

Decision = Literal[
    "argmax",  # deterministic best score
    "sample",  # Thompson / softmax sample
    "hold",  # hysteresis or min-dwell kept the current model
    "drift",  # mid-stream re-route aborted the previous model
    "speculative",  # chosen among parallel drafts
    "forced",  # single model / fallback after error
]


@dataclass
class Segment:
    index: int
    model: str
    phase: Phase
    text: str
    decision: Decision
    topic_posterior: dict[str, float]
    scores: dict[str, float]  # routing score per model at decision time
    started_at: float
    ended_at: float = 0.0
    aborted: bool = False
    usage: dict[str, Any] | None = None
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RelayEvent:
    """Events emitted by ``Relay.run`` (consumed by CLI, server SSE, OpenAI-compatible stream)."""

    type: Literal["segment_start", "delta", "segment_end", "phase", "error", "done"]
    segment: int | None = None
    model: str | None = None
    phase: Phase | None = None
    text: str = ""
    decision: Decision | None = None
    topic_posterior: dict[str, float] | None = None
    scores: dict[str, float] | None = None
    data: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in dataclasses.asdict(self).items() if v is not None and v != ""}


@dataclass
class RelayTrace:
    question: str
    segments: list[Segment] = field(default_factory=list)
    answer: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0

    @property
    def thinking(self) -> str:
        return "".join(s.text for s in self.segments if s.phase == "think")

    def switches(self) -> int:
        ms = [s.model for s in self.segments if s.phase == "think"]
        return sum(1 for a, b in zip(ms, ms[1:]) if a != b)

    def share_by_model(self) -> dict[str, float]:
        tot: dict[str, int] = {}
        for s in self.segments:
            if s.phase == "think":
                tot[s.model] = tot.get(s.model, 0) + len(s.text)
        n = sum(tot.values()) or 1
        return {k: v / n for k, v in tot.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "segments": [s.to_dict() for s in self.segments],
            "answer": self.answer,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "switches": self.switches(),
            "share_by_model": self.share_by_model(),
        }


# --------------------------------------------------------------------------------------
# Benchmark records
# --------------------------------------------------------------------------------------

# current   : current-state question, no date, neutral system prompt (comparable to the apoorvumang "direct" probe)
# current_d : same question under the deployment system prompt incl. today's date (PRIMARY routing index = "virtual cutoff")
# recall    : question that names the period explicitly (knowledge availability)
# mcq       : 4-way multiple choice incl. the stale/pre-event option (upper bracket, 25% floor)
ProbeKind = Literal["current", "current_d", "recall", "mcq"]
# correct  : asserts the current / ground-truth state
# outdated : asserts the stale (pre-event) state, e.g. "X is alive", the predecessor as office holder
# incorrect: other wrong / hallucinated
# abstain  : declines, says it does not know
# ungraded : empty response, API error, judge failure (dropped from every denominator)
Grade = Literal["correct", "outdated", "incorrect", "abstain", "ungraded"]


@dataclass
class Event:
    """One dated, verifiable world event used to probe a model's topic-specific knowledge.

    Superset of the apoorvumang/knowledge-cutoff ``events.jsonl`` schema (CC BY 4.0).
    """

    id: str
    date: str  # ISO date YYYY-MM-DD (day may be 01 if unknown)
    month: str  # YYYY-MM
    topic: str  # cutoffmix topic id (see cutoffmix.taxonomy)
    category: str  # event type: death | office_change | election | release | award | sports_result | other | control_alive | fake_event
    subject: str
    fact: str
    region: str = "International"
    predictability: Literal["low", "medium", "high"] = "low"
    # "current" probe: no date mentioned, asks about the present state of the world (measures the *virtual* cutoff)
    question_current: str = ""
    expected_current: str = ""
    # "recall" probe: explicitly names the period (measures knowledge availability)
    question_recall: str = ""
    expected_recall: str = ""
    mcq_question: str = ""
    mcq_choices: list[str] = field(default_factory=list)  # ["A) ...", "B) ...", "C) ...", "D) ..."]
    mcq_answer: str = ""  # "A".."D"
    aliases: list[str] = field(default_factory=list)  # accepted surface forms for deterministic grading
    reject_aliases: list[str] = field(default_factory=list)  # stale answers that indicate an outdated world-state
    source: str = ""
    origin: str = "manual"  # kc | wikidata | wikipedia | manual | synthetic
    control: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Event":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class ProbeResult:
    event_id: str
    model: str
    probe: ProbeKind
    response: str
    grade: Grade
    hedged: bool = False  # e.g. "As of March 2026, X is alive" -> grade=outdated, hedged=True
    grader: str = "rule"  # rule | mcq | judge:<model>[+<model>] | kc
    latency_s: float = 0.0
    usage: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ProbeResult":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
