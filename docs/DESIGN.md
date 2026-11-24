# cutoffmix: design and module contracts

This is the binding contract for every module. `cutoffmix/types.py`, `cutoffmix/timeutil.py` and `cutoffmix/taxonomy.py`
already exist and are **frozen**. Do not change their public names; additive fields are allowed only via the integrator.

## 0. Goal

Run one chain of thought ("shared working notes") and, segment by segment, give the next segment to the model whose
**topic-specific effective knowledge cutoff** is the most recent for the topic of the current window. The three models
are `anthropic/claude-fable-5.2`, `openai/gpt-6.2-sol` and `google/gemini-4-neon` on OpenRouter. The user calls the
latter two "Gemini 6.2 Sol" and "Gemini Neon"; 6.2 Sol is OpenAI's model. The index comes from **external evaluation**,
never from self-report. The user assumes as a premise that the models can continue each other's chain of thought.

## 1. Verified constraints (research 2026-11-24) and what follows from them

| Fact | Consequence |
|---|---|
| Reasoning is mandatory on all three models (minimum effort `low`), and the raw CoT is never returned (only summaries, encrypted, or signed) | The shared chain must be **visible text** that we keep and re-send. Each call's hidden reasoning is discarded (`reasoning: {effort: low, exclude: true}`). |
| Fable 5.2 rejects assistant prefill (400). Thinking blocks are signed and model-bound | No native continuation. Every segment is a new request: `system` + one `user` message with the notes. |
| Fable 5.2 has a `reasoning_extraction` refusal category (thinking/scratchpad tags, "running log of reasoning", "verbatim reasoning") | **Template B wording for all models**: "Several analysts … contribute the next intermediate result + evidence". Never use the words thinking/reasoning/scratchpad and **no XML tags**. A refusal arrives as `finish_reason: content_filter`: drop it, re-route the segment, never retry with a reworded prompt. |
| Reasoning tokens share `max_tokens` | `max_tokens = target_tokens + model.max_tokens_slack`. Visible length is controlled by the sentinel `<<END_STEP>>` plus a client-side cut. `length` with empty content: retry once with doubled `max_tokens`. |
| `stop` support: Fable yes; GPT-6.2 Sol no; Neon only on `google-vertex` | Send `stop` only if `model.supports_stop`. Always detect the sentinel client-side; then close the stream once its usage chunk is read, or at once if the model runs on past it (short grace window). |
| Abort stops billing on anthropic/openai/azure, **not on Google** | `provider: {only: [...]}` pinned per model (Fable `[anthropic, azure]`, Sol `[openai, azure]`, Neon `[google-vertex]`). Fallbacks stay enabled: `only` alone keeps routing on these providers, while `allow_fallbacks: false` without `order` would serve only the top-ranked one (no failover). Keep Neon's `max_tokens_slack` tight (1536). An aborted stream has no usage chunk: keep its generation id (`X-Generation-Id`) and report the run's usage as partial. |
| No logprobs on any of the three | No white-box switching signal (Co-LLM, CITER, R2R and others don't apply). Routing uses an external index + text classifiers. |
| Caching: Anthropic needs explicit `cache_control` (≤4 breakpoints, 20-block lookback); OpenAI is automatic (+`prompt_cache_key`); Gemini 4 is implicit only, ≥4096 tokens, re-bills 2,050 tokens on small appends | Request bodies are **append-only**: static system prompt and one content block per committed segment, never edited. Anthropic: `cache_control` on the last segment block (plus the system block). OpenAI: `prompt_cache_key = run_id + model`. |
| Temperature: only Neon accepts it | Never send temperature to Fable or Sol. |
| Existing benchmark data (apoorvumang/knowledge-cutoff, report_answers.json, 2026-11-23) only reaches 2026-06. Sol is freshest overall (direct 4PL τ: Fable 2026-02.8, Sol 2026-04.75, Neon 2026-03.0). Per-topic cells are too sparse | Ship a **prior index** fitted from that data, with heavy shrinkage. A new topic-stratified probe set (pipeline below, extended to 2026-11) is needed to find topic reversals. Be honest that always-Sol is the baseline to beat. |

Month axis: `timeutil` month index, 0 = 2024-01, 34 = 2026-11. τ is fractional: 27.75 means three quarters of the way through 2026-04.

## 2. Package layout and ownership

```
cutoffmix/
  types.py timeutil.py taxonomy.py      FROZEN (integrator only)
  config.py  data/default.yaml          WP1
  prompts.py                            WP1  (relay prompt templates + segment post-processing)
  backends/__init__.py base.py          WP1
  backends/openrouter.py                WP1
  backends/mock.py                      WP1
  router/__init__.py                    WP2
  router/features.py topics.py gate.py  WP2
  router/index.py policy.py router.py   WP2
  router/train.py train_gpu.py          WP2  (+ scripts/train_router_gpu.sh)
  bench/__init__.py io.py               WP3
  bench/kc_import.py                    WP3
  bench/sources/wikidata.py wikipedia.py WP3
  bench/probegen.py bench/prompts.py    WP3
  bench/runner.py grading.py            WP4
  bench/estimate.py simulate.py         WP4
  relay.py server.py cli.py web/index.html  WP5
tests/test_<module>.py                  owner of the module
```

Runtime deps: `numpy`, `httpx`, `pyyaml`. Optional: `fastapi`/`uvicorn` (server), `sentence-transformers` (embedding
classifier), `torch`/`transformers` (GPU router). Tests use `pytest` and `pytest-asyncio` (`asyncio_mode=auto`) and
**must not hit the network**: use fixtures, mocks and `httpx.MockTransport`. Python ≥3.10, typed, docstrings, no
dead code. Every module must import cleanly without the optional deps.

## 3. Contracts

### 3.1 config.py (WP1)

```python
@dataclass
class RelayConfig:
    segment_tokens: int = 128          # K, visible tokens asked per segment
    hard_cut_chars: int = 600
    max_segments: int = 60
    window_segments: int = 3           # routing window = question + last N segments
    strategy: str = "sequential"       # "sequential" | "speculative"
    speculative_gate: float = 0.5      # draft with all models only if r(W) >= gate
    answer_mode: str = "routed"        # "routed" (policy picks the answer model) | "inline" (text after FINAL ANSWER:)
    answer_max_tokens: int = 1024
    sentinel: str = "<<END_STEP>>"
    final_marker: str = "FINAL ANSWER:"
    inject_date: bool = True
    max_retries: int = 2
    disable_after_refusals: int = 2    # a model that refuses this often is excluded for the rest of the run
    first_content_timeout_s: float = 60.0  # per attempt: wall clock until the first visible text
    segment_timeout_s: float = 120.0   # per attempt: the whole stream (keep-alive comments reset neither)

@dataclass
class RouterConfig:
    policy: str = "softmax"            # softmax | thompson | argmax | fixed:<key> | round_robin | random
    T0: float = 0.5                    # months
    beta: float = 0.25                 # stickiness bonus for the current model, months
    kappa: float = 1.0                 # uncertainty penalty
    lam: float = 0.5                   # stale-assertion hazard penalty, months per unit hazard; a tie-breaker
                                       # (lam <= T0), so it cannot overturn a cutoff gap the index resolves
    r_min: float = 0.05
    delta: float = 0.5                 # argmax hysteresis margin, months
    ema: float = 0.5                   # topic-posterior EMA weight on the newest window
    index_path: str | None = None      # None -> packaged cutoffmix/data/index_prior.json (if present)
    probe_preference: tuple[str, ...] = ("current_d", "current", "recall", "mcq")
    classifier: str = "auto"           # auto | keyword | linear | embedding
    classifier_path: str | None = None
    gate_path: str | None = None       # trained gate (router train: gate_linear.npz); None -> lexical gate
    seed: int | None = None

@dataclass
class AppConfig:
    models: list[ModelSpec]
    relay: RelayConfig
    router: RouterConfig
    today: str | None = None           # ISO date; None -> date.today()
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_title: str = "cutoffmix"
    def model(self, key: str) -> ModelSpec
    def model_keys(self) -> list[str]
    def today_str(self) -> str

def load_config(path: str | None = None, overrides: dict | None = None) -> AppConfig
    # path None -> packaged cutoffmix/data/default.yaml; overrides = nested dict deep-merged ("relay": {...})
def default_config() -> AppConfig
```

`data/default.yaml` holds the three models as in §1: keys `claude-fable-5.2`, `gpt-6.2-sol`, `gemini-4-neon`. Colours:
`#d97757` (Fable), `#10a37f` (Sol), `#4285f4` (Neon). GPT gets `extra_body: {verbosity: low}`. Never send `verbosity`
to Claude.

### 3.2 prompts.py (WP1): relay templates and post-processing

```python
def relay_system_prompt(target_tokens: int, today: str | None, sentinel: str, final_marker: str) -> str
def relay_user_blocks(question: str, segments: list[str]) -> list[dict]
    # [{"type":"text","text": HEADER(question)}, one block per segment (segment text + "\n\n"), {"type":"text","text": TAIL}]
    # HEADER/TAIL constant apart from the question; with no segments a single "(no notes yet)" block.
def answer_system_prompt(today: str | None) -> str
def answer_user_blocks(question: str, segments: list[str]) -> list[dict]
def trim_overlap(notes_tail: str, new: str, min_len: int = 12) -> str
def trim_to_sentence(text: str, max_chars: int) -> str
    # a dot after an abbreviation (z. B., bzw., e.g., Okt., "No. 5"), an initial or a standalone 1-2 digit ordinal
    # ("zum 24. November") is no sentence end; years ("2026.") are. No sentence end in the cut: last whitespace.
def strip_preamble(text: str) -> str          # "Next paragraph:", "Continuation:", code fences, leading quotes
    # also template echoes ("Next result:", "Zwischenergebnis:", "Nächster Schritt:"), not "Correction:"/"Korrektur:"
def clean_segment(raw: str, notes_tail: str, *, sentinel: str, hard_cut_chars: int) -> tuple[str, bool]
    # -> (clean text, saw_sentinel). Cut at sentinel, strip preamble, trim echoed overlap, hard-cut, trim to sentence.
def cut_at_sentinel(raw: str, sentinel: str) -> tuple[str, bool]   # text before the first exact or mangled sentinel
def complete_sentences(text: str) -> str      # up to the last sentence end ("" if none): for cut-off paragraphs
def notes_echo_window(segments: list[str]) -> str   # the end of the notes (<= 8000 chars) a model may repeat
def split_final(text: str, final_marker: str) -> tuple[str, str | None]   # (notes part, answer part or None)
```

System prompt (template B, static for the whole run; nothing in it may vary per segment):
"Several analysts are jointly working through the problem in the user message; their shared notes so far are included.
Contribute the next intermediate result: state it and the evidence or calculation that supports it, in about {K} tokens,
written so it can be appended to the notes as the next paragraph. Use the same language as the notes (or as the
problem if there are no notes yet). If an earlier note is wrong, start with 'Correction:' and fix it. Prefer the most
recent facts you know and say when a fact may have changed. Output only that paragraph, then {sentinel}. If the notes
already reach a conclusion, write '{final_marker}' followed by the answer. Today's date: {today}."

### 3.3 backends (WP1)

`backends/base.py` re-exports `Backend`, `StreamEvent`, `ChatResult` and defines
`class BackendError(Exception)` with `status: int | None`, `error_type: str | None`, `retryable: bool`.

```python
# backends/openrouter.py
def parse_sse(lines: Iterable[str]) -> Iterator[dict]       # pure; skips ':' comments and blank lines, stops at [DONE]
class OpenRouterBackend:                                       # implements types.Backend
    def __init__(self, api_key: str | None = None, *, base_url: str = "https://openrouter.ai/api/v1",
                 timeout: float = 120.0, app_title: str = "cutoffmix", referer: str | None = None,
                 client: httpx.AsyncClient | None = None, max_retries: int = 3)
        # api_key None -> os.environ["OPENROUTER_API_KEY"] at first request; missing -> BackendError (not at import)
    def build_continuation_body(self, model: ModelSpec, req: ContinuationRequest) -> dict
    async def stream_continuation(self, model, req) -> AsyncIterator[StreamEvent]
        # yields kind="text" deltas as they arrive (raw, uncleaned); nothing after req.sentinel is emitted. After
        # the sentinel it reads on for a grace window to collect the finish and usage chunks (Sol writes the
        # sentinel as its last content), closing the connection when the usage chunk arrives, when more than
        # SENTINEL_GRACE_CHARS (32) of further content arrive, or after SENTINEL_GRACE_S (2.0 s); failures in that
        # window do not fail the segment. It also stops if model.cancellable and the text exceeds
        # req.hard_cut_chars*1.5 plus len(notes_echo_window(req.segments)) (an echo of the notes may come first; a
        # non-cancellable model is read on: closing would not stop its billing; the relay bounds it); final
        # kind="done" with usage (the usage chunk if received, plus "generation_id" from X-Generation-Id or the
        # chunk id when known), finish_reason ("stop"|"length"|"content_filter"|"sentinel"|"cut"|"error"), and
        # StreamEvent.text = "" ; on errors kind="error" with error message (never raise inside the iterator after
        # the first yield; raising BackendError before the first yield is ok for HTTP errors).
        # The first text or reasoning event also carries usage={"generation_id": ...} (when known), so a caller
        # that closes the stream before "done" (final marker, mangled sentinel, overflow) keeps the id.
        # Chunks of an unexpected shape (choices [null], a non-object delta, content as a list of parts) are read
        # tolerantly and never raise.
        # Refusal text (delta.refusal) with no content counts as finish_reason "content_filter".
        # Reasoning deltas (delta.reasoning) -> kind="reasoning" (streamed for Fable despite reasoning.exclude; the
        # relay drops them, and error messages never quote raw payloads, so they cannot reach notes or traces).
        # Mid-stream error chunk (top-level "error", finish_reason "error") -> kind="error".
    def build_chat_body(self, model: ModelSpec, req: ChatRequest) -> dict
    async def chat(self, model, req) -> ChatResult   # non-streaming; retries 408/429/5xx/502/503 with Retry-After + backoff
    async def aclose(self) -> None
```

Body rules: `model.provider_id`, `stream`, `max_tokens`, `reasoning` (from ModelSpec), `provider` (ModelSpec
routing), `stop: [sentinel]` only if `supports_stop`, `temperature` only if `supports_temperature` and not None,
`extra_body` merged, `session_id`/`prompt_cache_key` from `run_id` (+ model key) when `cache_style` is "openai",
`cache_control: {"type":"ephemeral"}` on the system block and on the last segment block when `cache_style` is
"anthropic". Messages: `[{"role":"system","content":[{"type":"text","text":SYS}]}, {"role":"user","content":blocks}]`.
Headers: `Authorization: Bearer`, `X-Title: app_title`, optional `HTTP-Referer`. No personal data anywhere.

```python
# backends/mock.py
@dataclass
class MockModelTruth:
    key: str
    tau: dict[str, float]          # true per-topic virtual cutoff (month index); missing topics -> default_tau
    default_tau: float
    slope: float = 0.6
    plateau: float = 0.95
    stale_rate: float = 0.7        # P(assert stale state | unknown); else abstain
    latency_s: float = 0.02        # per streamed chunk
    refusal_rate: float = 0.0
def default_mock_truths() -> dict[str, MockModelTruth]
    # keys = the three config keys; Sol freshest on most topics, but with clear reversals
    # (e.g. Fable on ai_ml and computing_software, Neon on science_research and space_astronomy)
class MockBackend:                 # implements types.Backend, fully deterministic given seed
    def __init__(self, truths: dict[str, MockModelTruth] | None = None, *, seed: int = 0,
                 classifier: "TopicClassifier | None" = None, segments_to_answer: int = 6,
                 events: dict[str, Event] | None = None, today: str | None = None)
    # stream_continuation: writes a plausible paragraph about the dominant topic of question+notes (keyword
    #   classifier from cutoffmix.router.topics if none given; import lazily), mentioning what the model "knows as of"
    #   month_label(its tau for that topic); streams ~8-char chunks with asyncio.sleep(latency_s); appends the
    #   sentinel; after segments_to_answer total segments emits final_marker + a short answer; honours refusal_rate.
    # chat: if req.metadata has "event_id" and "probe" and the event is known -> answers per truth:
    #   P(correct) = g + (plateau - g) * sigmoid((tau_mz - t_event)/slope), g = 0.25 for mcq else 0;
    #   otherwise stale answer (event.reject_aliases[0] or a generic "still in office"/"is alive") with stale_rate,
    #   else abstain text. mcq returns a letter. Deterministic per (seed, model, event, probe).
```

### 3.4 router (WP2)

```python
# router/features.py
def tokenize(text: str) -> list[str]                      # lowercase word tokens, unicode-aware (German umlauts)
def hashed_features(text: str, dim: int = 2**18) -> dict[int, float]   # word uni+bigrams + char 3-5grams, l2-normalised
# router/topics.py
class KeywordTopicClassifier:      # zero-training default; implements types.TopicClassifier
    def __init__(self, topics=TOPICS, *, prior: float = 0.3, sharpness: float = 1.5)
    topics: list[str]
    def posterior(self, text: str) -> list[float]         # keyword+seed matching with smoothing; uniform if no signal
class LinearTopicClassifier:       # multinomial logistic regression on hashed features (numpy, CPU)
    topics: list[str]
    @classmethod
    def train(cls, texts, labels, *, dim=2**18, epochs=8, lr=0.5, l2=1e-6, seed=0) -> "LinearTopicClassifier"
    def posterior(self, text) -> list[float]
    def save(self, path) / @classmethod load(cls, path)    # .npz
class EmbeddingTopicClassifier:    # optional sentence-transformers; ImportError message if missing
def load_classifier(kind: str = "auto", path: str | None = None) -> TopicClassifier
    # auto: linear if path exists, else keyword
# router/gate.py
class TimeSensitivityGate:                                 # __init__(..., today=None): which years are recent
    def score(self, text: str) -> float                   # r(W) in [0,1]; lexical baseline (temporal cues EN/DE,
                                                           # years from today-2 to today+1, "current/latest/now/
                                                           # aktuell/derzeit", office/role nouns, capitalised-entity
                                                           # density, news-topic mass). The current or previous year
                                                           # counts as a strong temporal cue ("this year") unless the
                                                           # text has math cues.
    @classmethod train(...)/save/load(path, *, today=None) # optional LR on hashed features
def load_gate(path: str | None = None, *, today=None) -> TimeSensitivityGate   # router.gate_path, else lexical
# router/index.py
@dataclass
class ProbeIndex:
    probe: str
    tau: np.ndarray        # [M,T] posterior mean virtual cutoff (month index)
    var: np.ndarray        # [M,T]
    hazard: np.ndarray     # [M,T] P(outdated|incorrect | item after cutoff)
    abstain: np.ndarray    # [M,T]
    n: np.ndarray          # [M,T] graded items used
    draws: np.ndarray | None   # [B,M,T] posterior/bootstrap draws of tau
    global_tau: np.ndarray # [M]
    slope: np.ndarray      # [M]
    plateau: np.ndarray    # [M]
class CutoffIndex:
    def __init__(self, models: list[str], topics: list[str], probes: dict[str, ProbeIndex], meta: dict)
    models; topics; probes; meta
    def primary(self, preference=("current_d","current","recall","mcq")) -> ProbeIndex
    def scores(self, posterior: dict[str, float], *, probe: str | None = None, kappa=1.0, lam=0.5,
               models: list[str] | None = None) -> dict[str, float]
        # U_m = Σ_z π_z τ_mz − κ·sqrt(Σ_z π_z² Var_mz) − λ·Σ_z π_z H_mz ; topics absent from the index use the
        # model's global_tau (with var = mean var); models absent from the index -> -inf (never chosen by argmax)
    def sample_scores(self, posterior, rng: np.random.Generator, *, probe=None, lam=0.0,
                      models=None) -> dict[str, float]
        # one Thompson draw: Σ_z π_z τ^(b)_mz (draws if present, else Normal(tau, var)) − λ·Σ_z π_z H_mz;
        # the router passes cfg.lam so Thompson ranks a window like softmax/argmax (no κ: the draw is the uncertainty)
    def p_best(self, posterior, *, probe=None, models=None) -> dict[str, float]
    def table(self, probe=None) -> list[dict]             # rows for UI/CLI: topic, model, tau, tau_label, sd, n, hazard, p_best
    def to_json(self) -> dict ; @classmethod from_json(cls, d) ; save(path) ; @classmethod load(path)
def load_index(path: str | None = None) -> CutoffIndex | None    # None -> packaged cutoffmix/data/index_prior.json if it exists
# router/policy.py
@dataclass
class PolicyDecision: model: str; decision: Decision; probs: dict[str, float]
class RoutingPolicy(Protocol):
    def choose(self, scores: dict[str, float], *, current: str | None, r: float, available: list[str],
               rng: random.Random, sample_scores: Callable[[], dict[str, float]] | None = None) -> PolicyDecision
SoftmaxPolicy(T0, beta, r_min)  # P ∝ exp((U_m + β·1[m=current]) / T), T = T0/max(r, r_min); decision "sample"
                                # or "hold" when the sampled model == current
ThompsonPolicy()                # argmax of sample_scores(); decision "sample"/"hold"
ArgmaxHysteresisPolicy(delta)   # switch only if best − current ≥ delta; "argmax" or "hold"
FixedPolicy(model) RoundRobinPolicy() RandomPolicy()    # baselines; decision "forced"/"sample"
def make_policy(name: str, cfg: RouterConfig) -> RoutingPolicy
# router/router.py
@dataclass
class RouteDecision:
    model: str; decision: Decision; probs: dict[str, float]; scores: dict[str, float]
    posterior: dict[str, float]; r: float
class Router:
    def __init__(self, models: list[str], *, index: CutoffIndex | None, classifier: TopicClassifier | None = None,
                 gate: TimeSensitivityGate | None = None, policy: RoutingPolicy | None = None,
                 cfg: RouterConfig | None = None, seed: int | None = None)
    @classmethod
    def from_config(cls, cfg: AppConfig) -> "Router"     # index, classifier and load_gate(gate_path, today)
    def reset(self) -> None
    def window(self, question: str, segments: list[str], n: int) -> str
    def decide(self, window_text: str, *, current: str | None, available: list[str] | None = None,
               update: bool = True) -> RouteDecision
        # posterior = EMA(prev, classifier(window)) (weight cfg.ema on the new one) if update else uses it without
        # storing; r = gate.score(window); scores = index.scores(...) (all-zero if index None); policy.choose(...)
    def score_draft(self, window_text: str, draft: str, model: str) -> float
        # U_model on posterior(window + draft) -- look-ahead score for speculative drafting
```

Training (CPU, `router/train.py`): `train_topic_classifier(texts, labels, out_path)` and
`train_gate(texts, labels, out_path)`; `build_training_corpus(events_paths) -> (texts, labels)` uses Event
facts/questions + taxonomy seeds; `cutoffmix router train`. GPU (`router/train_gpu.py`, optional v2): fine-tune a
multilingual encoder (default `jhu-clsp/mmBERT-base`, fallback `answerdotai/ModernBERT-base`) with heads topic(15) +
time-sensitivity(1) + per-model currentness(M) on probe outcomes; hard wall-clock guard (`--budget-min 55`), bf16,
torchrun DDP for 1-10 GPUs, saves `router_models/encoder/`. Import torch lazily; the module must import without torch.

### 3.5 bench (WP3 = data, WP4 = evaluation)

```python
# bench/io.py
def read_jsonl(path) -> list[dict]; def write_jsonl(path, rows); def append_jsonl(path, row)
def load_events(path) -> list[Event]; def save_events(path, events)
def load_results(path) -> list[ProbeResult]; def save_results(path, results)
# bench/kc_import.py  (apoorvumang/knowledge-cutoff, data CC-BY-4.0; the code has no license -> reimplemented)
KC_EVENTS_URL = "https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark/resolve/main/events.jsonl"
KC_ANSWERS_URL = "https://raw.githubusercontent.com/apoorvumang/knowledge-cutoff/main/report_answers.json"
def download_kc(dest_dir) -> tuple[Path, Path]
def tag_topic(kc_event: dict) -> str                   # taxonomy topic id (rules + KeywordTopicClassifier fallback)
def convert_events(kc_events: list[dict]) -> list[Event]   # question_current := question_direct, mcq passthrough,
                                                           # origin="kc", control for control_alive/fake_event
def convert_answers(report_answers: dict, events: dict[str, Event], models: list[str] | None = None) -> list[ProbeResult]
    # probes: direct->"current", mcq->"mcq"; labels c->correct, w->outdated (death/office_change) else incorrect,
    # a->abstain, u->ungraded; grader="kc"
# bench/sources/wikidata.py
USER_AGENT = "cutoffmix/0.1 (https://github.com/Erikiss/EscapeTheCutOff)"
def run_sparql(query: str, *, client=None, retries=4) -> list[dict]   # POST, honours Retry-After, flattens bindings
QUERIES: dict[str, str]                                 # heads (Q2), elections (Q7), awards (Q8), sports (Q3)
def events_from_heads(rows) -> list[Event]             # office_change; current: "Who is the {office}?" (no date);
                                                       # recall: "Who became {office} in {Month YYYY}?"; mcq incl.
                                                       # predecessor (P1365) as stale option; reject_aliases=[predecessor]
def events_from_elections(rows, ctx, *, as_of) / events_from_awards(rows, *, since, as_of)
    / events_from_sports(rows, ctx, *, editions, as_of) -> list[Event]
    # question_current ("most recent ...") is empty when a later edition is known (rows, P156 successor, or every
    # edition of the competition from the comp_editions query) or, for series recurring at most every 400 days
    # (all curated awards; sports with a P155 period or a "2024-25" season label), when the next edition is
    # overdue by 30 days as of `as_of`. Sports editions use their most specific competition (fewest editions,
    # P179 series first); one result filed under two items (same winner within 3 days) is one event. Work awards
    # (best film/series/album/song, festival prizes) answer the work, people credited via P1686 are aliases.
    # MCQ distractors only of the answer's type (same election/country, award family and person-vs-work, sport and
    # team kind, office kind); with fewer than 3 the event has no MCQ. Team aliases add short names
    # ("Manchester United", "Spain", "Kansas City Chiefs").
def build_wikidata_events(start="2024-01-01", end=None, *, client=None) -> list[Event]
# bench/sources/wikipedia.py
def fetch_ce_days(start: date, end: date, *, client=None) -> dict[date, str]    # batched revisions API, 50 titles/req,
                                                                                # maxlag=5, Retry-After, concurrency 1
def parse_ce_wikitext(text: str, day: date) -> list[dict]   # {date, ce_category, story_path, text, links, sources}
def fetch_deaths_month(year, month, *, client=None) -> str ; def parse_deaths(text, year, month) -> list[dict]
def ce_item_topic(item: dict) -> str
# bench/probegen.py  (LLM-generated probes for free-text Current-events items; needs a key)
async def generate_probes(backend, model: ModelSpec, items: list[dict], *, concurrency=4) -> list[Event]
def validate_event(e: Event) -> list[str]               # V3 gates: answer not in question, no year/month tokens in
                                                        # question_current, 4 unique mcq choices, answer letter valid
# bench/prompts.py
PROBE_SYSTEM_NEUTRAL (current/recall; kc-compatible wording), PROBE_SYSTEM_MCQ (kc wording)
def probe_messages(event: Event, probe: str, *, today: str | None, deployment_system: str | None) -> list[dict]
    # current_d: system = relay deployment preamble with today's date + answer instruction; user = question_current
JUDGE_SYSTEM (labels correct/outdated/incorrect/abstain + hedged; "As of <date>, <pre-event state>" = outdated, hedged)
def judge_messages(event, probe, response) -> list[dict]
# bench/grading.py (WP4)
def extract_letter(response: str) -> str | None        # kc regex order
def grade_mcq(event, response) -> Grade
def grade_rule(event, probe, response) -> tuple[Grade, bool] | None   # alias / reject-alias / abstain phrases; None = unsure
    # a correct or stale name in succession context ("Y, who replaced X", "her predecessor was X") is history, not
    # an assertion; a correct name next to an unknown name and a succession cue -> None (judges)
async def grade_with_judges(backend, judges: list[ModelSpec], event, probe, response) -> tuple[Grade, bool, str]
    # two judges, third on disagreement; parse {"label","hedged"} JSON; failures -> "ungraded"
# bench/runner.py (WP4)
async def run_probes(backend, models: list[ModelSpec], events: list[Event], probes: list[str], out_path, *,
                     today=None, concurrency=8, resume=True, use_batch_variant=False, judges=None,
                     deployment_system=None, max_tokens=4096, progress=None) -> list[ProbeResult]
    # use_batch_variant=True raises ValueError: OpenRouter serves ':batch' ids only via its async Batch API
    # (/api/v1/batches), which is not implemented; the backends call /chat/completions
def estimate_cost(models, events, probes, *, out_tokens=600, in_tokens=200, pricing=None, batch=False,
                  max_tokens=4096) -> dict
    # no batch discount; "max_usd" = bound when every call uses max_tokens (out_tokens=600 is unmeasured: pilot first)
# bench/estimate.py (WP4)
def fit_index(events: list[Event], results: list[ProbeResult], *, models=None, topics=None, probes=None,
              priors: dict[str, float] | None = None, prior_sd=3.0, bootstrap: int = 200, seed: int = 0,
              grid_step: float = 0.05) -> CutoffIndex
    # §4 below; excludes controls, high predictability, ungraded
def kc_style_cutoffs(events, results, *, threshold=0.5, min_tail=2) -> dict    # last_above / crossover per model+probe
# bench/simulate.py (WP4)
def synthetic_events(topics, months, per_cell, seed) -> list[Event]
def simulate_results(truths: dict[str, MockModelTruth], events, probes, seed) -> list[ProbeResult]
def recovery_report(index: CutoffIndex, truths, probe) -> dict   # RMSE, rank agreement, P_best calibration
```

## 4. Estimator (bench/estimate.py)

For each probe q separately. Rows: graded results on non-control, low/medium-predictability events. `y = 1` iff
grade == correct. `t_i` = `timeutil.parse_fractional_month(event.date)`.

1. **Model-level 4PL.** `P(y=1) = g + (a − g)·σ((τ − t)/s)`, with g = 0.25 for mcq and 0 otherwise. Grid MAP over
   τ ∈ [−6, 40] (step `grid_step`), s ∈ {0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0}, a ∈ {0.55, …, 0.99}.
   Priors: τ ~ N(priors[m] or the data median, prior_sd²), log s ~ N(log 1, 0.5²), a ~ Beta(9, 1). Gives
   global_tau, slope s_m and plateau a_m.
2. **Per-cell posterior.** For each (m, z), fix s_m and a_m and compute the 1-D grid posterior of τ_mz:
   likelihood × N(τ; global_tau_m + γ_z, σ_ε²).
   - γ_z = topic offset shared by all models: a precision-weighted mean over models of (cell MLE − global_tau_m),
     shrunk to 0 with σ_γ = 2.
   - σ_ε by method of moments on cells with n ≥ 8, clipped to [0.5, 3]. Default 1.5 if fewer than 3 such cells.
   - tau = posterior mean; var = posterior variance. Empty cells give the prior.
3. **Draws.** Paired cluster bootstrap: resample events with replacement within each topic, using the same resample
   for all models, and recompute the step-2 posterior means. That gives draws[B, M, T]; B = `bootstrap`. If
   bootstrap = 0, draw from N(tau, var).
4. **Hazard.** Among graded items with t_i > tau_mz + 2·s_m (k = outdated + incorrect of n): two-level Beta shrinkage
   with 5 pseudo-observations per level. Pooled over all models H_pool = (k + 1)/(n + 2); model level
   H_m = (k_m + 5·H_pool)/(n_m + 5); cell H_mz = (k_mz + 5·H_m)/(n_mz + 5). A model with few items after its cutoff
   (the freshest one, when the data end near it) thus gets about the pooled rate. Abstain rate is computed the same
   way.
5. `meta`: created (ISO), sources, n_events, n_results, probe list, method string, priors, and the month axis
   definition.

## 5. Relay (relay.py, WP5)

```python
class Relay:
    def __init__(self, cfg: AppConfig, backend: Backend, router: Router, *, models: list[ModelSpec] | None = None)
    async def run(self, question: str, *, run_id: str | None = None) -> AsyncIterator[RelayEvent]
    async def run_to_trace(self, question: str, **kw) -> RelayTrace
```

Loop, with i = segment index:
1. W = router.window(question, segments, cfg.relay.window_segments).
2. Sequential strategy: `d = router.decide(W, current=current, available=avail)`. Emit `segment_start` with model,
   decision, posterior, scores and probs. Stream `backend.stream_continuation(model, req)`. Buffer the first 40 chars
   so that preamble or echoed overlap can be stripped, then emit `delta` events. Never emit text after the sentinel,
   nor trailing whitespace at the end of the segment.
   A model may repeat earlier notes (whole notes, or the end of the last one) before its new paragraph: the buffer is
   held while the text is still a piece of `notes_echo_window(notes)`, the hard cut counts from where the new text
   starts, and the repeat is stripped. An attempt that is only a repeat (possibly cut off) is `empty`.
   At the end, `clean_segment()` and emit `segment_end` with the final text. The final text is authoritative; the UI
   replaces the streamed text with it.
3. Speculative strategy, when r(W) ≥ speculative_gate and more than one model is available:
   - Draft with all available models concurrently (non-emitting).
   - Score each draft with `router.score_draft(W, draft, m)`, then `policy.choose` over those scores with
     current = current model. Decision "speculative".
   - Emit the chosen draft as segment_start, one delta and segment_end, plus `data={"drafts": {m: text}}`.
   - Below the gate, use the sequential path. So do the policies that ignore scores (`fixed:<key>`, `round_robin`,
     `random`): they would pick the same model with or without drafts, so the other drafts would be paid for nothing.
4. Errors:
   - `finish_reason == content_filter`: refusal. Drop the text, count it, exclude that model for this segment, and
     re-decide.
   - `length` with empty text: retry once with doubled max_tokens. A think paragraph cut off (`length`, `cut`,
     `overflow`, no sentinel) loses its incomplete last sentence; with no complete sentence it counts as empty.
   - Nothing after the first sentinel counts, also a mangled one: a final marker written after it is no answer.
   - error or exception: retry ≤ max_retries with backoff, then exclude the model for this segment.
   - Each attempt (drafts included) has a wall-clock budget that keep-alive comments cannot reset:
     `first_content_timeout_s` until the first text, `segment_timeout_s` in total. Running out of it is a retryable
     error (`finish_reason` "timeout"; its usage is unknown, so it is listed in `usage_missing`), unless the
     paragraph was already complete (sentinel, marker, overflow), which is then kept.
   - If no model is left: emit `error` and stop. Exception: in the think phase, if every remaining model returned
     nothing usable (`empty`, e.g. it stopped at once or wrote only the sentinel) and the notes are not empty, the
     notes count as concluded and the answer phase runs.
5. If a segment contains final_marker: commit the text before it, then go to the answer phase.
   - answer_mode "inline": the text after the marker is the answer.
   - answer_mode "routed": the router picks the answer model on the whole-notes window; one answer call with
     `answer_*` prompts, streamed as `phase` and then `delta` events with phase "answer".
6. Stop when max_segments is reached and force the answer phase. Emit `done` with
   `data = trace.to_dict() | {"usage": per-model usage, "latency": ...}`. A call whose stream was closed before its
   usage chunk is listed in `usage_missing` (with its `generation_id`, which the backend also reports on its first
   event, so a stream the relay closes itself keeps it), and `usage_complete` is then false: the usage and cost
   totals are a lower bound and clients label them (cost and tokens) as partial. `Segment.aborted` is true whenever
   the stream was closed before the provider finished (by the relay, or by the backend at the cut or at the
   sentinel without the usage chunk); a sentinel stream whose usage chunk arrived in the grace window is not. When
   the relay closes a stream itself, `Segment.note` and `finish_reason` say why: `marker`, `mangled`, `overflow`,
   `sentinel` (a runaway past it) or `timeout`.

## 6. Server, UI and CLI (WP5)

- `server.py` (FastAPI, `create_app(cfg, backend=None, mock=False)`):
  - `GET /` returns the UI. `GET /api/config` returns models with colours and defaults. `GET /api/index` returns the
    index table. `POST /api/relay` returns rich SSE events (`data: RelayEvent.to_dict()`).
  - OpenAI-compatible `POST /v1/chat/completions`, stream and non-stream. Model ids are `cutoffmix`,
    `cutoffmix-speculative` and `cutoffmix-fixed-<key>`. The question is the conversation flattened into
    PROBLEM text. Thinking goes in `delta.reasoning` plus `delta.cutoffmix` metadata; the answer goes in
    `delta.content`. Non-stream returns `message.content`, `message.reasoning` and a `cutoffmix` trace. The
    `usage` object carries `cutoffmix_partial: true` when the run's usage is incomplete.
  - `GET /v1/models`.
  - Per request, options `policy`, `strategy` and `segment_tokens` can override the config.
  - The POST endpoints start paid calls, so they take only `Content-Type: application/json` (415 otherwise, so a
    cross-site page cannot send them without a CORS preflight, which is never granted) and refuse a foreign
    `Origin` (403): only the server's own (`Host` or `X-Forwarded-Host`) or one passed via `allowed_origins` /
    `serve --allow-origin` (compared without the scheme's default port). Every request must name a served host in
    `Host` (403 otherwise), so a DNS-rebinding page cannot reach the server: `localhost`, `127.0.0.1`, `::1`, the
    hosts of the allowed origins and `allowed_hosts` / `serve --allow-host` (`serve` adds its `--host` unless it is
    `0.0.0.0`/`::`; `*` allows any). A client disconnect stops the relay on every path (stream and non-stream). The
    flattened `/v1` conversation is capped at 20,000 characters like the `/api/relay` question (400). The UI's
    `?q=` link only prefills the question; a run always needs a click.
- `web/index.html`: a single self-contained file with no external network. It has the question box, strategy and
  policy controls, and a live stream with segments coloured per model (legend, hover shows topic posterior, r and
  probs). It also shows a topic bar for the current window, an index heatmap (topics × models, τ as YYYY-MM, P_best
  highlighted), and stats (switches, share per model, latency). Dark and light mode.
- `cli.py` (`cutoffmix` entry point, argparse):
  - `run "question" [--mock] [--policy] [--strategy] [--segments] [--json out] [--config]` prints coloured
    segments to the terminal.
  - `serve [--host] [--port] [--mock] [--allow-origin ORIGIN] [--allow-host HOST]`
  - `index show [--path] [--probe] [--config]` (default index and probe from `router.index_path` and
    `router.probe_preference`, as routing and `/api/index` use them)
  - `bench import-kc --out DIR`
  - `bench build-wikidata --out FILE`
  - `bench build-wikipedia --out FILE --start --end [--probe-model KEY]`
  - `bench run --events --out --models --probes [--dry-run] [--batch] [--limit]` (`--batch` is refused outside
    `--dry-run`, see §3.5)
  - `bench grade`
  - `bench fit --events --results --out [--priors]`
  - `router train`
  - `simulate [--out DIR]`
  - `--mock` uses MockBackend everywhere. Without `--mock` and without `OPENROUTER_API_KEY`, print a clear message.
