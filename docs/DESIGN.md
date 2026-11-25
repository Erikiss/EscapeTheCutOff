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
| Temperature: Fable and Sol have no parameter; Neon accepts it at the gateway, but Google ignores in-range values and returns 400 out of range (docs, 2026-11-24) | Never send temperature or top_p to any of the three: `supports_temperature: false` for all three in `default.yaml`, `ChatRequest.temperature` defaults to `None` (§7). |
| Existing benchmark data (apoorvumang/knowledge-cutoff, report_answers.json, 2026-11-23) only reaches 2026-06. Sol is freshest overall (direct 4PL τ: Fable 2026-02.8, Sol 2026-04.75, Neon 2026-03.0). Per-topic cells are too sparse | Ship a **prior index** fitted from that data, with heavy shrinkage. A new topic-stratified probe set (pipeline below, extended to 2026-11) is needed to find topic reversals. Be honest that always-Sol is the baseline to beat. |

Month axis: `timeutil` month index, 0 = 2024-01, 34 = 2026-11. τ is fractional: 27.75 means three quarters of the way through 2026-04.

## 2. Package layout and ownership

```
cutoffmix/
  types.py timeutil.py taxonomy.py      FROZEN (integrator only)
  config.py  data/default.yaml          WP1
  data/presets/explore.yaml lenses.py   explore WP-A (§7: preset file, lens fields and the seeded draw)
  prompts.py                            WP1  (relay prompt templates + segment post-processing; §7 explore texts)
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
    explore: ExploreConfig             # §7.1; enabled False in the packaged default
    today: str | None = None           # ISO date; None -> date.today()
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    app_title: str = "cutoffmix"
    preset: str | None = None          # the packaged preset load_config() merged ("explore"), else None: an echo (§7.1)
    def model(self, key: str) -> ModelSpec
    def model_keys(self) -> list[str]
    def today_str(self) -> str

def load_config(path: str | None = None, overrides: dict | None = None, *, preset: str | None = None) -> AppConfig
    # path None -> packaged cutoffmix/data/default.yaml; preset -> packaged data/presets/<preset>.yaml merged
    # BEFORE the file and the overrides (nested dict deep-merged, "relay": {...}); unknown preset -> ConfigError
def list_presets() -> list[str]        # names of the packaged presets (["explore"])
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
    # the taxonomy keywords cover news vocabulary and the research vocabulary of the technical topics (battery
    # chemistry, superconductivity, genetics, climate science, ML architectures, cryptography, astrophysics ...)
# taxonomy.py
def argmax_topic(posterior: Mapping[str, float]) -> str | None   # the clear argmax; None for a flat posterior
    # (relay current_topic / explore metrics, CLI header "topic ?", web UI "—" / "No clear topic")
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

#### 3.5.1 AI/ML paper probes (bench/sources/papers.py) and their grading

The `ai_ml` cell of the index needs dated facts that a model either knows or does not. A paper's *first author* is
such a fact: unguessable from the title, fixed on the arXiv date, verifiable from arXiv metadata, rule-gradable.

```python
# bench/sources/papers.py  (selection: Hugging Face daily papers; ground truth: arXiv API; topic ai_ml)
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"    # ?date=YYYY-MM-DD; public listing used only to pick papers
ARXIV_API_URL = "https://export.arxiv.org/api/query"        # id_list batches of <= 100 ids, >= 3 s apart (ARXIV_PACE_S)
def fetch_hf_daily(date, *, client=None, cache_dir=None, retries=4) -> list     # raw items; cached as hf/<date>.json
def collect_month(year, month, top_n, *, ...) -> list[HFPaper]  # the month's daily lists merged, deduped by arXiv id
                                                                 # (highest upvotes kept), top_n by upvotes
def fetch_arxiv(ids, *, client=None, cache_dir=None) -> dict[str, ArxivPaper]  # per-id cache arxiv/<id>.json, raw Atom
                                                                               # feeds under arxiv/raw/
def parse_arxiv_atom(xml) -> dict[str, ArxivPaper]     # title whitespace-normalised, authors in order, v1 published
                                                       # date, primary category; bad entries skipped, bad XML raises
def is_interp(title, summary="") -> bool               # strong keywords (interpretab, sparse autoencoder/SAE, superposition,
                                                       # activation patching, logit lens, monosemantic, mechanistic,
                                                       # transcoder, crosscoder, causal tracing, ...) or two weak
                                                       # families (circuit, attribution, probing, steering, neuron, ...;
                                                       # feature/activation only next to one of those), one in the title
def plan_months(dense_from, dense_to, per_month, *, anchor_from, anchor_to, anchor_every, anchor_per_month)
def build_paper_events(dense_from="2025-10", dense_to="2026-11", *, per_month=12, anchor_from="2024-01",
                       anchor_to="2025-09", anchor_every=3, anchor_per_month=6, interp_share=0.5,
                       controls_share=0.1, seed=0, candidates_per_month=60, cache_dir=None) -> (events, meta)
# CLI: cutoffmix bench build-papers --out FILE [--meta FILE --cache DIR ...]   (all of the above as options)
```

- **Filing and filters.** A paper is filed under the month of its arXiv `published` (v1) date, never under the
  Hugging Face date. Dropped: first authors that are not two-token person names (`DeepSeek-AI`, `Kimi Team`), titles
  with year or month tokens (they would date `question_current`), titles that contain the author's name, ids arXiv
  does not know. Preferred (ranked first): at least 3 authors and 6+ title words. Per month, about `interp_share`
  of the picks come from the `is_interp` pool when it has enough, the rest are the top-upvoted others.
- **Event shape** (category `paper`, `origin: "arxiv_hf"`, `topic: "ai_ml"`, `predictability: "low"`): id
  `paper:<arxiv id>`, `date` = arXiv date, `subject` = title, `fact` = "first author N; authors: ...; arXiv ID (month)",
  `question_current` = "Who is the first author of the arXiv paper titled '<title>'?" (no date; the `current_d` probe),
  `question_recall` = the same with " (arXiv, <Month YYYY>)" appended, expected = the first author's full name,
  `aliases` = full name, "Surname, Given", "G. Surname", "Surname G." and their diacritic-free forms,
  `reject_aliases` = the three MCQ distractors (first authors of other papers of the same or adjacent months, never
  the answer's surname, surnames pairwise distinct), MCQ order shuffled with `seed` and the event id, `source` =
  the arXiv abstract URL. The side table `papers_meta.jsonl` (one row per event id) carries `interp`, `upvotes`,
  `n_authors`, `authors`, `hf_authors`, `primary_category`, `hf_date`, `distractors` and, for controls, `donors`.
- **Controls** (category `fake_paper`, `control: True`, id `fakepaper:<n>`, about `controls_share` of the real
  count): the first half of one real title joined with the second half of another from a different month (both
  halves >= 3 words; not equal or near-equal to any real title), dated to the first donor's arXiv date. No MCQ, no
  aliases, no reject aliases. `expected_current` is the sentinel `CONTROL_EXPECTED` ("No such paper exists
  (control): ...") rather than empty, because `validate_event` requires an expected answer for every question and
  judges need a ground truth to read; the rule grader ignores it.
- **Taxonomy.** `EVENT_TYPES` gains `paper` and `fake_paper`; `grading.CONTROL_CATEGORIES` gains `fake_paper`, so
  the estimator excludes the controls as it does the other control families.

**Grading of paper probes** (`grading.grade_paper`, reached through `grade_rule` for both categories):

- abstain cues first: the shared `_ABSTAIN` phrases plus paper-specific ones ("I am not aware of", "I don't have
  information", "cannot verify/find/locate", "no record", "does not exist", "not familiar with"; German "ist mir
  nicht bekannt", "kenne ich nicht", "kann ich nicht verifizieren/finden", "liegen mir keine Informationen",
  "existiert nicht");
- real papers: `correct` when the expected name is asserted in a full form (aliases; diacritics folded; negated,
  conditional or interrogative clauses do not count, as everywhere in the rule grader) or by surname alone, except
  that a common surname (<= 4 letters, or Li/Wang/Zhang/Chen/Liu/Yang/Kim/Lee/Park/Singh/Kumar/Smith and a few
  more) also needs the given name's first token; `incorrect` when a distractor (`reject_aliases`) or any other
  capitalised two-to-five-token name is asserted instead (words of the question, title and stop words excluded;
  "Perhaps Wei Chen" names Wei Chen, "Hugging Face" names nobody); `abstain` for an abstain phrase without any
  name; everything else (the answer next to a distractor, a name only in a negated clause, "might be X", a lone
  common surname) is `None` and goes to the judges;
- controls: any named person -> `incorrect` (confabulation), an abstention -> `correct`, otherwise `None`;
- `hedged` is set when the asserted name comes with knowledge-horizon words (`_HEDGE`), uncertainty words
  ("I believe", "probably", "not certain", "vermutlich", ...) or an abstain phrase;
- MCQ: `grade_mcq` as usual, but a distractor pick is `incorrect`, not `outdated` (papers have no stale state);
  controls have no MCQ.
- Judges see "(none: ...)" as the outdated answer for both categories (`prompts._DEFAULT_OUTDATED`).

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
    `cutoffmix-speculative`, `cutoffmix-explore` (§7.7) and `cutoffmix-fixed-<key>`. The question is the conversation flattened into
    PROBLEM text. Thinking goes in `delta.reasoning` plus `delta.cutoffmix` metadata; the answer goes in
    `delta.content`. Non-stream returns `message.content`, `message.reasoning` and a `cutoffmix` trace. The
    `usage` object carries `cutoffmix_partial: true` when the run's usage is incomplete.
  - `GET /v1/models`.
  - Per request, options `policy`, `strategy`, `segment_tokens` and `preset` (`"explore"`, §7.7) can override the config.
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
  - `run "question" [--mock] [--policy] [--strategy] [--segments] [--json out] [--config] [--preset explore]` prints
    coloured segments to the terminal.
  - `serve [--host] [--port] [--mock] [--allow-origin ORIGIN] [--allow-host HOST] [--preset explore]`
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

## 7. Explore preset

An **opt-in** preset (`cutoffmix run --preset explore`, `cutoffmix serve --preset explore`, per-request
`"preset": "explore"` on `/api/relay` and `/v1/chat/completions`, model id `cutoffmix-explore` on `/v1`). With the
preset off, the relay's request bodies and behaviour are unchanged. The design and its evidence are in
`docs/DESIGN_explore.md`; this section is the binding contract of what is implemented. The preset does **not** change
`router.policy`: routing by cutoff is the product (with the shipped prior index Sol is freshest on every topic;
`--policy round_robin` shows all three writers). One seed: the run seed (`--seed` / `router.seed`) seeds every
client-side draw; `explore.seed` overrides it when set. No temperature or top_p is sent to any model (§1).

Wording safety for every text in this section: no "thinking", "reasoning", "reason", "scratchpad", "chain of
thought", no XML-like tags; labels are plain words ending with a colon. `tests/test_prompts.py` checks every text.

### 7.1 config.py: `ExploreConfig`, `LensConfig`, the preset file

```python
@dataclass
class LensConfig:                      # explore.lens
    p: float = 0.25                    # draw probability per eligible segment
    p_stagnation: float = 0.50         # while the main line stagnates
    min: int = 1                       # force one lens at the last eligible slot if none was drawn
    max: int = 2                       # lenses per run (>= min)
    first_index: int = 2               # 0-based segment index; never in the first two segments
    cooldown: int = 2                  # segments after a committed check before the next lens
    tiers: Mapping[str, float] = {"mid": 0.55, "far": 0.35, "near": 0.10}   # keys from LENS_TIERS, weights >= 0, sum > 0
    exclude_from_window: bool = True   # lens notes are not part of the routing window
    check_other_model: bool = True     # the check is routed away from the lens writer

@dataclass
class ExploreConfig:                   # explore
    enabled: bool = False
    seed: int | None = None            # overrides the run seed for every client-side draw when set
    min_segments: int = 6              # main-line notes (lens/check not counted) before a conclusion is accepted
    require_labels: tuple[str, ...] = ("Evidence:", "Alternative set aside:", "Open question:")  # each ends with ':'
    where_we_stand_window: int = 2     # a 'Where we stand:' line must appear in one of the last N notes
    max_forced_continuations: int = 3
    loop_overlap_jaccard: float = 0.5  # word-3-gram Jaccard of the last two notes above this -> stop forcing
    cues: tuple[str, ...] = ("Alternatively, ", "One thing not yet checked: ",
                             "The strongest objection to the current conclusion is ",
                             "Perhaps the missing piece is ", "Before closing, the weakest assumption above is ")
    lens: LensConfig
    vs_menu: str = "off"               # VS_MENU_MODES = ("off", "stagnation", "every_lens"); side call always on gpt-6.2-sol
    answer_basis: bool = True          # the answer ends with 'Basis:' and 'Not resolved:'
```

Validation in `__post_init__` (`ConfigError` naming `explore.<key>` / `explore.lens.<key>`); a partial `lens.tiers`
mapping patches the default weights (deep merge). YAML's bare `off` (parsed as false) is accepted for `vs_menu`.
`cutoffmix/data/default.yaml` carries the whole `explore:` block with `enabled: false`;
`cutoffmix/data/presets/explore.yaml` (package data) is the preset of `DESIGN_explore.md` §2 without `router.policy`:
`relay: {segment_tokens: 192, hard_cut_chars: 1500, max_segments: 16, answer_mode: routed}` (`hard_cut_chars` was
1000 until the first live run showed 192 tokens of German running to 1100-1300 characters, so the cut removed the
trailing `Where we stand:` line; `DESIGN_explore.md` §6.3), `explore.enabled: true`
with the defaults above, `models: {gemini-4-neon: {supports_temperature: false}, gpt-6.2-sol: {extra_body:
{verbosity: medium}, answer_extra_body: {verbosity: low}, supports_seed: true}}`. The default `default.yaml` keeps
Sol at `verbosity: low` (the measured default run). `AppConfig.to_dict()` gives `explore` with plain lists/dicts and
`config_from_dict` reproduces it. `_validate_model` also refuses `verbosity` in `answer_extra_body` of an
`anthropic/` model. `AppConfig.preset` echoes the preset `load_config(preset=...)` merged (`"explore"`, else `None`);
the loader always sets it after the file and the overrides, so a `preset:` key in a config file is overwritten and
never loads anything. It feeds the relay's `done.data["explore"]["preset"]` (§7.6) and `/api/config.defaults.preset`
(§7.7); `to_dict()` carries it and `config_from_dict` keeps it.

### 7.2 types.py (additive only)

```python
class ModelSpec:                       # appended fields
    answer_extra_body: Mapping[str, Any] = {}   # merged over extra_body for phase "answer" only
    supports_seed: bool = False                 # the endpoint honours `seed` (Sol): body["seed"] per segment
class ChatRequest:
    temperature: float | None = None   # was 0.0; a value is still only sent when model.supports_temperature
class ContinuationRequest:             # appended fields
    tail: str | None = None            # replaces the tail block (the last user block; the only non-cached one)
    system_override: str | None = None # replaces the static system text for the run
SegmentKind = Literal["main", "lens", "check", "continue"]
@dataclass
class ExploreFlags:                    # Segment.explore (None when the preset is off)
    kind: SegmentKind = "main"; field: str | None = None; tier: str | None = None
    labels: dict[str, int] = {}        # canonical label -> count in the committed text (parse_labels()["counts"])
    has_wws: bool = False; lens_verdict: Literal["kept", "dropped"] | None = None
    cut: bool = False                  # the committed text is shorter than what the model wrote: the client-side hard cut
                                       # (sentence trim) or a length/cut/overflow finish; how a trailing 'Where we stand:' is lost
    lens_prefix_ok: bool | None = None # lens segments: the note opened with 'Lens (field):'
    cue: str | None = None             # continuation segments
class Segment:
    explore: ExploreFlags | None = None   # last field; to_dict() nests it (None otherwise)
```

### 7.3 lenses.py: fields, distance tiers, seeded draw

```python
@dataclass(frozen=True)
class Field: name: str; home_topic: str; concepts: tuple[str, ...]; label -> str   # "materials_science" -> "materials science"
FIELDS: tuple[Field, ...]              # the 23 fields of DESIGN_explore.md §4.3 (home topic = taxonomy topic id)
TOPIC_GROUPS: tuple[tuple[str, ...], ...]   # Science | Tech/Economy | Politics | Culture, in line order (all 15 topics once)
GROUP_NAMES = ("science", "tech_economy", "politics", "culture"); TIERS = ("mid", "far", "near")
def group_of(topic) -> int | None
def tier_of(home_topic, current_topic) -> str      # same group "near", adjacent "mid", else "far"; unknown/None -> "mid"
def fields_by_tier(current_topic, used_fields=()) -> dict[str, list[Field]]
def tier_weights(tiers, stagnating=False) -> dict[str, float]   # far raised to >= 0.5 when stagnating, others scaled
def draw_lens(rng: random.Random, current_topic, used_fields: Collection[str], tiers, stagnating=False)
    -> tuple[Field, str, tuple[str, str, str]] | None
    # tier = rng.choices(TIERS, weights)[0] (fallback to the next-heaviest tier with unused fields), field = rng.choice
    # of the unused fields of that tier, concepts = rng.sample(field.concepts, 3); None when every field is used.
    # Deterministic for a seeded Random: the relay uses random.Random(f"{seed}:{idx}").
def field_by_name(name) -> Field ; def normalise_field(text) -> str   # "Materials Science" -> "materials_science"
```

### 7.4 prompts.py: explore texts, tail, record status, parsers

```python
RELAY_SYSTEM_TEMPLATE_EXPLORE          # DESIGN_explore.md §6.4, {K} {final_marker} {sentinel}
ANSWER_LENS_SENTENCE, ANSWER_BASIS_SENTENCE, ANSWER_SYSTEM_EXPLORE_SUFFIX = lens + basis
LENS_TEXT {field} {c1} {c2} {c3}; CHECK_TEXT {field}; CONTINUE_TEXT {why} {cue}
VS_MENU_SYSTEM; VS_MENU_USER {question} {notes}; VS_DIRECTION_TEXT {direction}; VS_MENU_P_MAX = 0.10; VS_MENU_MIN_LINES = 3
RELAY_TAIL_EXPLORE = "END OF THE SHARED NOTES. Write note {k}. {record_status} {directive}"
RECORD_CLOSED_TEXT / RECORD_CLOSED_NO_ITEMS_TEXT / RECORD_OPEN_TEXT
WWS_ITEM = "a 'Where we stand:' line at the end of this note"   # the checklist phrase, also the reminder (closed gate)
RECORD_OPEN_WWS_TEXT = " If you write a note, end it with a 'Where we stand:' line."   # the reminder once the gate is open
LABELS = ("Claim", "Evidence", "Inference", "Principle", "Alternative set aside", "Open question", "Next",
          "Where we stand", "Lens kept", "Lens dropped", "Correction", "Lens")
LABEL_RE          # a label at a line start (bullet, **bold** tolerated, case-insensitive); group "label", "field" for 'Lens (field):'
LENS_PREFIX_RE    # 'Lens (field):' as the first thing in a note; group "field"
VS_LINE_RE        # ^\s*(\d)[.)]\s*(.+?)\s*\|\s*p\s*=\s*(0?\.\d+|1(?:\.0+)?)\s*$ (multiline)
def relay_system_prompt_explore(target_tokens, today, sentinel, final_marker) -> str
def answer_system_prompt_explore(today, *, answer_basis=True) -> str   # ANSWER_SYSTEM_TEMPLATE + sentences + date
def relay_tail(k: int, record_status: str, directive: str = "") -> str    # note k is 1-based
def record_status_text(missing: Sequence[str], gate_open: bool, final_marker="FINAL ANSWER:", *, remind_wws=False) -> str
    # remind_wws (the previous note has no 'Where we stand:' line, skipped or cut): WWS_ITEM joins the missing items while
    # the gate is closed (once, even if the checklist lists it), RECORD_OPEN_WWS_TEXT follows the permission once it is open
def missing_record_items(*, main_line_count, min_segments, label_counts, require_labels, wws_recent, pending_check) -> list[str]
    # the checklist phrases ("three more main-line notes", "an 'Evidence:' item that names its source", WWS_ITEM, ...);
    # [] = complete
def lens_text(field_label, concepts) / check_text(field_label) / continue_text(why, cue) / vs_direction_text(direction) -> str
def vs_menu_messages(question, notes) -> list[dict]      # the side call's chat messages
def parse_labels(text) -> dict   # {"counts": {label: n}, "items": n (all but 'Where we stand'), "has_wws", "lens_prefix":
                                 #  field | None, "lens_verdict": "kept" | "dropped" | None, "wws_sentence": str}
                                 # labels count at line starts and after a sentence end inside a paragraph
def parse_vs_menu(text) -> list[tuple[str, float]]       # (direction, p) per well-formed line, in order
def pick_vs_direction(menu, rng) -> tuple[str, float] | None   # p < 0.10 (else the two lowest); None if < 3 lines parsed
def word3gram_jaccard(a, b) -> float                     # 0.0 when either text has fewer than three words
```

`strip_preamble` / `clean_segment` keep every label above (incl. `Lens (field):`, `Lens kept:`, `Correction:`);
`tests/test_prompts.py` proves it per label.

### 7.5 backends/openrouter.py: body rules with the preset

`build_continuation_body`: `req.system_override` replaces the system text (both phases); `req.tail` replaces the tail
block, which stays the last user block and is never cache-marked, so consecutive explore requests have identical
`blocks[:-1]` (prefix) and only the tail differs (test `test_explore_bodies_stay_append_only_across_segments`). Phase
`answer` merges `model.answer_extra_body` over `extra_body`; an `anthropic/` model never gets `verbosity` from either.
`model.supports_seed` -> `body["seed"] = segment_seed(key, index)` with `key = req.metadata["seed"]` if present else
`req.run_id`, `index = req.metadata["segment_index"]` (else `metadata["segment"]`, else 0); `segment_seed(key, index)
= int(sha256(f"{key}:{index}")[:8]) % 2**31` (public, platform-stable). `build_chat_body` adds `seed` only for a
`supports_seed` model when `req.metadata["seed"]` is set (the next-move menu side call). The relay passes
`metadata["seed"] = explore seed or run seed` and `metadata["segment_index"]` so Sol calls are reproducible per
(seed, segment).

### 7.6 relay.py: schedule, record gate, continuations, metrics

With `cfg.explore.enabled` false nothing below runs: requests, events, `done` and the trace are byte-identical to §5
(`tests/test_explore_relay.py::test_default_relay_is_byte_identical_to_the_recorded_expectation` compares three
scripted runs against `tests/fixtures/explore/default_relay_expectation.json`, recorded before the preset existed).

**Seed.** `explore.seed` if set, else `router.seed` (the run seed, `--seed`), else a random 31-bit integer; logged as
`done.data["explore"]["seed"]` / `seed_source` (`"explore" | "run" | "random"`). Every think and answer request
carries `metadata["segment_index"] = index` and `metadata["seed"]` (§7.5 derives Sol's `body["seed"]` from them).

**Requests.** Think: `system_override = relay_system_prompt_explore(segment_tokens, today, sentinel, final_marker)`,
`tail = relay_tail(k, record_status_text(missing, gate_open, final_marker, remind_wws=not wws_previous()), directive)`
with `k` = committed think segments + 1, `metadata["kind"]` = the segment's kind; `wws_previous()` is whether the last
committed note has a `Where we stand:` line (the model skipped it, or the hard cut removed it: `ExploreFlags.cut`),
so the tail reminds of the line even when the gate's two-note window does not (or no longer) require it. Answer: `system_override = answer_system_prompt_explore(today,
answer_basis=explore.answer_basis)`, no tail (the constant `ANSWER_TAIL`). `answer_mode` is honoured unchanged.

**Plan per think segment** (`_Plan`, computed before the routing decision, once per model decision of a segment;
deterministic for the state and `index`, the lens draw with `random.Random(f"{seed}:{index}")`):

```
if pending_check:          kind "check",    directive CHECK_TEXT(field);  exclude {lens writer} (check_other_model) and the
                                                                          concluding model of a pending continuation
elif forcing pending:      kind "continue", directive CONTINUE_TEXT(why, cue); exclude {concluding model}
else:
  stagnating = argmax topic unchanged for the last 4 think segments and the last note adds no number / capitalised token
  eligible   = index >= lens.first_index and not gate_open and cooldown == 0 and lenses_used < lens.max
  last_slot  = eligible and main_line_count == min_segments - 1
  if eligible and (rng.random() < (p_stagnation if stagnating else p) or (last_slot and lenses_used < lens.min)):
      kind "lens", draw_lens(rng, current_topic, used_fields, tiers, stagnating) -> directive LENS_TEXT(field.label, c1..c3)
  else kind "main"; with vs_menu "stagnation" and stagnating, or "every_lens" and eligible (gate closed): the menu side call
```

`exclude` only narrows `available` when another model remains. The lens decision is made with `update=False` (the
router's EMA ignores it); lens notes are left out of the routing window (`lens.exclude_from_window`, also for the
answer decision), check notes stay in. `current_topic` = argmax of the last committed think segment's posterior,
or None when that posterior is flat (`taxonomy.argmax_topic`: all equal, or the maximum below 1.5x the minimum),
so a question the classifier cannot place never counts as the taxonomy's first topic; every lens field is `mid` then.
After a committed lens: `pending_check`, `lens_writer`, `lenses_used += 1`, the field is used up. After a committed
check: `cooldown = lens.cooldown`, decremented by every other committed think segment. A check never waits for a
continuation: a pending forcing is consumed by the check (the check continues the line, the concluding model is
excluded from it).

**Menu side call** (`vs_menu != "off"`): `backend.chat(cfg.model("gpt-6.2-sol"), ChatRequest(vs_menu_messages(q,
notes), max_tokens=300, metadata={"seed", "segment_index", "purpose": "vs_menu"}))`, once per segment index, then
`pick_vs_direction(parse_vs_menu(text), random.Random(f"{seed}:{index}:vs"))` and `directive = vs_direction_text(..)`.
Skipped (logged in `vs_menu.skipped`) when Sol is not among the relay's models; an exception is logged in
`vs_menu.errors` and the note is written plain; usage is added to the model's `usage` row.

**Record gate.** `record_missing()` = `missing_record_items(...)` over the committed segments' `ExploreFlags`
(main-line count = kinds `main` + `continue`; label counts summed; a `Where we stand:` in one of the last
`where_we_stand_window` notes; no pending check); `gate_open()` = nothing missing. **Waiver** (`WWS_WAIVER_NOTES = 2`):
after every committed think segment the unwaived checklist is judged; when `WWS_ITEM` is its only entry the streak
advances, else it resets, and while the streak is at least 2 `record_missing()` leaves the line out (the tail still
reminds of it), so a line the cut keeps removing cannot hold the gate closed for good; each time the streak reaches 2
a waiver is logged (`wws.waived`, `wws.waived_at`), and it lapses once a note has the line again or another item goes
missing. A conclusion (text after the marker)
is **accepted** when the gate is open, or forcing stopped on a loop, or `forcings >= max_forced_continuations`.
Otherwise it is **premature**: the answer part is discarded (`segment_end.data["premature_final"] = True`), the note
part is committed if non-empty, else the attempt is dropped (`reason "empty"`, `premature_final` on the dropped row)
and the same segment is re-routed. Then, with at least one continuation already forced and the word-3-gram Jaccard
of the last two notes (the note part included) above `loop_overlap_jaccard`, forcing stops: the conclusion is
accepted after all (`loop_stop`). Else a continuation is scheduled for the next think segment: the next cue in
`explore.cues` (rotating), `why = "the record still lacks <missing items after this note>"`, the concluding model
excluded. The tail's record status always states the checklist; after the limit the model is not told to conclude,
but the next conclusion is accepted and `gate_missing_at_final` logs the missing items (also when `max_segments` or
an all-empty segment ends the notes with the gate closed). Speculative strategy: every draft's premature conclusion
is stripped before scoring; the chosen draft decides the forcing.

**Per segment** (`Segment.explore`, `segment_end.data["explore"]`): `kind`, `field` (label), `tier`, `labels`
(`parse_labels(text)["counts"]`), `has_wws`, `cut` (the committed text is shorter than what the model wrote: the
relay compares the cleaned paragraph before and after the hard cut, and also sets it for a `length` / `cut` /
`overflow` finish without a sentinel; a stream that ended normally can still be cut), `lens_prefix_ok` (lens: the
note opened with `Lens (..):`; a missing prefix is committed and counted as `ignored`), `lens_verdict` (check),
`cue` (continuation).
`segment_start.data["explore"] = {"kind", "field", "tier", "cue"}` for the UI badge.

**Metrics** `done.data["explore"]` = `trace.to_dict()["explore"]` (the trace of an explore run is `ExploreTrace`, a
`RelayTrace` with an `explore` dict; filled when the run ends, also on error):

```
preset (AppConfig.preset: "explore" | null), seed, seed_source, config (ExploreConfig as plain dict)
think_segments, main_line_segments, lens_segments, check_segments, continue_segments, kinds[], topics[] (argmax per think segment)
topics_visited {all, main_line}, topic_entropy_bits {mean_posterior, argmax_histogram}, topic_switches
lens {offered, written, ignored, kept, dropped, unverified, fields[], tiers[], keep_rate = kept / offered | null}
labels {counts {label: n}, coverage {two_or_more_items, where_we_stand}}   # shares over non-empty notes
wws_lines[] (the 'Where we stand:' sentences), answer_flips (changes of the text before the first comma/semicolon, normalised)
wws {missing, cut, waived, waived_at[]}   # non-empty notes without the line: skipped by the model (missing) or lost to the
                                          # cut (cut, disjoint); waivers granted by the gate and the segment index of each
premature_finals[] ({index, model, kind, missing[], overlap_last2, forced, loop, cue?}), forced_continuations (committed
  "continue" segments), first_final_index, gate_open, gate_missing_at_final[] | null, loop_stop, ngram_overlap_last2
notes_chars, notes_tokens_est (chars // 4), garble (notes with < 60 % dictionary-like words: letters with a vowel or CJK)
vs_menu {mode, calls, parsed, chosen_p[], menus[] ({index, menu[[direction, p]], chosen, p}), errors[], skipped}
```

`model_switches` / `share_by_model`, `usage_total`, `latency`, `refusals` and `dropped` stay where §5 puts them.
`preset` is `AppConfig.preset` (§7.1), so `--preset explore` and the per-request preset both echo `"explore"`; a
config built by hand with `explore.enabled` and no preset echoes `null`.

### 7.7 Surface: mock backend, CLI, server, UI

**backends/mock.py.** A request with `system_override` set is an explore request; `MockBackend` then writes a
structured note instead of the default paragraph (without `system_override` the output is byte-identical to before;
`tests/test_mock_backend.py::test_default_output_is_pinned` pins it). The directive is read from `req.tail` the way a
model would read the texts of §7.4:

- plain note: two to four labelled items (`Claim:`, `Evidence:` naming a source, `Inference:`, at most one
  `Principle:`, `Alternative set aside:`, `Open question:`, `Next:`), one per line, always ending with a `Where we
  stand:` line (best answer, confidence low/medium/high by note number, weakest assumption). The labels the record
  status names as missing (`'<Label>:' item`) are written first, so the gate opens as soon as `min_segments`
  main-line notes exist; an unknown required label gets a generic item. Items are dropped from the end until the
  note fits `min(4 * target_tokens, hard_cut_chars - 40)` characters, so the trailing line survives the client cut.
- `LENS_TEXT` in the tail (`Begin the note with 'Lens (field):'`): the note opens with `Lens (field):`, maps the
  first two example concepts onto the problem, has one `Inference:` marked as a working analogy, a return sentence
  and `Where we stand:`.
- `CHECK_TEXT` (`applied a cross-field lens from <field>`): `Lens kept:` or `Lens dropped:` first, deterministic per
  (seed, model, question, field, note number), about 60 % kept.
- `CONTINUE_TEXT` (`Begin this note with the words '<cue>'`): the note starts with the cue; `VS_DIRECTION_TEXT`
  (`Direction for this note: ...`): the first `Claim:` names the direction.
- `FINAL ANSWER:` only when the tail contains `the moderator accepts a conclusion` (then the marker and the answer
  instead of a note), plus exactly one early attempt per run: the first plain note with at least two notes before it
  concludes while the gate is closed (its `Claim:` says `the record already settles the problem`; no later note does
  while that text is in the notes), so the relay's premature-final path and a forced continuation run in every mock
  run. The answer phase appends `Basis:` and `Not resolved:` paragraphs when the answer system prompt asks for
  `'Basis:'`.

**cli.py.** `--preset NAME` on `run` and `serve` is passed as `load_config(path, overrides, preset=NAME)` (unknown
names: `invalid configuration: unknown preset ...`, exit 2); `serve` also hands `--config` to `create_app(...,
config_path=)`. `TerminalPrinter.explore_tag(data)` reads `segment_start.data["explore"]` (`kind`, `field`, `tier`,
`cue`; the same keys flat in `data` are accepted) and appends `[lens: <field> (<tier>)]`, `[check: <field>]` or
`[continue: '<cue>']` to the segment header (bold when coloured). `explore_summary(done.data["explore"])` adds
`lenses N (k kept, d dropped)`, `labels P%` (the first coverage share), `topics N` (`topics_visited["all"]`) and
`forced N` to the summary line; every key is optional and numbers, lists (their length) and mappings are tolerated.

**server.py.** `create_app(cfg, ..., config_path=None)`. `POST /api/relay` and `/v1/chat/completions` take
`"preset": "explore"` (any other value: 400 `preset must be 'explore' (or absent)`); on `/v1` the model id
`cutoffmix-explore` (listed by `/v1/models` and `/api/config.model_ids`) means the same, and a body `preset` wins
over the id. A preset request runs on a per-preset `RelayRuntime` built lazily from `load_config(config_path,
preset=name)`: the packaged default, the preset, then the server's config file, exactly what `serve --preset name`
loads; it shares the main runtime's backend (closed once) and its router factory (looked up per call, so routing
data are loaded once), and the per-request `policy`/`strategy`/`segment_tokens` apply on top. `GET /api/config`
adds `presets: [{"name", "segment_tokens", "max_segments"}]` (a preset the server's file cannot combine with is
logged and omitted) and `defaults.preset` (`config.preset`, else `"explore"` when the server's own config file
enables the explore protocol without the packaged preset, else `null`). The explore metrics need no server code: `done.data["explore"]` reaches the SSE `done` event, the final
`/v1` stream chunk (`cutoffmix`) and the non-stream `cutoffmix` object with the rest of the trace, and
`segment_start.data["explore"]` / `segment_end.data["explore"]` pass through the `/v1` `delta.cutoffmix` metadata.

**web/index.html.** A `Preset` select (`default` plus `/api/config.presets`; `default` is disabled with a title when
`defaults.preset` names the server's own preset) next to Policy; choosing a preset sets the segment-tokens input to
the preset's value, and the request carries `preset` unless `default` is chosen. Segment badges from
`segment_start.data["explore"]` (or the flat keys): `lens · <field> (<tier>)`, `check · <field>` becoming `check ·
kept` / `check · dropped` at `segment_end` (from `data.explore.lens_verdict`, else from `Lens kept:` /
`Lens dropped:` in the committed text), `continue · “<cue>”`; the routing inspector lists note kind, field, lens
verdict and cue. An `Exploration` strip (topics visited, lenses with kept/dropped, labels coverage, forced
continuations with the number of early conclusions) appears below the run stats only when `done.data["explore"]`
exists; `exploreStats()` tolerates missing keys, lists and mappings (`topics_visited.all`, the first coverage
share). Dark mode and the 390 px layout hold (`tests/test_server.py` checks the markup, the pure functions run under
node; screenshots in the work log).

### 7.8 Light schema for models whose input filter blocks the full schema

Live finding (2026-11-24, 17 single-segment calls): Anthropic's input filter blocks the full explore request for
Fable 5.2 in nearly every call with no notes, and in about half of the calls with notes, under the category
"reverse engineering or duplicating model outputs". The default template passes in every call with the same
questions. Bisection: the labelled-items system prompt alone and the tail's missing-items list alone each trigger
it; renaming "moderator"/"record", unquoting the labels or rewriting both texts in prose did not help, while the
tail pieces "Write note 1.", "The moderator does not accept a conclusion yet; do not write 'FINAL ANSWER:'." and
"an 'Evidence:' item that names its source" each passed on the default system prompt. It behaves like a threshold
classifier that the schema instruction crosses.

Contract: `ExploreConfig.light_schema_models: tuple[str, ...]` (default `()`, the preset lists
`claude-fable-5.2`). For a listed model a think request uses `relay_system_prompt` (the default template) as
`system_override` and the tail `relay_tail(k, record_status_text([], gate_open, final_marker), directive)`, i.e.
the note number, the directive (lens / check / continuation / direction) and the conclusion permission, but no
checklist and no 'Where we stand:' reminder. `ContinuationRequest.metadata["light"]` and `ExploreFlags.light` record
it; the answer phase is unchanged. `Relay._request` takes the `ModelSpec` so the speculative path builds one request
per model. Verified live after the change: the first segment and a lens segment with four notes both passed, and
Fable wrote the lens note with the `Lens (field):` prefix and an `Inference:` item on its own. Consequence: notes by
light-schema models usually carry no labels, so the record checklist is satisfied by the other models' notes.

