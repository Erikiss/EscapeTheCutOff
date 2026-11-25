PRIOR WORK: SWITCHING GENERATION BETWEEN DIFFERENT LLMs INSIDE ONE GENERATION OR REASONING CHAIN (checked 2026-11-24)

How I checked: every arXiv ID below was confirmed through the export.arxiv.org API (title, date and abstract retrieved). Where I also read the HTML full text, I give the numbers. "UNVERIFIED" means I could not confirm it. Notes are in (scratchpad)/handoff-lit/ (abs1.txt–abs5.txt hold the abstracts; dl/ holds the OpenRouter model and endpoint JSON).

== 0. BOTTOM LINE ==
1. Almost all token-level mid-generation switching work needs white-box access: logits or hidden states, a shared KV cache, or a trained small model. Examples: Co-LLM, CITER, R2R, RelayLLM, SplitReason, BiLD, speculative cascades, CoS, AdaFuse, SpecFuse, Cool-Fusion, STEER. None of it can be used as-is with our three closed API models.
2. Work that transfers to a black-box relay operates at step or segment level. The useful pieces are:
   - Speculative Thinking: "\n\n" boundaries plus keyword triggers.
   - SpecReason: a text-only judge score from 0 to 9, accept at 7 or above.
   - Lookahead Reasoning: multi-step drafts kept only if a semantic verifier accepts them.
   - Policy-Guided Stepwise Routing: a tiny MLP policy.
   - LE-MCTS: picks each step from several models using a process reward model.
   - Semantic-agreement cascades and Stepwise Confidence Attribution: consensus across outputs as a black-box confidence signal.
   - SpecEM: online multiplicative weights per model.
3. No paper routes between models by knowledge recency or topic-specific cutoff. Searches with no hits: arXiv abstracts for "knowledge cutoff" AND routing, "recency" AND "LLM routing", "time-sensitive" AND router; plus two web searches.
   - The closest pieces are "effective cutoff per sub-resource/topic" (Dated Data, 2403.12958) and "models default to older knowledge than their cutoff" (Set the Clock, 2402.16797). The second is the user's "virtual cutoff" idea.
   - 2608.15507 shows that self-reported current year and implicit (associative) year diverge. Prompting moves the self-report 94.6% of the time but the implicit year only 1.7%. This supports the user's preference for external evaluation over self-report.
4. No LLM routing paper uses explicit hysteresis. The arXiv searches for hysteresis/switching-cost/thrashing with LLM routing returned nothing relevant. The closest analogs:
   - BiLD: separate fallback and rollback thresholds.
   - "Reasoning that Travels": a two-step agreement rule, i.e. the decision must persist for two consecutive steps.
   - Trajectory-Triggered Sparse Collaborative Decoding: per-token uncertainty predicts failure at near chance (AUC 0.50–0.53), while features aggregated over the trajectory reach AUC 0.86. So route on windowed or aggregated features, not per-token signals.
5. The user's premise that models can continue each other's chain of thought is partly supported and partly contradicted. Taken as a premise and not argued, but the numbers matter for design:
   - Cross-model handoffs shift results by −8 to +13 percentage points (2603.03111).
   - Stronger receivers often copy the source model's errors unless they are allowed to keep reasoning (2601.23163).
   - Transfer is most reliable within one model family (CoT-X, Reasoning Relay).

== 1. VERIFIED API CONSTRAINTS (decide which prior work can be reused) ==
Source: public https://openrouter.ai/api/v1/models plus the /endpoints pages, fetched 2026-11-24.
- All three model IDs exist: anthropic/claude-fable-5.2 (slug -20261109), openai/gpt-6.2-sol (-20261117), google/gemini-4-neon (-20261118).
- Reasoning is "mandatory": true for all three.
  - Default effort: Fable high, Sol medium, Neon medium.
  - Supported efforts: Fable and Sol low…max; Neon low…xhigh.
  - So every call runs hidden reasoning that cannot be turned off.
- None of the three lists logprobs or top_logprobs in supported_parameters. 163 of the 473 OpenRouter models do support logprobs.
  - Consequence: every switching signal built on logprobs is unavailable for these three. That includes Nudging's top-1 probability threshold, STEER, the API mode of Policy-Guided routing, speculative cascades, Cool-Fusion and SpecFuse scoring, and trajectory-confidence features.
  - Workaround: an auxiliary open model that does return logprobs could act as a scorer.
- Parameter support:
  - `stop`: Fable yes, on all providers. GPT-6.2-sol on no provider. Gemini-4-neon only on the google-vertex endpoints, not on AI Studio.
  - `temperature`: Neon only.
  - `seed`: Sol and Neon.
- Limits and prices:
  - Max completion tokens: Fable 128000, Sol 128000, Neon 943718 (65536 on one AI Studio endpoint).
  - Price in $ per million tokens, input/output: Fable 8/40, Sol 2/10 (4/15 above 272k prompt tokens), Neon 3/15.
  - Cache reads in $ per million tokens: Fable 0.20, Sol 0.10, Neon 0.15.
- OpenRouter reasoning docs (openrouter.ai/docs/use-cases/reasoning-tokens):
  - Reasoning comes back as reasoning.summary, reasoning.encrypted or reasoning.text blocks.
  - OpenAI returns summaries. For Claude, OpenRouter defaults to summarized display.
  - Reasoning blocks passed back "must match the outputs generated by the model… you cannot rearrange or modify". Writing a mixed chain into another model's native thinking channel is therefore impossible.
  - The docs' own cross-model example pastes one model's reasoning into the next prompt as plain text.
- Anthropic API docs (via the claude-api skill, cached 2026-11-03), for Fable 5.2:
  - Assistant-message prefill returns HTTP 400.
  - Thinking cannot be disabled (400).
  - The raw chain of thought is never returned (summarized or omitted).
  - Thinking blocks from other models are silently dropped.
  - Replayed history must be append-only; Anthropic checks for edits on every account.
  - temperature, top_p and top_k return 400.
  - `reasoning_extraction` is one of the refusal categories. Prompts that read as "reveal your hidden thinking" are at risk; when that category triggers is UNVERIFIED.
  - Whether OpenRouter's assistant-prefill feature works for these reasoning-mandatory models is UNVERIFIED (no API key). For Fable it should be assumed not to work.
- Consequence: the shared "chain of thought" has to be an explicit, visible scratchpad that we keep and re-send. Each call adds its own hidden reasoning, which we can never mix. Handoff means "continue this trace" given as context, the free-generation mode in 2605.28913, not true continuation of a raw token prefix.

== 2. CATALOGUE ==
Field order per entry: ID | year | what it does | granularity | switching signal | access needed | measured effect | what we can reuse.

A. Token-level learned routing and collaborative decoding (white-box)
- Co-LLM, "Learning to Decode Collaboratively with Multiple Language Models" | arXiv 2403.03870 | 2024 (ACL 2024).
  - Mechanism: the choice of which model emits each token is a latent variable, learned by marginal likelihood. Gate is a linear head σ(⟨θ,h_t⟩) on the base model's last hidden state; defer when the gate exceeds η, grid-searched per task (e.g. 0.80 AlpacaEval, 0.57 MATH, 0.38 BioASQ).
  - Granularity: token. Access: base model white-box. The assistant needs only next-token probabilities ("commercial LLMs… exposing next-probability output can be used").
  - Deferral rates 0.1–0.9.
  - Results: GSM8k 34.5→43.5 with Llemma-34B; MATH 7.6→24.5. Accuracy is concave in deferral rate (there is an optimal switching frequency). No backtracking, so a bad assistant causes cascading errors. Only two models tested. Training about 2 h on 4×A100 (it fine-tunes the base model).
  - Reuse: the "template-filling" pattern, where the base writes the scaffold and the expert fills in facts or math, is exactly the shape of our recency use case. Also reuse the concave rate–quality curve and the random-deferral baseline (Co-LLM-Random: 17.0 on GSM8k).
- CITER | arXiv 2502.01976 | 2025.
  - Token-level router trained by policy optimization with a quality-plus-cost reward, plus a shortcut for reward estimation. Small/large pair; white-box.
  - Reuse: only the idea of a reward that includes future impact.
- R2R, "Roads to Rome" | arXiv 2505.21600 | 2025.
  - A router of about 56M parameters (six-layer residual FFN) reads the small model's top-100 logits, the token embedding and the last hidden state. It routes only "path-divergent" tokens to the large model.
  - Labels: Qwen2.5-72B judged continuations run to sentence end. 7.6M labels, 4.9% divergent; labeling took about 2.3 days on 8×A800.
  - The large model handles 11–15% of tokens, mostly at the start and end of each thought.
  - Results: 84.3 vs 30.5 tok/s against R1-32B (about 2.8×); AIME 55 vs 57.
  - White-box with shared prefill. The labeling cost alone breaks the user's 1-hour training budget.
- RelayLLM | arXiv 2601.05167 | 2026.
  - The small model, trained with warm-up plus GRPO, emits a special command to call the large model for critical tokens.
  - The large model generates 1.07% of tokens; average accuracy 49.52%; 98.2% cost reduction against performance-matched random routers.
  - White-box small model. Whether the large model could be an API doing raw-prefix continuation is UNVERIFIED.
- Nudging | arXiv 2410.09300 | 2024.
  - When the base model's top-1 probability falls below γ (0.3–0.4), a small aligned model drafts 16 tokens and only its first space-delimited word is accepted. About 10% of tokens come from the nudger.
  - Works across families (Gemma-2-27B with Llama-2-7b-chat) because it operates on words, not subword tokens.
  - Needs only top-1 logprob of the base model. API runs: 286 s vs 3026 s (average ensemble) vs 5330 s (proxy tuning); 47.5–76% of base tokens are discarded.
  - Reuse: word-level boundaries handle tokenizer mismatch. The trigger itself is not usable because our three models return no logprobs.
- CDS, "Collaborative decoding of critical tokens for boosting factuality" | arXiv 2402.17982 | 2024.
  - A classifier picks which model decodes each "critical" token, sending factual tokens to the pretrained model. Token-level, white-box.
  - Reuse: the concept of routing only fact-bearing spans, the analog of time-sensitive entities.
- BiLD, "Speculative Decoding with Big Little Decoder" | arXiv 2302.07863 | 2023.
  - Two policies: fallback (hand over to the large model when the small model is not confident) and rollback (undo small-model tokens when the large model disagrees). Up to 2.12× faster on a T4. White-box.
  - Reuse: the two-threshold structure is the closest published analog of hysteresis.
- Speculative cascades, "Faster Cascades via Speculative Decoding" | arXiv 2405.19261 | 2024.
  - The deferral rule is carried out by speculative execution; the paper derives the optimal deferral rule. Token-level; needs both models' distributions.
- CoS, "Fast LLM Collaborative Decoding via Speculation" | arXiv 2502.01662 | 2025.
  - Models alternate as proposer and verifier; verification uses the combined distribution. 1.11–2.23× faster than standard collaborative decoding. White-box.
- AdaFuse | arXiv 2601.06022 | 2026.
  - Word-level fusion that ensembles only when uncertainty is high; 6.88% average relative gain. The paper itself says it needs token likelihoods, unavailable from black-box APIs.
- Token-level ensembles (all need probabilities): GaC 2406.12585, UniTE 2410.03777, PackLLM 2404.11531.
  - UniTE's finding is still useful: model performance, vocabulary size and response style decide whether models are compatible to ensemble. Style compatibility matters for handoffs too.
- Trajectory-Triggered Sparse Collaborative Decoding | arXiv 2610.23613 | 2026-10.
  - Local per-token uncertainty (entropy, negative log-likelihood, margin, top-k mass) predicts failure at AUC 0.50–0.53. Aggregate features over the confidence trajectory, fed to a logistic predictor, reach AUC 0.86.
  - Gains of +5.3 to +20.0 points on GSM8K over the best open-loop schedule. Calling the target early has more leverage but weaker evidence.
  - Reuse: decide on aggregated windows, and use a tiny logistic router.

B. Step and segment-level handoff within one reasoning chain (most relevant)
- Speculative Thinking | arXiv 2504.12329 | 2025. Training-free.
  - Boundaries: "\n\n". After a boundary, keywords trigger a takeover:
    - Reflection: "wait", "alternatively", "hold on", "another", "verify", "think again", "recap", "check".
    - Affirmation: "yeah", "yes", "final answer", "confident".
  - Takeover length: the large model writes n1=20 tokens (affirmation/reflection) or n2=n3=125 tokens (verification, or exiting an over-long reflection loop, governed by counter c), then control returns.
  - The target writes about 18–22% of tokens. MATH500 83.2→89.4 (1.5B+32B); output 5439→4583 tokens.
  - Cross-family pairs: Llama3-8B with Qwen-32B; Qwen-1.5B with Phi-4.
  - A weaker or equal model guiding does not help (AIME −1.1).
  - Handoff interface not specified; triggers are text-only.
  - Reuse: "\n\n" boundaries, text cue triggers, and the 20/125-token takeover lengths.
- SplitReason | arXiv 2504.16379 | 2025.
  - The small model (R1-Distill-1.5B, trained with SFT then RL) emits <bigmodel> … </bigmodel>. Annotations: the "20% most difficult" spans in 18k OpenR1-Math traces, labeled by DeepSeek (the paper names two different annotator models).
  - Offload 1.35–5% of tokens; AIME24 +24% / +28.3%. Without pipelining the large model re-prefills serially.
  - Reuse: explicit begin/end tags in the trace to mark which model wrote which span.
- SpecReason | arXiv 2504.07891 | 2025.
  - A small model writes the next semantic step. The base model scores it with one generated token on a 0–9 scale; accept at 7 or above, otherwise the base model regenerates the step.
  - The score is generated text, not logprobs, so this works with black-box APIs.
  - Verification reuses the chain-of-thought prefix plus about 70 extra tokens.
  - 1.4–3.0× faster; +0.4–9.0% accuracy; up to 80% of steps accepted.
  - Reuse: a text-only judge for step acceptance or selection.
- Lookahead Reasoning | arXiv 2506.19830 | 2025.
  - A draft model proposes several future steps; the target expands them in one batch; a semantic verifier keeps correct steps. Speculative decoding speedup 1.4×→2.1×.
  - Reuse: drafting the next step(s) in parallel, judged for meaning rather than exact token match.
- RSD, Reward-Guided Speculative Decoding | arXiv 2501.19324 | 2025.
  - A process reward model scores each step; a threshold decides whether to call the target. Up to 4.4× fewer FLOPs; +3.5 vs parallel decoding.
- STEER | arXiv 2511.06190 | 2025 (AAAI 2026).
  - The small model's logit confidence before each step, fitted with a two-component Gaussian mixture. Up to +20% accuracy with 48% fewer FLOPs on AIME. White-box.
- Policy-Guided Stepwise Model Routing | arXiv 2605.06116 | 2026.
  - Steps split on "\n\n". Features:
    - uncertainty: average logit confidence on math tokens (open models), or minimum top-1 probability among the API's top-5 logprobs;
    - step index;
    - optional difficulty;
    - whether this is the final step;
    - step length.
  - Policy: a three-layer MLP with 64 units, trained 1000 steps (V-trace plus a constrained update), with threshold calibration to α=0.02.
  - Open model to closed: Qwen draft escalating to GPT-4.1-mini. MATH500 85.0 at 0.0493¢ vs 84.8 at 0.1313¢ for the target alone.
  - Escalation only; no switching back; switching frequency not studied.
  - Reuse: a tiny RL policy that trains in minutes (fits the 1-hour budget) and the feature list.
- RoRo | arXiv 2605.29310 | 2026. Rubric-guided process rewards plus GRPO for stepwise routing; same-family and cross-family settings.
- EcoTab | arXiv 2605.29319 | 2026. Estimates uncertainty separately for table tokens and text tokens before routing.
  - Reuse: model separately the fact/entity/date spans, which carry recency, and the logic spans.
- AdaSwitch | arXiv 2410.13181 | 2024. The local agent introspects on errors and asks the cloud agent for help at step level.
- AgentCollab | arXiv 2603.26034 | 2026. The agent's own self-reflection progress signal triggers escalation, with "difficulty-aware cumulative escalation". It also ran closed API models; it notes server-side scheduling affects API latency.
- DMR, Dynamic Mixed-Precision Routing | arXiv 2602.02711 | 2026. Switches precision within the same model per step via a shared KV cache, with under 1 ms overhead.
  - A secondary summary says it reports 218–324 ms re-prefill for switching across separate models; that figure is UNVERIFIED in the paper text.

C. Draft-with-all, then select (segment or process-level ensembles)
- Cool-Fusion | arXiv 2407.19807 | 2024. Each model generates up to a shared word boundary; every model reranks the candidates by perplexity. GSM8K +17.4%. Needs logprobs.
- SpecFuse / SpecEM | arXiv 2412.07380 | 2024. The arXiv title is SpecFuse; the abstract and code call it SpecEM.
  - Segment length L=10 tokens. Every model drafts; every model verifies by average segment likelihood, which needs white-box verify-in-line attention masks.
  - Online weights: ω_i(k) = ω_i(k−1)·exp(η·γ_i(k)), then normalized; η = α·sqrt(1/k)/N with α=1; γ_i = normalized count of pairwise wins.
  - First-token latency under 0.6 s; about 20% overhead over the slowest single model.
  - Reuse: the multiplicative-weights update, adapted to per-topic trust.
- LE-MCTS | arXiv 2412.15797 | 2024. Tree search over steps where each action is "the next step from model m"; a process reward model guides it. +3.6 MATH, +4.3 MQA. Black-box-compatible if the reward model is external.
- Mixture-of-Agents | arXiv 2406.04692 | 2024. Layered, response-level aggregation; AlpacaEval 2.0 65.1% vs 57.5%. Black-box, but not mid-generation.
- Trace-level synthesis in MoA | arXiv 2605.29116 | 2026. An aggregator reading full traces beats voting by assembling correct steps from minority chains. Also: a single model with perturbed inputs beats heterogeneous pools. Counter-evidence to assume, not ignore.
- ReM-MoA | arXiv 2606.24437 | 2026. Ranked reasoning memory across layers.
- Router-R1 | arXiv 2506.09033 | 2025. An LLM router alternates "think" and "route" actions, calls models from a pool (up to four routing steps), and is trained with RL. Conditions only on model descriptors such as price, latency and example performance.
- Semantic Agreement Cascades | arXiv 2509.21837 | 2025. Meaning-level agreement among models is a better deferral signal than token confidence. Needs no model internals; matches target quality at 40% of cost.
  - Reuse: disagreement among the three drafts is our signal that recency matters for this segment.

D. Can model B continue model A's chain of thought? Handoff robustness
- Reasoning Relay | arXiv 2512.20647 | 2025 (NeurIPS 2025 workshop).
  - Truncation at 25/50/75% of cumulative log-probability. The prefix appears to sit in the user message ("Solve this problem step by step: Question … [truncated reasoning]").
  - Cross-family is not uniformly worse:
    - From a Gemma-4B base (full chain 68.06%), the cross-family Llama-8B continuation beats the intra-family Gemma-1B: 63.80 vs 55.26 at 75%.
    - From a Llama-70B base (60.80%), cross-family Gemma-1B is worse: 41.98 vs 50.06 at 75%.
  - The paper's own normalized metric looks internally inconsistent.
- Reasoning that Travels | arXiv 2605.28913 | 2026. Open models only, provider/receiver setup.
  - Partial prefixes help in free-generation mode. In force-answer mode, AIME transfer is mostly answer leakage.
  - Gold-free early stopping: receivers must agree, and the two-step persistence rule is more accurate than one-step (a hysteresis analog).
- Probing the Trajectories of Reasoning Traces | arXiv 2601.23163 | 2026. Stronger models frequently repeat the weaker source model's errors when given its partial trace. Allowing further reasoning substantially improves recovery.
  - Implication: the receiver must be allowed to critique or revise, not only extend.
- CoT-X | arXiv 2511.05747 | 2025. 64 thinking/answering pairs from R1 and Qwen3. Transfer is most reliable within a family and stays effective across families once compression normalizes the trace.
  - Implication: write the shared trace in a model-neutral, normalized format.
- Evaluating Performance Drift from Model Switching | arXiv 2603.03111 | 2026.
  - A single-turn handoff changes Multi-IF strict success by −8 to +13 points and CoQA F1 by ±4.
  - Some suffix models degrade under any foreign history; others improve.
  - Decomposing drift into prefix influence plus suffix susceptibility explains about 70% of variance.
  - Reuse: estimate per-model handoff susceptibility and use it as a switching cost.
- The Replay Gap | arXiv 2608.08239 | 2026. Swapping models mid-trajectory rewrites 61–94% of later actions; replaying logs with substituted outputs mispredicts outcomes.
  - Implication: evaluate the relay with live branched rollouts, never by stitching logs.
- Measuring Faithfulness in CoT | arXiv 2307.13702 | 2023. How much models condition on their CoT varies widely, and larger models are less faithful.
  - Risk: receivers may partly ignore the shared trace, so mixing may do less than assumed.
- Reasoning Models Don't Always Say What They Think | arXiv 2505.05410 | 2025. Models verbalize hint use in often under 20% of cases.
  - So the visible trace is a lossy proxy for the knowledge actually used, which limits topic classification of the trace.
- Capable yet Parsimonious | arXiv 2609.26637 | 2026. Registering a custom tool through a standard API feature induces frontier models (e.g. GPT-6 Astra) to externalize reasoning, matching native chain-of-thought performance.
  - Supports a visible-scratchpad design. Watch Fable's reasoning_extraction refusals.
- Stepwise Confidence Attribution | arXiv 2605.19228 | 2026. Step-level confidence for closed models from generated traces only, via consensus with correct solutions (NIBS, GIBS). A black-box step-quality signal.

E. Knowledge recency and temporal knowledge boundaries (no routing work exists)
- Dated Data | arXiv 2403.12958 | 2024. Defines an effective cutoff per sub-resource and topic, which often differs from the reported one, because of CommonCrawl temporal skew and deduplication. Estimated by perplexity across dated versions of a document, which needs logprobs.
  - For our black-box models: build monthly QA probes per topic instead, like the user's benchmark.
- Set the Clock | arXiv 2402.16797 | 2024. LLaMA-2 (2022 cutoff) answers mostly with 2019 knowledge. Temporal alignment by prompting or fine-tuning gives up to +62% on 2022 answers.
  - Implication: the prompt itself can push models toward newer knowledge ("use the most recent information you have, state as-of dates"). Add this alongside routing.
- Do Language Models Consistently Encode the Current Year? | arXiv 2608.15507 | 2026. Associative year is within one year of the cutoff (about 10 months error for base models), and the declarative and associative mechanisms diverge.
- Is Your LLM Outdated? | arXiv 2405.08460 | 2024. Title confirmed; content not reviewed.
- Knowledge Card | arXiv 2305.09955 | 2023. Selects specialized LMs at inference time as sources of up-to-date knowledge, with filters for relevance, brevity and factuality. Black-box-style.
- Router-R1 (above): its routing is triggered by knowledge gaps, not dates.
- Fresh at Decision, Stale at Action | arXiv 2610.18126 | 2026. Tangential: routes new information to decisions already made.
- Community reference only (not peer-reviewed; cutoffs self-reported): github.com/Tibsfox/LLM-Knowledge-Cutoff-Reference.

F. Surveys for further leads: LLM Ensemble survey 2502.18036 (token, span and process-level "during inference" taxonomy); Dynamic Model Routing and Cascading survey 2603.04445; multi-LLM collaboration position paper 2502.04506 (API / text / logit / weight-level hierarchy; we sit at text level).

== 3. WHICH SWITCHING SIGNALS WORK OVER BLACK-BOX APIs ==
- Usable now:
  - text cue triggers (Speculative Thinking);
  - LLM-judge score generated as text (SpecReason, 0–9, threshold 7);
  - semantic agreement or disagreement across drafts (2509.21837, 2605.28913);
  - consensus-based step confidence (2605.19228);
  - external process reward model or judge (LE-MCTS, RSD);
  - tiny learned policy over text-derived features (2605.06116, 2610.23613);
  - online multiplicative weights (SpecEM);
  - our external topic-specific cutoff table (new).
- Not usable for these three models: anything using logprobs, hidden states or KV cache (Co-LLM gate, CITER, R2R, STEER, Nudging γ, BiLD, speculative cascades, Cool-Fusion and SpecFuse scoring). It becomes usable only if an auxiliary open model that returns logprobs serves as a proxy scorer.

== 4. DESIGN IMPLICATIONS FOR A BLACK-BOX API RELAY ==
Items marked [lit] come from the papers above; [rec] are my recommendations and need ablation.

4.1 Where the shared trace lives
- [verified] Native thinking cannot be shared or edited, and Fable rejects prefill. The "chain" is therefore a visible SHARED_TRACE that we keep and re-send. Each segment call also runs that model's own hidden reasoning, which cannot be disabled; we discard it.
  - [rec] Set reasoning effort to "low", and benchmark "medium", to keep latency down. With OpenRouter `exclude`, the reasoning still runs but is not returned.
- Option A, one shared trace per call: the trace sits in one user message, wrapped neutrally (e.g. `<trace>…</trace>`) with the instruction "write only the next 1 paragraph of the working notes; you may correct earlier steps".
  - Allowing correction is [lit] 2601.23163: receivers copy errors unless allowed to revise.
  - Role-neutral and simple. Cache only the static prefix (system prompt plus question) and append the growing trace at the end.
- Option B, per-model append-only threads: each model sees its own past segments as assistant turns, with reasoning_details passed back unchanged, and other models' segments as user turns ("[gemini-4-neon wrote]: …").
  - Satisfies Anthropic's append-only check and OpenRouter's "do not modify reasoning blocks" rule. Keeps each model's own hidden-reasoning continuity and prompt cache.
  - Risk: role asymmetry (foreign text arrives as user text).
  - [rec] Ablate A against B.
- [lit] Normalize the format of the shared trace (CoT-X): plain prose or markdown, no model-specific tags.
- [rec] Tag each segment's author out-of-band (SplitReason-style tags, kept only in the log) so authorship isn't visible in-band and can't bias the receiver.
- [verified/UNVERIFIED] Do not word prompts as "reveal your thinking"; Fable has a reasoning_extraction refusal category, and its trigger conditions are UNVERIFIED.

4.2 Segment boundaries
- [lit] Use "\n\n"-delimited reasoning steps as the unit (Speculative Thinking, Policy-Guided 2605.06116, STEER).
- [verified] Ending a segment:
  - Fable: `stop: ["\n\n"]`.
  - Neon: `stop` only when the provider order pins google-vertex.
  - GPT-6.2-sol: no `stop`, so stream the response and close the connection client-side at the first "\n\n" or a sentinel, with max_tokens as a hard cap. Whether tokens generated after the client closes the stream are still billed is UNVERIFIED.
- [lit] Takeover sizes: Speculative Thinking uses 20 tokens (short interjection) and 125 tokens (verification block). SpecFuse's 10-token and Nudging's one-word segments need logprobs and white-box access, so they do not apply.
- [rec] Default to one paragraph, minimum one sentence, hard cap 160 output tokens (a 192 ceiling for Sol, which is cut client-side).
- [rec] Use word or sentence boundaries, never subwords, since tokenizers differ (Nudging, Cool-Fusion).
- [lit/rec] Every handoff is a new HTTP request, with full re-prefill (minus cache hits) plus mandatory hidden reasoning before the first visible token. Live switching at token level is impossible over these APIs; switch at most once per segment.
  - Time to first token per model with effort=low is UNVERIFIED (OpenRouter shows latency_last_30m as null). Measure it first; it sets the achievable switching rate.

4.3 Routing window and score ([rec], informed by the literature)
- Topic of the current window = original question + last 2 segments + the candidate next segment (look-ahead, available in 4.5).
  - [lit] Use windowed, aggregated features; per-token signals are near chance (2610.23613: AUC 0.50–0.53 vs 0.86).
- Recency score: r_m = Σ_k p(topic k | window) · c_{m,k}, where c_{m,k} is the topic-specific effective cutoff of model m from external probes.
  - Carry a confidence interval: 330 benchmark events split over many topics gives very few events per topic and wide uncertainty.
  - [lit] Prefer external probes over self-report (2608.15507, Dated Data, Set the Clock).
- Smooth the topic distribution with an exponential moving average over segments, α≈0.5.
- [rec] Weight the score by time-sensitivity: when the window is temporally insensitive, recency differences matter little.
  - Either fall back to the current model or to the best default reasoner, or allow free mixing.
  - A cheap classifier, or judge/draft disagreement as in 4.5, decides sensitivity.

4.4 Hysteresis against thrashing vs. the desired mixing
- [lit] No paper does explicit hysteresis for LLM routing. Analogs:
  - BiLD's separate fallback and rollback thresholds;
  - the two-step persistence rule in 2605.28913, which is more conservative and more accurate;
  - Co-LLM's concave quality-vs-deferral curve, which implies an optimal switching rate.
  - Handoffs themselves cost or gain −8 to +13 points (2603.03111), so switching is not free.
- [rec] Stochastic, sticky policy: P(m) ∝ exp((r_m + β·1[m = current] − λ·h_m)/τ).
  - h_m = how sensitive model m is to receiving a foreign prefix, estimated as in 2603.03111.
  - β = stickiness, which acts as hysteresis.
  - Set τ about equal to the confidence-interval width of the c_{m,k} estimates. Models that are statistically indistinguishable for a topic then mix roughly uniformly (the user's intent), while real recency gaps dominate.
  - Deterministic variant: switch only if r_b − r_a > δ, with δ = max(1 month, CI half-width); minimum dwell of one segment.
  - Ablate β ∈ {0, 0.5σ, 1σ}, τ, and segment cap ∈ {64, 160, 320}. Plot quality vs switch rate, expecting the concave curve Co-LLM found.
- Baselines (required): each model alone; random routing at a matched switch rate (Co-LLM-Random, RelayLLM's matched-random comparison); round-robin; oracle best-of-three per question.

4.5 Speculative parallel drafting of the next segment
- [lit] SpecFuse/SpecEM (all models draft, all verify), Lookahead Reasoning (multi-step drafts kept after semantic check), LE-MCTS (pick each step from the model pool), SpecReason (judge score ≥7), semantic-agreement cascades.
- [rec] Protocol:
  - Each segment, fire all three requests in parallel on the same trace. Wall-clock ≈ the slowest of the three, which hides switching latency.
  - Selection, in order:
    - (a) classify the topic and time-sensitivity of each draft itself (look-ahead topic);
    - (b) if the drafts semantically agree, take the current model's draft (stickiness, no information lost);
    - (c) if they disagree on time-sensitive claims, take the draft from argmax r_m on the draft's topic;
    - (d) optionally run a text-only judge (SpecReason 0–9) or an external process reward model to veto incoherent drafts.
  - After each segment, update per-topic trust online with the SpecEM rule ω ← ω·exp(η·γ), normalized, η = sqrt(1/k)/N. γ comes from external checks such as agreement or judge wins, never from self-report.
- Cost and gating: about 3× output tokens and 3× hidden reasoning per segment.
  - [rec] Draft with all three only when the window is time-sensitive or the router margin is under δ. Otherwise use one model, with Speculative-Thinking-style text triggers (reflection/verification cues) to call a second opinion.
  - Prices: Fable $40 per million output tokens vs $10 (Sol) and $15 (Neon).
  - Prompt caching is essential because the trace is re-sent every call (cache reads $0.10–0.20 per million tokens). Keep the trace append-only so the cached prefix stays valid.

4.6 Learned router within the user's budget (at most 1 h on at most 10×B300)
- [lit] Feasible: a 3-layer, 64-unit MLP policy trained for 1000 steps (2605.06116), or a logistic regression over trajectory features (2610.23613). Both train in minutes on CPU; no GPU needed.
- Not feasible: R2R-style token labeling (2.3 days on 8×A800).
- [rec] Features: topic embedding of the window and of each draft; c_{m,k} plus its confidence interval; time-sensitivity score; draft disagreement; step index; current model; h_m. Reward: correctness on held-out time-sensitive QA plus a cost term.

4.7 Evaluation hygiene
- [lit] Use live branched rollouts, never replay stitching (2608.08239).
- Measure answer leakage separately from scaffolding (2605.28913).
- Allow revision of inherited steps (2601.23163).
- Report the share of segments per model and the run length between switches; these are missing from nearly all prior work.
