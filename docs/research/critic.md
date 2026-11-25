CRITIC REPORT: cutoff-mix, completeness and correctness review of the 5 research reports (checked 2026-11-24)

My working files are in (scratchpad)/critic/:
- arx.xml: arXiv batch check of 37 IDs
- models_live.json: live OpenRouter /api/v1/models
- live_streaming.md, live_reasoning-tokens.md, live_gemini-4.md, live_claude-5-5.md, live_prompt-caching.md, live_gpt-6.md: live OpenRouter docs
- sim/paired_power.py: sample-size simulation

== 0. WHAT I VERIFIED MYSELF ==
- **arXiv:** all 37 cited arXiv IDs resolve on export.arxiv.org, and each title matches its claimed paper. That includes every 2026 ID: 2611.11588 FinCutBench, 2611.06808, 2608.15507, 2607.18867 HindsightBench, 2610.23613, 2603.03111, 2601.23163, 2605.28913, 2608.08239, 2609.26637, 2605.06116, 2606.07587, 2605.19228, 2605.13045, 2603.16544. I found no fabricated IDs.
- **Abstract numbers that match:**
  - 2603.03111: −8 to +13 pp Multi-IF, ~70% of variance explained.
  - 2610.23613: AUC 0.50–0.53 for local signals vs 0.86 for trajectory features.
  - 2601.23163: stronger models repeat the source model's errors; allowing further reasoning improves recovery.
  - 2607.18867: effective cutoffs precede vendor dates by up to 8 months.
  - 2608.15507: associative and declarative year diverge; associative error is about 10 months.
- **OpenRouter live catalogue (470 models):**
  - All three IDs exist. reasoning.mandatory is true for all three.
  - Efforts: Fable and Sol low…max; Neon low…xhigh. Defaults: Fable high, Sol medium, Neon medium.
  - Prices: Fable $8/$40, Sol $2/$10, Neon $3/$15.
  - knowledge_cutoff is null for all three.
  - `:batch` variants exist for all three at 50% of list price. These belong to the asynchronous Batch API (docs: "Discounted asynchronous batch processing endpoints", not usable interactively). Use them for probing only, never for the relay.
- **Endpoint `stop` support** (from the endpoints JSON the api-mechanics researcher saved; I re-read it):
  - Fable: all 4 endpoints (anthropic, anthropic/fast, azure, google-vertex/global).
  - Neon: only google-vertex/global and google-vertex/global/priority. No AI Studio endpoint supports it.
  - Sol: none of its 6 endpoints (openai, openai/fast, openai/ultrafast, azure, amazon-bedrock/global, google-vertex/global).
- **OpenRouter docs:**
  - Stream cancellation stops billing for OpenAI, Azure and Anthropic. It is not supported for Google, Google AI Studio or AWS Bedrock.
  - Reasoning tokens count against `max_tokens`. If reasoning uses all of it, the response has finish_reason "length", empty content, and is still billed.
  - `reasoning.exclude` hides the reasoning, but it is still billed.
  - A Claude refusal comes back as `finish_reason: "content_filter"` plus `message.refusal`.
  - `allow_fallbacks` defaults to true, so a request with `order` can still fall through to other public providers.
- **Anthropic** (claude-api skill, plus a live fetch of platform.claude.com refusals-and-fallback):
  - Fable 5.2 rejects prefill with a 400 and rejects temperature/top_p/top_k. Thinking is always on and the raw chain of thought is never returned.
  - `reasoning_extraction` is defined as "asks the model to reproduce its internal reasoning in the response text". Listed triggers:
    - a `<thinking>`, `<reasoning>`, or scratchpad section the model fills in;
    - "a running log of its reasoning";
    - reasoning asked for verbatim or in full;
    - earlier turns showing reasoning in tags.
  - A `reasoning_extraction` refusal that arrives before any output IS billed, and the category has no fallback model ("change the prompt rather than retry").
  - Allowed: "a short explanation, the evidence behind a result, or a summary of the actions it took".
- **Gemini** (live WebFetch of aistudio.google.com/docs/caching):
  - Explicit caching is not supported for Gemini 4.
  - The implicit-cache minimum is 4,096 tokens.
  - "Gemini caches the full previous request only if the new request adds at least 2,050 tokens… For smaller additions, Gemini re-processes the last 2,050 tokens of the previous request at the standard input rate." This confirms the api-mechanics claim.
- **Benchmark data:** recomputed from kc-repo report_answers.json (HEAD a7bef1e). Monthly direct and MCQ counts for the three targets match the benchmark-internals tables exactly.

== 1. CONTRADICTIONS BETWEEN REPORTS → RESOLUTION ==
1. **Segment length via `max_tokens`.**
   - Claims: benchmark-internals says "use max_tokens instead [of stop]"; switching-papers says "hard cap 160 output tokens".
   - Opposing claim: api-mechanics says max_tokens = K + thinking allowance.
   - RESOLVED for api-mechanics (verified in the OpenRouter docs): reasoning shares the max_tokens budget. A max_tokens of 160 would usually come back empty and still be billed. Visible length must be controlled with a sentinel and a client-side cut.
2. **Stop string `"\n\n"` (switching-papers) vs a non-whitespace sentinel (api-mechanics).**
   - Use the sentinel `<<END_STEP>>`. Whether Anthropic accepts whitespace-only stop sequences is still UNVERIFIED.
   - Cutting at "\n\n" also breaks steps that contain lists or equations. Keep "\n\n" only as a fallback cut once a segment passes the cap.
3. **Abort billing** (switching-papers: UNVERIFIED; api-mechanics: specific).
   - VERIFIED as api-mechanics states.
   - NEW ISSUE: the api-mechanics request bodies and relay_client.py use `order` with `allow_fallbacks: true`. That lets Sol fall through to amazon-bedrock or google-vertex, and Fable to google-vertex, none of which can be cancelled. Use `provider.only` instead (decisions below).
4. **Trace wrapper and revision.**
   - switching-papers: wrap the trace in `<trace>…</trace>` and say "you may correct earlier steps".
   - api-mechanics: no tags, "no restating or summarising earlier notes", start exactly where the notes end.
   - RESOLVED: no XML-like tags at all (Anthropic lists tagged reasoning as a trigger), but explicitly allow corrections ("Correction: …"). 2601.23163 (verified) shows receivers copy errors unless they may revise.
5. **Two templates (A literal continuation for GPT/Gemini, B "next intermediate result + evidence" for Claude) vs one shared trace.**
   - Mixing styles works against CoT-X and UniTE (style and format compatibility).
   - RESOLVED: use one neutral step-level template, B wording, for all three. A is an ablation for Sol and Neon only.
   - Mid-word continuation is unnecessary because segments end at sentence boundaries.
6. **Event and item budget.**
   - cutoff-metrics: 8–10 per topic×month, about 1.2k items.
   - event-sources: 25 per topic×month, about 7.7k events.
   - My paired simulation (sim/paired_power.py: 12-month band, s=1, plateau 0.9, shared item offset SD 1 month), probability that the sign of the per-topic Δτ is correct:

     | n per topic-month | Δ=1 month | Δ=0.5 month |
     |---|---|---|
     | 5 | 0.92 | 0.75 |
     | 10 | 0.97 | 0.83 |
     | 25 | 1.00 | 0.96 |

   - RESOLVED: n=10 resolves 1-month topic gaps. 0.5-month "fine differences" need about 25 per topic-month, but only near τ̂. Use a two-phase adaptive design of about 3.3k items (section 4E).
7. **Taxonomy.** benchmark-internals has 7 regex topics; cutoff-metrics 12–16; event-sources 16 with notable_deaths as its own topic.
   - RESOLVED: 15 topics (event-sources T01–T15) plus a separate event_type facet (death, office_change, election, release, award, sports_result, other).
   - Deaths are assigned to the topic of the person's occupation. event_type becomes a fixed effect in the τ model.
8. **Hysteresis.** switching-papers: δ = max(1 month, CI half-width). cutoff-metrics: h ≈ 0.5 month.
   - Both are untested defaults. Global gaps between the models are only 1–2 months, so δ=1 would pin the router to Sol.
   - RESOLVED: softmax with a small stickiness term; the deterministic δ variant becomes an ablation (section 4D).
9. **Window.** switching-papers: question + last 2 segments. cutoff-metrics: last 256–512 tokens.
   - Consistent at K=128: use question + last 3 segments, about 384 tokens.
10. **Oracle headroom** (benchmark-internals: "+2 of 280"). My recount over all 280 eligible events:
    - Direct: union 264 vs Sol 261, i.e. +3. The cases are José Jerí 2025-10 (Fable only right), and Hanabusa and Oddsson in 2026-03 (Neon only right).
    - MCQ: union 275 vs Sol 274, i.e. +1 (FanDuel CEO, 2026-05).
    - This is a minor counting slip; the conclusion stands. On existing data, per-topic routing over always-Sol has almost no measurable gain.
11. **Neon MCQ cutoff.** cutoff-metrics gives "2026-Apr+0.75, not censored" (4PL with 0.25 floor). benchmark-internals gives "no_tail, last_above 2026-05" (repo rule).
    - Both are correct under their own definitions. Use the 4PL.
12. **Probe cost.** cutoff-metrics: $170 Fable for 1.2k items at 1k output tokens per call. event-sources: about $150 Fable at batch price for about 7.7k events × 3 probes ≈ 26.7k prompts at 250 output tokens.
    - Both rest on UNVERIFIED output-token counts with mandatory reasoning. Run a pilot first (section 4E).
13. **Model naming.** All reports agree that "Gemini 6.2 Sol" means openai/gpt-6.2-sol (verified).
    - Not mentioned by any report: openai/gpt-6.2-sol-pro also exists (same model in "pro" reasoning mode, far more tokens).
    - google/gemini-4-hydrogen exists too (created 2026-11-12, $1.5/$7.5). It is not in the benchmark. The index pipeline should accept extra models.
14. **Missed by all reports.** The benchmark ran the targets at OpenRouter's default effort (Fable high, Sol and Neon medium), with max_completion_tokens=2048 and no date in the prompt. The relay will run at effort low with a date-bearing system prompt.
    - Recall and stale-defaulting may depend on both settings (HindsightBench's date-trigger reflex; 2611.06808 date prefixes).
    - So benchmark cutoffs are priors only. The routing index must be measured under deployment conditions (CUR-D).

== 2. CLAIMS THAT LOOK FABRICATED OR UNVERIFIED ==
- **None of the arXiv IDs or API claims I checked turned out fabricated.** The 5 most decision-relevant checks:
  - (a) stop support and cancellation per provider;
  - (b) reasoning shares max_tokens;
  - (c) Fable reasoning_extraction scope and billing;
  - (d) the Gemini 4 caching rules;
  - (e) benchmark curves and oracle counts.
  All verified as reported, with the caveats in section 1.
- **Still UNVERIFIED** (numbers not in the abstracts; I did not read the full texts). Treat them as unconfirmed:
  - 2608.15507 "94.6% vs 1.7%";
  - FinCutBench per-axis boundaries for Claude-Haiku-5.5, and "GPT-5.6-Sol 2026-06 vs official 2026-02-16";
  - 2605.06116 architecture details (3-layer 64-unit MLP, 1000 steps, α=0.02);
  - DMR 218–324 ms re-prefill;
  - RouteLLM wall-clock training times;
  - the GPT-6.2 Sol vendor cutoff "Jun 01, 2026".
- **Runtime behaviours with no key to test them (all UNVERIFIED):**
  - TTFT and hidden-reasoning tokens at effort low;
  - Sol and Neon behaviour with a trailing assistant turn;
  - whether `cache_control` is a no-op for Neon on OpenRouter;
  - whether a whitespace-only stop is accepted;
  - exact tokens billed on abort;
  - whether `max_tokens: 0` keep-alive works via OpenRouter;
  - the item types of GPT and Gemini reasoning summaries.
- **Artificial Analysis latencies** in api-mechanics are at max/xhigh effort and say nothing about low effort.
- **RecencyQA (2603.16544) dataset:** not found on HF (search "recencyqa" returns 0), so its availability is UNVERIFIED.

== 3. WHAT IS STILL MISSING, AND WHAT I FILLED ==
**Filled:**
- (a) Provider pinning for cancellation (`only` instead of `order`+fallbacks). See 4A.
- (b) Neon economics. Neon cannot be cancelled. Small appends (<2,050 tokens) re-bill 2,050 tokens at full input price per call, about $0.006. Prompts under 4,096 tokens are never cached. Keep Neon's max_tokens tight and its stop on Vertex.
- (c) Fable refusal handling. A refusal is billed, there is no fallback model, and rewording to evade the classifier is not allowed. Re-route the segment and track the refusal rate.
- (d) Sample-size resolution (section 1, item 6).
- (e) Judge rubric fix. The benchmark judge graded Neon's "As of March 2026, X is alive" as abstain but Fable's "As far as I know X is alive" as incorrect. For the hazard term H, a hedged stale assertion must count as outdated (with a hedged flag).
- (f) Asset availability, verified on HF:
  - models: BAAI/bge-small-en-v1.5, BAAI/bge-m3, answerdotai/ModernBERT-large, katanemo/Arch-Router-1.5B;
  - FreshQA as HF mirror natyou/freshqa_10_06 (2023, parquet) and on GitHub as Google Sheets (latest 2026-04-21).
- (g) Judge and baseline models exist on OpenRouter: meta-llama/llama-3.1-70b-instruct, qwen/qwen-2.5-72b-instruct, openai/gpt-4o-2024-05-13, anthropic/claude-opus-4.8, deepseek/deepseek-v4-pro, x-ai/grok-4.5.
- (h) relay_client.py: its 4 offline tests pass. `httpx` is not installed in this container; it is needed for stream_segment.
- (i) No prior work routes by knowledge recency. A re-run of the arXiv search `abs:"knowledge cutoff" AND abs:routing` returned only unrelated hits.

**Still missing (needs an API key or human work):**
1. Pilot measurements per model at effort low, about 50 calls each:
   - TTFT and time to first visible token;
   - reasoning_tokens p50/p95;
   - empty-output rate;
   - Fable content_filter rate on the shared-notes template;
   - cache hits (cached_tokens);
   - cost per segment.
   The achievable switching rate follows from these.
2. An end-to-end evaluation set and acceptance criterion (4H).
3. A topic-classifier check against domain shift: about 300 hand-labelled reasoning windows. Event texts are not chain-of-thought text.
4. Judge validation: κ against human labels on about 200 items.
5. Terms: OpenRouter distillation policy (`provider.enforce_distillable_text` exists). Keep any LLM-generated training labels for the topic classifier from non-target models. The benchmark code has no LICENSE (reimplement score.py); the data is CC-BY-4.0.
6. Wikimedia token and rate tiers.
7. Re-run the index whenever a pinned slug date changes; log `model` and `provider` from every response.

**Honesty flag for the user.** On the only existing data, GPT-6.2 Sol is the newest in every topic. Its 2026H1 direct correct/n versus the best other model:

| topic | Sol | best other |
|---|---|---|
| entertainment | 17/24 | 10/24 |
| business | 7/8 | 2/8 |
| sports | 4/7 | 1/7 |
| politics-intl | 9/14 | 7/14 |
| politics-US | 3/4 | 4/4 (Neon, not significant) |

Sol also never abstains: when it is past its horizon it asserts stale facts confidently. Topic reversals may exist, but only new Jul–Nov 2026 data, plus denser data in 2025-12..2026-06, can show them. Expect a router that mostly picks Sol unless the new index finds reversals. Always-Sol is the baseline to beat.

== 4. IMPLEMENTATION DECISIONS ==
**A. Models, providers, request params**
- Models:
  - `anthropic/claude-fable-5.2` with provider `{"only":["anthropic","azure"]}`;
  - `openai/gpt-6.2-sol` with `{"only":["openai","azure"]}`;
  - `google/gemini-4-neon` with `{"only":["google-vertex"]}`, so stop works. If Vertex fails, retry explicitly on google-ai-studio with the client-side cut and the same tight max_tokens.
- `reasoning: {"effort":"low","exclude":true}` for all three. Never send temperature/top_p (ignored on Fable and Sol), never `verbosity` to Claude, never `:online` or web plugins.
- Use streaming, and parse SSE as api-mechanics describes. Switch on `error.metadata.error_type`, and read `finish_reason` (content_filter / length / error).

**B. Continuation technique (black-box relay, segment level)**
- The shared chain lives only as visible text. Each call's native hidden reasoning is discarded. Prefill and thinking-block injection are impossible (verified).
- Request layout, append-only:
  - `system` = static and byte-identical for the run, with "Today's date: 2026-11-24" inside it;
  - one user message = [header block: PROBLEM + "Shared working notes so far:"] + [one block per segment, never edited or merged] + [constant tail block];
  - Fable: `cache_control` on the last segment block. Sol: `prompt_cache_key` per run. Neon: implicit caching only.
- One template for all three, Template B wording: "Several analysts are jointly working through the problem… Contribute the next intermediate result: state it and the evidence or calculation that supports it, in about {K} tokens, as the next paragraph of the notes. If an earlier note is wrong, start with 'Correction:' and fix it. Output only that paragraph, then <<END_STEP>>. If the notes already reach a conclusion, write 'FINAL ANSWER:' and the answer."
- Never use the words thinking, reasoning or scratchpad, and no XML tags. Record authorship out of band only.
- Post-process each segment: strip preambles and code fences; trim echoed overlap of 12 characters or more; trim to the last sentence end.
- Fable `content_filter`:
  - drop the output, mark it, give the segment to the next model by U_m, and never retry a reworded prompt;
  - if the pilot refusal rate is above 2%, switch Fable to "sub-question mode", where the controller asks it one explicit factual sub-question drawn from the notes.
- v2 ablations: Template A literal continuation for Sol and Neon; per-model append-only native threads (Option B).

**C. Segment and window defaults**
- Unit: one step (one paragraph), target K=128 visible tokens.
- Stopping: `stop:["<<END_STEP>>"]` for Fable and Neon-on-Vertex. Sol is aborted client-side on the sentinel.
- Universal hard cut at 600 visible characters, then trim to the last sentence; minimum one sentence.
- `max_tokens`:
  - Fable and Sol: K + 2048 (cancellable, so the slack is free);
  - Neon: start at 1536, then set it to p95(reasoning@low) + 2K from the pilot, because Neon cannot be cancelled.
- On `length` with empty content: retry once with max_tokens doubled.
- At most one routing decision per segment; minimum dwell 1 segment; chain cap 60 segments.
- Routing window W = question + last 3 segments. The first segment uses the question alone. Topic posterior smoothed by EMA with η=0.5.
- v1.1: when r(W) ≥ 0.5, draft the next segment with all three in parallel and choose by the U_m of each draft's own topic.
  - If drafts agree semantically, keep the current model's draft.
  - Cost is about 3× per drafted segment. Latency equals the slowest model's, which hides switch latency.

**D. Index definition and routing rule**
- **Primary index:** τ̃^CUR-D_{m,z}, the plateau-normalised 50%-crossing month of P(current-correct | m, topic z). It is measured with no date in the question, under the exact relay system prompt and effort=low. This is the user's "virtual cutoff".
  - Fit a hierarchical 4PL as in cutoff-metrics §4.3: model mean, topic effect, model×topic deviation shrunk with σ ~ HalfNormal(2 months), shared slope per model.
  - Covariates: event_type fixed effect and log-pageview popularity.
  - Fit in NumPyro, or by empirical Bayes; keep the posterior draws.
- **Secondary per (m,z):** τ^AV (dated question), Λ = τ^AV − τ^CUR, hazard H = P(outdated or other-wrong | t > τ+2s), abstain rate.
- **Priors** (benchmark direct 4PL): Fable 2026-Feb+0.8, Sol 2026-Apr+0.75, Neon 2026-Mar+0.0, each with SD 3 months.
- **Window score:** U_m(W) = Σ_z π̄_z(W)·τ̃_{m,z} − κ·√(Σ_z π̄_z²·Var τ_{m,z}) − λ·Σ_z π̄_z·H_{m,z}, with κ=1 and λ=2 months.
- **Rule:** P(m|W) ∝ exp((U_m + β·1[m=current]) / T(W)), with T(W) = T0 / max(r(W), 0.05), T0 = 0.5 month, β = 0.25 month.
  - Time-insensitive windows mix almost uniformly, which is what the user wants.
  - Time-sensitive windows route sharply.
- **Ablations:** Thompson sampling over the posterior draws; deterministic switching at δ ∈ {0.25, 0.5, 1} month; β ∈ {0, 0.25, 0.5}; K ∈ {64, 128, 256}. Plot quality against switch rate; expect a concave curve, as Co-LLM found.

**E. Probe design (external evaluation, never self-report)**
- **Items:** built with the event-sources pipeline (CE portal, Deaths lists, year-in-topic pages, WDQS Q1–Q8, pageviews).
  - Settle 14 days before use; require two-source date agreement.
  - Use only low/medium predictability, with the external check: if 2 of 3 old models (llama-3.1-70b, qwen-2.5-72b, gpt-4o-2024-05-13) answer correctly, the item is excluded.
  - Add the benchmark's 280 eligible events, tagged with topic and event_type.
- **Probes per item:**
  - CUR-D: primary; current-state question with no date, deployment prompt.
  - AV: question with an explicit date.
  - MCQ: 4 options including the stale predecessor (WD P1365); bracket only.
  - USE, on 25% of items: a short notes context that needs the fact; the judge checks which state the next step assumes.
- **Controls, about 15%:** alive/unchanged, fake_event, future_scheduled.
- **Sampling, phase 1:** 10 items per topic-month for 2025-12..2026-11, and 3 per topic-month for 2025-01..2025-11. That is 15 topics × (120 + 33) ≈ 2.3k items.
- **Sampling, phase 2 (adaptive):** top up to 25 per topic-month in months within τ̂ ± 2, for topics where P_best is between 0.2 and 0.8. About +1k items. Thin topics (AI/ML, tech, space, science) use 2-month bins.
- **Execution:** `:batch` variants (async, 50% price), one sample per probe. Repeat 10% of items twice to measure sampling noise, since Fable and Sol accept no seed or temperature.
  - Run a synchronous pilot first: 50 items × all probes × 3 models.
- **Grading:**
  - Labels: {current, outdated, other_wrong, abstain, ungraded} plus a hedged flag. "As of <date>, <pre-event state>" counts as outdated (hedged).
  - Two judges from different non-target families (e.g. deepseek/deepseek-v4-pro and x-ai/grok-4.5; their judge quality is UNVERIFIED), a third judge on disagreement, a 5% human audit, and κ ≥ 0.8 required.
  - MCQ is graded deterministically, with the benchmark's letter regexes.

**F. Router model and training budget**
- **v1 needs no GPU.**
  - Topic classifier: bge-small-en-v1.5 embeddings plus multinomial logistic regression. Train it on the 17.7k CE leaves, year-page events and deaths, with weak rule labels refined by a non-target LLM tagger. Validate on about 300 hand-labelled reasoning windows. Arch-Router-1.5B is a zero-shot fallback.
  - Time-sensitivity gate r(W): logistic regression on the same embeddings. Positives: FreshQA slow/fast-changing questions and CE event sentences. Negatives: FreshQA never-changing questions and math/logic step snippets.
  - Both train in minutes on 4 CPUs.
- **v2 (optional, within the user's ≤1 h budget):** a currentness predictor f(text, m) → P(current).
  - ModernBERT-large with 3 sigmoid heads, trained on the probe outcomes (about 3k items × probes × 3 models), on 1×B200.
  - My estimate is under 30 minutes; the wall-clock is UNVERIFIED.
  - Ship it only if it beats index plus kNN on held-out events (split by event, temporally stratified) on Brier score and AUC. The Routing Plateau (2606.07587) and the kNN-router paper (2505.12601) suggest it often will not.
- **Do not attempt:** token-level routers (R2R labelling alone took 2.3 days on 8×A800), or anything that needs logprobs. None of the three models exposes logprobs.
- **Online adaptation:** per-topic multiplicative weights (SpecEM rule), updated only from external signals (judge, agreement), never from self-report.

**G. Data sources**
- Wikipedia: Current Events portal (17,718 events parsed, 2024-01..2026-11-23), Deaths lists, year-in-topic pages. Use the REST and Action API with batched titles, maxlag=5, Retry-After honoured, concurrency 1, and a descriptive User-Agent without personal data.
- WDQS queries with selective entry points: Q2 heads of state, Q3 sports, Q4 software, Q5 films/albums, Q7 elections, Q8 awards, Q1 death enrichment.
- Pageviews top-1000 lists as a salience filter.
- HF apoorvumang/knowledge-cutoff-benchmark (CC-BY-4.0), plus repo report_answers.json, which has 34 models and includes the targets; the HF results.jsonl is stale.
- FreshQA (natyou/freshqa_10_06, or the GitHub sheets).
- GDELT only for salience (DOC API unreachable). Do not use HF createdAt or PyPI dates as truth for "release" items without a cross-check.

**H. Evaluation and acceptance**
- **Test set:** held-out events disjoint from the index items. About 300 time-sensitive multi-step questions plus 100 time-insensitive reasoning questions, to measure harm.
- **Arms:**
  - each model alone, under the same notes protocol and effort low;
  - always-Sol;
  - random routing at a matched switch rate;
  - round-robin;
  - cutoff-mix (softmax);
  - cutoff-mix with drafting (v1.1);
  - oracle best-of-3.
- **Metrics:** currentness accuracy, stale-assertion rate, abstain rate, final-answer accuracy, dollars and latency per question, switch rate, run-length distribution, share of segments per model.
- **Statistics:** paired bootstrap or McNemar; at least 2 repeats per arm, since Fable and Sol cannot be seeded. Use live branched rollouts only, never replayed logs (2608.08239).
- **Acceptance:** beat always-Sol on currentness (p < 0.05) and be non-inferior on the time-insensitive set.
- **Rough cost** (UNVERIFIED; assumes about 500 hidden reasoning tokens per call at low effort): about $0.025 per Fable segment, $0.006 per Sol segment, and $0.009 + $0.006 per Neon segment (the second figure is the 2,050-token reprocess). A 3-way drafted segment costs about $0.05, so a 40-segment chain with drafting costs about $2. Replace with pilot numbers.
