# Topic-specific effective cutoff for cutoff-mix: prior work, metrics and recommended routing index (checked 2026-11-24)

## 0. Bottom line
- **No model-internal index is available for these three models.** No OpenRouter provider endpoint lists `logprobs` or `top_logprobs` for `anthropic/claude-fable-5.2`, `openai/gpt-6.2-sol` or `google/gemini-4-neon` (checked via `/api/v1/models/{id}/endpoints`). All white-box methods are therefore out: perplexity over versioned documents, activation probes, temporal heads and min-k% probability. The index has to be an external, behavioural evaluation, which is what the user prefers anyway.
- **Self-report should not be used, even as a feature.** LLMLagBench, 2611.06808 and HindsightBench all show that stated or self-reported cutoffs are off by months to years.
- **Topic-specific effective cutoffs are documented for black-box frontier models.** FinCutBench measured separate boundaries per financial axis within one model. For Claude-Haiku-5.5: market path 2026-03, sector 2026-06, macro 2026-08, and no decline for asset moves.
- **The "virtual cutoff" is documented as stale defaulting.** The model knows the newer state when cued, but defaults to an older state when asked about "now".
  - evolveQA: outdated-response rate 23–54%, yet models often pick the current option in MCQ.
  - DyKnow: GPT-4 gives 13% outdated answers, GPT-3 47%.
  - Set the Clock: LLaMA2-70B, with a 2022 cutoff, answers most like 2019.
  - Kairos: shuffled pretraining "peaks on older data", likely because older facts are repeated more.
- **I found no paper that fits per-topic logistic decay curves with hierarchical shrinkage.** The recommendation in sections 4–5 is a synthesis of the closest methods: FinCutBench's 3-phase piecewise fit, 2611.06808's half-way threshold, and the 50% crossover in the apoorvumang benchmark.
- **The global gaps between the three models are only about 1–3 months.** Routing therefore needs about 0.5-month resolution per topic, which takes about 8–10 items per topic×month in the transition band (simulation in §4.6).
- **No learned router is needed for a first version.** The index is a lookup table τ[m,z] combined with a topic classifier (embedding + kNN or logistic regression), which trains in seconds on CPU. A learned correctness predictor (RouteLLM-MF/BERT style) fits easily in 1 h on 1×B200, but the kNN-router literature says it rarely beats kNN.

## 1. Facts about the base benchmark and the three models

### apoorvumang/knowledge-cutoff
- **Method** (README and `kc/score.py`):
  - Per-month known_rate = correct/n, computed over low/medium-predictability events only.
  - `last_above` = latest month with known_rate ≥ 0.5.
  - `crossover` = last month ≥ 0.5 that is followed by at least 2 months (MIN_TAIL=2) all below 0.5.
  - Status values: `never_below` (right-censored), `no_tail` (dips and recovers), `pinned`.
  - Grading is three-way (correct / incorrect / abstain), with `control_alive` and `fake_event` diagnostics.
  - Two probes bracket the truth: `direct` (graded by an LLM judge) and `mcq` (4-way).
  - Greedy decoding by default.
  - There is no curve fit and no confidence interval.
- **Coverage:** months 2024-01..2026-06, about 7–11 events per month. Categories: death 201, office_change 111, control_alive 10, fake_event 8.
- **HF data:** `results.jsonl` (last modified 2026-07-10) does not contain our three models. Their monthly curves exist only in the JSON embedded in the explorer page, which I extracted.
- **Advertised cutoffs on the explorer:** Fable 5.2 Jun 2026, GPT-6.2 Sol Jun 2026, Gemini 4 Neon Mar 2026.
- **Explorer direct curves for the three models** (k = known, w = wrong, a = abstain):

| Model | Months | Pattern |
|---|---|---|
| Fable 5.2 | 26-02 k=.67, 26-03 .40, 26-04 .20, 26-05 0 | After the cutoff: wrong .5–.8, abstain .1–.5 |
| Sol | 26-03 .8, 26-04 .9, 26-05 .5, 26-06 0 | Always wrong, 0 abstain: asserts confidently but stale |
| Neon | 26-03 .5, 26-04 0 | Abstains .6–.9 |

- **MCQ curves are right-censored at 2026-06** for Fable and Sol ("never_below"). The event window must be extended to Nov 2026.

### My 4PL fits to these curves
Script: `fit_global.py`. Floor g=0 (direct) or 0.25 (mcq). Grid MLE. Parametric binomial bootstrap with B=300. Month index 0 = 2024-01; "Feb+0.8" means 0.8 of the way from the Feb bucket to the Mar bucket.

| Probe | Model | τ | 95% CI |
|---|---|---|---|
| direct | Fable 5.2 | 2026-Feb+0.80 | [Feb+0.35, Mar+0.20] |
| direct | Sol | 2026-Apr+0.75 | [Apr+0.25, May+0.23] |
| direct | Neon | 2026-Mar+0.00 | [Feb+0.70, Mar+0.30] |
| mcq | Fable 5.2 | ≈2026-May+0.75 | censored |
| mcq | Sol | ≈2026-May+0.75 | censored |
| mcq | Neon | 2026-Apr+0.75 | not censored |

- Fitted slopes are s ≈ 0.4–0.6 months on direct, i.e. steep decay.
- These CIs understate the real uncertainty: only 2 categories, about 10 items per month, no item heterogeneity modelled. Use them only as global priors.
- On recall (direct), Sol leads by about 2 months. On recognition (MCQ), Fable and Sol are about equal.
- The direct-vs-MCQ gap (Fable about 3 months, Neon about 1.75, Sol about 1) is itself a "virtual cutoff" signal.

### OpenRouter metadata (2026-11-24)
- **Pinned slugs:** `anthropic/claude-fable-5.2-20261109`, `openai/gpt-6.2-sol-20261117`, `google/gemini-4-neon-20261118`.
- **Reasoning is mandatory on all three.** OpenRouter's `knowledge_cutoff` field is null for all three.
- **Supported parameters:**
  - Fable 5.2 and Sol do not list `temperature`. Sol lists `seed`; Fable lists `stop`.
  - Neon lists `temperature`, `top_p`, `seed` and `stop`.
  - So greedy decoding cannot be guaranteed for Fable and Sol. What OpenRouter does with a passed temperature for them is UNVERIFIED.
- **Prices per million tokens, input/output:** Fable 5.2 $8/$40, Sol $2/$10, Neon $3/$15. The `:batch` variants cost half.
- **There is no "Gemini 6.2 Sol" on OpenRouter.** The model the user means is OpenAI's `gpt-6.2-sol`.

## 2. Prior work
Every arXiv ID below was checked against the arXiv API. BB = black-box (text only); WB = needs logits or weights.

### A. Effective and topic/resource-specific cutoffs

**Dated Data** (Cheng et al.), arXiv 2403.12958 — WB
- What it does: introduces the "effective cutoff", defined separately per sub-resource and topic.
- Data and method:
  - Perplexity of monthly versions of the 5,000 most-edited Wikipedia articles (2016-04..2023-04) and NYT articles bucketed by month.
  - Perplexity uses the first 512 (Wikipedia) or 256 (NYT) tokens; each month is averaged after trimming 2.5% at each tail.
  - Scores are min-max normalised: r_t = (p̄_t − min)/(max − min). The cutoff is argmin_t r_t, treated as a distribution.
  - The estimate is stable when there are more than 50 documents per bucket.
- Findings: effective cutoffs are often older than reported (e.g. LLaMA's Wikipedia cutoff lands at 2017–2020 against a reported Aug 2022). Causes are old content inside new CommonCrawl dumps and deduplication failures.
- Not usable for us (needs logprobs), but it is the conceptual source. The Ai2 blog post of 2025-10-24 summarises it.

**LLMLagBench**, arXiv 2511.12116 — BB
- Data: 1,713 validated news questions, 2021–2025, with "I don't know" allowed. A DeepSeek-V3 judge scores faithfulness 0–2 (κ = 0.81/0.83 against humans).
- Method: PELT change-point detection (Killick et al., arXiv 1101.1438) on time-ordered scores.
- Findings:
  - Multiple boundaries per model (e.g. Claude Sonnet 4 at Feb 2023 and Dec 2024), interpreted as training phases.
  - No per-topic breakdown.
  - Table 3 compares detected, declared and self-reported cutoffs: GPT-OSS-120B detected Sep 2023, declared Jul 2024, self-reported Sep 2021.

**FinCutBench**, arXiv 2611.11588 — BB; the strongest per-topic precedent
- Data: 9,000 four-option MCQs, 2023-01..2026-09, 200 per month, 50 per axis on 4 axes: macro/policy, sectors, market paths, asset moves.
- Method:
  - One-sided Hoeffding decline tests with Holm correction (α_j = 0.05/[j(j+1)]).
  - Then a 3-phase fit (plateau, linear decline, low plateau) by least squares.
  - Boundary = first month on the low plateau.
  - 1,000 resamples give a 90% Boundary Range ("Tight" if ≤ 6 months).
- Findings: per-axis boundaries differ within one model by up to about 5 months. GPT-5.6-Sol's boundary is 2026-06 against an official 2026-02-16.

**2611.06808**, "LLMs Don't Know What They Know: Temporal Self-Knowledge for Financial Backtesting" — BB
- Data: 2,226 dated US facts (1999-01..2026-09) in 3 topics: FRED macro series, stock prices, Fed decisions.
- Method:
  - Threshold T_K = C + ½(B_K − C), where C is chance and B_K is accuracy on 2015–2020.
  - "Knowledge ends" = last quarter whose 2-quarter rolling accuracy stays above T_K.
  - This is exactly a plateau-normalised 50% crossing.
- Findings:
  - GPT-5 answers 76% of controlled facts correctly but claims to know only 38%.
  - Llama-3.3's knowledge claims outlast its real knowledge by 7 quarters.
  - Per-topic gaps differ: stock prices accuracy−claim +0.54 for GPT-5; Fed decisions −0.16..+0.04.
  - "Assume today is [date]" prefixes change the labels, not the answers.

**HindsightBench**, arXiv 2607.18867 — BB
- Metric: "behaviourally effective cutoff" = last month with outcome-recall propensity LAP > 0.1. LAP comes from 20 samples at T = 1.0 per month, gated on the recall hit rate being above chance; the threshold sweep 0.05–0.2 moves it by at most 1 month on 9 of 12 models.
- Findings:
  - Measured cutoffs are 3–8 months earlier than vendor dates (Gemini 2.5 Pro 2024-05 vs 2025-01; GPT-5.5 2025-06 vs 2025-12; Claude Sonnet 5 2025-10 vs 2026-01).
  - The "date-trigger reflex" (stating a date changes behaviour) is present in every 2026-generation model tested.

**ChroKnowledge / ChroKnowBench**, arXiv 2410.09870 — BB-capable
- Domains: general (Wikidata), biomedical (UMLS), legal (CFR), commonsense, math. Facts are time-variant vs invariant and dynamic vs static; years 2010–2023 (biomedical 2020–2024).
- Method: Known / Partial / Unknown labels from greedy plus T = 0.7 samples over 5 few-shot sets.
- Findings: per-domain per-year curves drop in recent years. MCQ and true/false hold up better than generation.

**evolveQA**, arXiv 2510.19172 — BB
- Data: AWS, Azure and WHO change feeds, with gold answers tailored to each model's cutoff; 12 models.
- Findings:
  - Per-domain drops of 6–31%.
  - Outdated-response rate (ORR) 23–54%: outdated answers are more common than wrong ones.
  - The MCQ current option is often chosen even when open answers are stale.

**TempoMed-Bench**, arXiv 2605.13045 — access mode not stated in the abstract
- Findings: up-to-date medical-guideline knowledge declines gradually and linearly, with no sharp cutoff. Historical knowledge is only 25–54% of up-to-date knowledge.
- Implication: decay shape is topic-specific, so the slope s should be per model (and possibly per topic family).

**PinTrace**, arXiv 2605.06279 — BB
- Findings: LLM-chosen Python library versions converge on the same risky versions, and 72–91% of the CVEs were disclosed before the model's cutoff.
- My inference, not the paper's framing: a stale default for the software-versions topic.

**Kairos**, arXiv 2605.22769 — WB (pretraining study)
- Findings: the shuffled CommonCrawl-pretrained 6B model has accuracy peaks at 2015 and 2020 although its data runs to 2024. Temporally ordered training gives fresher knowledge. Attributed to repetition (a hypothesis).
- This is a mechanism for the virtual cutoff.

**"Is Your LLM Outdated?" / FreshBench**, arXiv 2405.08460
- Methods: Temporal Bias Index = linear slope of bits-per-character over 2-month points (WB); event-prediction accuracy (BB).
- Finding: "Nostalgia" bias (better on older events) is common.

**Time-stratified and continual-learning corpora** — WB (loss-based); useful mainly as design references
- TiC-LM, arXiv 2504.02107: 114 CommonCrawl dumps; per-domain time-stratified evaluation (Wikipedia, StackExchange, code docs).
- TemporalWiki, arXiv 2204.14211: diffs between Wikipedia/Wikidata snapshots.
- EvolvingQA ("Carpe Diem"), arXiv 2311.08106.

### B. Currentness / "internal clock" (virtual cutoff) probes, black-box

**Set the Clock**, arXiv 2402.16797
- Data: TAQA, 20,148 questions with answers changing across 2000–2023; metric is per-year F1.
- Findings:
  - LLaMA2-70B (cutoff Sep 2022) peaks at 2019.
  - The "as of year yy" prompt: 2022 F1 17.2→27.4.
  - Target-year fine-tuning: 2022 F1 17.2→27.9 (≈ +62% relative; WB).
- No topic breakdown; breakdown by popularity only.

**DyKnow**, arXiv 2404.08700
- Data: 130 Wikidata facts (heads of state/government, athletes' teams, CEOs). Answers are mapped to Wikidata validity intervals and labelled correct / outdated / irrelevant; the paper infers each model's training span from those intervals.
- Findings: GPT-4 80% correct / 13% outdated; GPT-3 42 / 47; prompt agreement 94–98% for GPT-4 and ChatGPT.
- This is the template for my interval-censored anchor-date estimator (§4.3).

**FreshQA / FreshLLMs**, arXiv 2310.03214
- Data: 600 questions; test set 125 each never-changing / slow-changing / fast-changing / false-premise; Relaxed and Strict grading.
- Use: labels for the recency-sensitivity gate.

**RealTime QA**, arXiv 2207.13332
- Weekly questions; GPT-3 returns outdated answers when retrieval is insufficient.

**RecencyQA**, arXiv 2603.16544
- 4,031 questions with recency and stationarity labels; difficulty rises with update frequency.
- Use: labels for the gate.

**Fact duration prediction**, arXiv 2305.14824
- Predicts how long a fact stays true. Use: volatility weight.

**TimeQA**, arXiv 2108.06314
- Time-sensitive questions with explicit time qualifiers, built from Wikidata. Use: template for availability probes with the date given.

**Time-sensitive fact-recall benchmarks**
- 2409.13338 (Herel et al.): over 8,000 events 2018–2024; base models beat instruction-tuned models on time-sensitive recall; brittle to paraphrase.
- TempUN, 2402.11997.

**Prompted-cutoff studies** (relevant because chain of thought leaks implicit knowledge)
- 2510.02340: prompted cutoffs fail on causally related knowledge.
- 2601.13717: a 52% gap between simulated and true ignorance; chain of thought does not suppress prior knowledge even when traces contain no explicit references.
- 2606.05804: recall-based prompting.

### C. Self-knowledge and internal signals
- **Language Models (Mostly) Know What They Know**, arXiv 2207.05221: P(IK) calibration; needs fine-tuning or logits.
- **SelfAware**, arXiv 2305.18153.
- **Knowledge-boundary survey**, arXiv 2412.12472.
- **Price et al., arXiv 2407.04108** (WB): activation probes tell pre-cutoff from post-cutoff events with about 90% accuracy, and a steering vector for the model's sense of the date exists. Precision/recall and model sizes were not shown on the abstract page (UNVERIFIED).
- **Temporal Heads**, arXiv 2502.14258 (WB): attention heads for time-specific recall.
- Negative evidence on self-report: LLMLagBench Table 3, 2611.06808, HindsightBench.
- Conclusion: for these API models, no faithful internal index is available. Use external evaluation.

### D. Routers and their reported training cost

**RouteLLM**, arXiv 2406.18665 — 65k Chatbot Arena pairs plus about 120k GPT-4-judged samples (about $700)

| Router | Training | Hardware | Serving cost per 1M requests | Throughput |
|---|---|---|---|---|
| Similarity-weighted ranking | none | — | $39.26 | 2.9 req/s |
| Matrix factorisation | ≈10 epochs, batch 64 | "8GB GPU" | $3.32 | 155 req/s |
| BERT-base classifier | ≈2,000 steps, batch 16 | 2×L4 | $3.19 | — |
| Llama-3-8B classifier | ≈2,000 steps, batch 8 | 8×A100 | $5.23 | — |

- No wall-clock training hours are reported (UNVERIFIED). By step counts, all four should fit far inside 1 h on 1×B200; that is my estimate.

**kNN router**, arXiv 2505.12601
- A tuned kNN matches or beats learned routers; CPU index build and query in the sub-millisecond to few-second range.

**The Routing Plateau**, arXiv 2606.07587
- 21 methods on 5 benchmarks; kNN is top-2 on all 5; routers mostly learn global averages.
- It suggests using partial output trajectories as routing input. This supports routing on chain-of-thought windows.

**Other routers** (training cost not stated in the abstracts: UNVERIFIED)
- IRT-Router, arXiv 2506.01048: IRT-style model ability and query difficulty; 20 LLMs, 12 datasets; interpretable.
- EmbedLLM, arXiv 2410.02223.
- RouterBench, arXiv 2403.12031: 405k inference outcomes.
- Arch-Router, arXiv 2506.16655: 1.5B model, base Qwen2.5-1.5B-Instruct per the HF card; maps text to user-defined domain labels without retraining. Usable as a zero-shot topic classifier.
- GraphRouter 2410.03834, RouterDC 2409.19886, Hybrid LLM 2404.14618, FrugalGPT 2305.05176.

**Intra-generation switching precedents**
- Co-LLM, arXiv 2403.03870: token-level latent-variable deferral.
- CITER, arXiv 2502.01976: token-level routing trained with RL.
- R2R, arXiv 2505.21600: routes only path-divergent tokens.
- All three assume open models or logits. For three black-box APIs, segment-level switching is the only option (my inference).

**IRT for cheap adaptive evaluation**
- tinyBenchmarks, arXiv 2402.14992: precedent for choosing informative items.

## 3. Gap analysis
- No paper found fits a per-topic decay curve with hierarchical shrinkage to a model's global curve, or turns topic cutoffs into a routing policy.
- No paper found separates availability from currentness on the time axis as two cutoffs per topic. evolveQA, DyKnow and Set the Clock measure the phenomenon but not as a dated cutoff.

## 4. Recommended topic cutoff index (black-box)

### 4.1 Item bank
- **Items:** dated world-state changes i with:
  - change time t_i in fractional months (day-level: t = month_index + (day−1)/days_in_month),
  - topic z_i,
  - popularity u_i (log Wikipedia pageviews),
  - predictability (keep only low/medium for τ),
  - for currentness items, the value history v_0..v_K with change times t_{i,1} < … < t_{i,K}, taken from Wikidata start/end qualifiers as in DyKnow.
- **Taxonomy:** 12–16 leaf topics, e.g.:
  - heads of state/government (US, EU, Asia, rest of world)
  - notable deaths
  - corporate leadership/M&A
  - markets/prices
  - central banks/macro
  - AI models/products
  - software library versions/APIs
  - medicine/drug approvals/guidelines
  - sports results/transfers
  - entertainment/awards
  - law/courts/regulation
  - conflicts/elections
- **Per-topic controls:** add control_alive / unchanged-state controls and fake events, as the benchmark does.
- **Time range:** extend to 2026-11; the current benchmark stops at 2026-06, so the target models are right-censored.

### 4.2 Probe types (same item, all 3 models, paired)
- **AV (availability):** an event-specific or explicitly dated question, e.g. "Who was CEO of Y in September 2026?" (TimeQA/TAQA style), or the benchmark's neutral "Is X alive or has he died? If so, when?". Run an MCQ variant as the upper bracket.
- **CUR (currentness, no date):** "Who is the CEO of Y?" This captures the virtual cutoff (DyKnow, FreshQA, evolveQA style).
- **CUR-D (deployment-matched):** CUR under the exact cutoff-mix system prompt, including any "today is 2026-11-24" injection. This is needed because of the date-trigger reflex (HindsightBench) and date prefixes relabelling answers (2611.06808).
- **USE (implicit), about 25% of items:** "Draft a 2-sentence note to Y's CEO about …". The judge checks which state is assumed. This is the closest proxy for in-chain-of-thought use (2510.02340, 2601.13717).
- **Grading:** an LLM judge with the ground truth and the value history assigns {current, outdated(k), irrelevant/hallucinated, abstain}, as in DyKnow and evolveQA. Use reasoning_effort=low to cut cost. Keep unparseable responses as `ungraded`, not abstain, as the benchmark does.

### 4.3 Model
Outcome c = current-correct. For model m, probe q, item i:

```
P(c | m,q,i) = g_q + (a_mq(u_i) − g_q) · σ((τ_{m,z_i,q} − t_i) / s_mq)
```

- g_q: floor. 0 for open probes; for MCQ, 1/k, or estimate it from the fake/unknown-item floor.
- Plateau: logit a_mq(u) = α_mq + β_mq·(u − ū). This separates coverage and popularity (PopQA, 2212.10511) from recency.
- Slope s_mq is shared across topics. A per-topic s is not identifiable at these n; at most use one s per topic family, which TempoMed suggests.
- Topic cutoff, with crossed hierarchical shrinkage:

```
τ_{m,z,q} = μ_mq + γ_{z,q} + ε_{m,z,q}
γ_{z,q} ~ N(0, σ_γ²)      topic effect common to all models; can be stabilised with the benchmark's 30+ models
ε_{m,z,q} ~ N(0, σ_ε²)    model-specific topic deviation = the "relative fine differences" the user wants
```

- Priors:
  - μ_mq ~ N(global prior from §1, 3²) in months
  - log s ~ N(log 1, 0.5²)
  - σ_γ, σ_ε ~ HalfNormal(2 months)
  - optional item random intercept on the logit for item difficulty (IRT-style)
- **Anchor-date interpretation, used for CUR with multi-change items.** Let the model's per-topic anchor date be A ~ Logistic(τ^CUR_{m,z}, s). Then

```
P(answer = v_k) = a_i · [F(t_{i,k+1}) − F(t_{i,k})]
```

  - For k = K (current): P = a_i · σ((τ − t_{i,K})/s), which reduces exactly to the decay curve above.
  - Outdated answers therefore carry information about *how* stale the model is (interval-censored likelihood).
  - **Virtual cutoff ≡ the median anchor date τ^CUR_{m,z}.**
- **Fitting:**
  - Full Bayes in NumPyro/Stan: about 10⁴ Bernoulli rows, minutes on CPU.
  - Or empirical Bayes: per-cell MLE (τ̂, SE), then

```
τ̃ = w·τ̂ + (1−w)·(μ̂_m + γ̂_z),   w = σ_ε² / (σ_ε² + SE²)
```

### 4.4 Outputs per (m, z)
- τ^AV and τ^CUR: the plateau-normalised 50% crossing (continuous month). This is the smooth version of 2611.06808's T_K = C + ½(B − C).
- Absolute-50% month, defined only if a > 0.5:

```
τ_abs = τ − s · logit((0.5 − g) / (a − g))
```

- Retrieval lag Λ = τ^AV − τ^CUR ≥ 0.
- Stale-assertion hazard H = P(outdated or hallucinated | t_i > τ + 2s), measured on post-cutoff items. It differs strongly by model: Sol asserts, Neon abstains.
- 90% interval from posterior draws, or a paired cluster bootstrap (resample items within topic×month strata, the same draw for all models, B = 1000).
- P_best(m | z) = Pr(τ_{m,z} = max over models), from draws.

### 4.5 Comparison with existing estimators
| Estimator | Source | Note |
|---|---|---|
| Threshold crossover | apoorvumang | No CI; breaks at n ≈ 10 |
| PELT change-point | LLMLagBench | Detects multiple phases; no per-topic CI |
| Piecewise 3-phase + resampling | FinCutBench | Closest to this proposal |
| Last month with LAP > 0.1 | HindsightBench | — |
| Argmin normalised perplexity | Dated Data | WB |

- The 4PL gives a continuous τ with standard errors, shrinkage, and the direct-vs-MCQ and CUR-vs-AV contrasts.
- Optional robustness check: run PELT or the FinCutBench decline test on the CUR series to detect multi-phase curves before trusting a single τ.

### 4.6 Sample sizes (simulation `sim.py`)
Setup: s = 1.2 months, plateau 0.9, independent item sets per model, 400 replicates.

| n per topic×month | RMSE(τ), months | P(correct order), Δτ = 1 month | P(correct order), Δτ = 2 months |
|---|---|---|---|
| 2 | 0.95 | 0.75 | 0.92 |
| 3 | 0.81 | 0.81 | 0.96 |
| 5 | 0.59 | 0.92 | 0.99 |
| 10 | 0.45 | 0.94 | 1.00 |
| 20 | 0.33 | 0.98 | 1.00 |
| 30 | 0.27 | 0.99 | 1.00 |

- MCQ floor 0.25 at n = 10: RMSE 0.62 and P(order) 0.85. So estimate τ from open probes and use MCQ only as a bracket.
- Analytic: Fisher information I(τ) ≈ n/s per month of band, so SE(τ) ≈ 1.3·√(s/n) once nuisance parameters are estimated.
- Paired items across models reduce variance further.
- **Recommendation:**
  - At least 5 (target 8–10) items per topic×month×probe inside the band [global τ − 4, global τ + 3], i.e. about 2025-11..2026-09.
  - 2–3 items per month on older months, to estimate the plateau.
  - Cells below 3 fall back to shrinkage (μ_m + γ_z).
  - Use adaptive item selection near the current τ̂, where p(1−p) is maximal (tinyBenchmarks/IRT precedent).
- Example budget: 14 topics × 11 band months × 8 = 1,232 items, plus about 250 sparse older items.
- Cost: at about 3 probes per item and about 1k output tokens per call (UNVERIFIED token count), that is roughly $170 Fable / $45 Sol / $65 Neon at list price, about half with `:batch`, plus the judge.
- Re-measure whenever the pinned slug changes.

## 5. Per-window routing score

### Inputs
- **Window W:** the last 256–512 thinking tokens plus the user prompt.
- **Topic posterior:** π_z(W) from kNN over item and topic embeddings (any small embedding model, e.g. `BAAI/bge-small-en-v1.5`, which I confirmed exists on HF):

```
π_z ∝ Σ_{j ∈ kNN} exp(cos(e_W, e_j) / T_e) · 1[z_j = z]
```

  Or a logistic-regression probe, or Arch-Router-1.5B zero-shot. Smooth it with an EMA: π̄_t = (1−η)·π̄_{t−1} + η·π(W_t), η ≈ 0.3–0.5.
- **Recency gate:** r(W) = P(time-sensitive | W), from logistic regression on embeddings trained on FreshQA never/slow/fast labels plus RecencyQA (seconds on CPU).

### Scores
- Recency score (months):

```
S_m(W) = Σ_z π_z(W) · τ̃^CUR_{m,z}
```

  Use τ^CUR-D if the date is injected in deployment.
- Risk- and staleness-adjusted version:

```
U_m(W) = S_m(W) − κ · √(Σ_z π_z² · Var τ_{m,z}) − λ · Σ_z π_z · H_{m,z}
```

  Suggested starting values: κ ≈ 1, λ ≈ 2 months per unit hazard. These are engineering defaults, not literature values.
- Alternative single-unit utility, "expected net currentness" over recent change months:

```
Q_{m,z} = Σ_t w(t) · [P_c(m,z,t) − ρ · P_stale(m,z,t)]
w(t) ∝ exp(−(T_now − t)/λ_t),  t ∈ last 12 months,  ρ ≈ 1
```

### Decision rules
- **Deterministic:** switch to m* = argmax U_m only if r(W) ≥ r_0 and U_{m*} − U_current ≥ h (h ≈ 0.5 month hysteresis). Otherwise keep the current model.
- **Softmax:**

```
P(m | W) = exp(U_m / T(W)) / Σ_{m'} exp(U_{m'} / T(W)),   T(W) = T_0 / max(r(W), r_min),   T_0 ≈ 0.5–1 month
```

  Time-insensitive windows mix almost uniformly, which gives the mixing the user wants. Time-sensitive windows route sharply. Example: S = (Sol 0, Neon −1.5, Fable −2) relative months at T = 1 gives P ≈ 0.74 / 0.16 / 0.10.
- **Thompson / probability matching** (no temperature to tune): draw τ^(b) from the posterior or bootstrap and choose m = argmax_m Σ_z π_z · τ^(b)_{m,z}. Then P(m) equals the posterior probability that m is the most recent model for this topic mix.

### Optional learned component (fits the ≤1 h budget)
- Train a correctness predictor f(e_W, m) → P(current) on the probe outcomes: RouteLLM-MF style or BERT-base, minutes on 1 GPU (my estimate).
- Deploy it only if it beats kNN plus the index on held-out topic×month items. The Routing Plateau and kNN-router results suggest it usually will not.

## 6. Unverified points
- RouteLLM wall-clock training times.
- IRT-Router, EmbedLLM and Arch-Router training costs.
- What OpenRouter does with `temperature` for Fable 5.2 and Sol.
- Output-token cost per probe.
- Price et al. precision/recall and model sizes.
- Whether per-topic slopes differ materially for these three models.
- All engineering defaults: h, κ, λ, T_0, η, window size.

Files are in (scratchpad)/topic-cutoff-lit:
- sim/sim.py
- sim/fit_global.py
- dl/kc_data.json
- dl/score.py
- hf/events.jsonl
- hf/results.jsonl
- or/models.json
