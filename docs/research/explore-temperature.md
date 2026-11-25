# Temperature for cross-domain exploration in cutoffmix (Question A)

Research date: 2026-11-24. Working notes with all raw numbers and the verification log:
`(scratchpad)/explore-research/temperature_notes.md`

How the sources were checked: for each paper I pulled the arXiv export-API record (title, authors, date, comment, journal ref, abstract) and the full-text PDF. I extracted the text with pdftotext and read the numbers from the paper itself. Where a venue comes only from a web search (for example an ICLR or NeurIPS listing), the source table says so.

---

## Kurzfassung (Deutsch, für den Nutzer)

1. **Der Temperatur-Regler ist bei Gemini 4 Neon wirkungslos.** Google schreibt in der offiziellen Neon-Doku wörtlich: *"Remove `temperature`, `top_p`, and `top_k` from requests. The server ignores in-range custom values if sent. Out-of-range values return a `400` validation error."* OpenRouter listet `temperature` zwar als "accepted", aber das heißt nur, dass der Parameter durchgereicht wird. Er hat keine Wirkung. Fable 5.2 und GPT-6.2 Sol haben den Parameter gar nicht. **Damit hat im Relay keines der drei Modelle eine echte Temperatur.** Für den Vertex-Pfad gibt es keine eigene Doku, ein kleiner A/B-Test (siehe 1.3) würde das klären.
2. **Auch wo Temperatur funktioniert, bringt sie selten Ideen aus fremden Fachgebieten.** Die Studien zeigen: Höhere Temperatur verändert vor allem die Formulierung, kaum den Inhalt. Der Effekt auf Neuheit ist schwach (β = .308), der auf Inkohärenz mittelstark (β = .240). Bei GPT-4o hebt eine Erhöhung von 1.0 auf 1.5 die Ideen-Kategorien nur von 22 auf 23 (Menschen: 28). Bei 2.0 kommt "garbled" Text heraus.
3. **Lange Ketten brechen bei hoher Temperatur zuerst.** Mit Mistral-7B fällt GSM8K-CoT bei T = 1.5 auf 0 %, mit min-p auf 18 %. Bei GPT-3.5 fällt die Genauigkeit ab T = 1.0 schnell, bei 1.6 ist der Text inkohärent. In einem Relay wird jeder Fehler zum Kontext für das nächste Modell.
4. **Mode Collapse lässt sich mit Temperatur nicht beheben.** Selbst bei T = 2.0 mit min-p sind 61 % der Antwortpaare zu mehr als 0.8 ähnlich. Verschiedene Hersteller landen bei denselben Attraktoren (71–82 % Ähnlichkeit zwischen Modellen).
5. **Was nachweislich wirkt, und zwar bei allen drei Modellen gleich:** Die Zufälligkeit auslagern, also per Client-Zufall eine fachfremde "Linse" pro Explorationssegment vorgeben (+40 % Originalität). Dazu eine Pflichtstruktur: Abbildung, Vorhersage, Bruchstelle, Rückkehr zur Wissenschaft, die Kohärenz und Nutzen sichert. Weitere Mittel sind IFG (erst ein kurzer "Intent", dann die Notiz), Verbalized Sampling (1.6–2.1× mehr Diversität), bewusste Rotation der Schreiber und mehrere kurze Läufe mit verschiedenen Linsen statt eines "heißen" Laufs.

---

## 1. The hard constraint first: Gemini 4 Neon ignores temperature

### 1.1 Primary source (verified 2026-11-24)
- `https://aistudio.google.com/docs/whats-new-gemini-4-neon` (raw HTML fetched; verbatim, under "Migration checklist"):
  > "**Remove sampling parameters:** Remove `temperature`, `top_p`, and `top_k` from requests. The server ignores in-range custom values if sent. Out-of-range values return a `400` validation error."
- The same text appears on `whats-new-gemini-4-hydrogen`. The Gemini 3.6 Flash page says: "`temperature`, `top_p`, and `top_k` are deprecated and ignored. In future model generations, supplying these parameters returns an HTTP 400 error." It says this applies to "all future Gemini model releases".
- Gemini API changelog: 2026-07-21 "The sampling parameters `temperature`, `top_p` and `top_k` are now deprecated." 2026-11-18 `gemini-4-neon` GA, with thinking levels `low`/`medium`/`high` (default)/`xhigh`. Neon "rejects `minimal` with an error".
- OpenRouter model page for `google/gemini-4-neon` lists `temperature`, `top_p` and `seed` as "accepted by this model". Its providers are Google AI Studio and Google Vertex. **"Accepted" means the parameter passes validation. It does not mean the model applies it.**
- Not verified: a Vertex-specific statement for Neon. I could not find a Vertex model page for Neon. Vertex very likely serves the same model behaviour, but this is unconfirmed.
- Secondary context, not verified: a web-search summary of the Vertex "Gemini 3 prompting guide" says temperatures below 1.0 "may lead to unexpected behavior, looping, or degraded performance". A direct fetch of that page redirected elsewhere, so I am not relying on it.

### 1.2 Consequence for the codebase
- `/home/user/EscapeTheCutOff/cutoffmix/types.py:31` has the comment `supports_temperature ... Gemini 4 Neon: yes`. `/home/user/EscapeTheCutOff/cutoffmix/backends/openrouter.py:389` then sends `temperature` to Neon. Per Google's docs, an in-range value is a **silent no-op**. A value above 2.0 is a **400** error (2.0 is the last documented Gemini range maximum).
- `ChatRequest.temperature` defaults to `0.0` (`types.py:79`). For Neon this is harmless only because it is ignored. It should default to `None` (omit) so the request bodies say what actually happens.
- **A "Neon temperature schedule" therefore cannot be the mechanism for exploration mode.** Cross-domain wandering has to come from the prompt or protocol. That is also the only lever that works the same way on Fable and Sol (see section 4).

### 1.3 Cheap empirical check (recommended before deciding; about $0.10–0.20 at Neon's $3/$15 per M)
1. Arm A: omit `temperature`. Arm B: `temperature: 2.0` (in range). n = 8 each, same short prompt (for example "Give one unusual analogy for <science concept> in at most 40 words"), reasoning effort `low`, small `max_tokens`.
2. Score both arms with the topic classifier's encoder: mean pairwise cosine similarity, distinct-2, and a garbling check.
3. Run one call with `temperature: 2.5`. If the docs are right, it should return 400.
4. Decision rule: if arm B is not more diverse (permutation test) and is not garbled at 2.0, the parameter is ignored. Mark Neon `supports_temperature = False` and drop the schedule.
5. A real effect is possible if OpenRouter routes to Vertex and Vertex still honours the parameter. Only in that case does the conditional schedule in 3.2 apply.

---

## 2. Does high temperature produce creative cross-domain leaps, or mainly incoherence?

**Short answer: mainly surface variation, a small increase in novelty, and a sharp coherence cliff. It does not systematically produce domain-crossing inferences.**

### 2.1 Novelty gains are small and mostly lexical
| Study (verified) | Setup | Finding |
|---|---|---|
| Peeperkorn et al. 2024, ICCC'24, arXiv 2405.00492 | Llama-2-Chat-70B stories; T ∈ {.001 … 2.0}; 36 human raters | Temperature → novelty β = .308 (SE .138, weak). Temperature → **incoherence** β = .240 (SE .122, moderate). No effect on typicality or cohesion. Embedding distance to the exemplar barely grows: "higher temperatures (t > 1.0) do not necessarily lead to more diversity". Conclusion: no compelling support for temperature as "the creativity parameter". |
| Ahmed, Berdica et al. 2025 (IFG), arXiv 2506.09659 | Theory plus experiments | Raising T increases the entropy of both intent and phrasing, but "the increase tends to be dominated by changes in phrasing". |
| Deng, Brucks, Toubia 2026, arXiv 2602.20408 | GPT-4o product ideation, 990 ideas/condition | T 1.0 → 1.5: idea categories 22 → 23, combinations 89 → 97. Humans: 28 / 209. At T 2.0 outputs were "frequently incoherent or garbled". "Higher temperature does not close the human–LLM diversity gap." |
| Terwiesch, Meincke et al. 2026, arXiv 2607.27553 | GPT-4o ideation | Diversity 0.618 (T 0) → 0.679 (T 1). ">1.0 often lead to unstable output"; above 1.0 "often becomes nonsensical". |
| Chen & Ding 2023, Findings EMNLP 2023, arXiv 2310.11158 | Divergent Association Task (10 unrelated words) | T 0.1 → 1.0 raises DAT for weaker models but **not for GPT-4** ("high-probability tokens are well aligned with high-quality answers"). Higher T "bring[s] instability and produce[s] invalid answers". |
| Bellemare-Pepin et al., Sci. Reports 16 (2026), arXiv 2405.13012 | DAT vs 100,000 humans | GPT-4 DAT rises with T (0.5/1.0/1.5). Mean 85.6 at 1.5 is above 72 % of humans, but highly creative humans remain ahead. **A prompted strategy ("varying etymology") also beat the plain prompt.** Temperature mattered for long, loosely constrained texts, not for haiku. |
| Li et al. 2025, Procedia CS / IJCNN-2025 workshop, arXiv 2506.07295 (weak method: GPT-3.5 judge) | Open models 1–80B, T 0.1–1.9 | Creativity is an inverted U. Optimum ≈ **1.3** for medium/large models, with a drop at **1.7**. Instruction following is flat up to 1.0, then hits an abrupt "mutation" at 1.6–1.9 for 40–80B models. Larger models tolerate higher T. |

Reading across these studies: temperature mostly helps with single-word or very short divergence (DAT). For idea-level or argument-level novelty the gain is a few percent. Nothing in the literature shows that temperature makes a model reach into a different scientific field. That kind of jump depends on content, and content follows the prompt.

### 2.2 Where the incoherence cliff is
- Renze & Guven 2024 (Findings EMNLP 2024, arXiv 2402.05201): 9 LLMs (Claude 3 Opus, GPT-4, Gemini 1.0/1.5 Pro, …). Problem-solving accuracy shows **no significant change for T 0.0–1.0**. For GPT-3.5 CoT, "performance began to drop rapidly after a temperature of 1.0 until the generated text became incoherent at 1.6", with accuracy reaching about 0.
- Du, Yang, Welleck 2025 (ICML 2025, arXiv 2502.05234): token entropy rises slowly with T, then jumps (super)exponentially at an "entropy turning point". The jump reflects a real shift in the sampling distribution, which means a drop in quality. Fine-tuned (sharper) models put that point higher, around T ≈ 0.9–1.1 for task-fine-tuned models. This explains why the cliff is sudden rather than gradual.
- Holtzman et al. 2020 (ICLR, arXiv 1904.09751): sampling from the full distribution produces incoherent text because of the unreliable tail. Truncation (top-p) is what keeps high-T text fluent.

### 2.3 Long chains and reasoning models (the relay is a long chain)
- Min-p paper, Table 2 (Mistral-7B, GSM8K CoT 8-shot accuracy):
  - Temperature only: 29.6 (T 0.7) / 17.5 (T 1.0) / **0.0** (T 1.5).
  - Top-p: 36.1 / 27.7 / 0.7.
  - Min-p: 35.2 / 30.9 / **18.4** (T 1.5) / 6.2 (T 2.0) / 0.0 (T 3.0).
  - Short-answer GPQA survives high T with min-p; **multi-step chains do not**. Errors compound with every sampled token of the chain.
- Vendor guidance for reasoning models (model cards fetched): DeepSeek-R1, "Set the temperature within the range of 0.5–0.7 (0.6 is recommended) to prevent endless repetitions or incoherent outputs". Qwen3 thinking mode, T 0.6 / top-p 0.95, "DO NOT use greedy decoding … endless repetitions". So reasoning models break at both ends: low T loops, high T derails.
- Audran-Reiss et al. 2025 (Meta, arXiv 2511.15593): ideation diversity causally improves AI-research-agent performance on MLE-bench, but only when diversity is **controlled by the prompt**. Sweeping DeepSeek-R1 temperature over 0.05 / 0.2 / 0.6 / 1 / 2 "does not have an impact on performance" (medal rate).
- Wu, Mirhoseini, Tambe 2025 (arXiv 2510.02611): different temperatures (0.0–1.2) solve different problem subsets. Spreading samples across temperatures adds 7.3 points over single-temperature test-time scaling for Qwen3 0.6–8B. **The benefit comes from diversity across independent runs, not from one long hot chain.**
- Salah & Muneer 2026 (arXiv 2604.08563; small, 39 problems, Grok-4.1): the best T depends on the prompting style. This is weak evidence that no single fixed T fits every task.

### 2.4 Why temperature does not fix mode collapse, even across vendors
- Kirk et al. 2024 (ICLR 2024, arXiv 2310.06452): RLHF "significantly reduces output diversity" compared with SFT.
- West & Potts 2025 (arXiv 2505.00047): aligned models lose to base models on randomness and creative originality. Higher benchmark scores correlate with worse performance on these tasks.
- Zhang et al. 2025, Verbalized Sampling (ICML 2026 per PDF header, arXiv 2510.01171): the root cause is typicality bias in preference data. Temperature sweeps (0.4–1.4, GPT-4.1 and Gemini-2.5-Flash) trade diversity against quality. VS shifts the whole Pareto front out at every temperature.
- Jiang et al. 2025, Artificial Hivemind (NeurIPS 2025 D&B oral, arXiv 2510.22954):
  - At top-p 0.9 and T 1.0, average pairwise similarity is above 0.8 in 79 % of cases.
  - **Even with min-p 0.1 at T 2.0, 81 % of pairs are above 0.7 and 61.2 % above 0.8.**
  - Between different models, similarity is 71–82 %. For "metaphor about time", 25 models × 50 samples fall into two clusters ("river", "weaver").
- Si, Yang, Hashimoto (ICLR 2025, arXiv 2409.04109): LLM research ideas are rated more novel than experts' (p < 0.05), but of 4000 generated seed ideas only about 200 (5 %) are non-duplicates.
- Implication for cutoffmix: three frontier vendors in one relay **share attractors**. Pooling vendors helps somewhat (Terwiesch et al. 2026: pooling "almost reaches human-level diversity"), but it does not replace an explicit diversification mechanism.

### 2.5 Min-p: not available here, and contested
- Nguyen et al. 2025 (ICLR 2025 oral, arXiv 2407.01082) recommends min-p 0.05–0.1 at T 2–3. Schaeffer et al. 2025 (arXiv 2506.13681) re-analysed the evidence and found min-p "did not outperform baselines in quality, diversity, or a trade-off" once hyperparameter count is controlled. Top-nσ (arXiv 2411.07641) is another temperature-invariant truncation method.
- None of these samplers is exposed for the three relay models, so this section is background on the mechanism only.

---

## 3. What high temperature can and cannot be expected to do

**It can:**
- Increase lexical and phrasing variety, and lower repetition of stock words (DAT, Bellemare-Pepin).
- Slightly raise novelty ratings (β ≈ .3), mainly in long, loosely constrained text.
- Diversify outcomes **across independent runs**, which improves coverage when you pick the best of several (Wu et al.; IFG pass@k).

**It cannot:**
- Reliably make the notes borrow from another field. Domain choice is content, temperature mostly changes phrasing, and aligned models collapse to shared attractors even at T 2.0 with min-p.
- Overcome mode collapse or the inter-model homogeneity of the three vendors.
- Stay coherent in long multi-step chains above about 1.3–1.5. Instruction following breaks abruptly around 1.6–1.9 for large models and earlier for smaller ones.
- Do anything at all on Gemini 4 Neon via the Gemini API, which ignores the value. Fable 5.2 and GPT-6.2 Sol have no such parameter.

---

## 4. Recommendations for gemini-4-neon in the relay

### 4.1 Default (high confidence)
- **Omit `temperature` and `top_p` for Neon.** Change `supports_temperature` for Neon to `False`, or rename the flag to something like `accepts_temperature` with a comment saying the value is ignored. Change the `ChatRequest.temperature` default from `0.0` to `None`.
- Never send values above 2.0 (400 error).
- Neon thinking level: use `low` for middle and explore segments (fewer hidden tokens that can drift; cheaper). Use `high`, the default, only for the final synthesis or answer if needed. `minimal` is rejected.
- Build exploration from the prompt and protocol (4.3). That works the same on all three writers, so exploration does not depend on whether the router happens to pick Neon.

### 4.2 Only if the A/B test in 1.3 shows the parameter is honoured (for example via Vertex)

| Relay phase | Segments | Temperature | top_p | Rationale |
|---|---|---|---|---|
| Anchor | First 1–2 (restate the question, plan) | omit (server default) | omit | Restating the question and planning need instruction following; most problem solving is flat from T 0 to 1.0. |
| Explore | Middle segments flagged as lens segments | **1.2** (hard cap 1.3) | 0.95 (0.9 if T ≥ 1.2) | Creativity optimum ≈ 1.3 for large models (Li). 1.0 → 1.5 gives only +1 category (Deng). Above 1.5, chain collapse (min-p table) and incoherence (Renze, Deng). Top-p trims the garbage tail (Holtzman). Do not lower top-p below 0.9 at the same time, or it cancels the effect. |
| IFG variant (preferred) | Within an explore segment | ≤ 30-token "lens/intent" line at 1.3, then the 128-token note at default | 0.95 | High T only where one token changes the concept, low T where coherence matters (IFG; segment-level alternation is described in the paper). |
| Converge | Last 2 segments + final answer | omit | omit | The answer must be coherent and consistent. |

- Never schedule T ≥ 1.5 in this relay. Never go below 1.0 on Gemini (secondary report of looping below 1.0).
- Assumption, not verified: if honoured, temperature probably also applies to Neon's hidden reasoning. That makes derailment more likely than in the visible notes alone, so keep thinking at `low` during hot segments.

### 4.3 Mechanisms that actually produce cross-domain inference (model-agnostic, evidence-backed)
1. **Externalise the randomness.** For selected middle segments, a client-side random generator draws a lens domain from a curated list (for example thermodynamics, ecology, economics, control theory, immunology, linguistics, music theory, materials science). This is true randomness without any decoding cost.
   - Evidence: Mehrotra et al. 2024 (arXiv 2405.06715), random-concept association gave +40 % originality, but lower usefulness. Bellemare-Pepin, an explicit strategy prompt beat the default prompt. Terwiesch et al. 2026, heterogeneous personas and constraints restore diversity.
2. **Make integration mandatory so the lens is not decoration.** Required template inside a lens segment: (a) the mapping (which element corresponds to which), (b) one testable prediction or inference the mapping yields for the science question, (c) where the analogy breaks, (d) the return to the original question. This addresses the usefulness drop Mehrotra found. It mirrors MOOSE-Chem (ICLR 2025, arXiv 2410.07076), where hypotheses are composed from background plus inspirations, and analogical prompting (Yasunaga et al., ICLR 2024, arXiv 2310.01714).
   - Fable constraint: do not use the words thinking/reasoning/scratchpad/chain of thought or tag syntax. Frame it as "shared working notes: one analyst tests an outside-field analogy".
3. **Verbalized Sampling for the lens choice.** "List 4 candidate analogies from different fields with a rough probability each; develop the least obvious one that still yields a testable inference." VS gives 1.6–2.1× diversity, orthogonal to temperature, and stronger models benefit more (arXiv 2510.01171).
4. **Counter fixation in the shared notes.** Deng et al. 2026: early outputs constrain later ones, so every model will elaborate on the first metaphor it sees. Rotate lenses (no domain repeated within N segments). Optionally ask for short titles of 3 directions before elaborating one; CoT-style decomposition reduced fixation and gave the highest diversity in Meincke 2024 and Deng 2026.
5. **Use vendor pooling deliberately.** In explore segments, bias the router toward a writer who did not write the previous segment, regardless of the cutoff index. Inter-model similarity is high (71–82 %), but pooling still helps (Terwiesch 2026).
6. **Several short seeded runs rather than one hot run.** Diversity across runs is what pays off (Wu et al. 2025; Si et al.: duplication saturates). For example, 3 runs with different lens seeds, then a convergent synthesis.
7. **Measure both breadth and coherence.** Breadth: entropy of the topic classifier's domain distribution over segments; mean embedding distance of lens segments from the question centroid; count of complete mapping-prediction-break-return blocks. Coherence: a judge check that each lens segment returns to the question, plus the garbling rate.

### 4.4 Expected failure modes in a high-temperature relay (if T were honoured) and mitigations
| Failure mode | Evidence | Mitigation |
|---|---|---|
| Incoherent or garbled tokens; cliff rather than slope | Renze (1.6), Deng (2.0 garbled), entropy turning point (Du) | Cap at 1.3; top-p 0.9–0.95; garble detector, resample the segment |
| A spurious claim enters the shared notes and the next writer builds on it | GSM8K-CoT collapse at 1.5; relay passes notes as context | Lens segments must tag analogies as hypotheses ("analogy, not evidence"); converge segments must re-check them |
| Instruction-following break (format, segment length, forbidden words leaking, which can trigger Fable refusals) | IF "mutation" 1.6–1.9 for large models (Li) | High T only on short lens lines (IFG); validate format before appending |
| Topic-classifier and router drift: metaphor-heavy segments get classified as the lens domain, so routing switches model by the wrong cutoff | cutoffmix design | Exclude marked lens segments from the routing window, or down-weight them |
| Homogeneous "creativity" (every model picks the same metaphor) | Hivemind; VS; Deng fixation | Client-side lens draw, lens rotation, VS |
| Loops or repetition, at the low end on Gemini and at the extremes on reasoning models | DeepSeek-R1/Qwen3 cards; Gemini 3 report (secondary) | Never send T < 1.0 to Gemini; keep the default |

---

## 5. Sources

### Verified (arXiv record and full text, or official docs)
| # | Paper | Authors | Year | ID / venue | Status |
|---|---|---|---|---|---|
| 1 | Is Temperature the Creativity Parameter of Large Language Models? | Peeperkorn et al. | 2024 | arXiv 2405.00492; ICCC'24 | verified |
| 2 | The Effect of Sampling Temperature on Problem Solving in Large Language Models | Renze & Guven | 2024 | arXiv 2402.05201; Findings EMNLP 2024 | verified |
| 3 | Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs | Nguyen et al. | 2024/25 | arXiv 2407.01082; ICLR 2025 oral | verified |
| 4 | Min-p, Max Exaggeration: A Critical Analysis of Min-p Sampling | Schaeffer et al. | 2025 | arXiv 2506.13681 (preprint) | verified |
| 5 | Artificial Hivemind: The Open-Ended Homogeneity of LMs (and Beyond) | Jiang et al. | 2025 | arXiv 2510.22954; NeurIPS 2025 D&B oral | verified |
| 6 | Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity | Zhang et al. | 2025 | arXiv 2510.01171; ICML 2026 (PMLR 306, PDF header) | verified |
| 7 | Understanding the Effects of RLHF on LLM Generalisation and Diversity | Kirk et al. | 2023/24 | arXiv 2310.06452; ICLR 2024 (proceedings page via search) | verified |
| 8 | Base Models Beat Aligned Models at Randomness and Creativity | West & Potts | 2025 | arXiv 2505.00047 (venue not confirmed) | verified (arXiv) |
| 9 | Probing the Creativity of LLMs: Can models produce divergent semantic association? | Chen & Ding | 2023 | arXiv 2310.11158; Findings EMNLP 2023 | verified |
| 10 | Divergent Creativity in Humans and Large Language Models | Bellemare-Pepin et al. | 2024/26 | arXiv 2405.13012; Sci. Reports 16 (2026), doi 10.1038/s41598-025-25157-3 (journal via search) | verified |
| 11 | Exploring the Impact of Temperature on LLMs: Hot or Cold? | Li et al. | 2025 | arXiv 2506.07295; Procedia CS (IJCNN 2025 INNS workshop) | verified, weak method |
| 12 | Intent Factored Generation: Unleashing the Diversity in Your Language Model | Ahmed, Berdica et al. | 2025 | arXiv 2506.09659 (ICML 2025 / NeurIPS 2026 listings via search) | verified (arXiv) |
| 13 | Optimizing Temperature for Language Models with Multi-Sample Inference | Du, Yang, Welleck | 2025 | arXiv 2502.05234; ICML 2025 | verified |
| 14 | On the Role of Temperature Sampling in Test-Time Scaling | Wu, Mirhoseini, Tambe | 2025 | arXiv 2510.02611 (preprint) | verified |
| 15 | What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity | Audran-Reiss et al. | 2025 | arXiv 2511.15593 (preprint) | verified |
| 16 | A Cognitive Framework for Idea Diversity in Large Language Models | Deng, Brucks, Toubia | 2026 | arXiv 2602.20408 (preprint) | verified |
| 17 | AI and Its Impact on Creativity and Diversity: An Empirical Study of LLM-Generated Product Ideas | Terwiesch, Meincke et al. | 2026 | arXiv 2607.27553 (preprint) | verified |
| 18 | Prompting Diverse Ideas: Increasing AI Idea Variance | Meincke, Mollick, Terwiesch | 2024 | arXiv 2402.01727 (working paper) | verified |
| 19 | Enhancing Creativity in LLMs through Associative Thinking Strategies | Mehrotra, Parab, Gulwani | 2024 | arXiv 2405.06715 (venue not confirmed) | verified (arXiv) |
| 20 | Can LLMs Generate Novel Research Ideas? | Si, Yang, Hashimoto | 2024/25 | arXiv 2409.04109; ICLR 2025 (poster page via search) | verified |
| 21 | The Curious Case of Neural Text Degeneration | Holtzman et al. | 2019/20 | arXiv 1904.09751; ICLR 2020 | verified (abstract) |
| 22 | MOOSE-Chem | Yang et al. | 2024/25 | arXiv 2410.07076; ICLR 2025 | verified (abstract) |
| 23 | Large Language Models as Analogical Reasoners | Yasunaga et al. | 2023/24 | arXiv 2310.01714; ICLR 2024 | verified (abstract) |
| 24 | Temperature-Dependent Performance of Prompting Strategies in Extended Reasoning LLMs | Salah & Muneer | 2026 | arXiv 2604.08563 (preprint, n = 39) | verified, weak |

**Docs:**
- Google AI Studio, "What's new in Gemini 4 Neon" / "… Hydrogen" / "… Gemini 3.6 Flash", and the Gemini API changelog (all fetched 2026-11-24).
- OpenRouter `google/gemini-4-neon` page.
- DeepSeek-R1 and Qwen3-8B Hugging Face model cards.

### Unverified (not relied on)
- Whether Vertex AI (one of the two OpenRouter providers for Neon) ignores temperature like the Gemini API. No Vertex Neon page was found.
- The Vertex "Gemini 3 prompting guide" wording about looping below T 1.0. I saw it only in a search summary; the direct fetch redirected.
- The final venues of West & Potts, Mehrotra et al., and IFG (ICML 2025 vs NeurIPS 2026 listings).
