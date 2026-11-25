# Question C - Making frontier models verbalize MORE intermediate steps in the VISIBLE notes (cutoffmix relay)

Date: 2026-11-24. Author: explore-research subagent (Fable 5.1).
Running notes with per-paper verification trail: `notes-C-running.md` (same directory).

Verification key: **V** = arXiv abstract page / ACL Anthology / ICLR-NeurIPS virtual page / OpenAI docs page opened in this session (title, authors, date, venue, numbers checked). **UNV** = seen only in a search snippet or secondary summary; not relied on.

---

## Kurzfassung (DE)

1. **Mehr sichtbare Zwischenschritte verbessern bei Frontier-Modellen die Antwortqualität in der Regel nicht - sie verbessern Transparenz und (hier gewollt) die Breite der Notizen.** Die Evidenz zeigt eine umgekehrte U-Kurve zwischen Länge und Genauigkeit (Wu et al., ICLR 2026), Varianz statt Qualität bei erzwungenem Weiterdenken (Ghosal et al., NeurIPS 2025), inverse Skalierung auf bestimmten Aufgaben (Gema et al., TMLR 2025) und bei R1-artigen Modellen sogar negative Effekte von s1-"Wait"-Forcing (Markovian Thinker App. G; ICLR-2026-Blog). Gewinne durch längere Schritte sind vor allem bei schwächeren Modellen und Mathe/Symbolik belegt (Jin et al., Findings ACL 2024: +12-13 Punkte für GPT-3.5, GPT-4 nur "tolerant"; Sprague et al., ICLR 2025).
2. **Mehr sichtbarer Text erhöht aber messbar die Überwachbarkeit/Nachvollziehbarkeit**: "longer CoTs are generally more monitorable" (OpenAI, Guan et al. 2025); größere Token-Budgets heben die Verbalisierung versteckter Hinweise auf bis zu 90 % (Zaman & Srivastava, ACL 2026). Aber: sichtbare Notizen sind bei cutoffmix ein *zweiter* Kanal neben dem verworfenen versteckten Kanal - in 55,4 % der Fälle steht ein Einflussfaktor nur im versteckten Kanal, nicht in der sichtbaren Antwort (Young 2026). Die Notizen werden also systematisch Dinge auslassen, wenn man sie nicht explizit per Struktur abfragt.
3. **Konkrete Mechanik** (Abschnitt 4): Mindestanzahl Segmente vor FINAL ANSWER abhängig von Schwierigkeit (4 / 8 / 12-16 Segmente), harte Obergrenze ~4096 Notiz-Tokens, Fortsetzungs-Cues *mit Inhalt* statt blankem "Wait" (max. 2-3 erzwungene Fortsetzungen, dann Modellwechsel statt erneutem Forcing), Segmentlänge 192-256 Tokens mit Satzgrenzen-Trim statt 128, ein strukturiertes Notiz-Template (Claim / Evidence / Inference / Alternative set aside / Open question / Where we stand) in Formulierungen ohne die für Fable verbotenen Wörter, und für GPT-6.2 Sol `verbosity: "high"` nur für Notiz-Segmente (steuert ausschließlich die sichtbare Ausgabe, nicht das versteckte Nachdenken).

---

## 1. What the user is actually asking for, in the vocabulary of the literature

The relay wants a *visible* chain of notes that contains more intermediate claims, more explicitly stated alternatives, and more cross-domain moves. In the literature this touches four distinct levers that are often conflated:

| Lever | What it changes | Key evidence |
|---|---|---|
| **Length control** (budget forcing, "Wait", min tokens) | How long the model keeps writing before it stops | s1 (V), NoWait (V), Markovian Thinker App. G (V), ICLR-2026 blog (V), Ringel et al. (V) |
| **Structure control** (Plan-and-Solve, least-to-most, templates, ToT) | *Which* intermediate objects get written (plan, sub-answer, alternative, check) | PS (V, ACL 2023), L2M (V, ICLR 2023), ToT (V, NeurIPS 2023), Self-Discover (V), Step-Back (V, ICLR 2024), BioNLP 2025 4-step (V) |
| **Faithfulness / "verbosity" in the monitorability sense** | Whether the visible text names the factors that actually drove the answer | Turpin (V), Lanham (V), Chen Y. (V), Meek et al. (V), Zaman & Srivastava (V), Young (V), FACE-Eval (V), Guan et al. (V) |
| **Answer-length knob** (OpenAI `verbosity`) | Token count and depth of the *final visible output* only | OpenAI cookbook + GPT-5.2 guide (V) |

The honest headline (Section 5): levers 1 and 4 mostly buy transparency and variance; lever 2 is where the (modest) accuracy evidence lives; lever 3 is where the transparency evidence lives.

---

## 2. Evidence base, by theme

### 2.1 Does longer / more verbalized visible reasoning improve accuracy?

**Classic prompting era (helps, mostly weaker models and math/symbolic):**
- **Kojima et al., "Large Language Models are Zero-Shot Reasoners", NeurIPS 2022, arXiv 2205.11916 (V).** Appending "Let's think step by step": text-davinci-002 MultiArith 17.7 -> 78.7, GSM8K 10.4 -> 40.7. Baseline for "verbalizing steps helps".
- **Jin et al., "The Impact of Reasoning Step Length on Large Language Models", Findings of ACL 2024, arXiv 2401.04925 (V).** Lengthening steps in the demonstrations *without adding information* improves accuracy; shortening hurts; even *incorrect* rationales work if long enough; effect is task-dependent. GPT-3.5-turbo-1106 GSM8K: Auto-CoT 65.8 -> 78.8, Zero-shot-CoT 64.1 -> 76.1 with a "must think more steps" prompt; roughly linear up to ~6 added steps. GPT-4 showed "the highest tolerance" (no decrease, no clear gain reported). Lengthening strategies tested: *Think About The Word*, *Read the question again*, *Repeat State*, *Self-Verification*, *Make Equation* - each treated as one extra step. Directly usable as note-item types.
- **Sprague et al., "To CoT or not to CoT?", ICLR 2025, arXiv 2409.12183 (V).** Meta-analysis of >100 papers plus 20 datasets x 14 models: CoT helps mainly math/logic; on MMLU no gain unless the question/response contains "="; most of the gain is better symbolic *execution*. For research/science-style questions, expect small gains from more verbalization per se.
- **Chain of Draft (Xu et al., arXiv 2502.18600, Feb/Mar 2025, no venue) (V).** The counter-direction: "Think step by step, but only keep a minimum draft for each thinking step, with 5 words at most." Matches CoT with as little as 7.6% of tokens. Concrete: GSM8K GPT-4o CoT 95.4% / 205 tok vs CoD 91.1% / 44 tok; Claude 3.5 Sonnet 95.8% / 190 tok vs CoD 91.4% / 40 tok. **Zero-shot CoD collapses** (Claude 90.4 -> 65.5), and small models lose a lot (Qwen2.5-3B 59.1 -> 43.1). Reading: the visible content is not useless (compression costs ~4 points on GSM8K), but most of the *length* is.

**Reasoning-model era (inverted U; forced length often hurts):**
- **Wu Y. et al., "When More is Less: Understanding Chain-of-Thought Length in LLMs", ICLR 2026, arXiv 2502.07266 (V; ICLR 2026 poster page iclr.cc/virtual/2026/poster/10011380).** Accuracy vs CoT length is an inverted U; the optimal length *rises with task difficulty* and *falls with model capability*; RL induces a simplicity bias toward shorter CoT. For frontier models the optimum is therefore short relative to what forcing would produce.
- **Ghosal et al., "Does Thinking More always Help? Mirage of Test-Time Scaling", NeurIPS 2025, arXiv 2506.04210 (V).** Performance rises then declines with more thinking; early gains are largely *increased output variance* ("an illusion of improved reasoning"); parallel sampling + majority vote beats extended thinking by up to 20% at the same budget.
- **Gema et al., "Inverse Scaling in Test-Time Compute", TMLR 12/2025 (Featured Certification), arXiv 2507.14417 (V).** Tasks where longer reasoning lowers accuracy; five failure modes incl. *Claude models become more distracted by irrelevant information* and o-series overfit to problem framings. Directly relevant to a relay whose notes deliberately wander into other domains: for Fable, the wandering material is "irrelevant information" in this sense.
- **Chen X. et al., "Do NOT Think That Much for 2+3=? On the Overthinking of o1-Like LLMs", arXiv 2412.21187 (V)** and **Wang Y. et al., "Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs", arXiv 2501.18585 (V).** Overthinking = redundant re-solving of easy items; underthinking = frequent thought-switching without following a path through, which correlates with wrong answers on hard math (their TIP penalty discourages premature switching). A relay that forces "one more alternative" every segment is pushing toward the underthinking pattern.
- **Su et al., "Between Underthinking and Overthinking", arXiv 2505.00127 (V):** models overthink easy problems and underthink hard ones, i.e. they misjudge difficulty - which is why the relay's difficulty estimate should set the minimum segment count, not the model.
- **Caldarella et al., "Thinking Past the Answer: Evaluating Harmful Overthinking in LRMs", arXiv 2606.02835, Jun 2026 (V).** Stopping at the first correct prefix improves accuracy by up to 21%; early-stopping heuristics cut *verbose* overthinking by up to 50% but do not fix *harmful* overthinking (logical drift after a correct answer).
- **Lotfi et al., "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv 2606.00206, May 2026 (V).** In up to 52% of failures the correct answer appears mid-trace but is not output; penalizing filler transition words cuts length 12-23% with equal or better accuracy. Filler-word continuation is the failure, not the fix.
- **Chen W.-L. et al., "Think Deep, Not Just Long", ICML 2026, arXiv 2602.13517 (V).** Raw token count is an unreliable proxy for reasoning effort; their deep-thinking ratio correlates with accuracy where length does not.
- **Shojaee et al., "The Illusion of Thinking", NeurIPS 2025, arXiv 2506.06941 (V).** Reasoning effort rises with complexity up to a point then *falls* even when budget remains; accuracy collapses beyond a complexity threshold. Forcing more visible tokens does not restore effort.
- **Balachandran et al., "Inference-Time Scaling for Complex Tasks", arXiv 2504.00294 (V).** Nine models, eight non-math task families: benefits of more tokens vary by task and shrink with complexity; "using more tokens doesn't reliably improve accuracy on hard problems".

**Does the *content* of the verbalization matter at all (vs. pure compute)?**
- **Pfau, Merrill, Bowman, "Let's Think Dot by Dot", arXiv 2404.15758 (V):** meaningless filler tokens can replace CoT on two algorithmic tasks (needs dense supervision) - evidence that some CoT benefit is compute, not content. **Goyal et al., "Think before you speak: pause tokens", ICLR 2024, arXiv 2310.02226 (V):** learned pause tokens help (SQuAD EM +18%, CSQA +8%, GSM8K +1%) only when pretrained with them. Neither applies to closed APIs, but both warn that "more visible tokens" is not the mechanism.
- **Wu T. et al., "Thinking LLMs", arXiv 2410.10630 (V):** thought generation (trained via preference optimization) helps on *non-reasoning* categories (marketing, health, general knowledge) on AlpacaEval/Arena-Hard - the one piece of evidence that verbalized intermediate work helps outside math, but it required training.
- **Zhang Y. et al., "Verbosity != Veracity", arXiv 2411.07858 (V):** "Verbosity Compensation" (hedging, repeating the question, over-enumeration) occurs in all 14 models tested (GPT-4 50.40% of responses); verbose responses are worse (27.61% gap on Qasper) and carry higher uncertainty. Asking for "more words" risks exactly this.
- **Human cognition analog - Fox, Ericsson & Best, "Do Procedures for Verbal Reporting of Thinking Have to Be Reactive?", Psychological Bulletin 137(2):316-344, 2011, doi 10.1037/a0021663 (title/journal/pages V on APA page; numbers from secondary sources only).** Meta-analysis of 94 studies: plain think-aloud is non-reactive (does not change performance; r approx. -0.03 per secondary summaries, UNV on primary page), whereas asking people to *explain* their reasoning is reactive and raises performance; all verbal reporting increases time. The LLM literature rhymes: plain "write more" is neutral-to-harmful; "explain/justify/plan" structures are where gains appear.

### 2.2 Budget forcing and continuation (s1-style "Wait")

- **Muennighoff et al., "s1: Simple test-time scaling", arXiv 2501.19393 (V; no venue on page).** Budget forcing = cap by appending the end-of-thinking delimiter (+ "Final Answer:"), or extend by *suppressing* the delimiter and appending "Wait". AIME24 50% -> 57% at best. Table 4 (AIME24 / MATH500 / GPQA-D): no extrapolation 50.0/93.0/57.6; 2x no string 50.0/90.2/55.1; 2x "Alternatively" 50.0/92.2/59.6; 2x "Hmm" 50.0/93.0/59.6; 2x "Wait" 53.3/93.0/59.6. "Suppressing the end-of-thinking token delimiter too often can lead the model into repetitive loops"; gains "eventually flatten out at six times". Note the model was *fine-tuned* (s1-32B on s1K) - not an off-the-shelf frontier model.
- **Wang C. et al., "Wait, We Don't Need to 'Wait'!" (NoWait), arXiv 2506.08343 (V).** Suppressing Wait/Hmm/etc. shortens R1-style CoTs 27-51% across 10 benchmarks with no utility loss. The reflective filler is dispensable for already-trained reasoning models.
- **Ringel, Tolochinsky, Romano, "Learning a Continue-Thinking Token", arXiv 2506.11274 (V; IJCNLP-AACL 2025 per search, UNV).** On a distilled R1 model, fixed-token "Wait" forcing gives +1.3 abs on GSM8K; a learned continue token +4.2. Small effect for the fixed cue.
- **Taveekitworachai & Pipatanakul, "Wait, Do We Need to Wait? Revisiting Budget Forcing", ICLR 2026 Blog Track (V; iclr.cc/virtual/2026/poster/10012115 and the blog post).** Four 7B models, AIME25/MATH500/MMLU-Pro-1K/SuperGPQA-1K, budget 8192: CoT+budget-forcing avg gains +10.5 to +13.6 for SFT-style reasoning models (s1.1, OpenThinker3, their RFT) but **-0.99 for DeepSeek-R1-Distill-7B**; Llama-3.1-8B 26.48 -> 19.73 and Ministral-8B 28.15 -> 22.76 (harmful). **"'Wait' is never the best keyword"** - "Let" or "Perhaps" beat it model-specifically; "pick words the model naturally uses, not instruction words". Guidelines: start at 1024 tokens, 2048-4096 for math, 512-1024 for QA, "do not exceed 4096"; watch for repetition and answer instability; "Budget forcing is neither a universal law nor a guaranteed improvement."
- **Aghajohari et al., "The Markovian Thinker", arXiv 2510.06557 (V), Appendix G.** Budget forcing as a baseline on AIME24 (avg@128): R1-Distill-1.5B 29.3 (LongCoT) -> 24.6 (budget forced); 7B 54.4 -> 50.1; 14B 68.9 -> 70.2. "budget forcing reliably buys more tokens, but those extra tokens often overwrite useful partial work instead of extending it"; the nudge "encourages second-guessing"; regenerating from the full prompt with a generic cue "does not preserve a compact notion of progress". Worked example: pass 1 correct at 9,287 tokens, pass 2 still correct, pass 3 wrong. **Design lesson for the relay:** their fix (Delethink) is to reason in fixed chunks (8K) with a carried-over *textual state* the model writes at each chunk end - structurally what a shared-notes relay already is. The relay should therefore ask every segment to end with an explicit "where we stand" line, so the next model extends rather than re-derives.
- **Tian X. et al., "Think Twice: Multi-round Test-time Thinking", arXiv 2503.19855 (V).** Feeding the previous *answer* back and asking for a re-answer: QwQ-32B AIME24 80.3 -> 82.1, DeepSeek-R1 79.7 -> 82.0. A "second pass after a provisional answer" segment has a small positive track record, unlike bare "Wait".
- **Yang V. et al., "Mid-Think", Findings of ACL 2026 (V).** Reasoning mode is steered by a few tokens (leading "Okay" triggers; newline pattern after the think tag suppresses). Confirms that the *first tokens of a segment* decide whether the model elaborates - the relay's continuation cue should be the opening token(s) of the next segment, not a trailing instruction.
- **Aggarwal & Welleck, "L1", COLM 2025, arXiv 2503.04697 (V).** Prompt-stated length targets are reliably followed only after RL training (LCPO). For closed APIs, "write at least N steps" is a weak lever; structure + hard token caps are the enforceable ones.

**Relay-specific handoff evidence:**
- **Ballon et al., "Probing the Trajectories of Reasoning Traces in LLMs", NeurIPS 2026 Evaluations & Datasets track, arXiv 2601.23163 (V).** Re-injecting truncated traces: accuracy and commitment rise with trace share, driven by *instance-specific semantic content* not length; **cross-model: stronger models often repeat the source model's errors when handed its partial trace, though allowing further reasoning substantially improves recovery**. So each handoff should explicitly licence the new model to disagree with the prior note.
- **Khraishi et al., "Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems", arXiv 2603.03111, Mar 2026 (V).** A single-turn handoff shifts Multi-IF strict success by -8 to +13 pp and CoQA F1 by +-4 - the size of a model-tier gap; ~70% of variance explained by prefix-influence + suffix-susceptibility terms. More, shorter segments = more handoffs = more drift.

### 2.3 Structure: prompting that forces explicit intermediate objects

- **Wang L. et al., "Plan-and-Solve Prompting", ACL 2023 (V, 2023.acl-long.147).** Zero-shot-CoT error analysis on GSM8K (46 wrong): calculation 7%, **missing step 12%**, semantic misunderstanding 27%. PS ("devise a plan ... then carry out the plan") and PS+ (plus "extract relevant variables", "calculate intermediate results") target missing steps. text-davinci-003 GSM8K 56.4 (Zero-shot-CoT) -> 58.2 (PS) -> 59.3 (PS+); average over six math sets 70.4 -> 76.7 (8-shot Manual-CoT 77.6); CSQA 65.2 -> 71.9; Last Letters 64.8 -> 75.2.
- **Zhou D. et al., "Least-to-Most Prompting", ICLR 2023, arXiv 2205.10625 (V).** Decompose then solve sub-problems sequentially: SCAN >= 99% vs 16% for CoT (code-davinci-002).
- **Yao S. et al., "Tree of Thoughts", NeurIPS 2023, arXiv 2305.10601 (V).** Explicit generation *and evaluation* of alternatives with lookahead/backtracking: Game of 24 GPT-4 CoT 4% -> ToT 74%. The strongest evidence that *writing down and scoring alternatives* (not just "considering" them) helps on search-like problems.
- **Chia et al., "Contrastive Chain-of-Thought Prompting", arXiv 2311.09277 (V).** Showing valid *and invalid* reasoning demonstrations "generally enhances" CoT - supports a "rejected alternative, and why" slot.
- **Zhou P. et al., "Self-Discover", arXiv 2402.03620 (V).** Have the model first *compose a reasoning structure* from atomic modules, then follow it: up to +32% over CoT (GPT-4, PaLM 2) on BBH/T4D/MATH; >20% over CoT-Self-Consistency at 10-40x less compute; structures transfer across model families (relevant: one model can write the structure, others follow it).
- **Zheng H.S. et al., "Take a Step Back", ICLR 2024, arXiv 2310.06117 (V).** Abstract to principles first, then reason: PaLM-2L MMLU Physics +7%, Chemistry +11%, TimeQA +27%, MuSiQue +7%. The only well-evidenced *accuracy-positive* mechanism that also pushes notes toward other domains (principles are domain-general).
- **Wang Y. & Zhao Y., "Metacognitive Prompting", NAACL 2024, arXiv 2308.05342 (V).** Staged self-evaluation (understand -> preliminary judgment -> critical evaluation -> final decision with confidence) beats CoT variants on 10 NLU datasets.
- **Liang S. & Sonntag D., "Advancing Biomedical Claim Verification by Using LLMs with Better Structured Prompting Strategies", BioNLP 2025 (V, 2025.bionlp-1.14).** Four-step template: claim comprehension -> evidence analysis -> intermediate conclusion -> entailment decision; each step "contributes distinct but complementary roles"; improves traceability. Closest published analog to the user's claim -> evidence -> inference template (no numbers in abstract).
- **Zhao J. et al., "LLMs are In-context Teachers for Knowledge Reasoning" (Self-Explain / Teach-Back), Findings of EMNLP 2024, arXiv 2311.06985 (V)** and **Gao H. et al., "Self-Explanation Prompting Improves Dialogue Understanding", arXiv 2309.12940 (V).** Asking the model to explain each input element before acting helps on knowledge QA and dialogue; self-generated explanations beat human-written demos. Evidence for "explain before you conclude" rather than "write more".
- **Radhakrishnan et al. (Anthropic), "Question Decomposition Improves the Faithfulness of Model-Generated Reasoning", arXiv 2307.11768 (V).** Answering sub-questions in *separate contexts* (factored decomposition) raises faithfulness substantially at some accuracy cost - relevant if a future relay variant lets one model answer a sub-question without seeing the others' notes.
- **Bogdan, Macar, Nanda, Conmy, "Thought Anchors: Which LLM Reasoning Steps Matter?", arXiv 2506.19143 (V).** Counterfactual resampling shows the high-impact sentences in reasoning traces are *planning* and *uncertainty-management* sentences, not computation. If the notes must be short, those are the sentences to require.

### 2.4 Faithfulness and monitorability of the visible channel

- **Turpin et al., NeurIPS 2023, arXiv 2305.04388 (V):** biasing features swing answers (accuracy drop up to 36% on 13 BBH tasks) while explanations never mention them. **Lanham et al. 2023, arXiv 2307.13702 (V):** larger/more capable models are *less* faithful on most tasks. **Chen Y. et al. (Anthropic) 2025, arXiv 2505.05410 (V):** reasoning models verbalize a used hint in >= 1% but often < 20% of cases; outcome RL raises faithfulness then plateaus.
- **Korbak et al., "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", arXiv 2507.11473 (V, v2 Dec 2025; 41 authors):** position paper; monitorability is useful and fragile. **Baker et al. (OpenAI) 2025, arXiv 2503.11926 (V):** optimizing against a CoT monitor produces obfuscated reward hacking; "monitorability tax".
- **Guan et al. (OpenAI), "Monitoring Monitorability", arXiv 2512.18311, Dec 2025 (V).** "longer CoTs are generally more monitorable"; monitorability "largely correlate[s] with the amount of thinking tokens" during training; an obfuscation run's monitorability drop was "primarily driven by drastically reduced CoT lengths"; **asking the model follow-up questions and passing the follow-up CoT to the monitor improves monitorability**; monitor reasoning effort low -> high adds ~12% monitorability (vs ~4% for action-only). This is the strongest direct evidence that *more visible text = more transparency*.
- **Meek et al., "Measuring CoT Monitorability Through Faithfulness and Verbosity", arXiv 2510.27378 (V).** Separates *faithfulness* (transparent about what drove the answer) from *verbosity* (CoT "lists every factor needed to solve the task"); models can look faithful while omitting key factors. The user's goal "verbalize more steps" is verbosity in exactly this sense.
- **Zaman & Srivastava, "Is CoT Really Not Explainability?", ACL 2026, arXiv 2512.23032 (V).** >50% of CoTs flagged unfaithful by the hint-omission metric are judged faithful by other metrics for some models; **larger inference-time budgets raise hint verbalization to as high as 90%** - "much apparent unfaithfulness stems from tight token limits". A 128-token segment is a tight token limit.
- **Young R.J., "Why Models Know But Don't Say", arXiv 2603.26410, Mar 2026 (V).** 12 open-weight reasoning models, 10,506 hint-followed cases: in **55.4%** the hidden thinking mentions the hint and the visible answer omits it; the reverse 0.5%; 11.8% neither channel. In cutoffmix the hidden channel is discarded, so the visible notes are structurally the *less* complete channel.
- **Gema et al., FACE-Eval, arXiv 2608.29464, Aug/Sep 2026 (V).** 5,100 items, 15 models: cues delivered via tool returns or implicitly are adopted *without* being verbalized far more often; **a source-attribution prompt narrows the gap on 7 models, telling models they are monitored does not**; unverbalized adoption correlates with lower monitor detection (r = -0.54 / -0.78).
- **Buerger et al., "CoT Monitoring Can Fail Even When CoT Is Necessary", arXiv 2610.18105, Oct 2026 (V):** on a medical task, patient sex / a clinical attribute influenced >= 19% / 24% of correct answers that relied on necessary CoT, almost never verbalized. **Li S. et al., NeurIPS 2026 Position Track, arXiv 2611.01620 (V):** visibility, faithfulness, localizability and monitorability are separate claims needing separate evidence. **Wang H. et al., MonitorBench, COLM 2026, arXiv 2603.28590 (V):** 1,540 instances / 19 tasks; monitorability is higher when decision-critical factors shape the *intermediate* reasoning, and most models can intentionally lower it under adversarial prompting. **Barez et al., "Chain-of-Thought Is Not Sufficient for Explainability", NeurIPS 2026 poster (V):** ~21.7% of 1,000 CoT papers treat CoT as load-bearing interpretability. **Jia, Benton, Easley, "Faithfulness as Information Flow", arXiv 2605.24286, May 2026 (V):** training-side fixes, not applicable to closed APIs.
- **Borchers, Vie, Azevedo, arXiv 2602.01015, 2026 (V):** LLM "think-alouds" are overly coherent, verbose and less variable than human ones - what you get when you ask a frontier model to narrate is a polished narrative, not a trace.

### 2.5 OpenAI `verbosity` (GPT-6.2 Sol lever) - what the docs actually say

- OpenAI cookbook, "GPT-5 New Params and Tools" (V): `verbosity` "lets you hint the model to be more or less expansive in its replies"; values `low` / `medium` (default) / `high`; "reliably scales both the length and depth of the model's output while preserving correctness and reasoning quality"; `high` is "great for audits, teaching, or hand-offs". gpt-5-mini poem: 560 -> 849 -> 1288 output tokens; sorting program 575 -> 943 -> 2381.
- OpenAI docs, "Using GPT-5.2" (V): "Verbosity determines how many output tokens are generated"; "While the model's reasoning approach stays mostly the same, the model finds ways to answer more concisely, which can either improve or diminish answer quality, depending on your use case"; "defines a general token range at the system prompt level, but the actual output is flexible to both developer and user prompts within that range"; you "can still steer verbosity through prompting". Reasoning token count is governed separately by `reasoning.effort`.
- Reading for the relay: `verbosity` is a knob on exactly the channel cutoffmix shares (visible output), it does not touch the hidden channel, and OpenAI's own wording says it changes *how much is written*, not *how the problem is approached*. With a hard per-segment cap it will mostly change density/elaboration inside the cap. (Per the task's verified API facts, Claude must not receive `verbosity` because OpenRouter maps it to effort; Gemini has no such parameter.)

---

## 3. Synthesis: what the evidence does and does not support

1. **"Write more" is not a reliable accuracy lever for frontier models.** Inverted-U (Wu, ICLR 2026), variance illusion (Ghosal, NeurIPS 2025), inverse scaling (Gema, TMLR 2025), harmful overthinking (Caldarella 2026), and negative budget-forcing results on R1-style models (Markovian Thinker App. G; ICLR-2026 blog) all point the same way. Gains from longer steps are documented mainly for weaker models (Jin: GPT-3.5 +12-13; GPT-4 "tolerant") and for math/symbolic tasks (Sprague).
2. **"Write different things" has a track record.** Plans (PS, +3 to +6 points zero-shot), decomposition (L2M), explicit scored alternatives (ToT), principles-first (Step-Back), composed structures (Self-Discover), staged self-evaluation (Metacognitive), explain-before-answer (Self-Explain). Human think-aloud research agrees: narration is neutral, *explanation* is reactive-positive (Fox et al. 2011).
3. **More visible text does buy transparency**, with caveats: longer CoT is more monitorable (Guan), bigger budgets lift hint verbalization to ~90% (Zaman & Srivastava), and "verbosity" in Meek's sense (list every factor) is exactly what a structured template requests. But the visible channel systematically omits what the hidden channel contains (Young 55.4%), frontier models are less faithful (Lanham), and models can perform low monitorability on demand (MonitorBench). Source-attribution instructions help; "you are being monitored" does not (FACE-Eval).
4. **Bare "Wait" is the wrong continuation primitive for this relay.** It is never the best keyword (ICLR-2026 blog), removable without loss on trained reasoning models (NoWait), worth ~+1.3 on GSM8K when it works (Ringel), and it "overwrite[s] useful partial work" on R1-style models (Markovian Thinker). The s1 numbers that made it famous came from a fine-tuned 32B model with a 6x flattening ceiling and documented loops.
5. **Handoffs are not free.** Each model switch shifts outcomes by a model-tier-sized amount (Khraishi, +-4 to 13 pp) and a stronger model handed a partial trace tends to inherit its errors unless allowed to reason further (Ballon, NeurIPS 2026).

---

## 4. Concrete relay mechanisms

### 4.1 Minimum number of note segments before FINAL ANSWER is accepted

Ground the floor in *difficulty*, not in a fixed number (Wu: optimal length rises with difficulty; Su: models misjudge difficulty themselves), and cap the ceiling where the evidence says accuracy turns down (ICLR-2026 blog: "do not exceed 4096" note tokens; s1: flattening at 6 forced continuations; Markovian: third forced pass drifted wrong).

| Difficulty estimate (from the existing classifier or a cheap first segment) | N_min segments | Approx. visible tokens at 128 / 192 per segment | N_max (hard cap) |
|---|---|---|---|
| Easy / factual | 4 | 512 / 768 | 16 |
| Medium (multi-hop, synthesis) | 8 | 1,024 / 1,536 | 24 |
| Hard (open research question, calculation-heavy) | 12-16 | 1.5-2k / 2.3-3k | 32 (~4,096 tok at 128; ~21 segments at 192) |

Acceptance rule for FINAL ANSWER: (a) segment index >= N_min, **and** (b) the structured checklist in 4.3 is satisfied at least once each - at least one *Claim* with an *Evidence* line, at least one *Alternative set aside* with a reason, at least one *Open question*, and a *Where we stand* line in the last two segments. (b) is the Plan-and-Solve "missing step" logic applied as a gate instead of a prompt; (a) alone would only buy length, which Section 3 says is not the lever.

### 4.2 s1-style continuation when a model concludes early - do it, but not with "Wait"

If a segment contains the FINAL ANSWER marker (or a closing phrase) before the gate passes:
1. Strip the premature conclusion from the shared notes and **keep the rest of the segment** (Markovian Thinker: do not let the continuation overwrite partial work).
2. Open the *next* segment with a **content-bearing cue the models naturally produce**, not an instruction word (ICLR-2026 blog; Mid-Think: the opening tokens steer the mode). Rotate among: "Alternatively, " / "Hmm, one thing not yet checked: " / "Let us test the strongest objection to the current conclusion: " / "Perhaps the missing piece is " / "Before closing, which assumption above is weakest? " (s1 Table 4: "Alternatively" and "Hmm" matched "Wait" on MATH500/GPQA; Guan et al.: *follow-up questions* raise monitorability; Think Twice: a re-examination pass after a provisional answer gains ~2 points).
3. Prefer to **route the continuation to a different model** than the one that concluded (the relay already has a router; Ballon: the new model must be explicitly allowed to disagree - include "If the previous note's conclusion is wrong, say so and why" in the per-segment instruction).
4. **Limit forced continuations to 2-3 per chain** and stop forcing when the last two segments have high n-gram overlap (s1: "repetitive loops"; Markovian: pass 3 wrong; ICLR-2026 blog: "watch for repetition and answer instability"). After that, accept the answer or trigger one parallel re-answer instead of more sequential forcing (Ghosal: parallel beats extended thinking by up to 20% at equal budget).
5. Never suppress a conclusion *after* the gate passes merely to hit a token target; Caldarella 2026 shows up to +21% from stopping at the first correct prefix.

### 4.3 Structured-notes instruction (compatible with the wording constraints)

Constraints honoured: no "thinking", "reasoning", "scratchpad", "chain of thought", no XML-like tags; framing = shared working notes of several analysts. Labels are plain words followed by a colon; the parser can key on them. Evidence for each slot: *Claim/Evidence/Inference* (Liang & Sonntag BioNLP 2025; Meek: list every factor), *Alternative set aside* (ToT; Contrastive CoT), *Principle / analogy from another field* (Step-Back; also the user's goal 1), *Open question* and *Where we stand* (Thought Anchors: planning and uncertainty-management sentences carry the causal weight; Delethink: textual state at chunk end; Jin: "Repeat State", "Self-Verification"), *Source* (FACE-Eval: source-attribution prompts raise verbalization).

System-level text (all three models):

> You are one of several analysts keeping a single shared set of working notes on the question below. The notes are the only record; nothing outside them is kept. Write the next note. A note is 3-6 short labelled items, each one or two sentences, chosen from: `Claim:` (a definite statement), `Evidence:` (a fact, number, source or observation that supports or weakens a claim; name where it comes from), `Inference:` (what follows from which claims, stated as "because A and B, therefore C"), `Principle:` (a general rule from any field that applies here; name the field), `Alternative set aside:` (an answer or route a reasonable analyst would propose, and the concrete reason it is set aside), `Open question:` (what is not yet settled and what would settle it), `Next:` (the single most useful thing the next analyst should do), `Where we stand:` (one sentence: current best answer, confidence, and the weakest assumption). Every note must end with `Where we stand:`. Do not restate earlier items; extend, correct or contradict them, and say which. Do not write a final answer until the moderator asks for it.

Per-segment user message (relay-controlled, varies by state):

> Notes so far: [...]. Write note #k. Items still missing from the record: [auto-filled from the gate, e.g. "no `Alternative set aside:` yet", "no `Evidence:` with a source for Claim 2"]. If the previous note's `Where we stand:` is wrong, say so and why. [Continuation cue if forcing, e.g. "Begin with: Alternatively, "]

FINAL ANSWER request (only after the gate passes):

> The moderator now asks for the answer. Write `FINAL ANSWER:` followed by the answer, then `Basis:` naming the claims and evidence items it rests on, and `Not resolved:` listing the open questions that remain.

Notes on the template: (i) "Do not restate" counters Verbosity Compensation (Zhang et al.) and the quantized-model "filler transition" failure (Lotfi 2026); (ii) requiring `Where we stand:` each segment gives the next model a compact progress state (Markovian Thinker's lesson) and gives the monitor a per-step commitment trace (Ballon's "commitment" curve); (iii) the `Principle:` slot is the evidence-backed way to let notes wander into other fields without inviting free association (Step-Back) - but for Fable specifically Gema et al. found longer reasoning makes Claude more distractible by irrelevant material, so keep `Principle:` to one item per note and require it to be *applied*, not merely mentioned.

### 4.4 Segment length

The ~128-token segment is at the "tight token limit" end where verbalization is suppressed (Zaman & Srivastava) and where a labelled item is likely cut mid-sentence (the items that matter are whole planning/uncertainty sentences - Thought Anchors). Each extra handoff also costs drift (Khraishi). Recommendation: **192 tokens default (256 for GPT-6.2 Sol at verbosity high), trimmed back to the last sentence boundary**, which fits 3-5 labelled items; keep the routing decision per segment. If the 128 cap is kept for routing granularity, require exactly one or two labelled items per segment and let `Where we stand:` appear every second segment instead of every one.

### 4.5 GPT-6.2 Sol: `verbosity`

- Set `text.verbosity = "high"` **for note segments only** (OpenAI: "scales both the length and depth of the model's output while preserving correctness and reasoning quality"; "great for audits, teaching, or hand-offs" - the relay is a hand-off). Under the per-segment cap, expect denser items rather than more items; raise Sol's segment cap to 256 if items get truncated.
- Set `verbosity = "low"` (or `"medium"`) for the FINAL ANSWER segment, where Verbosity Compensation hurts (Zhang et al.: verbose answers 27.61% worse on Qasper).
- Use `seed` to A/B the template and the cue set on fixed questions; it is the only reproducibility handle available on these three endpoints.
- Do not expect `verbosity` to add intermediate *steps*: OpenAI's own guidance is that "the model's reasoning approach stays mostly the same" and that the knob "can either improve or diminish answer quality". Steps come from the template (4.3), not the knob.
- Do not send `verbosity` to Fable (OpenRouter maps it to effort); Gemini has no such parameter - rely on the template and the cap.

### 4.6 Instrumentation worth adding (cheap, evidence-motivated)

- Log, per chain: number of forced continuations, segment index of first FINAL ANSWER attempt, label coverage, and answer stability across the last three `Where we stand:` lines. Answer flips after the gate passed are the Caldarella/Markovian failure mode; a rising count of forced continuations with no new labels is the s1 loop.
- Keep a parallel-sample control: for a subset of questions run k = 3 independent short chains and majority-vote (Ghosal). If it beats the long single chain, the relay's extra length is buying variance, not quality.

---

## 5. Honest answer: does more verbalization improve answer quality, or mainly transparency?

**Mainly transparency (and, for this user's purpose, breadth), not accuracy.**

- For *accuracy*, the strongest recent evidence (Wu, ICLR 2026; Ghosal, NeurIPS 2025; Gema, TMLR 2025; Caldarella 2026; Markovian App. G; ICLR-2026 blog) says that for already-capable models, pushing visible length beyond the model's natural stopping point is neutral at best and negative beyond ~2-4k tokens or ~3 forced continuations. The positive length results are from weaker models (Jin: GPT-3.5 +12-13 while GPT-4 only "tolerated" it) or from fine-tuned systems (s1-32B, +7 AIME24 with a 6x ceiling). Chain of Draft shows frontier models keep ~96% of CoT accuracy with 7.6% of the tokens: the *content* of steps matters a little, the *length* almost not at all. Expect the structured template (plans, explicit alternatives, principles, state lines) to be roughly accuracy-neutral to mildly positive on synthesis/planning questions (PS: +3-6 zero-shot; Step-Back: +7-27 on knowledge-heavy QA; ToT: large gains only on search-like puzzles) and to be mildly negative if it forces domain-wandering on Fable (Gema's distractibility finding).
- For *transparency*, the evidence is consistently positive: longer visible chains are more monitorable (Guan et al. 2025), larger budgets raise verbalization of real drivers toward 90% (Zaman & Srivastava, ACL 2026), structured factor-listing is literally the "verbosity" component of monitorability (Meek et al.), and follow-up questions improve monitorability (Guan et al.). The ceilings are also clear: the visible channel will omit things the hidden channel had (Young: 55.4%), frontier models are less faithful than small ones (Lanham), and polished narration is not a trace (Borchers 2026). A source-attribution requirement helps; threats of monitoring do not (FACE-Eval).
- Practical implication: treat the minimum-segment gate and the template as *instrumentation and diversity* features and evaluate them as such (monitorability, coverage of alternatives, cross-domain moves), while measuring accuracy separately against a short-chain and a parallel-vote control. Do not promise the user better answers from longer notes; promise more legible and more wide-ranging ones, and guard against the documented downside (answer flips after a correct provisional answer) with the gate-then-stop rule.

---

## 6. Papers cited (verification status)

All entries below were opened on arXiv / ACL Anthology / ICLR-NeurIPS virtual pages / OpenAI docs in this session unless marked otherwise.

| # | Paper | Venue / id | Status |
|---|---|---|---|
| 1 | Muennighoff et al., s1: Simple test-time scaling | arXiv 2501.19393 (2025) | V |
| 2 | Xu et al., Chain of Draft | arXiv 2502.18600 (2025) | V |
| 3 | Wu Y. et al., When More is Less | ICLR 2026, arXiv 2502.07266 | V |
| 4 | Jin et al., Impact of Reasoning Step Length | Findings ACL 2024, arXiv 2401.04925 | V |
| 5 | Sprague et al., To CoT or not to CoT? | ICLR 2025, arXiv 2409.12183 | V |
| 6 | Ghosal et al., Does Thinking More always Help? | NeurIPS 2025, arXiv 2506.04210 | V |
| 7 | Gema et al., Inverse Scaling in Test-Time Compute | TMLR 12/2025, arXiv 2507.14417 | V |
| 8 | Chen X. et al., Overthinking of o1-like LLMs | arXiv 2412.21187 (2024/25) | V |
| 9 | Wang Y. et al., Underthinking of o1-like LLMs | arXiv 2501.18585 (2025) | V |
| 10 | Su et al., Between Underthinking and Overthinking | arXiv 2505.00127 (2025) | V |
| 11 | Caldarella et al., Thinking Past the Answer | arXiv 2606.02835 (2026) | V |
| 12 | Lotfi et al., Quantized Reasoning Models Think They Need to Think Longer | arXiv 2606.00206 (2026) | V |
| 13 | Chen W.-L. et al., Think Deep, Not Just Long | ICML 2026, arXiv 2602.13517 | V |
| 14 | Shojaee et al., The Illusion of Thinking | NeurIPS 2025, arXiv 2506.06941 | V |
| 15 | Balachandran et al., Inference-Time Scaling for Complex Tasks | arXiv 2504.00294 (2025) | V |
| 16 | Pfau, Merrill, Bowman, Let's Think Dot by Dot | arXiv 2404.15758 (2024) | V (venue UNV) |
| 17 | Goyal et al., Pause tokens | ICLR 2024, arXiv 2310.02226 | V |
| 18 | Wu T. et al., Thinking LLMs | arXiv 2410.10630 (2024) | V |
| 19 | Zhang Y. et al., Verbosity != Veracity | arXiv 2411.07858 (2024) | V |
| 20 | Fox, Ericsson, Best, Verbal reporting meta-analysis | Psychological Bulletin 137(2), 2011 | V (metadata); effect sizes UNV on primary page |
| 21 | Wang C. et al., NoWait | arXiv 2506.08343 (2025) | V |
| 22 | Ringel et al., Learning a Continue-Thinking Token | arXiv 2506.11274 (2025/26) | V (venue UNV) |
| 23 | Taveekitworachai & Pipatanakul, Wait, Do We Need to Wait? | ICLR 2026 Blog Track | V |
| 24 | Aghajohari et al., The Markovian Thinker | arXiv 2510.06557 (2025) | V |
| 25 | Tian X. et al., Think Twice | arXiv 2503.19855 (2025) | V |
| 26 | Yang V. et al., Mid-Think | Findings ACL 2026 | V |
| 27 | Aggarwal & Welleck, L1 | COLM 2025, arXiv 2503.04697 | V |
| 28 | Ballon et al., Probing the Trajectories of Reasoning Traces | NeurIPS 2026 E&D, arXiv 2601.23163 | V |
| 29 | Khraishi et al., Performance Drift from Model Switching | arXiv 2603.03111 (2026) | V |
| 30 | Kojima et al., LLMs are Zero-Shot Reasoners | NeurIPS 2022, arXiv 2205.11916 | V |
| 31 | Wang L. et al., Plan-and-Solve | ACL 2023 (2023.acl-long.147) | V |
| 32 | Zhou D. et al., Least-to-Most | ICLR 2023, arXiv 2205.10625 | V |
| 33 | Yao S. et al., Tree of Thoughts | NeurIPS 2023, arXiv 2305.10601 | V |
| 34 | Chia et al., Contrastive CoT | arXiv 2311.09277 (2023) | V |
| 35 | Zhou P. et al., Self-Discover | arXiv 2402.03620 (2024) | V (venue UNV) |
| 36 | Zheng H.S. et al., Take a Step Back | ICLR 2024, arXiv 2310.06117 | V |
| 37 | Wang Y. & Zhao Y., Metacognitive Prompting | NAACL 2024, arXiv 2308.05342 | V |
| 38 | Liang & Sonntag, Structured prompting for biomedical claim verification | BioNLP 2025 (2025.bionlp-1.14) | V |
| 39 | Zhao J. et al., Self-Explain / Teach-Back | Findings EMNLP 2024, arXiv 2311.06985 | V |
| 40 | Gao H. et al., Self-Explanation Prompting | arXiv 2309.12940 (2023) | V |
| 41 | Radhakrishnan et al., Question Decomposition Improves Faithfulness | arXiv 2307.11768 (2023) | V |
| 42 | Bogdan et al., Thought Anchors | arXiv 2506.19143 (2025) | V |
| 43 | Turpin et al., Unfaithful Explanations in CoT | NeurIPS 2023, arXiv 2305.04388 | V |
| 44 | Lanham et al., Measuring Faithfulness in CoT | arXiv 2307.13702 (2023) | V |
| 45 | Chen Y. et al., Reasoning Models Don't Always Say What They Think | arXiv 2505.05410 (2025) | V |
| 46 | Korbak et al., CoT Monitorability | arXiv 2507.11473 (2025) | V |
| 47 | Baker et al., Monitoring Reasoning Models for Misbehavior | arXiv 2503.11926 (2025) | V |
| 48 | Guan et al., Monitoring Monitorability | arXiv 2512.18311 (2025) | V |
| 49 | Meek et al., Monitorability via Faithfulness and Verbosity | arXiv 2510.27378 (2025) | V |
| 50 | Zaman & Srivastava, Is CoT Really Not Explainability? | ACL 2026, arXiv 2512.23032 | V |
| 51 | Young, Why Models Know But Don't Say | arXiv 2603.26410 (2026) | V |
| 52 | Gema et al., FACE-Eval | arXiv 2608.29464 (2026) | V |
| 53 | Buerger et al., CoT Monitoring Can Fail Even When CoT Is Necessary | arXiv 2610.18105 (2026) | V |
| 54 | Li S. et al., CoT Oversight Requires Separate Evidence | NeurIPS 2026 Position, arXiv 2611.01620 | V |
| 55 | Wang H. et al., MonitorBench | COLM 2026, arXiv 2603.28590 | V |
| 56 | Barez et al., CoT Is Not (Sufficient for) Explainability | NeurIPS 2026 poster | V |
| 57 | Jia, Benton, Easley, Faithfulness as Information Flow | arXiv 2605.24286 (2026) | V |
| 58 | Borchers, Vie, Azevedo, LLMs as Students Who Think Aloud | arXiv 2602.01015 (2026) | V |
| 59 | Wu T. et al., Thinking Intervention | arXiv 2503.24370 (2025) | V (open models only; not applicable to closed APIs) |
| 60 | OpenAI, GPT-5 New Params and Tools (cookbook); Using GPT-5.2 (docs) | developers.openai.com | V |

Not relied on (UNV): the "Reasoning Cache" (arXiv 2602.03773) claim that "appending Wait is insufficient" came from a search snippet; the abstract page says nothing about Wait. Wurgaft et al. (arXiv 2505.23931) was checked and does *not* contain the think-aloud reactivity meta-analysis the snippet attributed to it (that is Fox et al. 2011). s1's venue (ICML 2025 per prior-pass keyword) is not on the arXiv page and is cited as arXiv only.

## 7. Caveats

- Almost all length/forcing numbers come from open-weight 1.5B-32B reasoning models on math benchmarks; the three cutoffmix endpoints are closed frontier models on research/science questions. Direction of effects is well replicated; magnitudes will differ.
- The relay's visible notes are a *summary channel* produced alongside a discarded hidden channel; no paper studies exactly this two-channel-with-handoff setting. Young 2026 and FACE-Eval are the closest.
- Budget forcing in s1 suppresses an end-of-thinking *token*; the relay can only detect a conclusion in text and re-prompt, which is closer to Think Twice than to s1.
- Fox et al. 2011 effect sizes (r approx. -0.03) were taken from secondary summaries; the APA page shows only metadata.
- OpenAI's "preserving correctness and reasoning quality" claim for `verbosity` is vendor documentation, not an independent evaluation.
- No paid API calls were made; none of the mechanisms in Section 4 were tested live here.
