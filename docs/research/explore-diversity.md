# Question B: Temperature-like diversity and cross-domain wandering without temperature

Scope: how to get diversity and deliberate cross-domain excursions in the cutoffmix relay from models that do not
accept `temperature` (Fable 5.2, GPT-6.2 Sol), using only prompting and client-side sampling, plus a concrete,
cheap "domain lens" mechanism. Every paper below was opened on its arXiv / ACL / OpenReview / CVF / journal page
unless marked UNVERIFIED. Running notes: `notes-B-running.md` in the same folder.

## Kurzfassung (Deutsch)

- Hohe Temperatur allein bringt laut Literatur keine Wanderung über Themengebiete. Sie macht den Text vor allem
  unzusammenhängender: Neuheit hängt nur schwach mit der Temperatur zusammen, Inkohärenz mäßig (Peeperkorn et al.,
  ICCC 2024). Auch bei T = 2.0 mit min-p sind 61 % der Antwortpaare desselben Modells noch > 0.8 ähnlich (Artificial
  Hivemind, NeurIPS 2025). Verschiedene Modellfamilien ähneln sich ebenfalls stark (mittlere Ähnlichkeit 0.71–0.82).
  Der Drei-Modell-Mix ist für sich also keine starke Quelle für Vielfalt.
- Gezielte Analogie-Anweisungen wirken stark. Bei Shen, Druckmann und Zou (NeurIPS 2026) steigt der Anteil neuartiger
  Lösungen bei Claude von 1.6 % auf 50.4 %, die Diversität um 90–173 %. Der Preis: Menschliche Gutachter halten dann
  nur 67 % statt 86 % der Lösungen für vernünftig. Eine Prüfung der Analogie ist also Pflicht.
- Vorschlag „Domain Lens“: Der Client wählt per Seed ein fernes Themengebiet und drei Konzepte daraus. Der nächste
  Abschnitt stellt eine funktionale Zuordnung her und kehrt zur Hauptlinie zurück. Der folgende Abschnitt, möglichst
  von einem anderen Modell, prüft die Analogie: hält sie, wo bricht sie, behalten oder verwerfen. Eine Linse kostet
  einen Zusatzabschnitt (≈ 4.7–9.8 s, 0.4–2.2 ct). Bei p = 0.15 sind das im Mittel ≈ 0.7–1.5 s pro Abschnitt.
- Der Zufall gehört in den Client. Instruktionsmodelle würfeln schlecht: Ein Modell nennt bei der Bitte um einen
  zufälligen Vornamen „Avery“ 40-mal zu oft. LLM-Analogien kommen außerdem aus wenigen Domänen: Bei 10 erbetenen
  Analogien entstehen im Mittel ≈ 2 semantische Cluster.

## 1. Constraints recap (from the task, not re-verified here)

- Only gemini-4-neon accepts temperature/top_p. Fable and Sol accept neither. Sol has `verbosity` and `seed`.
- No logprobs and no prefill on Fable. Fable prompts must avoid the words thinking/reasoning/scratchpad/chain of
  thought and XML-like tags. Notes are framed as "shared working notes" of several analysts.
- Hidden reasoning is mandatory and discarded. **Any cross-domain step must therefore be written in the visible
  notes.** A lens that happens only inside a model's hidden reasoning is lost to the next writer and cannot be checked.
- Project facts (README, live run 24.11.2026): per segment, Sol takes ≈ 4.7 s (TTFT 1.1 s), Fable ≈ 9.8 s
  (TTFT 6.3 s) and Neon ≈ 9.2 s (TTFT 7.2 s). Cost per segment is 0.4–0.8 ct (Sol), 1.7–2.2 ct (Fable) and 0.9–1.6 ct
  (Neon). The prior index routes mostly to Sol. The topic taxonomy has 15 **news-style** topics: conflict_security,
  international_relations, government, elections, law_crime, accidents, natural_disasters_climate, health_medicine,
  business_economy, ai_ml, computing_software, space_astronomy, science_research, sports, arts_entertainment.

## 2. What the literature says about temperature and why aligned models collapse

| Finding | Source (verified) |
|---|---|
| Temperature is only **weakly** correlated with novelty and **moderately** with incoherence. It shows no relation to cohesion or typicality. "More nuanced and weak" than the "creativity parameter" folklore. | Peeperkorn, Kouwenhoven, Brown, Jordanous, *Is Temperature the Creativity Parameter of LLMs?*, ICCC 2024, arXiv:2405.00492 |
| Same model, top-p 0.9 and T = 1.0: average pairwise similarity > 0.8 for 79 % of queries. With min-p 0.1 and **T = 2.0**, 61.2 % of pairs are still > 0.8. Across models, average pairwise similarity is 0.71–0.82 (DeepSeek-V3 vs GPT-4o 0.81). 70+ LMs tested, 26K open-ended queries. | Jiang L. et al. (Choi), *Artificial Hivemind*, NeurIPS 2025 D&B (oral), arXiv:2510.22954 |
| RLHF significantly reduces output diversity compared with SFT. | Kirk et al., ICLR 2024, arXiv:2310.06452 |
| Aligned Llama-2 models show lower token entropy, embedding clusters and "attractor states". | Mohammadi, *Creativity Has Left the Chat*, arXiv:2406.05587 |
| Aligned models are biased in random tasks ("7"), predictable in mixed-strategy games, and prefer pleasant prose over originality. Benchmark strength anti-correlates with these abilities. | West & Potts, *Base Models Beat Aligned Models at Randomness and Creativity*, arXiv:2505.00047 |
| 20 models produce less diversity than humans. **Larger models in a family are often less diverse.** | Zhang Y. et al., *NoveltyBench*, arXiv:2504.05228 |
| RLVR improves pass@1, but base models win at large k. RL narrows the explored space. | Yue et al., NeurIPS 2025 oral, arXiv:2504.13837 |
| Mode collapse is partly caused by typicality bias in preference data. Prompting for a verbalized distribution recovers diversity (see VS below). | Zhang J. et al., *Verbalized Sampling*, arXiv:2510.01171 |
| Asked for a uniform 1–10, Llama-2-13B-chat favours 5. Asked for a random first name, Mistral-7B-Instruct picks "Avery" 40× too often. | Zhang Y., Schwarzschild, Carlini, Kolter, Ippolito, *Forcing Diffuse Distributions out of LMs*, arXiv:2404.10859 |

Implications for cutoffmix:
1. Even on Neon, a temperature schedule will add noise, not cross-domain structure. Use high T on Neon only for
   short, checked exploratory segments (lens or move menu), never for the main line.
2. The three-model mix is not a strong diversity source in itself, given the inter-model homogeneity. In practice the
   prior index makes Sol write most segments anyway. It does help **accuracy** when the models disagree (ReConcile,
   §3.8).
3. Do not ask a model to "pick something random". The client draws the randomness with a seeded RNG and passes the
   result in the prompt. This is the String-Seed-of-Thought principle (§3.6), with the client generating the seed.

## 3. Technique catalogue: evidence, cost per use in the relay, fit to 128–300-token segments

Cost basis: one extra relay segment adds ≈ 4.7 s and 0.4–0.8 ct on Sol, or ≈ 9.2–9.8 s and 0.9–2.2 ct on
Neon/Fable (README measurement). Token counts are my estimates from typical template lengths, not measurements.

### 3.1 Verbalized Sampling (VS)
- **Paper:** Zhang J., Yu, Chong, Sicilia, Tomz, Manning, Shi, *Verbalized Sampling: How to Mitigate Mode Collapse and
  Unlock LLM Diversity*, arXiv:2510.01171 (Oct 2025, v4 Jul 2026). arXiv only; no venue shown on the page.
- **Finding:** Asking for k candidates with probabilities raises creative-writing diversity 1.6–2.1× over direct
  prompting. Human diversity ratings (1–4) improve: poem 1.90 → 2.39, story 2.74 → 3.06, joke 1.83 → 3.01. The authors
  report no loss of factuality or safety. Diversity is **tunable by the probability threshold in the prompt**: a lower
  threshold gives more diversity. Gains grow with model capability. Tested on GPT-4.1, Gemini-2.5-Pro, Claude-4-Sonnet,
  o3, DeepSeek-R1 and others. Uses k = 5 per call (k = 20 for open QA). The official repo's system prompt asks for five
  responses "each within a separate `<response>` tag … sample at random from the tails of the distribution, such that
  the probability of each response is less than 0.10".
- **Relay use:** a "next-move menu" side call before selected segments. The model lists 5 possible next moves with
  probabilities. The client picks one with a seeded RNG, either from the tail (p < 0.10) or weighted, and injects it as
  "Suggested next move: …".
- **Cost:** +1 call per use with ~150–250 output tokens, about +4.7 s on Sol or about +9 s on Neon. On Neon a higher T
  is allowed here. **Fit:** does not fit inside a 128-token segment. As a separate 300-token side call it fits.
  **Fable constraint:** replace the XML tags with numbered lines such as `1. (p≈0.07) …`.
- **Verdict:** adopt as an optional "stagnation breaker" (p ≈ 0.1, or triggered by stagnation). It complements the
  lens, which governs the domain, by governing the **move type**.

### 3.2 Analogical prompting
- **Paper:** Yasunaga, Chen X., Li Y., Pasupat, Leskovec, Liang, Chi, Zhou, *Large Language Models as Analogical
  Reasoners*, ICLR 2024, arXiv:2310.01714.
- **Finding:** Self-generated exemplars beat 0-shot CoT. GSM8K with GPT-3.5-turbo goes 75.8 → 77.8 and MATH 33.9 →
  37.3. PaLM2 GSM8K goes 78.2 → 81.7. K = 3–5 exemplars works best. K = 1 is worse (GSM8K 76.1) because of "excessive
  reliance on a single exemplar". The prompt asks for problems "distinct from each other". Self-generated exemplars
  beat retrieved ones only for larger models. In the failure analysis, 8 of 50 wrong answers came from over-reliance on
  exemplars (misdirection) and 12 of 50 from a generalization gap.
- **Cost:** 3 exemplars add +300–600 output tokens. **Fit:** no for 128-token segments. Compressing to K = 1 is the
  worst setting.
- **Verdict:** do not use as-is. The useful parts are "distinct" exemplars and the warning about over-reliance, which
  motivates the check segment.

### 3.3 Thought Propagation (TP)
- **Paper:** Yu J., He, Ying, *Thought Propagation: An Analogical Approach to Complex Reasoning with LLMs*, ICLR 2024,
  arXiv:2310.03965.
- **Finding:** The model proposes and solves analogous problems, then aggregates them. It reports +12 % absolute
  optimal shortest paths, +13 % human preference in creative writing and +15 % agent task completion.
- **Cost:** k analogous problems, each solved, plus an aggregation step: ≥ k + 1 extra calls and thousands of tokens,
  roughly +30–90 s. **Fit:** no.
- **Verdict:** reject for the relay. It could be an offline "deep analogy" side-branch at most.

### 3.4 Diversity of thought (DIV-SE / IDIV-SE), persona and approach rotation
- **Paper:** Naik, Chandrasekaran, Yuksekgonul, Palangi, Nushi, *Diversity of Thought Improves Reasoning Abilities of
  LLMs*, arXiv:2310.07088 (2023/2024). No venue on the arXiv page.
- **Finding (paper tables):** GPT-4 zero-shot on Blocksworld 4/5: CoT 40.0, SC-10 41.2, **DIV-SE 69.6**, IDIV-SE 57.0.
  Graph colouring: CoT 15, SC-10 23, **DIV-SE 97**, IDIV-SE 74. On GPT-4 CommonsenseQA the gain is only +0.1 to +0.3.
  Approaches are elicited from the LLM (≈ 100 queries, keeping the 3–5 most frequent) and paired with personas.
  DIV-SE sends one call per approach. IDIV-SE puts all approaches in one call. The pair is Pareto-optimal on
  accuracy vs cost. Several in-text numbers differ from the tables; I use the table values.
- **Related:** Meincke, Mollick, Terwiesch, *Prompting Diverse Ideas: Increasing AI Idea Variance*, arXiv:2402.01727.
  In GPT-4 ideation across 35 prompts, human groups reach cosine similarity 0.243. GPT-4 pools range 0.255–0.432, and
  the multi-step CoT prompt is best at 0.255. A "Steve Jobs" persona barely helps (0.368 vs base 0.377), and HBR
  creativity tips do worse (0.387). The estimated unique-idea space grows from ~3700 to ~4700 with CoT.
- **Related:** Lu L.-C. et al., *LLM Discussion*, COLM 2024, arXiv:2405.06373. With 4 role-played agents over
  5 rounds, LLM-judged originality is AUT 4.44 (single agent 3.47, LLM Debate 3.73) and Scientific Creativity Test
  3.95 (3.18 / 3.52). The role ablation is mixed: on AUT, roles without discussion scored best (4.58).
- **Cost:** a per-model **approach tag** costs +20–40 input tokens and 0 extra output or calls. **Fit:** yes. IDIV-SE
  ("two approaches, briefly") needs about 2× segment length, so it fits only at 300 tokens. Full DIV-SE is a run-level
  ensemble at 3–5× cost.
- **Verdict:** adopt **approach tags** (e.g. "mechanistic model", "order-of-magnitude estimate", "historical
  precedent", "statistical/base-rate view", "engineering constraint view"), seeded per segment or per model. Skip
  celebrity personas, since the measured diversity effect is tiny. DIV-SE-style parallel seeds are possible later at
  the run level.

### 3.5 Random stimulus, associative thinking, Leap-of-Thought
- **Mehrotra, Parab, Gulwani**, *Enhancing Creativity in LLMs through Associative Thinking Strategies*,
  arXiv:2405.06715. Asking GPT-4 to link the task to a **random object or attribute** significantly improves
  originality in product design, storytelling and marketing. The abstract gives no numbers.
- **Zhong S. et al.**, *Let's Think Outside the Box: Exploring Leap-of-Thought in LLMs with Creative Humor Generation*,
  CVPR 2024 (CVF open access, pp. 13246–13257), arXiv:2312.02439. The explorative self-refinement samples a **random
  noun** as a "condition", which is empty with probability ρ. It generates candidates, ranks them, and then
  **fine-tunes** on the result. The paper reports that CoT, CoT-SC and prompt-only LoT did **not** produce leaps on
  their benchmark: "LoT cannot be directly achieved by prompting alone". That is a caution for pure prompting, measured
  on humour (Oogiri), not science.
- **Cost:** +5–20 input tokens. **Fit:** yes. **Verdict:** use the random-concept condition **inside** the domain lens
  by sampling 3 concepts from the chosen domain. Fine-tuning is not available for these closed models.

### 3.6 String Seed of Thought (SSoT)
- **Misaki & Akiba**, *String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation*,
  ICLR 2026, arXiv:2510.21150. The LLM first writes a random string and then derives its answer from it. This
  approaches an ideal PRNG on probabilistic instruction following and increases NoveltyBench diversity.
- **Relay use:** the model-written string would pollute the shared visible notes (+20–50 tokens). The client should
  generate the seed instead: `seed = hash(run_seed, segment_idx)`. **Verdict:** adopt the principle, not the
  in-notes string.

### 3.7 Multi-agent debate (MAD) and "tit-for-tat" challenge segments
- **Liang T. et al.**, *Encouraging Divergent Thinking in LLMs through Multi-Agent Debate*, EMNLP 2024,
  arXiv:2305.19118. Identifies **Degeneration-of-Thought**: once a model is confident, self-reflection stops producing
  new ideas. A **moderate** tit-for-tat level and adaptive stopping work best.
- **Du Y. et al.**, *Improving Factuality and Reasoning in LMs through Multiagent Debate*, arXiv:2305.14325. Multiple
  instances debate over rounds and improve math, strategy and factuality.
- **Choi, Zhu, Li**, *Debate or Vote*, NeurIPS 2025 spotlight, arXiv:2508.17536. Across 7 benchmarks, **majority voting
  explains most MAD gains**. Debate alone is a martingale and does not raise expected correctness.
- **Cost:** a "challenge" segment (state one objection to the current line) costs one segment, ≈ 4.7–9.8 s.
  **Fit:** yes. **Verdict:** low priority as an accuracy tool. Useful only as a DoT breaker when the chain stagnates.

### 3.8 Model diversity (the relay itself)
- **Chen J. C.-Y., Saha, Bansal**, *ReConcile*, ACL 2024, arXiv:2309.13007. A round table of diverse LLMs beats single-
  and multi-agent baselines by up to 11.4 %. Mixing API, open and domain models gives +8 % on MATH. Gains are attributed
  to model diversity.
- **Hegazy**, *Diversity of Thought Elicits Stronger Reasoning in Multi-Agent Debate*, arXiv:2410.12853 (J. Robotics &
  Automation Research, 2024; **low-tier venue, weak evidence**). Diverse mid-size models reach 91 % on GSM8K vs 82 % for
  3× Gemini-Pro.
- **Hivemind** (above): inter-model similarity is 0.71–0.82 on open-ended prompts.
- **Verdict:** model switching helps the **check** step (independent critic), not the **exploration** step.

### 3.9 Knowledge-graph random paths (SciAgents)
- **Ghafarollahi & Buehler**, *SciAgents: Automating scientific discovery through multi-agent intelligent graph
  reasoning*, arXiv:2409.05556. The published version is in *Advanced Materials* (DOI 10.1002/adma.202413523, seen in
  search results; journal page not opened). Two concepts are chosen by the user or **at random**. The path between them
  is a **random path**: a modified Dijkstra with a randomness factor of 0.2 plus random waypoints, with second-hop
  neighbours as context. An Ontologist → Scientist 1 → Scientist 2 → Critic chain turns the path into a hypothesis.
  Example: silk → … → energy-intensive led to "silk + dandelion-based pigments".
- **Buehler**, arXiv:2403.11996: graph reasoning finds isomorphisms between distant domains (biological materials
  and Beethoven's 9th).
- **Relay use:** a "path lens" with 1–2 waypoint topics, run over 2–3 segments. **Fit:** as a multi-segment excursion
  only. **Verdict:** later extension of the lens.

### 3.10 Temperature on Neon
- **Cost:** 0. **Fit:** yes. **Verdict:** a high T (≈ 1.2–1.5, with top_p ≈ 0.95) is reasonable **only** on Neon's
  lens or menu segments, which a different model then checks. Peeperkorn and Hivemind say not to expect domain
  wandering from T alone.

### Summary table (per use; "segment" = one extra relay call)

| Technique | Extra calls | Extra output tokens | Extra latency | Fits 128 / 300? | Verdict |
|---|---|---|---|---|---|
| Domain lens (seeded) + check | +1 segment (the check). The lens replaces a main-line segment. | lens +0–80; check ~80–128 | ≈ 4.7 s (Sol) to 9.8 s (Fable) | 128: tight; 200–300: yes | **Adopt** |
| VS move menu | +1 side call | 150–250 | ≈ 4.7–9.2 s | side call ≤ 300 | Adopt (optional) |
| Approach tag (DIV-SE-lite) | 0 | 0 (+20–40 in) | ~0 | yes | **Adopt** |
| IDIV-SE "two approaches" | 0 | ×2 segment length | +2–5 s | 300 only | Occasional |
| DIV-SE parallel seeds | ×3–5 runs | ×3–5 | ~0 wall (parallel) | n/a | Run-level, later |
| Analogical prompting (K = 3) | 0 | +300–600 | +3–8 s | no | Reject as-is |
| Thought Propagation | +k+1 | 1000+ | +30–90 s | no | Reject |
| Random concept condition (CLoT, associative) | 0 | 0 | 0 | yes | Inside the lens |
| SSoT (model writes random string) | 0 | +20–50 in notes | small | pollutes notes | Client seed instead |
| Challenge / debate segment | +1 segment | ~128 | 4.7–9.8 s | yes | On stagnation only |
| LLM Discussion (4 × 5) | ~20 | thousands | minutes | no | Borrow roles only |
| KG random path | +1–2 segments | 256–384 | 10–20 s | multi-segment | Later |
| Neon high T | 0 | 0 | 0 | yes | Lens/menu only |

## 4. Cross-domain analogies and scientific reasoning / hypothesis generation

| Paper | Verified finding | Lesson for the lens |
|---|---|---|
| Shen A., Druckmann, Zou, *Unlocking LLM Creativity in Science through Analogical Reasoning*, arXiv:2605.11258 (May 2026), NeurIPS 2026 poster (neurips.cc page opened) | Two-step analogical reasoning (AR): (1) extract the problem's objects and relations and map to a cross-domain problem ("map objects by FUNCTION, not surface similarity"); (2) search for a solution the analogy supports. Run on Claude Sonnet 4.5, GPT-5.2 and Gemini 3 Flash at T = 1.0, 2 calls, ≈ $0.02/problem. Solution Vendi +90–173 %, domain Vendi +100–115 %, ≈ 3.5× more unique domains. LLM-judged novelty, no-domain / cross-domain / AR: Claude 1.6 % / 21.6 % / 50.4 %, Gemini 8.6 / 37.9 / 68.8. Humans rate AR more novel than the cross-domain baseline 78 % of the time, **but reasonable only 67 % vs 86 %**. Domain distance ≈ 7 vs 2.3–3.4. Uses 48 non-biomedical × 47 biomedical domains. | Explicit **relational mapping** matters more than just naming a domain: "cross-domain" without an analogy gets less than half of AR's novelty. Expect about 1 in 3 lenses to be unsound, so a check is required. |
| Shen Y. et al. (X. Zhu), *On the Diversity of Analogy Making in LLMs*, arXiv:2608.03233 (Aug 2026) | 10 LLMs, including Claude Sonnet 4.5/4.6, GPT-5.2, Gemini 2.5/3.1 Pro and Grok 4.5. Asked for 10 analogies, models average **≈ 2 semantic clusters** (best ≈ 3.5). Target-domain diversity is ≤ 0.63 for all models. Diversity methods (decoding and prompting) trade quality for diversity, and none dominates. | Do not let the model choose the domain: the **client picks domain and concept candidates**. |
| Qin C. et al. (Joty), *Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?*, arXiv:2404.12728 | GPT-3.5: self-generated **random biology** examples beat relevant ones on math (GSM8K 75.3 vs 71.5; MATH 34.6 vs 33.3). On BBH, relevant examples are better (63.2 vs 61.7). The correctness of the exemplars matters more than their relevance. | A distant excursion need not hurt main-line accuracy, provided the content of the excursion is correct. (Older model, different setting.) |
| Si, Yang, Hashimoto, *Can LLMs Generate Novel Research Ideas?*, ICLR 2025 (iclr.cc poster page), arXiv:2409.04109 | AI ideas are rated more novel than expert ideas (5.64 vs 4.84, p < 0.01) and slightly less feasible. **Of 4000 seed ideas per topic, only ≈ 200 (5 %) are unique** at cosine 0.8. Duplicates plateau. | Science ideation collapses without injected diversity. Use the 0.8-cosine duplicate threshold as a stagnation detector. |
| Wang Q., Downey, Ji, Hope, *SciMON*, ACL 2024, arXiv:2305.14259 | GPT-4 ideas have low technical depth and novelty. Retrieved "inspirations" plus iterative novelty boosting against the literature partly fix this. | Inspirations should be concrete. Add a novelty check. |
| Yang Z. et al., *MOOSE-Chem*, ICLR 2025, arXiv:2410.07076 | Hypothesis = background + retrieved inspirations, then ranking. 51 chemistry papers published after Jan 2024. Inspiration retrieval by LLM is "surprisingly" accurate. The abstract gives no numbers. | Treat the inspiration as a separate, retrievable unit. |
| Ghafarollahi & Buehler, *SciAgents*, arXiv:2409.05556 | Random KG paths connect distant concepts, followed by a dedicated Critic agent (§3.9). | A random path is a good extension, and the critic step is built in. |
| Radensky et al. (Hope, Weld), *Scideator*, arXiv:2409.14634 | Recombines facets (purpose, mechanism, evaluation). Retrieves analogous papers at an **adjustable distance**, from the same topic to unrelated fields. Gives more creativity support than the baseline. | Make distance a user-visible knob. |
| Ding Z., Srinivasan, MacNeil, Chan J., *Fluid Transformers and Creative Analogies*, arXiv:2302.12832 | LLM cross-domain analogies get median helpfulness 4/5. ≈ 80 % led to changes in the problem formulation. ≤ 25 % were potentially harmful, mostly upsetting content. | Analogies mainly **reframe** the problem, so the check should ask "what changes in the formulation". |
| Gu T. et al., *LLMs can Realize Combinatorial Creativity…*, arXiv:2412.14141 | Cross-domain, generalization-level retrieval gives +7–10 % similarity to real later research (OAG-Bench). | Abstract up a level before mapping. |
| Guo H., *LacMaterial*, arXiv:2510.22312 | Cross-domain analogy retrieval for battery materials beats standard prompting. Candidates fall outside known compositional spaces. | Domain evidence (materials). |
| Sternlicht & Hope, *CHIMERA*, arXiv:2505.20779 | A knowledge base of idea recombinations, including cross-domain analogies, mined from papers. A hypothesis model trained on it gives directions researchers rate as inspiring. | Recombination is a real pattern in science. |
| Lewis & Mitchell, *Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in LLMs*, arXiv:2402.08955 | GPT models drop sharply on counterfactual analogy variants, while humans stay high. | LLM analogies are brittle, so check the mapping and not just the plausibility of the prose. |
| Chan J., Dow, Schunn, *Do the best design ideas (really) come from conceptually distant sources of inspiration?*, Design Studies 2015 (doi 10.1016/j.destud.2014.08.001, full PDF opened) | OpenIDEO: 2341 concepts across 12 challenges, 707 citing sources. **Inverse** relation between source distance and shortlisting: +1 unit of maximum distance gives −0.33 log-odds (p < .05), mean distance −0.40. Far sources need more processing, and many yield shallow inferences ("hidden gems" have to be sifted). | Far sources do **not** directly give the best final ideas. Weight moderate distance and filter far ones. |
| Chan J. & Schunn, *The impact of analogies on creative concept generation: lessons from an in vivo study in engineering design*, Cognitive Science (PDF header: 2014 online; text read from the same PDF, which bundles it) | Using a **far analogy** is associated with a higher rate of new-concept generation: odds ratio 1.99 at lag 0 and 1.88 at lag 1. | Far analogies help **divergence**, opening new directions. Pair them with convergence. |
| Kwon & Goucher-Lambert, *Examining the Boundary Between Near and Far Design Stimuli*, Proc. Design Society (ICED23), 2023 | Distinct "too far" boundaries appear in embedding distances, and they are wider for visual than for functional similarity. | A "too far" zone exists, so cap the distance. |
| Fu K. et al., *The meaning of "near" and "far"…*, J. Mechanical Design 135(2), 2013 | **UNVERIFIED** (PDF 404; only search snippets): far stimuli gave lower novelty and quality, with a possible curvilinear optimum at moderate distance. | Not relied on. Consistent with the verified Chan 2015 result. |

## 5. Proposed mechanism: "domain lens"

### 5.1 Flow per segment (client side, before the routing decision)

```
eligible = idx >= 2 and idx < planned_segments - 2 and cooldown == 0 and not pending_check
if pending_check:                       # segment right after a lens
    instruction += CHECK_TEXT(lens)     # forced
    prefer_writer != lens.writer        # soft: add a router penalty for the lens writer (cross-model check)
elif eligible and rng(seed, idx).random() < p_lens(stagnation):
    tier   = rng.choices(["mid","far","near"], weights=[0.55, 0.35, 0.10])
    domain = sample_without_replacement(tier_of(current_topic)[tier])
    concepts = rng.sample(CONCEPTS[domain], 3)
    instruction += LENS_TEXT(domain, concepts)
    pending_check = True; cooldown = 3
```

- **Seeded RNG:** `seed = hash(run_seed, idx)`. This makes runs reproducible, and changing `run_seed` gives
  DIV-SE-like run diversity for free. Sol's `seed` parameter is a separate, extra reproducibility knob.
- **Routing interplay:** the lens segment is about another news topic, e.g. sports. **Exclude the lens and check
  segments from the topic-classifier window**, or down-weight them (×0.3). Otherwise the cutoff router drifts toward
  the lens domain. The lens text also forbids recent news events, so the lens does not create cutoff-sensitive claims.
  An alternative for later: route the lens segment by the lens domain's cutoff.

### 5.2 Prompt texts (Fable-safe: no thinking/reasoning/scratchpad/CoT wording, no XML-like tags)

LENS_TEXT:
> Cross-field lens for your next note: the field is {domain_name}. Pick one specific, well-established concept from
> this field (for example {c1}, {c2} or {c3}) whose structure resembles the current intermediate result. In 2–3 short
> mappings, state what in our problem plays the role of what in that concept. Match by function and relations, not by
> surface features. Then state one concrete consequence or testable question this suggests for the main line. Start
> the note with "Lens ({domain_name}):" and finish with one sentence that returns to the main line. Use textbook-level
> knowledge of the field, not recent news.

CHECK_TEXT:
> The previous note drew a cross-field lens from {domain_name}. Start your note by checking it in 1–3 sentences: which
> mapping holds, where the comparison breaks (name at least one real difference), and whether its suggested
> consequence survives. Write "Lens kept:" or "Lens dropped:" followed by the reason, then continue the main line. If
> kept, use the consequence concretely.

These texts follow the literature: structure mapping by function (Shen 2026 AR); "distinct" candidates and a guard
against over-reliance on the source (Yasunaga 2024); an independent check (CoVe, Findings ACL 2024; Huang et al.,
ICLR 2024 show intrinsic self-correction fails, hence a *different* model); and the reframing focus (Ding 2023).

### 5.3 How often: probability per segment

There is no paper that measures a per-segment injection rate for a reasoning relay. The values below are
**engineering heuristics** anchored in the following evidence:
- Analogies are valuable but noisy. About 1 in 3 is unreasonable (AR 67 % vs 86 %), and far sources need processing
  time and sifting (Chan 2015). Every lens therefore costs about 2 segments (lens plus check), so the rate must stay low
  enough for the main line to dominate.
- Divergence helps most when the chain stagnates. Degeneration-of-Thought (Liang 2024) and idea duplication plateaus
  (Si 2025: 5 % unique) both point there.
- A moderate challenge level beats a high one (MAD "tit-for-tat" result).

Recommended defaults:

| Preset | p_lens (eligible segments) | On stagnation | Cooldown | Max lenses | Share of segments in lens + check |
|---|---|---|---|---|---|
| off | 0 | 0 | – | 0 | 0 % |
| balanced | 0.08 | 0.30 | 4 | ceil(N/8) | ≈ 12–18 % |
| explore (user's wish) | 0.15 | 0.40 | 3 | ceil(N/6) | ≈ 25–30 % |

- **Stagnation detector (cheap):** the last segment has cosine ≥ 0.85 to the mean of the previous 3 segments. Si et
  al. and Meincke both use 0.8 as their duplicate threshold. Without an embedding model, use the proxy "no new noun
  phrases or numbers, and the classifier topic has been unchanged for ≥ 5 segments".
- **Never** place a lens in the first 2 segments (the main line must be set) or in the last 2 (convergence and answer).
  Never place one while a check is pending.
- **Expected cost at "explore" with 20 segments:** ≈ 3 lenses → +3 check segments, ≈ +15–30 s, +1.2–6.6 ct per run
  (README per-segment figures). Amortized, that is ≈ +0.7–1.5 s per segment.

### 5.4 How distant

- Build a 15 × 15 topic distance matrix. A cheap option is to embed each topic's description plus seed sentences and
  take 1 − cosine. Alternatively, hand-code it once.
- For the current classifier topic, split the other 14 into near (closest 4), mid (next 5) and far (5 farthest). From
  science_research, for instance: near = health_medicine, space_astronomy, ai_ml, computing_software; mid =
  natural_disasters_climate, business_economy, accidents, law_crime, …; far = sports, arts_entertainment, elections, …
- Sample the tier as **mid 0.55 / far 0.35 / near 0.10**. Raise far to 0.5 when stagnation is detected. Rationale:
  closer sources win on final quality (Chan 2015, verified); far analogies win on new directions (Chan & Schunn 2015,
  verified); there is a "too far" boundary (Kwon 2023); a curvilinear optimum is suggested but UNVERIFIED (Fu 2013).
  AR's large novelty gain came at high distance (≈ 7/10) with a reasonableness cost.
- Sample domains **without replacement** within a run. This counters the ≈ 2-cluster domain homogeneity (Shen Y. 2026).
- **Concept lists:** the 15 topics are news categories, so give each topic 8–12 *structural* concepts. The client
  samples 3 of them, which is the CLoT / associative random-condition idea. Examples:
  - elections: Arrow's impossibility, gerrymandering, swing voters, turnout effects, runoff vs plurality
  - sports: home advantage, regression to the mean, doping arms race, formation trade-offs, talent scouting base rates
  - conflict_security: attrition, supply lines, deterrence, fog of war, escalation ladder
  - international_relations: balance of power, treaty verification, sanctions leakage, coalition free-riding
  - government: principal-agent, succession crises, checks and balances, policy lag
  - law_crime: burden of proof, precedent, adversarial testing, plea bargaining
  - accidents: Swiss-cheese model, normal accidents, root-cause analysis, near-miss reporting
  - natural_disasters_climate: tipping points, return periods, cascading failures, early-warning signals
  - health_medicine: dose-response, placebo control, triage, immune memory, screening base rates
  - business_economy: arbitrage, network effects, moral hazard, bubbles, comparative advantage
  - ai_ml: overfitting, distribution shift, attention, exploration vs exploitation
  - computing_software: caching, race conditions, technical debt, version control, graceful degradation
  - space_astronomy: orbital resonance, gravitational lensing, Lagrange points, redshift as a clock
  - science_research: replication, falsification, instrument bias, preregistration
  - arts_entertainment: counterpoint, narrative arc, remix and sampling, genre conventions, negative space

### 5.5 Keeping coherence

1. The lens segment must end with a "return to the main line" sentence.
2. A **forced check segment** follows immediately. Prefer a different model than the lens writer by applying a router
   penalty to the lens writer for that segment.
3. In the final answer phase, mention kept lenses only with their concrete consequence. Dropped lenses must not appear
   as claims.
4. Track the **keep rate**. Shen 2026 found ≈ 67 % of AR solutions reasonable, so a keep rate around 40–70 % is
   plausible. Below 25 %, lower the far weight. Above 85 %, the lens is probably too near or the check too lenient.
5. Breadth metrics for evaluating the preset: distinct topics touched per run; embedding diversity (or Vendi score)
   across runs with different `run_seed`; LLM-judged novelty of the final answer vs lens = off; and answer
   correctness on the existing bench, to confirm the main line is not degraded.

### 5.6 Optional add-ons (in priority order)
1. **Approach tag** per segment (DIV-SE-lite): +20–40 input tokens, no extra call.
2. **VS move menu** on stagnation: a Neon side call, optionally at high T, listing five moves with probabilities on
   numbered lines (no tags). The client picks from the tail.
3. **Path lens** (SciAgents-style): 2 waypoint topics over 2–3 segments, followed by a check.
4. **Challenge segment** (MAD) on stagnation: "state the strongest objection to the current line".

## 6. Caveats
- Most evidence is on creative writing, ideation, math or design. Only Shen 2026 (AR) and the SciAgents / SciMON /
  MOOSE-Chem line address science directly, and none tests injection inside a multi-model note relay.
- The per-segment probabilities, tier weights and thresholds are heuristics derived from the evidence, not measured
  optima. Tune them on the bench.
- VS was evaluated on older model generations (GPT-4.1, Claude-4-Sonnet, Gemini-2.5). Its effect on Fable 5.2 or
  Sol 6.2 is untested.
- Relevant-or-Random and Analogical Prompting are GPT-3.5/PaLM-2 era results.
- Several sources are arXiv-only (VS, DIV-SE, Relevant or Random, Mehrotra, Ding, Meincke, Shen Y. 2026, NoveltyBench,
  West & Potts, Lewis & Mitchell, Zhang Y. 2024, SciAgents arXiv). Hegazy appeared in a low-tier journal.
