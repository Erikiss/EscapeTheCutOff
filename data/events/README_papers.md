# `papers_events.jsonl` and `papers_meta.jsonl`: AI/ML paper probes (topic `ai_ml`)

231 events dated 2024-01-04 to 2026-11-23, one `cutoffmix.types.Event` per line (`origin: "arxiv_hf"`,
`topic: "ai_ml"`): 210 real papers (category `paper`) and 21 fake-paper controls (category `fake_paper`,
`control: true`). `papers_meta.jsonl` is the side table, one row per event keyed by `event_id` (interp tag,
upvotes, author count, author lists, distractors, control donors).

Every probe asks the same thing: **"Who is the first author of the arXiv paper titled '<title>'?"** A first author
cannot be guessed from the title, is fixed on the day the paper appears on arXiv, and is verifiable from arXiv
metadata, so the probe measures whether a model has *seen* the paper. The dated recall variant appends
"(arXiv, <Month YYYY>)"; the MCQ offers four first authors. The fake titles measure confabulation: naming anyone is
wrong, only an abstention is right.

## Provenance

- Built on 2026-11-25 with `cutoffmix bench build-papers` at its defaults (dense range 2025-10..2026-11 with 12
  papers per month; anchor months 2024-01, 2024-04, ..., 2025-07 (every 3rd month of 2024-01..2025-09) with 6
  papers each; `--interp-share 0.5 --controls-share 0.1 --seed 0 --candidates 300`). Builder:
  `cutoffmix.bench.sources.papers.build_paper_events`. The file was rebuilt the same day (02:17 UTC) after a data
  review: version-1 metadata, stricter author and control rules, hand-corrected interp tags (details below).
- **Selection: Hugging Face daily papers** (`https://huggingface.co/api/daily_papers?date=YYYY-MM-DD`, a public
  listing). For each of the 21 planned months every daily list was fetched (01:16-01:28 UTC, 638 requests one at a
  time, at least 1 s apart, project User-Agent; 635 lists cached, the current UTC day is refused by the API with
  HTTP 400 and is skipped; the builder now defaults to yesterday UTC). Per month the lists were merged, deduplicated
  by arXiv id (highest upvote count kept) and the top 300 by upvotes formed the candidate pool.
- **Ground truth: the arXiv API** (`https://export.arxiv.org/api/query?id_list=...`, batches of 100 ids, at least
  3 s apart). Two fetches per id: the bare id (the latest version, the one a paper is cited under: 63 batches,
  6 083 ids, every id answered) and `<id>v1` (the paper as it appeared on its first day; asked only for the 3 189
  ids whose latest version is not v1: 32 batches, every id answered). The probe's title (whitespace-normalised),
  authors in order and `published` date come from the **v1** record; the primary category and the summary read by
  the interp tag from the latest. A candidate whose title (word tokens) or first author differs between v1 and the
  latest version is dropped (362 and 32 of the 6 083 candidates: a model that saw one version would be asked for
  the other), so every probe names a title and first author that have been the same since the paper appeared.
  Nothing from the Hugging Face listing ends up in a probe. Papers are filed under the month of the arXiv v1 date;
  1 of the 210 differs from the month of Hugging Face's own publication date and 14 were first listed by Hugging
  Face in the following month (`hf_published_at` and `hf_date` in the meta); candidates whose arXiv month is not a planned month
  (typically late-September papers that Hugging Face listed on 1 October, before an anchor gap) are dropped, the
  others are filed under their arXiv month.
- Every raw response is cached (`--cache DIR`, default `$CUTOFFMIX_PAPERS_CACHE` or `~/.cache/cutoffmix/papers`;
  this build: `hf/<date>.json`, `arxiv/<id>.json` (latest), `arxiv/<id>v1.json` (version 1), `arxiv/raw/<batch>.xml`,
  about 130 MB). A rebuild from the warm cache takes seconds and needs no network unless new days or ids are
  requested; the review rebuild needed the network only for the 32 v1 batches (2 minutes at the arXiv pace).
- No LLM was used. Questions, answers and MCQ options are templates filled from arXiv metadata.

**Reproduce.** `cutoffmix bench build-papers --out data/events/papers_events.jsonl --meta data/events/papers_meta.jsonl
--cache DIR` (about 12 minutes from the network, seconds from a warm cache). Hugging Face lists and upvotes change
over time, so a fresh build can pick different papers; the selection itself is deterministic, `--seed` drives the
MCQ order, the distractor draw and the control titles.

## What an event contains

| Field | Real paper (`paper`) | Control (`fake_paper`) |
|---|---|---|
| `id` | `paper:<arxiv id>` | `fakepaper:<n>` |
| `date` / `month` | arXiv v1 date | the first donor's arXiv date |
| `subject` | the title (identical in v1 and the latest version) | the spliced title |
| `fact` | "first author N; authors: ... (up to 12, then et al.); arXiv ID (month)" | "control: no such arXiv paper exists; the title splices the subtitles of two real papers" |
| `question_current` | "Who is the first author of the arXiv paper titled '<title>'?" (no date: the `current_d` probe) | same |
| `question_recall` | the same with " (arXiv, <Month YYYY>)" | same |
| `expected_current` / `expected_recall` | the first author's full name (identical in v1 and the latest version) | `""` (no right answer; the judges read the `fact`) |
| `mcq_*` | 4 options: the answer plus 3 first authors of other papers of the same or adjacent months, surnames pairwise distinct, order shuffled per seed; letters A 47, B 45, C 58, D 60 | none |
| `aliases` | full name, "Surname, Given", "G. Surname", "Surname G.", "Surname Given", ("First Surname" with a middle name), diacritic-free forms | `[]` |
| `reject_aliases` | the three distractors (naming one grades `incorrect`) | `[]` |
| `source` | `https://arxiv.org/abs/<id>` | `""` |
| `predictability` | `low` | `low` |

Filters (see `cutoffmix.bench.sources.papers._unusable`): first authors that are not a person's name the probe can
ask for are skipped (153 in the pool: "Team"/"Lab"/"AI"/"Robotics"/"Seed" first authors such as `Qwen Team`,
`DeepSeek-AI`, `Kimi Team`, `Tencent Robotics X`, `Bytedance Seed`; surnames of one letter; given names that are
only initials such as `B. An` or `I. Apanasevich`, which a rule grader cannot match against a written-out name),
titles or first authors that changed after v1 (362 and 32), titles with year or month tokens (they would date
`question_current`), titles containing the author's name, ids arXiv does not know. Preferred (ranked first) are
papers with at least 3 authors and 6+ title words: 207 of 210 have that shape; 1 paper has two authors, 2 titles
have 5 words. Every event passes `cutoffmix.bench.probegen.validate_event`.

Meta row (`papers_meta.jsonl`): `event_id, month, control, arxiv_id, interp, upvotes, n_authors, first_author,
authors, hf_authors, primary_category, published, hf_date, hf_published_at, title_words, preferred, distractors,
latest_version` (the arXiv version the paper is at today: 100 at v1, 66 v2, 26 v3, 13 v4, 3 v5, 2 v6); controls
carry `donors` (the two arXiv ids) and `donor_titles` instead.

## Counts

| month | papers | interp-tagged | preferred shape | with MCQ | controls |
|---|---:|---:|---:|---:|---:|
| 2024-01 | 6 | 1 | 6 | 6 | 1 |
| 2024-04 | 6 | 2 | 5 | 6 | 1 |
| 2024-07 | 6 | 2 | 6 | 6 | 1 |
| 2024-10 | 6 | 3 | 6 | 6 | 1 |
| 2025-01 | 6 | 3 | 6 | 6 | 1 |
| 2025-04 | 6 | 2 | 6 | 6 | 1 |
| 2025-07 | 6 | 1 | 6 | 6 | 1 |
| 2025-10 | 12 | 0 | 12 | 12 | 1 |
| 2025-11 | 12 | 1 | 12 | 12 | 1 |
| 2025-12 | 12 | 1 | 12 | 12 | 1 |
| 2026-01 | 12 | 3 | 12 | 12 | 1 |
| 2026-02 | 12 | 3 | 12 | 12 | 1 |
| 2026-03 | 12 | 1 | 12 | 12 | 1 |
| 2026-04 | 12 | 1 | 11 | 12 | 1 |
| 2026-05 | 12 | 1 | 12 | 12 | 1 |
| 2026-06 | 12 | 3 | 11 | 12 | 1 |
| 2026-07 | 12 | 1 | 12 | 12 | 1 |
| 2026-08 | 12 | 0 | 12 | 12 | 1 |
| 2026-09 | 12 | 2 | 12 | 12 | 1 |
| 2026-10 | 12 | 2 | 12 | 12 | 1 |
| 2026-11 | 12 | 1 | 12 | 12 | 1 |
| **all** | 210 | 34 | 207 | 210 | 21 |

2026-11 covers the lists up to 2026-11-24. Primary categories: cs.CV 66, cs.CL 54, cs.AI 42, cs.LG 24, cs.RO 7,
cs.SE 6, others 11. Authors per paper: 2 to 215, median 11. Upvotes: 7 to 745, median 124; the non-interp picks are
the top of each month (floor 71 upvotes), the interp picks reach down to 7 upvotes because the pool is deep (below).
MCQ answer letters: A 50, B 48, C 53, D 59.

### The interp subset (34 papers, keyword tag `interp` in the meta)

`is_interp` is a keyword tag over title and abstract: specific terms anywhere (sparse autoencoder / SAE with an
autoencoder context word, activation patching, logit lens, monosemantic, representation engineering, transcoder,
crosscoder, causal tracing, steering vector, feature attribution, dictionary learning, ...), the generic words
interpretab / mechanistic / superposition in the title or three times in the abstract, or two weak families
(circuit, attribution, probing, steering, neuron, residual stream, attention head; feature/activation only next to
one of those) with a specific one in the title. A title that names an applied domain (medical, clinical, driving,
MEG/EEG/fMRI/brain, compression, pruning, quantisation) is tagged only by a specific term in the title itself:
"Interpretable MEG Decoding" is not interpretability research, "Sparse Autoencoders for Brain Decoding" is. On top
of that, `INTERP_FALSE_POSITIVES` in the source is a hand-checked list of nine ids of the first build whose
abstracts mention a technique only as prior work or as a tool (token-level data filtering, emergent misalignment,
balanced thinking, GradCuit, ...); the tag is removed for them. The requested share of 50 % could not be met: the
Hugging Face daily lists carry only 0-3 such papers per month even in a 300-deep pool, so most months fill up with
top-upvoted papers (the first build tagged 46 with the looser rules; 9 of those were false positives, 4 more were
dropped for a title change after v1, 1 was added).

| month | arXiv id | upvotes | title |
|---|---|---:|---|
| 2024-01 | 2401.10831 | 9 | Understanding Video Transformers via Universal Concept Discovery |
| 2024-04 | 2404.08636 | 14 | Probing the 3D Awareness of Visual Foundation Models |
| 2024-04 | 2404.14394 | 21 | A Multimodal Automated Interpretability Agent |
| 2024-07 | 2407.01100 | 7 | Eliminating Position Bias of Language Models: A Mechanistic Approach |
| 2024-07 | 2407.14435 | 7 | Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders |
| 2024-10 | 2410.07656 | 16 | Mechanistic Permutability: Match Features Across Layers |
| 2024-10 | 2410.12409 | 26 | Revealing the Barriers of Language Agents in Planning |
| 2024-10 | 2410.15999 | 18 | Steering Knowledge Selection Behaviours in LLMs via SAE-Based Representation Engineering |
| 2025-01 | 2501.06751 | 30 | Padding Tone: A Mechanistic Analysis of Padding Tokens in T2I Models |
| 2025-01 | 2501.08319 | 9 | Enhancing Automated Interpretability with Output-Centric Feature Descriptions |
| 2025-01 | 2501.18009 | 23 | Large Language Models Think Too Fast To Explore Effectively |
| 2025-04 | 2504.02821 | 10 | Sparse Autoencoders Learn Monosemantic Features in Vision-Language Models |
| 2025-04 | 2504.15133 | 25 | EasyEdit2: An Easy-to-use Steering Framework for Editing Large Language Models |
| 2025-07 | 2507.12990 | 12 | Teach Old SAEs New Domain Tricks with Boosting |
| 2025-11 | 2511.15210 | 91 | Unveiling Intrinsic Dimension of Texts: from Academic Abstract to Creative Story |
| 2025-12 | 2512.23988 | 18 | Fantastic Reasoning Behaviors and Where to Find Them: Unsupervised Discovery of the Reasoning Process |
| 2026-01 | 2601.10825 | 14 | Reasoning Models Generate Societies of Thought |
| 2026-01 | 2601.14004 | 33 | Locate, Steer, and Improve: A Practical Survey of Actionable Mechanistic Interpretability in Large Language Models |
| 2026-01 | 2602.00462 | 20 | LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs |
| 2026-02 | 2602.02343 | 13 | Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics |
| 2026-02 | 2602.05027 | 62 | AudioSAE: Towards Understanding of Audio-Processing Models with Sparse AutoEncoders |
| 2026-02 | 2602.14111 | 57 | Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines? |
| 2026-03 | 2603.19028 | 17 | SEM: Sparse Embedding Modulation for Post-Hoc Debiasing of Vision-Language Models |
| 2026-04 | 2604.01202 | 31 | Therefore I am. I Think |
| 2026-05 | 2605.30076 | 26 | UniSteer: Text-Guided Flow Matching in Activation Space for Versatile LLM Steering |
| 2026-06 | 2606.06735 | 26 | A Geometric Account of Activation Steering through Angle-Norm Decomposition |
| 2026-06 | 2606.13106 | 23 | Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning |
| 2026-06 | 2606.20097 | 17 | HydraHead: From Head-Level Functional Heterogeneity to Specialized Attention Hybridization |
| 2026-07 | 2607.08393 | 13 | Towards Mechanistically Understanding Why Memorized Knowledge Fails to Generalize in Large Language Model Finetuning |
| 2026-09 | 2609.09113 | 20 | SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research? |
| 2026-09 | 2609.29845 | 89 | Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs |
| 2026-10 | 2610.09087 | 44 | U-Space: Uncovering When and Why Uncertainty Arises in Language Models |
| 2026-10 | 2610.40636 | 20 | Probing Spatial State Representations in Visual Foundation Models |
| 2026-11 | 2611.26627 | 22 | RL Post-Training in Diffusion Transformers Is Activation Steering |

The tag is still a keyword heuristic. About 28 of the 34 are interpretability or representation-analysis papers in
the narrow sense (SAEs, steering, mechanistic analyses, probing studies, surveys of the field); borderline are
2410.12409 (agent planning, feature attribution as a tool), 2511.15210 (intrinsic dimension, SAE as a tool),
2606.13106 (latent reasoning, "mechanistic" in the abstract), 2606.20097 (attention-head specialisation),
2610.09087 (uncertainty, mechanistic analysis) and 2601.10825 (societies of thought). The `interp` flag lives in
the meta; to correct one, add the id to `INTERP_FALSE_POSITIVES` and rebuild from the cache (seconds), or edit the
meta row by hand.

### Controls (21 fake papers)

Each title is the first half of one real title's *subtitle* (the part after the colon; a title without a colon is
used whole) joined with the second half of another's from a different month. The "Name:" head of a real paper is
never used, so no control opens with the distinctive prefix of a real paper (the first build's "SWE-Game: ...",
"VideoChat3: ..." controls made a model that knew the donor name the donor's author, which the control semantics
graded as a confabulation). Both halves have at least 3 words; the seam is not on punctuation or a function word
(the split moves one word when it would be, so no "Boosting mathematical capability of | Large Solve ..."); the
result equals or nearly equals (similarity >= 0.9) no real title among the planned months' candidates and neither
donor's subtitle; donors are never papers probed in this file. One per planned month, dated to the first donor's
arXiv date; the donors are in the meta.

| id | month | fake title |
|---|---|---|
| fakepaper:1 | 2024-01 | Efficient Temporal Adaptation Modeling through Language-Conditioned Video Generation |
| fakepaper:2 | 2024-04 | A Large Language Model Enhanced Rule-based Refinement for Perfect Local Details |
| fakepaper:3 | 2024-07 | Cross-Morphology Motion Alignment Quantum-Augmented Diffusion Language Models |
| fakepaper:4 | 2024-10 | Harnessing Webpage UIs Step in Agent-Based Program Synthesis from Scratch |
| fakepaper:5 | 2025-01 | A Preliminary Exploration Dialogue Agents via Self-Talk |
| fakepaper:6 | 2025-04 | Evolving Self-Play Critic via Adversarial Generation with Masked Discrete Diffusion |
| fakepaper:7 | 2025-07 | Autonomous High-Quality Image Models with Natural Language Actions |
| fakepaper:8 | 2025-10 | Direct Semantic Communication Answering with Multi-Granularity Answers |
| fakepaper:9 | 2025-11 | Reinforcement Learning with Evolving Datasets for Vision-Language Models |
| fakepaper:10 | 2025-12 | Can LLMs Predict Their Own Generation with Long Context |
| fakepaper:11 | 2026-01 | Evaluating and Advancing Process Code Evolution in Dynamic Environments |
| fakepaper:12 | 2026-02 | Privileged Information Distillation LLMs with Large-Scale Synthetic Personas |
| fakepaper:13 | 2026-03 | Versatile Editing of Video World Model in Minecraft |
| fakepaper:14 | 2026-04 | Context-Aware Hybrid Attention Dataset for Robust Chart Understanding |
| fakepaper:15 | 2026-05 | Group-based RLVR as Target-Projection Preparation and Evaluation for Generative AI |
| fakepaper:16 | 2026-06 | Correcting Flow Matching SLAM in Dynamic Environments |
| fakepaper:17 | 2026-07 | Infinite Interactive World Rollout Model that Reasons, Aligns, and Acts |
| fakepaper:18 | 2026-08 | Agentifying the Evaluation Model for Complex Meteorological Systems |
| fakepaper:19 | 2026-09 | Building a Self-Hosted LLM That Covers Control for Multi-Event Video Generation |
| fakepaper:20 | 2026-10 | LLMs Keep Thinking Real-World Simulation at Scale |
| fakepaper:21 | 2026-11 | Training Vision Models Self-Improving LLMs from Raw Corpora |

The splices are noun-phrase pile-ups of the kind arXiv is full of; a few read oddly ("A Preliminary Exploration
Dialogue Agents via Self-Talk"), none is a real paper's title and none starts with a real system name. They test
whether a model invents an author for a title it cannot have seen. The grader counts a named person as `incorrect`
and an abstention as `correct`; a response that disclaims the paper and then names the author of a similar real
title ("I cannot find this paper; you may mean '...' by X") goes to the judges. The MCQ is not run for controls
(a four-way question without a right answer is not representable).

### Five examples

| id | date | question (shortened) | expected | MCQ (answer in bold) | meta |
|---|---|---|---|---|---|
| `paper:2602.02343` (interp) | 2026-02-02 | ... titled 'Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics'? | Ziwen Xu | **A) Ziwen Xu**, B) Shiting Huang, C) Shiva Krishna Reddy Malay, D) Yiqi Lin | interp, 13 upvotes, 12 authors, cs.CL, latest v4 |
| `fakepaper:20` (control) | 2026-10-08 | ... titled 'LLMs Keep Thinking Real-World Simulation at Scale'? | `""` (no such paper) | none | donors 2610.11765, 2511.23369 |
| `paper:2401.02385` (anchor) | 2024-01-04 | ... titled 'TinyLlama: An Open-Source Small Language Model'? | Peiyuan Zhang | **A) Peiyuan Zhang**, B) Zizhang Li, C) Boyuan Chen, D) Xiaoran Fan | 92 upvotes, 4 authors, cs.CL, latest v2 |
| `paper:2510.04618` (dense) | 2025-10-06 | ... titled 'Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models'? | Qizheng Zhang | A) Zihao Wang, B) Harold Haodong Chen, C) Wei He, **D) Qizheng Zhang** | 135 upvotes, 13 authors, cs.LG, latest v3 |
| `paper:2611.29897` (newest) | 2026-11-23 | ... titled 'Motion Mamba 2.0: Autoregressive Diffusion Mamba for Long-Horizon Humanoid Control'? | Yue Huang | A) Ruopeng Gao, B) Yukyeong Lee, C) Quanyu Long, **D) Yue Huang** | 171 upvotes, 8 authors, cs.RO, latest v1 |

Aliases of the first example: `Ziwen Xu`, `Xu, Ziwen`, `Z. Xu`, `Xu Z.`, `Xu Ziwen`; its `reject_aliases` are the
three distractors. The recall question is the same text with " (arXiv, February 2026)" before the question mark.

## Grading

Rule grading for both categories is `cutoffmix.bench.grading.grade_paper` (docs/DESIGN.md §3.5.1): abstain phrases
first (English and German, including "I am not aware of", "I don't have information", "cannot verify", "no record",
"does not exist", "there is no paper", "no idea", "unknown", "Ich kann ... nicht nennen"); `correct` when the
expected name is asserted in a full form or by an uncommon surname alone (common surnames - four letters or fewer,
or Li/Wang/Zhang/Chen/Liu/Yang/Kim/Lee/Park/Singh/Kumar/Smith and a few more - also need the given name's first
token; 13 of the 210 answers are a Zhang, 12 a Wang, 12 a Yang, 10 a Li). Spaces and hyphens inside a name are free
in both directions ("Jun-sol Kim", "Jun Sol Kim" and "Junsol Kim" are one name; Korean first authors are frequent).
`incorrect` when a distractor or any other capitalised name is asserted instead; capitalised services, venues,
fields and German hedges ("Semantic Scholar", "OpenReview", "Mechanistic Interpretability", "Meines Wissens") are
not names. A hedged assertion keeps its label with `hedged: true`. To the judges go: negated or conditional
mentions, mixed answers, an unknown name next to a phrase that disclaims knowledge of the paper, and a name that
could be the expected one misspelt ("Maciej Piorro" for Maciej Pióro) or written out from initials. Controls: a
named person is `incorrect`, an abstention `correct`, a disclaimer that redirects to another paper's author goes to
the judges. A distractor pick in the MCQ is `incorrect`, not `outdated`. Judges see "(none: ...)" as the outdated
answer for both categories and, for controls, the `fact` as the ground truth.

Surface forms (fixed after the 20-event pilot, `runs/papers_pilot.jsonl`, 111 rows): markdown emphasis around a
name ("The first author is **Sho Maeoki**.") is stripped before name detection, a middle initial is part of a name
("**Bardienus P. Duisterhof**"), an affiliation in parentheses or a quoted title after a name is not, and the
paper's own title is masked before the cue search, so the "No" of "Leave No Context Behind: ..." is not a negation.
The three pilot rows these shapes had left `ungraded` grade `incorrect`, `incorrect` and `correct` after the fix.
`cutoffmix bench grade --events data/events/papers_events.jsonl --in runs/papers_pilot.jsonl --out NEW.jsonl
[--regrade-all]` re-grades a results file offline with the rules into a new file (never in place, usage and latency
kept) and prints the grade counts per model and probe before and after.

## Licence and attribution

- **arXiv metadata** (titles, author names, dates, categories) is used through the arXiv API under the
  [arXiv API terms of use](https://info.arxiv.org/help/api/tou.html): one request at a time, at least 3 s apart,
  a descriptive User-Agent. arXiv asks users of its API to acknowledge it: *Thank you to arXiv for use of its open
  access interoperability.* The events hold titles and author names (factual metadata); abstracts are not
  redistributed (they stay in the local cache and are read only for the keyword tag).
- **Hugging Face daily papers** is a public listing used only to choose which papers to probe; the meta keeps the
  upvote count, the listing day and the author list as shown there (`hf_authors`, which occasionally differs in
  name order from arXiv) for provenance.
- The repository code that built the files is MIT.

## Caveats

- **The interp tag is a heuristic** (see above): hand-checked once for this build (nine false positives of the
  first build are excluded by `INTERP_FALSE_POSITIVES`), about six of the 34 remain borderline. The share of
  interpretability papers on the daily lists is small, so the interp subset has 0-3 papers per month; a cutoff
  curve for it needs the probes of several months pooled, or a dedicated interp source (an arXiv `search_query`
  over interpretability keywords would be the natural extension).
- **Interp picks are obscure.** Because the pool is 300 deep, interp papers down to 7 upvotes are included; a model
  may never have seen a 7-upvote paper regardless of its cutoff, which lowers the pre-cutoff hit rate of the subset.
  The non-interp picks are the month's top (71 upvotes or more).
- **Only version-stable papers.** A paper whose title or first author changed after v1 is excluded (394 of the
  6 083 candidates), so the file under-represents papers that were retitled for a conference version. For the kept
  papers the probe is version-independent; the meta's `latest_version` says how many revisions a paper has had.
- **Controls have no expected answer** (`expected_current == ""`): `validate_event` allows that for control events
  and the judges read the `fact`. The rule grader needs no ground truth for them.
- **Same names, different people.** `Jian Yang` is the first author of 3 papers, 11 other names appear twice; these
  are distinct papers and in several cases distinct people. Distractors never share the answer's surname and are
  never co-authors of the paper.
- **Author strings are arXiv's.** Spelling, diacritics and name order follow the arXiv submission; the aliases
  cover "Surname, Given", initials, the reversed order and diacritic-free forms, not transliteration variants.
- **The last weeks are provisional.** Lists are complete up to 2026-11-24; upvotes on recent papers keep changing,
  so a rebuild can swap the newest month's picks. Events less than 14 days old should be rebuilt before use.
- **Not yet probed.** No model has been run on this file; the predictability labels are all `low` by construction
  and the "hard to guess" assumption (first authors are essentially unguessable from titles) has not been checked
  against a baseline. Run the pilot before fitting the index.
- **Cost.** `current_d` + `mcq` for the three configured models is 1 323 calls (441 per model: 231 `current_d`,
  210 `mcq`). With prompts of about 110 tokens and the hidden reasoning measured on the live fixtures (Fable
  176/280/396, Neon 315/830/1 446, Sol 17/30/46 completion tokens per call in a low/mid/high scenario), the dry run
  projects 5.9 / 11.2 / 17.4 USD for the whole file and 0.49 / 0.94 / 1.46 USD for a 20-event pilot (`--limit 20`:
  111 calls, 17 papers and 3 controls). Judges are not included (about 0.3 USD at 15 % of the `current_d` rows).
  `--max-tokens 1500` caps the worst case at 43.6 USD (118.6 USD at the 4 096 default). Reproduce:
  `cutoffmix bench run --events data/events/papers_events.jsonl --out runs/papers.jsonl --models claude-fable-5.2
  gpt-6.2-sol gemini-4-neon --probes current_d mcq --dry-run --in-tokens 110 --out-tokens claude-fable-5.2=280
  gemini-4-neon=830 gpt-6.2-sol=30 --max-tokens 1500 [--limit 20]`; after the pilot, read
  `usage.completion_tokens` per model from the result rows and re-estimate with `--out-tokens-measured`.
