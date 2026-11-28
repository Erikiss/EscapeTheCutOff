# Lens A: ARC-AGI as the grounding substrate for a weight-space DSL

Research date: 2026-11-25. Every claim below was checked by opening the primary page (arcprize.org, arXiv, GitHub, Hugging Face API/model files, Kaggle-linked repos) or by running code on downloaded files. Anything not checked that way is marked **UNVERIFIED**. "Spike" means a measurement I ran on this machine (4-core Intel Xeon @ 2.3 GHz with AVX-512 and AMX-bf16, 15 GB RAM, no GPU, PyTorch 2.15.0+cpu). The scripts are in `/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/scripts/`.

## TL;DR

1. **Status.** ARC-AGI-1 and ARC-AGI-2 are effectively saturated for frontier models. On the arcprize.org leaderboard the relay's own models score 85.8–93.8 % on ARC-AGI-2 semi-private. ARC-AGI-3 (interactive games, launched 2026-03-25) is the live frontier: Claude Fable 5.2 is at 81.2 % and GPT-6.2 Sol at 54.9 %, with no Gemini 4 Neon entry. ARC Prize 2026 is closed and its results are due 2026-12-04. This year is the last Kaggle competition on ARC-AGI-2.
2. **Open-weight ARC models you can actually load on CPU** are the tiny recursive models.
   - TRM has 6.83 M core weights (verified by loading the checkpoint).
   - HRM has 27.28 M core weights (verified).
   - The "tiny" label hides a per-(task × augmentation) **puzzle-embedding table**: 876,406 × 512 rows in TRM, 1,045,829 × 512 in HRM, about 0.45–0.54 B floats. So the checkpoints are 1.8–2.5 GB.
   - Both models are *transductive*. They can only answer tasks whose IDs were in training.
   - Measured CPU cost for TRM is 2.8 s (bf16 with AMX) to 11.7 s (fp32) per augmented sample. The official ~1000-augmentation vote takes about 0.8–3.3 h per task. A reduced 8-view vote takes about 0.5–1.5 min per task.
3. **ARC as a programming language exists in several formal forms:**
   - Hodel's arc-dsl: 160 primitives and 400 solvers, MIT.
   - re-arc: 400 generators and 400 verifiers, MIT.
   - Google ARC-GEN: Apache-2.0.
   - BARC: Python seeds plus an 8B *induction* model that writes programs.
   - The 2025 Google Code Golf Championship (NeurIPS 2025): 400 ARC-AGI-1 tasks as files `task001..task400.json`, each with train, test and about 262 ARC-GEN examples. Sakana AI published all 400 solutions (Apache-2.0).
   - **No ARC/grid dataset called "CodeBench" exists** as far as HF/GitHub/web search shows. The owner most likely means Code Golf. The 2026 follow-up, the **NeuroGolf Championship** (IJCAI-ECAI 2026), is the most relevant find of all: participants submit a *hand-built ONNX neural network per ARC task*, scored by size. That turns "ARC program → weights" into a literal benchmark.
4. **ARC primitives as weight operations.** On real TRM weights, a residual-stream rotation applied consistently to all weights and embeddings is an exact symmetry (relative logit change 2.3e-15). The *same* rotation applied to one layer only destroys the function (argmax agreement 0 %). Joint permutation of SwiGLU units and the up/down rescaling are exact symmetries. Rescaling inside the gate, or literally rot90-ing a block of `down_proj`, changes the function.
   - So "rotate the middle layer" is only well-defined *modulo the gauge group*. A DSL must type every statement as either a gauge move (no-op) or an edit.
   - Grid primitives act naturally on weights only where a weight axis indexes grid cells: token-mixing MLPs, conv kernels, positional embeddings, NeuroGolf ONNX graphs. Even there, the 30×30 zero-padded canvas breaks the analogy for variable-size grids. A compiled `vmirror` permutation is exact on 39/266 examples of its task; a compiled `rot180` permutation is exact on 266/266.

---

## 1. Status of ARC-AGI-1/2/3 and ARC Prize 2025/2026

### Datasets

| Benchmark | Public data (verified in repo) | Licence | Format |
|---|---|---|---|
| ARC-AGI-1 | `fchollet/ARC-AGI`: 400 training + 400 evaluation JSON tasks | Apache-2.0 | static grid pairs, ≤30×30, 10 colours |
| ARC-AGI-2 | `arcprize/ARC-AGI-2`: 1000 training + 120 public evaluation tasks | Apache-2.0 | same format, harder compositional tasks (paper arXiv:2505.11831, Chollet et al., 2025-05-17) |
| ARC-AGI-3 | interactive turn-based game environments, "hundreds of original environments … thousands of levels", hand-designed, no instructions (arcprize.org/blog/arc-agi-3-launch, 2026-03-25; paper arXiv:2603.24621, v2 2026-04-17) | not stated on launch page (**UNVERIFIED**) | agent API (docs.arcprize.org), not static grids |

### Leaderboard snapshot

This was extracted from the data embedded in arcprize.org/leaderboard on 2026-11-25 (Semi-Private sets). Cost is USD/task for v1/v2. For v3 the "cost" field is an unlabelled run total.

| System | ARC-AGI-1 | ARC-AGI-2 | ARC-AGI-3 | Notes |
|---|---|---|---|---|
| Human panel | 98 % | 100 % | – | $17/task |
| Claude Fable 5.2 (XHigh) | 98.5 % ($0.36) | 93.75 % ($0.98) | 81.2 % (standard); 97.3 % with "provider adapter" harness | best v3 score on the board |
| GPT-6.2 Sol (Max) | 97.5 % | 85.8 % ($0.26) | 54.9 %; 98.2 % with provider adapter | |
| Gemini 4 Neon (Medium/Low) | 98.5 % | 93.75 % ($0.32–0.55) | no entry found | |
| ARChitects (ARC Prize 2024) | 56 % | 2.5 % | – | $0.20 |
| NVARC (ARC Prize 2025) | – | 27.64 % | – | $0.20 |
| TRM (2025-10-07) | 40 % ($1.76) | 6.25 % ($2.10) | – | ARC Prize verification run |
| HRM (2025-08) | 32 % ($1.48) | 2 % ($1.68) | – | |

Consequence for the relay: ARC-AGI-1/2 no longer separate the three relay models. ARC is useful here as a *formal, well-understood substrate*, not as a hard benchmark. To avoid contamination of the public tasks, use freshly generated ARC-GEN/re-arc instances.

### ARC Prize 2025 (verified: arcprize.org/blog/arc-prize-2025-results-analysis, 2025-12-05; technical report arXiv:2601.10904, 2026-01-15)

- 1,455 teams and 15,154 entries. The Grand Prize was not claimed.
- Kaggle top 5 (ARC-AGI-2 private):

| Place | Team | Score |
|---|---|---|
| 1 | NVARC | 24.03 % |
| 2 | the ARChitects | 16.53 % |
| 3 | MindsAI | 12.64 % |
| 4 | Lonnie | 6.67 % |
| 5 | G. Barbadillo | 6.53 % |

- The blog says "all winning solutions and papers are open-source" (Kaggle notebooks).
- Paper awards:
  - 1st: TRM ("Less is More", arXiv:2510.04871).
  - 2nd: Pourcel/Colas/Oudeyer, self-improving LMs for evolutionary program synthesis.
  - 3rd: Liao & Gu, CompressARC ("ARC-AGI Without Pretraining", arXiv:2512.06104).
  - Runners-up: ARC-VSA, Berman, Pang, ARC-NCA, ArcMemo.
- The technical report names "small-network weight-space approaches" as one of the 2025 trends.

### ARC Prize 2026 (verified: arcprize.org/competitions/2026 and its sub-pages)

- Three tracks with $2M total: an ARC-AGI-3 competition, an ARC-AGI-2 competition, and a Paper Prize.
- Dates: start 2026-03-25; ARC-AGI-3 milestones 2026-06-30 and 2026-09-30; submissions due 2026-11-02; papers due 2026-11-08; **results 2026-12-04**. The site shows "has finished".
- **ARC-AGI-3 track** ($850K):
  - $700K Grand Prize for 100 %.
  - $75K Top Score Award.
  - $75K milestone prizes.
  - No internet during evaluation; open-sourcing is required.
- **Milestone #1** (blog 2026-07-06):
  - 1st: Tufa Labs "The Duck", a small LLM (Qwen 3.6 27B FP8) writing Python in a REPL.
  - 2nd: Reki, Gemma-4-31B VLM.
  - 3rd: "forge".
  - All solutions are public Kaggle notebooks.
- **ARC-AGI-2 track** ($700K):
  - $275K progress prizes.
  - $275K Grand Prize for the best solution write-up.
  - $150K bonus for ≥85 % on the private set.
  - Organisers state this is the final year ARC-AGI-2 is used on Kaggle.
- Kaggle compute limit, per the NVARC paper for 2025: 12 h on 4× L4 for 240 tasks.

---

## 2. Open-weight models published in or around ARC Prize

### 2.1 Summary table

| Model | Params (core) | Licence | Where the weights are | I/O format | ARC scores | CPU (4 cores, no GPU) |
|---|---|---|---|---|---|---|
| **TRM**, Tiny Recursive Model (Jolicoeur-Martineau, Samsung SAIL Montreal; arXiv:2510.04871) | **6,830,082** (verified by loading) + puzzle-embedding table **876,406 × 512** in `arc_v1_public` | code MIT; ARC Prize checkpoints MIT | No official author weights. Official repo `SamsungSAILMontreal/TinyRecursiveModels` is archived (read-only since 2026-04-01). Reference weights: HF **`arcprize/trm_arc_prize_verification`** (`arc_v1_public/step_518071` 1.82 GB, `arc_v2_public/step_723914` 2.47 GB, plus `trm.py` and `all_config.yaml`). Community: `Trelis/TRM-pretrain_arc2_rearc_200k` (MIT; 40 checkpoints plus the *built dataset* with identifiers), `gaoxin492/TinyRecursiveModels-ARC-AGI-1/-2` (**UNVERIFIED** contents). NVARC's TRM checkpoints are on Kaggle dataset `cpmpml/arc-prize-trm-031` (per NVARC `TRM/README.md`). | grid flattened to 900 tokens (30×30), vocab 12 (pad = 0; 10 colours plus boundary token, exact mapping **UNVERIFIED**), plus a 16-slot puzzle-embedding prefix → sequence 916. No demonstration pairs are fed; the task is identified only by `puzzle_identifier`. | paper: 45 % ARC-AGI-1 / 8 % ARC-AGI-2. ARC Prize verified: 40 % / 6.25 %. Trelis card claims ~15 % on ARC-AGI-2 public eval after 200k epochs (card claim). NVARC's Kaggle TRM: 10.0 % public LB. | **Feasible** for inspection and single tasks. Measured: one sample at 16 ACT steps = **11.7 s fp32 / 2.8 s bf16** (§2.3). Full ~1000-augmentation vote: ~0.8–3.3 h per task. New tasks need test-time training (**not feasible** at scale on CPU). |
| **HRM**, Hierarchical Reasoning Model (Sapient; arXiv:2506.21734) | **27,277,314** (verified) + puzzle-embedding table **1,045,829 × 512** | code Apache-2.0; HF checkpoint has **no model card or licence** | HF `sapientinc/HRM-checkpoint-ARC-2` (single file `checkpoint`, 2.25 GB, 2025-07-21). Also Sudoku/Maze checkpoints. No ARC-AGI-1 checkpoint. | same 900-token / vocab-12 scheme; H_layers 4, L_layers 4, H_cycles 2, L_cycles 2, 16 ACT steps | ARC Prize verified: 32 % ARC-AGI-1, 2 % ARC-AGI-2. ARC Prize analysis (blog 2025-08-15): the outer refinement loop drives the gains; the H/L hierarchy matters little; it is effectively zero-pretraining TTT. | **Feasible.** Estimated ~0.8× TRM per sample: 24 vs 30 block applications per ACT step, derived from config, not measured. |
| **CompressARC** (Liao & Gu, CMU; arXiv:2512.06104) | **~76K** | MIT (`iliao2345/CompressARC`) | **No pretrained weights by design.** A fresh model is trained on each single puzzle (MDL objective, ~2000 steps). | "multitensor" over (example, colour, direction, height, width, channel). **Equivariant by construction to example permutations, colour permutations, rotations and flips** (paper §4, App. D.5). | 20 % ARC-AGI-1 eval, 34.75 % training | Code hard-codes `torch.set_default_device('cuda')` and needs a one-line patch. ~20 min/puzzle on an RTX 4070, so CPU is likely hours per puzzle (**UNVERIFIED**, not measured). |
| **ARChitects 2024** (Franzen, Disselhoff, Hartmann) | 8B (Mistral-NeMo-Minitron-8B); 3B variant | `…-Full-bnb-4bit`: NVIDIA Open Model License; 3B: llama3.2; ReArc1200: "other" | HF `da-fr/Mistral-NeMo-Minitron-8B-ARChitects-Full-bnb-4bit`, `da-fr/Llama-3.2-3B-ARChitects-ReArc-bnb-4bit` (2.9B), `da-fr/Mistral-NeMo-Minitron-8B-ARChitects-ReArc1200-bnb-4bit`. Code: `da-fr/arc-prize-2024`. | text tokens; **vocabulary reduced to 77 tokens**; augmentation, DFS sampling and product-of-experts rescoring (ICML 2025, arXiv:2505.07859: 71.6 % ARC-AGI-1 public eval, ~2 ct/task on a 4090) | 53.5 % ARC Prize 2024 private (1st). Leaderboard: 56 % v1, 2.5 % v2. Note: the public eval set was used as training data. | 4-bit bitsandbytes is CUDA-oriented. The 8B model in bf16 (~16 GB) does not fit 15 GB RAM. The 3B model fits. Per task with augmentation scoring this is likely tens of minutes; per-task TTT is not feasible (**UNVERIFIED** estimate). |
| **ARChitects 2025** | LLaDA-8B-Base (masked diffusion, MIT, 8.0B) | the repo has no LICENSE file; write-up is CC BY-SA 4.0 | Code and write-up: `LambdaLabsML/ARC2025_Solution_by_the_ARChitects`. Fine-tuned weights **not found**. | 2D "Golden Gate" RoPE; grids padded to 32×32 with delimiter row and column; LoRA r=512 pretraining (39 h on 8×H100); per-task TTT (128 steps, LoRA r=32); recursive re-masking | 16.53 % private (2nd) | Not feasible (8B diffusion with many refinement passes plus TTT). |
| **NVARC** (Sorokin & Puget, NVIDIA; `1ytic/NVARC`, paper `nvarc_2025.pdf`) | Qwen3-4B (post-deadline: Qwen3-4B-Thinking-2507 and Qwen3-VL-2B LLM part) plus improved TRM | **no LICENSE file in the repo** | Official Qwen weights **not found** on HF or Kaggle Models. `iamPi/Qwen-NVARC` (3.63B, no card, no licence) is **unofficial**. Data on Kaggle: `sorokin/nvarc-synthetic-puzzles` (~103k puzzles), `sorokin/nvarc-augmented-puzzles` (~3.2M samples). | **16-token vocab**: digits, newline, "user"/"assistant" as input/output markers, im_start/im_end, pad; one grid row per line. Full fine-tune on 3.26M samples (4 nodes × 8×H100, 27 h), then per-puzzle LoRA TTT (r=256). | 24.03 % private (1st); 27.64 % public LB (= leaderboard v2 entry); 29.72 % post-deadline. arXiv:2611.06827 (2026-11-04) reports continued fine-tuning of "the NVARC Qwen3-4B grid model" to 33.06 % semi-private. | 4B in bf16 (~8 GB) fits in RAM. Without TTT, a ~7k-token prompt plus decoding is minutes per attempt (**UNVERIFIED** estimate). TTT is not feasible. |
| **BARC** (Li, Hu, … Ellis; arXiv:2411.02272; `xu3kev/BARC`) | 8.03B (Llama-3.1-8B-Instruct base) | llama3.1 (models); datasets MIT; repo has no licence file | HF `barc0/Llama-3.1-ARC-{Potpourri,Heavy}-{Induction,Transduction}-8B` plus `barc0/*` datasets (100k–200k synthetic problems) | **Induction model outputs Python programs** against `seeds/common.py`; the transduction model outputs grids | paper: induction and transduction solve different tasks, and the ensemble "approaches human-level" on ARC-AGI-1 | 16 GB bf16 does not fit; a GGUF 4-bit build via llama.cpp would (**UNVERIFIED**). A few program samples per task are feasible; the paper's thousands of samples are not. |
| MindsAI (2025, 3rd) | – | – | Kaggle code per ARC Prize blog; weights **UNVERIFIED** | – | 12.64 % | – |

### 2.2 How to load and run (TRM/HRM, verified parts)

- Checkpoints are plain `torch` `state_dict`s with `torch.compile` prefixes (`_orig_mod.model.inner.*`). `torch.load(path, map_location="cpu", weights_only=True, mmap=True)` loads them instantly on CPU (spike: `scripts/inspect_ckpt.py`).
- **TRM tensors (arc_v1_public):**
  - `embed_tokens` [12,512]
  - `lm_head` [12,512]
  - `q_head` [2,512] (halting)
  - `H_init` and `L_init` [512] (bf16)
  - `puzzle_emb.weights` [876406,512]
  - a single 2-layer `L_level`, each layer with `qkv_proj` [1536,512], `o_proj` [512,512], `gate_up_proj` [3072,512] and `down_proj` [512,1536]
  - **no normalisation parameters**: RMSNorm is parameter-free, which matters for §4
- **HRM tensors:** same, plus a separate 4-layer `H_level` and 4-layer `L_level`.
- **TRM recursion (from `trm.py`):** per ACT step, `H_cycles=3` × (`L_cycles=4` updates of `z_L ← L_level(z_L, z_H + x)` plus one `z_H ← L_level(z_H, z_L)`) = 15 calls of the 2-layer stack = 30 block applications. At evaluation it "always uses max steps" (16). The verification checkpoint uses L_cycles=4, while the paper's best model used 6.
- **The official inference path needs the dataset build.** Each (task × augmentation) has its own `puzzle_identifier`, so you must rebuild IDs with `dataset/build_arc_dataset.py` (same seed and subsets) to address the right embedding rows. The Trelis repo ships the built dataset (`identifiers.json`, `all__inputs.npy`, …).
  - Its ARC-AGI-2 test split has 118,765 augmented instances over 120 tasks, i.e. ~990 views per task (verified in `dataset.json`). That makes it the most plug-and-play option for real CPU inference.
- **Transduction limit** (ARC Prize HRM analysis; arXiv:2512.11847 on the ARC Prize TRM checkpoint): accuracy drops to zero if the puzzle ID is blanked or randomised. The published checkpoints therefore answer *only* the tasks they were trained on (ARC-AGI-1 eval for `arc_v1_public`, ARC-AGI-2 eval for `arc_v2_public`). A new task needs new embedding rows trained by gradient descent; NVARC did this with TTT in ~2 h on Kaggle GPUs.

### 2.3 CPU spike (measured)

`scripts/bench_trm_cpu.py` uses my own reimplementation of the TRM block (post-norm attention with 8×64 heads, SwiGLU 1536, parameter-free RMSNorm; RoPE omitted) with the **real `L_level` weights** of the ARC Prize TRM checkpoint. Sequence length is 916.

| dtype | batch | one ACT step (30 blocks) | per sample (16 ACT steps) |
|---|---|---|---|
| fp32 | 1 | 0.73 s | **11.7 s** |
| fp32 | 8 | 5.96 s | 11.9 s |
| bf16 | 1 | 0.17 s | **2.8 s** |
| bf16 | 8 | 1.69 s | 3.4 s |

- Each block is about 8 GFLOP, so fp32 runs at ~330 GFLOP/s and bf16 at ~1.4 TFLOP/s. The bf16 speed comes from AMX; a laptop without AMX will be near the fp32 row or slower.
- **Per task** (single test input):
  - ~1000-view official vote: 46 min (bf16) to 3.2 h (fp32).
  - 8-view D4 vote: 22–94 s.
- **Weight-space work** (loading, rotating, permuting, diffing, single forward passes) is interactive on this machine. The gauge spike in §4, with 11 full TRM forwards in float64, ran in 12.7 s wall time.

---

## 3. ARC as a programming language

| Resource | What it is (verified) | Licence | Use for the DSL |
|---|---|---|---|
| **arc-dsl** (`michaelhodel/arc-dsl`, last commit 2024-10-11) | `dsl.py` with **160 primitives** (counted), `arc_types.py` (Grid, Object, Patch, Indices …), `solvers.py` with **400 solvers** for the ARC-AGI-1 training tasks (counted), `arc_dsl_writeup.pdf`. Example solvers: `solve_67a3c6ac(I): return vmirror(I)`, `solve_a416b8f3(I): return hconcat(I, I)`. | MIT | the vocabulary of grid operations (list below) |
| **re-arc** (`michaelhodel/re-arc`; arXiv:2404.07353, Hodel, 2024-04-10) | 400 procedural **generators** and 400 **verifiers** (counted); `re_arc.zip` holds 1000 verified examples per task with difficulty metrics; bundles the DSL | MIT | unlimited fresh instances per program, for contamination-free probes |
| **ARC-GEN** (`google/arc-gen`; arXiv:2511.00162, Moffitt, 2025-10-31) | "mimetic" generators for all 400 ARC-AGI-1 tasks; ARC-GEN-100K; news item 2026-03-25 says +500 ARC-AGI-2 tasks; official generator of **Code Golf 2025** and **NeuroGolf 2026** | Apache-2.0 | the generator behind both golf benchmarks |
| **BARC** (`xu3kev/BARC`; arXiv:2411.02272) | 162 hand-written seed programs (Python `main()` plus `generate_input()`, concept labels, NL description) over `seeds/common.py`; LLM-remixed to 100k–200k synthetic problems; induction (program-writing) and transduction 8B models | repo: none stated; datasets MIT; models llama3.1 | a Python grid library plus a program-writing open model |
| **2025 Google Code Golf Championship** (NeurIPS 2025 competition; organisers Moffitt, Thakkar, Burnell, Firat, Reade, Dane, Howard; Kaggle `google-code-golf-2025`) | Write the *shortest* Python program per ARC task. Data (from the competition files mirrored in the Sakana repo): `task001.json … task400.json`, each `{train, test, arc-gen}` (task001: 5/1/262). `id_map.csv` maps them to ARC-AGI-1 training IDs (task001 = 007bbfb7). Score per task max(1, 2500 − bytes), from Sakana's judge code, a secondary source. | ARC data Apache-2.0; Kaggle data licence **UNVERIFIED** (Kaggle pages do not render without JS) | the ARC corpus as 400 minimal programs |
| ↳ public solutions | **`SakanaAI/google-code-golf-2025`** (Apache-2.0): **all 400 solutions** in `solutions/taskNNN.py`, total 954,571 in its `scores.csv`. Example: task001 `p=lambda g:[[X&x for X in A for x in a]for A in g for a in g]` (61 bytes). The repo also holds a snapshot of a community scoreboard (top: "Code Golf International" 962,070). | Apache-2.0 | ready-made program corpus |
| ↳ official winners | not verifiable from official pages (**UNVERIFIED**) | – | – |
| **NeuroGolf Championship 2026** (IJCAI-ECAI 2026; official site sites.google.com/view/neurogolf-2026) | **"Implement a variety of transformations using the simplest possible networks."** Organisers Moffitt, Oldacre, Reade, Howard. $50K. Ran 2026-04-15 to 2026-07-15; special session 2026-08-15 in Bremen. Kaggle page `kaggle.com/competitions/neurogolf-2026` (title verified). | – | **the literal "ARC program → network weights" benchmark** |
| ↳ rules | From the community repo `zikuanqi/NeuroGolf` (MIT), since the official site lists no rules: one ONNX network per ARC task; input/output one-hot float32 `(1,10,30,30)`, output thresholded at >0; ≤1.44 MB; banned ops include LOOP, SCAN, NONZERO, UNIQUE, Sequence*; static shapes; score max(1, 25 − ln(memory_bytes + params)) per task if every train/test/arc-gen example passes. That repo hand-builds networks (157 pattern solvers) and solves 232/400 locally. | MIT (that repo) | – |
| ↳ winners | only participant LinkedIn posts (**UNVERIFIED**) | – | – |
| **CodeARC** (arXiv:2503.23145; COLM 2025) | the closest *name* match: an inductive program-synthesis benchmark with 1114 general Python functions (not grids); HF `anjiangwei/CodeARC-Problems` | – | not ARC grids; probably not what the owner means |
| **"CodeBench" / "Codebench"** | **No ARC- or grid-related dataset of that name was found.** HF dataset search returns only generic code benchmarks (e.g. `Kiy-K/codebench-verified`, `toolevalxm/CodeBench-HumanEval-*`); GitHub returns unrelated projects (microVM runner, Kohana module, UFAM course data). | – | interpret as Code Golf 2025 (Python) and/or NeuroGolf 2026 (ONNX) |

**arc-dsl primitives**, grouped from `dsl.py`:
- **geometry:** `rot90 rot180 rot270 hmirror vmirror dmirror cmirror shift normalize`
- **size and tiling:** `hupscale vupscale upscale downscale hconcat vconcat hsplit vsplit crop subgrid trim compress tophalf bottomhalf lefthalf righthalf canvas`
- **colour:** `recolor replace switch fill paint underfill underpaint cover palette numcolors mostcolor leastcolor colorcount colorfilter ofcolor`
- **objects and partitions:** `objects partition fgpartition toobject asobject toindices sizefilter`
- **topology and metric:** `dneighbors ineighbors neighbors adjacent bordering manhattan hmatching vmatching gravitate position centerofmass`
- **shape predicates:** `square vline hline portrait height width shape`
- **lines, boxes and frontiers:** `connect shoot box inbox outbox backdrop delta corners frontiers hfrontier vfrontier`
- **periodicity:** `hperiod vperiod`
- **pattern matching:** `occurrences cellwise`
- **functional combinators:** `compose chain fork rbind lbind apply mapply mfilter sfilter power branch matcher`
- **arithmetic and set operations**

---

## 4. Which primitives can double as operations on weight matrices viewed as grids?

### 4.1 Spike 1: gauge versus edit on real TRM weights

Script: `scripts/gauge_spike.py`, float64, my reimplementation of the full H/L recursion for one ACT step. It uses the real `L_level`, embedding, `lm_head` and one puzzle-embedding row, on random input tokens.

| Operation | Relative logit change | Argmax agreement | Verdict |
|---|---|---|---|
| Rotate the residual basis by a random Q∈O(512) **in every reader/writer matrix and embedding** (readers W→WQ, writers W→QᵀW, E→EQ) | 2.3e-15 | 100 % | **exact symmetry.** RMSNorm has no gain, so the residual basis is arbitrary up to O(512). |
| Same Q applied to **one layer only** | 1.09 | 0 % | destroys the function |
| Jointly permute the 1536 SwiGLU units (gate rows, up rows, down columns) | 1.6e-15 | 100 % | exact symmetry |
| Permute gate rows only | 0.84 | 57 % | edit |
| Scale up-rows by s>0 and down-columns by 1/s | 9e-16 | 100 % | exact symmetry (up is bilinear) |
| Same scaling on gate rows (inside SiLU) | 0.147 | 100 % | edit (SiLU is not homogeneous) |
| Literal grid `rot90` of a 512×512 block of `down_proj` | 0.69 | 100 % | large edit. Argmax is uninformative on random inputs, so read the logit change. |

**Reading.** The owner's sentence "all weights in the middle layer rotate" has two precise and very different meanings:
- the **gauge move** (rotate and compensate in the neighbours), which does nothing; or
- the **edit** (rotate without compensation), which is a large perturbation.

A weight-space DSL must make that distinction explicit, check it numerically, and state the symmetry group of each architecture: for TRM, O(512) residual rotation × S₁₅₃₆ unit permutation × positive diagonal up/down scaling per layer, plus the standard per-head V/O and QK freedoms (not tested here).

TRM and HRM are also **weight-tied recurrences**. The "middle layer" is a middle *iteration* of the same two blocks, so statements about it are statements about activation trajectories (z_H, z_L) rather than about distinct matrices. arXiv:2512.11847 finds most accuracy is already reached at the first recursion step ("shallow effective recursion").

### 4.2 Spike 2: compiling arc-dsl primitives into weights (NeuroGolf encoding)

Scripts: `scripts/golf_linear.py` and `scripts/compile_perm.py`, run on the Code Golf task files.

- **Single primitives.** Six of the 400 tasks are solved exactly by one geometric primitive:

| Task | ARC id | Primitive | Distinct input sizes |
|---|---|---|---|
| task087 | 3c9b0459 | rot180 | 1 |
| task140 | 6150a2bd | rot180 | 1 |
| task150 | 67a3c6ac | vmirror | 7 |
| task155 | 68b16354 | hmirror | 5 |
| task179 | 74dd1130 | dmirror | 1 |
| task241 | 9dfd6313 | dmirror | 7 |

- **Compiled permutations.** Compiled to an explicit weight matrix kron(I₁₀, P) with P a 900×900 cell permutation:
  - `rot180` (task087) is exact on **266/266** examples.
  - `dmirror` (task179) is exact on 267/267.
  - The same construction for `vmirror` (task150) is exact on only **39/266**, and `dmirror` on task241 on 32/266.
  - Reason: on a zero-padded 30×30 canvas, mirroring a w×h grid is a *different* permutation for every size, so a size-detection gate (non-linear) is needed.
- **Shape statistics.**
  - Only 204/400 tasks have a constant input shape across train, test and arc-gen.
  - 262/400 keep the output shape equal to the input shape.
  - The other 138 change shape, which in weight terms means crop, tile or concat of matrices.
- **Unstructured linear fit.** A minimum-norm least-squares map (9000→9000) fitted on train+test reproduces them in 394/394 tasks but generalises to all ARC-GEN examples in only **1/394** (task056 = 27a28665). Unconstrained weights memorise; symmetry or structure must come from the parametrisation. This is the same lesson as HRM/TRM puzzle-ID transduction and CompressARC's built-in equivariance.

### 4.3 Primitive ↔ weight-operation dictionary

| arc-dsl primitive | Counterpart on a weight matrix W (d_out × d_in) | Function-preserving? | Where the analogy breaks |
|---|---|---|---|
| `rot90/180/270`, `hmirror`, `vmirror`, `cmirror` | W → P_out W P_in with reversal permutations, plus a transpose for odd rotations | Only if the compensating inverse permutation is applied to the adjacent layers (gauge). Alone it is an edit (spike 1). | Rows and columns of hidden layers have **no spatial order**: any permutation is equivalent, so "rotation" of a hidden×hidden matrix has no intrinsic meaning. It is meaningful only where an axis indexes grid cells: token-mixing MLP (TRM's `mlp_t` variant over seq_len+16 positions, where a grid rotation of the input acts as W → ΠWΠᵀ), conv kernels (ARC-NCA, NeuroGolf), positional encodings (TRM/HRM use 1D RoPE on the flattened grid, so grid rotation is *not* an architectural symmetry and D4 is learned from augmentation; ARChitects 2025 uses 2D Golden-Gate RoPE; CompressARC hard-wires D4). |
| `dmirror` (transpose) | W → Wᵀ (swaps reader and writer roles) | Generally not; it is only type-correct for square W | In tied-embedding LMs Wᵀ is meaningful (unembedding = embeddingᵀ). In general it changes the function. |
| `upscale`, `hupscale`, `vupscale` | Kronecker widening W ⊗ 1, i.e. Net2Net (arXiv:1511.05641) unit duplication | **Yes**, if outgoing weights are divided by the replication factor | Breaks symmetry only after further training |
| `hconcat`, `vconcat`, tiling | adding units or heads (widen) or stacking layers (deepen, identity init) | Yes with zero or identity initialisation | Not for weight-tied recursions (TRM): depth = iterations, not new matrices |
| `crop`, `subgrid`, `trim`, `compress`, `tophalf`… | pruning rows/columns, heads or units; SliceGPT (arXiv:2401.15024) deletes rows/columns *after* a gauge rotation (PCA) | Approximately, if the removed directions carry little variance | `compress` (drop uniform rows/columns) ≈ removing dead units, which is exact only if they are truly dead |
| `recolor`, `replace`, `switch`, `palette`, `numcolors` | elementwise value maps: scaling, sign flip, **quantisation to a k-level codebook** (palette = codebook, numcolors = bit width) | Scaling only across bilinear or homogeneous pairs (spike 1: up/down yes, gate no). (W_V, W_O) → (−W_V, −W_O) is exact. | ARC colours are categorical; weights are real-valued and ordered. Rendering a weight matrix as an ARC grid needs a quantisation choice, which the DSL must name. |
| `objects`, `partition`, `fgpartition` | blocks: attention heads (contiguous slices of qkv_proj, 8×64 in TRM), SwiGLU units, connected components of the thresholded **bipartite weight graph** (circuits) | – (they are analyses, not edits) | ARC objects use 4- or 8-*adjacency* on the grid. In weight matrices, adjacency is meaningless until the permutation gauge is fixed: sort units, or align with Git Re-Basin (arXiv:2209.04836). Graph connectivity replaces grid connectivity. |
| `dneighbors`, `ineighbors`, `neighbors`, `adjacent`, `gravitate`, `shift` | – | – | Need a canonical ordering. `shift` corresponds to a circulant permutation; only meaningful for position-indexed axes. |
| `hperiod`, `vperiod`, `occurrences` | periodic or repeated structure, e.g. Fourier-sparse embeddings in grokked modular-addition transformers (Nanda et al., arXiv:2301.05217) | analysis | This is where "grokking" becomes a weight-space *measurement*. Spectral periodicity is basis-dependent but well defined on the embedding axis (tokens). |
| `cellwise`, `fill`, `paint`, `underpaint` | masked edits, i.e. weight patches (W ← W ⊙ (1−M) + V ⊙ M) | Edits by definition | `underpaint` (paint only on background) ≈ editing only near-zero weights |
| colour-permutation augmentation | permutation of the 10 colour rows of `embed_tokens` and `lm_head` (TRM: 12×512) | Exactly corresponds to relabelling colours in input and output | The puzzle-embedding row also changes per augmentation; see the hypothesis below |

### 4.4 Hypothesis worth a spike (UNVERIFIED)

The TRM/HRM **puzzle-embedding tables** (876k or 1.05M rows × 512) are a weight-space object with known group structure. Each row is one (task, D4 × colour-permutation augmentation) pair.

If the model has internalised the augmentation group, rows of the same task under group element g should be related by a shared, task-independent linear map A_g. Fitting A_g on half the tasks and testing on the others gives a direct test of whether symmetries are represented in weight space.

This is testable on CPU with the Trelis repo, which ships `identifiers.json` mapping rows to (task, augmentation). It is a natural first "weight-space statement" grounded in ARC: "the task-embedding of rot90(T) equals A_rot90 · emb(T)". The Trelis repo also offers **40 checkpoints across training** (steps 18,433 to 737,320, 1.75 GB each, plus per-step `submission.json`). That is a ready-made training trajectory for the owner's grokking and steering questions ("is there a phase transition, and is the direction steerable?").

---

## 5. Recommendations (Lens A)

1. **Use NeuroGolf, ONNX on one-hot (1,10,30,30) grids, as the bridge between "ARC as programming language" and "weight space".**
   - Every arc-dsl, Code Golf or BARC program can in principle be compiled to a small network.
   - Geometry becomes permutations or Transpose/Slice ops, recolour becomes a 1×1 conv, and `objects` becomes unrolled conv flood-fill (LOOP is banned).
   - The result is ground-truth weight-space objects with known semantics, against which the relay's weight-space statements can be checked.
2. **Use TRM, via `arcprize/trm_arc_prize_verification` (MIT) and `Trelis/TRM-pretrain_arc2_rearc_200k` (MIT, with identifiers and 40 checkpoints), as the primary open-weight ARC substrate on CPU.**
   - The 6.83 M core weights fit in seconds of compute per forward pass.
   - Skip the 8B LLM solvers for weight-space work. HRM is the second substrate (27 M core, Apache code, but no licence on the HF checkpoint).
3. **Make the DSL gauge-aware.**
   - Every statement should carry its symmetry group and be classified automatically as *gauge* (no-op) or *edit*, with a numeric check like `gauge_spike.py`.
   - Rendering weights as ARC grids for the "graphical approach" requires (a) a canonical unit ordering and (b) a named quantisation (palette). Otherwise neighbourhood and object primitives are meaningless.
4. **Respect weight tying.** In TRM/HRM, "layer k" means "iteration k", so the DSL needs separate sorts for *parameters*, *iterations/activations* and *task embeddings*.
5. **Probe with freshly generated ARC-GEN/re-arc instances**, not the public ARC files. The relay models score ≥85.8 % on ARC-AGI-2, and the public tasks are contaminated (ARC Prize 2025 report).
6. **"Codebench".** Tell the owner it does not exist under that name. The formalised ARC-as-code resources are Code Golf 2025 (400 Python programs, Sakana's full set public) and NeuroGolf 2026 (ARC-as-networks).

## 6. Caveats

- **CPU timings** come from my reimplementation: RoPE, ACT bookkeeping, `lm_head` and the voting pipeline are omitted. They ran on a Xeon with AMX, so bf16 numbers are optimistic for typical laptops. HRM, CompressARC, ARChitects, NVARC and BARC CPU figures are estimates, not measurements.
- **The gauge spike** uses random tokens and an arbitrary puzzle-embedding row, not a real ARC task. Relative logit change is the reliable metric there; argmax agreement on random inputs can mislead (see the rot90 row).
- **Leaderboard figures** were parsed from embedded page data on 2026-11-25 and will change. The ARC Prize 2026 final results (2026-12-04) are not out yet. The v3 "cost" unit is unlabelled.
- **Code Golf 2025:** official scoring, data licence and winners could not be read from Kaggle (JS-only). The scoring formula comes from Sakana's judge code, and the ranking comes from a community scoreboard snapshot.
- **NeuroGolf 2026:** the rules come from a community repo, because the official site lists none. Winners are unverified.
- **Licences:** no licence file was found in `1ytic/NVARC`, `LambdaLabsML/ARC2025_Solution_by_the_ARChitects` or `xu3kev/BARC`, and the HRM HF checkpoint has no licence, so reuse terms are unclear.
- **Disk:** the downloaded checkpoints take about 4 GB (`dl_trm_ckpt/` 1.82 GB, `dl_hrm_ckpt/` 2.25 GB under the `wsdsl` scratch directory). Delete them when no longer needed.

## 7. Files

- Scripts: `/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/scripts/`
  - `inspect_ckpt.py` (tensor inventory)
  - `bench_trm_cpu.py` (CPU timing)
  - `gauge_spike.py` (gauge vs edit)
  - `golf_linear.py` (shape statistics and linear-fit test)
  - `compile_perm.py` (primitive → permutation weights)
- CPU-only PyTorch venv: `/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/venv/`
- Downloaded (untrusted) data, each in its own directory under `/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/`:
  - `dl_trm_ckpt/`, `dl_hrm_ckpt/`, `dl_trm_cfg/`, `dl_hrm_cfg/` (checkpoints and configs)
  - `dl_arcdsl/` (arc-dsl, re-arc), `dl_golf/sakana/` (Code Golf data and solutions)
  - `dl_nvarc/`, `dl_nvarc_repo/`, `dl_compressarc/`, `dl_arch25/`, `dl_arcdata/` (ARC-AGI-1/2)
  - `dl_trelis/` (card and configs), `dl_lb/` (leaderboard HTML)
