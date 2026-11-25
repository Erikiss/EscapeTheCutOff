# Lens C: operations, steering, tooling, and what fits on this machine

Research date: 2026-11-25. Every paper, model, repo and library below was opened on arXiv, GitHub, Hugging Face (web page or `/api/models`), PyPI (`/pypi/<pkg>/json`), docs.pytorch.org, arcprize.org or nvidia.com during this session. Anything else is marked **UNVERIFIED**. Numbers marked **measured** come from my own runs on this machine. The machine is 4 vCPU (Xeon @ 2.30 GHz), 15 GB RAM, no GPU, and it is shared with sibling agents, so wall-clock times are upper bounds.

My own scripts are in `ops_scripts/`, their outputs in `ops_out/`, and the downloads in `dl_ops_*`. All three directories are in this folder.

| Script | Output | What it does |
|---|---|---|
| `weight_ops.py [f64]` | `ops_out/weight_ops*.out` | 7 closed-form weight edits on Pythia-70M-deduped, each checked against its activation-space twin |
| `resid_rotation.py` | `ops_out/resid_rotation.out` | Residual-stream rotation symmetry on real Pythia-70M (LN folding, Q·1 = 1, all vs middle-only) |
| `tool_probe.py` | `ops_out/tool_probe.out` | TransformerLens 4 / nnsight / pyvene / SAELens on CPU |
| `ckpt_line.py` | `ops_out/ckpt_line.out`, `ops_out/ckpt_line_scan.out` | Pythia-14M checkpoint interpolation with `torch.func.functional_call` |
| `contrastive_ws.py` | `ops_out/contrastive_ws.out` | Mini contrastive weight steering vs activation steering on Pythia-70M |
| `grok.py` | `ops_out/*_s0.json`, `ops_out/log_*.txt` | Nanda-recipe grokking with steering variants |
| `install_measure.sh` | `ops_out/install.log` | Fresh CPU venv; size after each library |
| `count_ckpt.py` | (inline below) | Parameter census of the TRM/HRM ARC checkpoints a sibling lens downloaded |

---

## 0. TL;DR

* **Many activation-space operations compile exactly to weight edits.** I checked each one numerically on a real model (Pythia-70M-deduped, float64). The difference between "weight-edited model" and "hooked model" is **2.7e-11 to 6.0e-11** in logits, against edit effects of 4.5 to 250. The operations are:
  * directional ablation, i.e. weight orthogonalization of all residual writers (Arditi et al.);
  * an all-position steering vector, which becomes a bias edit;
  * MLP neuron permutation;
  * a per-head GL(d_head) V/O change of basis.

  A rotation of MLP neurons is **not** a symmetry: the loss goes from 4.66 to 9.85.
* **The owner's "rotate the weights" example, tested on a real LayerNorm model** (`resid_rotation.py`). Rotating the residual basis of *all* readers, writers, embed and unembed is exact (1.5e-11), but only under two conditions:
  1. LN gains and biases are folded into the readers first;
  2. Q fixes the all-ones vector, because LayerNorm centres.

  Each variant that breaks a condition fails badly (max \|Δlogit\|):

  | Variant | max \|Δlogit\| |
  |---|---|
  | Generic Q | 13.8 |
  | No folding | 1,232 |
  | Rotating **only the middle layer** | 366 |

  So "all weights in the middle layer rotate" is false as an invariance. It is true only in the global, folded, centring-respecting form.
* **Weight steering beats activation steering at CPU scale, at least in this toy.** I ran a miniature contrastive weight steering experiment (Fierro & Roger, arXiv 2511.05408) on Pythia-70M: two 60-step fine-tunes, 2.7 min on 1 CPU thread. At equal capability cost it gave about **2x** the held-out sentiment shift of a mean-difference activation vector:
  * Δloss about 0.12: +7.5 log-odds (weight) vs +4.6 at Δloss 0.19 (activation);
  * Δloss about 1.3: +24.4 vs +13.0 at Δloss 1.5.

  One seed, own toy data, and the activation vector was tried at one layer only.
* **"Project out an SAE direction" is not "ablate an SAE feature".** This is a type distinction the DSL must make. Measured on Pythia-70M-deduped with the SAELens `pythia-70m-deduped-res-sm` layer-3 SAE:
  * Removing feature 21809's decoder direction from all writers drives the feature to 0, but the loss jumps from 3.88 to 6.98. Every token carries a component of 4 to 8 along that direction, plus 123 and 191 on two tokens.
  * Ablating the feature only where it fires leaves the loss at 3.877.

  Only the second is the "feature" statement, and it is **not** a static weight edit: it needs the SAE encoder in the forward pass.
* **"Is grokking steerable?" Yes, modestly, and tested here.** The canonical Nanda et al. recipe runs at **43 ms/step** on 4 uncontended threads, about 7 min per 10k full-batch steps. Measured, seed 0, test accuracy ≥ 0.95:
  * base groks at step **9,900**;
  * Grokfast-EMA (α = .98, λ = 2) at **6,700** (1.48x; the paper reports ×44–50 only in its own mini-batch setup);
  * a single weight-space edit θ ← 0.7θ at the end of memorization at **8,900** (1.11x). The optimizer restores the norm within about 500 steps, so with one seed this result is inconclusive.

  Details are in §4.3.
* **Tooling fits.** One CPU venv holds torch 2.15 (CPU), TransformerLens 4.4.1, nnsight 0.8.4, pyvene 0.1.8 and SAELens 6.56.1. It is 1.66 GB and installs in under 30 s with uv. Everything runs on CPU.

  **Version trap for the relay:** TransformerLens **4.0 (2026-09-21) removed `HookedTransformer`**, and TransformerBridge arrived in 3.0 (2026-04-17). The measured import error reads: "'HookedTransformer' was removed in TransformerLens 4.0. Use TransformerBridge.boot_transformers(name), then enable_compa…". All three relay models' training predates 4.0, so they will write `HookedTransformer.from_pretrained(...)`, and that fails on the current release. The fix is to pin `transformer-lens<4`, or to give the models the TransformerBridge API in the prompt.
* **ARC models are mostly per-task embeddings, not network.** My parameter count of the two ARC checkpoints:

  | Checkpoint | Network | Per-puzzle embeddings | Share of weights in embeddings |
  |---|---|---|---|
  | TRM (`arcprize/trm_arc_prize_verification`, arc_v1) | 6.8M | 448.7M (876,406 x 512) | 98.5% |
  | HRM (`sapientinc/HRM-checkpoint-ARC-2`) | 27.3M | 535.5M (1,045,829 x 512) | 95% |

  For the owner's "interface to published ARC models", the natural weight-space object is therefore **one 512-d puzzle-embedding row per (augmented) task**, next to a tiny shared recursive core.

---

## 1. Activation steering vs weight steering: which operations are closed-form weight edits

Notation: the residual stream is x ∈ R^d. *Writers* are the matrices whose outputs are added to x: W_E, W_pos if learned, W_O of every head, W_out of every MLP, and their biases. *Readers* are W_Q, W_K, W_V and W_in, behind the norms.

### 1.1 Table of operations

| Operation (activation form) | Weight form | Exact? | Evidence |
|---|---|---|---|
| **Directional ablation everywhere**: x ← (I − r rᵀ)x after every writer | W ← (I − r rᵀ)W for every writer and its bias (W_E rows: E ← E(I − r rᵀ)) | **Exact** | Arditi et al., arXiv 2406.11717 §4.1: "this weight modification is equivalent to the previously described inference-time directional ablation" (proof in their App. E). **Measured**: 2.68e-11 (f64) between weight edit and hook; effect 4.5 logits. nnsight hook vs HF weight edit: 6.35e-3 in fp32 at max\|logit\| 1583 |
| **Steering vector at all positions of layer ℓ** (CAA-style, every token): x_ℓ ← x_ℓ + αv | b_O^(ℓ) ← b_O^(ℓ) + αv (or b_out of the MLP) | **Exact**, if the layer has an output bias | **Measured** 2.71e-11 (f64). For bias-free models (Llama/Qwen o_proj and down_proj) a bias parameter must be added. The weight-steering repo ships `models_with_mlp_bias.py`, plausibly for this reason (**UNVERIFIED** purpose) |
| **ActAdd at the first k prompt positions** (Turner et al., arXiv 2308.10248) | Only at layer 0 with learned absolute positions: W_pos[k] += v. Otherwise none | Not a static edit in general | Position-dependent. In GPT-2 the layer-0 case is the W_pos row edit; with RoPE nothing like it exists |
| **Directional ablation at one layer only** | No edit of the existing weights. It needs an inserted projection | Not a static weight edit | Later writers can write r again; multiplying all later readers by P also removes later writes, so it differs |
| **SAE feature ablation / clamping** (f_i(x) set to 0 or c) | No edit of the existing weights. The SAE (encoder plus decoder) has to be inserted as a module | Not a static weight edit | Input-dependent through the encoder. **Measured**: loss 3.877 → 3.877 (feature ablation) vs 3.877 → 6.975 (project out d_i in weights) |
| **Add an SAE decoder direction c·d_i everywhere** | Bias edit, as in the steering row | Exact | Same algebra as the steering row |
| **Zero-ablate head (l,h)** | W_O[:, h-block] ← 0 | Exact | **Measured** sweep of 48 heads (§5, #8) |
| **Mean-ablate head / neuron** | W_O[:,h] ← 0 and b_O += W_O[:,h]·E[z_h]; same for neurons with W_out, b_out | Exact for a fixed dataset mean | Algebra. The mean is a constant vector |
| **LoRA** ΔW = BA | Merged W + BA | Exact by construction | Hu et al., arXiv 2106.09685: the trained low-rank matrices merge, "no additional inference latency" |
| **ReFT / LoReFT** h ← h + Rᵀ(Wh + b − Rh) at selected positions | None in general | No | Wu et al., arXiv 2404.03592: interventions on hidden representations of a frozen model at chosen positions ("15x–65x more parameter-efficient than LoRA"). At all positions it is a rank-r affine map in the residual, so it needs an inserted layer (**UNVERIFIED** whether it can be folded) |
| **Task vector / task arithmetic** θ + αΔ | It *is* a weight edit | n/a | Ilharco et al., arXiv 2212.04089 (negation, addition, analogies) |
| **Contrastive weight steering** θ + α(Δ₊ − Δ₋) | Weight edit (difference of two small fine-tunes) | n/a | Fierro & Roger, arXiv 2511.05408 (ICLR 2026). It "often generalizes further than activation steering". Δ·w_evil is used to detect emergent misalignment. Code: github.com/safety-research/weight-steering (Llama-2-7b-chat; no license file shown) |
| **ROME / MEMIT** | Closed-form rank-one (or multi-layer) update of MLP W_out from key k* and value v* | Exact closed form | Meng et al., arXiv 2202.05262 (NeurIPS 2022) |
| **Persona-vector preventative steering** | Activation steering applied *during* fine-tuning | Changes the training dynamics, not the weights directly | Chen et al., arXiv 2507.21509. It is a Dynamics statement in Lens B's grammar |
| **Gauge symmetries**: neuron permutations, positive scaling (ReLU only), per-head GL on VO, QK (RoPE-restricted), global residual rotation after norm-gain folding | Function-preserving weight edits | Exact | **Measured** here for permutation, VO, and the global residual rotation on real Pythia-70M (LayerNorm: fold γ/β into the readers; Q·1 = 1 because of centring; 1.5e-11). Theory: SliceGPT (arXiv 2401.15024, ICLR 2024) "computational invariance". Lens B verified the RMSNorm version on random weights (`spike/symmetry_spike.out`, A1–A5) |
| **Layer swap / skip / parallel** | Structural weight edits | n/a (empirical) | Sun et al., "Transformer Layers as Painters", arXiv 2407.09298: middle layers show "a surprising amount of uniformity". **Measured** on Pythia-70M in §5 |

### 1.2 Weight-native interpretability: decompositions that are already statements about weights

* **Transcoders** (Dunefsky, Chlenski, Nanda; arXiv 2406.11944, NeurIPS 2024) replace an MLP with a wide, sparse MLP. They give circuits that "neatly factorize into input-dependent and input-invariant terms". The input-invariant term is a pure weight product (W_dec of an upstream feature times W_enc of a downstream feature). That makes "feature i feeds feature j with strength s" a checkable weight-space statement. SAELens ships `Transcoder`, `SkipTranscoder` and `JumpReLUTranscoder` classes, plus the pretrained release `gpt2-small-mlp-tm`.
* **Crosscoders** (Lindsey et al., transformer-circuits.pub, 2024-10-25) read from and write to several layers, or several models. Their main weight-space use is **model diffing** between checkpoints or fine-tunes.
* **Parameter decompositions.**
  * APD (arXiv 2501.14926).
  * SPD (Bushnaq, Braun, Sharkey; arXiv 2506.20790, v2 2025-09-04; code github.com/goodfire-ai/spd) splits every weight matrix into rank-one subcomponents with a learned causal importance.
  * Christensen & Riggs (arXiv 2511.08854, NeurIPS 2025 MI workshop) extend SPD to transformers, including GPT-2-small (subcomponents for "golf" and "basketball").

  "Ablate subcomponent k" is then literally a weight edit.
* **Weight-sparse transformers.**
  * Gao et al. (OpenAI; arXiv 2511.13653) find circuits about 16x smaller than in dense models of comparable loss. Weights are released at github.com/openai/circuit_sparsity (Apache-2.0; `csp_yolo1` has 118M params, `csp_yolo2` 475M; the README says the GPT code is suitable for CPU inference).
  * Marin-Llobet & Heimersheim (arXiv 2607.02964, 2026-07-03) report that 12–31% of individual weights in a sparse transformer have a single short, held-out-verified description.

  These are the cleanest "one weight, one meaning" substrates.
* **Context: "weight-space programming" became a named topic in November 2026.** All three papers below are beyond all relay models' cutoffs:
  * WSP-Scope (Hu et al., arXiv 2611.09382, 2026-11-06; code github.com/WSP-Scope/WSP-Scope-Project): agents edit Qwen3.5-0.8B/9B weights directly. MATH-500 gains up to +29.45 from modifying only 1,023 parameters; 6 of 16 trajectories kept their original weights.
  * Zhang, Wang, Lin (arXiv 2611.09376, 2026-11-06): GPT-6 Astra edits 65 token-embedding rows of π0.5 on a **CPU workspace** (86.3% → 92.0% on held-out paraphrases).
  * iWSP (arXiv 2611.18114): covered in Lens B.

---

## 2. Tooling: what each library does for weight edits and verification

Installed into a fresh venv with uv on this machine (**measured**, `ops_out/install.log`). Versions and dates are from PyPI JSON.

| Library | Version (date) | License | Size added to venv | Weight edits | Verification / hooks | CPU (measured) |
|---|---|---|---|---|---|---|
| **torch** (CPU wheel) | 2.15.0 (2026-10-28) | BSD-style composite | 800 MB | Any (`with torch.no_grad(): W.copy_(...)`) | `register_forward_hook`. **`torch.func`**: `functional_call` (forward with substituted params, so no mutation), `vmap`, `grad`, `jvp`/`vjp`, `linearize`, `jacrev`/`jacfwd`, `hessian`, `stack_module_state` (ensembles for vmap) | Yes. `functional_call` line scans: Pythia-14M 11 points in ~4 s |
| **TransformerLens** | 4.4.1 (2026-11-25); 3.0.0 2026-04-17; 4.0.0 2026-09-21 | MIT | +476 MB (pulls transformers 5.23, datasets, pandas, wandb, …) | `TransformerBridge` wraps the HF module (`bridge.original_model`), so edits go there. `enable_compatibility_mode()` gives HookedTransformer-style processed weights (folded LN, centred writers and unembed) | 259 hook points on Pythia-70M; `run_with_cache`; exports `FactoredMatrix` and `SVDInterpreter` (weights-based analysis); `head_detector` | Yes. Boot plus cached run 2.6 s. Logits vs HF 8.8e-3 (fp32, logit scale 1583); compat-mode log-probs vs HF 7.1e-3 |
| **nnsight** | 0.8.4 (2026-11-23) | MIT | +27 MB | Parameters are the real module's: an in-place edit is seen by the next trace (measured: logits change 53.6 after zeroing L0 `attention.dense`). Persistent `edit(*, inplace=False)` (**measured**): `with lm.edit() as (tracer, edited):` stores *activation* interventions that are replayed on every later trace of the edited shallow copy. The base model is untouched (Δ 0.0 vs 19.0 on the copy). `clear_edits()` removes them. Note the 0.8 binding is a tuple; older docs show `as edited` | `with lm.trace(ids):` read or assign `.output`. **Gotchas hit:** modules must be accessed in execution order (`OutOfOrderError`), and the `with` body must start on its own line. `LanguageModel` is deprecated in favour of `TransformersModel`. `remote=True` runs on NDIF (needs an API key) | Yes. Trace 1.6–1.9 s. Activation ablation reproduced the HF weight orthogonalization to 6.35e-3 (fp32) |
| **pyvene** | 0.1.8 (2025-05-26; no release since) | Apache-2.0 | +295 MB (matplotlib, plotnine, datasets, …) | Activation-level interventions only ("internal states of PyTorch models") | Declarative `IntervenableConfig` (component, unit, intervention type). The closest existing thing to a *declarative* intervention language | Yes. `ZeroIntervention` on Pythia `block_output`, 1.7 s. `pyreft` 0.1.0 pins `transformers==4.45.1`, so it conflicts with TL 4.x and needs a separate venv |
| **SAELens** | 6.56.1 (2026-11-20) | MIT | +60 MB | Gives `W_enc`/`W_dec`. Decoder rows become directions for orthogonalization or bias edits. Transcoder classes | 170 pretrained releases in the directory, including `pythia-70m-deduped-{res,mlp,att}-sm`, `gpt2-small-res-jb`, `gpt2-small-mlp-tm` (transcoders) and SAEBench Pythia-70M sweeps. `SAETransformerBridge` | Yes. `SAE.from_pretrained` for one Pythia-70M layer (d_sae 32768) took 1.5–8 s (134 MB download) |

**Total**: 1,658 MB for the whole stack, installed in about 25 s (torch 15 s).

**Verification pattern that worked on all of them.** For every weight statement, run the same model three ways and compare:
1. the edited weights;
2. the unedited weights plus the equivalent hook;
3. the unedited weights alone.

Do it in **float64** (`torch.set_default_dtype(torch.float64)`, `dtype=torch.float64` in `from_pretrained`). fp32 noise on Pythia-70M is about 1e-2 in logits, because max\|logit\| is 1583. In f64 the equality checks land at about 1e-11 and the separation from real effects is 10+ orders of magnitude.

---

## 3. Small open models for CPU experiments (verified on HF model pages or the `/api/models` endpoint)

| Model | Params | Weights file | License | Training checkpoints | Notes |
|---|---|---|---|---|---|
| `EleutherAI/pythia-14m` | 14.07M (1.19M non-embedding) | `model.safetensors` 28.1 MB (fp16) | Apache-2.0 | **154** (step0, step1…512 log-spaced, step1000…143000) | Re-trained on the Pile at interpretability researchers' request. **Before 2026-02-27 this URL hosted the deduped model**, which moved to `pythia-14m-deduped` (the card says so; the repo was created 2026-02-24). Measured checkpoint interpolation is in §5 |
| `EleutherAI/pythia-31m` | 30.49M (4.74M non-emb); 6 layers, d=256, 8 heads | 61.0 MB | Apache-2.0 | 154 | Same schedule |
| `EleutherAI/pythia-70m(-deduped)` | 70.43M (18.92M non-emb); 6 layers, d=512, 8 heads | 166 MB | Apache-2.0 | 154 | Pretrained SAEs (SAELens `pythia-70m-deduped-*-sm`, SAEBench sweeps). All my weight-edit checks ran on it. Parallel residual, partial RoPE |
| `EleutherAI/pythia-160m` | 162,322,944 (85.06M non-emb); 12 layers, d=768, 12 heads. The HF safetensors count of 212.65M also includes 12 causal-mask buffers of 2048² | 375 MB | Apache-2.0 | 154 | Suite paper: Biderman et al., arXiv 2304.01373 (16 models, 70M–12B, "same order") |
| `openai-community/gpt2` | 124M | 548 MB (f32) | MIT | none | Largest SAE and transcoder ecosystem in SAELens |
| `roneneldan/TinyStories-{1M,3M,8M,28M,33M,1Layer-21M}` | by name | `pytorch_model.bin` 48.6 / 66.7 / 112 / 242 / 291 / 269 MB | 33M README body says "License: mit". HF metadata has no license tag (others **UNVERIFIED**) | none | GPT-Neo architecture; paper arXiv 2305.07759 |
| `NeelNanda/GELU_1L512W_C4_Code`, `…/Attn_Only_1L512W_C4_Code` | 1 layer, d=512 | `.pth` | MIT (GELU-1L); none tagged (Attn-only) | **yes**: `checkpoints/tokens_*.pth` | Toy models with training checkpoints |
| **TRM** `arcprize/trm_arc_prize_verification` | **6.82M network + 448.7M puzzle_emb** (my count, arc_v1 `step_518071`, fp32) | 1.82 GB (v1), 2.47 GB (v2) | MIT | final only | Jolicoeur-Martineau, arXiv 2510.04871 ("7M parameters"; 45% ARC-AGI-1 / 8% ARC-AGI-2 claimed). ARC Prize replication: 40% / 6.2%. Code: SamsungSAILMontreal/TinyRecursiveModels (MIT); training takes about 3 days on 4 H100s per README |
| **HRM** `sapientinc/HRM-checkpoint-ARC-2` | **27.26M network (H 13.63M + L 13.63M) + 535.5M puzzle_emb** (my count) | 2.25 GB | Code Apache-2.0 (sapientinc/HRM); the HF repo has no license tag | final only | Wang et al., arXiv 2506.21734. ARC Prize analysis (arcprize.org/blog/hrm-analysis, 2025-08-15): reproduced 32% ARC-AGI-1 semi-private and 2% ARC-AGI-2; a same-size transformer comes within about 5 pp; the outer refinement loop is the main driver; puzzle-ID embeddings make it transductive |

**Why the puzzle-embedding count matters for the ARC interface.** For TRM and HRM, "the weights of the ARC model" are 95–98.5% one learned 512-vector per augmented puzzle ID. A weight-space statement about ARC is therefore most naturally a statement about rows of `puzzle_emb`, for example:
* the embedding of task A ⊕ the embedding of task B solves the composition of A and B;
* the embeddings of the dihedral augmentations of one task are related by a fixed linear map.

The 6.8M core is small enough to analyse as a function on CPU. The TRM forward cost per ARC sample on CPU was measured by a sibling lens (`scripts/bench_trm_cpu.py`); I did not run it.

---

## 4. Grokking that trains in minutes on CPU, and what "steering grokking" means here

### 4.1 Recipes (verified sources)

* **Canonical: Nanda et al., arXiv 2301.05217 (full text on ar5iv).**
  * Task and data: P = 113, 30% of all pairs for training.
  * Model: a one-layer ReLU transformer, d = 128, learned positional embeddings, 4 heads of 32, MLP 512. "We did not use LayerNorm or tie our embed/unembed matrices."
  * Training: AdamW, lr 1e-3, weight decay 1, **full batch**, 40k epochs.
  * Phases: memorization 0–1.4k, circuit formation 1.4k–9.4k, cleanup 9.4k–14k. Generalization comes "after around 10,000 epochs".
  * The fetched text does not state the Adam betas. I used (0.9, 0.98) (**UNVERIFIED** as the paper's value).
  * My implementation is `ops_scripts/grok.py`: 226,688 parameters, input tokens (a, b, =), readout at "=". The forward computes only the readout position, which is mathematically identical for the loss.
* **Grokfast** (Lee et al., arXiv 2405.20233; github.com/ironjr/grokfast, MIT). Grokfast-EMA adds the low-pass-filtered gradient to the gradient: `g ← g + λ·EMA_α(g)`.
  * Recommended settings for the algorithmic task: EMA with α = 0.98, λ = 2.0, wd = 0.005 → **910** iterations to 95% val, vs a **39,890** baseline. MA with window 100, λ = 5, wd = 0.01 → 790.
  * The abstract claims ">50x".
  * Note: the repo's code defaults (α = 0.99, λ = 5) differ from the README table.
  * Their algorithmic setup (paper HTML v2) is **different from Nanda's**: modular *multiplication* mod 97, a 2-layer decoder with LayerNorm and GELU, d = 128, 4 heads, Adam (0.9, 0.98), lr 1e-3, **batch 512** (not full batch). The paper also reports that weight decay on its own, without Grokfast, made training unstable. Transferring λ = 2 to the full-batch, wd = 1 Nanda recipe is therefore an extrapolation. That transfer is exactly what my run #1 tests.
* **Other steering handles in the literature (abs pages opened unless marked):**
  * **Weight norm / initialization scale.** Omnigrok, arXiv 2210.01117 (the "LU mechanism"): it induces grokking on non-algorithmic data and eliminates it on algorithmic data.
  * **Softmax collapse.** Prieto et al., arXiv 2501.04697: StableMax "enables grokking without regularization", and ⊥Grad prevents naive loss minimization.
  * **Optimizer.** Muon, arXiv 2504.16041 (mean grokking epoch 153.09 → 102.89 vs AdamW; from a search-result summary, abs page not opened, so **UNVERIFIED**). Wang, arXiv 2607.20512 (2026-07-06): the speed-up comes from orthogonalization (Newton–Schulz); spectral-norm-only is no faster.
  * **Logit scale is the proximal variable.** Truong, arXiv 2606.18465 (2026-06-16): with the norm clamped and temperature varied, the delay "collapses onto the logit scale alone" (R² = 0.97); matching the logit scale recovers about 85% of the delay. Code: github.com/ClevixLab/grokking-logit-scale.
  * **Direction vs magnitude.** Truong, arXiv 2607.06628 (2026-07-07): implanting a donor's weight *direction* at the recipient's norm drives the run to the donor's circuit in 40/40 cases; an angle-matched random control gives no shift.
  * **Closed-form solution.** Gromov, arXiv 2301.02679 gives "analytic expressions for the weights" of a 2-layer MLP that solves modular arithmetic. These are ground-truth weights to compare against, and a target for "weight-space programming".

### 4.2 Cost on this machine (measured)

* **4 threads, uncontended: 43 ms/step** (300-step smoke test). That is about **7 min per 10k steps**.
* 2 threads: 64 ms/step. 1 thread: 97 ms/step.
* With 3–4 concurrent jobs, and sibling agents also using the CPU, about 170 ms/step per job.
* Memorization (train acc ≥ 0.99) happens by step 200 in every run.

So a single Nanda-recipe run, carried to generalization, fits the ≤10-minute budget only with the whole machine. Two ways to make it cheaper:
* Raise the train fraction. A sibling lens's p = 97 / frac = 0.5 runs (`feas_out/`) generalize by step 1,250–1,500 (base) and step 1,000 (Grokfast). That leaves less room to see a steering effect.
* Run seeds in parallel with `torch.func.stack_module_state` + `vmap`.

### 4.3 Results: is grokking steerable here? (measured, seed 0, p = 113, frac = 0.3)

All three runs share the same initialization (seed 0), the same data split, and the Nanda recipe above. Test-acc thresholds are checked every 100 steps (`ops_out/{base,grokfast,shrink}_s0.json`).

| Run | Weight-space formal statement | train acc ≥ 0.99 | test ≥ 0.5 | **test ≥ 0.95 (grok)** | test ≥ 0.99 | Speed-up vs base |
|---|---|---|---|---|---|---|
| base (AdamW, wd = 1) | `Dynamics modadd(113, .3)` | 200 | 9,500 | **9,900** | 10,300 | 1.00x |
| Grokfast-EMA (α = .98, λ = 2, wd = 1) | `… with grad_ema_amplify(.98, 2)` | 200 | 6,000 | **6,700** | 7,000 | **1.48x** |
| One-shot shrink θ ← 0.7·θ at step 1500 | `… with at(1500) scale(θ, .7)` | 200 | 8,300 | **8,900** | 9,200 | **1.11x** |

Test accuracy every 1k steps:

| Run | 1k | 2k | 3k | 4k | 5k | 6k | 7k | 8k | 9k | 10k |
|---|---|---|---|---|---|---|---|---|---|---|
| base | .039 | .041 | .044 | .052 | .057 | .067 | .081 | .116 | .241 | .979 |
| Grokfast | .122 | .143 | .164 | .151 | .253 | .545 | .992 | | | |
| shrink | .039 | .047 | .053 | .060 | .071 | .092 | .130 | .327 | .978 | |

**Reading.**
* **Grokking is steerable here.** The base run reproduces Nanda et al.'s "after around 10,000 epochs" (grok at 9,900).
* **Grokfast moves the transition 3,200 steps earlier (1.48x).** Its test accuracy already rises above base from step 1,000 (0.122 vs 0.039). That suggests the slow-gradient amplification speeds up circuit formation and does not just shorten cleanup.
* **The speed-up does not reach the ×44–50 the paper reports in its own setup.** That setup is mini-batch 512, wd 0.005 and modular multiplication. Here the recipe is full batch with wd = 1, where weight decay already supplies much of the "slow" drift. λ = 2 also caused a transient instability: train accuracy fell to 0.966 at step 4,000.
* **A single weight-space edit had a small effect.** Shrinking every parameter by 0.7 at the end of memorization moved grokking 1,000 steps earlier (1.11x). But the optimizer undid most of the norm change within about 500 steps: ‖θ‖ went 57.0 → 39.9 at the edit, 48.2 at step 1,600 and 52.3 at step 2,000. The effect is therefore at best weak, and with one seed it is **not distinguishable from seed noise** (seed spread not measured).
* **This is consistent with the 2026 picture.** Truong (2606.18465) found that the norm is an *upstream* handle and the logit scale is the proximal one, and (2607.06628) that norm has only a modest, distributed effect on the delay while the *direction* carries the circuit identity (paraphrased from the abstract). The next experiments follow from that:
  1. a direction implant from a grokked donor run;
  2. a temperature / logit-scale clamp instead of a norm shrink;
  3. 5–10 seeds per arm, which on a free 4-core machine is about 1 h per arm, or one B200 sweep (§4.4).

Wall clock under contention (3 runs in parallel at 1 thread each, plus sibling jobs): 0.14–0.16 s/step, about 21–25 min per run. On the whole machine with nothing else running, one run to grokking takes about **7 min** (43 ms/step).

### 4.4 What a B200 for ≤ 1 hour adds

Hardware (nvidia.com DGX B200 spec table): 8 GPUs, 1,440 GB HBM3e in total, 64 TB/s, FP8 72 PFLOPS sparse (36 dense). Per GPU that is about 180 GB, 8 TB/s and about 4.5 PFLOPS dense FP8. The per-GPU split is derived by me; BF16/FP32 figures are **UNVERIFIED**.

1. **Phase diagrams instead of single runs.** Train ~10³ grokking models at once with `stack_module_state` + `vmap`, over grids of:
   * Grokfast (α, λ);
   * shrink (c, t_edit);
   * wd;
   * direction-implant donors.

   Compute: one run is about 5.7 GFLOP/step × 10k steps, so 1,000 runs are about 5.7e16 FLOP. At an assumed effective 100 TFLOPS that is about 10 minutes (**estimate, UNVERIFIED**). This turns "steerable?" into a statement with confidence intervals.
2. **Scale-up of the weight-edit equivalences and steering to Pythia 1.4B–6.9B across many of the 154 checkpoints.** Example question: does a refusal-style or sentiment direction exist at step N, and can it be orthogonalized away? Download bandwidth, not FLOPs, is the limit: one checkpoint of a 6.9B model in fp16 is about 14 GB (**UNVERIFIED** size).
3. **Train your own SAEs or transcoders** with SAELens on GPT-2-small / Pythia-160M activations, so that "SAE direction" statements are about dictionaries you control. That it fits in an hour is **UNVERIFIED**.
4. **Reproduce contrastive weight steering at 7B** (two LoRA fine-tunes plus the difference; Fierro & Roger used Llama-2-7b-chat). Feasible in an hour only for small fine-tune sets (**UNVERIFIED**).
5. **Run TRM/HRM ARC evaluation under puzzle-embedding edits.** It is not CPU-feasible at evaluation scale, and retraining TRM takes about 3 days on 4 H100s, so it is out of reach in an hour.

---

## 5. Ten weight-space experiments: one formal statement each, CPU ≤ 10 min, informative for MI

The syntax follows Lens B's sketch (`formal.md` §5): Symmetry / Edit / Dynamics, plus **Equiv**, which says that a weight edit equals an activation intervention. **Nine of the ten were run here.** "Result here" is measured. CPU times were taken on a machine shared with sibling agents, so they are upper bounds. The ranking weighs:
* the information gained for mechanistic interpretability;
* how directly the experiment answers the owner's three examples (rotate, interpolate, steer grokking);
* how cheap it is.

| # | Formal statement | Model / setup | CPU cost | Result here (measured) | Why it matters for MI |
|---|---|---|---|---|---|
| 1 | `Dynamics modadd(113, frac=.3) with grad_ema_amplify(α=.98, λ=2) => t_grok ↓` | Nanda recipe, `grok.py --mode grokfast` vs `--mode base` | ≈ 7 min per 10k steps (4 threads) | **TRUE (modest)**: grok at step **6,700 vs 9,900** for base (1.48x). Test accuracy already ahead from step 1k (0.122 vs 0.039). Transient train-accuracy dip to 0.966 at step 4k. One seed | Directly answers "is grokking steerable?". Grokfast steers along the slow, i.e. generalizing, gradient component |
| 2 | `Symmetry rotate(resid, Q) all_layers ∧ embed ∧ unembed requires fold(ln) ∧ Q·1 = 1`, and `¬Symmetry rotate(resid, Q) layers={mid}` | Pythia-70M-deduped, f64, `resid_rotation.py` | 30 s | **TRUE** for the global, folded, Q·1 = 1 form: 1.5e-11. **FALSE** for: generic Q (13.8), no folding (1,232), middle layer only (366) | The owner's "rotate the middle layer" example, decided on a real LayerNorm model. It shows the checker's preconditions (fold, centring) and the scope rule (all readers and writers) |
| 3 | `Equiv orth(writers, r) ≡ hook(each writer: x ↦ (I − rrᵀ)x)` | Pythia-70M-deduped, f64, `weight_ops.py` | 5 s | **TRUE**: 2.68e-11 vs effect 4.53 | Arditi's §4.1 equivalence made executable. Every "ablate direction r everywhere" activation result compiles to a shareable checkpoint |
| 4 | `Edit θ0 + α·(Δ₊ − Δ₋)/2 ≻ act_add(L3, α·v) on held-out topics at equal Δloss` (contrastive weight steering vs activation steering) | Pythia-70M-deduped; two 60-step full fine-tunes on 96 own toy sentences each; `contrastive_ws.py` | 2.7 min (1 thread) | At Δloss ≈ +0.12–0.19: weight steering raises the held-out sentiment log-odds by **+7.5**, activation steering by **+4.6**. At Δloss ≈ +1.3–1.5: **+24.4** vs **+13.0**. That is about 2x more held-out shift per unit of capability loss (one seed, toy data, activation vector at one layer only) | A CPU-scale replication of the direction of Fierro & Roger's finding. Gives "weight-space steering" a number to argue about |
| 5 | `Dynamics modadd(113) with at(t=1500) scale(θ, 0.7) => t_grok ?` (a one-shot weight-space edit at the memorization checkpoint) | `grok.py --mode shrink` | as #1 | Grok at **8,900 vs 9,900** (1.11x). The optimizer restores ‖θ‖ within about 500 steps (39.9 → 52.3). One seed, so within unmeasured seed noise: **inconclusive, weak at best** | Asks whether a *single* weight edit, rather than a changed optimizer, moves the transition. Links the Omnigrok norm and Truong's logit scale to an Edit statement |
| 6 | `Edit orth(writers≤3, W_dec[i]) ≢ ablate_feature(SAE_3, i)` | Pythia-70M-deduped + SAELens `pythia-70m-deduped-res-sm`, `tool_probe.py` | 15 s (+134 MB) | **TRUE (they differ)**. Projecting out the direction: loss 3.877 → 6.975 (the feature goes to 0; residual component 1e-5). Ablating the feature only where it fires: 3.877 → 3.877 | A type rule for the DSL: a *direction* is linear and static (a weight edit); a *feature* is nonlinear and gated by the input (it needs the encoder) |
| 7 | `Symmetry permute(mlp(2).hidden)` and `¬Symmetry rotate(mlp(2).hidden)` | Pythia-70M-deduped, f64 | 20 s | Permutation **3.6e-11**. Rotation: max\|Δlogit\| 249.8, loss 4.66 → 9.85 | GELU privileges the neuron basis. This is the canonical "is this a gauge or a real change?" test |
| 8 | `Edit line(θ_step k → θ_step143000)` => barrier(k) | Pythia-14M, `functional_call`, `ckpt_line.py` | about 4 s per pair (+28 MB per checkpoint) | Barrier in nats: k = 1k **3.46**, 3k **3.77**, 10k **3.68**, 30k **3.94**, 60k **1.96**, 100k **0.05**, 120k / 133k **0.00** | Linear mode connectivity *along one run*. Pythia-14M enters its final linearly connected basin between steps 60k and 100k (one 112-token text, so a weak estimate). Bisect over the 154 checkpoints |
| 9 | `Edit swap(L2, L3)`; `Edit avg(L2, L3)`; `Edit delete(L3)` => Δloss | Pythia-70M-deduped, `weight_ops.py` | 5 s | Δloss: swap **+2.17**, average **+1.14**, delete **+0.83** nats | Tests the owner's "middle layers interpolate" intuition at small scale. In a 6-layer model the middle layers are *not* interchangeable (Painters' uniformity was reported for larger models). Rerun on Pythia-160M/410M to find where uniformity begins |
| 10 | `Equiv add(resid_post(3), v) ≡ bias(attn(3).dense) += v` | Pythia-70M-deduped, f64 | 4 s | **TRUE**: 2.71e-11 vs effect 8.87 | All-position steering vectors (CAA, persona vectors) become weight statements; the ActAdd first-k-positions variant does not. With #3 it makes the activation→weight compiler complete for linear, position-independent interventions |

**Also run, but ranked below the ten:**
* `Symmetry head(1,0).vo by N ∈ GL(64)`: **6.0e-11** (cond(N) = 161). The OV product is the object, not W_V or W_O alone.
* `Edit ablate head(l,h) ∀ l,h`: the top heads are L0H1 **+0.255**, L0H4 +0.089 and L2H5 +0.080 nats; 48 heads in 37 s.
* `functional_call` along a random direction, scaled by the per-tensor std: Δloss 0.32 / 2.99 / 21.9 at t = 0.01 / 0.03 / 0.1. This is a sharpness baseline for any "edit size" claim.

**Not run, but feasible:** the ARC bridge on TRM, `Edit puzzle_emb[A] := slerp(puzzle_emb[A], puzzle_emb[D4·A])`. TRM has 6.8M core parameters, and a sibling lens benchmarked its CPU forward. Whether a full ARC evaluation fits in 10 CPU minutes is **UNVERIFIED**.

---

## 6. Implications for the relay and the DSL

1. **Give the checker three verdict types that need no LLM judgment:**
   * `Equiv` (f64, ≤ 1e-9 relative to the effect size);
   * `Symmetry` (the same, against the unedited model);
   * `Edit` / `Dynamics` (an effect size with a seed spread).

   All three run on this CPU with the stack above. The SAE case (§5 #6) shows why "direction" and "feature" need to be separate types.
2. **Pin library versions in the prompt context.** TransformerLens 4 removed `HookedTransformer`. nnsight 0.8 deprecates `LanguageModel`, requires accesses in execution order, and requires the `with` body to start on its own line. The relay models' cutoffs predate these changes. Add the installed API surface (`dir(transformer_lens)` shows `TransformerBridge`, `FactoredMatrix`, `SVDInterpreter`) to the system prompt, or pin `transformer-lens<4`.
3. **Freshness routing.** Fable is freshest on ML literature (arXiv to about 2026-04), so it should translate papers into statements. But the most relevant work is beyond every model's cutoff:
   * Weight-Space Programming, Nov 2026;
   * the grokking logit-scale and direction-implant papers, Jun/Jul 2026;
   * TL 4.0, Sep 2026.

   A retrieval step that injects these abstracts is needed.

4. **Status tags must carry the setup.** Grokfast's ×44–50 became 1.48x when transferred to the Nanda recipe (§4.3). A statement like `grad_ema_amplify => t_grok ↓` must be stored as `EMPIRICAL(ref, task, batch, wd, seeds)`. A relay model citing a paper's headline number is not evidence for a different setup, and the CPU check is cheap enough to run every time.
5. **ARC interface.** For TRM/HRM, address `puzzle_emb[task_id]` (512-d, one per augmented task) and the small core separately. Statements about "the ARC model's weights" are almost entirely statements about those rows (98.5% / 95% of the parameters).

---

## 7. Sources opened in this session

* **Steering and editing:**
  * arXiv 2406.11717 (abs + ar5iv §4.1); github.com/andyrdt/refusal_direction (Apache-2.0)
  * arXiv 2511.05408; github.com/safety-research/weight-steering
  * arXiv 2308.10248; 2312.06681; 2310.01405; 2404.03592; 2106.09685; 2212.04089; 2202.05262; 2507.21509
* **Weight-native interpretability:**
  * arXiv 2406.11944; transformer-circuits.pub/2024/crosscoders
  * arXiv 2506.20790; 2511.08854
  * arXiv 2511.13653; github.com/openai/circuit_sparsity; arXiv 2607.02964
  * arXiv 2309.08600; 2401.15024; 2407.09298
* **Weight-space programming:** arXiv 2611.09382; 2611.09376
* **Grokking:**
  * arXiv 2301.05217 (abs + ar5iv); 2201.02177; 2210.01117; 2501.04697; 2301.02679
  * arXiv 2405.20233; github.com/ironjr/grokfast
  * arXiv 2504.16041 (search result only); 2606.18465; 2607.06628; 2607.20512
* **Models:**
  * HF pages and API: EleutherAI/pythia-{14m, 14m-deduped, 31m, 70m, 70m-deduped, 160m}, openai-community/gpt2, roneneldan/TinyStories-* (incl. raw README of 33M), NeelNanda/GELU_1L512W_C4_Code, NeelNanda/Attn_Only_1L512W_C4_Code, arcprize/trm_arc_prize_verification, sapientinc/HRM-checkpoint-ARC-2
  * arXiv 2304.01373; 2510.04871; 2506.21734
  * github.com/SamsungSAILMontreal/TinyRecursiveModels; github.com/sapientinc/HRM; arcprize.org/blog/hrm-analysis
* **Tools:**
  * github.com/TransformerLensOrg/TransformerLens; github.com/ndif-team/nnsight; github.com/stanfordnlp/pyvene; github.com/decoderesearch/SAELens
  * PyPI JSON for transformer-lens, nnsight, pyvene, pyreft, sae-lens, torch, transformers
  * docs.pytorch.org/docs/2.15/func.api.html
* **Hardware:** nvidia.com/en-gb/data-center/dgx-b200 (spec table)
