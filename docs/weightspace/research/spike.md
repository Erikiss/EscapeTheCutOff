# Feasibility spike: weight-space statements on a CPU box

Date 2026-11-25. Machine: 4 CPUs, 15 GB RAM, no GPU, Python 3.13.18, PyTorch 2.15.0+cpu.
Every number below was measured on this machine by the scripts in
`/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/feas_scripts/`. Raw outputs are in
`/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl/feas_out/`.

**Contention warning.** A sibling agent ran its own grokking jobs (`ops_scripts/grok.py`, 4 threads) at the same time
(load average 6.6 to 7.6 on 4 cores). So all wall-clock times below are pessimistic, roughly 2x what an idle box
would give. Step counts are not affected.

Base dir used below: `W=/tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad/wsdsl`.

---

## 0. Environment (task 1)

`$W/venv` already existed (made at 08:34 by a sibling agent: `python3 -m venv`, torch 2.15.0+cpu, numpy 2.5.5).
I reused it and added packages to it. To get an honest **cold install time**, I installed the same stack into a
throwaway venv without the pip cache and then deleted it:

```bash
python3 -m venv $W/venv_timing
$W/venv_timing/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
$W/venv_timing/bin/pip install --no-cache-dir numpy safetensors huggingface_hub matplotlib
```

| step | wall time | size |
|---|---|---|
| `python3 -m venv` | 2.1 s | |
| torch 2.15.0+cpu (`--index-url .../whl/cpu`) | 15.6 s | torch package 747 MB |
| numpy 2.5.5, safetensors 0.8.0, huggingface_hub 2.4.0, matplotlib 3.11.2 | 6.4 s | |
| **total cold install** | **24.2 s** | **966 MB** venv |

Additions to the shared `$W/venv`: safetensors, huggingface_hub, matplotlib (5.6 s); tokenizers (1.0 s);
transformers 5.24.1 + scipy 1.18.1 (4.9 s). transformers is used **only** as a reference check of my hand-written
forward pass, and scipy only for Hungarian matching. Final `$W/venv` size: **1.3 GB**. My own disk footprint is about
1.35 GB in total (venv plus 29 MB of models plus outputs), which is under the 3 GB budget.

---

## 1. pythia-14m: weight-space statements, verified numerically (task 2)

```bash
$W/venv/bin/python -I -c "from huggingface_hub import snapshot_download; snapshot_download('EleutherAI/pythia-14m',
  revision='cf967c0a9a04383db6f7b1108d86b2962634b4ac', allow_patterns=['config.json','model.safetensors',
  'tokenizer.json','tokenizer_config.json','special_tokens_map.json'], local_dir='$W/models/pythia-14m')"
$W/venv/bin/python -I $W/feas_scripts/pythia_ws.py $W/models/pythia-14m $W/feas_out/pythia_ws.json --check-hf
```

* Download: 2.5 s. `model.safetensors` is 28,143,920 bytes (fp16, 76 tensors). The config was verified from the
  file: GPT-NeoX, d=128, 6 layers, 4 heads of 32, d_mlp=512, `rotary_pct=0.25` (so only **8 of 32** head dims are
  rotary), `use_parallel_residual=true`, exact GELU, LayerNorm eps 1e-5, untied unembedding **without bias**.
* `pythia_ws.py` is a **hand-written GPT-NeoX forward pass in float64** over the raw safetensors dict, so every
  weight-space statement is literally a function `params -> params`.
* Text: 149 tokens of my own prose plus a 3-line Python snippet. **Base loss 4.3177 nats/token** (uniform: 10.83).
  One forward pass takes 103 ms. The whole table below (29 statements) runs in **13.7 s**, including the transformers reference load.
* Reference check against `transformers` 5.24.1 (`GPTNeoXForCausalLM`, float64): max |Δlogits| = 8.3e-6. I traced
  the gap to HF building its RoPE cos/sin tables in float32. With float32 tables my forward matches HF to
  **1.8e-11** (`feas_scripts/hf_diff.py`). Max |logit| is 25.8.

### 1a. Results (Δ against the unedited model; float64)

"≈0" means a max |Δlogits| of about 2e-11, i.e. about 1e-12 relative, which is the float64 round-off floor of this
forward pass.

| id | weight-space statement | max \|Δlogits\| | loss | Δloss |
|---|---|---|---|---|
| **Exact symmetries (no reparametrisation needed)** |||||
| S1 | middle layer 3: permute the 512 MLP neurons (rows of `W_in`, `b_in`, columns of `W_out`) | 2.7e-11 | 4.3177 | 0.0000 |
| S1b | same, an independent permutation in every layer | 2.4e-11 | 4.3177 | 0.0000 |
| S2 | layer 3: permute the 4 heads (fused QKV row blocks + `dense` column blocks) | 2.9e-11 | 4.3177 | 0.0000 |
| S3 | layer 3, head 0: `V := N V`, `O := O N⁻¹`, N random in GL(32) | 2.1e-11 | 4.3177 | 0.0000 |
| S4 | layer 3, head 0: `Q := A Q`, `K := A⁻ᵀ K` on the **24 non-rotary dims** only, A in GL(24) | 2.5e-11 | 4.3177 | 0.0000 |
| S4b | layer 3, head 0: the same 2x2 rotation φⱼ on each RoPE pair (j, j+4) of Q **and** K, plus `Q·s`, `K/s` | 2.6e-11 | 4.3177 | 0.0000 |
| **Residual-stream rotations (need a canonical form first)** |||||
| S5a | `fold_ln`: move every LN γ/β into the following linear map. This needs a **new** `embed_out.bias = W_U β_final`, because NeoX's unembedding has no bias | 2.4e-11 | 4.3177 | 0.0000 |
| S5b | then centre every residual writer (embedding, `attention.dense`, `mlp.dense_4h_to_h`, and their biases) with `C = I − 11ᵀ/d` | 2.2e-11 | 4.3177 | 0.0000 |
| S5c | after fold_ln: `resid := R resid` with **R in O(128) and R·1 = 1** (the stabiliser of the mean direction, ≅ O(127)). LN is kept | 2.0e-11 | 4.3177 | 0.0000 |
| S5d | after centring: replace LayerNorm with RMSNorm (no rotation) | 2.2e-11 | 4.3177 | 0.0000 |
| S5e | centred + RMSNorm: `resid := R resid` with **any R in O(128)** (SliceGPT-style) | 2.0e-11 | 4.3177 | 0.0000 |
| **Broken or only approximate symmetries** |||||
| B1 | scale the whole residual stream by 2 (embedding and all writers). LN is scale-invariant **except for eps** | 0.45 | 4.3178 | +0.0002 |
| B2 | rotate the residual stream **without** folding γ/β (γ is elementwise, so it is basis-dependent) | 22.6 | 7.3240 | +3.006 |
| B3 | after folding, R in O(128) that does **not** fix the ones vector, with LN kept (the centring breaks it) | 31.3 | 5.0609 | +0.743 |
| B4 | layer 3: positive per-neuron rescale, c in [0.5, 2] (`W_in·c`, `W_out/c`). Exact for ReLU, **not for GELU** | 6.5 | 4.4474 | +0.130 |
| B5 | layer 3: rotate the MLP hidden units with an orthogonal 512x512 Q | 23.4 | 8.5588 | +4.241 |
| B6 | layer 3, head 0: GL(32) on Q/K including the 8 rotary dims (RoPE breaks it) | 11.9 | 4.5909 | +0.273 |
| **Real edits (the owner's examples)** |||||
| E0 | delete the middle layer 3 (zero its writers) | 25.1 | 8.3606 | +4.043 |
| E1 | swap layers 2 and 3 | 23.9 | 7.5924 | +3.275 |
| E2 | layers 2 and 3 := (layer2 + layer3)/2, naive and unaligned | 23.9 | 7.2761 | +2.958 |
| E2s | sanity check: layer 3 permuted into alignment with layer 2 (heads by brute force over S₄, neurons by Hungarian matching) | 2.7e-11 | 4.3177 | 0.0000 |
| E2a | layers 2 and 3 := (layer2 + align(layer3))/2 | 22.8 | 6.9855 | +2.668 |
| E2r | "all weights in the middle layer rotate": only layer 3's readers get `W Rᵀ` and its writers get `R W` | 31.0 | 10.6870 | +6.369 |
| E3 | **antisymmetric pair**: layer 4 := −layer 2 as a residual update (copy layer 2, negate its writer weights and biases) | 24.1 | 6.8283 | +2.511 |
| E3ref4 | reference: delete layer 4 only | 24.0 | 6.4754 | +2.158 |
| E3ref | reference: delete layers 2 and 4 (what "the pair cancels" would predict) | 27.7 | 8.3816 | +4.064 |
| E3n | naive antisymmetry: layer 4 := −(every non-LN parameter of layer 2) | 28.8 | 10.6098 | +6.292 |
| E4 | "interpolates in between": layer 3 := (layer2 + layer4)/2 | 27.8 | 9.3100 | +4.992 |
| E3ctl | control: layer 4 plus a random direction with the same ‖Δθ‖ as E3 | 46.0 | 20.3196 | +16.002 |

### 1b. Which residual-stream symmetry holds for GPT-NeoX with LayerNorm

`LN(x) = γ ⊙ (x − mean(x)·1)/sqrt(var(x) + eps) + β`, with parallel residual `x ← x + Attn(LN₁ x) + MLP(LN₂ x)`.

1. **Stock parameters:** no non-trivial residual rotation is an exact symmetry. γ and β are elementwise, so they
   pin a basis (B2: Δloss +3.0).
2. **After `fold_ln`** (γ into the reader weights, β into the reader biases, plus a new unembedding bias): LN is
   pure centre-and-normalise. A rotation commutes with it **iff R maps the all-ones vector to ±itself**, so the
   symmetry group is O(127) × O(1), not O(128) (S5c exact, B3 broken: +0.74). eps does no harm because R preserves
   the variance.
3. **After also centring the writers** (exact under LN, since every reader of the stream is an LN, which ignores
   the mean component) **and swapping LN for RMSNorm**, which is an architecture change, **all of O(128)** is
   exact (S5d, S5e). This is SliceGPT's "computational invariance" construction (arXiv:2401.15024, §3.1–3.2,
   checked in the HTML version: the output matrices are multiplied by `M = I − 11ᵀ/D`, the embedding is
   mean-subtracted, and LN becomes RMSNorm up to √D).
4. The **scale** symmetry `x → c·x` is broken only by eps (B1: max |Δlogit| 0.45, Δloss +0.0002). This is an
   "approximate gauge".
5. Inside a block, the exact gauges are: neuron permutations S₅₁₂ (**not** positive rescaling, because GELU is not
   positively homogeneous: B4), head permutations S₄, V/O GL(32) per head, and Q/K GL(24) on the non-rotary dims.
   On the 8 RoPE dims, only the commutant of RoPE survives: one rotation per frequency pair, plus a global Q/K scale
   (S4b exact, B6 broken).

### 1c. What the edits say about the owner's examples

* **"All weights in the middle layer rotate"** has two readings that a DSL must keep apart. As a *gauge* (S3, S4,
  S4b, or S5 applied to all layers) it is a no-op. As a rotation of the middle layer's interface alone (E2r) it is
  the most destructive edit in the table (+6.37), worse than deleting the layer (+4.04).
* **The antisymmetric pair does not cancel and does not "interpolate".** Set layer 4 := −layer 2 functionally. If
  the residual updates were small, `x + f₂(x) + f₃ − f₂(x + f₂ + f₃) ≈ x + f₃ + O(J·f)`, i.e. the result would look
  like deleting both layers. Measured: the residual stream after layer 4 differs from the "delete 2 and 4" stream by
  **104 %** (relative norm), and the loss (+2.51) is between "delete layer 4 only" (+2.16) and "delete both" (+4.06).
  So the first-order argument fails: in a 6-layer model the layers are not small perturbations. The edit is still
  *structured*: a random edit of the same parameter norm costs +16.0.
* **Interpolation** `layer3 := (layer2 + layer4)/2` costs +4.99, which is worse than deleting layer 3 (+4.04).
  Adjacent layers do not lie on a line in weight space.
* **Averaging needs a quotient.** "Average layers 2 and 3" is only defined up to the permutation gauge. Aligning
  first (E2a) lowers the cost from +2.96 to +2.67, but the parameter distance barely moves (410.3 → 409.2). Two
  *different* layers are not permuted copies of each other, unlike two runs of the *same* layer (re-basin). Every
  DSL statement that compares or mixes weights must therefore name its gauge-fixing step.

---

## 2. Grokking and Grokfast on CPU (task 3)

**Setup.** I used the mainline architecture of Nanda et al. (arXiv:2301.05217 v3; I checked the setup text in the
PDF): one-layer ReLU transformer, d=128, learned positional embeddings, 4 heads of 32, 512 MLP units, no LayerNorm,
untied embed/unembed, full-batch AdamW, lr 1e-3, weight decay 1. **Deviations:** p=97 instead of 113 and Adam
betas (0.9, 0.98), both my choice. CPU contention made p=113 / 30 % (about 10k+ steps to grok) too slow for an
8-minute budget, so I ran two train fractions. The logits are read only at the "=" position.
Grokfast-EMA follows `gradfilter_ema` from `ironjr/grokfast` (`grokfast.py`, defaults α=0.98, λ=2.0, update
`g ← g + λ·EMA_α(g)`; I checked the source page). Paper: arXiv:2405.20233 (Lee et al., 2024), whose abstract claims
"more than ×50" acceleration. Both runs of a pair ran **concurrently, 1 thread each, 480 s wall budget each, seed 0,
same data split**, with evaluation on the full train and test sets every 50 steps.

```bash
for M in base grokfast; do for F in 0.5 0.3; do
  $W/venv/bin/python -I $W/feas_scripts/grok_spike.py --mode $M --p 97 --frac $F --budget 480 --threads 1 \
     --out $W/feas_out/grok_${M}_p97_f${F/./}.json &      # each pair launched together
done; done
$W/venv/bin/python -I $W/feas_scripts/plot_grok.py $W/feas_out/grok_curves.png <the 4 JSONs>
```

| config | run | train acc ≥ 99 % | test ≥ 50 % | **test ≥ 90 %** | test ≥ 99 % | steps in 480 s | final test acc |
|---|---|---|---|---|---|---|---|
| p=97, frac 0.5 (4704 train / 4705 test) | baseline | step 150 (21 s) | 550 | **1250 (181 s)** | 1500 (223 s) | 3350 (6.9/s) | 1.000 |
| | Grokfast-EMA | 200 (25 s) | 200 | **1000 (138 s)** | 1200 (173 s) | 3200 (6.7/s) | 1.000 |
| p=97, frac 0.3 (2822 train / 6587 test) | baseline | 150 (10 s) | – | **not reached** | – | 6350 (13.2/s) | 0.208 |
| | Grokfast-EMA | 150 (13 s) | – | **not reached** | – | 5700 (11.8/s) | 0.218 |

Curves: `feas_out/grok_curves.png`. JSON per run (step, wall time, train/test loss and accuracy, weight norm):
`feas_out/grok_{base,grokfast}_p97_f{05,03}.json`.

**Reading.**
* **Grokking happens at frac 0.5.** Training accuracy is 100 % at step 150. Test accuracy then sits on a plateau of
  **0.488–0.517** (steps 200–700), then rises from about step 750 and reaches 0.907 at step 1250 and 0.993 at step 1500. The plateau has a clean explanation,
  which I checked: **49.3 %** of the test pairs (a,b) have their swap (b,a) in the training set, so the plateau is
  "memorisation plus commutativity". Weight norm: 56.7 at memorisation, 52.7 at step 1000, 35.1 at the end, which
  matches the cleanup phase.
* **Grokfast speeds this up only modestly here: 1.25× in steps** (90 % at step 1000 vs 1250; 99 % at 1200 vs 1500).
  That is far from the paper's ">×50". Its effect is mostly *earlier* (test 0.5 at step 200 vs 550). A plausible
  cause, not tested: with weight decay 1.0, AdamW's per-coordinate normalisation absorbs much of the
  λ-amplification. The paper's headline setting differs from this one.
* **frac 0.3: no run reaches 90 % within 8 minutes.** Test accuracy reaches only 0.21–0.22 after about 6k steps,
  below the 0.295 commutativity floor (I checked: 29.5 % of the test pairs have their swap in train). This fits
  Nanda's 40k-epoch schedule at 30 %. The crossing step is therefore **> 6350 (baseline) and > 5700 (Grokfast)**,
  so I cannot measure an acceleration here within budget. In the Grokfast run, train accuracy dips once to 0.943,
  with train loss 0.96, at step 3800 (a slingshot-like spike). The baseline shows no such dip. My estimate, not
  measured: a 20–30k-step run takes about 15–30 minutes on an idle 4-core box, so this is feasible but outside the
  spike's budget. A sibling agent is running p=113 / 30 % with 20–25k steps (`$W/ops_out/`).
* **CPU throughput:** 7–13 full-batch steps/s per single-thread run under contention. Each step is a full pass over
  2.8–4.7k examples. Grokking experiments of this size are therefore CPU-feasible, with turnaround measured in
  minutes, not hours.

### 2b. Extra: "steering along grokking" in weight space (beyond the brief, about 6 min)

Grokfast steers in *gradient* space. The owner's question ("what happens if I steer along grokking?") can also be
asked directly in *weight* space. I trained two seeds (A=0, B=1) on p=97 / frac 0.5, saved θ at step 200
(memorised: train 1.0, test 0.488 / 0.485) and at step 2000 (grokked: test 1.0 / 1.0), and evaluated three
weight-space statements:

```bash
$W/venv/bin/python -I $W/feas_scripts/steer_grok.py train --seed 0 --steps 2000 --save 200,2000 --out $W/feas_out/steer/seed0 &
$W/venv/bin/python -I $W/feas_scripts/steer_grok.py train --seed 1 --steps 2000 --save 200,2000 --out $W/feas_out/steer/seed1 &
$W/venv/bin/python -I $W/feas_scripts/steer_grok.py analyse --a $W/feas_out/steer/seed0 --b $W/feas_out/steer/seed1 \
    --mem 200 --end 2000 --out $W/feas_out/steer/steer_results.json
```

Test accuracy (train accuracy in parentheses):

| statement | t = −0.5 | 0 | 0.25 | 0.5 | 0.75 | 1 | 1.25 | 1.5 | 2 |
|---|---|---|---|---|---|---|---|---|---|
| P1 `θ_A(mem) + t·(θ_A(grok) − θ_A(mem))` | .106 (.295) | .488 (1.0) | .370 (.808) | .515 (.821) | **.969 (.997)** | 1.0 (1.0) | **.999 (1.0)** | .727 (.726) | .205 (.189) |
| P2 `θ_B(mem) + t·(θ_A(grok) − θ_A(mem))` (direction transplanted from another seed) | .091 | .485 | .258 | .088 | .042 | **.024** | .017 | .015 | .011 |
| P2b `θ_B(mem) + t·(θ_B(grok) − θ_B(mem))` (B's own direction) | .037 | .485 | .313 | .384 | .806 | 1.0 | .259 | .025 | .011 |

| P3 `c · θ_A(mem)` (pure weight-norm shrink) | c = 1.0 | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 |
|---|---|---|---|---|---|---|
| test (train) | .488 (1.0) | .470 (.999) | .308 (.817) | .115 (.401) | .036 (.143) | .013 (.042) |

cos(dir_A, dir_B) = **−0.001**. cos(dir_A, −θ_A(mem)) = **0.739**. ‖dir_A‖ = 38.3.

* **Within one run, grokking is linearly steerable but not linearly connected.** The chord from the memorising to
  the grokked point first crosses a barrier (train accuracy drops to 0.81 at t = 0.25–0.5). From t = 0.75 it
  generalises (0.969), and extrapolating a little past the end point (t = 1.25) still gives 0.999.
* **The grokking direction is seed-specific.** The directions of two seeds are orthogonal (cos −0.001), and
  transplanting A's direction into B destroys B (test 0.024 at t=1). A shared "grokking direction" would only make
  sense *modulo the symmetry group* (align first, as in E2a). That is the same lesson as §1c.
* **Grokking is not just shrinking.** 74 % of the direction (by cosine) points toward the origin, i.e. weight decay.
  But shrinking alone (P3) only destroys memorisation, and the generalising circuit lives in the remaining
  component.

---

## 3. TRM / HRM ARC checkpoints: metadata only (task 4)

Command (no weight downloads; only `HfApi.list_models` / `model_info(files_metadata=True)` plus two config YAMLs of
about 1 KB each):

```bash
$W/venv/bin/python -I $W/feas_scripts/hf_meta.py $W/feas_out/trm_hrm_meta.json   # 32 candidate repos, full listing in feas_out/trm_hrm_meta.log
$W/venv/bin/python -I -c "from huggingface_hub import hf_hub_download as h; \
  h('arcprize/trm_arc_prize_verification','arc_v1_public/all_config.yaml',revision='55ced5dd59de74c52f53d47aa2898232b5a15b7a',local_dir='$W/feas_dl_trm_cfg'); \
  h('sapientinc/HRM-checkpoint-ARC-2','all_config.yaml',revision='ee2c595e8b9dd061a448f6f65fca997bd2227c74',local_dir='$W/feas_dl_hrm_cfg')"
```

| repo (HF, verified via API) | sha / last modified | files (bytes) | licence (card) | provenance |
|---|---|---|---|---|
| `arcprize/trm_arc_prize_verification` | 55ced5dd, 2025-10-16 | `arc_v1_public/step_518071` **1,822,205,258**; `arc_v2_public/step_723914` **2,467,988,810**; plus `trm.py`, `losses.py`, `all_config.yaml` (total 4.29 GB) | MIT | ARC Prize's re-training for score verification (card: 40 % ARC-AGI-1, 6.2 % ARC-AGI-2) |
| `sapientinc/HRM-checkpoint-ARC-2` | ee2c595e, 2025-07-21 | `checkpoint` **2,250,983,821**, `all_config.yaml` | none (the page says "No model card") | official HRM authors (Sapient) |
| `sapientinc/HRM-checkpoint-maze-30x30-hard`, `-sudoku-extreme` | 2025-07-21 | `checkpoint` 109,103,458 each | none | official, non-ARC (no puzzle table → shows the core net size) |
| `gaoxin492/TinyRecursiveModels-ARC-AGI-1` / `-2` | 2025-10-15 | `step_155718` 1,822,205,258 / `step_217602` 2,467,988,810 | apache-2.0 | community |
| `alphaXiv/trm-model-arc-agi-1` | 2025-10-22 | two 1,822,205,258-byte checkpoints (attention variants) | MIT | community |
| `Trelis/TRM-ARC-AGI-II` | 2025-10-17 | 11 checkpoints of 2,467,988,810 bytes (steps 100 to 723,914) plus per-step `submission.json` (27.2 GB) | apache-2.0 | community **training trajectory** |
| `Trelis/TRM-pretrain_arc2_rearc_200k` | 2025-11-19 | 107 files, 79.7 GB including the built dataset (`identifiers.json`, `all__inputs.npy`, ...) | MIT | community |

The official TRM GitHub repo `SamsungSAILMontreal/TinyRecursiveModels` is MIT and was **archived on 2026-04-01**.
Its README mentions no released weights and assumes 4×H100 for about 3 days per ARC run (checked on the GitHub
page). TRM paper: arXiv:2510.04871, "7M parameters, 45 % ARC-AGI-1 / 8 % ARC-AGI-2" (abstract). HRM paper:
arXiv:2506.21734, "27 million parameters", 1,000 training samples (abstract).

**Configs (read from the YAMLs).** TRM `arc_v1_public`: hidden 512, 8 heads, `L_layers 2`, `H_layers 0`,
`H_cycles 3`, `L_cycles 4`, `halt_max_steps 16`, `puzzle_emb_len 16`, `puzzle_emb_ndim 512`, RoPE, `forward_dtype
bfloat16`, data `arc1concept-aug-1000`. HRM ARC-2: hidden 512, 8 heads, `H_layers 4`, `L_layers 4`, `H_cycles 2`,
`L_cycles 2`, `halt_max_steps 16`, `puzzle_emb_ndim 512`.

**Why the files are GBs while the nets are tiny (my arithmetic from the byte counts).** The HRM non-ARC checkpoints
are 109.1 MB, i.e. 27M params × 4 B in fp32. So the ARC-2 HRM checkpoint (2.25 GB) is about **95 % puzzle-embedding
table** (about 2.14 GB, roughly 1.05M rows × 512 fp32). For TRM-ARC-1, 1.82 GB ≈ 6.8M × 4 B (27 MB core net) plus
about 1.80 GB of embedding table. A sibling agent loaded this checkpoint and reports 6,830,082 core params plus
876,406 × 512 embeddings, consistent with this (`$W/arc.md`). **The task identity lives in the embedding rows, not
in the net.** For TRM, each (task, augmentation) is a 16×512 block.

**CPU plausibility (estimate, not measured by me).** TRM core net: 2 layers × about 3.4M params, sequence 916
(900 grid + 16 puzzle tokens). Per net pass, about 2 × 6.8M × 916 ≈ 12.5 GFLOP for matmuls plus about 3.4 GFLOP for
attention, so about 16 GFLOP. Per ACT step: 3 H-cycles × (4 latent + 1 answer) = 15 passes ≈ 240 GFLOP. Times 16
ACT steps: **about 3.8 TFLOP per sample**. HRM: (2×2 L + 2 H) module passes × 4 layers ≈ 190 GFLOP per ACT step, so
**about 3 TFLOP per sample**. I measured one fp32 matmul core at about 88 GFLOP/s here (`feas_scripts/prof.py`,
under contention), which predicts about 10–45 s per sample. The sibling agent measured TRM at 11.7 s fp32 / 2.8 s bf16
for one sample at 16 ACT steps (`$W/arc.md`). **Verdict: CPU inference is plausible for single samples and for
weight-space probes.** Load needs about 2–2.5 GB RAM in fp32, which fits in 15 GB. The full ARC protocol
(about 1000 augmentations per task, about 3 h per task in fp32) and any test-time training are **not** feasible on
this box. Both checkpoints are already on disk from sibling agents (`$W/dl_trm_ckpt` 1.7 GB, `$W/dl_hrm_ckpt`
2.1 GB), so nobody needs to download them again. I did not open them (untrusted pickles; load with
`weights_only=True` and `python -I`).

---

## 4. Consequences for a weight-space DSL ("weight-space instructions")

What the spike shows is feasible today on this CPU box, and what a statement language therefore needs:

1. **The language can operate directly on weights.** Every statement in §1 and §2b is a pure function
   `state_dict → state_dict` in about 5–20 lines of torch. Checking one statement on a ≤ 30M-parameter net takes
   **0.1–1 s** (pythia-14m: 29 statements in 13.7 s, including a 512×512 Hungarian match). Executing statements
   needs no GPU.
2. **Statements need types**, because the same English sentence ("the middle layer rotates") denotes completely
   different things:
   * `gauge`: claimed to be an exact symmetry. Check: max |Δlogits| below about 1e-9 (float64). It fails loudly
     when false (B2–B6: 0.27 to 4.2 nats).
   * `reparam`: a function-preserving change of *parametrisation class* (`fold_ln` adds a parameter, `ln→rms`
     changes the architecture). It must be explicit, because some gauges only exist after it (S5c/S5e vs B2/B3).
   * `approx_gauge(tol)`: e.g. residual scaling, which LN eps breaks (B1).
   * `edit`: semantics is the measured Δloss / Δlogits / residual-stream difference on a probe set, always with a
     random direction of equal ‖Δθ‖ as control (E3 vs E3ctl: +2.5 vs +16.0).
   * `path` / `traj`: statements about θ(t) or between checkpoints (P1–P3). These need saved checkpoints or a
     published trajectory, such as Trelis' TRM checkpoints at many training steps.
3. **Comparing or mixing two weight sets needs an explicit quotient** (`align(· → ·)` over S_n per layer, S_heads,
   GL per head). Without it, "average" (E2) and "steer along another run's grokking direction" (P2) are ill-posed,
   and empirically they are destructive.
4. **Notation sketch** (illustrative only, not implemented):
   ```text
   gauge   L3.mlp.neurons  := perm(π)                         # check |Δlogits| < 1e-9
   reparam fold_ln ; center_writers ; ln -> rms               # check |Δlogits| < 1e-9
   gauge   resid := R · resid,  R ∈ O(d)    requires rms      # S5e
   edit    L4.write := −L2.write                              # report Δloss, rand-control(‖Δθ‖)
   edit    L3 := lerp(L2, align(L4 → L2), 1/2)
   path    θ(t) := θ(mem) + t·(θ(grok) − θ(mem)),  t ∈ [−0.5, 2]   # report test acc(t)
   ```
   The compile target is exactly what `pythia_ws.py` / `steer_grok.py` do. The relay models would *propose*
   statements, a CPU checker would *execute* them, and the measured semantics would flow back into the shared
   notes. A model's claim that "X is a symmetry" is then falsifiable in under a second.
5. **A model zoo the checker can run on CPU:** pythia-14m (28 MB), modular-addition models (trained in 2–8 min),
   and the TRM and HRM core nets (6.8M / 27M params). The TRM/HRM puzzle-embedding tables (1.8–2.5 GB) fit in RAM.
   That is the ARC interface: an ARC task is a 16×512 block of rows, so "task-level" weight statements are row
   statements. Full ARC scoring and TTT stay off-box.
6. **Not covered by this spike:** the "Codebench" / ARC-as-programs side (DSL programs, code-golf solutions). A
   sibling report covers it (`$W/arc.md`).


---

## 5. Caveats

* Wall-clock times were measured under heavy CPU contention from a parallel sibling job. Step counts are exact.
* Grokking: one seed per configuration, p=97 rather than Nanda's 113, and an 8-minute budget. The frac-0.3 runs
  did not grok within budget, so they give no crossing step.
* pythia-14m results use one 149-token probe text. The Δloss values of the edits are indicative, not
  benchmark-grade. The symmetry results (about 1e-11) do not depend on the text.
* TRM/HRM: metadata and configs only. The CPU-time estimate is my FLOP arithmetic. The 11.7 s measurement and the
  6,830,082 parameter count come from a sibling agent's notes (`$W/arc.md`), not re-measured by me.
* The sibling-created `$W/venv` was reused and extended (not recreated). The cold-install timing comes from a
  separate throwaway venv, which was deleted afterwards.

## 6. Files

* Scripts: `$W/feas_scripts/{timed_install.sh, pythia_ws.py, hf_diff.py, grok_spike.py, plot_grok.py, steer_grok.py, hf_meta.py, prof.py}`
* Outputs: `$W/feas_out/{timed_install.log, pythia_ws.json, grok_*_p97_f0{5,3}.json, grok_*.log, grok_curves.png, steer/steer_results.json, trm_hrm_meta.json, trm_hrm_meta.log}`
* Downloads (untrusted, separate directories): `$W/models/pythia-14m/`, `$W/feas_dl_trm_cfg/`, `$W/feas_dl_hrm_cfg/`
