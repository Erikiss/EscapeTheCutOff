# Lens B: formal languages and theory whose semantics are weights

Research date: 2026-11-25. Every paper, repo or tool below was opened on arXiv, GitHub or a conference page during this session, unless it is marked **UNVERIFIED**. A venue is given only when the opened page states it.
CPU spike (numpy only, my own code): `spike/symmetry_spike.py`, output in `spike/symmetry_spike.out`, both in this directory.

---

## 0. TL;DR

* **The owner's three example statements fall into three different logical types.** A useful weight-space language has to tell them apart:
  1. **Symmetry statements** ("rotate all weights"). They are function-preserving and checkable by algebra alone, with a THEOREM-grade answer. The owner's statement (a) is one of these. It is **true only in a precise form**: the *residual-stream* rotation has to be applied to every reader and writer at once (SliceGPT Thm. 1). A rotation of *only the middle layer* is true only for the per-head pairs (Q with K, V with O). With RoPE, the Q/K part is further restricted to RoPE-commuting 2x2 blocks. My spike confirms all of this numerically (deltas of 1e-14 against O(1)).
  2. **Edit statements** ("set layers l-1 and l+1 antisymmetric, and the net interpolates"). They change the function and need an empirical check. Statement (b) splits into:
     * a **true theorem**: an antisymmetric generator A gives a rotation exp(A), and an outer pair exp(A) / exp(-A) *conjugates* the middle layer, which preserves its spectrum;
     * an **empirical true** part: a slerp interpolation of adjacent layers can be inserted as an extra layer (PLI, 2025), and middle layers can be run in parallel with averaged outputs (Painters, 2024);
     * a **false** part: making the middle layers copies of the centre layer is catastrophic (Painters "Middle Repeat");
     * an **open** part: whether trained LLM weights are smooth enough in depth for "middle = interpolation of neighbours". It is proven only for ResNet→neural-ODE settings.
  3. **Dynamics statements** ("is grokking steerable?"). They are claims about training trajectories and need training runs. Statement (c) is **empirically true on toy algorithmic tasks**:
     * the timing can be accelerated, delayed or eliminated by weight norm (Omnigrok), slow-gradient amplification (Grokfast, >50x), ⊥Grad/StableMax and Muon;
     * the *circuit identity* can be transplanted via the weight **direction** (40/40 runs, July 2026);
     * the delay is mediated by logit scale (R²=0.97, June 2026).

     It is **open** for LLM scale, and for a training-free "add one vector and the checkpoint groks".
* **Program↔weights languages exist and are mature only at toy scale:** RASP → Tracr, ALTA, C-RASP, Learning Transformer Programs, and D-RASP (ICML 2026), which decompiles *trained* small transformers to RASP.
* **Declarative weight-edit languages exist but have no semantics.** mergekit YAML and BrainSurgery YAML (June 2026, MIT, with assertions) manipulate tensors *by name*. They know nothing about symmetry groups, residual bases or claims.
* **Formal interpretability-claim frameworks exist but have no concrete syntax** for weights: causal abstraction, causal scrubbing, compact proofs and interpretive equivalence. pyvene and nnsight specify *activation* interventions, not weight statements.
* **The closest existing work to the owner's idea is iWSP** ("Interpretable weight-space programming", arXiv 2611.18114, **13 Nov 2026**, repo MIT). In it, Claude Code agents write transformer weights in closed-form Python code, with no training. This is weight-space programming by LLMs at ~166-parameter scale. It is 12 days old and beyond all three relay models' cutoffs.
* **What is missing (the gap the owner could fill):** a *typed, symmetry-aware* statement language with:
  * a group-theoretic checker ("is this edit a symmetry of this architecture?");
  * a status lattice: THEOREM / EMPIRICAL(paper, model, scale) / OPEN / FALSE;
  * provenance links to papers;
  * an executable CPU semantics on small testbeds (Tracr/ALTA models, InterpBench, grokking toys, small Pythia/GPT-2).

**Kurzfassung (DE):**
* Deine Beispiel-Aussagen sind drei verschiedene Typen: Symmetrie (beweisbar), Edit (empirisch) und Trainingsdynamik (nur mit Trainingsläufen prüfbar).
* "Alle Gewichte rotieren" ist exakt wahr, wenn die Residual-Rotation global angewandt wird. Lokal gilt es nur paarweise pro Head (Q mit K, V mit O).
* "Antisymmetrisch davor und danach" heißt mathematisch Rotation und Konjugation, keine Interpolation. Interpolation benachbarter Layer per Slerp ist empirisch belegt (PLI). Den mittleren Layer überall zu kopieren ist falsch (Painters).
* Grokking ist auf Toy-Tasks steuerbar. Für LLM-Skala ist das offen.
* Eine semantische Weight-Space-DSL gibt es noch nicht. mergekit/BrainSurgery sind nur Tensor-Skripte. iWSP (Nov 2026) zeigt, dass LLM-Agenten Gewichte direkt "programmieren" können.

---

## 1. Languages whose compiled semantics are transformer weights (RASP family)

| Name | What it expresses | Maturity / code | CPU | Verified |
|---|---|---|---|---|
| **RASP**: Weiss, Goldberg, Yahav, "Thinking Like Transformers", arXiv 2106.06981, ICML 2021 | Sequence primitives (select / aggregate / elementwise) that model encoder attention and FFN. Programs for histograms, sorting and Dyck. Gives a layer/head upper bound for a task. | Language and paper. The arXiv page lists no code. | n/a | yes |
| **Tracr**: Lindner, Kramár, Farquhar, Rahtz, McGrath, Mikulik, arXiv 2301.05062, NeurIPS 2023 Spotlight | Compiles RASP (embedded in Python) into **decoder-only transformer weights with known structure**. Compiled models have **no LayerNorm** (`layer_norm=False` in `compiler/assemble.py`) and need a BOS token. Residual stream = direct sum of labelled 1-D subspaces. Paper shows frequency, sort and parentheses programs. | github.com/google-deepmind/tracr. Apache-2.0, 574 stars. **Archived 10 Apr 2025, read-only.** Last commit 2024-02-05 (from my clone). JAX + Haiku. | `pip install` into a venv with jax 0.12.1 / dm-haiku 0.0.17 on Python 3.13 **succeeded**. Running it was **not attempted** (executing downloaded code was refused by the session's permission policy). So CPU execution is **UNVERIFIED**, though JAX-CPU is the default backend. | yes (repo, paper, source) |
| **ALTA**: Shaw, Cohan, Eisenstein, Lee, Berant, Toutanova, arXiv 2410.18077, TMLR 2025 | RASP/Tracr successor with **loops** and compilation to **Universal Transformers**. Constructive length-invariant parity and addition. SCAN via shift-reduce parsing. **Trace supervision**: training on ALTA execution traces as finer-grained supervision. "Minimality" of a program with respect to a training set. | github.com/google-deepmind/alta. Apache-2.0 (code) / CC-BY 4.0. 36 stars. Not archived. Deps: absl-py, tensorflow, numpy, jax, apache-beam, scipy. | Probably CPU, since the dependencies are all CPU-capable. **UNVERIFIED (not run).** | yes |
| **Learning Transformer Programs**: Friedman, Wettig, Chen, arXiv 2306.01128, NeurIPS 2023 oral | Trains a *constrained* transformer by gradient descent, then converts it to **discrete human-readable Python** (RASP-like). Covers in-context learning, sort, Dyck, NER and text classification. | github.com/princeton-nlp/TransformerPrograms. PyTorch 1.13.1, Python 3.8. 167 stars. No license shown on page. | Small models, so CPU is plausible. **UNVERIFIED.** | yes |
| **Adaptive Transformer Programs**: Lai-Dang, Kang, Son, ICLR 2025 poster | Extends Transformer Programs with redesigned attention modules for categorical and numerical reasoning, plus a reparameterisation scheme. Narrows the gap to black-box transformers. | ICLR page. No arXiv id found. Code UNVERIFIED. | ? | yes (ICLR page) |
| **What Accuracy Hides In Learned Transformer Programs**: Lee, Oozeer, Quirke, arXiv 2610.22891, 16 Oct 2026 | Critique of Transformer Programs across 9 tasks. Token accuracy overstates correctness. Exhaustive testing found a counting program at 100% on samples that still fails a valid input. Restricted ops can need many layers or large lookup tables. | arXiv only. No code link. | n/a | yes |
| **C-RASP / K_t[#]**: Yang & Chiang, arXiv 2404.04393 | Temporal counting logic, equivalent to C-RASP. Every formula **compiles** into a future-masked softmax transformer, giving the best-known lower bound on expressivity. | arXiv. No code link on the page. | n/a | yes |
| **RASP-Generalization Conjecture**: Zhou et al., arXiv 2310.16028 | Transformers length-generalise when the task has a *short RASP program*. The abs page does not use the name "RASP-L". | arXiv | n/a | yes |
| **Neural Decompiling of Tracr Transformers**: Thurnherr & Riesen, arXiv 2410.00061, ANNPR 2024 (LNCS) | A learned decompiler from Tracr weights to RASP. >30% exact reproduction, >70% functionally equivalent. | No code link on the page. | n/a | yes |
| **D-RASP**: Huang, Bakalova, Bhattamishra, Merrill, Hahn, "Discovering Interpretable Algorithms by Decompiling Transformers to RASP", arXiv 2602.08857, ICML 2026 | **Decompiles normally *trained* small transformers.** It reparameterises the transformer as a RASP program, then uses causal interventions to isolate a small sufficient sub-program. It often recovers simple programs from length-generalising models. | No code link on the abs page. | n/a | yes |
| **Tracr-Injection**: Vergara-Browne & Soto, arXiv 2505.10719, ACL Findings 2025 | Distils RASP algorithms into a **pre-trained LM**. The residual stream then contains an interpretable subspace that decodes the RASP variables. Better OOD performance. | The paper says code is released (URL not on the abs page). | ? | yes |
| **InterpBench**: Gupta, Arcuschin, Kwa, Garriga-Alonso, arXiv 2407.14494, NeurIPS 2024 D&B | 86 semi-synthetic transformers with **known circuits**: Tracr circuits re-trained with **Strict Interchange Intervention Training (SIIT)** so that their weights look natural. Later expanded with TracrBench circuits. | Code: github FlyingPumba/InterpBench (MIT). Models: HF cybershiptrooper/InterpBench (CC-BY). | small | abs page yes. The model count (86) and the repo/HF links come from the v3 HTML via search, not opened directly. |
| **MIPS**: Michaud et al., "Opening the AI black box: program synthesis via mechanistic interpretability", arXiv 2402.05110 | RNN weights → finite-state machine (integer autoencoder) → symbolic regression → **Python code**. Solves 32 of 62 tasks, 13 of them not solved by GPT-4. | No code link on the abs page. | small | yes |
| **Langosco et al. "meta-models" decompiler** | (owner-adjacent idea) | **UNVERIFIED.** Not found. Related: William Baker's "Inverse Tracr" PDF. | | no |

**Assessment.**
* Program→weights is solved for *toy, LayerNorm-free* models: Tracr is frozen but still useful; ALTA is maintained.
* Weights→program now works on *small trained* models (D-RASP, 2026), and partly on compiled ones (learned decompilers).
* None of these handles an LLM-scale model. None expresses statements *about* weights such as "this rotation is a no-op". Their statements are *programs*, and the weights are the compilation target.

**Bridge to the ARC/"Codebench" idea (pointer only, not researched in this lens).** If ARC tasks exist as short programs (DSL / Python), RASP/ALTA is the formal bridge from "program" to "weights". D-RASP is the bridge back.

---

## 2. Weight-space learning, symmetries, editing and merging

### 2.1 Networks that take weights as input (the "graphical approach")

* **DWSNets.** Navon, Shamsian, Achituve, Fetaya, Chechik, Maron, "Equivariant Architectures for Learning in Deep Weight Spaces", arXiv 2301.12780, ICML 2023. It characterises *all* affine equivariant and invariant layers for MLP neuron-permutation symmetry.
* **NFN.** Zhou et al., "Permutation Equivariant Neural Functionals", arXiv 2302.14040, NeurIPS 2023. NF-layers by parameter sharing. Code: github.com/AllanYangZhou/nfn.
* **Neural Functional Transformers (NFT).** Zhou et al., arXiv 2305.13546. Attention-based permutation-equivariant weight-space layers, plus Inr2Array (+17% INR classification). Same repo.
* **Graph Metanetworks (GMN).** Lim, Maron, Law, Lorraine, Lucas, arXiv 2312.04501. A network is turned into a *graph*, processed by a GNN. Covers multi-head attention, normalisation, conv, ResNet and group-equivariant layers. Equivariance to parameter permutations is proven.
* **Neural Graphs.** Kofinas et al., "Graph Neural Networks for Learning Equivariant Representations of Neural Networks", arXiv 2403.12143, ICLR 2024. Neurons are nodes and weights are edges. Code: github.com/mkofinas/neural-graphs.
  *This is the literal "graphical approach" to weight space: a weight-space statement can be a graph-rewrite rule on this graph.*
* **Transformer-NFN.** Tran et al., "Equivariant Neural Functional Networks for Transformers", arXiv 2410.04209, ICLR 2025.
  * It proves the **maximal symmetry group of multi-head attention** is S_h × GL_{D_k}(R)^h × GL_{D_v}(R)^h. Thm 3.2, under full-rank and distinct-heads assumptions: W̄_Q = W_Q M^T, W̄_K = W_K M^{-1}, W̄_V = W_V N, W̄_O = N^{-1} W_O, plus a head permutation.
  * It releases >125k transformer checkpoints.
* **Survey.** Han et al., "A Survey of Weight Space Learning: Understanding, Representation, and Generation", arXiv 2603.10090 (Mar 2026). Companion list: github.com/Zehong-Wang/Awesome-Weight-Space-Learning.

### 2.2 Symmetries and mode connectivity

* **Linear mode connectivity.** Frankle, Dziugaite, Roy, Carbin, arXiv 1912.05671, ICML 2020. Networks become stable to SGD noise early in training: linear interpolation between them shows no loss barrier.
* **Git Re-Basin.** Ainsworth, Hayase, Srinivasa, arXiv 2209.04836 (venue not listed on the abs page). Permute units to align two models, then merge. Zero-barrier LMC for ResNets on CIFAR-10. It also discusses a counterexample to the LMC hypothesis.
* **Rotation symmetry for transformers.** Zhang, Zheng, Chen, Li, "Beyond the Permutation Symmetry of Transformers: The Role of Rotation for Model Fusion", arXiv 2502.00264, ICML 2025.
  * It uses per-head rotations R_qk^h on (W_Q, W_K) together, and R_vo^h on (W_V, W_O) together.
  * The FFN has only permutation symmetry, because of its element-wise activation.
  * It does not discuss RoPE.
  * Code: github.com/zhengzaiyi/RotationSymmetry.
* **Computational invariance (the TRUE "rotate all weights").** SliceGPT, Ashkboos et al., arXiv 2401.15024, ICLR 2024, **Theorem 1**.
  * For an RMSNorm-connected transformer and an orthogonal Q:
    * W_embd → W_embd·Q;
    * every block input W_in → Qᵀ·W_in;
    * every block output W_out → W_out·Q, with b_out → Qᵀ b_out;
    * W_head → Qᵀ·W_head.

    This leaves the function unchanged, because RMSNorm(XQ)Qᵀ = RMSNorm(X).
  * LayerNorm is first converted to RMSNorm: the mean subtraction is folded into W_out and the scale diag(α) into the next W_in.
  * A per-block Q_ℓ is also possible, if the residual connection gets the extra matrix Q_{ℓ-1}ᵀQ_ℓ.
  * Code: github.com/microsoft/TransformerCompression.
* **QuaRot.** Ashkboos et al., arXiv 2404.00456 (venue not listed on the abs page). A rotation that "removes outliers from the hidden state without changing the output". 4-bit weights, activations and KV cache. Code: github.com/spcl/QuaRot.
* **SpinQuant.** Liu et al., arXiv 2405.16406, ICLR 2025. Four rotation placements:
  * R1 = residual-stream rotation, *mergeable* into the weights;
  * R2 = per-head rotation on V and O-input, mergeable, chosen independently per layer;
  * R3 and R4 = *online* Hadamard rotations (KV cache, before the MLP down-projection). These are not weight symmetries.

  R1 and R2 are learned by **Cayley SGD on the Stiefel manifold**: each update is the Cayley transform of a skew-symmetric matrix. This is the antisymmetric↔rotation link used in §3b. Code: github.com/facebookresearch/SpinQuant.

### 2.3 Editing and merging (function-changing weight statements)

* **Task arithmetic.** Ilharco et al., arXiv 2212.04089, ICLR 2023. The task vector τ = θ_ft − θ_pre, with negation, addition and analogies.
* **Contrastive weight steering.** Fierro & Roger, "Steering Language Models with Weight Arithmetic", arXiv 2511.05408, ICLR 2026. A behaviour direction is the difference of two opposing small fine-tunes. It often generalises OOD better than activation steering. There is preliminary evidence that emergent misalignment can be detected by comparing updates to an "evil" weight direction.
* **ROME.** Meng, Bau, Andonian, Belinkov, arXiv 2202.05262, NeurIPS 2022. Causal intervention localises facts to mid-layer MLPs, which are then edited with a **rank-one** update. Code: rome.baulab.info.
* **MEMIT.** Meng et al., arXiv 2210.07229. Thousands of edits on GPT-J and GPT-NeoX. Code: memit.baulab.info.
* **Parameter decomposition: weights → mechanisms, the most "machine-code" interpretability line.**
  * APD, Braun, Bushnaq, Heimersheim, Mendel, Sharkey, arXiv 2501.14926. Minimises mechanistic description length.
  * SPD, Bushnaq, Braun, Sharkey, arXiv 2506.20790. Code: github.com/goodfire-ai/spd.
  * tPD, Vigouroux & Sharkey, "Targeted Recovery of Weight-Space Mechanisms From Neural Networks", arXiv 2607.13047, ICML 2026 MI workshop. It extracts only the components that process target inputs. It extracts a CSS-only submodel with 7% of the FLOPs, and surgically ablates and rewires memorised sequences in a 12-block transformer.
* **Weight-sparse transformers.** Gao et al. (OpenAI), arXiv 2511.13653; the search snippet lists ICML 2026 / PMLR 306. Training with most weights at zero yields small interpretable circuits, about 16x smaller at equal loss (per the search snippet). Scaling beyond tens of millions of non-zero parameters is open.

---

## 3. The owner's example statements: TRUE / FALSE / OPEN

### Spike results (numpy, CPU, <1 s)

The model is a tiny pre-norm decoder: d=16, 2 heads of d_head=8, ReLU MLP of width 32, 5 layers, RMSNorm, optional RoPE. The table shows max |Δlogits|; ~1e-14 means an exact symmetry.

| id | statement | result |
|---|---|---|
| A1 | rotate the residual stream in **all** layers, plus E and U (gains = 1) | **1.5e-14 (exact)** |
| A2 | same, with RoPE | **1.3e-14 (exact)**. RoPE acts inside the heads, not on the residual stream. |
| A3 | global rotation with non-unit RMSNorm gains **not folded** | 6.4 (broken) |
| A4/A5 | fold the gains into the next W_in, then rotate | 5e-15 / 1.8e-14 (exact) |
| A6 | rotate **only the middle layer's** residual-facing readers and writers | **8.1 (broken)** |
| B1 | middle layer, one head: Q→QM, K→KM^{-T}, M ∈ GL(d_head), no RoPE | 5e-13 (exact) |
| B2 | same, **with RoPE** | 3.6 (broken) |
| B3 | M restricted to RoPE-commuting 2x2 blocks [[a,b],[-b,a]] per frequency pair, with RoPE | 4e-15 (exact) |
| B4 | V→VN, O→N^{-1}O, with RoPE | 5e-15 (exact) |
| B5 | rotate W_Q only (K untouched) | 1.4 (broken) |
| C1 | middle MLP: permute plus positive per-neuron scaling of the hidden units (ReLU) | 2.7e-15 (exact) |
| C2 | middle MLP: *rotate* the hidden units | 6.0 (broken) |
| D1 | exp(A) is orthogonal for antisymmetric A | 2e-15 |
| D2 | Euler residual step x+εAx changes the norm only at O(ε²) | 1.5e-3 at ε=0.01 |
| D3 | linear (lerp) midpoint of two rotations | singular values off by up to 0.96 (not a rotation) |
| D4/D5 | geodesic midpoint R0·exp(½·log(R0ᵀR1)) is orthogonal, and the log is antisymmetric | 9e-16 / 7e-15 |
| D6/D7 | exp(A)·W·exp(−A): the spectrum of W is unchanged (2e-14), but W itself changes (4.8) | conjugation, not cancellation |
| D8 | residual triple (I+εA)(I+εW)(I−εA) − (I+εW) scales as ε²·([A,W] − A²) | ratio → 50.1 (predicted 50.17) |

B3 is a derivation made here, not a cited result. RoPE multiplies q and k by position-dependent block rotations, so only an M in the *commutant* of all those rotations survives. For distinct frequencies that means per-pair "complex scalars", i.e. scaled 2x2 rotations. This matches SpinQuant putting its learnable per-head rotation R2 on V/O, not on Q/K. SpinQuant's stated reason was not verified.

### (a) "All weights in the middle layer rotate"

* **TRUE (theorem)** as: "rotate the residual-stream basis by an orthogonal Q in *every* layer, plus the embedding and unembedding, after folding the norm gains". Source: SliceGPT Thm 1; QuaRot; SpinQuant R1. Spike: A1, A2, A5.
* **TRUE (theorem), local to one layer**, as: "inside layer ℓ, transform each head's (W_Q, W_K) by (M, M^{-T}) and its (W_V, W_O) by (N, N^{-1})". Source: Transformer-NFN's maximal group S_h × GL^h × GL^h; Zhang et al. 2025 for the orthogonal subgroup. With RoPE, M must commute with RoPE. MLP hidden units: permutation and, for ReLU, positive scaling only (DWSNets/NFN). Spike: B1, B3, B4, C1.
* **FALSE** as: "rotate only the middle layer's residual-facing weights" (A6), "rotate only W_Q" (B5), or "rotate the MLP hidden basis" (C2).
* **Basis caveat (important for interpretability claims):** a global rotation preserves the function but *destroys the readable basis*. In Tracr, the residual stream is a labelled direct sum, and after a rotation it is not. So "neuron/channel k means X" is never invariant; only subspace statements are. This ties directly to Tracr's design, and to InterpBench's motivation that Tracr weights are unnaturally sparse and axis-aligned.
* **Empirical reading** ("during training, middle-layer weights rotate"): UNVERIFIED in this lens. No paper was opened for weight-rotation dynamics across depth.

### (b) "Layers before and after the middle set antisymmetric → the net interpolates in between"

* **TRUE (theorem), but it is rotation/conjugation, not interpolation.**
  * An antisymmetric matrix A (Aᵀ = −A) has purely imaginary eigenvalues, so exp(A) is a rotation. The AntisymmetricRNN (Chang, Chen, Haber, Chi, arXiv 1902.09689, ICLR 2019) uses h_t = h_{t−1} + ε·tanh((W_h − W_hᵀ − γI)h_{t−1} + V_h x_t + b_h). The diffusion term γ fixes the outward spiral of forward Euler.
  * Haber & Ruthotto ("Stable Architectures for Deep Neural Networks", arXiv 1705.03341, *Inverse Problems* 34(1)) write: "the antisymmetric matrix K0 yields a rotation in the feature space, which preserves the distances between the features and leads to a stable forward propagation and a well-posed learning problem" (verified in the PDF text).
  * If layer ℓ−1 applies exp(A) and layer ℓ+1 applies exp(−A), the block is the middle layer *seen in a rotated frame*: the spectrum is preserved and the eigenbasis is rotated (D6/D7). This is the local form of SliceGPT's per-block Q_ℓ.
  * In residual (Euler) form the pair does not cancel exactly. It leaves an ε²([A,W] − A²) term (D8).
  * In a **pre-norm transformer** a block x + g(RMSNorm(x)) cannot implement an exact linear rotation, because g∘RMSNorm is scale-invariant. So the theorem holds only to first order. **Exact realisability inside a real LLM block is OPEN / unverified.**
* **"Interpolates" — TRUE in the geodesic sense.** The geodesic between two rotations R0 and R1 is R0·exp(t·log(R0ᵀR1)), and that log is antisymmetric (D4, D5). A linear average of two rotations is *not* a rotation (D3). PLI (below) also uses slerp rather than lerp for layer interpolation. So the precise statement is: "the layer between two rotation-layers is the geodesic midpoint, whose generator is half the antisymmetric log".
* **EMPIRICAL TRUE (LLMs):**
  * Transformer Layers as Painters (Sun, Pickett, Nain, Jones, arXiv 2407.09298, v4 Feb 2025; Llama2-7B, BERT-Large, Pythia/Mistral in the appendix) finds that middle layers share a representation space.
  * Skipping a few middle layers degrades gracefully.
  * Reversed or randomised middle order degrades gracefully (random better than reversed).
  * **Running the middle layers in parallel and averaging their *outputs* works** for most benchmarks except GSM8K. Looping the parallel block helps. Math and reasoning (GSM8K, ARC) are more order-sensitive.
  * PLI (Chen et al., "Expanding before Inferring… Premature Layers Interpolation", arXiv 2506.02973) inserts training-free layers built by **slerp** between adjacent layers' weights. Upper-layer positions 24/28 of 32 work best, 2–3 layers are optimal, and TruthfulQA MC1 rises about +1.08 on average over 8 models. Lerp appears only as an ablation.
  * Related: Lad, Lee, Gurnee, Tegmark (arXiv 2406.19384) keep 72–95% of top-1 accuracy after deleting or swapping adjacent layers. Middle layers are most robust ("stages of inference").
  * Gromov et al. (arXiv 2403.17887) remove up to ~half the layers with QLoRA healing.
  * Layer-merging compression: LaCo (arXiv 2402.11187, EMNLP Findings 2024; "rear layers collapse into a prior layer"; merge mechanics not verified) and Sliding-Window Merging (arXiv 2502.19159; "parameter consolidation" of similar consecutive layers; mechanics not verified).
* **FALSE:** "the middle layers are all the same function". Painters' **Middle Repeat**, which replaces the middle layers by copies of the centre layer, "quickly degrades to random baseline". It is worse than skipping.
* **OPEN (theory):** "trained LLM weights are smooth in depth, so θ_ℓ ≈ ½(θ_{ℓ−1}+θ_{ℓ+1})". This is proven only for ResNet→neural-ODE settings:
  * Sander, Ablin, Peyré, arXiv 2205.14612, NeurIPS 2022: linear residuals with small initial loss → limit ODE at rate 1/N;
  * Marion et al., arXiv 2309.01213, ICLR 2024 spotlight: a ResNet initialised as an ODE discretisation stays one during gradient flow.

  A natural CPU test: measure ‖θ_ℓ − ½(θ_{ℓ−1}+θ_{ℓ+1})‖ on Pythia/GPT-2 *after* Git-Re-Basin-style alignment of adjacent layers, since neighbouring layers need not share a neuron basis.
* **Not verified:** whether trained decoders' W_QK has a measurable *antisymmetric* part. Saponati et al., arXiv 2502.10927, report "symmetry" for bidirectional training and "directionality / column dominance" for autoregressive training. The symmetric/antisymmetric decomposition itself was not seen on the abs page.

### (c) "Is grokking steerable? What happens if I steer along grokking?"

* **EMPIRICAL TRUE on toy algorithmic tasks**, through optimiser and weight-space interventions:
  * **Weight norm.** Omnigrok (Liu, Michaud, Tegmark, arXiv 2210.01117): a large initialisation *induces* grokking (MNIST; LSTM on IMDb with α=6; GCNN on QM9 with α=3). "Constrained optimization at constant weight norm (α=0.8) largely eliminates grokking" (1-layer transformer, modular addition). The time to generalise scales about as 1/γ in weight decay.
  * **Slow-gradient amplification.** Grokfast (Lee, Kang, Kim, Lee, arXiv 2405.20233): amplify the low-frequency component of each parameter's gradient history, giving a >50x speedup. Code: github.com/ironjr/grokfast.
  * **Numerical-stability view.** Prieto, Barsbey, Mediano, Birdal, arXiv 2501.04697; the PDF header in the search snippet says ICLR 2025. After overfitting, gradients align with a "naïve loss minimisation" direction that just scales the logits, *scaling weights along their current direction*. This leads to Softmax Collapse. **StableMax** and **⊥Grad** (which removes that component) enable or accelerate grokking without regularisation. Code on GitHub.
  * **Logit scale is the proximal variable.** Truong, arXiv 2606.18465, Jun 2026: clamp the norm and vary the output temperature. The delay tracks logit scale (R²=0.97), and the norm adds only 1–2%. Under MSE the route differs. Code: github.com/ClevixLab/grokking-logit-scale.
  * **Direction carries circuit identity.** Truong, "Cross-Trajectory Chimera Interventions…", arXiv 2607.06628, Jul 2026: transplanting a donor's weight *direction* into a recipient with a different norm drove the run to the donor's circuit in 40/40 cases. An angle-matched random control produced no shift. The transfer is threshold-like, with the threshold location predicted by the recipient's norm.
    This is the closest verified result to "steer along grokking": **steer by setting the direction, then continue training.**
  * **Low-dimensional trajectory.** Xu, arXiv 2602.16746, v3 Apr 2026:
    * one PC captures 68–83% of the attention-weight trajectory variance;
    * suppressing orthogonal gradient flow *prevents* generalisation, with a monotonic dose-response;
    * artificially raising curvature is not sufficient.

    Xu, arXiv 2604.07380: at grokking, gradient and weight-decay updates align on a "compression axis" that is >4000x more impactful than random directions when ablated. Removing weight decay after grokking reverses the compression and keeps the algorithm.
  * **Embedding transfer.** GrokTransfer (Xu, Ni, Wang, Hu, arXiv 2504.13292, ICLR 2025): initialise with a weaker model's embedding. The delay disappears, provably on XOR.
  * **Optimiser.** Muon (Tveit et al., arXiv 2504.16041): mean grokking epoch 153.1 → 102.9 against AdamW.
  * **Mechanism.** Nanda et al., arXiv 2301.05217: the modular-addition circuit is the Fourier "addition as rotation about a circle", with phases memorisation → circuit formation → cleanup. Note the coincidence with (a): the *grokked algorithm itself is a rotation*.
* **OPEN:**
  * (i) a *training-free* single additive weight vector that turns a memorising checkpoint into a generalising one (no verified paper found);
  * (ii) any of the above at LLM scale;
  * (iii) "steering along grokking" in the *activation-steering* sense (no verified paper found).

  Weight-space behaviour steering in LLMs exists (contrastive weight steering, arXiv 2511.05408), but not for grokking.
* **Formal type.** These are *dynamics* statements of the form "∀ runs with intervention I on the optimiser or weights, the delay changes by Δ". Unlike (a), they are only checkable by training runs. Toy modular-arithmetic grokking is CPU-feasible (a small transformer or MLP, p≈97–113). CPU runtime was not measured here.

---

## 4. Existing languages and DSLs for weight edits and interpretability claims

| System | What it is | Weight-level? | Semantics / checking | Status |
|---|---|---|---|---|
| **mergekit YAML** (Goddard et al., arXiv 2403.13257; github.com/arcee-ai/mergekit, LGPL-3.0, ~7.5k stars) | Declarative merge configs: `slices[].sources[].{model, layer_range}`, `merge_method`, and per-filter parameter gradients. Methods: linear, SLERP, NuSLERP, Multi-SLERP, Karcher, task arithmetic, TIES, DARE, DELLA, breadcrumbs, SCE, Model Stock, Nearswap, Arcee Fusion, passthrough. **CPU supported.** | yes, by tensor name and layer range | None. No notion of symmetry, basis or claim. | mature, widely used |
| **BrainSurgery** (Barmina et al., arXiv 2606.09707, v2 Oct 2026; github.com/schneiderkamplab/brainsurgery, MIT, PyPI) | YAML plans over checkpoint tensors. Ops: copy, move, delete, split, concat, reshape, permute, cast, add, subtract, scale, matmul, fill, phlora (low-rank), diff. Targets are regexes on tensor names, with slices. **Assertions**: exists, shape, dtype, equal, all. Web UI. Coding agents generate and review plans reliably, and the plans are shorter than the equivalent Python. | yes | Structural assertions only, no functional semantics | new and small (6 stars) |
| **Penzai** (github.com/google-deepmind/penzai, Apache-2.0, 1.9k stars) | JAX "model surgery": type-driven selectors (`pz.select`), named axes, treescope. Loads Gemma, Llama, Mistral and Pythia. | yes (pytree rewriting) | Python API | **Archived 25 Aug 2026** |
| **pyvene** (Wu et al., arXiv 2403.07809; github.com/stanfordnlp/pyvene, Apache-2.0, 920 stars) | Interventions "specified as dicts", serialisable and shareable via HuggingFace. Static or trainable interventions. | **activations**, not weights | Causal-abstraction-oriented | mature |
| **NNsight / NDIF** (Fiotto-Kaufman et al., arXiv 2407.14561) | Deferred-execution **intervention graph** that "decouples experimental design from model runtime". Remote execution on NDIF. | activations (and module access) | Python tracing API | mature |
| **circuit-tracer** (github.com/safety-research/circuit-tracer; README now lists the decoderesearch org; MIT, ~3k stars) | Attribution graphs over cross-layer transcoder features, with interventions that set feature values. Gemma-2-2B, Llama-3.2-1B, Qwen-3, GPT-OSS-20B, Gemma-3. `--device cpu` exists. | features / activations | Graph = an *explanation artefact* | mature |
| **Causal abstraction** (Geiger et al., arXiv 2301.04709, v4 May 2025) | Formal theory that unifies patching, mediation, causal scrubbing, causal tracing, circuits, concept erasure, SAEs, DAS and steering. Extended to arbitrary *mechanism transformations*. | the theory covers any mechanism | Semantics with no concrete syntax | theory |
| **Causal scrubbing** (Redwood, LessWrong/AF, ~Nov 2022) | A hypothesis is a computational graph plus a correspondence to model nodes, tested by resampling ablations. There is a known false-positive critique. | activations | Formal test procedure | method |
| **Compact proofs** (Gross et al., arXiv 2406.11779, NeurIPS 2024) | Mechanistic understanding turned into **computer-checked proofs** of accuracy lower bounds (Max-of-K, 151 seeds, 102 strategies). | yes: proofs are over weights | **Formal** | research |
| **Interpretive equivalence** (Sun & Toneva, arXiv 2603.30002, ICLR 2026) | When two models share an interpretation: two interpretations are equivalent when all of their implementations are. With an estimation algorithm. | representations | Formal definition | research |
| **iWSP** (Mehta, Singh, …, Jojic, Gao, arXiv 2611.18114, 13 Nov 2026; github.com/csinva/weight-space-programming-autoresearch, MIT, 137 stars) | **An LLM agent writes the weights of a small GPT-style transformer in closed-form Python (PyTorch). No training of any kind.** 17 tasks (13 algorithmic, 4 NLP). Algorithmic models shrink from ~3,400 to a median 166 parameters at 100%. NLP models ~190k parameters. Audited by an LLM judge, re-run invariance and a "text bottleneck" rebuild: all 13 algorithmic rebuilds reach 100%. Agents: Claude Code agents. Also brain-encoding models. About $17.5k of API spend. | **yes: the weights are the program** | Validation score plus audits. No formal statement language. | brand new |

**What is missing (design gap):**
1. **No typed statement language** spans the three statement types (symmetry / edit / dynamics), plus the fourth type, mechanism claims (causal abstraction).
2. **No symmetry checker.** Given an architecture description (pre/post-norm, LN vs RMSNorm, RoPE, gated MLP, tied embeddings), nothing decides whether an edit is in the function-preserving group. The answer is mechanical, as the spike shows: residual O(d) after gain folding; per-head GL(d_k) on QK, restricted by RoPE; GL(d_v) on VO; head permutations; MLP permutations, plus positive scaling for ReLU.
3. **No provenance or status lattice.** Nothing tags a statement as THEOREM(proof ref) / EMPIRICAL(paper, model, scale, metric) / OPEN / FALSE(counterexample).
4. **Tensor-name DSLs (mergekit, BrainSurgery) lack semantic types** such as "residual basis", "head", "reader/writer". Activation DSLs (pyvene, nnsight) do not address weights.
5. **RASP/ALTA statements are programs, not claims about weights.** They are still the best *ground-truth generator* for testing weight statements (known basis, no LayerNorm).

---

## 5. Sketch: a weight-space statement language for the relay

Proposed, not existing.

```
# types
Space   := resid | head(l,h).qk | head(l,h).vo | mlp(l).hidden
Group   := O(resid) | GL(qk) | Commutant(RoPE) ⊂ GL(qk) | GL(vo) | Perm(h) | Perm⋉Pos(mlp.hidden)
Stmt    := Symmetry  g:Group on Space [requires fold(norm_gains)]
         | Edit      op(weights) => predicted_effect(metric, eval_set)
         | Dynamics  train(cfg) with intervention I => delta(metric/time)
         | Mechanism program P (RASP/ALTA) ~ model M  under alignment τ   (causal abstraction)
Status  := THEOREM(ref) | EMPIRICAL(ref, model, scale) | OPEN | FALSE(counterexample)

# the owner's examples, typed
S1  Symmetry rotate(resid, Q:O(d)) all_layers requires fold(rmsnorm_gains)         -- THEOREM SliceGPT Thm1; spike A1–A6
S2  Symmetry head(mid,*).qk by M ∈ Commutant(RoPE); head(mid,*).vo by N ∈ GL     -- THEOREM Tran'25 Thm3.2 + RoPE derivation; spike B1–B4
E1  Edit insert between(l,l+1) := slerp(θ_l, θ_{l+1}, 0.5) => TruthfulQA ↑      -- EMPIRICAL PLI 2506.02973 (7–8B, upper layers)
E2  Edit replace(mid-k..mid+k) := repeat(θ_mid)            => collapse            -- FALSE→"degrades to random" Painters 2407.09298
E3  Edit parallel(mid-k..mid+k, combine=mean_outputs)       => small loss, ≠ math  -- EMPIRICAL Painters
E4  Edit W_{mid-1}:=exp(A), W_{mid+1}:=exp(-A), Aᵀ=-A       => conj(exp A, W_mid)  -- THEOREM linear/ODE; OPEN pre-norm block
D1  Dynamics modadd(p) with grad_ema_amplify(λ)             => t_grok ↓ (>50×)     -- EMPIRICAL Grokfast
D2  Dynamics modadd(p) with implant direction(θ_donor)      => circuit = donor     -- EMPIRICAL 2607.06628 (40/40)
```

**Checker tiers, all on CPU:**
* **Algebraic.** Symmetry statements are verified exactly on random weights of the declared architecture. This is the spike, generalised; it takes milliseconds.
* **Ground-truth.** Tracr/ALTA-compiled or InterpBench models, where the "true" mechanism is known.
* **Small real models.** GPT-2-small / Pythia-70M–410M for Edit statements (perplexity deltas).
* **Toy training.** Modular-arithmetic grokking for Dynamics statements.

**Relay roles (fits cutoffmix):**
* *Translator* (paper → statement): give this to the model that is freshest on the ML literature (Fable, arXiv to ~2026-04).
* *Typer/critic*: decides which type the statement is and whether it is a symmetry; another model, with the checker as an oracle.
* *Executor*: the CPU checker returns the status.

The human "weight-space editor" writes statements. The checker answers THEOREM/FALSE instantly for symmetries, and queues runs for the rest.

**Cutoff note for the relay.** These verified items post-date Fable's arXiv knowledge (~2026-04), so they must be injected as context:

| Item | Date |
|---|---|
| BrainSurgery | 2606 |
| Truong logit-scale paper | 2606 |
| tPD | 2607 |
| Chimera interventions | 2607 |
| "What Accuracy Hides" | 2610 |
| iWSP | 2611 |
| Penzai archival | Aug 2026 |

D-RASP (2602), the WSL survey (2603), interpretive equivalence (2603) and Xu's grokking papers (2602, 2604) sit at or just inside that edge. The titles may be recognised (Fable recognises titles to ~2026-09), but their content should not be trusted from memory.

---

## 6. Reproduction

```
python3 -I spike/symmetry_spike.py      # numpy only; prints the table in §3
```

Tracr was cloned to `dl_tracr/src` and installed into `venv_tracr`, but not executed: running downloaded third-party code was refused by the session's permission policy.

## 7. UNVERIFIED / not checked

* Tracr, ALTA and Learning Transformer Programs actually running on CPU: not executed.
* The "Langosco et al. meta-models" decompiler: not found.
* Weight-rotation *training dynamics* in middle layers.
* An antisymmetric-part analysis of W_QK in trained decoders (Saponati et al.'s abs page does not show it).
* The layer-merge mechanics of LaCo and SWM.
* SOLAR depth-up-scaling layer choice.
* Any training-free "grokking vector", or activation steering of grokking.
* Venues not shown on the abs pages: QuaRot, Git Re-Basin, Omnigrok, Grokfast, Nanda et al., MEMIT, Saponati et al.
* Code URLs not on the abs pages: D-RASP, APD, MIPS, Tracr-Injection, contrastive weight steering.
* The Prieto et al. ICLR 2025 venue and the weight-sparse ICML 2026 venue come from search-result snippets (the PDF header and the PMLR page respectively), not from the opened abs pages.
