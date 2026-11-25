# WSBench – Build Dossier, Phase 1 (for Codex)

**Kurzfassung für den Auftraggeber (Deutsch).** Dieses Dokument ist ein vollständiger, eigenständiger Bauauftrag. Es setzt kein Repository voraus.

- **Was gebaut wird:** **WSBench**, eine lokale Web-App mit Python-Backend. Damit schreibt man Anweisungen im Gewichtsraum in einer kleinen formalen Sprache (WSL). Die Maschine führt sie sofort auf echten Gewichten aus und sagt, ob eine Behauptung stimmt. Beispiele:
  - „rotiere den Residualstrom“ (Symmetrie oder nicht?);
  - „Layer 4 := −Layer 2“ (was kostet das?);
  - „ist Grokking steuerbar?“;
  - „löst ein kompiliertes Spiegel-Netz die ARC-Aufgabe?“.
- **Bilder:** Gewichte werden als Heatmap und als ARC-Raster gezeigt.
- **ARC und Code Golf:** Die 400 Aufgaben und die 400 Code-Golf-Lösungen (Apache-2.0) sind eingebunden.
- **Drei Modelle:** Sie übersetzen Paper-Text in Anweisungen, sagen das Ergebnis voraus und schlagen Kontrollen vor. Die Maschine misst.
- **Die Rechenkerne liegen als geprüfter Referenzcode bei** (Anhang A–C). Die erwarteten Zahlen stehen in Anhang F; sie wurden am 25.11.2026 auf einer 4-Kern-CPU gemessen.
- **Phase 2** (Details, Integration ins cutoffmix-Relay, TRM-Modell) macht danach Claude; siehe §12.

---

## 0. Rules for the implementer (read first)

1. **Everything you need is in this document.** There is no repository to read. Do not look for one.
2. **Build exactly what is specified. Add no features.** If something is unspecified, choose the simplest option that satisfies the acceptance tests and record the choice in `NOTES.md` (one line per decision).
3. **The reference code in Appendices A–C is verified. Copy it into the package unchanged**, at the paths given in §2.3. You may add type hints and docstrings, but do not change the arithmetic, the order of random draws, the dtype or the parameter names. The acceptance numbers in Appendix F depend on it.
4. **All weight-space arithmetic runs in float64 on CPU.** No GPU code paths.
5. **Never evaluate text produced by a language model as Python.** Model output is only ever parsed as WSL (§3). WSL is a data language, not code.
6. **Third-party code (the Code Golf solutions) runs only in a subprocess** with a timeout (§6.4).
7. **Work milestone by milestone (§11).** A milestone is done when its acceptance tests pass. Commit after each milestone.
8. **Tests must run offline.** Only tests marked `@pytest.mark.network` may download anything, and `@pytest.mark.live` tests call the paid LLM API. Both are skipped by default.
9. When the spec gives an exact string (error code, JSON key, file name, prompt text), use it exactly.

---

## 1. What we are building, in plain words

Researchers in mechanistic interpretability often state findings about weights in words:

- "rotating the residual stream does not change the model";
- "the middle layers are interchangeable";
- "grokking can be accelerated".

The same English sentence can mean very different things. "All weights in the middle layer rotate" is an exact no-op in one reading (a gauge symmetry, applied consistently to all readers and writers). In another reading (only that layer's matrices are rotated) it destroys the model.

**WSBench makes such statements precise and checkable.**

1. The user (a "weight-space editor") writes **statements** in **WSL** (Weight-Space Language). Each statement has two parts:
   - a **claim kind**: `gauge` = "this is an exact symmetry", `edit` = "this changes the function; measure it", and so on;
   - a **pipeline** of weight operations.
2. The **executor** applies the pipeline to real weights in float64 and measures what changed. It returns a **verdict**:
   - `CONFIRMED` or `REFUTED` for claims;
   - `MEASURED` for edits, together with a random control.
3. The **graphics** show weight matrices before, after and as a difference: as heatmaps, and for small blocks as ARC-style colour grids.
4. The **ARC / Code Golf module** connects this to ARC-AGI:
   - 400 ARC tasks;
   - their shortest-known Python solutions;
   - grid primitives (rot90, mirror, …) that work both on grids and on weight blocks;
   - a compiler that turns a grid primitive into an explicit network weight matrix, checked against all task examples.
5. The **assist** sends a text (for example a paper abstract) to three language models through OpenRouter. Each model has a role:
   - one translates the text into WSL statements;
   - one predicts each verdict and fixes mistakes;
   - one proposes control statements.

   The executor then runs everything and shows predicted next to measured results.

### 1.1 Glossary

| Term | Meaning |
|---|---|
| residual stream | The vector `x` (size `d`) that each transformer layer reads from and adds to. |
| reader / writer | A matrix that reads the residual stream (through a norm), or one whose output is added to it. |
| gauge (symmetry) | A weight change that leaves the network function exactly unchanged. Example: permuting MLP neurons together with the matching columns. |
| reparam | A function-preserving change that also changes the parametrisation class. Example: folding LayerNorm scales into the next matrix, which adds a bias parameter. |
| edit | A weight change that changes the function. Its effect is measured as Δloss. |
| GPT-NeoX | The architecture of the Pythia models. Parallel residual: `x ← x + Attn(LN1 x) + MLP(LN2 x)`. Fused QKV matrix. Rotary position embedding (RoPE) on part of each head. |
| RoPE | Rotary position embedding: position-dependent 2×2 rotations of pairs of query/key dimensions. |
| grokking | A small network first memorises its training set, and only much later suddenly generalises. Classic example: modular addition `(a + b) mod p`. |
| Grokfast | A gradient filter, `g ← g + λ·EMA(g)`, reported to make grokking happen earlier. |
| ARC-AGI | A benchmark of grid puzzles (≤ 30×30 cells, 10 colours). Each task has train and test input/output pairs. |
| Code Golf 2025 | The Google Code Golf Championship (NeurIPS 2025): the 400 ARC-AGI-1 training tasks, each solved by the shortest possible Python function `p(grid)`. Each task file has `train`, `test` and about 260 extra `arc-gen` examples. |
| NeuroGolf | The 2026 follow-up competition: each ARC task solved by the smallest possible neural network, on one-hot `(10, 30, 30)` grids. Our compiler (§6.5) produces networks in this encoding. |

---

## 2. Fixed technical decisions

### 2.1 Stack

- Python ≥ 3.11.
- Dependencies (pin lower bounds only):
  - `torch>=2.4` (CPU wheel);
  - `numpy>=1.26`;
  - `scipy>=1.11`;
  - `safetensors>=0.4`;
  - `tokenizers>=0.15`;
  - `fastapi>=0.110`;
  - `uvicorn>=0.29`;
  - `httpx>=0.27`;
  - `pytest>=8`.
- **Do not use** `transformers`, `transformer_lens`, `nnsight`, a JavaScript build step, or any CDN.
- Frontend: one static `index.html`, plus `app.js` and `style.css`. Vanilla JavaScript (ES2020), served by FastAPI. Charts are hand-written SVG; heatmaps and grids are drawn on `<canvas>`.
- Data directory: environment variable `WSB_HOME`, default `~/.cache/wsbench`. All downloads, training runs and logs go there.
- LLM key: environment variable `OPENROUTER_API_KEY`. Never log it and never write it to disk.

### 2.2 Commands (console script `wsbench`)

| Command | Effect |
|---|---|
| `wsbench serve [--host 127.0.0.1] [--port 8765]` | Start the web app. |
| `wsbench run FILE.wsl [--json OUT.json]` | Execute a script and print one line per verdict. Exit code 0 if no statement has status `ERROR`, otherwise 1. |
| `wsbench check FILE.wsl` | Parse and run the static hints (§3.8). No execution. Prints `file:line:col: severity: CODE message`. |
| `wsbench fetch pythia-14m` | Download the model (§4.2). |
| `wsbench fetch arc` | Download the Code Golf data (§6.1). |
| `wsbench fetch all` | Both. |

### 2.3 Package layout (create exactly these modules; you may add private helpers)

```
pyproject.toml                    # name "wsbench", version "0.1.0", console script wsbench = wsbench.cli:main
NOTES.md                          # your decisions (rule 0.2)
wsbench/
  __init__.py
  cli.py
  paths.py                        # WSB_HOME handling
  lang/
    lexer.py  parser.py  ast.py  printer.py  hints.py  card.py   # card.py holds LANGUAGE_CARD (Appendix E.1)
  models/
    neox.py                       # = Appendix A, verbatim
    modadd.py                     # = Appendix B, verbatim
    registry.py                   # model names -> loaders (§4)
    fetch.py                      # downloads with sha256 checks
  exec/
    executor.py  verdict.py  resolve.py  jobs.py  runs.py
  arc/
    grid.py                       # = Appendix C, verbatim
    data.py  golf_runner.py  render.py
  assist/
    openrouter.py  prompts.py  pipeline.py  mock.py
  server/
    app.py
    static/index.html  static/app.js  static/style.css
  examples/
    rotate_middle_layer.wsl  antisymmetric_pair.wsl  grokking_steerable.wsl  arc_mirror.wsl   # Appendix D
tests/
  test_lang.py  test_hints.py  test_neox_tiny.py  test_pythia.py  test_modadd.py  test_arc.py
  test_api.py  test_assist_mock.py
  fixtures/golf_mini/task087.json  fixtures/golf_mini/task150.json                           # Appendix D.5
```

---

## 3. The language WSL v0.1

### 3.1 Lexical rules

- WSL is line-oriented: one statement or directive per line.
- `#` starts a comment that runs to the end of the line. Blank lines are ignored.
- Lines are numbered from 1 (comment and blank lines count).
- **Identifiers:** `[A-Za-z_][A-Za-z0-9_]*`, case-sensitive.
- **Numbers:** an optional sign, digits, an optional fraction and an optional exponent (`-1.5e-3`).
- **Strings:** double-quoted, no escapes except `\"` and `\\`.
- **Booleans:** `true` and `false`.
- **Multi-character tokens** must be lexed before single characters: `:=`, `->`.
- **Single-character tokens:** `( ) [ ] , ; : = .` and also `+ -` inside layer indices (§3.3).
- Whitespace other than the newline is insignificant.
- **Keywords are contextual.** `model`, `seed` and the claim names are keywords only at the start of a line (after an optional `LABEL:`). Elsewhere they are ordinary identifiers, so `from=train` is valid. The words `all`, `pass`, `none`, `fixed`, `gated`, `mid`, `resid`, `orth`, `orth1`, `gl` and `randperm` are ordinary identifiers that the ops interpret.

### 3.2 Grammar (EBNF)

```
script       = { line } ;
line         = [ directive | statement ] [ comment ] NEWLINE ;
directive    = "model" model_name | "seed" INTEGER ;
model_name   = "neox-tiny" | "pythia-14m" | "modadd" | "grid:" TASK ;      (* TASK = "task" followed by 3 digits *)
statement    = [ IDENT ":" ] claim pipeline ;
claim        = "gauge" | "reparam" | "edit" | "approx" [ "(" NUMBER ")" ] | "train" | "path" | "solve" ;
pipeline     = op { ";" op } ;
op           = assignment | call ;
assignment   = layer ":=" layer_expr ;
call         = IDENT "(" [ arg { "," arg } ] ")" ;
arg          = [ IDENT "=" ] value ;
value        = NUMBER | STRING | "true" | "false" | IDENT | list | range | target | call ;
list         = "[" [ value { "," value } ] "]" ;
range        = "[" NUMBER ":" NUMBER ":" NUMBER "]" ;
target       = "resid" | layer [ "." part ] ;
layer        = "L" INTEGER | "L" "[" index "]" ;                 (* L3 and L[3] are the same *)
index        = INTEGER | "mid" [ ( "+" | "-" ) INTEGER ] ;
part         = "mlp" | "heads" | "H" INTEGER | "H" "[" INTEGER "]" ;
layer_expr   = layer | call ;                                   (* calls allowed here: lerp, neg_writers, neg, align *)
align_call   = "align" "(" layer_expr "->" layer_expr ")" ;     (* the only place where "->" may appear *)
```

Notes:

- `model_name`: `grid:` is lexed as one token followed by `taskNNN`, with no whitespace in between.
- `claim`: `approx` without a number means `approx(0.05)`.
- `layer`: the token `L3` must be lexed as layer 3, not as an identifier, wherever a target is expected. `L` followed by digits is reserved and cannot be a label.
- `index`: `mid` = `n_layers // 2` (integer division) of the current model.
- **Directives** take effect from their line to the next directive of the same kind:
  - `model` is required before the first statement; otherwise every statement errors with `E_MODEL`.
  - `seed` defaults to 0.
- **Labels** are optional and must be unique within a script (`E_LABEL`). Without a label, the statement is called `line<N>`.
- **Range** `[a:b:s]`: `a, a+s, a+2s, …`. It includes `b` if a term is within `1e-9·max(1,|s|)` of `b`, and stops before passing `b`. `s` must not be 0 (`E_BAD_ARG`). Example: `[1:0.5:-0.1]` = `1, 0.9, 0.8, 0.7, 0.6, 0.5`.

### 3.3 Values that denote random group elements

These identifiers (or calls) are valid as arguments where §3.4 says *group element*. Their size is inferred from the op.

| Value | Meaning | Reference function (Appendix A) |
|---|---|---|
| `orth` | random orthogonal matrix | `rand_orth(n, g)` |
| `orth1` | random orthogonal matrix with `R·1 = 1` (fixes the all-ones vector) | `rand_orth_fixing_ones(n, g)` |
| `gl` | random well-conditioned invertible matrix | `rand_gl(n, g)` |
| `randperm` | random permutation | `rand_perm(n, g)` |
| `perm([i0, i1, …])` | explicit permutation; must be a permutation of `0..n-1` (`E_BAD_ARG`) | – |
| `uniform(lo, hi)` | random positive vector, entries in `[lo, hi)`; requires `0 < lo < hi` | `rand_uniform(n, lo, hi, g)` |

**Determinism.** Each random value gets its own generator:

```
g = torch.Generator().manual_seed(script_seed * 1_000_003 + line_number * 1009 + k)
```

Here `k` is the 0-based index of that random value among the random values of the same statement, counted in source order, left to right. The same script always produces the same numbers.

### 3.4 Operation catalogue

Each row names the WSL call, the models it is valid for, and the reference function it must call. `i` and `j` are resolved layer indices (§3.5), `h` is a head index. An op used with the wrong model family gives `E_MODEL`. Bad argument types or values give `E_BAD_ARG`.

**Family `neox`** (models `neox-tiny`, `pythia-14m`). State = `(W, norm_kind)`; it starts as `(W0, 'ln')`.

| WSL | Arguments | Reference call | Kind of change |
|---|---|---|---|
| `fold_ln()` | – | `W = fold_ln(cfg, W)` | reparam (adds `embed_out.bias`) |
| `center_writers()` | – | `W = center_writers(cfg, W)` | reparam |
| `ln_to_rms()` | – | `norm_kind = 'rms'` | reparam (architecture change) |
| `rotate(resid, R, layers=all, embed=true, unembed=true)` | `R` ∈ {`orth`, `orth1`} of size `d`. `layers` = `all` or a list of integer layer indices (e.g. `[3]`; no `mid` inside the list). | `rotate_resid(cfg, W, R, layers=None if all else list, embed=…, unembed=…)` | depends |
| `scale(resid, c)` | `c` a number | `scale_resid(cfg, W, c)` | depends |
| `permute(L[i].mlp, π)` | `π` ∈ {`randperm`, `perm([...])`} of size `d_mlp` | `perm_mlp(cfg, W, i, π)` | depends |
| `scale(L[i].mlp, s)` | `s` = `uniform(lo, hi)` of size `d_mlp` | `scale_mlp(cfg, W, i, s)` | depends |
| `rotate(L[i].mlp, Q)` | `Q` = `orth` of size `d_mlp` | `rotate_mlp(cfg, W, i, Q)` | depends |
| `permute(L[i].heads, π)` | `π` of size `n_heads` | `perm_heads(cfg, W, i, list(π))` | depends |
| `vo(L[i].H[h], N)` | `N` ∈ {`gl`, `orth`} of size `dh` | `vo_gauge(cfg, W, i, h, N)` | depends |
| `qk(L[i].H[h], A, dims=pass)` | `dims` ∈ {`pass`, `all`}. `pass` = `torch.arange(rd, dh)` (the non-rotary dims), `all` = `torch.arange(0, dh)`. `A` ∈ {`gl`, `orth`} of size `len(dims)`. | `qk_gauge(cfg, W, i, h, A, dims)` | depends |
| `rope_pairs(L[i].H[h], phis=[…], scale=1.0)` | `phis`: exactly `rd/2` numbers (radians) | `rope_pairs(cfg, W, i, h, phis, scale)` | depends |
| `delete(L[i])` | – | `zero_writers(W, [i])` | edit |
| `swap(L[i], L[j])` | – | `A, B = get_layer(W,i), get_layer(W,j); W = set_layer(set_layer(W, i, B), j, A)` | edit |
| `L[i] := E` | `E` is a layer expression (below) | `W = set_layer(W, i, eval(E))` | depends |
| `gridop(P, block("NAME", r0, r1, c0, c1))` | `P` = a primitive name from §6.3. `NAME` = an existing 2-D parameter. `0 ≤ r0 < r1 ≤ rows`, `0 ≤ c0 < c1 ≤ cols`. | `gridop_block(W, P, NAME, r0, r1, c0, c1, TORCH_PRIMITIVES[P])` | edit |

**Layer expressions** (evaluated on the **current** pipeline state, before the assignment):

| Expression | Value |
|---|---|
| `L[j]` | `get_layer(W, j)` |
| `lerp(X, Y, t)` | `lerp_layer(X, Y, t)`; `t` a number |
| `neg_writers(X)` | `neg_writers(X)` |
| `neg(X)` | `neg_all(X)` |
| `align(X -> Y)` | `align_layer(cfg, X, Y)`; X permuted (heads, then MLP neurons) to match Y |

`swap` exists because `L[2] := L[3]; L[3] := L[2]` sets both layers to the old layer 3: the second assignment reads the already updated state.

`TORCH_PRIMITIVES` mirror §6.3 on 2-D torch tensors:

| Name | Torch form |
|---|---|
| `identity` | `b` |
| `rot90` | `torch.rot90(b, -1, (0, 1))` |
| `rot180` | `torch.rot90(b, 2, (0, 1))` |
| `rot270` | `torch.rot90(b, 1, (0, 1))` |
| `hmirror` | `b.flip(0)` |
| `vmirror` | `b.flip(1)` |
| `dmirror` | `b.T` |
| `cmirror` | `torch.rot90(b, 2, (0, 1)).T` |

`rot90`, `rot270`, `dmirror` and `cmirror` need a square block; otherwise `E_SHAPE`.

**Family `modadd`** (model `modadd`). These statements create or read **runs** (§4.4).

| WSL | Arguments | Effect |
|---|---|---|
| `modadd(run=NAME, p=97, frac=0.5, steps=3000, seed=0, lr=0.001, wd=1.0, filter=none, save=[], intervene=[])` | `filter` = `none` or `grokfast(alpha, lamb)`. `intervene` = list of `at(STEP, scale(C))`. `save` = list of steps. | Only valid with claim `train`. Calls `train(p, frac, steps, seed, lr, wd, filt, interventions, save_steps, eval_every=50, threads=…)` from Appendix B, then stores the run. `filt` is `None` or `('grokfast', alpha, lamb)`; `interventions` = `[(STEP, 'scale', C), …]`. Checkpoints are saved for the `save` steps **and always for step 0 and the final step**. |
| `ckpt(NAME, STEP)` | – | A parameter dict (state_dict) of a stored run. Unknown run → `E_MISSING_RUN`; step not saved → `E_MISSING_RUN`. |
| `diff(X, Y)` | `X`, `Y` are `ckpt(...)` | Per-key `X − Y`. |
| `lerp(X, Y, t=RANGE)` | – | `path`: for each `t`, evaluate `(1−t)·X + t·Y`. |
| `add(B, D, t=RANGE)` | `B` = `ckpt`, `D` = `diff` | `path`: for each `t`, evaluate `B + t·D`. |
| `scale(X, c=RANGE)` | `X` = `ckpt` | `path`: for each `c`, evaluate `c·X`. |
| `cosine(A, B)` | `A`, `B` ∈ {`ckpt`, `diff`} | `path`: cosine of the two flattened parameter vectors (concatenated in sorted key order), plus both norms. |

"Evaluate" means `eval_state(sd, p, frac)` from Appendix B. It uses the `p` and `frac` of the run named by the first `ckpt` in the expression.

**Family `grid`** (model `grid:taskNNN`). Only valid with claim `solve`.

| WSL | Meaning |
|---|---|
| `prim(NAME)` | Apply the grid primitive `NAME` (§6.3) to each input. |
| `chain(prim(A), prim(B), …)` | Apply the primitives left to right. |
| `compile(NAME, size=fixed, from=all)` | Compile `NAME` into a permutation network (§6.5) and run the network on each input. `size` ∈ {`fixed`, `gated`}. `from` ∈ {`all`, `train`}: which examples' input sizes the compiler may see (`all` = train + test + arc-gen, `train` = the `train` split only). |
| `golf()` | Run the task's Code Golf solution (§6.4). |

### 3.5 Target resolution

- `L[i]` must satisfy `0 ≤ i < n_layers`; otherwise `E_TARGET`, for example `E_TARGET layer 7 out of range (pythia-14m has L0..L5)`.
- `H[h]` must satisfy `0 ≤ h < n_heads` (`E_TARGET`).
- `mid ± k` is computed first, then range-checked.
- `block("NAME", …)`: an unknown `NAME` or a tensor that is not 2-D gives `E_TARGET`.

### 3.6 Statement semantics

- **Every statement is independent.** For family `neox`, it starts from the unedited model `(W0, 'ln')`, applies its pipeline ops left to right, and is compared with the unedited model.
- `train` statements create runs.
- `path` statements read runs created by earlier lines of the same script, or stored on disk from earlier scripts.
- **Statements execute in source order**, one at a time (§5.5).

### 3.7 Claim kinds and the verdict status

| Claim | Valid for | Status rule |
|---|---|---|
| `gauge` | neox | `CONFIRMED` iff measured class = `exact`, else `REFUTED`. |
| `reparam` | neox | `CONFIRMED` iff measured class = `exact`, else `REFUTED`. The verdict lists `new_params`, `removed_params` and the final `norm_kind`. |
| `approx(tol)` | neox | `CONFIRMED` iff `rel ≤ tol`, else `REFUTED`. |
| `edit` | neox | Always `MEASURED`. If the measured class is `exact`, add warning `W_NOOP` ("this edit is a symmetry: it does not change the function"). |
| `train` | modadd | `MEASURED` (the run summary). |
| `path` | modadd | `MEASURED`. |
| `solve` | grid | `CONFIRMED` iff every example passes, else `REFUTED`. |

Any exception during a statement gives status `ERROR`, with `error = {"code": …, "message": …}`. The remaining statements still run.

**Measured class** (family `neox`):

- Let `ℓ0` = logits of the unedited model on the probe, and `ℓ1` = logits after the pipeline.
- `max_abs_dlogits = max|ℓ1 − ℓ0|`.
- `rel = max_abs_dlogits / max(1e-12, max|ℓ0|)`.
- Class:
  - `exact` if `rel ≤ 1e-9`;
  - `near` if `rel ≤ 1e-3`;
  - `edit` otherwise.

### 3.8 Static hints (editor support; run by `check`, by the parser endpoint, and before execution)

Hints are **warnings**, never errors. They predict a likely `REFUTED` and say how to fix the statement. Implement exactly these rules for family `neox`. The models' activation is GELU, the norm is LayerNorm, and both have RoPE.

| Code | Condition (on the parsed statement) | Message |
|---|---|---|
| `H_GELU_SCALE` | claim `gauge`, op `scale(L[i].mlp, …)` | `GELU is not positively homogeneous, so per-neuron rescaling is not a symmetry. Use permute(L[i].mlp, randperm), or write 'edit' to measure it.` |
| `H_MLP_ROTATE` | claim `gauge`, op `rotate(L[i].mlp, …)` | `Rotating MLP neurons is not a symmetry through an elementwise activation. Only permutations are.` |
| `H_RESID_NOFOLD` | claim `gauge`, `rotate(resid, …)` with no earlier `fold_ln()` in the same pipeline | `LayerNorm gains pin a basis. Put fold_ln() before the rotation.` |
| `H_RESID_ONES` | claim `gauge`, `rotate(resid, orth)` and the pipeline has no earlier `ln_to_rms()` | `LayerNorm centres the stream. Use orth1 (R·1 = 1), or center_writers(); ln_to_rms() first.` |
| `H_RESID_PARTIAL` | claim `gauge`, `rotate(resid, …)` with `layers` not `all`, or `embed=false`, or `unembed=false` | `A residual rotation is a symmetry only if it is applied to every reader and writer, the embedding and the unembedding.` |
| `H_ROPE_QK` | claim `gauge`, `qk(…, dims=all)` | `RoPE breaks GL symmetry on the rotary dims. Use dims=pass, or rope_pairs(...) for the rotary dims.` |
| `H_RMS_ORDER` | `ln_to_rms()` appears in a pipeline without an earlier `center_writers()` | `ln_to_rms() is exact only after fold_ln(); center_writers().` |
| `H_LERP_ALIGN` | `lerp(X, Y, t)` where `X` and `Y` are different layers and neither is wrapped in `align(...)` | `Mixing two layers without align() depends on the permutation gauge.` |
| `H_EDIT_GAUGE` | claim `edit` and every op in the pipeline is one of `permute(...)`, `vo(...)`, `qk(…, dims=pass)`, `rope_pairs(...)`, `fold_ln()`, `center_writers()` | `Every op here is a known symmetry: expect W_NOOP.` |

Each hint is reported as `{"code", "line", "col", "message"}`, where `col` is the column of the op that triggered it (1-based).

### 3.9 Error codes

| Code | Raised when |
|---|---|
| `E_SYNTAX` | Lexer or parser error; give the line, the column and the expected token(s). |
| `E_MODEL` | No `model` directive yet, an unknown model, or an op or claim not valid for the model family. |
| `E_LABEL` | Duplicate label. |
| `E_UNKNOWN_OP` | Unknown call name. |
| `E_BAD_ARG` | Wrong number, type or value of arguments. |
| `E_TARGET` | Layer, head or tensor out of range or unknown. |
| `E_SHAPE` | Shape mismatch, for example rot90 of a non-square block. |
| `E_MISSING_RUN` | `ckpt(...)` refers to an unknown run or an unsaved step. |
| `E_DOWNLOAD` | Model or data not available locally. The message tells the user which `wsbench fetch …` command to run. |
| `E_RUNTIME` | Any other exception, with its message. |

The parser should recover at the next line and report all syntax errors, not only the first.

### 3.10 Printer

`printer.py` turns an AST back into canonical text:

- one space after commas and around `:=` and `->`;
- `L[i]` with brackets;
- `H[h]` with brackets;
- numbers via `repr(float)`, or `str(int)` for integers.

`parse(print(parse(s)))` must equal `parse(s)` for every statement in Appendix D.

---

## 4. Models

### 4.1 `neox-tiny` (offline, for tests)

- Weights: `make_tiny(seed=0)` from Appendix A.
- Config: `TINY_CFG`.
- Probe ids: `TINY_IDS`.
- No download.

### 4.2 `pythia-14m`

Download these three files into `WSB_HOME/models/pythia-14m/`:

```
https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/config.json
https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/model.safetensors
https://huggingface.co/EleutherAI/pythia-14m/resolve/cf967c0a9a04383db6f7b1108d86b2962634b4ac/tokenizer.json
```

Verify the SHA-256 of each file. On a mismatch, delete the file and raise `E_DOWNLOAD`.

```
config.json        f97f966a66c444890ed461fff2a51eefb15d74303df05b948124719f199b0b17     698 bytes
model.safetensors  116a02532db461f91386a5b20f942ff2c8d4de7341e21b55caafc3d7b25f49a1  28,143,920 bytes
tokenizer.json     870f4e2baa6b683221fa52004d5d6f40ab8c9d31961617304b78c910c2c3caf2   2,114,042 bytes
```

Loading:

- `cfg = cfg_from_hf_config(json.load(config.json))`.
- `W0 = {k: v.to(torch.float64) for k, v in safetensors.torch.load_file(model.safetensors).items()}`. There are 76 tensors, stored as fp16.
- Probe: `ids = torch.tensor([Tokenizer.from_file(tokenizer.json).encode(PROBE_TEXT).ids])`, with `PROBE_TEXT` from Appendix D.1. That gives exactly 149 tokens.
- Facts: d=128, 6 layers, 4 heads of 32, d_mlp 512, rotary dims per head `rd` = 8, vocab 50,304.

### 4.3 Caching

- Load each model once per server process and keep `W0`, `cfg`, `ids` and `ℓ0` in memory.
- Never mutate `W0`. All reference operators copy their input.

### 4.4 `modadd` and runs

- A **run** is the result of a `train` statement.
- It is stored in `WSB_HOME/runs/NAME/`:
  - `run.json`: the arguments, `curves`, `first`, the list of saved steps, `wall_s`;
  - `step{S}.pt`: one per saved checkpoint, written with `torch.save(state_dict)`.
- Load checkpoints with `torch.load(path, map_location='cpu', weights_only=True)`.
- **Run names are global.** Training a run whose name already exists overwrites it, with warning `W_OVERWRITE`.
- **Threads:** `threads = max(1, os.cpu_count() - 1)`.

---

## 5. Executor and verdicts

### 5.1 Verdict JSON (one per statement)

```json
{
  "line": 7,
  "label": "R1",
  "source": "gauge fold_ln(); rotate(resid, orth1)",
  "model": "pythia-14m",
  "claim": "gauge",
  "tol": null,
  "status": "CONFIRMED",
  "measured": "exact",
  "metrics": {},
  "hints": [],
  "warnings": [],
  "error": null,
  "elapsed_ms": 142
}
```

Fields:

- `source`: the canonical printer output (§3.10).
- `tol`: a number, only for `approx`.
- `status`: one of `CONFIRMED`, `REFUTED`, `MEASURED`, `ERROR`, `PENDING`, `RUNNING`.
- `measured`: `exact`, `near`, `edit`, or `null` when not applicable.
- `metrics`: per family (§5.2–5.4).
- `hints`: static hints (§3.8).
- `warnings`: entries such as `{"code": "W_NOOP", "message": "..."}`.
- `error`: `null`, or `{"code": "...", "message": "..."}`.

### 5.2 Metrics for family `neox`

```json
{
  "max_abs_dlogits": 2.9e-11, "rel": 1.1e-12,
  "loss0": 4.3177, "loss1": 4.3177, "dloss": 0.0,
  "param_dist": 12.3,
  "changed_tensors": [{"name": "gpt_neox.layers.3.mlp.dense_h_to_4h.weight", "shape": [512, 128], "rel_change": 1.41}],
  "new_params": [], "removed_params": [], "norm_kind": "ln",
  "control": null
}
```

- `loss` is `next_token_loss(logits, ids)` from Appendix A.
- `param_dist` = `sqrt(Σ_k ‖W1[k] − W0[k]‖²)` over keys present in both.
- `changed_tensors`: every key whose tensor changed by any amount, with `rel_change = ‖W1[k] − W0[k]‖ / max(1e-12, ‖W0[k]‖)`, sorted by `rel_change` descending.
- `new_params` / `removed_params`: keys present in only one of the two states.
- **`control` (only for claim `edit`, and only if `param_dist > 0`):**
  - Let `mask[k]` = the entries where `W1[k] != W0[k]` (exact inequality), and `δ = W1 − W0`.
  - Draw `ε[k] ~ N(0, 1)` with `torch.Generator().manual_seed(statement_seed + 999_983)`, iterating keys in **sorted** order. `statement_seed` = the seed formula of §3.3 with `k = 0`.
  - `Wc[k] = W0[k] + mask[k] · ε[k] · (‖δ‖ / ‖mask·ε‖)`, where both norms are over all keys.
  - Report `{"dloss": dloss_control, "max_abs_dlogits": …, "ratio": dloss / dloss_control}`. Set `ratio` to `null` if `|dloss_control| < 1e-12`.

### 5.3 Metrics for family `modadd`

- `train`: `{"run": NAME, "first": {...}, "curves": {...}, "saved_steps": [...], "wall_s": ...}`. The keys of `first` are exactly those of Appendix B.
- `path` with lerp / add / scale: `{"points": [{"t": 0.0, "train_acc": ..., "test_acc": ..., "train_loss": ..., "test_loss": ..., "wnorm": ...}, ...]}`. For `scale`, the key is `"c"` instead of `"t"`.
- `path cosine(...)`: `{"cosine": ..., "norm_a": ..., "norm_b": ...}`.

### 5.4 Metrics for family `grid`

```json
{"passed": 39, "total": 266,
 "by_split": {"train": [1, 3], "test": [0, 1], "arc-gen": [38, 262]},
 "first_failure": {"split": "train", "index": 1, "input": [[...]], "expected": [[...]], "got": [[...]] },
 "compiled": {"sizes": [[4, 4]], "params_dense": 810000, "nnz": 16}}
```

- `by_split` holds `[passed, total]` per split.
- `got` may be `null` (not decodable) or an error string.
- `compiled` is present only for `compile(...)`. `nnz` = the number of non-negative entries summed over all compiled tables.
- For `golf()`, add `"bytes_raw"` and `"bytes_submitted"` (§6.1).

### 5.5 Script execution as a job

- `POST /api/run` creates a **script run** with a new `run_id` (uuid4 hex).
- It executes the statements sequentially in one background worker thread. A process-wide FIFO queue allows only one script at a time.
- `GET /api/runs/{run_id}` returns:
  ```json
  {"run_id": "...", "status": "queued|running|done", "verdicts": [...], "current_line": 12}
  ```
  Statements not yet started are listed with status `PENDING`. A `train` statement in progress is `RUNNING`, and its `metrics.curves` grows after every evaluation (every 50 steps).
- **Cancelling:** `POST /api/runs/{run_id}/cancel` sets a flag. `train` checks it through `should_stop`. A cancelled statement gets `ERROR` with code `E_RUNTIME` and message `cancelled`; later statements get `ERROR` / `cancelled` without running.
- **Retention:** keep the last 50 script runs in memory. The worker must catch everything, so the server never dies on a bad script.
- **Python API (keep stable, Phase 2 builds on it):** `wsbench.exec.executor.run_script(script: str, on_update=None, should_stop=None) -> list[dict]` runs a script synchronously and returns the verdicts (§5.1). It calls `on_update(verdicts_so_far)` after each statement and after each training evaluation. The server's worker thread is a thin wrapper around it. `wsbench.lang.parser.parse(script) -> (statements, errors)` and `wsbench.lang.hints.hints(statements) -> list` are likewise public.

---

## 6. ARC / Code Golf module

### 6.1 Data

`wsbench fetch arc`:

1. Download `https://codeload.github.com/SakanaAI/google-code-golf-2025/tar.gz/8ccc4b44b65d933527c205c304cb918883d12270` (repository `SakanaAI/google-code-golf-2025`, Apache-2.0, commit dated 2025-10-31).
2. Strip the first path component of each tar member.
3. Extract **only** these members into `WSB_HOME/golf/`:
   - `google-code-golf-2025/task*.json` (400 files)
   - `compression/raw/task*.py` (400)
   - `solutions/task*.py` (400)
   - `data/id_map.csv`
   - `LICENSE`
4. Reject members whose normalised path is absolute or contains `..`.
5. Verify the counts.

The files:

- **Task file:** `{"train": [...], "test": [...], "arc-gen": [...]}`. Each example is `{"input": grid, "output": grid}`, and a grid is a list of rows of ints 0–9. Example: task087 has 4 / 1 / 261 examples.
- **`id_map.csv`:** header `task_id,original_id`, e.g. `task001,007bbfb7` (the ARC-AGI-1 id).
- **`compression/raw/taskNNN.py`:** the readable solution. It defines `p(grid)`. Some solutions mutate their input; the runner deep-copies it (§6.4).
- **`solutions/taskNNN.py`:** the submitted file. For 36 tasks it is a zlib-compressed form of the raw file. Display and run the **raw** file. Report the byte sizes of both: `bytes_raw` and `bytes_submitted` (`os.path.getsize`).

### 6.2 Task index

`GET /api/arc/tasks` returns one entry per task:

```json
{"task": "task150", "original_id": "67a3c6ac", "n_train": 3, "n_test": 1, "n_arcgen": 262,
 "bytes_raw": 30, "bytes_submitted": 30, "single_primitive": "vmirror"}
```

- `bytes_raw` / `bytes_submitted` may be `null` if the file is missing.
- `single_primitive` is the name of the first primitive in §6.3 table order (excluding `identity`) that solves **all** examples, otherwise `null`.
- Compute the index once after the fetch and cache it in `WSB_HOME/golf/index.json`.
- Expected: exactly 7 tasks have a single primitive (Appendix F.4).

### 6.3 Grid primitives

The semantics are identical to Hodel's arc-dsl (MIT). `rot90` is **clockwise**. The implementation is `PRIMITIVES` in Appendix C.

| Name | numpy form | Example on `[[1,2,3],[4,5,6]]` |
|---|---|---|
| `identity` | `g` | `[[1,2,3],[4,5,6]]` |
| `rot90` | `np.rot90(g, -1)` | `[[4,1],[5,2],[6,3]]` |
| `rot180` | `np.rot90(g, 2)` | `[[6,5,4],[3,2,1]]` |
| `rot270` | `np.rot90(g, 1)` | `[[3,6],[2,5],[1,4]]` |
| `hmirror` | `g[::-1, :]` | `[[4,5,6],[1,2,3]]` |
| `vmirror` | `g[:, ::-1]` | `[[3,2,1],[6,5,4]]` |
| `dmirror` | `g.T` | `[[1,4],[2,5],[3,6]]` |
| `cmirror` | `np.rot90(g, 2).T` | `[[6,3],[5,2],[4,1]]` |

### 6.4 Code Golf runner (third-party code)

The runner follows the official rules of the 2025 competition utilities.

`golf_runner.py` writes our own small runner script to a fresh temporary directory, then runs it:

```
[sys.executable, "-I", RUNNER, SOLUTION_PATH]
```

- cwd = the temporary directory.
- Environment: only `PATH` and `PYTHONHASHSEED=0`.
- The examples go in as JSON on stdin.
- Timeout: 60 s per task. On timeout, kill the process and report every example as failed with `got = "error: timeout"`.

The runner script, for each example:

1. Imports the solution with `importlib.util.spec_from_file_location`. It must define a callable `p`.
2. Calls `result = p(copy.deepcopy(example["input"]))`.
3. Computes `s = json.dumps(result).replace("true", "1").replace("false", "0")`. If `s` contains any character outside `[0-9,\[\]\s\.]`, the example fails.
4. Otherwise the example passes iff `numpy.array_equal(numpy.array(json.loads(s)), numpy.array(example["output"]))`.
5. Exceptions fail the example, with `got = "error: <ExceptionType>: <message>"`.

It prints one JSON object to stdout: `{"results": [{"ok": true, "got": ...}, ...]}`. Send `got` only for the first failure, to keep the output small.

### 6.5 Primitive → network compiler (NeuroGolf encoding)

The implementation is in Appendix C.

- **Encoding.** A grid becomes a one-hot float32 array `(10, 30, 30)`:
  - the grid sits at the top-left;
  - padding cells are all-zero in every channel;
  - colour 0 is a real colour (black) and is **not** padding.
- **Network.** For an input of exactly `h × w`, the primitive is a cell permutation `src[900]`: output cell `j` copies input cell `src[j]`, and `−1` leaves it empty. This equals the weight matrix `kron(I_10, P)` with `P[j, src[j]] = 1`.
- **Size modes:**
  - `size=fixed`: one table, compiled for the input size of the **first** example of the chosen examples (`from=all`: train[0]).
  - `size=gated`: one table per distinct input size among the chosen examples. An input of an unseen size fails (`got = null`).
- **Decoding.** Use the `decode` function in Appendix C.
- **Pass rule.** The decoded grid must equal the expected output exactly (`check` in Appendix C).

**The finding to show.** On a zero-padded 30×30 canvas, mirroring is a different permutation for every grid size. A single fixed network therefore fails on most examples of a multi-size task (task150: 39 / 266). A size-gated network passes all 266. A weight matrix only "is" a grid operation once its axes are tied to grid cells.

### 6.6 Grid rendering (shared by the ARC panel and the weight views)

- **Palette** (`ARC_PALETTE` in Appendix C, RGB for colours 0–9): black, blue, red, green, yellow, grey, magenta, orange, light blue, maroon.
- **Cell size:** 16 px for grids up to 15×15, 10 px up to 30×30.
- **Grid lines:** 1 px, `#555`.
- **Padding:** one cell of background `#222` around each grid.

---

## 7. Graphics for weights

### 7.1 Heatmap view (any tensor)

`GET /api/tensor?run_id=…&line=…&name=…&which=before|after|diff&max=256`

- `before` = W0, `after` = W1, `diff` = W1 − W0.
- **1-D tensors** are shown as a `1 × n` strip.
- **Downsampling:** if a dimension is larger than `max`, average non-overlapping blocks of `ceil(dim / max)` along that dimension. A partial last block averages what it has.
- **Response:**
  ```json
  {"name": ..., "shape": [r, c], "shown": [r', c'], "block": [br, bc], "scale": s, "data": [[...]]}
  ```
  - `scale` = `max|before|` for `before` and `after` (the same scale for both), and `max|diff|` for `diff`.
  - `data` holds floats rounded to 6 significant digits.
- **Colour:** diverging around 0. `v/scale` is clamped to `[−1, 1]`; −1 → `#2166ac`, 0 → `#f7f7f7`, +1 → `#b2182b`, with linear RGB interpolation in between. Show `scale` and the block size under the image.
- **Store state:** the executor must keep `W1` for every `neox` statement of the last 5 script runs (memory: pythia-14m is about 110 MB in float64 per state). For older runs, return `410` with `{"error": "state expired; re-run the script"}`.

### 7.2 ARC view of a weight block

For any 2-D block of at most 30×30 entries, the UI can switch to "ARC view". `GET /api/tensor?...&view=arc&r0=&r1=&c0=&c1=` returns `{"grid": [[0..9]], "quantiles": [q10, ..., q90]}`.

- The colour index is the **decile bin** of each value within the block: index `= #{q ∈ quantiles : value > q}`.
- Render the grid with §6.6.
- Show this legend text verbatim: `ARC view: colour = decile of the value inside this block (0 = lowest tenth, 9 = highest). Colours are bins, not categories.`

### 7.3 Charts

Line charts in hand-written SVG:

- 640 × 320 px;
- 5 ticks on each axis;
- a legend;
- one colour per series: `#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd`.

They are used for:

- `train` curves: x = step; y = train_acc and test_acc on the left axis, with a second chart for train_loss and test_loss on a log scale;
- `path` points: x = t or c; y = train_acc and test_acc.

---

## 8. Assist: three models in the loop

### 8.1 OpenRouter client (`assist/openrouter.py`)

- Endpoint: `POST https://openrouter.ai/api/v1/chat/completions`.
- Headers:
  - `Authorization: Bearer ${OPENROUTER_API_KEY}`
  - `Content-Type: application/json`
  - `X-Title: wsbench`
- Body:
  ```json
  {"model": MODEL_ID, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
   "max_tokens": 6000, "reasoning": {"effort": "low", "exclude": true}, "usage": {"include": true}}
  ```
  Send **no** `temperature`: none of the three models honours it.
- Timeout: 180 s. No streaming.
- Result:
  ```
  {"ok": bool, "text": choices[0].message.content or "", "model": MODEL_ID,
   "prompt_tokens": int, "completion_tokens": int, "cost_usd": float, "error": str|null,
   "finish_reason": choices[0].finish_reason}
  ```
  - `cost_usd` = `usage.cost` if present; otherwise `prompt_tokens · in + completion_tokens · out`, using the price table below.
  - HTTP ≠ 200, an empty text, or `finish_reason == "content_filter"` → `ok = false`, and `error` holds the reason (include the HTTP body, first 500 chars).

**Models and roles** (defaults; the user can change them in the UI):

| Role | Default model id | Price per 1M tokens (input / output, USD) | Why this role |
|---|---|---|---|
| `translate` | `anthropic/claude-fable-5.2` | 8 / 40 | Measured freshest knowledge of ML papers (to about 2026-04). |
| `critic` | `openai/gpt-6.2-sol` | 2 / 10 | Fastest (about 1 s to the first token); cheap. |
| `alternatives` | `google/gemini-4-neon` | 3 / 15 | A third, independent view. |

**Fable restriction.** The provider's input filter for `anthropic/claude-fable-5.2` has refused prompts that ask it to continue or rework other models' outputs. So never send Fable another model's output:

- if the user assigns Fable to `critic` or `alternatives`, show the warning `Fable may refuse prompts that contain other models' output.`;
- send the request anyway;
- report a refusal as `ok = false`.

### 8.2 Pipeline (`POST /api/assist`)

**Input:**

```json
{"text": "...", "model": "pythia-14m", "roles": {"translate": "...", "critic": "...", "alternatives": "..."}, "execute": true}
```

- `model` is the WSL model for the statements; it must be a neox model or `grid:taskNNN`.
- `roles` is optional.
- `execute` defaults to true.

**Steps:**

1. **Translate.**
   - System prompt = `TRANSLATE_SYSTEM` (Appendix E.2) with `{LANGUAGE_CARD}` and `{MODEL}` substituted.
   - User prompt = `Source text:\n<<<\n{text}\n>>>`.
   - From the reply, extract the first fenced block that starts with three backticks followed by `wsl` and ends with three backticks.
   - Remove every `model` directive from the block and prepend `model {MODEL}`. The user's choice of target model always wins.
2. **Repair (at most once).** Parse the block. If there are errors, call the same model again with the same system prompt. The messages are:
   - user = the original user prompt;
   - assistant = its previous reply;
   - user = `Your WSL did not parse. Errors:\n{one line per error: line N col C CODE message}\nReturn the corrected wsl block only.`

   Parse again, then drop lines that still fail and keep their errors for display.
3. **Critic.**
   - System prompt = `CRITIC_SYSTEM` (Appendix E.3).
   - User prompt = `Model: {MODEL}\nStatements:\n` followed by the valid statements, numbered `1. …`, `2. …` in canonical form.
   - Extract the first block fenced as json. Parse it as a list of objects `{"n": int, "predict": str, "reason": str, "fix": str}`. Accept the `predict` values `exact`, `near`, `edit`, `pass`, `fail`, `unknown`.
   - If `fix` is non-empty and parses as a single valid statement, add it as an extra statement labelled `fix<n>`.
4. **Alternatives.**
   - System prompt = `ALT_SYSTEM` (Appendix E.4).
   - User prompt = `Source text:\n<<<\n{text}\n>>>\nModel: {MODEL}\nStatements already proposed:\n{numbered statements}`.
   - Extract the wsl block and keep at most 3 valid statements, labelled `alt1..alt3`.
5. **Execute** (if `execute`). Build one script from:
   - the model directive;
   - the translator statements, labelled `t1..tN` (overwriting any labels the model gave);
   - the fixes;
   - the alternatives.

   Run it as in §5.5.

**Response:**

```json
{"run_id": "...", "script": "...",
 "roles": {"translate": {...client result...}, "critic": {...}, "alternatives": {...}},
 "predictions": {"t1": "exact", ...}, "parse_errors": [...], "cost_usd": 0.0123}
```

The UI then polls `/api/runs/{run_id}`.

**Agreement.** For every statement, compare the critic's prediction with the measured outcome:

- for neox, `measured`;
- for grid, `pass` if `CONFIRMED`, else `fail`.

Mark it `agree`, `disagree` or `n/a`. This is the visible "synergy": the models predict, the machine measures, and the disagreements are where something is learned.

**Mock mode.** If the environment has `WSB_LLM=mock`, `openrouter.py` returns the canned replies of Appendix E.5 instead of calling the network, keyed by role. Cost is 0. All assist tests use mock mode, with target model `neox-tiny` so that they run offline. If the source text contains the marker `[[mock-syntax-error]]`, the first translate reply is the broken reply of E.5, and the repair call gets the normal translate reply.

---

## 9. HTTP API (FastAPI, JSON)

| Method and path | Request | Response |
|---|---|---|
| `GET /api/health` | – | `{"ok": true, "version": "0.1.0", "models_local": {"pythia-14m": bool}, "arc_local": bool, "llm": "live" or "mock" or "no-key"}` |
| `GET /api/language` | – | `{"card": LANGUAGE_CARD, "ops": [{"family": ..., "call": ..., "args": ..., "doc": ...}], "examples": [{"name": ..., "script": ...}]}` |
| `POST /api/parse` | `{"script": str}` | `{"statements": [{"line", "label", "claim", "source", "family"}], "errors": [{"line", "col", "code", "message"}], "hints": [...]}` |
| `POST /api/run` | `{"script": str}` | `{"run_id": str}` (also when the script has syntax errors: those lines become `ERROR` verdicts) |
| `GET /api/runs/{run_id}` | – | §5.5 |
| `POST /api/runs/{run_id}/cancel` | – | `{"ok": true}` |
| `GET /api/tensor` | query, §7 | §7 |
| `GET /api/arc/tasks` | – | §6.2 list |
| `GET /api/arc/task/{task}` | – | `{"task", "original_id", "train", "test", "arc_gen_first20", "n_arcgen", "solution_raw": str or null, "bytes_raw", "bytes_submitted"}` |
| `GET /api/arc/compiled` | `task`, `prim`, `size`, `from`, `shape` (e.g. `4x4`) | `{"src_grid": 30×30 list (the source cell index per output cell, −1 if empty), "P_preview": 300×300 floats (block mean of the dense P)}` |
| `POST /api/assist` | §8.2 | §8.2 |
| `GET /` | – | `index.html` |

Errors: HTTP 400 with `{"error": {"code", "message"}}` for bad requests, and 404 for an unknown run or task.

---

## 10. User interface (one page, four tabs)

**Header:**

- the title `WSBench`;
- status chips from `/api/health`: pythia-14m local yes/no, ARC data yes/no, LLM live/mock/no-key.

### Tab 1 "Editor"

**Left: the editor.**

- A `<textarea>` with a monospace font and line numbers.
- Below it, an **op palette**: buttons that insert a template at the cursor, one per op in §3.4, grouped by family.
- An **examples** dropdown that loads the Appendix D scripts.
- Typing triggers `/api/parse`, debounced 300 ms. Each problem appears as a line marker:
  - errors: red underline and a gutter icon;
  - hints: amber.

  Hovering shows the message.
- Buttons `Check` (parse only) and `Run`.

**Right: the results table**, one row per verdict:

| label | claim | source | status | measured | dloss | rel | control ratio | hints and warnings |
|---|---|---|---|---|---|---|---|---|

- Status colours: `CONFIRMED` green, `REFUTED` red, `MEASURED` blue, `ERROR` grey, `RUNNING` and `PENDING` with an animated dot.
- **Clicking a neox row** opens the detail panel:
  - the changed tensors list;
  - clicking a tensor shows three heatmaps side by side (before / after / diff, §7.1);
  - for a 2-D tensor, a block selector (`r0, r1, c0, c1`) with an "ARC view" toggle (§7.2), and a button `Insert gridop(...)`, which inserts `edit gridop(rot90, block("NAME", r0, r1, c0, c1))` into the editor.
- **Clicking a train or path row** shows the charts (§7.3).
- **Clicking a grid row** shows the first failure (input / expected / got grids).

### Tab 2 "ARC / Code Golf"

**Left: a task list** (400 rows: task, original id, `bytes_raw`, a single-primitive badge).

- A filter box (substring on task or original id).
- A checkbox "single primitive only".

**Right, for the selected task:**

- the train and test pairs as grids (§6.6), and the first 3 arc-gen pairs;
- the raw solution source (read-only `<pre>`) with its byte counts;
- buttons:
  - `Run golf()`;
  - `prim ▾` (dropdown of primitives);
  - `compile ▾` (primitive + size fixed/gated + from all/train).

  Each button runs a one-line script `model grid:taskNNN` / `solve …` and shows the verdict.
- For `compile`, a panel shows `src_grid` as a 30×30 grid of numbers (small font), and `P_preview` as a heatmap.
- A button `Open in editor` copies the generated script into Tab 1.

### Tab 3 "Assist"

- A text area: "Paste a paper abstract or describe an idea".
- A target model selector (`pythia-14m`, `neox-tiny`, or the currently selected ARC task).
- Three role selectors, prefilled with the defaults of §8.1.
- A cost line after the run.
- Buttons `Translate & run`, and `Translate only` (`execute=false`).
- **Results:**
  - the three raw replies (collapsible);
  - the parse errors;
  - the combined script (with a button `Open in editor`);
  - the verdict table with two extra columns, `predicted` (critic) and `agree?`.

### Tab 4 "Language"

- The language card (Appendix E.1), rendered as preformatted text.
- The op table from `/api/language`.

---

## 11. Milestones and acceptance tests

Run `pytest -q`. Network and live tests are opt-in: `pytest -m network` and `pytest -m live`.

### M0 Skeleton

- `pip install -e .` works.
- `wsbench --help` lists the commands.
- `wsbench serve` serves `/api/health`.
- **AT0.1** `GET /api/health` → 200 with `version` `"0.1.0"`.

### M1 Language (no torch needed in these tests)

- **AT1.1** Every statement in Appendix D parses without errors. `parse(print(parse(s))) == parse(s)` holds for each (compare ASTs).
- **AT1.2** These inputs give exactly these error codes (line 1 is `model pythia-14m`, the statement is line 2):

  | Input (line 2) | Code |
  |---|---|
  | `gauge rotate(resid, orth` | `E_SYNTAX` |
  | `gauge frobnicate(L3)` | `E_UNKNOWN_OP` |
  | `edit delete(L[7])` | `E_TARGET` |
  | `edit delete(L[mid+3])` | `E_TARGET` (mid = 3) |
  | `gauge vo(L3.H[4], gl)` | `E_TARGET` |
  | `solve prim(rot90)` | `E_MODEL` |
  | `train modadd(run=A)` | `E_MODEL` |
  | `edit swap(L2)` | `E_BAD_ARG` |
  | `gauge permute(L3.heads, perm([0, 0, 1, 2]))` | `E_BAD_ARG` |

- **AT1.3** A script whose first statement precedes any `model` directive gives `E_MODEL` for that statement.
- **AT1.4** Ranges: `[1:0.5:-0.1]` → `[1.0, 0.9, 0.8, 0.7, 0.6, 0.5]` (within 1e-12); `[0:1:0.25]` → 5 values; `[0:1:0]` → `E_BAD_ARG`.
- **AT1.5** Hints: each row of §3.8 has one test that triggers it and one that does not.

### M2 Executor on `neox-tiny` (offline)

- **AT2.1** Base loss of `neox-tiny` = **5.9561** (±1e-4).
- **AT2.2** Run the script of Appendix D.6 on `neox-tiny`. The statuses and measured classes must match Appendix F.1, column "tiny".
- **AT2.3** Determinism: running the same script twice gives identical `max_abs_dlogits` (bitwise).
- **AT2.4** `W0` is unchanged after any run: compare a hash of every tensor before and after.
- **AT2.5** An `edit` statement has a `control` object. A `gauge` statement has `control = null`.

### M3 `pythia-14m` (mark `network`; the download is 28 MB)

- **AT3.1** `wsbench fetch pythia-14m` verifies the SHA-256 values. The probe has 149 tokens.
- **AT3.2** Base loss **4.3177** (±1e-4); `max|ℓ0|` = 25.85 (±0.01).
- **AT3.3** Run Appendix D.1 + D.2 + the `gridop` line of D.4. The results must match Appendix F.1/F.2 column "pythia-14m":
  - statuses and classes exactly;
  - the deterministic `dloss` values within ±0.0005;
  - for `exact` rows, `rel ≤ 1e-9`.
- **AT3.4** Time: the whole AT3.3 script runs in under 30 s on a 4-core CPU (reference machine: about 4 s).

### M4 modadd (mark `slow`; about 5–10 min)

Run Appendix D.3. The checks:

- **AT4.1 (run A, seed 0, base):**
  - `first["train_acc>=0.99"] ≤ 300`;
  - `800 ≤ first["test_acc>=0.9"] ≤ 2500`;
  - final test_acc ≥ 0.99.
  - Reference: 150 / 1200; spike 150 / 1250.
- **AT4.2 (run F, Grokfast):** `first["test_acc>=0.9"]` exists and is ≤ 2500. Report the speed-up `first_A / first_F` in the verdict; the reference is about 1.25×. Do **not** assert that F is faster: the effect is small and depends on the machine.
- **AT4.3 (path P1, lerp A@200 → A@2000):**
  - test_acc at t = 1.0 is ≥ 0.99;
  - train_acc at t = 0.25 is < 0.95 (a barrier on the straight line; reference 0.81);
  - test_acc at t = 0.0 is in [0.40, 0.60] (memorised; reference 0.49).
- **AT4.4 (path P2, transplant):** test_acc at t = 1.0 is < 0.2 (reference 0.02).
- **AT4.5 (cosine C1):** |cosine| < 0.05 (reference −0.001).
- **AT4.6** While the script runs, `GET /api/runs/{id}` shows the `train` statement as `RUNNING`, with growing curves. Cancelling stops it within 2 s.

### M5 ARC

- **AT5.1 (offline):** using `tests/fixtures/golf_mini/` (Appendix D.5) as the golf directory:

  | Script | Result |
  |---|---|
  | `model grid:task087` / `solve compile(rot180, size=fixed)` | `CONFIRMED`, 8 / 8 |
  | `model grid:task150` / `solve compile(vmirror, size=fixed)` | `REFUTED`, 1 / 7 |
  | `model grid:task150` / `solve compile(vmirror, size=gated)` | `CONFIRMED`, 7 / 7 |
  | `model grid:task150` / `solve prim(vmirror)` | `CONFIRMED` |
  | `model grid:task087` / `solve chain(prim(vmirror), prim(hmirror))` | `CONFIRMED` |

- **AT5.2 (offline):** `decode(onehot(g)) == g` for 200 random grids (sizes 1–30, colours 0–9). `decode` of an all-zero array is `None`.
- **AT5.3 (network):** `wsbench fetch arc` yields 400 / 400 / 400 files. The index lists exactly the 7 single-primitive tasks of Appendix F.4.
- **AT5.4 (network):** For the 7 tasks, `prim`, `compile fixed` and `compile gated` give exactly the counts of Appendix F.4.
- **AT5.5 (network):** `solve golf()` on all 400 tasks. **Report** the number of `CONFIRMED` tasks in `NOTES.md`; the source claims all 400 pass. Assert ≥ 390. Any failing task must have a `first_failure` with the error text.

### M6 UI

- **AT6.1** `test_api.py` covers every endpoint of §9 with FastAPI's `TestClient`: shapes of the responses, plus a 404 for an unknown run.
- **AT6.2** A manual checklist; record the results in `NOTES.md`:
  1. Load each example and run it.
  2. Open a heatmap triple for `R11` (the MLP rotation), and the ARC view of a 16×16 block.
  3. Insert a gridop from the block selector and run it.
  4. Open task150, run `golf()`, `compile fixed` and `compile gated`.
  5. Load task087 into the editor.
  6. Charts appear for D.3.

### M7 Assist

- **AT7.1 (mock):**
  - `POST /api/assist` with `WSB_LLM=mock` returns the three role results;
  - a script with the statements `t1, t2, t3`, `fix2` and `alt1`;
  - `predictions` for `t1..t3`;
  - after the run finishes, `agree` values exactly as in Appendix E.5.
- **AT7.2 (mock):** A translate reply with a syntax error triggers exactly one repair call. The canned repair reply then parses.
- **AT7.3 (live, opt-in; costs about $0.05):** One call per default model with the text from Appendix E.5. Each returns `ok = true` or an explicit refusal. Record the costs in `NOTES.md`.

---

## 12. Out of scope for Phase 1 (Phase 2, done by Claude afterwards)

Do **not** build these. They are listed so that the Phase 1 design leaves room for them:

1. `equiv` statements: a weight edit equals an activation intervention (directional ablation, steering vectors as bias edits), with forward hooks.
2. TRM / HRM (the published ARC models: 6.8 M and 27 M core parameters, plus 0.9–1.0 M puzzle-embedding rows):
   - a forward pass;
   - statements about puzzle-embedding rows (e.g. "the rows of the D4 augmentations of a task are related by one shared linear map");
   - the published 40-checkpoint training trajectory.
3. Export of compiled networks as NeuroGolf ONNX files, with NeuroGolf scoring.
4. Path statements over the 154 public Pythia training checkpoints; larger Pythia models.
5. A status lattice with provenance (THEOREM / EMPIRICAL / OPEN / FALSE plus paper references), and a source pack of papers newer than the models' knowledge cutoffs.
6. Integration into the `cutoffmix` relay: the three models write WSL into shared notes, the executor appends observations, and routing follows each model's measured topic freshness.
7. Multi-seed dynamics with `vmap`, and GPU sweeps.
8. Tracr / ALTA-compiled ground-truth models.

---

## 13. Licences and attribution (show on the "Language" tab, at the bottom)

| Item | Licence |
|---|---|
| WSBench code | MIT |
| pythia-14m (EleutherAI) | Apache-2.0 |
| Code Golf 2025 task files and solutions (SakanaAI/google-code-golf-2025) | Apache-2.0. The task files contain ARC-AGI-1 data (Apache-2.0, F. Chollet) and ARC-GEN examples (Apache-2.0, Google). |
| Grid-primitive semantics (michaelhodel/arc-dsl) | MIT. Reimplemented, not copied. |
| ARC palette | from the Code Golf 2025 utilities (Apache-2.0, Google) |

---

## Appendix A: `wsbench/models/neox.py` (verified reference, copy verbatim)

Float64 GPT-NeoX forward pass, random group elements and all weight-space operators. Tested against the pythia-14m numbers in Appendix F. The `forward` matches an independent implementation to 1.8e-11 in logits.

```python
"""Reference: GPT-NeoX forward pass and weight-space operators, float64, plain torch.

A model state is (W, norm_kind): W maps parameter names to float64 tensors, norm_kind is 'ln' or 'rms'.
Every operator is a pure function W -> W (it returns a new dict and never mutates its input).
"""
import itertools
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

P = 'gpt_neox.layers.{}.'
READERS = ['attention.query_key_value', 'mlp.dense_h_to_4h']   # read the residual stream through a norm
WRITERS = ['attention.dense', 'mlp.dense_4h_to_h']             # add into the residual stream


@dataclass(frozen=True)
class NeoxCfg:
    d: int            # hidden_size
    n_heads: int      # num_attention_heads
    n_layers: int     # num_hidden_layers
    d_mlp: int        # intermediate_size
    vocab: int        # vocab_size
    rotary_pct: float
    rope_base: float  # rotary_emb_base
    eps: float        # layer_norm_eps

    @property
    def dh(self):     # head dimension
        return self.d // self.n_heads

    @property
    def rd(self):     # rotary dimensions per head (the first rd dims of each head)
        return int(self.dh * self.rotary_pct)


def cfg_from_hf_config(c):
    assert c['use_parallel_residual'] is True and c['hidden_act'] == 'gelu'
    return NeoxCfg(d=c['hidden_size'], n_heads=c['num_attention_heads'], n_layers=c['num_hidden_layers'],
                   d_mlp=c['intermediate_size'], vocab=c['vocab_size'], rotary_pct=c['rotary_pct'],
                   rope_base=c.get('rotary_emb_base', c.get('rope_theta', 10000)), eps=c['layer_norm_eps'])


TINY_CFG = NeoxCfg(d=16, n_heads=2, n_layers=4, d_mlp=64, vocab=64, rotary_pct=0.5, rope_base=10000.0, eps=1e-5)


def make_tiny(seed=0, cfg=TINY_CFG):
    """Random weights for offline tests. Deterministic given the seed."""
    g = torch.Generator().manual_seed(seed)
    n = lambda *s, std=0.2: torch.randn(*s, generator=g, dtype=torch.float64) * std
    D, V, DM = cfg.d, cfg.vocab, cfg.d_mlp
    W = {'gpt_neox.embed_in.weight': n(V, D, std=1.0), 'embed_out.weight': n(V, D, std=0.5),
         'gpt_neox.final_layer_norm.weight': 1 + n(D, std=0.1), 'gpt_neox.final_layer_norm.bias': n(D, std=0.1)}
    for i in range(cfg.n_layers):
        p = P.format(i)
        for ln in ('input_layernorm', 'post_attention_layernorm'):
            W[p + ln + '.weight'] = 1 + n(D, std=0.1)
            W[p + ln + '.bias'] = n(D, std=0.1)
        W[p + 'attention.query_key_value.weight'] = n(3 * D, D, std=D ** -0.5)
        W[p + 'attention.query_key_value.bias'] = n(3 * D, std=0.1)
        W[p + 'attention.dense.weight'] = n(D, D, std=D ** -0.5)
        W[p + 'attention.dense.bias'] = n(D, std=0.1)
        W[p + 'mlp.dense_h_to_4h.weight'] = n(DM, D, std=D ** -0.5)
        W[p + 'mlp.dense_h_to_4h.bias'] = n(DM, std=0.1)
        W[p + 'mlp.dense_4h_to_h.weight'] = n(D, DM, std=DM ** -0.5)
        W[p + 'mlp.dense_4h_to_h.bias'] = n(D, std=0.1)
    return W


TINY_IDS = [(7 * i + 3) % 64 for i in range(32)]


# ---------------------------------------------------------------- forward

def _norm(cfg, x, w, b, kind):
    if kind == 'ln':
        return F.layer_norm(x, (cfg.d,), w, b, cfg.eps)
    if kind == 'rms':
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + cfg.eps) * w + b
    raise ValueError(kind)


def _rope(cfg, T):
    rd = cfg.rd
    inv = 1.0 / (cfg.rope_base ** (torch.arange(0, rd, 2, dtype=torch.float64) / rd))
    fr = torch.outer(torch.arange(T, dtype=torch.float64), inv)
    emb = torch.cat([fr, fr], -1)
    return emb.cos(), emb.sin()


def _apply_rope(cfg, x, cos, sin):          # x [B,H,T,DH]; rotate the first rd dims, pass the rest
    rd = cfg.rd
    xr, xp = x[..., :rd], x[..., rd:]
    a, b = xr[..., :rd // 2], xr[..., rd // 2:]
    rot_half = torch.cat([-b, a], -1)
    return torch.cat([xr * cos + rot_half * sin, xp], -1)


def forward(cfg, W, ids, kind='ln', return_resid=False):
    """ids: LongTensor [B,T]. Returns logits [B,T,V] (float64); optionally the residual stream after each layer."""
    B, T = ids.shape
    D, H, DH = cfg.d, cfg.n_heads, cfg.dh
    cos, sin = _rope(cfg, T)
    mask = torch.full((T, T), float('-inf'), dtype=torch.float64).triu(1)
    x = W['gpt_neox.embed_in.weight'][ids]
    resid = [x]
    for i in range(cfg.n_layers):
        p = P.format(i)
        h = _norm(cfg, x, W[p + 'input_layernorm.weight'], W[p + 'input_layernorm.bias'], kind)
        qkv = h @ W[p + 'attention.query_key_value.weight'].T + W[p + 'attention.query_key_value.bias']
        qkv = qkv.view(B, T, H, 3 * DH).permute(0, 2, 1, 3)       # per head: [q | k | v], each DH wide
        q, k, v = qkv[..., :DH], qkv[..., DH:2 * DH], qkv[..., 2 * DH:]
        q, k = _apply_rope(cfg, q, cos, sin), _apply_rope(cfg, k, cos, sin)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(DH) + mask, -1) @ v
        a = a.permute(0, 2, 1, 3).reshape(B, T, D)
        attn = a @ W[p + 'attention.dense.weight'].T + W[p + 'attention.dense.bias']
        h2 = _norm(cfg, x, W[p + 'post_attention_layernorm.weight'], W[p + 'post_attention_layernorm.bias'], kind)
        mlp = F.gelu(h2 @ W[p + 'mlp.dense_h_to_4h.weight'].T + W[p + 'mlp.dense_h_to_4h.bias'])
        mlp = mlp @ W[p + 'mlp.dense_4h_to_h.weight'].T + W[p + 'mlp.dense_4h_to_h.bias']
        x = x + attn + mlp                                         # parallel residual
        resid.append(x)
    x = _norm(cfg, x, W['gpt_neox.final_layer_norm.weight'], W['gpt_neox.final_layer_norm.bias'], kind)
    logits = x @ W['embed_out.weight'].T
    if 'embed_out.bias' in W:                                      # exists only after fold_ln
        logits = logits + W['embed_out.bias']
    return (logits, resid) if return_resid else logits


def next_token_loss(logits, ids):
    return F.cross_entropy(logits[0, :-1], ids[0, 1:]).item()


# ---------------------------------------------------------------- random group elements

def rand_orth(n, g):
    q, r = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return q * torch.sign(torch.diagonal(r))


def rand_orth_fixing_ones(n, g):
    """Random R in O(n) with R @ ones = ones."""
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    A[:, 0] = 1.0
    Hb, _ = torch.linalg.qr(A)
    if Hb[:, 0].sum() < 0:
        Hb = -Hb
    Q = torch.eye(n, dtype=torch.float64)
    Q[1:, 1:] = rand_orth(n - 1, g)
    return Hb @ Q @ Hb.T


def rand_gl(n, g):
    """Random well-conditioned invertible matrix."""
    return torch.randn(n, n, generator=g, dtype=torch.float64) / math.sqrt(n) + 2 * torch.eye(n, dtype=torch.float64)


def rand_perm(n, g):
    return torch.randperm(n, generator=g)


def rand_uniform(n, lo, hi, g):
    return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)


# ---------------------------------------------------------------- operators (pure: W -> W)

def cp(W):
    return {k: v.clone() for k, v in W.items()}


def _ln_of(reader):
    return 'input_layernorm' if reader.startswith('attention') else 'post_attention_layernorm'


def fold_ln(cfg, W):
    """Move every norm's gamma/beta into the following linear map. Adds the new parameter embed_out.bias."""
    W = cp(W)
    D = cfg.d
    for i in range(cfg.n_layers):
        p = P.format(i)
        for r in READERS:
            g_, b_ = W[p + _ln_of(r) + '.weight'], W[p + _ln_of(r) + '.bias']
            W[p + r + '.bias'] = W[p + r + '.bias'] + W[p + r + '.weight'] @ b_
            W[p + r + '.weight'] = W[p + r + '.weight'] * g_
        for n in ('input_layernorm', 'post_attention_layernorm'):
            W[p + n + '.weight'] = torch.ones(D, dtype=torch.float64)
            W[p + n + '.bias'] = torch.zeros(D, dtype=torch.float64)
    g_, b_ = W['gpt_neox.final_layer_norm.weight'], W['gpt_neox.final_layer_norm.bias']
    W['embed_out.bias'] = W.get('embed_out.bias', torch.zeros(W['embed_out.weight'].shape[0], dtype=torch.float64)) \
        + W['embed_out.weight'] @ b_
    W['embed_out.weight'] = W['embed_out.weight'] * g_
    W['gpt_neox.final_layer_norm.weight'] = torch.ones(D, dtype=torch.float64)
    W['gpt_neox.final_layer_norm.bias'] = torch.zeros(D, dtype=torch.float64)
    return W


def center_writers(cfg, W):
    """Remove the all-ones component of everything written to the residual stream."""
    W = cp(W)
    C = torch.eye(cfg.d, dtype=torch.float64) - 1.0 / cfg.d
    W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] @ C
    for i in range(cfg.n_layers):
        p = P.format(i)
        for w in WRITERS:
            W[p + w + '.weight'] = C @ W[p + w + '.weight']
            W[p + w + '.bias'] = C @ W[p + w + '.bias']
    return W


def rotate_resid(cfg, W, R, layers=None, embed=True, unembed=True):
    """x -> R x on the residual stream: writers W -> R W (and biases), readers W -> W R^T."""
    W = cp(W)
    if embed:
        W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] @ R.T
    for i in (range(cfg.n_layers) if layers is None else layers):
        p = P.format(i)
        for r in READERS:
            W[p + r + '.weight'] = W[p + r + '.weight'] @ R.T
        for w in WRITERS:
            W[p + w + '.weight'] = R @ W[p + w + '.weight']
            W[p + w + '.bias'] = R @ W[p + w + '.bias']
    if unembed:
        W['embed_out.weight'] = W['embed_out.weight'] @ R.T
    return W


def scale_resid(cfg, W, c):
    """Multiply the embedding and every writer (weights and biases) by c."""
    W = cp(W)
    W['gpt_neox.embed_in.weight'] = W['gpt_neox.embed_in.weight'] * c
    for i in range(cfg.n_layers):
        p = P.format(i)
        for w in WRITERS:
            W[p + w + '.weight'] = W[p + w + '.weight'] * c
            W[p + w + '.bias'] = W[p + w + '.bias'] * c
    return W


def perm_mlp(cfg, W, i, perm):
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = W[p + 'mlp.dense_h_to_4h.weight'][perm]
    W[p + 'mlp.dense_h_to_4h.bias'] = W[p + 'mlp.dense_h_to_4h.bias'][perm]
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'][:, perm]
    return W


def scale_mlp(cfg, W, i, c):
    """Per-neuron positive rescale: input side * c, output side / c. Exact only for homogeneous activations (not GELU)."""
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = W[p + 'mlp.dense_h_to_4h.weight'] * c[:, None]
    W[p + 'mlp.dense_h_to_4h.bias'] = W[p + 'mlp.dense_h_to_4h.bias'] * c
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'] / c[None, :]
    return W


def rotate_mlp(cfg, W, i, Q):
    """Orthogonal mix of the hidden units (not a symmetry through GELU)."""
    W = cp(W); p = P.format(i)
    W[p + 'mlp.dense_h_to_4h.weight'] = Q @ W[p + 'mlp.dense_h_to_4h.weight']
    W[p + 'mlp.dense_h_to_4h.bias'] = Q @ W[p + 'mlp.dense_h_to_4h.bias']
    W[p + 'mlp.dense_4h_to_h.weight'] = W[p + 'mlp.dense_4h_to_h.weight'] @ Q.T
    return W


def qkv_rows(cfg, h, part):
    """Fused QKV layout: head h owns rows [h*3*DH, (h+1)*3*DH); inside, part 0 = q, 1 = k, 2 = v (DH rows each)."""
    s = h * 3 * cfg.dh + part * cfg.dh
    return slice(s, s + cfg.dh)


def perm_heads(cfg, W, i, perm):
    W = cp(W); p = P.format(i); DH = cfg.dh
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    ow = W[p + 'attention.dense.weight']
    W[p + 'attention.query_key_value.weight'] = torch.cat([qw[h * 3 * DH:(h + 1) * 3 * DH] for h in perm])
    W[p + 'attention.query_key_value.bias'] = torch.cat([qb[h * 3 * DH:(h + 1) * 3 * DH] for h in perm])
    W[p + 'attention.dense.weight'] = torch.cat([ow[:, h * DH:(h + 1) * DH] for h in perm], 1)
    return W


def vo_gauge(cfg, W, i, h, N):
    """v -> N v, W_O -> W_O N^-1 for head h of layer i."""
    W = cp(W); p = P.format(i); r = qkv_rows(cfg, h, 2); DH = cfg.dh
    W[p + 'attention.query_key_value.weight'][r] = N @ W[p + 'attention.query_key_value.weight'][r]
    W[p + 'attention.query_key_value.bias'][r] = N @ W[p + 'attention.query_key_value.bias'][r]
    o = W[p + 'attention.dense.weight']
    o[:, h * DH:(h + 1) * DH] = o[:, h * DH:(h + 1) * DH] @ torch.linalg.inv(N)
    return W


def qk_gauge(cfg, W, i, h, A, dims):
    """q_d -> A q_d, k_d -> A^-T k_d on the head dims listed in `dims` (LongTensor)."""
    W = cp(W); p = P.format(i)
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    rq, rk = qkv_rows(cfg, h, 0), qkv_rows(cfg, h, 1)
    iq = torch.arange(rq.start, rq.stop)[dims]
    ik = torch.arange(rk.start, rk.stop)[dims]
    AinvT = torch.linalg.inv(A).T
    qw[iq] = A @ qw[iq]; qb[iq] = A @ qb[iq]
    qw[ik] = AinvT @ qw[ik]; qb[ik] = AinvT @ qb[ik]
    return W


def rope_pairs(cfg, W, i, h, phis, scale):
    """The same 2x2 rotation phi_j on each RoPE pair (j, j + rd/2) of q AND k, then q * scale, k / scale."""
    assert len(phis) == cfg.rd // 2
    W = cp(W); p = P.format(i)
    qw, qb = W[p + 'attention.query_key_value.weight'], W[p + 'attention.query_key_value.bias']
    for part, s in ((0, scale), (1, 1.0 / scale)):
        base = qkv_rows(cfg, h, part).start
        for j, phi in enumerate(phis):
            a, b = base + j, base + j + cfg.rd // 2
            c, sn = math.cos(phi), math.sin(phi)
            for M in (qw, qb):
                ra, rb = M[a].clone(), M[b].clone()
                M[a] = s * (c * ra - sn * rb)
                M[b] = s * (sn * ra + c * rb)
        # non-rotary dims of this head are scaled too, so that q.k is unchanged on them
        rows = qkv_rows(cfg, h, part)
        for M in (qw, qb):
            M[rows.start + cfg.rd:rows.stop] = M[rows.start + cfg.rd:rows.stop] * s
    return W


# ---------------------------------------------------------------- layer values

def get_layer(W, i):
    """A layer value: dict of that layer's parameters, keys relative to the layer prefix."""
    pre = P.format(i)
    return {k[len(pre):]: W[k].clone() for k in W if k.startswith(pre)}


def set_layer(W, i, layer):
    W = cp(W)
    for k, v in layer.items():
        W[P.format(i) + k] = v.clone()
    return W


def zero_writers(W, layers):
    W = cp(W)
    for i in layers:
        for w in WRITERS:
            W[P.format(i) + w + '.weight'].zero_()
            W[P.format(i) + w + '.bias'].zero_()
    return W


def lerp_layer(X, Y, t):
    return {k: (1 - t) * X[k] + t * Y[k] for k in X}


def neg_writers(X):
    return {k: (-v if any(k.startswith(w) for w in WRITERS) else v.clone()) for k, v in X.items()}


def neg_all(X):
    return {k: (-v if 'layernorm' not in k else v.clone()) for k, v in X.items()}


def _layer_as_W(cfg, X):
    return {P.format(0) + k: v for k, v in X.items()}


def align_layer(cfg, X, Y):
    """Return X permuted (heads, then MLP neurons) to best match Y. A within-layer gauge move."""
    from scipy.optimize import linear_sum_assignment
    DH, H = cfg.dh, cfg.n_heads

    def head_sig(L_, h):
        w = L_['attention.query_key_value.weight'][h * 3 * DH:(h + 1) * 3 * DH]
        return torch.cat([w.flatten(), L_['attention.dense.weight'][:, h * DH:(h + 1) * DH].flatten()])

    sim = torch.tensor([[float(head_sig(Y, j) @ head_sig(X, k)) for k in range(H)] for j in range(H)])
    if H <= 8:
        best = max(itertools.permutations(range(H)), key=lambda pm: sum(float(sim[j, pm[j]]) for j in range(H)))
    else:
        _, col = linear_sum_assignment(-sim.numpy())
        best = tuple(int(c) for c in col)
    Wx = perm_heads(cfg, _layer_as_W(cfg, X), 0, list(best))
    Xh = get_layer(Wx, 0)
    fy = torch.cat([Y['mlp.dense_h_to_4h.weight'], Y['mlp.dense_h_to_4h.bias'][:, None], Y['mlp.dense_4h_to_h.weight'].T], 1)
    fx = torch.cat([Xh['mlp.dense_h_to_4h.weight'], Xh['mlp.dense_h_to_4h.bias'][:, None], Xh['mlp.dense_4h_to_h.weight'].T], 1)
    _, col = linear_sum_assignment(-(fy @ fx.T).numpy())
    Wx = perm_mlp(cfg, _layer_as_W(cfg, Xh), 0, torch.as_tensor(col))
    return get_layer(Wx, 0)


def gridop_block(W, name, key, r0, r1, c0, c1, grid_fn):
    """Apply a grid primitive (function on a 2-D tensor) to the block W[key][r0:r1, c0:c1] in place of the block."""
    W = cp(W)
    blk = W[key][r0:r1, c0:c1]
    out = grid_fn(blk)
    assert out.shape == blk.shape, f'{name} changes the block shape {tuple(blk.shape)} -> {tuple(out.shape)}'
    W[key][r0:r1, c0:c1] = out
    return W
```

---

## Appendix B: `wsbench/models/modadd.py` (verified reference, copy verbatim)

The modular-addition grokking model (Nanda et al. 2023: one layer, no LayerNorm, ReLU MLP, full-batch AdamW with weight decay 1), the trainer with Grokfast-EMA and scale interventions, and evaluation.

```python
"""Reference: modular-addition grokking model (Nanda et al. 2023, 1-layer, no LayerNorm), trainer, Grokfast-EMA filter."""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_data(p=97, frac=0.5, split_seed=1234):
    """All pairs (a, b) with label (a + b) mod p. Token p is '='. Fixed split independent of the model seed."""
    X = torch.cartesian_prod(torch.arange(p), torch.arange(p))
    X = torch.cat([X, torch.full((p * p, 1), p)], 1)
    Y = (X[:, 0] + X[:, 1]) % p
    perm = torch.randperm(p * p, generator=torch.Generator().manual_seed(split_seed))
    n_tr = int(frac * p * p)
    return X[perm[:n_tr]], Y[perm[:n_tr]], X[perm[n_tr:]], Y[perm[n_tr:]]


class OneLayer(nn.Module):
    """1-layer attention + ReLU MLP transformer, untied unembed, readout at the '=' position only."""

    def __init__(s, vocab, d=128, H=4, dmlp=512, n_ctx=3):
        super().__init__()
        s.H, s.dh = H, d // H
        s.E = nn.Parameter(torch.randn(vocab, d) / math.sqrt(d))
        s.P = nn.Parameter(torch.randn(n_ctx, d) / math.sqrt(d))
        s.WQ = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WK = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WV = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.WO = nn.Parameter(torch.randn(d, d) / math.sqrt(d))
        s.Win = nn.Parameter(torch.randn(d, dmlp) / math.sqrt(d))
        s.bin = nn.Parameter(torch.zeros(dmlp))
        s.Wout = nn.Parameter(torch.randn(dmlp, d) / math.sqrt(dmlp))
        s.bout = nn.Parameter(torch.zeros(d))
        s.U = nn.Parameter(torch.randn(d, vocab - 1) / math.sqrt(vocab))

    def forward(s, tok):                                   # tok [B,3] -> logits [B, p]
        x = s.E[tok] + s.P
        B = x.shape[0]
        q = (x[:, -1] @ s.WQ).view(B, 1, s.H, s.dh)
        k = (x @ s.WK).view(B, 3, s.H, s.dh)
        v = (x @ s.WV).view(B, 3, s.H, s.dh)
        att = torch.softmax((q * k).sum(-1) / math.sqrt(s.dh), dim=1)
        r = x[:, -1] + (att.unsqueeze(-1) * v).sum(1).reshape(B, -1) @ s.WO
        r = r + F.relu(r @ s.Win + s.bin) @ s.Wout + s.bout
        return r @ s.U


def evaluate(model, data):
    Xtr, Ytr, Xte, Yte = data
    with torch.no_grad():
        lo_tr, lo_te = model(Xtr), model(Xte)
        return dict(train_acc=(lo_tr.argmax(-1) == Ytr).float().mean().item(),
                    test_acc=(lo_te.argmax(-1) == Yte).float().mean().item(),
                    train_loss=F.cross_entropy(lo_tr, Ytr).item(),
                    test_loss=F.cross_entropy(lo_te, Yte).item(),
                    wnorm=math.sqrt(sum((q ** 2).sum().item() for q in model.parameters())))


def gradfilter_ema(model, ema, alpha, lamb):
    """Grokfast-EMA (Lee et al. 2024): g <- g + lamb * EMA_alpha(g). Returns the updated EMA state."""
    if ema is None:
        ema = {n: q.grad.detach().clone() for n, q in model.named_parameters() if q.grad is not None}
    for n, q in model.named_parameters():
        if q.grad is not None:
            ema[n].mul_(alpha).add_(q.grad.detach(), alpha=1 - alpha)
            q.grad.add_(ema[n], alpha=lamb)
    return ema


def train(p=97, frac=0.5, steps=3000, seed=0, lr=1e-3, wd=1.0, filt=None, interventions=(), save_steps=(),
          eval_every=50, threads=1, on_eval=None, should_stop=None):
    """Full-batch AdamW. filt: None or ('grokfast', alpha, lamb). interventions: list of (step, 'scale', c), applied
    to all parameters (theta <- c * theta) after the evaluation/checkpoint of that step index and before its
    gradient computation.
    Returns dict(curves, first, checkpoints) where checkpoints maps step -> state_dict (CPU tensors)."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    data = make_data(p, frac)
    Xtr, Ytr = data[0], data[1]
    m = OneLayer(p + 1)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.98))
    ema = None
    curves = {k: [] for k in ('step', 't', 'train_loss', 'train_acc', 'test_loss', 'test_acc', 'wnorm')}
    first, ckpts = {}, {}
    thresholds = [('train_acc>=0.99', 'train_acc', 0.99), ('test_acc>=0.5', 'test_acc', 0.5),
                  ('test_acc>=0.9', 'test_acc', 0.9), ('test_acc>=0.99', 'test_acc', 0.99)]
    iv = {int(s): (kind, c) for s, kind, c in interventions}
    t0 = time.time()
    for step in range(steps + 1):
        if step in save_steps:
            ckpts[step] = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if step % eval_every == 0 or step == steps:
            ev = evaluate(m, data)
            ev['step'], ev['t'] = step, time.time() - t0
            for k in curves:
                curves[k].append(ev[k])
            for name, key, thr in thresholds:
                if ev[key] >= thr and name not in first:
                    first[name] = step
            if on_eval:
                on_eval(ev)
        if step == steps or (should_stop and should_stop()):
            break
        if step in iv and iv[step][0] == 'scale':
            with torch.no_grad():
                for q in m.parameters():
                    q.mul_(iv[step][1])
        loss = F.cross_entropy(m(Xtr), Ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if filt is not None and filt[0] == 'grokfast':
            ema = gradfilter_ema(m, ema, filt[1], filt[2])
        opt.step()
    return dict(curves=curves, first=first, checkpoints=ckpts)


def eval_state(state_dict, p=97, frac=0.5):
    m = OneLayer(p + 1)
    m.load_state_dict(state_dict)
    return evaluate(m, make_data(p, frac))
```

---

## Appendix C: `wsbench/arc/grid.py` (verified reference, copy verbatim)

Grid primitives (arc-dsl semantics), the one-hot NeuroGolf encoding, decoding, the permutation compiler and the official pass rule.

```python
"""Reference: ARC grid primitives, NeuroGolf-style one-hot encoding, primitive -> permutation-network compiler."""
import numpy as np

# Official ARC colours (index 0..9) as used by the 2025 Google Code Golf utilities (Apache-2.0).
ARC_PALETTE = [(0, 0, 0), (30, 147, 255), (250, 61, 49), (78, 204, 48), (255, 221, 0),
               (153, 153, 153), (229, 59, 163), (255, 133, 28), (136, 216, 241), (147, 17, 49)]

# Semantics identical to michaelhodel/arc-dsl (rot90 is CLOCKWISE). g is a 2-D numpy array (rows x cols).
PRIMITIVES = {
    'identity': lambda g: g,
    'rot90':    lambda g: np.rot90(g, -1),
    'rot180':   lambda g: np.rot90(g, 2),
    'rot270':   lambda g: np.rot90(g, 1),
    'hmirror':  lambda g: g[::-1, :],          # flip top <-> bottom
    'vmirror':  lambda g: g[:, ::-1],          # flip left <-> right
    'dmirror':  lambda g: g.T,                 # transpose (main diagonal)
    'cmirror':  lambda g: np.rot90(g, 2).T,    # counter-diagonal
}
SHAPE_PRESERVING_ON_SQUARE = list(PRIMITIVES)   # all of them keep an n x n block n x n


def upscale(g, k):
    return np.kron(g, np.ones((k, k), dtype=g.dtype))


def onehot(grid):
    """ARC grid (list of lists, values 0..9, at most 30x30) -> float32 array (10, 30, 30); grid at the top-left,
    padding cells are all-zero in every channel."""
    g = np.array(grid, dtype=np.int64)
    h, w = g.shape
    assert h <= 30 and w <= 30
    x = np.zeros((10, 30, 30), dtype=np.float32)
    x[g, np.arange(h)[:, None], np.arange(w)[None, :]] = 1.0
    return x


def decode(y):
    """(10, 30, 30) -> grid (list of lists) or None if not decodable. A cell is active if any channel > 0.
    Height/width = extent of active cells from the top-left. Every cell inside must be active with exactly one
    channel > 0, every cell outside must be inactive."""
    pos = y > 0
    active = pos.any(0)
    if not active.any():
        return None
    rows = np.where(active.any(1))[0]
    cols = np.where(active.any(0))[0]
    h, w = rows.max() + 1, cols.max() + 1
    inside = np.zeros_like(active)
    inside[:h, :w] = True
    if (active != inside).any() or (pos[:, :h, :w].sum(0) != 1).any():
        return None
    return pos[:, :h, :w].argmax(0).tolist()


def compile_permutation(h, w, prim):
    """Index map src[900]: output cell j (= r*30 + c) copies input cell src[j]; -1 means 'stays empty'.
    Built for an input of exactly h x w placed top-left. The network is y[k, j] = x[k, src[j]] for every
    colour channel k, i.e. the weight matrix kron(I_10, P) with P[j, src[j]] = 1."""
    idx = np.arange(900).reshape(30, 30)
    out = PRIMITIVES[prim](idx[:h, :w])
    src = np.full(900, -1, dtype=np.int64)
    oh, ow = out.shape
    for r in range(oh):
        for c in range(ow):
            src[r * 30 + c] = out[r, c]
    return src


def apply_permutation_net(src, x):
    """x (10, 30, 30) -> y (10, 30, 30)."""
    xf = x.reshape(10, 900)
    y = np.zeros_like(xf)
    m = src >= 0
    y[:, m] = xf[:, src[m]]
    return y.reshape(10, 30, 30)


def dense_weight_from_src(src):
    """Dense 900 x 900 matrix P (for display only; 810,000 entries, 1 per output cell)."""
    P = np.zeros((900, 900), dtype=np.float32)
    for j, s in enumerate(src):
        if s >= 0:
            P[j, s] = 1.0
    return P


class CompiledNet:
    """size='fixed': one permutation compiled for the first example's input size.
    size='gated': one permutation per input size seen in the task's examples (a size gate selects it)."""

    def __init__(self, prim, examples, size='fixed'):
        self.prim, self.size = prim, size
        shapes = [np.array(e['input']).shape for e in examples]
        if size == 'fixed':
            self.table = {'*': compile_permutation(*shapes[0], prim)}
        elif size == 'gated':
            self.table = {s: compile_permutation(*s, prim) for s in dict.fromkeys(shapes)}
        else:
            raise ValueError(size)

    def __call__(self, grid):
        shape = np.array(grid).shape
        src = self.table.get('*', self.table.get(shape))
        if src is None:
            return None
        return decode(apply_permutation_net(src, onehot(grid)))

    def n_params_dense(self):
        return len(self.table) * 900 * 900   # with colour-channel sharing


def check(fn, examples):
    """Official Code Golf rule: the output must equal the expected grid exactly. Returns (passed, total, first_fail)."""
    passed, first_fail = 0, None
    for i, e in enumerate(examples):
        try:
            got = fn([row[:] for row in e['input']])
            ok = got is not None and np.array_equal(np.array(got), np.array(e['output']))
        except Exception as exc:   # noqa: BLE001
            ok, got = False, f'error: {exc!r}'
        if ok:
            passed += 1
        elif first_fail is None:
            first_fail = dict(index=i, input=e['input'], expected=e['output'], got=got)
    return passed, len(examples), first_fail
```

---

## Appendix D: example scripts, probe text and fixtures

Store D.1–D.4 as `wsbench/examples/*.wsl`, exactly as shown (line numbers matter for the random seeds). Store D.6 as `tests/data/tiny_battery.wsl`.

### D.1 `rotate_middle_layer.wsl`, and the pythia-14m probe text

```
model pythia-14m
seed 0
# Idea 1: "all weights in the middle layer rotate" - the precise readings
R1: gauge fold_ln(); rotate(resid, orth1)
R2: gauge rotate(resid, orth1)
R3: gauge fold_ln(); rotate(resid, orth)
R4: gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
R5: edit rotate(resid, orth, layers=[3], embed=false, unembed=false)
R6: gauge vo(L[mid].H[0], gl)
R7: gauge qk(L[mid].H[0], gl, dims=pass)
R8: gauge qk(L[mid].H[0], gl, dims=all)
R9: gauge rope_pairs(L[mid].H[0], phis=[0.3, -1.1, 2.0, 0.7], scale=1.7)
R10: gauge permute(L[mid].mlp, randperm)
R11: gauge rotate(L[mid].mlp, orth)
R12: gauge permute(L[mid].heads, randperm)
R13: gauge scale(L[mid].mlp, uniform(0.5, 2))
R14: reparam fold_ln()
R15: reparam fold_ln(); center_writers(); ln_to_rms()
R16: gauge scale(resid, 2)
R17: approx(0.05) scale(resid, 2)
```

`PROBE_TEXT` for pythia-14m is the following Python string, exactly. It contains literal `\n` newlines and the four-space indentation shown.

```python
PROBE_TEXT = ("The river leaves the mountains in late spring, when the snow begins to melt and the water turns "
        "grey with silt. Farmers in the valley wait for this moment, because the flood brings soil to the "
        "fields and fills the canals that run between the villages. In the old town, the market opens at "
        "dawn. Traders set out baskets of bread, cheese and dried fish, and children carry water from the "
        "well to the stalls. By noon the square is crowded, and the sound of bargaining fills the narrow "
        "streets. When the sun goes down, the lamps are lit one by one, and the town falls quiet again. "
        "def add(a, b):\n    return a + b\n\nprint(add(2, 3))\n")
```

### D.2 `antisymmetric_pair.wsl`

```
model pythia-14m
seed 0
# Idea 2: "layers before and after the middle antisymmetric -> the net interpolates in between"
A1: edit L[mid+1] := neg_writers(L[mid-1])
A2: edit delete(L[mid+1])
A3: edit delete(L[mid-1]); delete(L[mid+1])
A4: edit L[mid+1] := neg(L[mid-1])
A5: edit L[mid] := lerp(L[mid-1], L[mid+1], 0.5)
A6: edit delete(L[mid])
A7: edit swap(L[mid-1], L[mid])
A8: edit L[mid-1] := lerp(L[mid-1], L[mid], 0.5); L[mid] := L[mid-1]
A9: gauge L[mid] := align(L[mid] -> L[mid-1])
A10: edit L[mid-1] := lerp(L[mid-1], align(L[mid] -> L[mid-1]), 0.5); L[mid] := L[mid-1]
```

### D.3 `grokking_steerable.wsl`

```
model modadd
seed 0
# Idea 3: "is grokking steerable? what happens if I steer along grokking?"
G1: train modadd(run=A, p=97, frac=0.5, steps=2000, seed=0, save=[200, 2000])
G2: train modadd(run=F, p=97, frac=0.5, steps=2000, seed=0, filter=grokfast(0.98, 2.0))
G3: train modadd(run=B, p=97, frac=0.5, steps=2000, seed=1, save=[200, 2000])
G4: train modadd(run=S, p=97, frac=0.5, steps=2000, seed=0, intervene=[at(200, scale(0.7))])
P1: path lerp(ckpt(A, 200), ckpt(A, 2000), t=[-0.5:2:0.25])
P2: path add(ckpt(B, 200), diff(ckpt(A, 2000), ckpt(A, 200)), t=[0:1.5:0.25])
P3: path scale(ckpt(A, 200), c=[1:0.5:-0.1])
C1: path cosine(diff(ckpt(A, 2000), ckpt(A, 200)), diff(ckpt(B, 2000), ckpt(B, 200)))
```

### D.4 `arc_mirror.wsl`

```
model grid:task150
V1: solve golf()
V2: solve prim(vmirror)
V3: solve compile(vmirror, size=fixed)
V4: solve compile(vmirror, size=gated)
V5: solve compile(vmirror, size=gated, from=train)
V6: solve prim(hmirror)
model grid:task087
V7: solve chain(prim(vmirror), prim(hmirror))
V8: solve compile(rot180, size=fixed)
model pythia-14m
W1: edit gridop(rot90, block("gpt_neox.layers.3.attention.dense.weight", 0, 64, 0, 64))
```

### D.5 Offline fixtures `tests/fixtures/golf_mini/`

These are the train and test examples plus the first 3 arc-gen examples of two tasks, taken from the Apache-2.0 data. Each file holds one line of JSON.

`task087.json`:

```json
{"train":[{"input":[[2,2,1],[2,1,2],[2,8,1]],"output":[[1,8,2],[2,1,2],[1,2,2]]},{"input":[[9,2,4],[2,4,4],[2,9,2]],"output":[[2,9,2],[4,4,2],[4,2,9]]},{"input":[[8,8,8],[5,5,8],[8,5,5]],"output":[[5,5,8],[8,5,5],[8,8,8]]},{"input":[[3,2,9],[9,9,9],[2,3,3]],"output":[[3,3,2],[9,9,9],[9,2,3]]}],"test":[{"input":[[6,4,4],[6,6,4],[4,6,7]],"output":[[7,6,4],[4,6,6],[4,4,6]]}],"arc-gen":[{"input":[[8,5,3],[8,3,8],[5,5,8]],"output":[[8,5,5],[8,3,8],[3,5,8]]},{"input":[[8,2,2],[2,8,8],[2,8,2]],"output":[[2,8,2],[8,8,2],[2,2,8]]},{"input":[[2,1,2],[1,2,2],[8,8,2]],"output":[[2,8,8],[2,2,1],[2,1,2]]}]}
```

`task150.json`:

```json
{"train":[{"input":[[6,6,6,2],[6,1,6,2],[7,2,7,2],[1,7,2,2]],"output":[[2,6,6,6],[2,6,1,6],[2,7,2,7],[2,2,7,1]]},{"input":[[7,7,7,6,6,6,2],[6,7,1,1,7,7,1],[7,7,2,1,2,6,6],[2,2,7,7,7,2,2],[7,2,7,1,2,7,2],[6,6,6,2,2,1,1],[6,2,6,6,6,6,6]],"output":[[2,6,6,6,7,7,7],[1,7,7,1,1,7,6],[6,6,2,1,2,7,7],[2,2,7,7,7,2,2],[2,7,2,1,7,2,7],[1,1,2,2,6,6,6],[6,6,6,6,6,2,6]]},{"input":[[1,2,7,1,1,1],[2,1,7,7,2,6],[2,1,2,6,2,1],[1,2,1,7,6,2],[2,7,1,2,7,1],[2,1,6,2,7,7]],"output":[[1,1,1,7,2,1],[6,2,7,7,1,2],[1,2,6,2,1,2],[2,6,7,1,2,1],[1,7,2,1,7,2],[7,7,2,6,1,2]]}],"test":[{"input":[[7,6,1],[6,7,6],[6,2,2]],"output":[[1,6,7],[6,7,6],[2,2,6]]}],"arc-gen":[{"input":[[6,7,2,6,1,7,2],[6,7,7,6,6,6,2],[7,6,7,6,6,6,2],[2,6,2,7,7,1,6],[6,2,2,6,2,6,6],[2,1,6,7,1,7,2],[1,7,7,7,6,7,6]],"output":[[2,7,1,6,2,7,6],[2,6,6,6,7,7,6],[2,6,6,6,7,6,7],[6,1,7,7,2,6,2],[6,6,2,6,2,2,6],[2,7,1,7,6,1,2],[6,7,6,7,7,7,1]]},{"input":[[1,6,2],[7,7,7],[2,6,6]],"output":[[2,6,1],[7,7,7],[6,6,2]]},{"input":[[7,6,6],[1,7,2],[7,2,1]],"output":[[6,6,7],[2,7,1],[1,2,7]]}]}
```

For offline tests, the golf directory can be overridden with the environment variable `WSB_GOLF_DIR`, which points at a directory of `taskNNN.json` files. `golf()` statements then error with `E_DOWNLOAD`, because there are no solutions there.

### D.6 `tests/data/tiny_battery.wsl` (no comment lines: the line numbers 3–21 matter)

```
model neox-tiny
seed 0
T1: gauge fold_ln(); rotate(resid, orth1)
T2: gauge rotate(resid, orth1)
T3: gauge fold_ln(); rotate(resid, orth)
T4: gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
T5: edit rotate(resid, orth, layers=[2], embed=false, unembed=false)
T6: gauge vo(L[mid].H[0], gl)
T7: gauge qk(L[mid].H[0], gl, dims=pass)
T8: gauge qk(L[mid].H[0], gl, dims=all)
T9: gauge rope_pairs(L[mid].H[1], phis=[0.3, -1.1], scale=1.7)
T10: gauge permute(L[mid].mlp, randperm)
T11: gauge rotate(L[mid].mlp, orth)
T12: gauge permute(L[mid].heads, perm([1, 0]))
T13: gauge scale(L[mid].mlp, uniform(0.5, 2))
T14: reparam fold_ln(); center_writers(); ln_to_rms()
T15: approx(0.001) scale(resid, 2)
T16: gauge scale(resid, 2)
T17: gauge L[mid] := align(L[mid] -> L[mid-1])
T18: edit permute(L[1].mlp, randperm)
T19: edit L[mid] := lerp(L[mid-1], L[mid+1], 0.5)
```

---

## Appendix E: language card, prompts and mock replies (verbatim)

### E.1 `LANGUAGE_CARD`

This is the text shown in the "Language" tab and inserted into every prompt. Store it as a Python string constant in `wsbench/lang/card.py`.

```text
WSL v0.1 - Weight-Space Language. One statement per line:  [LABEL:] CLAIM OP; OP; ...   (# starts a comment)
Directives: "model NAME" (neox-tiny | pythia-14m | modadd | grid:taskNNN) and "seed N". Every statement starts from
the unedited model and is compared with it. Layers: L[i] or Li (0-based), L[mid], L[mid-1], L[mid+1]; heads L[i].H[h].

CLAIMS
  gauge      the pipeline is an exact symmetry (logits unchanged up to float64 round-off)  -> CONFIRMED / REFUTED
  reparam    function-preserving change of parametrisation (may add parameters)           -> CONFIRMED / REFUTED
  approx(t)  relative logit change <= t                                                     -> CONFIRMED / REFUTED
  edit       changes the function; measured as delta-loss, with a random control of equal norm -> MEASURED
  train      train a modular-addition model (grokking toy)                                  -> MEASURED
  path       evaluate points between / along checkpoints                                     -> MEASURED
  solve      an ARC grid program or compiled network must pass every example of the task     -> CONFIRMED / REFUTED

OPS ON GPT-NeoX MODELS (neox-tiny, pythia-14m: LayerNorm, GELU MLP, parallel residual, partial RoPE)
  fold_ln()                         move every LayerNorm gain/bias into the next matrix (adds embed_out.bias)
  center_writers()                  remove the all-ones component from everything written to the residual stream
  ln_to_rms()                       replace LayerNorm by RMSNorm (exact only after fold_ln(); center_writers())
  rotate(resid, orth|orth1, layers=all|[i,...], embed=true|false, unembed=true|false)
                                    x -> R x on the residual stream; orth1 means R fixes the all-ones vector
  scale(resid, c)                   multiply the embedding and all writers by c
  permute(L[i].mlp, randperm|perm([...]))      permute MLP neurons      scale(L[i].mlp, uniform(lo, hi))   rotate(L[i].mlp, orth)
  permute(L[i].heads, randperm|perm([...]))    permute attention heads
  vo(L[i].H[h], gl|orth)            V -> N V, O -> O N^-1 for one head
  qk(L[i].H[h], gl|orth, dims=pass|all)        Q -> A Q, K -> A^-T K on the non-rotary dims (pass) or all dims
  rope_pairs(L[i].H[h], phis=[...], scale=s)   same 2x2 rotation on each RoPE pair of Q and K; Q*s, K/s
  delete(L[i])   swap(L[i], L[j])   L[i] := L[j] | lerp(X, Y, t) | neg_writers(X) | neg(X) | align(X -> Y)
  gridop(PRIM, block("tensor name", r0, r1, c0, c1))    apply an ARC grid primitive to a block of a weight matrix
  PRIM: identity rot90 rot180 rot270 hmirror vmirror dmirror cmirror  (rot90 is clockwise; hmirror flips top/bottom)

OPS ON modadd
  train modadd(run=NAME, p=97, frac=0.5, steps=3000, seed=0, lr=0.001, wd=1.0, filter=none|grokfast(alpha, lamb),
               save=[steps], intervene=[at(STEP, scale(C))])
  path lerp(ckpt(R, s1), ckpt(R, s2), t=[a:b:step])     path add(ckpt(R, s), diff(ckpt(A, s2), ckpt(A, s1)), t=[...])
  path scale(ckpt(R, s), c=[a:b:step])                  path cosine(diff(...), diff(...))

OPS ON grid:taskNNN (ARC-AGI-1 task from the 2025 Code Golf set)
  solve prim(PRIM)    solve chain(prim(A), prim(B), ...)    solve golf()
  solve compile(PRIM, size=fixed|gated, from=all|train)   one-hot (10,30,30) permutation network; 'gated' = one per input size

FACTS MEASURED ON THESE MODELS (float64; exact = relative logit change about 1e-12)
  exact:  MLP neuron permutation; head permutation; V/O change of basis per head; Q/K change of basis on non-rotary dims;
          one 2x2 rotation per RoPE pair shared by Q and K (plus Q*s, K/s); fold_ln; center_writers; after those two,
          ln_to_rms; residual rotation of ALL readers, writers, embedding and unembedding after fold_ln with R fixing
          the all-ones vector (orth1), or with any R after center_writers + ln_to_rms; aligning a layer to another.
  not exact: residual rotation without fold_ln; with LayerNorm and R not fixing ones; of one layer only (a large edit,
          worse than deleting the layer); per-neuron positive rescaling through GELU; rotating MLP neurons; Q/K basis
          change on rotary dims; scaling the residual stream (only LayerNorm eps breaks it: approx).
  pythia-14m edits (delta-loss in nats, base 4.3177): delete middle layer +4.04; swap layers 2,3 +3.27;
          L4 := -L2 (writers negated) +2.51 (delete L4 alone +2.16, delete L2 and L4 +4.06); L3 := mean(L2, L4) +4.99.
  modadd (p=97, 50 % train): memorised by step ~150, generalises (test >= 0.9) by step ~1200; Grokfast ~1.2x earlier;
          the straight line memorised -> grokked has a barrier (train acc ~0.8 at t=0.25); the direction of one seed
          is orthogonal to another seed's (cosine ~0) and transplanting it destroys the other model.
```

### E.2 `TRANSLATE_SYSTEM`

````text
You translate claims about neural-network weights into WSL, a small formal language. A machine executes WSL statements on real weights and checks them.

{LANGUAGE_CARD}

Target model for your statements: {MODEL}.

Task: read the source text the user gives you. Write at most 8 WSL statements that test its claims about weights on the target model. Prefer informative statements: a claimed symmetry as gauge, a weight change whose cost should be measured as edit, and a control next to it.

Output format, exactly:
1. One fenced block that starts with ```wsl and ends with ```. The first line inside it is: model {MODEL}
2. Before each statement, one comment line: # claim: <one sentence: what the source asserts and what you expect>
3. Nothing after the block.

Use only the ops listed above. If the text makes no claim that these ops can test, return a block that contains only the model line and one comment line explaining why.
````

### E.3 `CRITIC_SYSTEM`

````text
You review WSL statements before a machine executes them. WSL is described below.

{LANGUAGE_CARD}

For every numbered statement, predict what the machine will measure:
- for gauge, reparam, approx and edit statements: "exact" (logits unchanged up to round-off), "near" (relative logit change at most 1e-3) or "edit" (a larger change);
- for solve statements: "pass" (every example passes) or "fail";
- "unknown" if you cannot tell.
Give a one-sentence reason. If a statement has the wrong claim or will fail with an error, put one corrected statement in "fix" (no label, no model line). Otherwise "fix" is "".

Output format, exactly: one fenced block that starts with ```json and ends with ```, containing a JSON list with one object per statement:
[{"n": 1, "predict": "exact", "reason": "...", "fix": ""}]
Nothing else.
````

### E.4 `ALT_SYSTEM`

````text
You design control experiments on neural-network weights, written in WSL. WSL is described below.

{LANGUAGE_CARD}

You get a source text and the statements already proposed. Write at most 3 additional statements whose outcome would change the conclusion if it differed. Examples: a control of equal size, the same operation on another layer or head, or the exact symmetry that a proposed edit approximates. Do not repeat a proposed statement.

Output format, exactly: one fenced block that starts with ```wsl and ends with ```. The first line inside it is: model {MODEL}
Before each statement, one comment line: # control: <what it controls for>
Nothing after the block.
````

`{LANGUAGE_CARD}` and `{MODEL}` are replaced with `str.replace`, not with `str.format`: the card contains braces.

### E.5 Mock replies (`assist/mock.py`) and the expected outcome

Source text used by the tests:

```text
SliceGPT shows that the residual stream of a transformer can be rotated by any orthogonal matrix without changing its outputs, once LayerNorm is converted to RMSNorm. Rotating only one layer's weights, however, breaks the model.
```

Reply of role `translate`:

````text
```wsl
model pythia-14m
# claim: a global residual rotation is a symmetry after converting LayerNorm (SliceGPT).
gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
# claim: without converting LayerNorm, the same rotation is not a symmetry.
gauge rotate(resid, orth)
# claim: rotating only one layer breaks the model.
edit rotate(resid, orth, layers=[2], embed=false, unembed=false)
```
````

Broken first reply of role `translate`, used only with the marker `[[mock-syntax-error]]`:

````text
```wsl
model pythia-14m
gauge rotate(resid, orth
```
````

Reply of role `critic`:

````text
```json
[{"n": 1, "predict": "exact", "reason": "Computational invariance after LayerNorm is converted.", "fix": ""},
 {"n": 2, "predict": "edit", "reason": "LayerNorm gains and centring pin the basis.", "fix": "gauge fold_ln(); rotate(resid, orth1)"},
 {"n": 3, "predict": "edit", "reason": "The other layers' readers and writers are not compensated.", "fix": ""}]
```
````

Reply of role `alternatives`:

````text
```wsl
model pythia-14m
# control: rotation restricted to matrices fixing the all-ones vector, LayerNorm kept.
gauge fold_ln(); rotate(resid, orth1)
```
````

**Expected outcome with target model `neox-tiny`.** The model lines are replaced (§8.2), so the script reads:

```
model neox-tiny
t1: gauge fold_ln(); center_writers(); ln_to_rms(); rotate(resid, orth)
t2: gauge rotate(resid, orth)
t3: edit rotate(resid, orth, layers=[2], embed=false, unembed=false)
fix2: gauge fold_ln(); rotate(resid, orth1)
alt1: gauge fold_ln(); rotate(resid, orth1)
```

The test asserts the statuses, classes and `agree` values below. It does not assert the `line` numbers.

| Statement | Status | Measured | `agree` |
|---|---|---|---|
| `t1` | CONFIRMED | exact | agree |
| `t2` | REFUTED | edit | agree |
| `t3` | MEASURED | edit | agree |
| `fix2` | CONFIRMED | exact | n/a |
| `alt1` | CONFIRMED | exact | n/a |

---

## Appendix F: expected results (measured on 2026-11-25, 4-core x86 CPU, torch 2.15 CPU, float64)

**Statuses and classes must match exactly.** The `dloss` values below come from deterministic statements, which use no random group elements. They must match within ±0.0005. Statements with random group elements have no fixed number; only their class is fixed.

### F.1 `rotate_middle_layer.wsl` on pythia-14m (base loss 4.3177, max |logit| 25.85, probe of 149 tokens)

| Label | Status | Measured | Hint expected | Notes |
|---|---|---|---|---|
| R1 | CONFIRMED | exact | – | |
| R2 | REFUTED | edit | `H_RESID_NOFOLD` | |
| R3 | REFUTED | edit | `H_RESID_ONES` | |
| R4 | CONFIRMED | exact | – | `norm_kind` = `rms`, `new_params` = `["embed_out.bias"]` |
| R5 | MEASURED | edit | – | dloss large and positive (reference draws: +6.4 to +7.1) |
| R6 | CONFIRMED | exact | – | |
| R7 | CONFIRMED | exact | – | |
| R8 | REFUTED | edit | `H_ROPE_QK` | |
| R9 | CONFIRMED | exact | – | |
| R10 | CONFIRMED | exact | – | |
| R11 | REFUTED | edit | `H_MLP_ROTATE` | |
| R12 | CONFIRMED | exact | – | |
| R13 | REFUTED | edit | `H_GELU_SCALE` | |
| R14 | CONFIRMED | exact | – | `new_params` = `["embed_out.bias"]` |
| R15 | CONFIRMED | exact | – | `norm_kind` = `rms` |
| R16 | REFUTED | edit | – | deterministic: rel 0.0175, max_abs_dlogits 0.4519, dloss +0.0002 |
| R17 | CONFIRMED | edit | – | same numbers; rel 0.0175 ≤ 0.05 |

### F.2 `antisymmetric_pair.wsl` on pythia-14m (mid = 3), plus `W1` from D.4

All rows are deterministic.

| Label | Status | Measured | dloss | Hint |
|---|---|---|---|---|
| A1 | MEASURED | edit | +2.5106 | – |
| A2 | MEASURED | edit | +2.1577 | – |
| A3 | MEASURED | edit | +4.0639 | – |
| A4 | MEASURED | edit | +6.2921 | – |
| A5 | MEASURED | edit | +4.9923 | `H_LERP_ALIGN` |
| A6 | MEASURED | edit | +4.0429 | – |
| A7 | MEASURED | edit | +3.2747 | – |
| A8 | MEASURED | edit | +2.9584 | `H_LERP_ALIGN` |
| A9 | CONFIRMED | exact | 0.0000 | – |
| A10 | MEASURED | edit | +2.6678 | – |
| W1 | MEASURED | edit | +0.1764 | – |

How to read it:

- The antisymmetric pair A1 neither cancels (A3 would be "cancels") nor interpolates.
- Interpolating the middle layer (A5) is worse than deleting it (A6).
- Averaging two layers needs alignment first: A10 < A8.

### F.3 `tiny_battery.wsl` on neox-tiny (base loss 5.9561)

| Label | Status | Measured | Notes |
|---|---|---|---|
| T1 | CONFIRMED | exact | |
| T2 | REFUTED | edit | |
| T3 | REFUTED | edit | |
| T4 | CONFIRMED | exact | |
| T5 | MEASURED | edit | |
| T6 | CONFIRMED | exact | |
| T7 | CONFIRMED | exact | |
| T8 | REFUTED | edit | |
| T9 | CONFIRMED | exact | |
| T10 | CONFIRMED | exact | |
| T11 | REFUTED | edit | |
| T12 | CONFIRMED | exact | |
| T13 | REFUTED | edit | |
| T14 | CONFIRMED | exact | |
| T15 | CONFIRMED | near | deterministic rel 7.16e-6 |
| T16 | REFUTED | near | deterministic rel 7.16e-6 |
| T17 | CONFIRMED | exact | |
| T18 | MEASURED | exact | warning `W_NOOP`; hint `H_EDIT_GAUGE` |
| T19 | MEASURED | edit | deterministic dloss −0.3300 (random tiny model: a negative dloss is fine) |

### F.4 ARC: the 7 tasks solved by a single primitive (all examples = train + test + arc-gen)

| Task | ARC id | Primitive | Examples | `prim` | `compile fixed` | `compile gated` (from=all) | `bytes_raw` |
|---|---|---|---|---|---|---|---|
| task087 | 3c9b0459 | rot180 | 266 | 266 | 266 | 266 | 36 |
| task140 | 6150a2bd | rot180 | 265 | 265 | 265 | 265 | 36 |
| task150 | 67a3c6ac | vmirror | 266 | 266 | **39** | 266 | 30 |
| task155 | 68b16354 | hmirror | 266 | 266 | **51** | 266 | 18 |
| task179 | 74dd1130 | dmirror | 267 | 267 | 267 | 267 | 21 |
| task241 | 9dfd6313 | dmirror | 266 | 266 | **32** | 266 | 21 |
| task380 | ed36ccf7 | rot270 | 267 | 267 | 267 | 267 | 27 |

Further expected results for D.4:

- `V3` `by_split` = train [1, 3], test [0, 1], arc-gen [38, 262]; `compiled.nnz` = 16.
- `V4` `compiled.sizes` lists 7 sizes (3×3 to 9×9); `nnz` = 280.
- `V5` (gated, from=train) = 112 / 266, `REFUTED`.
- `V6` (hmirror on task150) = 0 / 266.
- `V7` = 266 / 266.
- `V1` (golf) is expected to pass 266 / 266.

### F.5 modadd reference values (this machine, 2 threads; your step counts may differ slightly)

**Training runs** (first step at which each threshold was reached):

| Run | train ≥ 0.99 | test ≥ 0.5 | test ≥ 0.9 | test ≥ 0.99 | Wall time |
|---|---|---|---|---|---|
| A (seed 0, base) | 150 | 550 | 1200 | 1400 | 152 s |
| F (seed 0, Grokfast 0.98 / 2.0) | 200 | 200 | 1000 | 1200 | 150 s |
| B (seed 1, base) | 150 | 250 | 650 | 800 | – |

**Path P1** (A@200 → A@2000):

| t | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|
| test acc | 0.488 | 0.363 | 0.543 | 0.979 | 1.000 | 1.000 |
| train acc | 1.000 | **0.798** | 0.822 | 0.999 | 1.000 | 1.000 |

**Path P2** (B@200 + t·dir_A), test acc:

| t | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 |
|---|---|---|---|---|---|---|
| test acc | 0.488 | 0.258 | 0.095 | 0.045 | **0.024** | 0.018 |

**C1** cosine(dir_A, dir_B) = **−0.0014**.
