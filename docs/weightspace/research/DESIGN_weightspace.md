# WSI: weight-space instructions for cutoffmix (design proposal)

Date: 2026-11-25. Status: **proposal, nothing built.** No file under `/home/user/EscapeTheCutOff` was changed.

Evidence base: four lens reports in this directory (`arc.md`, `formal.md`, `ops.md`, `spike.md`). Their authors opened the
primary pages (arXiv, GitHub, Hugging Face, arcprize.org, PyPI) and ran CPU spikes. Numbers marked **measured** come
from those spikes: 4 vCPU Xeon, 15 GB RAM, no GPU, torch 2.15 CPU, float64 unless stated, on a machine shared with
other jobs. I re-opened three central sources today:

- iWSP: arXiv abs and HTML. Confirmed 17 tasks, a median of 166 parameters, about $17,500, with Claude Code agents.
- WSP-Scope 2611.09382: abs page.
- The official NeuroGolf 2026 site: organisers, dates and prizes. The site states no network format or scoring rules.

Anything not verified this way is marked **UNVERIFIED**. "Estimate" means a number that was not measured.

---

## Kurzfassung (Deutsch)

**Die Idee in einem Satz.** Du willst auf „Maschinencode“ arbeiten, also auf den Gewichten statt auf Prompts. Vorschlag:
eine kleine formale Sprache **WSI (Weight-Space Instructions)**. Jede Zeile ist eine Anweisung aus drei Teilen:

- *Operation*;
- *Ziel* (Layer, Head, Neuron, Richtung, Checkpoint);
- *erwartete, messbare Wirkung*, also eine falsifizierbare Behauptung.

Ein **Executor** wendet die Anweisung mit torch auf echte Modelldateien an und misst. Im Relay schreiben die drei Modelle
solche Anweisungen in die gemeinsamen Notizen. Zwischen den Abschnitten hängt der Executor das Messergebnis als
`Observation:` an. So arbeiten alle drei Modelle und du mit echten Zahlen statt mit Behauptungen.

**Wichtigste Einsicht aus den Messungen.** Derselbe Satz („der mittlere Layer rotiert“) bezeichnet ganz verschiedene
Dinge. WSI hat deshalb Anweisungstypen:

- `gauge`: exakte Symmetrie. Die Funktion bleibt gleich, die Prüfung dauert unter 1 s.
- `reparam`: funktionserhaltende Umformung, z. B. LayerNorm einfalten.
- `edit`: ändert die Funktion. Gemessen wird immer mit einer Zufallskontrolle gleicher Norm.
- `train`: Trainingsdynamik, z. B. Grokfast oder Weight Decay.
- `path`: Linien zwischen Checkpoints.
- `equiv`: Gewichts-Edit gleich Aktivierungs-Eingriff.
- `observe`: Messgrößen und Bilder.

**Deine drei Beispiele, mit Status:**

1. **„Alle Gewichte im mittleren Layer rotieren.“**
   - **Falsch** als Invarianz, wenn nur der mittlere Layer rotiert. Gemessen auf pythia-14m: Δloss +6,37 nats, schlimmer
     als den Layer zu löschen (+4,04).
   - **Exakt wahr** (Abweichung 2e-11) als *globale* Rotation des Residualstroms über alle Leser, Schreiber, Embedding und
     Unembedding, nach dem Einfalten der LayerNorm und mit R·1 = 1 (SliceGPT, Theorem 1).
   - Lokal im mittleren Layer exakt: Neuronen-Permutation, Head-Permutation und Basiswechsel pro Head auf V/O und Q/K
     (auf den RoPE-Dimensionen nur eingeschränkt).
2. **„Layer vor und nach der Mitte antisymmetrisch, dann interpoliert das Netz dazwischen.“**
   - Mathematisch bedeutet Antisymmetrie Rotation bzw. Konjugation, nicht Interpolation (Satz für lineare Schichten).
   - Gemessen auf pythia-14m: Das Paar hebt sich nicht auf. Der Residualstrom weicht um 104 % von „beide löschen“ ab.
   - „Mittlerer Layer := Mittel der Nachbarn“ ist **schlechter** als Löschen (+4,99 gegenüber +4,04).
   - Bei 7–8B-Modellen ist empirisch belegt:
     - Slerp-Interpolation benachbarter Layer als *zusätzlicher* Layer hilft (PLI).
     - Parallele mittlere Layer mit gemittelter Ausgabe funktionieren (Painters).
     - Den mittleren Layer zu kopieren ist **falsch**: Das Modell kollabiert.
3. **„Ist Grokking steuerbar?“**
   - **Ja, mäßig**, auf dem Toy-Task gemessen: Grokfast verschiebt Grokking von Schritt 9.900 auf 6.700 (1,48×, ein Seed;
     das Paper meldet über 50× in einem anderen Setup).
   - Im Gewichtsraum ist ein Lauf entlang der Linie „auswendig gelernt → gegrokkt“ steuerbar, mit einer Barriere.
   - Die Richtung eines anderen Seeds zu übertragen zerstört das Modell: Testgenauigkeit 0,024, die beiden Richtungen sind
     orthogonal (cos −0,001).
   - **Offen** und ein guter eigener Versuch: Rettet ein vorheriges Alignment (Permutationssymmetrie) die Übertragung?

**Grafischer Ansatz.** Gewichtsmatrizen werden als Raster gezeigt; Reihenfolge und Palette werden immer ausdrücklich
genannt. ARC-DSL-Operationen (rot90, mirror, upscale, crop, recolor, objects) werden auf Gewichtsoperationen
abgebildet.

- Die Analogie **hält**, wo eine Achse wirklich räumlich ist: Positionen, Faltungskerne, Token- und Farbzeilen.
- Sie **bricht** auf versteckten Achsen. Die sind beliebig permutierbar und werden erst nach einer Gauge-Fixierung
  sinnvoll.
- Vorschlag: kleine, gauge-invariante Bilder als ARC-Raster (Ziffern 0–9) direkt in die Notizen. Das ist das Format, in
  dem alle drei Modelle ARC-AGI-2 zu 85,8–93,75 % lösen.

**ARC und „Codebench“.**

- Einen ARC-Datensatz namens „Codebench“ gibt es nicht. Gemeint ist sehr wahrscheinlich die **Google Code Golf
  Championship 2025** (NeurIPS 2025). Dort ist jede der 400 ARC-AGI-1-Aufgaben ein möglichst kurzes Python-Programm;
  Sakana hat alle 400 Lösungen unter Apache-2.0 veröffentlicht.
- Dazu passt der Nachfolger **NeuroGolf 2026** (IJCAI-ECAI 2026): Jede Aufgabe wird als möglichst kleines neuronales Netz
  gebaut. Das ist wörtlich „ARC-Programm → Gewichte“.
- Veröffentlichte ARC-Modelle, die auf dieser CPU laufen:
  - **TRM**: 6,83 M Kern-Parameter, MIT.
  - **HRM**: 27,3 M Kern-Parameter, der Checkpoint hat keine Lizenz.

  95–98,5 % ihrer Gewichte sind eine Tabelle mit einer 512-d-Zeile pro (Aufgabe × Augmentation).
- Die 8B-Löser (ARChitects, BARC, NVARC) sind für Gewichtsarbeit auf der CPU zu groß.

**Relay.** Neues Preset `weightspace`:

- Fable ist bei ML-Papern am aktuellsten und übersetzt Befunde aus Papern in Anweisungen.
- Ein anderes Modell prüft und korrigiert.
- Der Executor misst.
- Was nach den Cutoffs erschienen ist, kommt über eine Quellenliste in die Notizen. Beispiel: iWSP (13.11.2026), wo
  Claude-Agenten Transformer-Gewichte direkt in Code schreiben.
- Fables Eingangsfilter hat im Explore-Preset beschriftete Schemata gesperrt. Fable bekommt deshalb wieder die
  abgeschwächte Vorgabe. Ob Gewichts-Themen den Filter auslösen, muss ein Pilot mit etwa 20 Cent-Aufrufen zeigen.

**Du als Editor.**

- `cutoffmix ws check` parst die Datei und meldet Fehler mit Zeile und Spalte.
- Die Typprüfung kontrolliert die Ziele gegen die Formen des geladenen Modells, z. B. „L7 gibt es nicht, pythia-14m hat
  L0–L5“ oder „rescale ist bei GELU keine Symmetrie, schreib `edit`, um es zu messen“.
- Dazu kommen Vervollständigung, ein Vorschaubild und `ws run` für die Messung.
- Die Datei kann auch direkt als Frage an das Relay gehen. Du schreibst `expect … ?`, die Modelle sagen voraus, und die
  Maschine misst.

**Erstes Build (nur CPU, ohne API-Kosten).**

1. Parser, Typprüfung und Executor auf pythia-14m und dem Grokking-Toy, CLI, deine drei Beispiele als Testdatei.
2. Danach das Relay-Preset: Pilot etwa 0,5 $, pro Lauf etwa 0,3–0,6 $ (Schätzung).
3. Danach die ARC-Scheibe: kompilierte Code-Golf-Netze und der TRM-Kern.

Ein B200 für höchstens 1 h bringt Phasendiagramme (tausend Grokking-Läufe parallel) und Pythia 1,4–6,9B. Für das erste
Build ist er nicht nötig.

**Ehrliche Grenzen.**

- Alles läuft auf Toy-Skala. Ergebnisse von 14M-Modellen übertragen sich nicht automatisch auf große Modelle; Grokfast
  hatte im Paper über 50×, hier 1,25–1,48×.
- Die Modelle sehen keine Gewichte, nur Zahlen und kleine Raster.
- „Synergie“ ist eine Hypothese, die das erste Build erst messbar macht.

**Was ich von dir brauche:**

1. Stimmt „Codebench“ = Code Golf 2025 / NeuroGolf 2026, oder meinst du eine andere Quelle?
2. Freigabe der kleinen Pilot-Kosten in Phase 2.
3. Soll TRM schon in das erste Build oder erst in Phase 3?

---

## 0. The request and how it is read

The owner wants:

- to operate on **weights** ("machine code"), not prompts;
- a formal language of weight-space statements that (1) supports him as a "weight-space editor" (a kind of prompting
  in a formal language) and (2) lets the three relay models produce synergy by working in it;
- "pseudo weight space" as a compromise: a finding from a paper is *translated* into a weight-space statement;
- a "graphical approach";
- an interface to the **published open-weight ARC-AGI models**, combined with "Codebench, because there it is
  formalised as a programming language".

Reading of "Codebench": no ARC or grid dataset of that name exists. A Hugging Face dataset search and a GitHub search
return only generic code benchmarks and unrelated projects; the closest name, CodeARC (arXiv 2503.23145), contains
general Python functions, not grids. The formalised ARC-as-code resources are:

- **2025 Google Code Golf Championship** (NeurIPS 2025; Kaggle `google-code-golf-2025`): the shortest Python program for
  each of 400 ARC-AGI-1 tasks. Files `task001..task400.json` each hold train, test and about 262 ARC-GEN examples.
  `SakanaAI/google-code-golf-2025` (Apache-2.0) has all 400 solutions.
- **NeuroGolf Championship 2026** (IJCAI-ECAI 2026). Official site: "Implement a variety of transformations using the
  simplest possible networks"; organisers Moffitt, Oldacre, Reade and Howard; ran 2026-04-15 to 2026-07-15; $50K. The
  site gives no rules. A community repository (`zikuanqi/NeuroGolf`, MIT) describes them: one ONNX network per ARC task,
  one-hot input of shape (1,10,30,30), and a score of max(1, 25 − ln(memory + params)). Winners are **UNVERIFIED**.
- Supporting resources: Hodel's `arc-dsl` (MIT; 160 primitives, 400 solvers), `re-arc` (MIT; 400 generators and 400
  verifiers), Google `arc-gen` (Apache-2.0), and BARC (Python seed programs plus an 8B model that writes programs).

This design treats Code Golf as the **program** side and NeuroGolf as the **program → weights** side. The owner should
confirm this reading (§5.6).

Prior art: the idea has had a name since November 2026, and every item below is newer than all three relay models'
cutoffs.

- **iWSP** (arXiv 2611.18114, 2026-11-13; repo `csinva/weight-space-programming-autoresearch`, MIT)
  - Claude Code agents write the weights of small transformers in closed-form code, with no
    training.
  - 17 tasks. The algorithmic models reach 100 % accuracy at a median of 166 parameters.
  - Audits include a text-bottleneck rebuild.
  - Cost about $17,500.
- **WSP-Scope** (arXiv 2611.09382, 2026-11-06)
  - Agents edit the weights of Qwen3.5-0.8B and 9B.
  - Up to +29.45 points on MATH-500 from 1,023 edited parameters.
  - 6 of 16 trajectories kept the original weights.
- **arXiv 2611.09376** (verified by the ops lens): GPT-6 Astra edits 65 embedding rows of π0.5 in a CPU workspace.

What WSI adds to these:

- a **typed, falsifiable statement language** that knows the architecture's symmetry group;
- a measuring executor;
- a **multi-model relay** with a human editor in the loop.

---

## 1. The language: WSI

### 1.1 Principles

1. **A statement has three parts: an operation, a target selector and an expected observable effect.** The expectation
   is what makes a statement falsifiable. `expect … ?` leaves it open, which asks the relay to predict and the executor
   to measure.
2. **The kind is part of the statement.** The same English sentence denotes different things:
   - `gauge` claims an exact symmetry;
   - `edit` claims a measurable change.

   The measured examples:

   | Case | Result |
   |---|---|
   | Global residual rotation on Pythia-70M | 1.5e-11 (exact) |
   | Middle-layer-only rotation on Pythia-70M | max_dlogits 366 |
   | Middle-layer-only rotation on TRM | relative change 1.09, argmax agreement 0 % |

   The static checker therefore rejects a `gauge` that is not in the architecture's symmetry group *before* anything
   runs. Changing the kind to `edit` turns the same operation into a measurement.
3. **Reparametrisations are explicit.** Some symmetries exist only after a function-preserving change of form, measured
   on pythia-14m:

   | Step | Result |
   |---|---|
   | Rotate the residual stream on stock parameters | broken, Δloss +3.006 |
   | After `fold_ln`, a rotation R with R·1 = 1 | exact, 2.0e-11 |
   | After also `center_writers` and `ln_to_rms`, any R in O(128) | exact, 2.0e-11 |
4. **Comparing or mixing two weight sets needs a quotient.** Averaging, interpolating or transplanting is only defined
   up to the permutation and per-head gauge. `align(a -> b)` is a primitive, and the type checker warns when
   `lerp`/`slerp`/`+` mixes two layers or two runs without it. Measured on pythia-14m:

   | Case | Result |
   |---|---|
   | Average layers 2 and 3, naive | Δloss +2.958 |
   | Same, after aligning layer 3 to layer 2 | Δloss +2.668 |
   | Grokking direction transplanted between seeds, no alignment | test accuracy 0.024 |
5. **Directions and features are different types.** A direction is static and linear, so it compiles to a weight edit.
   An SAE feature is gated by the input and needs the encoder in the forward pass. Measured on Pythia-70M, SAE feature
   21809 at layer 3:

   | Operation | Loss |
   |---|---|
   | Baseline | 3.877 |
   | Project the decoder direction out of the weights | 6.975 |
   | Ablate the feature only where it fires | 3.877 |
6. **WSI is not Python.** Statements come from model output, which is untrusted. The parser accepts a whitelist
   grammar: no `eval`, no file paths, only registered model aliases, and size and step caps. Because the executor
   compiles every statement, relay models never write torch code, so they also avoid the TransformerLens 4 trap:
   `HookedTransformer` was removed in 4.0 (2026-09-21) and all three models' training data predates that.
7. **Canonical form.** Statements are ASCII. The parser normalises `θ − ≈ ∈ →` to `theta - ~ in ->`. The printer adds
   defaults (seed, group dimension) and echoes every statement in canonical form in the Observation.

### 1.2 Sorts

| Sort | Examples | Notes |
|---|---|---|
| `Params` | `L3`, `L3.attn`, `L*.writers`, `embed`, `theta` | a set of tensors in a state_dict |
| `Space` | `resid`, `L3.head[0].qk[nonrotary]`, `L3.head[0].vo`, `L3.neurons`, `L3.heads` | an axis that a group acts on |
| `Group` | `O(d)`, `Stab1(d)`, `GL(d)`, `S(n)`, `Pos(n)`, `RopeComm(d)` | the checker knows which apply to which `Space` |
| `Direction` | `dir(sae(L3.resid_post, 21809))`, `dir(mean_diff(pos, neg, L3.resid_post))` | a vector in `resid` |
| `Feature` | `feat(sae(L3.resid_post, 21809))` | input-gated; only in `hook`, `equiv` and observables |
| `Ckpt` / `Run` | `ckpt(A, 200)`, `pythia-14m@step60000` | named training states |
| `Iter` | `iter(H=2, L=3)` | weight-tied recursions (TRM/HRM): "layer k" is iteration k of the same weights |
| `TaskRow` | `puzzle_emb[task=007bbfb7, aug=rot90]` | TRM/HRM per-(task, augmentation) embedding row |
| `Observable` | `dloss`, `acc(test)`, `t_grok(0.95)`, `grid(L3.head[0].ov)` | see §1.4(d) |

### 1.3 Grammar sketch (EBNF; the MVP parser fixes the details)

```ebnf
(* WSI 0.1 - one statement per line, '#' starts a comment *)
program      = { line } ;
line         = [ header | statement ] , [ comment ] , NEWLINE ;
header       = "use" , model_ref                      (* use pythia-14m@cf967c0a *)
             | "probe" , NAME                          (* registered probe set, e.g. own149 *)
             | "seed" , INT
             | "let" , NAME , "=" , ( weight_expr | direction ) ;
model_ref    = NAME , [ "@" , REV ] , [ "(" , kwargs , ")" ] ;   (* pythia-14m, modadd, linear-resnet(d=16), golf:task087 *)

statement    = [ LABEL , ":" ] , body , [ expect ] , [ cite ] ;
body         = "gauge"   , sym_op , [ "requires" , reparam_op , { "," , reparam_op } ]
             | "reparam" , reparam_op , { "," , reparam_op }
             | "edit"    , edit_item , { ";" , edit_item }          (* applied in order, one statement *)
             | "equiv"   , edit_op , "==" , hook
             | "train"   , task , [ "with" , directive , { "," , directive } ]
                         , [ "save" , NAME , "at" , INT , { "," , INT } ]
             | "path"    , weight_expr , "for" , "t" , "in" , range
             | "observe" , observable , [ "on" , ( model_ref | "compile" , "(" , arc_prog , ")" ) ] ;
edit_item    = sel , ":=" , weight_expr | edit_op ;
range        = NUMBER , ".." , NUMBER , [ "step" , NUMBER ] ;
arc_prog     = NAME , [ "(" , arc_prog , { "," , arc_prog } , ")" ] ;   (* whitelisted arc-dsl primitives only *)

(* (a) exact symmetries *)
sym_op       = ( "perm" | "rotate" | "basis" | "rescale" ) , "(" , sel , [ "," , elem ] , ")" ;
elem         = NAME , ":" , group , [ "seed" , INT ] ;
group        = ( "O" | "Stab1" | "GL" | "S" | "Pos" | "RopeComm" ) , "(" , INT , ")" ;
reparam_op   = "fold_ln" | "center_writers" | "ln_to_rms" ;

(* (b) structural edits *)
edit_op      = "swap" , "(" , sel , "," , sel , ")"
             | "delete" , "(" , sel , ")"
             | "insert_after" , "(" , sel , "," , weight_expr , ")"
             | "rotate" , "(" , sel , "," , elem , ")"            (* as an edit: no compensation *)
             | "antipair" , "(" , sel , "->" , sel , ")"          (* copy a into b, negate b's writers *)
             | "parallel" , "(" , sel , ")"                        (* run a layer range in parallel, mean of outputs *)
             | "project_out" , "(" , direction , "," , sel , ")"
             | "steer" , "(" , direction , "," , NUMBER , "," , sel , ")"   (* all-position bias edit *)
             | "rank_one" , "(" , sel , "," , vec , "," , vec , "," , NUMBER , ")"
             | "patch" , "(" , sel , "," , mask , "," , weight_expr , ")"
             | "scale" , "(" , sel , "," , NUMBER , ")"
             | "quantize" , "(" , sel , "," , INT , ")" ;
weight_expr  = term , { ( "+" | "-" ) , term } ;
term         = [ NUMBER , "*" ] , factor ;
factor       = sel | "ckpt" , "(" , NAME , "," , INT , ")" | "neg" , "(" , weight_expr , ")"
             | ( "lerp" | "slerp" ) , "(" , weight_expr , "," , weight_expr , "," , ( NUMBER | "t" ) , ")"
             | "align" , "(" , weight_expr , "->" , weight_expr , ")"
             | "random_like" , "(" , weight_expr , ")" | "antisym" , "(" , weight_expr , ")"
             | "expm" , "(" , weight_expr , ")" | "(" , weight_expr , ")" ;

(* selectors *)
sel          = scope , { "." , part } ;
scope        = "resid" | "embed" | "unembed" | "pos" | "theta" | "L" , layers
             | "puzzle_emb" , "[" , kv , { "," , kv } , "]" | NAME ;
layers       = INT | "mid" , [ ( "+" | "-" ) , INT ] | "*" | INT , ".." , INT | "{" , INT , { "," , INT } , "}" ;
part         = "attn" | "mlp" | "readers" | "writers" | "resid_io" | "neurons" | "heads"
             | "head" , "[" , ( INT | "*" | INT , ".." , INT ) , "]"
             | "qk" , [ "[" , ( "rotary" | "nonrotary" | INT , ".." , INT ) , "]" ] | "vo" | "ov" | "qk_product"
             | TENSOR_NAME ;
direction    = "dir" , "(" , ( "sae" , "(" , hookpt , "," , INT , ")"
                             | "mean_diff" , "(" , NAME , "," , NAME , "," , hookpt , ")" | NAME ) , ")" ;
feature      = "feat" , "(" , "sae" , "(" , hookpt , "," , INT , ")" , ")" ;
hook         = "hook" , "(" , hookpt , "," , ( "proj" , "(" , direction , ")"
                                             | "add" , "(" , direction , "," , NUMBER , ")"
                                             | "zero" , "(" , feature , ")" ) , ")" ;

(* (c) training-dynamics directives *)
task         = "modadd" , "(" , "p" , "=" , INT , "," , "frac" , "=" , NUMBER , [ "," , "op" , "=" , NAME ] , ")"
             , [ "recipe" , NAME ] ;
directive    = "adamw" , "(" , kwargs , ")" | "grokfast_ema" , "(" , NUMBER , "," , NUMBER , ")"
             | "grokfast_ma" , "(" , INT , "," , NUMBER , ")" | "wd_schedule" , "(" , pieces , ")"
             | "clamp_norm" , "(" , NUMBER , ")" | "logit_scale" , "(" , NUMBER , ")"
             | "implant_direction" , "(" , weight_expr , "," , INT , ")"
             | "at" , "(" , INT , ")" , "edit" , edit_item
             | "seeds" , "(" , INT , ")" | "steps" , "(" , INT , ")" ;

(* (d) observables, expectations, provenance *)
expect       = "expect" , claim , { "and" , claim } ;
claim        = observable , ( relop , rhs , [ "tol" , NUMBER ] | "?" ) ;
relop        = "<" | "<=" | ">" | ">=" | "=" | "~" | "in" ;
rhs          = NUMBER | [ NUMBER , "*" ] , observable , [ "@" , edit_op ] | "[" , NUMBER , "," , NUMBER , "]" ;
               (* 'dloss@delete(L3)': the same observable under a reference edit *)
observable   = "max_dlogits" | "rel_dlogits" | "loss" | "dloss" | "kl"
             | "acc" , "(" , NAME , ")" | "logit_diff" , "(" , STRING , "," , STRING , ")"
             | "resid_diff" , "(" , sel , "," , edit_op , ")" | "t_grok" , "(" , NUMBER , ")"
             | "barrier" | "curve" , "(" , observable , ")" | "norm" , "(" , weight_expr , ")"
             | "cos" , "(" , weight_expr , "," , weight_expr , ")" | "spectrum" , "(" , sel , ")"
             | "fourier_share" , "(" , sel , "," , INT , ")"
             | ( "picture" | "grid" ) , "(" , sel , [ "," , view ] , ")"
             | "arc_grid" , "(" , NAME , "," , INT , ")" | "base" , "." , observable ;
view         = "order" , "=" , ( "index" | "norm" | "pca" | "align" , "(" , sel , ")" )
             , [ "," , "palette" , "=" , INT ] , [ "," , "size" , "=" , INT ] ;
cite         = "cite" , REF , { "," , REF } , [ "status" , ( "THEOREM" | "EMPIRICAL" | "OPEN" | "FALSE" ) ] ;
REF          = "arXiv:" , ARXIV_ID | "run:" , NAME ;
```

`base.X` means "the same observable on the unedited model, or on the baseline training run". `X@edit` means "the
same observable under a reference edit". A statement fits on one line (at most 200 characters), because relay notes
are short.

**Semantics of a file.** Every statement starts from the model of the current `use` line, so edits never accumulate
across statements. Order matters only through names that `let` and `save` create, which later statements can use.
Inside one `edit`, a `;` chain applies its parts in order. `save` names training checkpoints: with `seeds(n)` and
`save A`, the runs are `A0` to `A(n-1)`. For linear toys, `spectrum(La..b)` is the spectrum of the composite map.

### 1.4 Core primitives

#### (a) Exact symmetries (`gauge`): function-preserving

The static checker reads an **ArchCard**, a set of traits parsed from `config.json`: norm kind, activation, rotary
dims, parallel residual, biases, tied embeddings and weight tying. From it the checker looks up the symmetry group.
"Measured" values are max_dlogits on the given model in float64; the floor is about 1e-12 relative.

| Primitive | Space and group | Preconditions | Evidence |
|---|---|---|---|
| `perm(Lk.neurons, pi: S(n))` | MLP hidden units | any elementwise activation | measured: pythia-14m 2.7e-11; Pythia-70M 3.6e-11; TRM SwiGLU 1.6e-15 (relative) |
| `perm(Lk.heads, pi: S(h))` | heads | none | measured: pythia-14m 2.9e-11 |
| `basis(Lk.head[i].vo, N: GL(d_head))` | V := N V, O := O N⁻¹ | none | measured: pythia-14m 2.1e-11; Pythia-70M 6.0e-11 (cond 161). Theory: Tran et al. arXiv 2410.04209, maximal MHA group S_h × GL^h × GL^h |
| `basis(Lk.head[i].qk[nonrotary], A: GL(m))` | Q := A Q, K := A⁻ᵀ K | only on non-rotary dims | measured: pythia-14m GL(24) on dims 8..31, 2.5e-11; the same on rotary dims fails (Δloss +0.273) |
| `basis(Lk.head[i].qk[rotary], P: RopeComm(r))` | one 2×2 rotation per RoPE pair, plus Q·s, K/s | RoPE | measured: pythia-14m 2.6e-11. Derived in the formal lens (commutant of RoPE); no paper opened for it |
| `rescale(Lk.neurons, c: Pos(n))` | W_in·c, W_out/c | **ReLU only** (positively homogeneous) | random-weight ReLU toy 2.7e-15; **fails for GELU** (pythia-14m Δloss +0.130); TRM: up/down pair exact (9e-16), gate not (0.147) |
| `rotate(resid, R: Stab1(d))` | all readers, writers, embed, unembed | `requires fold_ln`, and the model has LayerNorm | measured: pythia-14m 2.0e-11, Pythia-70M 1.5e-11. A generic Q breaks it (Pythia-70M 13.8; pythia-14m Δloss +0.743) |
| `rotate(resid, R: O(d))` | same | `requires fold_ln, center_writers, ln_to_rms`, or a parameter-free RMSNorm (TRM) | measured: pythia-14m 2.0e-11; TRM relative 2.3e-15. Theory: SliceGPT arXiv 2401.15024 Thm 1; QuaRot; SpinQuant R1 |
| `rotate(resid, M: GL(d))` | same | no norm at all (modadd toy) | algebra; **not yet measured** (MVP test) |
| `reparam fold_ln` / `center_writers` / `ln_to_rms` | change of form | `fold_ln` on NeoX needs a *new* unembed bias W_U β | measured: pythia-14m 2.4e-11 / 2.2e-11 / 2.2e-11 |
| approximate gauge: `scale(resid, 2)` | stream scale | broken only by LN eps | measured: pythia-14m max_dlogits 0.45, Δloss +0.0002 |

Every statement that is not in the table for the loaded ArchCard is **rejected statically** as a `gauge`. The message
names the closest valid symmetry and offers the `edit` form.

#### (b) Structural edits (`edit`): change the function

The executor always reports Δloss, max_dlogits and a **random control of equal ‖Δθ‖**. On pythia-14m that control cost
+16.0, against +2.5 for the antisymmetric pair of equal norm. Measured values below are on pythia-14m with its
149-token probe unless stated.

| Primitive | Meaning | Measured / literature |
|---|---|---|
| `delete(Lk)` | zero the writers of layer k | Δloss +4.043 (L3); Pythia-70M +0.835 |
| `swap(Lj, Lk)` | exchange two blocks | +3.275 (L2↔L3); Pythia-70M +2.167 |
| `Lk := lerp(Lj, Lk, 0.5)` / `align` | average, naive or gauge-fixed | L2,L3 naive +2.958, aligned +2.668; Pythia-70M naive +1.138 |
| `Lk := lerp(Lj, Ll, 0.5)` | "interpolate in between" | +4.992 (L3 := mean(L2, L4)), worse than deleting L3 |
| `insert_after(Lk, slerp(Lk, align(Lk+1 -> Lk), 0.5))` | PLI-style inserted layer | EMPIRICAL: arXiv 2506.02973, slerp; upper positions 24/28 of 32; +1.08 TruthfulQA MC1 averaged over 8 models. Not measured here |
| `parallel(La..b)` | run the middle range in parallel, mean of outputs | EMPIRICAL: Painters arXiv 2407.09298 (Llama2-7B); works except GSM8K |
| `La..b := Lmid` | middle repeat | FALSE: Painters, "degrades to random baseline" |
| `antipair(Lj -> Lk)` | Lk := copy of Lj with its writers negated | +2.511; the residual stream differs by 104 % from "delete both" (+4.064); delete Lk alone +2.158 |
| `rotate(Lk.resid_io, R: O(d))` | rotate one layer's interface, uncompensated | +6.369, the most destructive edit tested |
| `project_out(dir(r), L*.writers)` | directional ablation by weight orthogonalisation | Pythia-70M, Equiv to the hook: 2.68e-11 (effect 4.53). Theory: Arditi et al. arXiv 2406.11717 §4.1 |
| `steer(dir(v), a, Lk)` | all-position steering, as a bias edit | Pythia-70M, Equiv to the hook: 2.71e-11 (effect 8.87). Needs an output bias (Llama/Qwen have none) |
| `rank_one(Lk.mlp.W_out, u, v, a)` | ROME-style closed-form update | literature: ROME arXiv 2202.05262; MEMIT arXiv 2210.07229 |
| `theta := theta + a*(ckpt(P) - ckpt(N))` | task vector / contrastive weight steering | literature: arXiv 2212.04089; Fierro & Roger arXiv 2511.05408. Measured toy on Pythia-70M: weight steering +7.5 vs activation steering +4.6 held-out log-odds at similar Δloss (one seed) |
| `patch`, `quantize`, `scale` | masked patch, k-level codebook, global scale | used by the graphical layer (§2) |

#### (c) Training-dynamics directives (`train`)

Toy task: `modadd(p, frac)` with `recipe nanda`. This is Nanda et al. (arXiv 2301.05217): a one-layer ReLU transformer,
d 128, 4 heads × 32, MLP 512, no LayerNorm, full-batch AdamW, lr 1e-3, weight decay 1. The Adam betas (0.9, 0.98) are
the spikes' choice; the paper's values are **UNVERIFIED**.

| Directive | Meaning | Evidence |
|---|---|---|
| `grokfast_ema(a, l)` | g ← g + l·EMA_a(g) | measured: p=113 / frac 0.3, t_grok(0.95) 6,700 vs 9,900 (1.48×, seed 0); p=97 / frac 0.5, 90 % at step 1,000 vs 1,250 (1.25×). Paper arXiv 2405.20233 reports ">50×" in a different setup (multiplication mod 97, batch 512, wd 0.005) |
| `at(t) edit scale(theta, c)` | one-shot weight edit during training | measured: c 0.7 at step 1,500 gives 8,900 vs 9,900 (1.11×). The optimiser restores the norm within about 500 steps. With one seed this is **inconclusive** |
| `clamp_norm(c)` | constant weight norm | EMPIRICAL: Omnigrok arXiv 2210.01117 ("largely eliminates grokking" at α=0.8) |
| `logit_scale(s)` | output temperature | EMPIRICAL: Truong arXiv 2606.18465 (delay tracks logit scale, R² 0.97) |
| `implant_direction(ckpt, t)` | donor weight direction at the recipient's norm | EMPIRICAL: Truong arXiv 2607.06628 (40/40 runs take the donor's circuit; a random control does nothing) |
| `wd_schedule(...)`, `adamw(...)`, `seeds(n)`, `steps(n)`, `save NAME at …` | schedule, optimiser, replication, checkpoints | — |
| reserved, not in the MVP | `stablemax`, `perp_grad` (Prieto et al. arXiv 2501.04697), `muon` | — |

#### (d) Observables

| Observable | Definition | Used for |
|---|---|---|
| `max_dlogits`, `rel_dlogits`, `kl` | against the unedited model on the probe set | gauge and equiv verdicts |
| `loss`, `dloss` | next-token loss on the probe set (pythia) or the train/test split (modadd) | edits |
| `acc(split)`, `t_grok(x)` | accuracy; first evaluated step with test accuracy ≥ x | dynamics |
| `logit_diff("a", "b")` | logit difference of two tokens | behaviour edits |
| `resid_diff(Lk, ref)` | relative norm of the residual difference after layer k against a reference edit | "does the pair cancel" |
| `barrier`, `curve(O)` | max loss above the chord along a path; O(t) on a t grid | path statements |
| `norm`, `cos`, `spectrum(sel)` | norms, cosines, singular values | gauge-invariant summaries |
| `fourier_share(embed, k)` | share of embedding norm in the top-k Fourier frequencies (definition proposed here) | grokking as a weight measurement |
| `picture(sel, view)` | PNG for the owner | §2 |
| `grid(sel, view)` | at most 16×16 digits 0–9, an ARC-format text grid for the notes | §2.4 |
| `arc_grid(task, view)` | TRM/HRM predicted output grid | Phase 3 |

### 1.5 Verdicts and status

The **executor verdict** is decided by a rule, with no LLM judging:

| Kind | Verdict rule |
|---|---|
| `gauge`, `reparam` | HOLDS if rel_dlogits ≤ 1e-9 in float64, else FAILS, with the value |
| `equiv` | HOLDS if the difference between weight edit and hook is ≤ 1e-9 relative to the effect size; the measured margin was 2.7e-11 vs 4.5–250 |
| `edit` | CONFIRMED or REFUTED per `expect` claim, plus the random control and `base.*` |
| `train` | CONFIRMED if the claim holds for the median and for at least ⌈2n/3⌉ of n seeds. With one seed the verdict is INDICATIVE |
| any kind | NOT_RUN with the static error or the budget that was hit |

The **literature status** is attached by `cite … status`:

- THEOREM(ref);
- EMPIRICAL(ref, model, scale);
- OPEN;
- FALSE(counterexample).

An executor run adds MEASURED(model@rev, probe, dtype, seeds, value). A literature status is never treated as a
measurement for a different setup. Grokfast's ">50×" became 1.25–1.48× here.

### 1.6 The owner's three ideas, written in WSI

All MEASURED values are from the lens spikes, on the model named in the `use` line.

#### Idea 1: "All weights in the middle layer rotate"

```text
use pythia-14m@cf967c0a        # GPT-NeoX, L0..L5, d 128, 4 heads x 32, rotary dims 0..7, MLP 512, GELU, LayerNorm
probe own149
R1: gauge rotate(Lmid.resid_io, R: O(128))                     # as said: claimed invariance of one layer
R2: edit  rotate(Lmid.resid_io, R: O(128) seed 1) expect dloss ?
R3: gauge rotate(resid, R: Stab1(128)) requires fold_ln  cite arXiv:2401.15024 status THEOREM
R4: gauge rotate(resid, R: O(128)) requires fold_ln, center_writers, ln_to_rms  cite arXiv:2401.15024 status THEOREM
R5: gauge basis(Lmid.head[0].vo, N: GL(32))
R6: gauge basis(Lmid.head[0].qk[nonrotary], A: GL(24))
R7: gauge basis(Lmid.head[0].qk[rotary], P: RopeComm(8))
R8: gauge perm(Lmid.neurons, pi: S(512))
R9: gauge rescale(Lmid.neurons, c: Pos(512))
```

| Statement | Static check | Result | Status |
|---|---|---|---|
| R1 | **rejected**: a residual rotation must cover every reader and writer, embed and unembed, and needs `fold_ln` on LayerNorm models | — | FALSE as an invariance |
| R2 | ok (edit) | pythia-14m: Δloss **+6.369**, max_dlogits 31.0. Pythia-70M: max_dlogits 366. TRM: relative 1.09, argmax 0 %. Random RMSNorm toy: 8.1 | MEASURED |
| R3 | ok | 2.0e-11 (pythia-14m); 1.5e-11 (Pythia-70M) | THEOREM + MEASURED |
| R3 without `requires` | rejected | Δloss +3.006 | FALSE |
| R3 with a generic O(128) | rejected (LN centring) | Δloss +0.743 | FALSE |
| R4 | ok | 2.0e-11 | THEOREM + MEASURED |
| R5, R6, R7, R8 | ok | 2.1e-11, 2.5e-11, 2.6e-11, 2.7e-11 | MEASURED. Theory: arXiv 2410.04209; arXiv 2502.00264 |
| R9 | **rejected**: GELU is not positively homogeneous | as an edit: Δloss +0.130 | FALSE for GELU; exact for ReLU |

Two editor notes:

- `Lmid` with 6 layers is ambiguous (L2 or L3). The checker warns and resolves it to L3, the layer the spikes used.
- Reading for the owner: "rotate the middle layer" is either a no-op (a gauge, compensated everywhere, or restricted to
  per-head pairs) or the most destructive edit tested. A global rotation keeps the function but **destroys the readable
  basis** ("neuron k means X" is never gauge-invariant). That is why §2 shows gauge-invariant pictures.

#### Idea 2: "Antisymmetric layers before and after the middle make the net interpolate in between"

```text
use pythia-14m@cf967c0a
I1: edit antipair(L2 -> L4)            expect resid_diff(L4, delete(L{2,4})) < 0.1      # "the pair cancels"
I2: edit L3 := lerp(L2, L4, 0.5)       expect dloss < dloss@delete(L3)                  # "interpolates in between"
I3: edit L3 := lerp(L2, align(L4 -> L2), 0.5)   expect dloss ?
I4: edit insert_after(L3, slerp(L3, align(L4 -> L3), 0.5))   expect dloss ?  cite arXiv:2506.02973 status EMPIRICAL
I5: edit parallel(L2..3)               expect dloss ?  cite arXiv:2407.09298 status EMPIRICAL
I6: edit L2..4 := L3                   expect dloss > dloss@delete(L3)  cite arXiv:2407.09298 status FALSE
use linear-resnet(d=16, layers=3)       # toy without norms, for the exact algebra
let A = antisym(random_like(L1.W))
I7: edit L0.W := expm(A) ; L2.W := expm(neg(A))   expect spectrum(L0..2) ~ spectrum(L1) tol 1e-12
```

| Statement | Result | Status |
|---|---|---|
| I1 | the residual differs by **104 %** from "delete both"; Δloss +2.511, vs delete L4 only +2.158 and delete both +4.064; random control of equal norm +16.002 | **FALSE**: no first-order cancellation in a 6-layer model, although the edit is structured |
| I2 | Δloss **+4.992** vs delete L3 +4.043 | **FALSE** at this scale: adjacent layers do not lie on a line |
| I3 | not run; the closest run is "average L2 and L3, aligned", +2.668 vs +2.958 naive | OPEN (MVP run) |
| I4 | not run at 14M | EMPIRICAL at 7–8B (slerp, upper layers, +1.08 TruthfulQA MC1 average); OPEN at 14M |
| I5 | not run | EMPIRICAL (Llama2-7B; not GSM8K) |
| I6 | not run | FALSE (Painters: middle repeat collapses to the random baseline) |
| I7 | the composite is exp(A)·W·exp(−A), a conjugation: spectrum unchanged (2e-14), W itself changes (4.8). In residual (Euler) form the pair leaves a term ε²([A,W] − A²), measured ratio 50.1 vs predicted 50.17 | THEOREM in linear form; OPEN inside a pre-norm block |

Reading for the owner, in two precise forms:

- **"Antisymmetric"** means a rotation generator. Haber & Ruthotto (arXiv 1705.03341): "the antisymmetric matrix K0
  yields a rotation in the feature space"; also AntisymmetricRNN (arXiv 1902.09689). An outer pair exp(A) / exp(−A)
  *conjugates* the middle layer.
- **"Interpolates"** in the exact sense is the geodesic midpoint R0·exp(½·log(R0ᵀR1)). The log is antisymmetric. A
  linear midpoint of two rotations is not a rotation: in the random-weight spike its singular values were off by up to
  0.96.

Smoothness of trained weights in depth is proven only for ResNet → neural-ODE settings (arXiv 2205.14612;
arXiv 2309.01213).

#### Idea 3: "Is grokking steerable? What happens if I steer along grokking?"

```text
use modadd
G0: train modadd(p=113, frac=0.3) recipe nanda with seeds(1)
G1: train modadd(p=113, frac=0.3) recipe nanda with grokfast_ema(0.98, 2)  expect t_grok(0.95) < base.t_grok(0.95)
      cite arXiv:2405.20233 status EMPIRICAL
G2: train modadd(p=113, frac=0.3) recipe nanda with at(1500) edit scale(theta, 0.7)  expect t_grok(0.95) < base.t_grok(0.95)
G3: train modadd(p=97, frac=0.5) recipe nanda with seeds(2) save A at 200, 2000    # runs A0 and A1
G4: path lerp(ckpt(A0, 200), ckpt(A0, 2000), t) for t in -0.5..2  expect curve(acc(test)) ?
G5: edit theta := ckpt(A1, 200) + (ckpt(A0, 2000) - ckpt(A0, 200))  expect acc(test) > 0.9
G6: edit theta := ckpt(A1, 200) + align(ckpt(A0, 2000) - ckpt(A0, 200) -> ckpt(A1, 200))  expect acc(test) ?
G7: train modadd(p=97, frac=0.5) recipe nanda with implant_direction(ckpt(A0, 2000), 200)  expect acc(test) ?
      cite arXiv:2607.06628 status EMPIRICAL
G8: observe cos(ckpt(A0, 2000) - ckpt(A0, 200), ckpt(A1, 2000) - ckpt(A1, 200))
```

| Statement | Result | Status |
|---|---|---|
| G0 | t_grok(0.95) = 9,900 (seed 0), matching Nanda's "around 10,000 epochs" | MEASURED |
| G1 | **6,700** (1.48×); test accuracy already ahead at step 1k (0.122 vs 0.039); a transient train-accuracy dip to 0.966 | CONFIRMED, INDICATIVE (one seed) |
| G2 | 8,900 (1.11×); ‖θ‖ goes 57.0 → 39.9 at the edit → 52.3 at step 2,000 | INDICATIVE, inside unmeasured seed noise |
| G4 | test accuracy (train accuracy) along the chord: 0.488 (1.0) at t=0; 0.370 (0.81) at t=0.25, the barrier; **0.969** at 0.75; 1.0 at 1; 0.999 at 1.25; 0.727 at 1.5; 0.205 at 2 | MEASURED: steerable along a line within one run, not linearly connected |
| G5 | test accuracy **0.024** | **FALSE**: the direction is seed-specific |
| G6 | not run | **OPEN**: the best original experiment here. A training-free transplant after alignment has no verified paper |
| G7 | not run here | EMPIRICAL (40/40, July 2026) |
| G8 | cos −0.001; cos(dir_A, −θ_A(mem)) = 0.739, yet a pure shrink of θ_mem by 0.7 gives test accuracy 0.115 | MEASURED: grokking is not just shrinking |

Reading for the owner:

- **Yes, grokking is steerable**: through the optimiser (Grokfast, weight norm, logit scale) and along a run's own
  weight-space chord.
- **"Steering along grokking" across runs needs a quotient** by the permutation gauge first.
- At LLM scale it is **OPEN**.

---

## 2. Graphical semantics

### 2.1 Views: a tensor as a grid

A **view** turns a selector into a 2-D grid. Its four parts are always named:

- the **tensor**;
- the **row and column order**: `index`, `norm`, `pca`, or `align(ref)`;
- the **palette**: a k-level codebook; `palette=10` means sign × magnitude quantiles mapped to 10 colours, with 0 as black;
- the **size**: downsampling by block means, at most 30×30 like ARC, and 16×16 for the notes.

Two outputs:

- `picture(...)` writes a PNG for the owner (matplotlib, optional).
- `grid(...)` writes a text grid of digits for the notes.

Hidden axes are permutable, so a picture of a raw W_in shows one arbitrary representative of its gauge orbit. The
editor therefore prefers **gauge-invariant pictures**:

| Picture | Invariant under | Notes |
|---|---|---|
| `Lk.head[i].ov` = W_O·W_V (d × d) | per-head VO GL (measured: "the OV product is the object", Pythia-70M 6.0e-11) | still conjugated by a residual rotation R |
| `Lk.head[i].qk_product` = W_Qᵀ·W_K on non-rotary dims | per-head QK GL on those dims | rotary dims: only the commutant is free |
| token-space `unembed · ov · embed` | all of the above **and** residual rotation | direct-path summary, not a function claim (norms are ignored). For TRM the vocabulary has 12 tokens, so this is a **12 × 12 colour → colour grid**, exactly ARC-shaped (proposal, not built) |
| `spectrum(sel)`, Gram matrices, norms per head or neuron | all permutations and rotations of the free axis | shown as bar strips |

### 2.2 ARC-DSL primitives on weight tensors

Grid operations on a view compile down to WSI core operations. Then the gauge/edit rule decides what they mean. This
dictionary extends the ARC lens's (`arc.md` §4.3).

| arc-dsl primitive | Weight-space counterpart | Gauge or edit |
|---|---|---|
| `rot90/180/270`, `hmirror`, `vmirror`, `cmirror` | W → P_out·W·P_in with reversal permutations (a transpose for odd rotations) | **gauge** only if the matching inverse permutation is applied to the neighbouring tensors (as in `perm`). Alone it is an edit. Measured on TRM: rot90 of a 512×512 block of `down_proj` gave a relative logit change of 0.69 |
| `dmirror` | Wᵀ (swaps reader and writer roles) | edit (type-correct only for square W); meaningful for tied embed/unembed |
| `upscale`, `hupscale`, `vupscale` | Net2Net widening (arXiv 1511.05641): duplicate units and divide outgoing weights | function-preserving when the outgoing weights are divided by the factor |
| `hconcat`, `vconcat` | add units, heads or layers (identity / zero init) | function-preserving with zero or identity init; not for TRM's tied recursion |
| `crop`, `trim`, `compress` | prune rows and columns; SliceGPT prunes *after* a PCA rotation | approximate (a variance criterion) |
| `recolor`, `replace`, `palette`, `numcolors` | elementwise value maps; `quantize(sel, k)` | edit. Scaling is exact only across homogeneous pairs; `(W_V, W_O) → (−W_V, −W_O)` is exact |
| `objects`, `partition`, `fgpartition` | heads, SwiGLU units, connected components of the thresholded bipartite weight graph | analysis only. Adjacency is meaningful only after gauge fixing (sort, or Git Re-Basin arXiv 2209.04836) |
| `neighbors`, `shift`, `gravitate` | circulant permutations | only on position-indexed axes |
| `hperiod`, `vperiod`, `occurrences` | periodic structure; `fourier_share(embed)` | where grokking becomes a weight measurement (Fourier embeddings, Nanda et al.) |
| `cellwise`, `paint`, `underpaint` | `patch(sel, mask, V)`; "paint only background" ≈ edit only near-zero weights | edit |
| colour-permutation augmentation | permute the 10 colour rows of TRM `embed_tokens` and `lm_head` (12 × 512) | exact relabelling by construction (embedding lookup); not yet measured. The puzzle-embedding row also changes per augmentation |

### 2.3 Where the analogy holds and where it breaks

| Holds | Breaks |
|---|---|
| Axes that really index space: positions in token-mixing MLPs, conv kernels (NeuroGolf, ARC-NCA), positional embeddings, token and colour rows (TRM vocabulary 12) | Hidden axes have **no order**. Any permutation is equivalent, so "rotate a hidden × hidden block" has no intrinsic meaning |
| Fixed-size geometry compiled into weights: `rot180` as kron(I₁₀, P) on the one-hot 30×30 canvas is exact on **266/266** examples of task087 (ARC-GEN included), and `dmirror` on 267/267 (task179) | Variable grid sizes: on a zero-padded canvas each size needs a different permutation. `vmirror` (task150, 7 sizes) is exact on only **39/266**, and `dmirror` on task241 on 32/266. A size gate is non-linear |
| Structure supplied by the parametrisation: CompressARC is D4- and colour-equivariant by construction | Unstructured weights memorise: a min-norm linear map fits train+test in 394/394 tasks but generalises to ARC-GEN in **1/394** |
| Colours as categories: rows of embedding tables | Weight values are real and ordered, so a palette is a quantisation choice that must be named |
| TRM/HRM tasks as rows: 95–98.5 % of the parameters are per-(task, augmentation) 512-d rows | TRM/HRM use 1-D RoPE on the flattened grid, so grid rotation is **not** an architectural symmetry; D4 is learned from augmentation |

### 2.4 Pictures in the notes: ARC-format text grids

The relay is text-only. The executor can append `grid(sel, view)` as a block of at most 16 rows of digits 0–9, with
the view stated, for example `grid(L3.head[0].ov, order=pca, palette=10, size=16)`. A 16×16 grid costs roughly 150–300
tokens (estimate).

The relay models solve ARC-AGI-2 semi-private at 85.8 % (Sol, Max) to 93.75 % (Fable, XHigh; Neon). Caveat: the relay
calls them at low reasoning effort, so these scores do not carry over directly. The grid lets them use that strength
on weight structure: symmetric blocks, periodic stripes, a few dominant units.

The caveat is printed with every grid: *the picture depends on ordering and quantisation; only gauge-invariant views
support claims.*

### 2.5 ARC open-weight models and the Code Golf / NeuroGolf bridge

- **Open ARC models that fit this CPU.**
  - **TRM**: `arcprize/trm_arc_prize_verification`, MIT. 6,830,082 core parameters and a puzzle-embedding table of
    876,406 × 512. ARC Prize verified 40 % on ARC-AGI-1 and 6.25 % on ARC-AGI-2. Measured: 11.7 s per sample in fp32,
    2.8 s in bf16 (AMX); an 8-view vote takes 22–94 s per task.
  - **HRM**: `sapientinc/HRM-checkpoint-ARC-2`. The code is Apache-2.0; the checkpoint has no licence. 27.3 M core
    parameters and a 1,045,829 × 512 table.
  - Both are **transductive**: they answer only tasks whose puzzle IDs were in training.
  - The 2-layer TRM block is reused across H/L cycles, so "layer k" means `iter(k)`.
  - The 8B solvers (ARChitects, BARC, NVARC/Qwen3-4B, which has no official weights) are out of scope for CPU weight
    work.
- **Code Golf / NeuroGolf bridge.** `use golf:taskNNN` loads a *compiled* network for a whitelisted arc-dsl program. MVP
  primitives: `rot90/180/270`, `hmirror`, `vmirror`, `dmirror`, `cmirror`, `replace/recolor` as a 1×1 conv, and
  fixed-factor `upscale`. Statements then check the compiled net on ARC-GEN examples:

  ```text
  use golf:task150                       # ARC 67a3c6ac, solver: vmirror
  C1: observe acc(arc-gen) on compile(vmirror)  expect acc(arc-gen) = 1    # measured 39/266 -> REFUTED
  C2: observe acc(arc-gen) on compile(size_gate(vmirror))  expect acc(arc-gen) ?
  ```

  This is a natural synergy point. The relay models are strong at ARC, so they propose the *program* fix (a size
  gate); the executor compiles it and measures. This is NeuroGolf's problem, scored by exactness here, not by size.

---

## 3. Grounding: models, executor, compute

### 3.1 Model zoo

| Tier | Substrate | Size, licence | What it grounds | Status |
|---|---|---|---|---|
| 0 | random-weight nets from an ArchCard (tiny NeoX, RMSNorm toy, linear ResNet) | in memory | the algebraic checker, offline tests (ms) | spike `spike/symmetry_spike.py` (numpy) |
| 1 | **pythia-14m** @cf967c0a (GPT-NeoX) | 28 MB fp16, Apache-2.0, 154 training checkpoints | gauge, reparam, edit, equiv, path | own float64 forward, matches HF to 1.8e-11 with float32 RoPE tables; 103 ms per pass; 29 statements in 13.7 s |
| 1b | Pythia-70M-deduped plus SAELens `pythia-70m-deduped-res-sm` | 166 MB + 134 MB per SAE layer | direction vs feature, Equiv rules | measured in `ops_scripts` |
| 2 | **modadd grokking toy** (Nanda recipe) | 226,688 params, own code | train, path | p=113 / 0.3: 43 ms/step on 4 idle threads (about 7 min per 10k steps); p=97 / 0.5: 90 % test at step 1,250 (181 s, contended) |
| 3 | **Golf-compiled nets** from arc-dsl primitives | own code; ARC data Apache-2.0; Code Golf file licence **UNVERIFIED** | program ↔ weights ground truth | spike `scripts/compile_perm.py` |
| 4 | **TRM core** + puzzle_emb; HRM second | 1.8–2.5 GB checkpoints (pickle) | ARC gauge, edit and TaskRow statements | gauge spike on real weights (11 f64 forwards in 12.7 s); a full forward with RoPE/ACT is not yet written |
| 5 | Pythia 1.4B–6.9B, 7B fine-tunes | GPU | scale checks | B200 (§3.3) |

Notes:

- The scratch directory already holds the TRM and HRM checkpoints (about 4 GB). It can be cleared by a reboot, so the
  build should keep models in `~/.cache/huggingface` or another durable path.
- Load pickles only with `torch.load(weights_only=True, mmap=True)` in a `python -I` subprocess, then convert them once
  to safetensors and `.npy`.
- How TRM's 16-slot puzzle prefix maps onto the 512-d table rows differs between the lens reports (one row per ID
  vs a 16×512 block per ID). The vocabulary-to-colour mapping is also **UNVERIFIED**. Settle both while porting the
  forward.

### 3.2 The executor

```
text ──parse──► AST ──normalise──► canonical text
     ──resolve(ArchCard)──► typed selectors (layer/head/dim ranges checked against shapes)
     ──static symmetry check──► reject | ok          (ms, no torch needed)
     ──compile──► chain of pure functions  dict[str, Tensor] -> dict[str, Tensor]   (or a TrainSpec)
     ──run──► float64 forward(s) on the probe set: edited, base, control (random_like of equal ‖Δθ‖), hook twin for equiv
     ──observe──► values ──claims──► verdict ──► Result + Observation text
```

- **Result record**:
  - `stmt_id`, `canonical`, `kind`;
  - `static` {ok, messages};
  - `verdict`, `values`, `control`, `base`;
  - `setup` {model@rev, probe id and hash, dtype, seeds, torch version, wsi version};
  - `runtime_s`, `cache_key`.

  The cache key is sha256 of the canonical text, setup and code version, so a repeated statement costs nothing.
- **Determinism**: float64 by default (in fp32 the noise floor on Pythia-70M is about 1.7e-2, because max|logit| is
  1583); group elements are drawn from `seed`.
- **Probe sets** are registered by name. The spikes used one 149-token text, so their Δloss values are indicative.
  The MVP ships `own149` plus a 20-passage self-written set, which avoids licence questions.
- **Budgets**: per statement 5 s CPU (static and edit), per relay segment 15 s. Training runs are capped by `steps` and
  run in a background process.
- **Safety**: the whitelist grammar never evaluates code from statements. Model files are untrusted data: safetensors
  for Pythia, and the `weights_only` pickle path in a subprocess for TRM/HRM. No downloaded third-party code is
  executed; Tracr/ALTA stay a stretch goal for that reason.
- **Reused code**: the forward and ops of `feas_scripts/pythia_ws.py`; the grokking loop of `ops_scripts/grok.py` /
  `feas_scripts/steer_grok.py`; the Equiv patterns of `ops_scripts/weight_ops.py`; `scripts/compile_perm.py`. All of
  them are the lens agents' own code, to be rewritten cleanly into the package.

### 3.3 What runs here, what needs a B200 (≤ 1 h), what neither covers

| Runs on this CPU (4 cores, 15 GB) | Needs a B200 for ≤ 1 h | Out of reach |
|---|---|---|
| every static check (ms); gauge, reparam, edit and equiv on pythia-14m (0.1–1 s each), Pythia-70M (2–18 s), Pythia-160M (estimate: seconds) | grokking **phase diagrams**: about 10³ toy runs with `vmap` over (Grokfast α, λ; shrink c, t; wd; donors). About 5.7e16 FLOP, estimate about 10 min, **UNVERIFIED** | retraining TRM (about 3 days on 4×H100) |
| path statements over Pythia-14M checkpoints (about 4 s per pair; barrier 3.46 nats at 1k, 1.96 at 60k, 0.05 at 100k, 0.00 at 120k against step 143k) | edits and checkpoint paths on Pythia 1.4B–6.9B (limited by download; about 14 GB per 6.9B checkpoint, **UNVERIFIED**) | test-time training for new ARC tasks at scale |
| modadd training: 1–7 min per run; 5–10 seeds per arm in about 1 h | PLI-style slerp insertion on a 7–8B model with a benchmark eval (**UNVERIFIED** in 1 h) | claims about frontier models' own weights (closed) |
| TRM core gauge and edit statements; single samples (2.8–11.7 s); 8-view votes (22–94 s per task) | own SAEs/transcoders; 7B contrastive weight steering (both **UNVERIFIED** in 1 h); the TRM 1000-view vote for many tasks | ARC-AGI-3 (interactive games, not static grids) |

The MVP needs no GPU. The executor takes `device=` and the same code runs on CUDA.

---

## 4. Relay integration: the `weightspace` preset

### 4.1 Roles and loop

```
question (owner's text or a .wsi file)  +  WSI reference card  +  source pack (arXiv IDs, short own summaries)
   │
[translate]  routed by the index; ai_ml → Fable      "Source: …"  "Statement: …"  "Prediction: …"
   │   executor: parse → type-check → run (≤ 5 s) → Observation note (or "not run" with the error)
[critique]   another model (the translator is excluded)     typed, corrected statement + new prediction
   │   executor → Observation
[formalise]  only if the last note had prose and no valid statement (Sol or Neon)
   │   train statements run in the background → their Observation is appended when ready
 … main notes …
FINAL ANSWER (routed answer model) → status table: statement · kind · measured · status · source
```

The three parties:

- **the owner** writes statements and questions, or leaves `expect … ?` open;
- **the models** translate, predict, critique and propose programs;
- **the executor** measures.

This is the "all three in the mix" loop, in both readings: three models, and human + models + machine.

### 4.2 Segment kinds and schedule

The preset builds on the explore machinery (`ExploreConfig`, plan per segment, record gate, continuations; DESIGN.md §7).
Lenses are off, and four kinds are added. `SegmentKind` in `types.py` is frozen except for additive changes through the
integrator, so `translate`, `critique`, `formalise` and `observation` are additive.

| Kind | When | Routing | Directive (prose; see §4.5) |
|---|---|---|---|
| `translate` | the next unused source-pack item; at most `ws.translate_max` per run (default 3) | index (topic ai_ml → Fable) | turn the source's finding into one statement plus a prediction |
| `critique` | after every Observation of a model statement | exclude the writer of the statement (as `check_other_model`) | type it, find the missing reparam or align, compare scales, correct it |
| `formalise` | the last note has no parseable statement, or the executor rejected one | exclude the writer; `ws.kind_models.formalise` defaults to Sol and Neon | write the statement in card notation |
| `main` | otherwise | index | continue |
| `observation` | after any committed note that contains statements | **no model call**; `model = "executor"`; excluded from the routing window (like lens notes) | — |

### 4.3 Observation notes

The executor appends one block per executed note. The relay keeps it append-only, so prompt caches still hit. Example
(the numbers are the measured pythia-14m values from `spike.md`):

```text
Observation: S4 'edit antipair(L2 -> L4)' on pythia-14m@cf967c0a (float64, probe own149, 0.4 s): Δloss +2.511
(delete L4 only +2.158, delete L2 and L4 +4.064); the residual after L4 differs from 'delete L2 and L4' by 104 %;
random control of equal norm +16.002. The prediction in note 5 ('the pair cancels, resid_diff < 0.1') is not
supported.
Observation: S5 'gauge rescale(L3.neurons, c: Pos(512))' not run: GELU is not positively homogeneous, so positive
rescaling is not a symmetry of pythia-14m. Closest valid symmetry: perm(L3.neurons). Write 'edit' to measure it.
```

Rules:

- Only the executor writes lines that begin with `Observation:`. A model-written line with that label is committed as
  `Claimed observation:` and flagged in `segment_end.data["ws"]`.
- A statement that is still pending (a training run) gets `Observation: S7 queued (train, about 3 min).` The result
  follows as a later block.

### 4.4 Routing by the cutoff index, and the source pack

The measured `ai_ml` cell, from README.md:

| Model | `current` (open question) | `mcq` (recognition) | Hallucination |
|---|---|---|---|
| Fable 5.2 | 2026-04.6 | 2026-09.8 | 4 % wrong authors, 0/21 fake titles |
| Gemini 4 Neon | 2026-01.3 | 2026-04.5 | 1 % wrong authors, 0/21 fake titles |
| GPT-6.2 Sol | 2025-09.9 | 2025-10.0 | 26 % wrong authors, 4/21 fake titles |

What follows from that:

- With the default `softmax` policy, Fable writes the translate and main notes.
- The `critique` exclusion brings in Neon or Sol. The H term (λ·H) prefers Neon.
- Sol is the fastest (about 4.7 s per segment) and cheapest, so it suits `formalise`, which needs no literature.

The most relevant work lies **beyond every model's cutoff**:

- iWSP, WSP-Scope and 2611.09376 (Nov 2026);
- BrainSurgery (Jun 2026);
- tPD and the Truong grokking papers (Jun/Jul 2026);
- "What Accuracy Hides" (Oct 2026);
- TransformerLens 4.0 (Sep 2026).

These reach the notes through the **source pack** (`data/ws/sources.jsonl`: arXiv ID, title, date, a one- or
two-sentence own summary, and key numbers). The pack is built from the four lens reports, and the arXiv metadata is
re-fetched with the existing `bench/sources/papers.py` client. Storing only IDs and own summaries avoids redistributing
abstracts.

A **citation check** compares every `cite arXiv:…` and `Source:` ID with the pack, and can query the arXiv API. Its
Observation reads `source not in pack` or `title mismatch`. This is a second machine check, aimed at Sol's measured 26 %
wrong-author rate.

### 4.5 Prompts: wording constraints and the Fable filter

Constraints carried over from DESIGN.md §1 and §7:

- no "thinking", "reasoning", "reason", "scratchpad" or "chain of thought";
- no XML-like tags;
- labels are plain words ending with a colon;
- `tests/test_prompts.py` checks every text.

WSI notation uses `:=`, `->`, `<` and `>=`, but never paired angle-bracket tags.

**Fable input filter** (README and DESIGN.md §7.8). In the explore preset, Anthropic's filter blocked 2 of 4 Fable calls
under the category "reverse engineering or duplicating model outputs". The trigger was the labelled-items system prompt,
and separately the missing-items list in the tail. The default template always passed, and single label phrases in the
tail passed.

The design therefore:

1. Lists `claude-fable-5.2` in `ws.light_schema_models`. Fable gets the **default** system template, and a tail with
   the note number plus one prose directive: no label list and no checklist.
2. Places the **reference card** (below, about 25 lines) and the source pack in the *question* block, not in the system
   text. Whether a notation card in the user content trips the same threshold classifier is **unknown**. So is the
   effect of the topic itself (editing model weights, with the category name mentioning model outputs).
3. **Pilot before any live run**: about 20 single-segment Fable calls, bisecting card / source pack / directive / notes
   with Observation blocks, as the explore bisection did with 17 calls. Cost about $0.3–0.5. Needs the owner's OK.
4. **Fallback**: if Fable is blocked whenever the card is present, Fable writes prose only (`translate` without
   `Statement:`). The next `formalise` note by Sol or Neon writes the statement. The relay already re-routes on
   `content_filter` and never rewords a refused prompt.

Drafts, to be checked against the wording tests and the pilot:

```text
WS_SYSTEM (non-light models):
Several analysts are jointly working on a question about the parameters of small neural networks and keep one
shared set of working notes; the notes so far are included, after a reference card for a one-line statement
notation and a list of sources. A program applies statements to real model files and adds its results to the notes
as lines that begin with 'Observation:'; only that program writes such lines. Contribute the next note in about {K}
tokens, written so it can be appended as the next paragraph. A note has two to four short items chosen from:
'Source:' (a finding from the source list, or from memory, and then say so); 'Statement:' (one line in the notation
of the card, at most three per note); 'Prediction:' (the number or sign you expect the program to report for a
statement, and on what grounds); 'Critique:' (what is wrong or unclear in an earlier statement, for example a missing
fold_ln or align step, or a different model scale); 'Open question:'. If a statement was rejected or refuted, correct
it. Keep the labels in English; write the rest in the language of the notes. The last paragraph of the user message
says whether a conclusion is accepted; only then may you write '{final_marker}' followed by the answer. Output only
the note, then {sentinel}.

TRANSLATE_TEXT (directive; light form for Fable):
For this note, take source {n} from the source list ({title}) and say in two or three sentences what it found about
model parameters or training, including the model size it was found at. If you can, end with one line that begins
'Statement:' and writes that finding in the notation of the reference card.

CRITIQUE_TEXT:
The program has just reported on statement {sid}. Begin this note by testing the statement against that result: is it
a symmetry, a change of form, an edit or a training statement in the sense of the card; does it need fold_ln or align
before it means anything; does the cited finding come from a different model scale. Then give a corrected
'Statement:' and a 'Prediction:' for it.

FORMALISE_TEXT:
The previous note states a finding in words, or its statement was rejected. Write that finding as one line beginning
'Statement:' in the notation of the reference card, and add a 'Prediction:'.

ANSWER_WS_SUFFIX:
End the answer with a table with one row per statement: the statement, its kind, the measured result from the
'Observation:' lines, a status (theorem, paper result, measured here, open, false) and the source.
```

Reference card (in the question block):

```text
WSI card v0. One statement per line. Kinds: gauge X (claims all outputs stay the same; checked to 1e-9) ·
reparam fold_ln | center_writers | ln_to_rms · edit X (changes weights; the program reports dloss and a random
control of equal size) · equiv E == hook(H, f) · train T with D · path E for t in a..b · observe O.
Targets: resid, embed, unembed, L3, Lmid, Lmid-1, L*.writers, L3.neurons, L3.heads, L3.head[0].vo,
L3.head[0].qk[nonrotary], dir(name). Groups: O(d), Stab1(d), GL(d), S(n), Pos(n), RopeComm(d).
Weights: lerp(a,b,s), slerp(a,b,s), align(a -> b), neg(a), ckpt(run,step), a + s*(b - c).
Edits: swap(a,b), delete(a), insert_after(a,E), rotate(a, R: group), antipair(a -> b), parallel(a..b),
project_out(dir,writers), steer(dir,s,layer), rank_one(a,u,v,s), scale(a,s).
Training: modadd(p=..,frac=..) recipe nanda; grokfast_ema(a,l), clamp_norm(c), logit_scale(s),
implant_direction(E,step), at(step) edit ..., seeds(n).
Observables: dloss, max_dlogits, rel_dlogits, acc(test), t_grok(0.95), barrier, resid_diff(a, edit), cos(a,b),
spectrum(a), grid(a). Expectations: expect O < x | O > x | O ~ x tol e | O in [a,b] | O ?   Sources: cite arXiv:ID.
Loaded model: pythia-14m (GPT-NeoX, L0..L5, d 128, 4 heads x 32, rotary dims 0..7, MLP 512, GELU, LayerNorm,
parallel residual).
```

### 4.6 Gate and answer

The explore record gate is reused with WS items. A conclusion is accepted when all of these hold:

- at least `ws.min_executed` statements were executed (default 3);
- every `Prediction:` has a later Observation, or its statement is marked `NOT_RUN` with a reason;
- one `Open question:` exists;
- no training run is pending, or `ws.max_wait_s` (default 240 s) has passed; the answer then lists the run as pending.

Otherwise the relay forces a continuation, as explore does. Light-schema notes carry few labels, so the checklist is met
by the other models' notes, as in explore.

### 4.7 The owner as weight-space editor

- **CLI**:
  - `cutoffmix ws check FILE|-` parses and type-checks, and needs no torch.
  - `cutoffmix ws run FILE [--model … --seed … --json]` executes.
  - `cutoffmix ws pic SELECTOR [--order … --palette … --out x.png]` draws a picture.
  - `cutoffmix run --preset weightspace --ws-file FILE "question"` sends the file to the relay.
- **Diagnostics** (`file:line:col: severity: message`, each with a fix):
  - `error: unknown selector 'L7' (pythia-14m has L0..L5)`
  - `warning: 'mid' is ambiguous for 6 layers (L2 or L3); using L3`
  - `error: gauge rescale(L3.neurons) is not a symmetry: activation GELU is not positively homogeneous; use perm(...) or write 'edit'`
  - `error: gauge rotate(L3.resid_io, ...) is not in the symmetry group: residual rotations must cover readers, writers, embed and unembed, and need fold_ln (LayerNorm)`
  - `error: GL(32) on L3.head[0].qk includes rotary dims 0..7; use qk[nonrotary] with GL(24) or qk[rotary] with RopeComm(8)`
  - `error: project_out expects a direction; feat(sae(...)) is an input-gated feature (use hook(..., zero(feat(...))))`
  - `warning: lerp(L2, L4) mixes two layers without align; result depends on the permutation gauge`
- **Completion**: `complete(text, cursor, arch)` offers layer ranges, parts, head indices, dimension ranges (rotary or
  non-rotary), groups with the correct dimension for the chosen space (O(128) for resid, GL(32) for vo), names from
  `let` and `save`, and registered probes and models.
- **Preview picture** for the selector under the cursor. With an edit statement selected, it shows before / after /
  difference, defaulting to the gauge-invariant view.
- **Web UI** (stretch): a WSI panel next to the relay with line diagnostics (`POST /api/ws/check`), a completion
  dropdown (`/api/ws/complete`), Run (`/api/ws/run`, streamed results), "Ask the relay" (preset weightspace), and
  Observation badges in the notes.

### 4.8 Metrics and a synergy test

`done.data["ws"]` holds:

- statements: proposed, parsed, rejected statically, executed, and per verdict;
- predictions: made, checked, sign hit, within a factor of 2, broken down by model;
- corrections: a statement revised after FAILS or REFUTED;
- citations: in pack, not in pack, title mismatch;
- pending training runs;
- executor seconds.

**Synergy is a hypothesis.** The MVP makes it measurable:

- **Questions**: 10–12 owner questions, the three ideas plus items from `ops.md` §5. Examples: "is the OV product the
  object?", "where does Pythia-14M enter its final basin?", "does Grokfast move circuit formation or cleanup?".
- **Arms**: `weightspace` (relay), and `fixed:claude-fable-5.2`, `fixed:gpt-6.2-sol`, `fixed:gemini-4-neon`.
- **Compare**: executed statements per run, prediction hit rate, corrected false statements, and cost per CONFIRMED or
  correctly refuted statement.
- **Claim**: the relay beats the best single model on these.

---

## 5. MVP, stretch, limits, costs

### 5.1 Modules (new subpackage `cutoffmix/ws/`, optional extra `.[ws]` = torch CPU, safetensors, scipy, matplotlib)

Line counts are estimates.

| File | Content | Needs torch | Lines |
|---|---|---|---|
| `ws/grammar.py` | tokenizer, recursive-descent parser → dataclass AST, unicode normaliser, canonical printer, positioned errors | no | ~450 |
| `ws/arch.py` | `ArchCard` from `config.json` (GPT-NeoX), modadd, tiny fixtures; selector resolution; static symmetry table; completion | no | ~350 |
| `ws/ops.py` | `state_dict → state_dict` ops: fold_ln, center_writers, ln_to_rms, rotate_resid, perm (neurons, heads), basis (VO/QK/RopeComm), rescale, swap, delete, lerp, slerp, align (Hungarian + head brute force), antipair, parallel, project_out, steer, rank_one, scale, patch, quantize, random_like | yes | ~500 |
| `ws/models/neox.py` | float64 GPT-NeoX forward with hook points and parallel residual (port of `pythia_ws.forward`) | yes | ~170 |
| `ws/models/modadd.py` | Nanda toy, trainer, directives (grokfast_ema/ma, wd_schedule, clamp_norm, logit_scale, at, implant_direction), checkpoints | yes | ~280 |
| `ws/observe.py` | observables, controls, claim evaluation, verdict rules | yes | ~260 |
| `ws/picture.py` | views, gauge-invariant composites, PNG and text grids | optional mpl | ~160 |
| `ws/executor.py` | pipeline, budgets, `python -I` subprocess for untrusted loads, background training, cache | yes, lazy | ~280 |
| `ws/notes.py` | extract statements from notes, Observation formatting, the `Claimed observation:` rewrite, citation check | no | ~150 |
| `cli.py` (+) | `ws check`, `ws run`, `ws pic`, `--ws-file` | — | ~120 |
| `data/ws/` | `card_v0.txt`, `owner_examples.wsi` (§1.6), `sources.jsonl`, `probes/own149.txt` and a 20-passage set | — | — |
| tests | parser golden tests (owner examples, error positions); static-check tests; symmetry tests on a random tiny NeoX (ms, offline); executor tests on fixtures; modadd smoke (100 steps); notes extraction | per module | ~800 |

Phase 2 (relay): `config.py` gets `WeightspaceConfig` (enabled, model, probe, card, sources, max_statements_per_note 3,
stmt_budget_s 5, segment_budget_s 15, dynamics background|cache_only|off, max_wait_s 240, min_executed 3,
light_schema_models, kind_models). Other Phase 2 pieces:

- `data/presets/weightspace.yaml`: segment_tokens 192, hard_cut_chars 1500, max_segments 16, answer_mode routed;
  explore protocol on, lenses off.
- the WS texts in `prompts.py`, with wording tests;
- a `relay.py` hook after `segment_end` (executor → observation segment);
- an additive `SegmentKind` in `types.py`;
- mock-backend behaviour for WS tails, so the whole loop runs offline;
- CLI/server pass-through.

About 700 lines plus 450 lines of tests (estimate).

Phase 3 (ARC slice):

- `ws/models/golfnet.py`: whitelisted arc-dsl primitives compiled to one-hot (10,30,30) torch nets, ARC-GEN accuracy.
- `ws/models/trm.py`: own TRM forward with RoPE and ACT, written from the MIT `trm.py` as a reference and not executed.
  Validated against Trelis per-step `submission.json` for a few tasks. Adds `TaskRow` selectors and
  `arc_grid` / `arc_vote` observables.

About 600 lines plus 250 lines of tests (estimate).

### 5.2 Phases and effort (estimates)

| Phase | Deliverable | Spend | Effort (estimate) |
|---|---|---|---|
| 1 | `ws check/run/pic` on pythia-14m and modadd; the owner's three examples reproduce the §1.6 numbers within probe variance; offline tests green | $0 | about 1–2 developer-days; with the agent workflow used for the explore preset, roughly one session plus review |
| 2 | `--preset weightspace` running offline (mock) and live; Fable pilot; 2–3 live runs; metrics | ≈ $0.5 pilot + $1–2 runs (needs OK) | about 1–2 days |
| 3 | golf-compiled nets; TRM core forward; first TaskRow hypothesis (D4 map between puzzle_emb rows) | $0; download of Trelis identifiers and one checkpoint (≥ 1.75 GB; identifiers size **UNVERIFIED**) | about 1–2 days |

### 5.3 Stretch goals

1. **TaskRow hypotheses on TRM.** Does a shared linear map A_g relate the rows of one task under augmentation g, fitted
   on half the tasks and tested on the rest? Does `slerp` between two tasks' rows produce a "between" program? Plus the
   Trelis 40-checkpoint trajectory as `path` statements ("is there a grokking-like transition in ARC training?").
2. **B200 hour**: `vmap` phase diagrams for G1, G2, G6 and G7 with confidence intervals; PLI and Painters statements on
   Pythia-1.4B/6.9B.
3. **Mechanism statements** (`mechanism P ~ M under tau`) via Tracr/ALTA/InterpBench. This needs permission to run
   third-party code, which the research session's policy refused.
4. **Export** of edit statements to BrainSurgery (MIT) or mergekit (LGPL-3.0) plans for large checkpoints.
5. **A "weight-space calibration" cell** in the router index: per-model prediction hit rate on executed statements.
6. **The web editor** with completion and pictures; Observation badges in the UI.
7. **The SAE / feature sort on Pythia-70M** (SAELens 6.56.1, +60 MB) and Equiv statements for SAE edits.

### 5.4 Honest limits: what this will not show

- **Scale.** Everything measured is on models of 14–70M parameters or toys. Painters' middle-layer uniformity was
  reported at 7B; on Pythia-14M/70M the middle layers are *not* interchangeable. Grokfast's ">50×" became 1.25–1.48×.
  WSI verdicts are statements about the loaded model only.
- **The models see numbers, not weights.** Their contribution is translation, prediction, typing and critique. Whether
  this beats one model alone is the hypothesis of §4.8, not a result.
- **Probe and seed variance.** The Δloss values came from one 149-token text and the grokking results from one seed.
  The MVP adds a 20-passage probe set and `seeds(n)`, but small effects (such as the 1.11× shrink) need many seeds.
- **The symmetry checker covers only the architectures it describes**: GPT-NeoX/LN, RMSNorm and SwiGLU, ReLU toys.
  Anything else returns "unknown", not "no symmetry". QK symmetries with RoPE rest on a derivation checked numerically,
  not on a cited paper.
- **ARC.**
  - ARC-AGI-1/2 are saturated for the relay models (85.8–93.75 % on ARC-AGI-2), so ARC is a formal substrate here, not
    a hard benchmark. Probes should use fresh ARC-GEN or re-arc instances.
  - TRM and HRM are transductive, and new tasks need gradient-based test-time training (GPU).
  - ARC-AGI-3 (interactive) is not covered.
  - ARC Prize 2026 results are due 2026-12-04, after this design.
- **Codebench** is an interpretation (§0). The NeuroGolf rules come from a community repository; winners and the Code
  Golf data licence are **UNVERIFIED**.
- **Fable's filter** may block weight-space notes. Then Fable contributes prose only, and the mix leans on Sol and Neon
  for statements (§4.5).
- **Sources after the cutoffs** reach the models only through the pack. Paper statuses are claims with references, not
  re-measurements.
- **Licences.** The HRM checkpoint, BARC, NVARC and the ARChitects 2025 repos have no licence file. They are not used
  in the MVP.

### 5.5 Costs

| Item | Cost |
|---|---|
| Phase 1 compute | CPU only. Disk: the optional `.[ws]` venv (torch CPU 747 MB; the whole stack about 1 GB, cold install about 25 s, measured) and pythia-14m (28 MB) |
| Phase 2 API | Fable pilot about 20 calls ≈ $0.3–0.5. A live run with about 14 segments, Fable-heavy (1.7–2.2 ct per segment) and a longer context (card plus Observations; caching helps) ≈ $0.3–0.6 (estimate; an explore run with 11 segments cost $0.29). |
| Synergy test | 10–12 questions × 4 arms ≈ 40–50 runs ≈ $12–25 (estimate) |
| Phase 3 | no API; downloads ≥ 1.75 GB (Trelis), plus the TRM/HRM checkpoints already on disk (about 4 GB, scratch) |
| B200 | optional, ≤ 1 h, owner-provided |

### 5.6 Decisions needed from the owner

1. **"Codebench"**: confirm Code Golf 2025 / NeuroGolf 2026, or name the resource.
2. **Phase 2 spend**: OK for about $0.5 of pilot calls plus a few live runs (about $1–2)?
3. **ARC timing**: TRM in the first build, or after the relay preset (recommended: after)?
4. **Dependency**: torch as an optional extra `.[ws]` (about 1 GB), keeping the default install lean.
5. **Storage**: a durable location for downloaded models, since the scratch directory can be cleared.

---

## Appendix: evidence index (source → what this design uses → where verified)

| Source | Used for | Verified in |
|---|---|---|
| SliceGPT arXiv 2401.15024 (Thm 1, §3.1–3.2) | global residual rotation, ln → rms | formal.md, spike.md (HTML sections) |
| Tran et al. arXiv 2410.04209; Zhang et al. arXiv 2502.00264 | per-head QK/VO groups | formal.md |
| Arditi et al. arXiv 2406.11717 §4.1 | project_out ≡ directional-ablation hook | ops.md |
| Painters arXiv 2407.09298; PLI arXiv 2506.02973 | parallel, middle repeat, slerp insertion | formal.md |
| Haber & Ruthotto arXiv 1705.03341; AntisymmetricRNN arXiv 1902.09689 | antisymmetric = rotation | formal.md (PDF text) |
| Sander et al. arXiv 2205.14612; Marion et al. arXiv 2309.01213 | depth smoothness, OPEN for LLMs | formal.md |
| Nanda et al. arXiv 2301.05217; Grokfast arXiv 2405.20233 | modadd recipe, grokfast_ema | ops.md, spike.md (PDF, repo) |
| Omnigrok 2210.01117; Truong 2606.18465, 2607.06628; Prieto 2501.04697 | clamp_norm, logit_scale, implant_direction, reserved directives | formal.md, ops.md |
| Fierro & Roger arXiv 2511.05408; task arithmetic 2212.04089; ROME 2202.05262 | weight steering, task vectors, rank_one | formal.md, ops.md |
| Git Re-Basin arXiv 2209.04836; Net2Net arXiv 1511.05641 | align, upscale | formal.md, arc.md |
| iWSP arXiv 2611.18114 + repo; WSP-Scope arXiv 2611.09382 | prior art | formal.md, ops.md; **re-opened today** |
| NeuroGolf official site; zikuanqi/NeuroGolf | ARC-as-networks | arc.md; **official site re-opened today** (no rules on it) |
| Code Golf 2025 (NeurIPS page; Sakana repo) | ARC-as-programs | arc.md (Kaggle pages unreadable; data licence UNVERIFIED) |
| TRM `arcprize/trm_arc_prize_verification`; HRM `sapientinc/HRM-checkpoint-ARC-2`; Trelis repos | ARC substrates | arc.md (loaded), spike.md (HF API) |
| arcprize.org leaderboard (2026-11-25) | relay models' ARC scores | arc.md |
| pythia-14m @cf967c0a | primary CPU substrate | spike.md (config from the file, HF check 1.8e-11) |
| TransformerLens 4.0 removal of HookedTransformer | why models must not write TL code | ops.md (measured import error) |
| Measured spikes | all MEASURED values | `feas_out/pythia_ws.json`, `ops_out/*.out`, `ops_out/*_s0.json`, `feas_out/steer/steer_results.json`, `scripts/gauge_spike.py`, `scripts/compile_perm.py`, `spike/symmetry_spike.out` |
