# cutoffmix `explore` preset: design and evidence

Date: 2026-11-24. Synthesised from the three research notes in this directory (`temperature.md`,
`diversity-without-temperature.md`, `verbalization.md`); only papers those notes marked `verified: true` are relied
on. Code references are to the current tree (`cutoffmix/prompts.py`, `relay.py`, `config.py`, `types.py`,
`backends/openrouter.py`, `router/router.py`).

## 0. Summary

The user wants (1) a chain of shared notes that wanders through other fields (metaphors, analogies, inferences from
other domains) between a research-style start and end, and (2) models that write down more intermediate steps.

- **Temperature cannot deliver (1).** Fable 5.2 and GPT-6.2 Sol have no temperature parameter, and Google's Gemini 4
  Neon documentation says the server *ignores* in-range `temperature`/`top_p`/`top_k` and returns 400 for out-of-range
  values. Even where temperature works, the literature finds it buys phrasing variety and incoherence, not new content
  directions (Peeperkorn 2024; Deng 2026; Jiang 2025). The preset therefore sends **no temperature or top_p to any
  model** and gets wandering from the protocol: a **client-side, seeded "domain lens"** injected into selected middle
  segments, each followed by a **forced check segment written by a different model**.
- **(2) comes from structure, not from "write more".** The preset replaces the one-paragraph note with a
  **labelled-items note** (Claim / Evidence / Inference / Principle / Alternative set aside / Open question / Next /
  Where we stand), raises the segment from 128 to **192 visible tokens**, gates `FINAL ANSWER:` behind a **minimum of
  6 main-line segments plus a record checklist**, and when a model concludes early, strips the conclusion and reopens
  the next segment with a **content-bearing cue** ("Alternatively, ...") written by another model, at most 3 times.
  Evidence predicts this makes the notes more legible and wider-ranging and roughly accuracy-neutral; it will not make
  answers better (Wu 2026; Ghosal 2025; Gema 2025; Caldarella 2026).
- **Cost.** The measured default live run (round-robin, 6 segments + answer, 128 tokens) was **0.15 USD / 57 s**. A
  typical `explore` run is 10-12 think segments + answer at 192 tokens: about **0.25-0.40 USD and 100-140 s with
  round-robin**, about **0.08-0.15 USD with the default softmax policy** (mostly Sol). With under 1 USD on the key:
  one `--mock` run (free), then **one live round-robin run**, then stop.
- **Routing caveat.** With the shipped prior index GPT-6.2 Sol is freshest on every topic, and lens segments are
  excluded from the routing window on purpose, so wandering topics will **not** change the writer until a
  topic-stratified index exists. Use `router.policy: round_robin` to see all three models.

## 1. Constraints the preset must satisfy

| Constraint (verified API facts, Nov 2026) | Consequence for the preset |
|---|---|
| Fable 5.2 and GPT-6.2 Sol: no `temperature`/`top_p`/`min_p`/`top_k`. Gemini 4 Neon: parameters accepted by OpenRouter, but Google's docs: "The server ignores in-range custom values if sent. Out-of-range values return a `400` validation error." | No sampling knob anywhere. Set `gemini-4-neon.supports_temperature: false`; `ChatRequest.temperature` default `None`. Exploration must come from the prompt and the protocol. |
| GPT-6.2 Sol supports `verbosity` (low/medium/high) and `seed`; OpenRouter maps `verbosity` to effort for Claude (already blocked in `config.py` `_validate_model` and `openrouter.py` `_apply_model_params`). | Sol: `verbosity: medium` for notes, `low` for the answer; `seed` per (run seed, segment) for A/B reproducibility. Nothing for Fable; Neon has no such knob. |
| Hidden reasoning is mandatory (minimum effort `low`), discarded, never shared. No logprobs. No prefill on Fable (400). | Continuation cues cannot be prefilled; they go into the per-segment tail instruction. Diversity cannot be read from token probabilities; it is measured on the visible notes. |
| Fable refuses prompts that look like reasoning extraction: never the words thinking / reasoning / scratchpad / chain of thought, no XML-like tags. | All new texts below are framed as "shared working notes of several analysts" with plain-word labels ending in a colon; they also avoid the stem "reason" altogether. The existing sentinel `<<END_STEP>>` is kept (it is live-verified). |
| Request bodies must stay append-only for prompt caches (static system prompt, one block per committed segment, constant tail). | Every per-segment instruction (lens, check, continuation, record status) goes into the **tail block**, which is already the only block never cache-marked (`openrouter.py build_continuation_body`: `cache_control` sits on `blocks[-2]`). Committed segments are never edited. |
| Live costs (README, 24.11.2026): Sol 0.4-0.8 ct, Fable 1.7-2.2 ct, Neon 0.9-1.6 ct per 128-token segment; 7 calls round-robin = 0.15 USD. | Hard cap on segments and lenses; the optional side calls run on Sol. |

## 2. The preset at a glance

New file `configs/explore.yaml` (deep-merged over `cutoffmix/data/default.yaml`; the `explore:` section is a new
`ExploreConfig` dataclass in `config.py`, `enabled: false` in the packaged default):

```yaml
relay:
  segment_tokens: 192          # was 128
  hard_cut_chars: 1500         # 192 tokens of German = 1100-1300 chars (measured); 'Where we stand:' must survive the cut
  max_segments: 16             # hard ceiling (16 x 192 ~ 3k note tokens, under the 4096 cap)
  answer_mode: routed

router:
  policy: round_robin          # to see all three writers; softmax+prior gives mostly Sol (see section 9)
  window_segments: 3           # unchanged; lens segments are excluded from the window by the relay

explore:
  enabled: true
  seed: null                   # run seed for every client-side draw; null -> random, logged in `done`
  min_segments: 6              # main-line notes (lens and check notes not counted) before a conclusion is accepted
  require_labels: ["Evidence:", "Alternative set aside:", "Open question:"]
  where_we_stand_window: 2     # a 'Where we stand:' line must appear in one of the last 2 notes
  max_forced_continuations: 3
  loop_overlap_jaccard: 0.5    # word-3-gram Jaccard of the last two notes above this -> stop forcing
  cues:
    - "Alternatively, "
    - "One thing not yet checked: "
    - "The strongest objection to the current conclusion is "
    - "Perhaps the missing piece is "
    - "Before closing, the weakest assumption above is "
  lens:
    p: 0.25                    # per eligible segment
    p_stagnation: 0.50
    min: 1                     # force one lens at the last eligible slot if none was drawn
    max: 2
    first_index: 2             # 0-based: never in the first two segments
    cooldown: 2                # segments after a check before the next lens may be drawn
    tiers: {mid: 0.55, far: 0.35, near: 0.10}
    exclude_from_window: true  # lens notes are not part of the routing window
    check_other_model: true    # the check is routed away from the lens writer
  vs_menu: off                 # off | stagnation | every_lens  (section 5)
  answer_basis: true           # answer ends with 'Basis:' and 'Not resolved:'

models:
  gemini-4-neon:
    supports_temperature: false            # Google ignores it; OpenRouter only passes it through
  gpt-6.2-sol:
    extra_body: {verbosity: medium}        # notes (was low)
    answer_extra_body: {verbosity: low}    # new optional field, applied in phase 'answer'
    supports_seed: true                    # new flag: body['seed'] = hash(run seed, segment index)
```

Everything else (sentinel, retries, refusal handling, provider pinning, caching) is unchanged.

## 3. Component 1: Neon temperature / top_p schedule per phase

**Decision: omit `temperature` and `top_p` in every phase, for every model.**

| Phase | Segments | temperature | top_p | Why |
|---|---|---|---|---|
| Anchor | 0-1 | omit | omit | Neon ignores it (Google docs); Fable/Sol have no parameter. |
| Main line | 2 .. | omit | omit | same |
| Lens / check | flagged | omit | omit | same; exploration is injected by the lens text (section 4) |
| Converge + answer | last 2 + answer | omit | omit | same |

Evidence that nothing is lost by this: temperature correlates weakly with novelty (beta .308) and moderately with
incoherence (beta .240), with no effect on typicality (Peeperkorn 2024); GPT-4o at T 1.0 -> 1.5 adds one idea
category (22 -> 23) and is garbled at 2.0 (Deng 2026); even min-p at T 2.0 leaves 61% of same-model answer pairs
above 0.8 similarity (Jiang 2025); varying DeepSeek-R1 temperature 0.05-2 has no effect on research-agent
performance while prompt-controlled ideation diversity does (Audran-Reiss 2025); strong aligned models gain nothing
from temperature even on word-level divergence (Chen & Ding 2023). Multi-step chains collapse at T >= 1.5 with any
truncation sampler (Nguyen 2024: GSM8K-CoT 0.0 at T 1.5 temperature-only), and the relay is one long chain.

**Conditional schedule, only if a cheap A/B test shows the Vertex path honours the parameter** (not run in this
budget; about 0.15-0.20 USD: Neon, one short analogy prompt, n=8 with temperature omitted vs n=8 at 2.0, effort low,
compare mean pairwise similarity and distinct-2, plus one call at 2.5 that must return 400): anchor and converge
omit; lens segments only `temperature 1.2` (hard cap 1.3) with `top_p 0.9`; preferably IFG-style, a lens line of at
most 30 tokens hot and the note itself at default (Ahmed 2025). Never >= 1.5 (Nguyen 2024; Renze & Guven 2024:
incoherent at 1.6), never below 1.0 on Gemini. Creativity optimum for large models is about 1.3 with an abrupt
instruction-following break at 1.6-1.9 (Li 2025, weak method); degradation is a cliff at an entropy turning point,
not a slope (Du 2025); top-p must trim the tail if T is raised (Holtzman 2019).

Neon `reasoning.effort` stays `low` for all think segments (`minimal` is rejected); `high` only for the answer if a
later budget allows it.

Code: `cutoffmix/data/default.yaml` Neon `supports_temperature: false`; `types.py` `ChatRequest.temperature: float |
None = None`; `_apply_model_params` already omits temperature when unsupported or None.

## 4. Component 2: domain lens (all three writers)

### 4.1 Why a client-side lens

- The model must not choose the field. Asked for 10 analogies, 10 frontier LLMs (incl. Claude Sonnet 4.6, GPT-5.2,
  Gemini 3.1 Pro) produce about 2 semantic clusters; target-domain diversity <= 0.63 for every model (Shen Y. 2026).
  Asked for randomness, aligned models return the same values (Zhang Y. 2024: "Avery" 40x; West & Potts 2025: "7").
  Explicit entropy injection works, but the seed should come from the client, not pollute the shared notes (Misaki &
  Akiba 2025).
- Random cross-field association raised originality by 40% but lowered usefulness (Mehrotra 2024); explicit
  function-based relational mapping in science tasks gives +90-173% solution diversity and ~3.5x more unique domains,
  with novelty 50.4% vs 21.6% for a plain cross-domain prompt, but reasonableness 67% vs 86% (Shen, Druckmann & Zou
  2026). Hence: mandatory mapping template and a mandatory check.
- Analogies mostly reframe the problem (80% changed the formulation; Ding 2023) and are brittle (Lewis & Mitchell
  2024), so the check tests the mapping, not the prose. Intrinsic self-correction does not work (Huang 2024);
  independent verification does (Dhuliawala 2024); diverse models help checking (Chen, Saha & Bansal 2024). Hence
  the check is routed to a different model.
- Far sources yield new directions (odds ratio ~1.9 for new-concept generation; Chan & Schunn 2014) but lower final
  quality (-0.33 log-odds per unit distance; Chan, Dow & Schunn 2015) and there is a "too far" boundary (Kwon &
  Goucher-Lambert 2023). Hence a mid/far/near mixture, not always the farthest field.

### 4.2 Schedule (client side, before the routing decision of segment `idx`)

```
rng = random.Random(f"{explore.seed}:{idx}")
eligible = (idx >= lens.first_index and not gate_open and cooldown == 0 and not pending_check
            and lenses_used < lens.max and forcing is None)
last_slot = (eligible and main_line_count == min_segments - 1)          # last chance before the gate can open
if pending_check:
    directive = CHECK_TEXT(field);  exclude = {lens_writer} if check_other_model and len(available) > 1
elif eligible and (rng.random() < (lens.p_stagnation if stagnating else lens.p)
                   or (last_slot and lenses_used < lens.min)):
    tier   = rng.choices(["mid", "far", "near"], weights=[.55, .35, .10])[0]     # far 0.50 when stagnating
    field  = rng.choice([f for f in FIELDS if tier_of(f.home_topic, current_topic) == tier and f not in used])
    c1, c2, c3 = rng.sample(field.concepts, 3)
    directive = LENS_TEXT(field, c1, c2, c3);  pending_check = True;  lens_writer = chosen model
else:
    directive = ""   # plain main-line note
after a committed check: cooldown = lens.cooldown
```

- Probability per eligible segment: **0.25** (0.50 when stagnating), `min 1`, `max 2`, cooldown 2. With
  `min_segments 6` the eligible slots are roughly segments 2-7, so a run gets 1-2 lenses (1 guaranteed) and 1-2 checks.
  These rates are engineering heuristics (no paper measures a per-segment injection rate); they are anchored in
  "about one in three analogies is unreasonable" (Shen 2026), the Degeneration-of-Thought finding that divergence is
  most useful when the chain stagnates (Liang 2024), and duplication plateaus in science ideation (Si 2024: 5% unique
  at cosine 0.8). A deterministic alternative (lens at segments 2 and 5) is one line to change; keep the field draw
  random either way.
- **Stagnation** (cheap proxy, no embedding model): the classifier argmax topic is unchanged for >= 4 segments and
  the last note introduces no number or capitalised token not already in the notes. With `classifier: embedding`
  available: cosine >= 0.85 between the last note and the mean of the previous three (duplicate thresholds of 0.8 in
  Si 2024 and Meincke 2024).

### 4.3 How the lens field is chosen relative to the question's topic

The 15 router topics are news categories; the lenses the user wants are fields of knowledge. The preset keeps a
curated list `FIELDS`, each with a *home topic* (for distance) and 6-8 structural concepts:

```yaml
# field: [home topic, concepts]
thermodynamics:       [science_research, [entropy, free energy, phase transition, Carnot limit, fluctuation-dissipation, equipartition]]
materials_science:    [science_research, [fatigue, grain boundaries, annealing, creep, composite laminate, phase diagram]]
chemistry:            [science_research, [catalysis, activation energy, rate-limiting step, equilibrium shift, buffer, chirality]]
evolutionary_biology: [science_research, [fitness landscape, Red Queen dynamics, drift vs selection, exaptation, punctuated equilibrium]]
immunology:           [health_medicine, [clonal selection, immune memory, tolerance, antigenic drift, cytokine feedback, herd immunity]]
epidemiology:         [health_medicine, [reproduction number, base rate, lead-time bias, confounding, dose-response, screening trade-off]]
ecology:              [natural_disasters_climate, [carrying capacity, niche partitioning, keystone species, trophic cascade, succession, r/K strategies]]
geoscience_climate:   [natural_disasters_climate, [tipping point, return period, feedback loop, early-warning signal, cascading failure]]
safety_engineering:   [accidents, [Swiss-cheese model, normal accidents, redundancy, common-mode failure, near-miss reporting, root cause]]
orbital_mechanics:    [space_astronomy, [orbital resonance, Lagrange point, gravitational lensing, redshift as a clock, Hohmann transfer, parallax]]
control_theory:       [computing_software, [feedback gain, stability margin, observability, hysteresis, dead time, PID control]]
information_theory:   [computing_software, [channel capacity, compression, error-correcting code, caching, race condition, graceful degradation]]
machine_learning:     [ai_ml, [overfitting, distribution shift, exploration vs exploitation, regularisation, curriculum, reward hacking]]
economics:            [business_economy, [arbitrage, moral hazard, network effects, comparative advantage, bubble, Goodhart's law, principal-agent]]
game_theory:          [international_relations, [deterrence, credible commitment, prisoner's dilemma, focal point, tit-for-tat, free-riding]]
military_logistics:   [conflict_security, [attrition, supply line, fog of war, escalation ladder, centre of gravity, decision loop]]
political_institutions: [government, [checks and balances, policy lag, succession crisis, veto player, path dependence]]
electoral_systems:    [elections, [Arrow's theorem, median voter, gerrymandering, Duverger's law, turnout effect, runoff vs plurality]]
law_and_evidence:     [law_crime, [burden of proof, precedent, adversarial testing, chain of custody, standard of evidence]]
linguistics:          [arts_entertainment, [grammaticalisation, sound change, Zipf's law, code-switching, creole formation, semantic drift]]
music_theory:         [arts_entertainment, [counterpoint, tension and resolution, motif development, harmonic series, polyrhythm]]
narrative_and_visual_arts: [arts_entertainment, [narrative arc, negative space, foreshadowing, remix, genre convention]]
sports_science:       [sports, [home advantage, regression to the mean, periodisation, doping arms race, scouting base rate, formation trade-off]]
```

Distance tiers come from a fixed ordering of four topic groups along a line, Science - Tech/Economy - Politics -
Culture:

| Group | Topics |
|---|---|
| Science | science_research, health_medicine, space_astronomy, natural_disasters_climate, accidents |
| Tech/Economy | ai_ml, computing_software, business_economy |
| Politics | government, elections, international_relations, conflict_security, law_crime |
| Culture | sports, arts_entertainment |

`tier_of(home, current)`: same group -> **near**; adjacent group -> **mid**; two or more groups apart -> **far**.
For a `science_research` question (the user's usual case): near = thermodynamics, materials, chemistry, evolution,
immunology, epidemiology, ecology, geoscience, orbital mechanics, safety engineering; mid = control theory,
information theory, machine learning, economics; far = game theory, military logistics, institutions, electoral
systems, law, linguistics, music theory, arts, sports science. Tiers are sampled mid 0.55 / far 0.35 / near 0.10
(far 0.50 when stagnating); fields are sampled **without replacement within a run** (Shen Y. 2026: the models
themselves would reuse two clusters). `current_topic` is the argmax of the router's smoothed posterior before the
segment, or None when that posterior is flat (`taxonomy.argmax_topic`; the keyword classifier matched nothing):
then every field is `mid`. The 15 x 15 distance can later be replaced by 1 - cosine of embedded topic descriptions.

### 4.4 Exact instruction texts (English; Fable-safe)

`LENS_TEXT` (appended to the tail block of the lens segment):

> Cross-field lens for this note: the field is {field}. Choose one specific, well-established concept from this
> field (for example {c1}, {c2} or {c3}) whose structure resembles the current intermediate result. Give two or three
> short mappings stating what in our problem plays the role of what in that concept; match by function and relations,
> not by surface features. Then give one 'Inference:' item with a concrete consequence or testable question this
> suggests for the problem, marked as a working analogy rather than evidence. Begin the note with 'Lens ({field}):'
> and, before 'Where we stand:', write one sentence that returns to the problem. Use textbook-level knowledge of the
> field, not recent news.

`CHECK_TEXT` (tail block of the segment right after a lens; routed away from the lens writer):

> The previous note applied a cross-field lens from {field}. Begin this note by testing it in one to three
> sentences: which mapping holds, where the comparison breaks (name at least one real difference), and whether its
> suggested consequence survives. Write 'Lens kept:' or 'Lens dropped:' followed by why. If kept, use the
> consequence concretely in the rest of the note; if dropped, continue the main line without it.

The lens text forbids recent news so a lens cannot introduce cutoff-sensitive claims into the notes (the taxonomy is
news-based). "Working analogy rather than evidence" keeps the check and the final answer from treating the mapping as
a fact (Qin 2024: exemplar correctness matters more than relevance).

### 4.5 How coherence is restored

1. The lens note must open with `Lens ({field}):` and end with a return sentence plus `Where we stand:`. The relay
   checks the prefix with a regex; a missing prefix is committed anyway but logged as `lens_ignored`.
2. A **check note follows immediately**, written by a different model when more than one is available (the relay
   passes `available` minus the lens writer to `router.decide`). Rationale: AR solutions were reasonable only 67% of
   the time (Shen 2026), self-correction fails (Huang 2024), independent verification works (Dhuliawala 2024).
3. **Routing window:** lens notes are excluded from the window text (`Router.window(question, [s for s in notes if
   not s.is_lens], n)`), and the router's EMA is not updated on the lens decision. Check notes stay in the window
   (they carry main-line content). This stops a music-theory lens from steering the cutoff router.
4. **Final answer:** the answer system prompt gets one sentence: "Where the notes mark a cross-field lens as kept,
   use only its concrete consequence; never present a lens, kept or dropped, as evidence."
5. Cooldown of 2 segments after a check; no lens once the record gate is open (section 6.1), in the first two
   segments, or while a continuation is being forced.
6. Track the **keep rate** (`Lens kept:` / lenses). Expected 40-70% (Shen 2026). Below 25%: lower the far weight.
   Above 85%: the lens is too near or the check too lenient.

## 5. Component 3: optional verbalized-sampling "next-move menu" (Fable/Sol, client-side pick)

Verbalized Sampling asks for k candidates with verbalized probabilities and samples from the tail; it gives
1.6-2.1x diversity in creative writing, works on every model, grows with capability, and is orthogonal to
temperature (Zhang J. 2025). In the relay it is a **side call** (plain `chat`, not a segment):

System: "You advise a group of analysts who keep shared working notes on a problem."
User:

> PROBLEM: {question}
>
> SHARED NOTES SO FAR: {notes}
>
> List five candidate directions for the next note. Write each on its own line in the form 'n. direction | p=0.xx',
> where p is your estimate of the share of analysts who would take that direction next; the five values need not sum
> to one. At least three directions must have p below 0.10 and draw on fields of knowledge not yet used in the
> notes. One line each, at most 25 words per direction, no other text.

Parse: `^\s*(\d)[.)]\s*(.+?)\s*\|\s*p\s*=\s*(0?\.\d+|1(?:\.0+)?)\s*$` (multiline). Keep lines with p < 0.10 (if fewer
than two parse, take the two lowest p; if fewer than three lines parse at all, discard the menu and write a plain
note). Pick one with `random.Random(f"{seed}:{idx}:vs")`. Inject into the tail block: "Direction for this note:
{direction}. Develop it with the usual labelled items and return to the problem before 'Where we stand:'." Log the
menu, the chosen line and its p.

Cost multiplier: +1 call per use. On Sol (cheapest, fastest, supports `seed`; effort low, 200-250 output tokens):
about +0.4-0.6 ct and +3-5 s per use, i.e. +3-5% cost and +4-8% latency on a run with two uses. On Fable: about +2 ct
and +8-10 s per use. Numbered lines, no tags, no forbidden words, so the Fable refusal risk is low but the price is
not worth it.

**Verdict: not worth it as a default; ship as `vs_menu: stagnation` opt-in, always on Sol.** The client-side field
draw already supplies the entropy VS would supply; what VS adds is a *model-informed* direction that is relevant to
the notes, which matters when the main line has stalled, not every segment. Two cautions: models over-report the
probabilities, and asking a model for randomness in general is unreliable (Zhang Y. 2024; West & Potts 2025), so the
client still does the pick.

## 6. Component 4: verbalization

### 6.1 Minimum segments and the record gate before `FINAL ANSWER:`

A conclusion is accepted only when **both** hold:
(a) `main_line_count >= explore.min_segments` (6; lens and check notes do not count), and
(b) the record checklist passes: at least one `Evidence:` item, one `Alternative set aside:` item, one
`Open question:` item anywhere in the notes, a `Where we stand:` line in one of the last two notes, and no pending
lens check.

Why a floor plus a checklist rather than a token target: optimal chain length rises with difficulty and falls with
model capability (Wu 2026), so frontier models stop early on their own; the missing-step error class is what plans
and intermediate results fix (Wang L. 2023: 12% missing steps; PS+ +3-6 points zero-shot); prompt-stated length
targets are followed reliably only after RL training (Aggarwal & Welleck 2025), so an enforceable gate beats "write
at least N steps". Ceiling: `max_segments 16` keeps the notes near 3k tokens, under the "do not exceed 4096"
guideline (Taveekitworachai & Pipatanakul 2026); s1's gains flatten at six forced continuations (Muennighoff 2025).
Fixed `min_segments` is the implementable version today; a difficulty-scaled floor (4 / 8 / 12) needs a difficulty
estimator the relay does not have.

Mechanics: `split_final` already separates a note from the text after the marker. If the gate is closed, the answer
part is discarded, the note part is committed if non-empty (Markovian Thinker: forced restarts "overwrite useful
partial work"; keep what was written), the attempt is logged as `premature_final`, and the next segment is a forced
continuation. If the note part is empty, the attempt is dropped and not committed (append-only caching is unaffected).

The `Where we stand:` line in practice (first live run, `tests/fixtures/live_sse/relay_trace_explore_round_robin.json`):
only 1 of 10 notes ended with it. Two causes, two measures:

- **Reminder.** Whenever the previous committed note has no `Where we stand:` line, whether the model skipped it or
  the client-side hard cut removed it (the relay detects the cut by comparing the cleaned paragraph before and after
  the cut, and also counts a `length` / `cut` / `overflow` finish without a sentinel; `ExploreFlags.cut`), the record
  status says so: while the gate is closed "a 'Where we stand:' line at the end of this note" joins the missing items
  (once, even when the two-note window already lists it), once the gate is open "If you write a note, end it with a
  'Where we stand:' line." follows the permission to conclude. The metrics split the notes without the line into
  `wws.missing` (skipped) and `wws.cut` (lost to the cut).
- **Waiver.** The window rule stays, but a note the cut shortened must not hold the gate closed for good: after every
  committed note the unwaived checklist is judged, and once the line has been its *only* entry for two consecutive
  notes (so only after `min_segments` and with every required label in) the gate leaves it out (`wws.waived`,
  `wws.waived_at`; the tail still reminds). The waiver lapses when a note has the line again or another item goes
  missing; it is granted again after two further such notes. In the live run above the gate would have opened for
  segment 9 instead of waiting until segment 10, at the cost of nothing the model had not been told.

### 6.2 Continuation when a model concludes early (s1-style, without "Wait")

The next segment's tail block gets `CONTINUE_TEXT`, and the concluding model is excluded from `available` for that
segment:

> The moderator set aside the conclusion in the previous note because {why, e.g. "the record has no
> 'Alternative set aside:' item yet"}. Begin this note with the words '{cue}' and add what is missing. If the previous
> note's 'Where we stand:' is wrong, say so and why; you may contradict it.

Cues rotate in order: "Alternatively, " / "One thing not yet checked: " / "The strongest objection to the current
conclusion is " / "Perhaps the missing piece is " / "Before closing, the weakest assumption above is ". Evidence:
"Wait" is never the best keyword and forcing hurts distilled/RL models (Taveekitworachai & Pipatanakul 2026:
R1-Distill -0.99, Llama-3.1-8B 26.5 -> 19.7); "Alternatively"/"Hmm" matched "Wait" on MATH500/GPQA (Muennighoff
2025, Table 4); reflective filler is removable without loss (Wang C. 2025) and worth +1.3 GSM8K at best (Ringel
2025); the opening tokens of a segment decide whether the model elaborates (Yang V. 2026), which is why the cue is
an opening phrase, not a trailing instruction; a stronger model handed a partial trace inherits its errors unless
explicitly allowed to go further (Ballon 2026), hence "you may contradict it" and the model switch; follow-up
questions raise monitorability (Guan 2025); a re-examination pass after a provisional answer has a small positive
record (Tian 2025: +1.8 to +2.3).

Limits: at most **3** forced continuations per run; stop forcing when the word-3-gram Jaccard overlap of the last two
notes exceeds 0.5 (s1 loops; Markovian Thinker: the third forced pass turned a correct answer wrong; "answer
instability" in the ICLR-2026 blog). After the limit, the next `FINAL ANSWER:` is accepted even if the checklist is
incomplete, and the missing items are logged. Never force after the gate has passed (Caldarella 2026: stopping at the
first correct prefix +21%).

### 6.3 `segment_tokens`

**192 visible tokens, `hard_cut_chars 1500`, sentence-boundary trim (existing `trim_to_sentence`).** 128 tokens is
the "tight token limit" regime in which models drop the real drivers from their visible text (Zaman & Srivastava
2026: larger budgets raise hint verbalization to ~90%); the sentences that carry causal weight are planning and
uncertainty-management sentences, which a 128-token cut truncates first (Bogdan 2025); and every extra handoff costs
a model-tier-sized drift (Khraishi 2026: -8 to +13 pp), so fewer, longer segments are better than more, shorter ones.
192 fits 2-4 labelled items plus `Where we stand:`. The hard cut was first raised to 1000 chars so the mandatory last
line survives an overshoot; the first live run showed that this is too tight for German: 192 tokens run to 1100-1300
characters, every stream ended normally (`sentinel` / `stop`), the notes came out at 800-996 characters after the cut
and its sentence trim, and 9 of 10 had lost their last line. The cut is therefore **1500** characters (about 2x the
token target in English, 1.2-1.4x in German), and a note the cut shortened is flagged (`ExploreFlags.cut`) and
counted (`wws.cut`, distinct from `wws.missing` for a line the model skipped; section 6.1 says what the tail and the
gate do about it). The surrounding limits scale with it and still make sense: the relay stops reading a cancellable
stream once the visible text exceeds 1.5x = 2250 characters (`overflow`; the backend closes the response at the same
point plus the echo window, `cut`), a non-cancellable stream (Neon) is read on until 4x = 6000 characters (`runaway`),
which is about where Neon's `max_tokens_slack` of 1536 tokens ends anyway, and the display cap of the stream filter
is the cut itself. Routing granularity stays per segment. Not 256 for everyone: cost grows with visible tokens and
Neon's abort does not stop billing.

### 6.4 Structured-notes instruction (replaces `RELAY_SYSTEM_TEMPLATE` when `explore.enabled`)

System prompt (static for the run, cacheable; no forbidden words, no tags):

> Several analysts are jointly working through the problem in the user message and keep one shared set of working
> notes; the notes so far are included and are the only record that is kept. Contribute the next note in about {K}
> tokens, written so it can be appended as the next paragraph. A note is two to four short labelled items, each one
> or two sentences, chosen from: 'Claim:' (a definite statement); 'Evidence:' (a fact, number, observation or source
> that supports or weakens a claim; name where it comes from); 'Inference:' (what follows from which claims, as
> 'because A and B, therefore C'); 'Principle:' (a general rule from any field that applies here; name the field and
> apply it to the problem); 'Alternative set aside:' (a route or answer a careful analyst would propose, and the
> concrete grounds on which it is set aside); 'Open question:' (what is unsettled and what would settle it); 'Next:' (the single
> most useful thing the next analyst should do). Every note ends with 'Where we stand:' followed by one sentence
> giving the current best answer, a confidence from low to high, and the weakest assumption. Keep the labels exactly
> as given, in English, even when the notes are in another language; write everything else in the language of the
> notes (or of the problem if there are no notes yet). Do not restate earlier items; extend, correct or contradict
> them and say which. If an earlier note is wrong, start with 'Correction:' and fix it. Prefer the most recent facts
> you know and say when a fact may have changed. The last paragraph of the user message says whether the moderator
> accepts a conclusion; only then may you write '{final_marker}' followed by the answer instead of a note. Output
> only the note, then {sentinel}.

Tail block (varies per segment; the only non-cached block):

> END OF THE SHARED NOTES. Write note {k}. {record_status} {directive}

`record_status` when the gate is closed: "Items still missing from the record: {missing, e.g. 'three more main-line
notes', 'an Evidence: item that names its source', 'an Alternative set aside: item'}. The moderator does not accept
a conclusion yet; do not write 'FINAL ANSWER:'." When open: "The record is complete. If the notes settle the
problem, the moderator accepts a conclusion: write 'FINAL ANSWER:' followed by the answer instead of a note."
`directive` is empty, `LENS_TEXT`, `CHECK_TEXT`, `CONTINUE_TEXT` or the VS direction.

Answer system prompt: the existing `ANSWER_SYSTEM_TEMPLATE` plus "Where the notes mark a cross-field lens as kept,
use only its concrete consequence; never present a lens, kept or dropped, as evidence." and, with `answer_basis`,
"After the answer add 'Basis:' naming the claims and evidence it rests on, and 'Not resolved:' listing the open
questions that remain."

Evidence per slot: Claim / Evidence / Inference mirrors the four-step claim-verification template (Liang & Sonntag
2025) and "list every factor" verbosity (Meek 2025); `Evidence:` must name its source because source-attribution
prompts raise verbalization of real drivers while "you are monitored" framing does not (Gema 2026, FACE-Eval);
`Alternative set aside:` with a reason because written and evaluated alternatives are where gains appear (Yao 2023;
Chia 2023) and because frequent switching without follow-through correlates with wrong answers, so at most one per
note (Wang Y. 2025); `Principle:` is the one accuracy-positive abstraction mechanism that also pushes notes toward
other fields (Zheng 2024: +7 to +27 on knowledge-heavy QA), limited to one applied item per note because Claude gets
more distracted by irrelevant material as chains lengthen (Gema 2025); `Where we stand:` gives the next model a
compact textual state (Aghajohari 2025) and a confidence (Wang & Zhao 2024), and factors must appear in intermediate
notes, not only in the answer, to be monitorable (Wang H. 2026); "do not restate" counters verbosity compensation
(Zhang Y. 2024: verbose answers 27.6% worse on Qasper) and filler-driven continuation (Lotfi 2026).

Parsing: labels are matched with `^\s*(Claim|Evidence|Inference|Principle|Alternative set aside|Open question|Next|
Where we stand|Lens kept|Lens dropped|Correction|Lens \([^)]+\)):` per line (case-insensitive, tolerating `**`).
`strip_preamble`'s `_LABEL` regex does not match any of these labels, so they survive post-processing.

### 6.5 GPT-6.2 Sol: `verbosity` and `seed`

- Notes: `verbosity: medium` (default was `low`). OpenAI documents the knob as scaling "both the length and depth of
  the output while preserving correctness and reasoning quality", with the approach "mostly the same" and reasoning
  tokens governed separately (OpenAI docs). Under a 192-token target with client-side sentinel detection (Sol has no
  `stop`), `high` tends to overshoot, costs output tokens until the sentinel, and risks the trailing `Where we
  stand:` being cut; `medium` is the budget-safe step up. `high` is the A/B arm (the 1500-character cut leaves it
  room; `wws.cut` in the metrics says whether it still overshoots).
- Answer: `verbosity: low` (new `answer_extra_body`), where verbosity compensation hurts (Zhang Y. 2024).
- `seed = hash(explore.seed, idx) mod 2^31` on every Sol call (`supports_seed: true`): the only reproducibility
  handle on the three endpoints; used for template and cue A/B tests.
- Never `verbosity` to Fable (mapped to effort; already blocked). Neon: no knob; `reasoning.effort low` throughout.
- Do not expect `verbosity` to add steps; the steps come from the template.

## 7. Component 5: metrics the relay logs (in the `done` event and the trace)

| Metric | Definition | Shows |
|---|---|---|
| `topics_visited` | distinct argmax topics over think segments (from the stored `topic_posterior`), all and main-line only | breadth as the router sees it |
| `topic_entropy_bits` | Shannon entropy of the mean posterior across think segments, and of the argmax histogram | breadth; compare `explore` vs default on the same question |
| `topic_switches` | argmax changes between consecutive think segments | wandering |
| `model_switches`, `share_by_model` | existing `RelayTrace.switches()` / `share_by_model()` | whether wandering changed the writer (section 9: it mostly will not) |
| `lens.{offered, written, kept, dropped, ignored}`, `lens.fields`, `lens.tiers`, `lens.keep_rate` | from directives and `Lens (..):` / `Lens kept:` / `Lens dropped:` regexes | the lens mechanism works and is checked |
| `lens_distance` (optional, embedding classifier present) | mean cosine distance of lens notes to the question | how far the lenses go |
| `labels.counts`, `labels.coverage` | per-label counts; share of notes with >= 2 labelled items and with `Where we stand:` | verbalization in Meek's sense (factor listing) |
| `wws_lines`, `answer_flips` | the sequence of `Where we stand:` best answers; number of changes (string-normalised, or judge) | commitment trajectory; flips after the gate opened are the failure signal |
| `wws.{missing, cut, waived, waived_at}` | notes without the line: skipped by the model vs lost to the hard cut (disjoint); waivers the gate granted (section 6.1) and at which segment | whether the cut is too tight (`cut`), whether the template is followed (`missing`), whether the gate had to give in (`waived`) |
| `premature_finals`, `forced_continuations`, `first_final_index`, `gate_missing_at_final` | gate bookkeeping | how often models conclude early; whether forcing loops |
| `ngram_overlap_last2` | word-3-gram Jaccard of the last two notes | loop detection |
| `notes_chars`, `notes_tokens_est`, `think_segments`, `main_line_segments` | sizes | cost driver; not a quality metric (Chen W.-L. 2026) |
| `refusals`, `dropped`, `garble` (notes with < 60% dictionary words) | existing + one check | wording safety |
| `vs_menu.{calls, parsed, chosen_p}` | side-call bookkeeping | whether the menu was used |
| `usage_total.cost`, `latency` | existing | cost vs the 0.15 USD default |
| `preset`, `explore.seed` | configuration echo | reproducibility |

Protocol for the first comparison: same question, same `explore.seed`, run `default` and `explore` once each
(mock first); report the table above side by side. Accuracy must be measured separately on the existing bench against
a short-chain control and, when budget exists, a k=3 parallel-vote control (Ghosal 2025: parallel beats extended
thinking by up to 20% at equal budget; if the vote wins, the long chain buys variance, not quality).

## 8. Cost and latency vs the default

| | Default (measured 24.11.2026) | `explore`, round_robin (estimate) | `explore`, softmax + prior index (estimate) |
|---|---|---|---|
| Calls | 6 segments + 1 answer = 7 | 6 main-line + 1-2 lens + 1-2 check + 0-2 continuations + answer = 10-13 | same |
| Visible tokens / segment | 128 | 192 | 192 |
| Per-call cost factor vs default | 1.0 | 1.1-1.3 (hidden reasoning at `low` dominates output; +64 visible tokens; +~250 cached system tokens) | 1.1-1.3 |
| Writers | all three | all three | mostly Sol (0.4-0.8 ct) |
| **Cost per run** | **0.15 USD** | **0.25-0.40 USD** | **0.08-0.15 USD** |
| Wall time | 57 s | 100-140 s | 60-80 s |
| Optional VS menu (2 uses, Sol) | - | +0.01-0.02 USD, +6-10 s | same |
| Optional Neon temperature A/B (once) | - | 0.15-0.20 USD, deferred | - |

With under 1 USD on the key: run `cutoffmix ... --mock --config configs/explore.yaml` until the trace looks right
(free), then **one** live round-robin explore run (about 0.3 USD) with a fixed `explore.seed`, then stop. Budget
knobs if needed: `lens.max 1`, `min_segments 5`, `max_forced_continuations 1`.

## 9. What this will and will not achieve

Will:
- Notes with labelled claims, sourced evidence, set-aside alternatives, open questions and a per-note "where we
  stand" line: more legible and more monitorable visible text (Guan 2025; Zaman & Srivastava 2026; Meek 2025), and a
  per-step commitment trace (Ballon 2026).
- One or two genuine cross-field excursions per run, each with a function-based mapping, a testable consequence, an
  explicit break point and a check by another model; fields drawn from a client-side list, so no two runs reuse the
  same two clusters (Shen Y. 2026) and runs are reproducible via `explore.seed`.
- Longer chains (10-13 segments instead of 6) without bare "Wait" loops, at 1.5-2.5x the default cost.
- Diversity across runs when `explore.seed` varies, which is where diversity actually pays off (Wu 2025; Si 2024).

Will not:
- **Change the sampling.** No temperature or top_p takes effect on any of the three models; the preset does not try.
- **Change which model writes, with the current index.** The prior index (knowledge-cutoff bench data to June 2026)
  puts GPT-6.2 Sol ahead on every topic; with `softmax` the others only get turns through sampling noise, and lens
  notes are excluded from the routing window by design. Wandering topics will therefore rarely re-route until a
  topic-stratified index from the July-November 2026 Wikidata events (`data/events/`, 961 events; costs money) exists.
  Until then, `round_robin` is the way to see all three writers; `topic_switches` and `model_switches` will be
  uncorrelated and the log will show it.
- **Improve answers.** For capable models, pushing visible length past the natural stop is neutral at best and
  negative beyond ~2-4k tokens or ~3 continuations (Wu 2026; Ghosal 2025; Gema 2025; Caldarella 2026; Aghajohari
  2025); positive length effects are from weaker models (Jin 2024: GPT-3.5 +12-13, GPT-4 only "tolerant") or
  fine-tuned systems (Muennighoff 2025); CoT helps mainly math/symbolic tasks (Sprague 2025), and Chain of Draft keeps
  ~96% of accuracy with 7.6% of the tokens (Xu 2025). Expect accuracy-neutral on synthesis questions, mildly negative
  if lenses distract Fable (Gema 2025) or a lens is kept wrongly (Shen 2026: 67% reasonable).
- **Make the notes a faithful record.** Hidden drivers are mentioned only in the discarded channel in 55.4% of cases
  (Young 2026); frontier models are the least faithful (Lanham 2023; Chen Y. 2025); a frontier model asked to narrate
  produces a polished narrative, not a trace (Borchers 2026); part of CoT's benefit is compute, not content (Pfau
  2024). Do not present the notes as an explanation of the answer (Barez 2026), and do not grade or pressure them
  hard, or they become performative (Baker 2025).
- **Guarantee deep analogies.** Prompt-only methods produced surface associations on CLoT's humour benchmark (Zhong
  2024); the check and the keep-rate metric exist to catch that. Models may also ignore labels or the `Lens (..):`
  prefix; the relay commits such notes and logs them rather than retrying (retrying a reworded prompt is forbidden
  by the refusal policy).
- **Measure field breadth with the news classifier.** A thermodynamics lens maps to `science_research` or to no
  topic at all, so `topic_entropy` under-counts field diversity; `lens.fields` and the optional embedding distance are
  the real breadth metrics.
- **Be validated.** No paper studies visible shared notes with mandatory discarded hidden reasoning and model handoffs
  every ~200 tokens; p_lens, tiers, cooldowns, the 6-note floor and the overlap threshold are heuristics to tune on
  the bench, and nothing here has been tested live.

## 10. Implementation map

1. `config.py`: `ExploreConfig` (fields of section 2) and `AppConfig.explore`; `ModelSpec.answer_extra_body`,
   `ModelSpec.supports_seed` in `types.py`; `ChatRequest.temperature` default `None`; Neon
   `supports_temperature: false` in `default.yaml`; `configs/explore.yaml`.
2. `prompts.py`: `RELAY_SYSTEM_TEMPLATE_EXPLORE`, `LENS_TEXT`, `CHECK_TEXT`, `CONTINUE_TEXT`, `VS_MENU_TEXT`,
   `relay_tail(k, record_status, directive)`, `LABEL_RE`, `lens_prefix_re`, `parse_vs_menu`; `FIELDS` and the topic
   groups in a new `cutoffmix/lenses.py`.
3. `types.py` `ContinuationRequest`: `tail: str | None` (explore tail text) and `system_override: str | None`;
   `openrouter.py build_continuation_body` uses them, applies `answer_extra_body` in phase `answer`, and sets
   `body["seed"]` when `supports_seed`.
4. `relay.py` `_RunState`: `segment_flags` (main/lens/check/continue), `pending_check`, `lens_writer`, `cooldown`,
   `lenses_used`, `used_fields`, `forced`, `premature_finals`, gate helpers (`record_missing()`, `gate_open()`,
   `stagnating()`); `_run` handles `premature_final` (commit note part, schedule continuation); `_think` passes
   `exclude` to `_produce`/`router.decide`; `router.window` receives main-line + check notes only; `done_data()` adds
   the section 7 metrics.
5. `cli.py` / `server.py`: `--config configs/explore.yaml` already works via `load_config`; add `--seed` to set
   `explore.seed`; UI badges for lens/check/continue segments (optional).
6. Tests (offline, mock backend): lens scheduling is deterministic for a seed and never in the first two or after the
   gate; check follows every lens and excludes the lens writer; `premature_final` strips the answer and forces at most
   3 continuations; loop detection stops forcing; `parse_vs_menu` on well-formed, malformed and short menus; label
   regex vs `strip_preamble`; Neon body has no `temperature`; Sol body has `seed` and phase-specific `verbosity`;
   Fable body never has `verbosity`.

## 11. Citations

All entries below were opened on their arXiv / ACL / OpenReview / official page by the researchers whose notes
this document synthesises (status `verified: true` in those notes). Venue given where the page showed it.

Temperature and sampling
- Google, "What's new in Gemini 4 Neon" (Gemini API docs, fetched 2026-11-24): temperature/top_p/top_k ignored in range, 400 out of range; GA 2026-11-18; thinking levels low/medium/high/xhigh.
- Peeperkorn, Kouwenhoven, Brown, Jordanous, "Is Temperature the Creativity Parameter of Large Language Models?", ICCC 2024, arXiv 2405.00492.
- Renze, Guven, "The Effect of Sampling Temperature on Problem Solving in Large Language Models", Findings of EMNLP 2024, arXiv 2402.05201.
- Nguyen et al., "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs", ICLR 2025, arXiv 2407.01082.
- Jiang et al., "Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond)", NeurIPS 2025 D&B, arXiv 2510.22954.
- Zhang J., Yu, Chong et al., "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity", arXiv 2510.01171 (ICML 2026 per PDF header).
- Ahmed, Berdica, Elliott, Horak, Foerster, "Intent Factored Generation: Unleashing the Diversity in Your Language Model", arXiv 2506.09659.
- Deng, Brucks, Toubia, "A Cognitive Framework for Idea Diversity in Large Language Models", arXiv 2602.20408.
- Terwiesch, Meincke, Girotra, Mollick, Nave, Ulrich, "AI and Its Impact on Creativity and Diversity: An Empirical Study of LLM-Generated Product Ideas", arXiv 2607.27553.
- Li, Sleem, Gentile, Nichil, State, "Exploring the Impact of Temperature on Large Language Models: Hot or Cold?", arXiv 2506.07295 (IJCNN 2025 workshop).
- Du, Yang, Welleck, "Optimizing Temperature for Language Models with Multi-Sample Inference", ICML 2025, arXiv 2502.05234.
- Wu Yuheng, Mirhoseini, Tambe, "On the Role of Temperature Sampling in Test-Time Scaling", arXiv 2510.02611.
- Audran-Reiss et al., "What Does It Take to Be a Good AI Research Agent? Studying the Role of Ideation Diversity", arXiv 2511.15593.
- Chen H., Ding N., "Probing the Creativity of Large Language Models: Can models produce divergent semantic association?", Findings of EMNLP 2023, arXiv 2310.11158.
- Holtzman, Buys, Du, Forbes, Choi, "The Curious Case of Neural Text Degeneration", ICLR 2020, arXiv 1904.09751.
- West, Potts, "Base Models Beat Aligned Models at Randomness and Creativity", arXiv 2505.00047.
- Zhang Yiming, Schwarzschild, Carlini, Kolter, Ippolito, "Forcing Diffuse Distributions out of Language Models", arXiv 2404.10859.
- Misaki, Akiba, "String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation", ICLR 2026, arXiv 2510.21150.

Diversity, analogy and science ideation
- Shen A., Druckmann, Zou, "Unlocking LLM Creativity in Science through Analogical Reasoning", NeurIPS 2026 poster, arXiv 2605.11258.
- Shen Y., de Sousa, Barcelos, Guo, Zhu, "On the Diversity of Analogy Making in Large Language Models", arXiv 2608.03233.
- Mehrotra, Parab, Gulwani, "Enhancing Creativity in Large Language Models through Associative Thinking Strategies", arXiv 2405.06715.
- Bellemare-Pepin, Lespinasse et al., "Divergent Creativity in Humans and Large Language Models", arXiv 2405.13012 (Scientific Reports 2026 per search).
- Meincke, Mollick, Terwiesch, "Prompting Diverse Ideas: Increasing AI Idea Variance", arXiv 2402.01727.
- Si, Yang, Hashimoto, "Can LLMs Generate Novel Research Ideas? A Large-Scale Human Study with 100+ NLP Researchers", ICLR 2025, arXiv 2409.04109.
- Naik, Chandrasekaran, Yuksekgonul, Palangi, Nushi, "Diversity of Thought Improves Reasoning Abilities of LLMs", arXiv 2310.07088.
- Yasunaga et al., "Large Language Models as Analogical Reasoners", ICLR 2024, arXiv 2310.01714.
- Qin et al., "Relevant or Random: Can LLMs Truly Perform Analogical Reasoning?", arXiv 2404.12728.
- Lewis, Mitchell, "Using Counterfactual Tasks to Evaluate the Generality of Analogical Reasoning in Large Language Models", arXiv 2402.08955.
- Ding, Srinivasan, MacNeil, Chan, "Fluid Transformers and Creative Analogies", arXiv 2302.12832.
- Chan, Dow, Schunn, "Do the best design ideas (really) come from conceptually distant sources of inspiration?", Design Studies 2015, doi 10.1016/j.destud.2014.08.001.
- Chan, Schunn, "The impact of analogies on creative concept generation: lessons from an in vivo study in engineering design", Cognitive Science (2014 online).
- Kwon, Goucher-Lambert, "Examining the Boundary Between Near and Far Design Stimuli", Proc. Design Society ICED23, 3:1725-1734.
- Zhong et al., "Let's Think Outside the Box: Exploring Leap-of-Thought in Large Language Models with Creative Humor Generation", CVPR 2024, arXiv 2312.02439.
- Huang et al., "Large Language Models Cannot Self-Correct Reasoning Yet", ICLR 2024, arXiv 2310.01798.
- Dhuliawala et al., "Chain-of-Verification Reduces Hallucination in Large Language Models", Findings of ACL 2024, arXiv 2309.11495.
- Chen J. C.-Y., Saha, Bansal, "ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs", ACL 2024, arXiv 2309.13007.
- Liang T. et al., "Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate", EMNLP 2024, arXiv 2305.19118.
- Choi, Zhu, Li, "Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?", NeurIPS 2025, arXiv 2508.17536.

Verbalization, length and continuation
- Muennighoff et al., "s1: Simple test-time scaling", arXiv 2501.19393.
- Taveekitworachai, Pipatanakul, "Wait, Do We Need to Wait? Revisiting Budget Forcing for Sequential Test-Time Scaling", ICLR 2026 Blog Track.
- Aghajohari et al., "The Markovian Thinker: Architecture-Agnostic Linear Scaling of Reasoning", arXiv 2510.06557.
- Wang C. et al., "Wait, We Don't Need to 'Wait'! Removing Thinking Tokens Improves Reasoning Efficiency", arXiv 2506.08343.
- Ringel, Tolochinsky, Romano, "Learning a Continue-Thinking Token for Enhanced Test-Time Scaling", arXiv 2506.11274.
- Wu Yuyang et al., "When More is Less: Understanding Chain-of-Thought Length in LLMs", ICLR 2026, arXiv 2502.07266.
- Ghosal et al., "Does Thinking More always Help? Mirage of Test-Time Scaling in Reasoning Models", NeurIPS 2025, arXiv 2506.04210.
- Gema et al., "Inverse Scaling in Test-Time Compute", TMLR 2025, arXiv 2507.14417.
- Jin et al., "The Impact of Reasoning Step Length on Large Language Models", Findings of ACL 2024, arXiv 2401.04925.
- Sprague et al., "To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning", ICLR 2025, arXiv 2409.12183.
- Xu, Xie, Zhao, He, "Chain of Draft: Thinking Faster by Writing Less", arXiv 2502.18600.
- Caldarella et al., "Thinking Past the Answer: Evaluating Harmful Overthinking in Large Reasoning Models", arXiv 2606.02835.
- Lotfi, Kirichenko, Li, Liu, "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", arXiv 2606.00206.
- Chen W.-L. et al., "Think Deep, Not Just Long: Measuring LLM Reasoning Effort via Deep-Thinking Tokens", ICML 2026, arXiv 2602.13517.
- Wang Y. et al., "Thoughts Are All Over the Place: On the Underthinking of o1-Like LLMs", arXiv 2501.18585.
- Zhang Yusen, Das, Zhang R., "Verbosity != Veracity: Demystify Verbosity Compensation Behavior of Large Language Models", arXiv 2411.07858.
- Pfau, Merrill, Bowman, "Let's Think Dot by Dot: Hidden Computation in Transformer Language Models", arXiv 2404.15758.
- Tian et al., "Think Twice: Enhancing LLM Reasoning by Scaling Multi-round Test-time Thinking", arXiv 2503.19855.
- Yang V. et al., "Mid-Think: Training-Free Intermediate-Budget Reasoning via Token-Level Triggers", Findings of ACL 2026.
- Aggarwal, Welleck, "L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning", COLM 2025, arXiv 2503.04697.
- Ballon, Verbeken, Ginis, Algaba, "Probing the Trajectories of Reasoning Traces in Large Language Models", NeurIPS 2026 E&D, arXiv 2601.23163.
- Khraishi, Zafar, Myles, Cowan, "Evaluating Performance Drift from Model Switching in Multi-Turn LLM Systems", arXiv 2603.03111.

Structure and transparency
- Wang L. et al., "Plan-and-Solve Prompting", ACL 2023 (2023.acl-long.147).
- Yao et al., "Tree of Thoughts: Deliberate Problem Solving with Large Language Models", NeurIPS 2023, arXiv 2305.10601.
- Chia, Chen, Luu, Poria, Bing, "Contrastive Chain-of-Thought Prompting", arXiv 2311.09277.
- Zheng H. S. et al., "Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models", ICLR 2024, arXiv 2310.06117.
- Wang Y., Zhao Y., "Metacognitive Prompting Improves Understanding in Large Language Models", NAACL 2024, arXiv 2308.05342.
- Liang S., Sonntag, "Advancing Biomedical Claim Verification by Using Large Language Models with Better Structured Prompting Strategies", BioNLP 2025 (2025.bionlp-1.14).
- Bogdan, Macar, Nanda, Conmy, "Thought Anchors: Which LLM Reasoning Steps Matter?", arXiv 2506.19143.
- Guan et al. (OpenAI), "Monitoring Monitorability", arXiv 2512.18311.
- Zaman, Srivastava, "Is Chain-of-Thought Really Not Explainability? Chain-of-Thought Can Be Faithful without Hint Verbalization", ACL 2026, arXiv 2512.23032.
- Young, "Why Models Know But Don't Say: Chain-of-Thought Faithfulness Divergence Between Thinking Tokens and Answers in Open-Weight Reasoning Models", arXiv 2603.26410.
- Gema, Rajani, Saxena, Kwan, Minervini, "Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered (FACE-Eval)", arXiv 2608.29464.
- Meek, Sprejer, Arcuschin, Brockmeier, Basart, "Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity", arXiv 2510.27378.
- Lanham et al., "Measuring Faithfulness in Chain-of-Thought Reasoning", arXiv 2307.13702.
- Chen Y. et al. (Anthropic), "Reasoning Models Don't Always Say What They Think", arXiv 2505.05410.
- Wang H. et al., "MonitorBench: A Comprehensive Benchmark for Chain-of-Thought Monitorability in Large Language Models", COLM 2026, arXiv 2603.28590.
- Baker et al. (OpenAI), "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation", arXiv 2503.11926.
- Barez et al., "Chain-of-Thought Is Not Sufficient for Explainability", NeurIPS 2026 poster 140085.
- Borchers, Vie, Azevedo, "Large Language Models as Students Who Think Aloud: Overly Coherent, Verbose, and Confident", arXiv 2602.01015.
- OpenAI, "GPT-5 New Params and Tools" (cookbook) and "Using GPT-5.2" (API guide): `verbosity` low/medium/high; reasoning tokens governed by `reasoning.effort`.

Not relied on (UNVERIFIED in the source notes): Fu et al. 2013 (JMD, near/far stimuli; PDF 404); Fox, Ericsson &
Best 2011 (Psychological Bulletin; effect sizes only from secondary sources); the Vertex "Gemini 3 prompting guide"
warning about T < 1.0 (seen only in a search summary); the Vertex path's handling of Neon temperature (no page found;
the A/B test in section 3 settles it).
