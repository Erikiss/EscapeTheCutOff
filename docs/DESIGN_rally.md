# Rally preset: models pass the record back and forth to lift each other's recency

Status: design contract (2026-11-26). Everything below is implemented offline against the simulated backend; the
live behaviour of the three models under these prompts is **not** verified (no key was available when it was built).

## 1. Idea

The default relay lets the router pick one writer per note, and the writer is usually the model with the freshest
knowledge of the topic. The rally does the opposite on purpose: two or three models with **different training data**
hand one record to each other in a fixed ring (default Fable, Neon, Fable, Neon, ...).

1. Model A writes what it knows, every item with its date, plus the newest date it can place (`Newest I know:`).
2. Model B receives A's note and a directive: confirm what you can confirm, correct what is outdated or wrong
   (with what you know instead, dated), add what you know that is *newer* than A's newest date, say plainly when you
   know nothing newer. B's note ends with B's own newest date.
3. A receives B's note: B's newer items are a cue (and a check) for A, who may now recall related newer facts it
   would not have produced unprompted. And so on.
4. The exchange ends when one full revolution of the ring adds nothing (every model of the last `quiet_turns` notes
   wrote `Nothing newer:` and no `Update:` or `Newer:`), or at `max_turns`. One answer call writes the answer from
   the record: statements that a second model confirmed are stated plainly, single-source statements are marked.

What is **not** claimed: that the exchange pushes the effective cutoff beyond the best single model. That is a
hypothesis. The metrics in section 6 are self-reports (the project's rule is that self-reports do not count), so
they only show what the exchange did; whether it helps is measured with the benchmark (section 9, not built yet).
Known risk: models can reinforce each other's mistakes or invent something newer to please. The prompts forbid
inventing and treat `Nothing newer:` as a good answer; the answer marks single-source statements.

## 2. Config (`config.py`)

```python
@dataclass
class RallyConfig:
    enabled: bool = False
    ring: tuple[str, ...] = ()       # model keys in hand-over order; () = all configured models in config order
    min_turns: int | None = None     # None -> 2 * len(ring): no stop before this many think notes
    max_turns: int = 8               # hard stop of the exchange (think notes), then the answer phase
    quiet_turns: int | None = None   # None -> len(ring): consecutive quiet notes that end the exchange
    answer_basis: bool = True        # the answer ends with 'Basis:' and 'Not resolved:'
    answer_from_ring: bool = True    # the answer model is chosen by the router among the ring's models only
    partner_failures: int = 2        # consecutive failed hand-overs (errors, not refusals) that take a ring model out
    light_schema_models: tuple[str, ...] = ()   # models that get the light notes (section 4.1)
```

`AppConfig.rally: RallyConfig`; `config_from_dict` builds it like `explore` (key `rally`, unknown keys rejected,
`AppConfig.to_dict` includes it with `ring` as a list). Validation (`ConfigError`): ring entries are non-empty,
unique and known model keys; `max_turns >= 2`; `min_turns` (if set) in `[1, max_turns]`; `quiet_turns` (if set) >= 1;
`partner_failures` >= 1; `light_schema_models` entries are known model keys; with `enabled`: the effective ring has
at least two models and `explore.enabled` is false (the two presets are
mutually exclusive). `AppConfig.rally_ring() -> list[str]` is the effective ring. `relay.strategy` `speculative` is
ignored while the rally is on (drafts would pay for models the ring does not use).

Preset `cutoffmix/data/presets/rally.yaml` (merged like `explore.yaml`; `--preset rally`, model id `cutoffmix-rally`):

```yaml
relay: {segment_tokens: 160, hard_cut_chars: 1500, max_segments: 12, answer_mode: routed, disable_after_refusals: 3}
rally: {enabled: true, ring: [claude-fable-5.2, gemini-4-neon], max_turns: 8, partner_failures: 2,
        light_schema_models: [claude-fable-5.2]}
```

## 3. Hand-over (`relay.py`)

With the rally on, a think segment's candidates are the ring's models that are available (not disabled, not excluded
by a refusal or error for this segment), **in ring order**. The router still decides (`router.decide(window, current=
st.current, available=ring_avail)`, so `topic_posterior`, `scores` and `r` are reported as always), then the relay
replaces the choice: the model after `st.current` in ring order (the ring's first model for the first note), with
`decision = "forced"` and `probs = {model: 1.0}`. If the next model is unavailable the following one in ring order is
taken, which can be the model that wrote the previous note: that note is a **self-reply** (`self_reply: true`). A
self-reply gets its own directive (4: "the other analyst could not take its turn"), is never quiet and cannot close
the exchange; the answer prompt is told which notes they are. A ring model that fails a hand-over with an error
(not a refusal) is counted: a non-retryable error takes it out for the run at once, a retryable one after
`partner_failures` consecutive failures (a note by that model resets the count); refusals use
`relay.disable_after_refusals`. Every skipped hand-over is listed in the metrics. The exchange ends with
`stop_reason = "no_partner"` (and the answer phase runs from the record) when fewer than two ring models can still
write, when two self-replies follow each other, or when no ring model can take a turn although notes exist. A failure
before the first note stays an error. The answer phase restricts the candidates to the ring when `answer_from_ring`
(if none of them is left, all available models), and the router's normal policy picks among them.

## 4. Notes and prompts (`prompts.py`)

Labels (English, case-insensitive, optional markdown bold, at the start of a line or after a bullet): `Known:`,
`Confirm:`, `Update:`, `Newer:`, `Unverified:`, `Nothing newer:` and the closing line `Newest I know:`. The prompt asks
for the label exactly as given, directly followed by the colon, dates after it; the parser also accepts a
parenthesised or numbered qualifier (`Update (March 2026):`, `Update 1:`), `Updates:`, `Correction:` (an update),
`Nothing newer than <date>` with or without the colon and a heading prefix.

- **Opening note** (turn 1): 3 to 6 `Known:` items, each with the date or period it refers to; the weakest item
  marked; `Newest I know: <month year>` = the latest month relevant to the problem that the writer really knows
  something about (not today's date).
- **Reply note** (turn >= 2): 2 to 5 labelled lines chosen from `Confirm:`, `Update:`, `Newer:`, `Unverified:`,
  `Nothing newer:`; ends with `Newest I know:`. `Update:` = an earlier claim is outdated or wrong, with what the
  writer knows instead and its date. `Newer:` = a development after the previous writer's newest date, dated.
  `Unverified:` = a claim the writer cannot confirm and doubts. `Nothing newer:` = the writer knows nothing newer
  than the previous writer's newest date that the notes lack, followed by what it checked. Never invent a date or
  event to have something to add.

System prompt (static for the run; the date sentence as in the default relay):

```
RELAY_SYSTEM_TEMPLATE_RALLY = (
    "Several analysts, each trained on different material, hand one set of working notes to each other. The "
    "problem and the notes so far are in the user message. Every analyst knows some recent facts the others do "
    "not, and none knows everything: your job is to add what you know that the notes lack or get wrong, and to "
    "say honestly where you know nothing more. Write the next note in about {K} tokens, as short labelled lines "
    "(labels in English, everything else in the language of the problem): 'Known:' (first note only), 'Confirm:', "
    "'Update:', 'Newer:', 'Unverified:', 'Nothing newer:'. Every claim carries the month and year it refers to. "
    "Never invent a date, number or event to have something to add: 'Nothing newer:' is a valuable answer. Do not "
    "agree to be polite and do not repeat earlier lines. End every note with 'Newest I know:' and the latest "
    "month and year relevant to the problem that you genuinely know about (not today's date). Do not write a final "
    "answer; the moderator ends the exchange. Output only the note, then {sentinel}."
)
```

Tail (`RELAY_TAIL_RALLY = "END OF THE SHARED NOTES. Write note {k}. {directive}"`):

- `RALLY_OPEN_TEXT`: "This is the first note. List what you know about the problem as three to six 'Known:' items,
  each with the date or period it refers to, and mark the one you are least sure of. Then give 'Newest I know:'."
- `RALLY_REPLY_TEXT`: "The previous note was written by {writer}, an analyst whose knowledge differs from yours;
  their newest knowledge on this problem is {newest}. Go through its dated claims. Use 'Confirm:' for what you can
  confirm, 'Update:' for what is outdated or wrong (say what you know instead and when), 'Newer:' for developments
  after {newest} that you know, including ones the previous note reminded you of, 'Unverified:' for claims you
  doubt but cannot refute, and 'Nothing newer:' if you know nothing newer than {newest} that the notes lack. Then
  'Newest I know:'." (`Newer:` is for developments "that you know and the notes do not yet contain": the bar stays
  the previous writer's own date, not the running frontier, so one inflated date cannot hide real facts.)
  `{writer}` is the display name of the previous model (`ModelSpec.label()`); `{newest}` is the previous note's
  `Newest I know:` month as `Month YYYY`, `not stated`, or, for an approximate date (a year, a quarter, a season), the
  writer's own words in quotes.
- `RALLY_SELF_TEXT` (the writer is the author of the previous note): "The other analyst could not take its turn, so
  the previous note is your own. Do not confirm your own claims. Add with 'Update:' or 'Newer:' only what the notes do
  not yet contain, with its date; otherwise write 'Nothing newer:' and what you checked. Then 'Newest I know:'."
- Answer system prompt (`answer_system_prompt_rally(today, answer_basis=True)`): the default answer prompt plus
  "The notes are an exchange between analysts with different knowledge. State plainly what a second analyst
  confirmed or corrected; mark a statement only one analyst made with '(single source)'; where they disagreed say so
  and give the better supported side; name the newest date the record supports." and, with `answer_basis`, the
  existing `ANSWER_BASIS_SENTENCE`. Notes that are self-replies are named by their 1-based position ("do not count
  its 'Confirm:' lines as confirmation by a second analyst").

### 4.1 Light notes (`rally.light_schema_models`)

Anthropic's input filter blocked schema-heavy prompts for Fable 5.2 in the explore preset (DESIGN.md 7.8: a threshold
on the amount of schema; rewording did not help). The rally prompt has seven labels in the system prompt and up to
six in the reply tail, which is nearer to the blocked shapes than to the verified ones, and it has not been tried
live. Models in `light_schema_models` (the preset lists Fable) therefore get the default relay system prompt and a
prose tail: the one-sentence ban on a conclusion that passed in 7.8, plus a directive with only `Correction:` and
`Newest I know: <Month YYYY>` quoted; the other analyst is not named. Their replies are read without labels: a reply
without any label counts as quiet when it declares a readable newest month that is not above the frontier so far.
After a live probe (4 to 6 calls) shows that the full prompt passes, empty the list.

### 4.2 Echo guard

The relay drops a note that only repeats earlier text. A quiet rally note (labels `Confirm:`, `Nothing newer:` or
`Unverified:` only, plus its `Newest I know:` line) is exempt from the whole-note repeat test and from the overlap trim,
because models write it in nearly the same words twice in a row; repeats of earlier notes at the start of a note are
still removed, and a loud repeat (an `Update:`, `Newer:` or `Known:` copy) is still dropped.

A conclusion (`FINAL ANSWER:` text) written inside a think note is never accepted while the rally runs: the note
part stays, the answer part is discarded (`segment_end.data["premature_final"] = True`); a note that is empty after
that is dropped as `empty` and the segment is re-routed (exactly the explore preset's handling, without gate or
forced continuation).

## 5. Parsing and state (`cutoffmix/rally.py`)

- `parse_note(text) -> dict`: `labels` (counts: `known`, `confirm`, `update`, `newer`, `unverified`, `nothing_newer`),
  `newest_raw` (text after `Newest I know:` on its line, else None) and the date tokens of that line.
- `parse_month(text, today) -> (YYYY-MM | None, approx: bool, future: bool)`: the **latest** date token of the text:
  `2026-03`, `03/2026`, `March 2026`, `Mar 2026`, `März 2026` (English and German month names, full and abbreviated),
  `Q2 2026` (first month of the quarter, `approx`), a bare year (January, `approx`). A date after `today`'s month is
  `future`: it is reported but never enters the frontier.
- `RallyState(cfg, ring, today)`: `notes` (one dict per committed think note: `index`, `model`, `kind` (`"open"` for
  the first note, else `"reply"`), `labels`, `newest` (YYYY-MM or None), `newest_raw`, `approx`, `future`, `quiet`),
  `quiet` = reply note, not a self-reply, with `update == 0 and newer == 0`, whose declared newest month is not above
  the frontier before it (a rising date without a `Newer:` line is not quiet), that is not unreadable (cut and no
  closing line), and that says something: a `Confirm:`, `Nothing newer:` or `Unverified:` line, or no label at all with
  a readable newest month (light notes). The opening note is never quiet. `turns`, `previous()`, `frontier()`,
  `next_model(available)`, `hand_over_failed()`, `hand_over_ok()`, `note_skipped()`,
  `should_stop(partners_left) -> "converged" | "max_turns" | "no_partner" | None`, `metrics()`.
- A note is parsed from the writer's whole paragraph, before the hard cut (`cut` says that the committed text is
  shorter); the next writer is shown the trimmed text.
- Stop rule, checked after every committed think note, `n` = notes so far: `n >= max_turns` -> `max_turns`;
  `n >= min_turns` (default `2 * len(ring)`) and the last `quiet_turns` notes (default `len(ring)`) are all quiet and
  written by at least `min(quiet_turns, len(ring))` different models -> `converged`; fewer than two ring models able
  to write, or two self-replies in a row -> `no_partner`. If the relay ends the note phase otherwise, `stop_reason` is
  `max_segments` (relay cap), `empty` (every model had nothing to add) or `error`.

## 6. Metrics (`done.data["rally"]`, also `Segment.rally` and `segment_end.data["rally"]` per note)

`RallyTrace(RelayTrace)` with `rally: dict | None`, `to_dict()` adds `"rally"` (like `ExploreTrace`).

```
ring [keys], preset (AppConfig.preset), turns, stop_reason, quiet_streak (at the end), min_turns, max_turns, quiet_turns
notes [{index, model, kind, labels{known, confirm, update, newer, unverified, nothing_newer}, newest, newest_raw,
        approx, future, mentioned, quiet, self_reply, cut, light}]
frontier {self_reported: true, by_turn [YYYY-MM|null] (running maximum of the non-future declared dates),
          by_turn_approx [bool], first_declared {model: YYYY-MM|null} (each model's first note), final, gain_months,
          gain_months_approx}
        gain_months = months between `final` and the latest of the first declarations; null unless every model that
        wrote has a dated first note and both endpoints are exact dates (gain_months_approx then holds the number)
self_replies (count), skipped_hand_overs [{index, model, reason}]
adds_by_model {model: {update, newer, confirm, unverified, nothing_newer}}  (label counts)
unlabelled_replies (reply notes with none of the five reply labels)
hand_overs (= turns - 1)
```

`Segment.rally` / `segment_end.data["rally"]` = the note's dict from `notes`. Non-rally runs carry none of this and
are byte-identical to before (the existing default-relay expectation test must stay green).

## 7. Surface

- Mock backend: when the request's `metadata["kind"]` is `"open"` or `"reply"` (set by the rally request builder)
  the simulated model writes rally notes. Each simulated model has a "newest I know" month derived from its truth
  tau for the topic (tau as a month index via `month_label`), so the exchange has something to lift: early replies
  carry `Update:` / `Newer:` lines when the writer's newest month is later than the previous note's declared one,
  and `Nothing newer:` otherwise, so the stop rule converges. Deterministic per seed.
- CLI: `--preset rally`; `--ring KEY,KEY[,KEY]` (run and serve) overrides `rally.ring` (and implies the preset's
  config when `--preset rally` is also given; with `--ring` alone the rally is switched on). Segment headers show
  the note kind and counts (`open`, `reply · +2 update · +1 newer · newest 2026-03`); a summary after the run shows
  turns, stop reason, the frontier curve and `adds_by_model`.
- Server: `cutoffmix-rally` model id (automatic with the packaged preset), `"preset": "rally"` and an optional
  `"ring": [keys]` on `POST /api/relay` (validated against the model keys). `/v1/chat/completions` returns the
  metrics under `cutoffmix.rally`.
- UI: the segment badge shows `open` / `reply`, the label counts and the declared newest month; a strip under the
  run shows the frontier curve (declared newest per turn, per model colour) and the stop reason. The text says
  "self-reported".

## 8. Tests

`tests/test_rally_core.py` (config, parsing, state, prompts), `tests/test_rally_relay.py` (scripted backend: strict
ring order, convergence, `max_turns`, conclusion stripping, refusal skip, answer restricted to the ring, default
relay untouched), `tests/test_rally_surface.py` (mock backend end to end, CLI, server, UI markup).

## 9. Measuring whether it helps (not built)

Run `bench` probes (current-events questions after the models' cutoffs, with known answers) through three arms:
each model alone, the router relay, the rally. Compare accuracy by event month. If the rally's accuracy curve
stays above the best single model's for events after that model's cutoff, the exchange lifts recency. This costs
real money (about 8 to 20 cents per question for the rally) and needs approval and an OpenRouter key.
