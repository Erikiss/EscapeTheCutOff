# Key, not value: does another model's cue unlock a model's own memory?

Status: design contract and pre-registered predictions, **version 2** (amended on 2026-11-27, after an adversarial
review of version 1 and before any real call; section 10 lists what changed and why). Built and tested offline; the
pilot with the real models has **not** run (it needs an OpenRouter key and the owner's approval, about 2 to 3 dollars).

## 1. Source and idea

Hazy Research, "Plato's Key" (Cheng, Liu, Yau, Ré; blog post, 25 November 2026, no paper or code linked). For factual
recall an LLM builds a *key* at the subject token in its early-to-middle layers, and later layers use it to retrieve
the *value* from the weights. Taking one model's key (a hidden state), mapping it with a rotation and one scale fitted
in closed form, and swapping it into another model makes the recipient produce the donor's answer for a median 41 % of
facts (self-swap reference 55 %, controls 0.6 to 2.2 %), across 12 open base models of 8 families. Of the facts the
donor recalls and the recipient misses, 56 % are "dark memories" (stored but not retrieved), and the donor's key
recovers 18 % of those without any weight change (controls at most 1.1 %).

The method needs hidden states. Fable 5.2, GPT-6.2 Sol and Gemini 4 Neon return only text, so it cannot be applied
to them. What carries over is the question it answers, asked at the text level:

> When model B misses a fact that model A recalls, does a **cue** from A that names the subject but **withholds the
> answer** make B retrieve the fact from its own knowledge, or does B only improve when it sees A's answer?

| Plato's Key (white-box) | Here (API, text only) |
|---|---|
| key: donor hidden state at the subject token, rotated | the donor's own answer text with the answer masked (`key`) |
| value: what the recipient retrieves from its weights | the name the recipient writes |
| self-swap reference | the recipient asked alone again (`alone`, the noise floor of re-asking) |
| control "mismatched facts" | `wrong_key`: the masked note of another event |
| control "random maps" | `blank`: a content-free note in the same wrapper |
| dark memory: stored but not retrieved | the recipient misses the open question but passes the multiple-choice question |

Why it matters for the relay: notes are handed between models. If a masked cue recovers B's own memory, then B's
agreement after a key-only hand-over is an independent confirmation, and the rally preset could use a blind key round
(keys first, values after). If B only improves by copying, a `Confirm:` after a full note is an echo and the answer
must treat it as a single source. The same numbers are a text-level transfer matrix between the three models: how well
each one uses another's notes, which is what switching between Sol, Neon and Fable rests on.

## 2. Data (v1)

The AI-paper probe set (`data/events/papers_events.jsonl`: 231 events, 21 of them fake-paper controls, question "Who is
the first author of the arXiv paper titled ...?") and its graded results of 25 November 2026
(`data/results/papers_probes_2026-11-25.jsonl`, see `data/results/README.md`: probes `current_d` and `mcq` for all
three models, 1323 rows, graded by the rules and, for 57 answers, by the judge `gpt-6.2-sol`). The stored `current_d`
rows give every (event, model) its alone outcome and its response text, so the donor notes cost nothing.

Candidate counts in that file (donor correct, recipient not correct, real papers): Fable to Neon 50, Fable to Sol 96,
Neon to Sol 47; every other direction 1. The knowledge is nested in this topic (Fable knows almost everything the
others know), so the reverse directions cannot be measured with this set. Neon answers in one sentence, so most of its
correct answers mask to the bare sentence "The first author of the arXiv paper "[title]" is [withheld]." and carry no
cue (section 3): the Neon to Sol pair has almost no usable key. Section 9 gives the planned counts. The Wikidata events
have no stored results; they need a donor run first (later step).

## 3. Item sets per directed pair (donor A, recipient B)

| set | selection (stored `current_d` grades) | arms |
|---|---|---|
| `dark` | real paper, A `correct`, B not `correct`; A's key carries a cue | alone, key, full, wrong_key, blank |
| `donor_wrong` | real paper, A `incorrect` with a name A offers, B not `correct` | alone, full |
| `both_right` | real paper, A and B `correct`, A's key carries a cue; seeded sample of `--both-right` (default 10) | alone, key, full |
| `control` | fake paper, A `incorrect` (it offered an author for a paper that does not exist) | alone, full |

Rules, in the order the plan applies them:

- **Leak** (section 5): an item whose key still gives the answer away is left out of `dark` and `both_right`
  (`leaks`).
- **Cue**: a key must carry something the question does not. After removing `[withheld]`, `[title]`, markdown and
  quotes, its *cue tokens* are the words of three letters or more and the numbers that are not percentages, that are
  not words of the question (accent- and case-free), not in a fixed list of template and hedge words ("first",
  "author", "paper", "arxiv", "titled", "believe", "certain", "sure", "verify", "confirm", "recollection", ...), and
  not the paper's own acronym in parentheses. An item whose key has no cue token is left out (`no_cue`); the plan row
  stores the tokens (`cues`).
- **Wrong key**: drawn (seeded) among the donor's other correct, non-leaking notes that carry a cue themselves, never
  the same event and never one whose expected answer is this item's person (the two names compared letters-only, so
  "Jun-sol Kim" is "Junsol Kim"). It is masked with that event's answer forms and with this item's title; one that
  then equals the key on letters and digits (accents and case folded, so two notes that differ only in punctuation,
  markdown, quotes or whitespace are one note), or that still contains this item's answer, is not used. No candidate
  left: `no_wrong_key`.
- **Pointer**: a key that ties `[withheld]` to another work ("who also led X", "first author of the earlier X",
  "the original X paper lists [withheld]", "first-authored by ... [withheld]"), or that names a short handle (the main
  title before a colon, or a CamelCase or model-id token of a title) of another event in the events file with the
  same expected answer, gets `pointer: true`; `related_known` lists the recipient's stored `current_d` and `mcq`
  grades on those events. Pointer items are planned and run, but kept out of predictions 1, 1b and 2 (section 7): a
  success there may be B resolving A's pointer from a different fact, not B's own memory of this one.
- **min_items** (default 5) is applied after these filters: a pair with fewer surviving `dark` items loses its
  `dark` and `both_right` sets (`keyed: false`, listed under `excluded` with its counts). Its `donor_wrong` and
  `control` items are still planned, because the model that is rarely a useful donor (Sol in this topic) gives most of
  the wrong answers and every confabulated control.
- **Offered name** (`donor_wrong`, `control`): `donor_answer` is the name the donor offers (`asserted_name`, text in
  double quotes skipped); the item is kept only if the donor's own full note offers it (`offers`, section 6). No name:
  `no_name`; a name the note does not offer (it rules the name out, or names it only in a condition): `unasserted`.
- `recognised` (`dark` and `both_right`): B's stored `mcq` grade for the event is `correct`.
- The `both_right` sample is drawn among the candidates that pass the leak and cue rules.
- A plan with a repeated pair is refused.

## 4. Prompts

System prompt: `current_d_system(today)` for every arm (the probe the index is built on), `today` = the plan's date.

- `alone`: user = the event's `question_current` (the stored probe, asked again).
- `key`, `full`, `wrong_key`, `blank`: user = `KEY_USER_TEMPLATE`, identical for the four arms, so only the note
  differs:

```
{question}

Another analyst wrote this note about the question:
<<<
{note}
>>>
Use the note only as a pointer and answer from what you know yourself. If you do not know, say so.
```

- `full`: the donor's stored response, with the paper's title replaced by `[title]`.
- `key`: the same text with every form of the answer replaced by `[withheld]` (section 5).
- `wrong_key`: the donor's masked note about a different real paper (section 3). It has the shape of a key and points
  elsewhere.
- `blank`: `BLANK_NOTE` = `The first author of the arXiv paper "[title]" is [withheld].` The wrapper and the shape of
  a key with nothing the question does not say. The wrapper's last sentence ("If you do not know, say so") is in every
  note arm and not in `alone`; `key` against `blank` is controlled for it.

Calls: `ChatRequest(messages, max_tokens=4096, temperature=0.0, metadata={"event_id", "probe": "current_d", "model",
"arm", "set", "donor", "item", "donor_answer"})`, the same request shape as `bench run`. The configured models do not
take a temperature (`supports_temperature: false`), so every call, the stored ones included, is a sample: re-asking
alone is not deterministic, which is why `alone` is planned in `dark` and `both_right`.

## 5. Masking and leak check (`mask_note`)

- Forms of the answer: `expected_current`, every entry of `aliases`, and each name token of `expected_current`
  (surname and given names, at least 2 letters) as a whole word. Matching is case-insensitive, accents folded (the
  text is matched in both its own and its accent-free spelling), spaces and hyphens inside a name interchangeable.
  Every match becomes `[withheld]`; adjacent placeholders collapse into one. Name tokens under 4 letters are matched
  case-sensitively (the pronoun "he" survives a surname "He"). The dot of an initial is optional inside a name, but
  the initial must be upper case in the text ("M Pióro" is "M. Pióro"; "University's Yuan" is not "S. Yuan").
- The title (`subject`) becomes `[title]` (case-insensitive, punctuation between its words free). `[title]` also
  covers the part of the title before a colon when it has at least 4 letters ("SmolVLM: ...").
- Leak check: the masked note leaks when the paper rule grader (`grade_paper(event, "current_d", note)`) reads it as
  `correct`, when an answer token of 3 or more letters or an accepted answer form remains, or when a placeholder is
  joined by a space or hyphen to a capitalised word, so that the masked token was part of another person's or a lab's
  name ("Mengdi [withheld]", "[withheld] Balagansky", "[withheld] Group": a recipient who knows that co-author or lab
  learns part of the answer). A capitalised function word at the start of a sentence ("By", "Probably", ...) is not a
  name.
- Wrong-key notes additionally get the item's own title masked, and a wrong-key note that happens to contain the
  item's own answer is not used. An item whose key leaks is excluded from `dark` and `both_right` (`leaks`); its note
  may still serve other items as a wrong key once masked with its own forms and checked.

## 6. Plan, run, report

```
cutoffmix bench keys plan   --events E.jsonl --results R.jsonl --out runs/keys_plan.jsonl [--force]
                            [--models K,K,K] [--pairs A:B,...] [--both-right 10] [--min-items 5] [--seed 0] [--today D]
cutoffmix bench keys run    --plan runs/keys_plan.jsonl --events E.jsonl --out runs/keys_results.jsonl
                            [--mock] [--concurrency 8] [--max-cost USD] [--limit N] [--judges KEY ...] [--resend-stale]
cutoffmix bench keys report --plan runs/keys_plan.jsonl --results runs/keys_results.jsonl [--json OUT] [--md OUT]
```

A mock run writes to its own file (`runs/keys_results_mock.jsonl`); `run` refuses to mix backends in one file.

**plan** (offline, deterministic for its inputs and seed) writes one row per call: `id` (16 hex of sha1 of
"set|donor|recipient|event|arm"), `set`, `donor`, `recipient`, `event_id`, `arm`, `recognised`, `donor_grade`,
`donor_answer` (the expected answer for `dark` and `both_right`, the name their `full` answers are checked against;
the offered wrong name for `donor_wrong` and `control`), `note` (None for `alone`), `note_event` (the wrong key's
source event), `today`, `cues`, `pointer`, `related_known` (the last three None outside `dark` and `both_right`) and
`messages`. It prints the counts per pair, set and arm, the excluded pairs, the drops (`leaks`, `no_cue`,
`no_wrong_key`, `pointers`, `no_name`, `unasserted`), and a cost estimate (stored mean cost per `current_d` call per
model, scaled by prompt length). It refuses to replace an existing plan file whose ids carry other messages or another
`donor_answer` unless `--force` is given; `--force` never lets `--out` be the `--results` or `--events` file, or an
existing file that is not a plan. A repeated pair is refused.

**run** sends the planned calls and grades each answer with the existing graders (`grade_response`: the paper rules;
the judges only when `--judges` is given). It appends one row per answer:

`{id, set, donor, recipient, event_id, arm, response, grade, hedged, grader, adopted, adoption, adoption_grader,
latency_s, usage, judge_usage, backend, prompt_sha, request_sha, note_event, today, error}`

- **Provenance.** `backend` (`mock`, `openrouter`, ...), `prompt_sha` (sha1 of the canonical JSON of the planned
  messages, `prompt_sha()`), `request_sha` (sha1 of the messages, `max_tokens`, the temperature as the gateway sends it,
  none for a model that takes no temperature, and the recipient's model settings: provider id, reasoning, provider
  routing, extra body). Resuming counts a stored row as done only if
  it has provenance, the same backend and the same `request_sha`. A stored row without provenance or from another
  backend stops the run before any call (with the counts and the remedy: a new `--out`, or the original plan); never
  re-sent. A row of the same backend with another `request_sha` (the plan or the settings changed) also stops the run,
  unless `--resend-stale` is given; then those calls are sent again and the newest row counts.
- **Safety.** The rows of `--out` must be rows of `bench keys run` (with `id`, `arm`, `grade`; no `probe`, `model` or
  `messages`), else the run stops before any call. A plan with repeated ids is refused. An exclusive lock on
  `<out>.lock`, with `<out>` resolved to its real path (a symlink and its target share one lock), keeps a second run
  from writing the same file. An `--out` that exists but is not a regular file, or cannot be written, is refused
  before any call; a missing or wrong `--plan`, `--events` or `--results` file is a one-line error.
- **Resume.** An id that has a current row without `error` is not sent again; a send error (a failed call, an empty
  answer: Grading, below) is sent again; readers keep the last row per id. Before the first call the last line of
  `--out` is mended: a complete row without its newline (a tool that dropped the final newline) gets the newline and
  counts like every other row; only a torn line that is no valid JSON (an append the process did not finish) is cut,
  with a warning on the log. The report reads the file the same way (a complete last row counts, a torn one is
  skipped with a warning). The first Ctrl-C (SIGINT, on POSIX) stops the run from starting new calls; the calls in
  flight finish and write their rows, with their judge calls, and the run prints its usual summary, then
  "interrupted: N of M planned calls not sent; rerun to resume", and exits with status 130 (after the last call has
  started there is nothing left to stop, and the run ends as usual). A second Ctrl-C aborts at once: the calls still
  in flight write no row, and the next run sends them again. On other systems the first Ctrl-C aborts at once.
- **Cost and stops.** `--max-cost` stops sending new calls once the summed cost of **all** rows in `--out` (recipients
  and judges: `usage` and every entry of `judge_usage`) reaches it; calls in flight finish, with their judge calls. A
  run stopped by the cap says so (how many planned calls were not sent) and exits with status 3. An error that no
  retry can fix stops the run the same way: HTTP 401, 402 or 403 from the gateway (a missing or wrong key, no
  credits, no access; a 402 that the gateway marks temporary with Retry-After is retried instead). No new call
  starts, the calls in flight finish, and the run prints the error and how many planned calls were not sent and exits
  with status 4. Any other failed call writes an `error` row and the run goes on (the next run sends the call again).
  At the end such rows are counted apart from the grades (`errors N`), and when nothing else stopped the run it
  prints "N of M planned calls failed (first error: ...); rerun to send them again" and exits with status 4. A judge
  call that fails is a missing vote: its entry in `judge_usage` keeps the `error` (cost 0), the run prints "K judge
  calls failed (first error: ...)", and an answer it leaves undecided is asked again by the next run with `--judges`.
  `--limit` caps the calls a run sends. Real calls need `--max-cost`. `--mock` uses the simulated backend. Exit
  statuses, the first that applies: 130 after Ctrl-C, 3 at the cap, 4 after an error no retry can fix, 0 at
  `--limit`, 4 when planned calls failed, else 0; a refused input or option is a one-line error with status 2, before
  any call.
- **Grading.** A policy refusal (`content_filter`) is `ungraded` with grader `refusal`; the gateway reports OpenAI's
  native refusal shape (no content, a `refusal` text) as `content_filter` on the non-streaming path too. An exception
  becomes an `ungraded` row with `error`; a completion that failed inside an HTTP 200 (`choices[0].error`, or
  `finish_reason` `error`) raises a retryable error in the gateway, so it is retried and, while it keeps failing,
  written as such a row. An answer without visible text whose `finish_reason` is neither `length` nor
  `content_filter` (an empty reply) is no answer either: its row keeps the `usage` for the cap, has the `error` "empty
  answer (finish_reason ...)", and is sent again by the next run. An empty answer cut at `max_tokens` (`length`) stays
  an answer, because `max_tokens` is pre-registered. With `--judges`, stored rows that are `ungraded` without error
  and not refusals, and stored rows whose label is `reported` and undecided (adoption rows, and the `full` rows of the
  next point), are graded again from their stored response (no new recipient call); the new row replaces the old one
  under the same id. For the pilot the judge is `gpt-6.2-sol`, the judge of the stored results that define the sets.
  An exception while grading an answer that came back (the rules, the judges, the label) writes the row with `error`
  "grading failed: ..." and keeps the response, its usage and the judge calls made; the next run grades it again from
  the stored response and never sends the call again. A stored row whose label differs from the label its response
  gets now (for the plan's current `donor_answer` on an adoption row, for the right name on a `full` row of the next
  point) is labelled again from its stored response, without a new call; a row whose label is current is left as it
  is.
- **The right name on `full` rows** (`dark`, `both_right`). Their note shows the expected answer (the plan row's
  `donor_answer`), so an answer can name it without giving it as its own: "The note says X, but I believe the first
  author is Y.", "The note's answer, X, is wrong.", "According to the note, the first author is X." The paper rules
  grade all three `correct`. So after the grading (the rules, and the judges with `--judges`) a `correct` answer is
  labelled for that name with `adoption_label` (next point): `endorsed` keeps `correct`, and so does `absent` (the
  rules or the judges found the name where `offers` does not, e.g. "Z. Wang" for "Ziqi Wang"); `reported` makes the
  row `ungraded` until the adoption judges decide, with the question of the next point and the right name as the
  candidate: ENDORSE keeps `correct`, REPORT or REJECT make it `incorrect` (the answer does not give the right name as
  its own), no majority leaves it `ungraded`. `hedged` and `grader` keep what the grading said; `adoption` holds the
  label, `adopted` the decision (true for endorsed, absent and ENDORSE, false for REPORT and REJECT, null while
  undecided) and `adoption_grader` `rule` or `judge:<keys>`, so the row shows why. A `full` row with another grade,
  and every other row of `dark` and `both_right`, has no label.
- **Adoption** (`donor_wrong`, `control`; on the `full` rows of `dark` and `both_right` the same fields record the check
  of the right name, above; None on every other row). `adoption` labels how the answer treats the donor's name:
  - `endorsed`: the name is offered (`offers`, below) in a clause that does not refer to the note;
  - `reported`: it is offered only in clauses that refer to the note ("the note says / claims / suggests / names X",
    "according to the note", "the note's answer (X)"): the rules cannot tell a pass-through ("According to the note,
    the first author is X") from a reported claim the answer declines. A clause refers to the note when it names it:
    "the", "this", "that", "your", "another" or "other" and up to two words before "note", "pointer" or "hint", or
    "note's", or "analyst". `reported` also covers a name given only as the author of another work ("the original T
    paper's first author was X", "if you are thinking of T, its first author is X"); a guess that a related work only
    justifies ("My best guess is X, who was first author of the earlier T paper") stays `endorsed`. That case needs no
    note, so an `alone` row can be `reported` too;
  - `absent`: it is not offered (missing, negated, conditional, asked).

  `adopted`: false when the answer is graded `correct`; else true for `endorsed`, false for `absent`, and for
  `reported` null (undecided) unless judges decide it: each judge answers one dedicated question (does the answer give
  X as its own answer to this question, also when it cites a note or another work as its source: ENDORSE; does it only
  report a claim about X, or name X only as the author of another work: REPORT; does it reject X: REJECT; a reply
  counts only where it states its verdict: its first word, a line that is only the label or ends in it after a colon
  or a dash, or a label field of a JSON object), the majority decides (true for ENDORSE, false otherwise), and `adoption_grader`
  records `rule` or `judge:<keys>`.

  `offers(text, name, event)`: a full form of the name, or its surname alone when that surname is distinctive (longer
  than four letters, not common), in an asserted clause. Sentences end at `.`, `!`, `?` (also after a period inside
  closing quotes); a clause ends at a semicolon or a contrastive conjunction ("but", "though", ...). A negation ("not",
  "n't", "never", "no", "cannot", "doubt", "unlikely", ...) before the name anywhere in its clause, or after it inside
  the name's own comma clause, a question, or "if", "whether", "unless" inside the name's own comma clause make it no
  assertion, so "The note names X, which I cannot verify." still offers X (and is `reported`). "would", "could",
  "might" are hedges, so "It might be X" and "X would be my best guess" offer X. The paper's own title is blanked
  first, so a title word ("Never", "No") negates nothing.

**report** (offline) joins plan and results (the last row per id). Rows whose `prompt_sha` differs from the plan's
messages (`stale`), rows without provenance and rows of ids not in the plan (`foreign`, their cost reported
separately) are left out and counted. `stale` compares the messages only: the report has no configuration, so a change
of the model settings is caught by `run` (`request_sha`), which refuses to resume until every changed call has been
sent again with `--resend-stale` (run that to the end before reporting). Its inputs are checked like those of `run`: a
missing or unreadable `--plan` or `--results` file, a plan that fails the plan check, and a results file with a line
that is not a row of `run` are one-line errors that name the option, and an empty plan is refused. `--json` and `--md`
may not be the `--plan` or the `--results` file (a symlink or a hard link to it included), nor one file; that is
refused before anything is read or written. Per pair, set and arm: planned, answered, missing, errors, refusals,
ungraded and the correct rate with a 95 % Wilson interval, in the primary analysis (below) and with ungraded rows left
out (the sensitivity line); an arm with more than 10 % ungraded rows is flagged. Paired comparisons (exact McNemar,
two-sided binomial on the discordant pairs) on the items with both compared arms answered: on `dark` items `key` vs
`alone`, `key` vs `wrong_key`, `key` vs `blank`, `blank` vs `alone`, `blank` vs `wrong_key`, `full` vs `key`, over all
items and without pointer items; the `dark` set split by `recognised`, with the guess-corrected recognised share
`(s - 0.25) / 0.75` (four MCQ options); adoption per arm (endorsed, reported and undecided, absent; `adopted` rate) and
`full` vs `alone`; on `both_right` the not-correct rate per arm and `key` vs `alone`. Costs: recipients, judges and
foreign rows separately, over every row of the results file. Markdown on stdout, JSON with `--json`. When every joined
row comes from the mock the report is headed SIMULATED and each verdict is prefixed "simulated:"; when the joined rows
come from more than one backend it gives no verdicts.

**Primary analysis.** An answered row (no error, not a refusal) counts; `ungraded` (the rules could not decide, or a
`full` answer gave the right name only in reported speech, and no judge decided) counts as **not correct**. That is
close to exact, because the rules leave almost only misses undecided (stored results: 1 correct answer among 57
undecided ones), while dropping ungraded rows would inflate every arm's rate and hide harm. It is conservative only
for the ungraded rows of the `key` arm, which count against the key and as harm (and for those of `full` in 3a, which
count against `full`). An ungraded row of the other arm of a comparison works the other way: an ungraded `alone`,
`wrong_key` or `blank` row counts for the key, an ungraded `key` row counts for `full` in 3a, and an ungraded `alone`
row on `both_right` lowers the excess harm. The sensitivity line, with ungraded rows left out, shows the difference.
Refusals are left out and counted.

## 7. Pre-registered predictions (version 2, written before any real call)

Taken from the paper and stated so the pilot can refute them.

1. On `dark` items, `key` beats both `alone` and `wrong_key`. Some, not most, of the misses are recovered (the paper:
   18 % of dark memories); we expect 10 to 30 % of `dark` items correct under `key` against under 10 % under
   `wrong_key`.
   - **1b.** `key` beats `blank`: the donor's content, not the presence of a note, does the work.
2. The recovery is concentrated on `recognised` items: the key's gain over `alone` is larger on recognised items than
   on unrecognised ones.
3. `full` beats `key` (the value is copied), and `full` makes the recipient endorse a wrong donor answer (`donor_wrong`,
   `control`) far more often than `alone` does.
4. The notes rarely break a correct recall: on `both_right`, the excess harm of `key` over re-asking alone is under
   10 %.

How the report decides (written before any real call; "interim" and "not testable" give no verdict):

- **Decisive pair**: the keyed pair with the most planned `dark` items without pointer (on the v1 data Fable to Sol);
  ties go to the pair with more planned `dark` items, then to the first pair name in alphabetical order. It is fixed
  by the plan, not by the results. Predictions 1, 1b and 2 use `dark` items without pointer; their numbers
  over all items, and the pointer items on their own ("B resolves A's pointer"), are printed next to them.
- **Interim**: a prediction whose planned rows (the decisive pair's `dark` set for 1 and 1b, the keyed pairs' `dark`
  sets for 2 and 3a, the adoption sets for 3b, the `both_right` sets for 4) are not all answered (missing, error,
  stale, without provenance) is "interim (k of N planned items answered)". Verdicts count only on the complete plan. A refusal is a final outcome
  and keeps no prediction interim; a row whose grading failed does until a later run grades it. Each half of
  prediction 3 is interim on its own rows. An interim, mixed-backend or simulated report states every part of a
  prediction (each comparison, each half) the same way as its headline.
- **Floor**: a comparison on fewer than 6 items is "not testable (n too small)": an exact two-sided McNemar test
  cannot reach p < 0.05 with fewer than 6 discordant pairs.
- **Three-way verdict** for a paired comparison "X beats Y" with margin m: *supported* if b > c and p < 0.05;
  *refuted* if the 95 % Wilson upper bound of b / n is below m (the data rule out a gain of m: the paired gain
  (b - c) / n is at most b / n); otherwise *inconclusive* ("k items; about N would be needed for a gain of m"). N is the
  smallest number of items at which a gain of m would be significant at the observed share of discordant items (with
  no discordance against X: 51 items at m = 0.10, 26 at m = 0.20). When N is at most k the items sufficed for such a
  gain and the data show none: "k items; a gain of m is neither shown nor ruled out". Testability is never decided by
  the observed number of discordant pairs.

1. Supported if `key` vs `alone` and `key` vs `wrong_key` are both supported (margin 0.10); not supported if either is
   refuted; not testable if either is; else inconclusive. Printed with it: `blank` vs `alone` (the note's presence) and
   `blank` vs `wrong_key` (the harm of a mismatched cue). If `blank` beats `wrong_key` (b > c, p < 0.05), a supported
   verdict reads "supported (confounded: the wrong key misleads)": then key against wrong key may be won by the
   control's harm.
   - **1b.** `key` vs `blank` with margin 0.10, the same rule.
2. Testable only when prediction 1 is supported and each group has at least 10 units. Units: distinct (recipient,
   event) over the keyed pairs, `dark` items without pointer with `alone`, `key` and `wrong_key` answered (an event that
   is `dark` in two pairs with the same recipient keeps the decisive pair's row, else the row of the pair with more
   planned `dark` items). Per unit d = [key correct] - [alone correct] and d_w = [key correct] - [wrong_key correct].
   Supported if (a) on recognised units `key` beats `alone` (b > c, p < 0.05); (b) the mean of d is higher on
   recognised than on unrecognised units by a permutation test stratified by pair (the recognised flags permuted within
   each pair; statistic: the sum over pairs of n_r n_u / n times (mean d over recognised - mean d over unrecognised);
   20 000 permutations with a fixed seed; two-sided p < 0.05 and the observed statistic positive); and (c) the same
   holds for d_w. Otherwise inconclusive, with the stratified difference and its p. At this n prediction 2 can be
   supported or inconclusive, not refuted. A rate of "key correct and alone not" is not compared between the groups:
   it rises with the re-ask success rate, which is higher on recognised items, even with no key effect.
3. Two halves, both pooled over the keyed pairs (3a) or over `donor_wrong` and `control` of every pair (3b).
   - **3a.** `full` beats `key` on `dark` items, margin 0.10. A `full` row is correct only when the answer gives the
     right name as its own (section 6, the right name on `full` rows): an answer that only relays or rejects the
     note's name is `incorrect` once the judges decide, and `ungraded` until they do (not correct in the primary
     analysis, left out of the sensitivity line). As in 3b, only endorsement counts.
   - **3b.** `full` vs `alone` on adoption (`adopted` true), margin 0.20 ("far more often"). If more than 20 % of the
     `full` rows are undecided (`adopted` null: reported speech the rules cannot decide), 3b is "not testable
     (reported speech undecided; grade with --judges)". The checks run in this order: the floor, the undecided share,
     the three-way verdict.
   - Supported if both halves are supported; not supported if one is refuted; not testable if one is; else
     inconclusive.
4. On `both_right` items with `alone` and `key` answered (pooled over pairs): b = alone correct and key not, c = key
   correct and alone not; excess harm (b - c) / n. Supported if n >= 10 and the excess harm is under 10 %; not
   supported if n >= 10 and it is 10 % or more. Printed with it: the not-correct rate of `alone` (re-ask noise), `key`
   and `full`. Re-ask noise only adds misses, so the excess harm of `key` over `alone` is what the note itself does.

If 1 fails while 3 holds, a model's agreement with a hand-over is copying, and the rally's confirmations after full
notes must count as single source. If 1 and 1b hold, a blind key round in the rally is worth a test of its own; if 1
holds and 1b does not, any note in that shape helps, and the donor's content is not what the recipient uses.

## 8. Mock

`MockBackend.chat` answers a keyed request (`metadata["arm"]`) by the simulated model's truth like any probe, plus:
`key` turns a miss into the right answer with a fixed probability when the event lies within a few months after the
model's simulated cutoff (a simulated dark memory); `full` copies the donor's name (from the note) with a fixed
probability, as a bare answer ("X.", which section 6 reads as endorsed); `wrong_key` and `blank` behave like `alone`.
Judge requests get no scripted answer, so a mock run with `--judges` leaves undecided rows undecided. Deterministic per
seed. The mock exists to run the pipeline offline; its numbers say nothing about the real models, and the report marks
them SIMULATED.

## 9. Costs

The version 2 plan (seed 0, `--today 2026-11-27`) has **746 calls**:

| set | items | arms | calls |
|---|---:|---|---:|
| `dark` | 116 (Fable to Sol 79, Fable to Neon 37) | alone, key, full, wrong_key, blank | 580 |
| `donor_wrong` | 45 | alone, full | 90 |
| `both_right` | 20 (10 per Fable pair) | alone, key, full | 60 |
| `control` | 8 | alone, full | 16 |

- Funnels of the `dark` candidates: Fable to Sol 96: 4 leaks, 13 without a cue, 79 kept; Fable to Neon 50: 2 leaks, 11
  without a cue, 37 kept; Neon to Sol 47: 1 leak, 45 without a cue, 1 kept, so the pair is unkeyed (below
  `--min-items`). No wrong key had to be dropped. Pointers: 17 (8 `dark` items in each Fable pair, the same events, and
  one `both_right` item). Adoption items: 53 (`no_name` 4, `unasserted` 2 dropped).
- Calls by pair: Fable to Sol 439, Fable to Neon 231, Sol to Neon 44, Sol to Fable 26, Neon to Sol 4, Neon to Fable 2.
  By recipient: Sol 443, Neon 275, Fable 28.
- Estimate from the stored usage (mean cost per `current_d` call per recipient, prompt part scaled by the planned
  prompt length): **2.17 USD** (Neon 1.47, Sol 0.45, Fable 0.24). The judges come on top: on the version 1 plan (1161
  calls) an offline replay with the stored answers needed 96 judge calls for 0.08 USD. The check of the right name on
  `full` rows (section 6) adds at most 136 adoption-judge calls with one judge (the 116 `full` rows of `dark` and the
  20 of `both_right`, if every one gave the name only in reported speech). The stored results hold no usage of the
  judge's own calls (a judged row keeps the usage of the answer it grades), so these are priced from the stored
  `current_d` usage of the judge model `gpt-6.2-sol` (231 calls; 2 USD per million prompt tokens, 10 per million
  completion tokens): with the recipient's stored answer in the judge's prompt and the mean stored completion (69
  tokens) a call costs about 0.0014 USD, 136 calls about 0.18 USD; with the longest stored answer or note and the
  longest stored completion (184 tokens) about 0.0028 USD, 136 calls about 0.38 USD. So the judges cost a few cents up
  to about 0.35 USD (about 0.55 USD at worst), and with the recipients the pilot stays under its cap of 3 USD.
- Power: the decisive pair Fable to Sol has 71 `dark` items without pointer. A gain of 0.10 needs at least 51 items;
  the Wilson upper bound of 0 of 71 is 5.1 %, so a key that does nothing can be refuted there. Prediction 2 has 100
  units (63 recognised, 37 unrecognised).

Approval and an OpenRouter key are needed first; `run` refuses real calls without `--max-cost` (the pilot:
`--max-cost 3`), and the cap counts the judges.

## 10. Changes from version 1 (2026-11-27, before any real call)

An adversarial review (three lenses: validity and statistics, masking on the real data, run safety; every finding
verified against the code and simulated on the real plan) confirmed 16 findings. The design changed as follows.

| finding | version 1 | version 2 |
|---|---|---|
| adoption counted reported speech | `adopted` = the name appears outside a negated clause | `adoption` endorsed / reported / absent, judges decide reported speech, prediction 3b on endorsed only |
| prediction 2 compared levels | key rate on recognised vs unrecognised items, pooled | the key's gain over `alone`, stratified by pair, distinct units, gated on prediction 1 |
| ungraded rows dropped | left out of every n, an item needed all four arms graded, decisive pair = most complete | ungraded counts as not correct, pairs per comparison, decisive pair fixed by the plan |
| no content-free control | wrong key only | `blank` arm and prediction 1b |
| cue-free keys | any non-leaking note was a key; Neon's bare sentences made key = wrong key | cue rule (`no_cue`), wrong key with a cue and never equal to the key, `min_items` after the filters |
| pointers to other papers | not detected | `pointer` flag, kept out of predictions 1, 1b and 2 |
| no re-ask baseline on `both_right` | harm = share not correct under key | `alone` arm, excess harm over alone |
| tokens shared with co-authors | "Mengdi [withheld]" passed the leak check | a placeholder joined to a capitalised word is a leak; initials must be upper case |
| donor's wrong name not offered | `asserted_name` alone | `offers` check at plan time (`unasserted`), the same rule for adoption, "she" no longer a surname stop word |
| rows matched by id only | mock and real rows, old and new prompts mixed silently | provenance per row, resume refuses mismatches, the report drops stale rows and marks mock reports |
| partial runs gave refutations | testable from 10 items, two-way verdict | interim, floor of 6, three-way verdict with margins |
| ungraded rows could not be repaired | `bench grade` could not read keys rows | `run --judges` re-grades stored rows without new recipient calls |
| judge cost outside the cap | judges' usage dropped | `judge_usage` per row, counted by the cap and the report |
| double sends | repeated pairs, two runs on one file | repeated pairs and ids refused, lock file, cost summed over every row |
| `--out` not checked | a probe-results file as `--out` got keys rows appended | `--out` must hold keys rows only |

The rejected finding (wrong keys that name their source paper by arXiv id or acronym) is by design: a wrong key is the
mismatched-facts control and points elsewhere; keys name their own paper as often.

### Refinements during the implementation review (2026-11-27, still before any real call)

Each area (masking and the plan, the run, the report) was reviewed against this document once built, and then all
areas together. The confirmed issues refined the rules below; sections 3 to 7 above already state the refined rules.
The planned calls are the same 746; only the count of surviving Neon to Sol items changed (2 to 1; the pair stays
unkeyed). The review across the areas added the last ten rows. The first of them (the right name on `full` rows)
leaves the planned calls as they are and may need up to 136 more judge calls (section 9). The other nine change how
the report checks and writes its files, how a run fails, stops and resumes, and how a wrong key is compared with its
key; the 746 planned calls stay the same, ids and messages byte for byte.

| issue | before | now |
|---|---|---|
| a name given only as the author of another work | `endorsed` (final) | `reported`, the judges decide; their question covers it |
| negation scope in `offers` | anywhere in the clause | before the name in its clause, after it only in its comma clause; the title is blanked first |
| pointer relation words ("also", "original", "had") | matched with no other work named | another work (a work noun, a capitalised name, a model id) must follow |
| the paper's own acronym | any capitals in the order of the title's initials | from the first content word on; consecutive initials need 3 capitals |
| hedges as cues | "right now", the lower bound of a range ("90-95%") were cues | stop words and percentages |
| a wrong key about the same person | its expected answer compared with this item's as normalised text | the two names compared letters-only ("Jun-sol Kim" is "Junsol Kim"), as masking and pointers compare |
| adoption judge votes | a label anywhere in the reply, also negated | only where the reply states its verdict |
| an exception while grading | ended the run | a `grading failed` row, graded again by the next run without a new call |
| a changed `donor_answer` | went unnoticed | `plan` refuses without `--force`; `run` labels the stored rows again |
| temperature in `request_sha` | always 0.0 | as the gateway sends it |
| the lock | per spelled path | per real path; an `--out` that cannot take rows is refused |
| parts of a prediction | showed verdicts in interim, mixed or simulated reports | stated like the headline |
| "about N would be needed" | a fixed N | N at the observed discordance; "neither shown nor ruled out" when N is at most k |
| joined cost | the last row per id | every row of the request that row answered |
| the right name on `full` rows (`dark`, `both_right`) | `correct` whenever the rules find the name in an asserted clause, also in "The note says X, but I believe it is Y" and "The note's answer, X, is wrong" (3a counted relaying, 3b only endorsement) | labelled like adoption: endorsed (or absent) stays `correct`, reported speech is `ungraded` until the adoption judges decide (ENDORSE `correct`, REPORT or REJECT `incorrect`) |
| the report's input and output files | read unchecked: a missing file was a traceback, an events or probe-results file gave a plausible report on "missing" calls; `--json` or `--md` could replace the plan or the paid results | one-line errors that name the option (the plan check, every results line a row of `run`, an empty plan refused); `--json` and `--md` may not be `--plan`, `--results` or one file |
| a complete last row without its newline | cut like a torn line, and its call paid again | kept (the newline is added); only a torn line is cut, with a warning on the log |
| failed calls at the end of a run | folded into `ungraded`, exit status 0 | counted apart (`errors N`); "N of M planned calls failed (first error: ...)", exit status 4 |
| an error no retry can fix (HTTP 401, 402, 403) | every remaining call failed the same way | no new call starts, the calls in flight finish; the error and the calls not sent are printed, exit status 4 |
| a failed judge call | a missing vote, recorded nowhere | an entry with its `error` in `judge_usage`; "K judge calls failed" |
| non-streaming shapes: a completion that failed inside an HTTP 200, OpenAI's native refusal | an empty answer, a miss that was never sent again | a retryable error (retried, then a send error), and a refusal (`content_filter`, left out) |
| an empty answer (no visible text, not cut at `max_tokens`) | an answered row, `ungraded`, so a miss in the primary analysis | a send error, sent again by the next run; an empty `length` answer stays an answer |
| Ctrl-C | the calls in flight were dropped without a row, still billed, and sent again on resume | the first Ctrl-C starts no new call and lets those in flight finish and write their rows (exit status 130); a second aborts at once |
| a wrong key equal to the key up to punctuation | the notes compared with markdown, quotes and whitespace removed, punctuation kept ("It is [withheld], with X." differed from "It is [withheld] with X") | the notes compared on letters and digits, accents and case folded (`no_wrong_key`); the real plan is unchanged |
