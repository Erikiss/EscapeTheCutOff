# Key, not value: does another model's cue unlock a model's own memory?

Status: design contract and pre-registered predictions, **version 2** (amended on 2026-11-27, after an adversarial
review of version 1 and before any real call; section 10 lists what changed and why). Built and tested offline; the
pilot with the real models has **not** run (it needs an OpenRouter key and the owner's approval, about 3 dollars; the
cap the owner is asked to approve is 3.5 dollars, section 9).

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
the judges only when `--judges` is given; without them the paper rules' `incorrect` next to the expected answer is
unsure, and with them the grading judges decide every rule `incorrect` or `abstain` of an answer that names the
expected answer, Grading and the grading scope below). It appends one row per answer:

`{id, set, donor, recipient, event_id, arm, response, grade, hedged, grader, offers_expected, adopted, adoption,
adoption_name, adoption_grader, rule_adopted, latency_s, usage, judge_usage, backend, prompt_sha, request_sha,
note_event, today, error}`

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
  counts like every other row; only a torn line that is no valid JSON (an append the process did not finish, also one
  cut inside a multi-byte character: a character cut at the very end of the file is read as a replacement character) is
  cut, with a warning on the log. Any other byte that is not UTF-8, in the last line as in every other (a complete last
  row without its newline included), refuses the file before it is read, mended or appended to, with its offset in the
  file. The report reads the file the same way (a complete last row counts, a torn one is skipped with a warning, a bad
  byte refuses the file). The first Ctrl-C (SIGINT, on POSIX) stops the run from starting new calls from the moment it
  arrives (its handler sets the stop at once, so a Ctrl-C before the first call starts none); the calls in flight finish
  and write their rows, with their judge calls, and the run prints its usual summary, then "interrupted: N of M planned
  calls not sent; rerun to resume", and exits with status 130 (after the last call has started there is nothing left to
  stop, and the run ends as usual). A second Ctrl-C aborts at once: the calls still in flight write no row, and the next
  run sends them again. On other systems the first Ctrl-C aborts at once.
- **Cost and stops.** `--max-cost` stops sending new calls once the summed cost of **all** rows in `--out` (recipients
  and judges: `usage` and every entry of `judge_usage`) reaches it; calls in flight finish, with their judge calls, so a
  run can end above the cap by what the items in flight cost (up to `--concurrency` items, the one that reached the cap
  included; section 9). A run stopped by the cap says so (how many planned calls were not sent) and exits with status 3.
  An error that no retry can fix stops the run the same way: HTTP 401 or 402 from the gateway, on a recipient's call or
  a judge's (a missing or wrong key, no credits; a 402 that the gateway marks temporary with Retry-After is retried
  instead). No new call or re-grade starts, the calls in flight finish, and the run prints the error and how many
  planned calls failed and were not sent and exits with status 4 (also when the error came after the last call had
  started: the next run would meet it again). A 403 is no such error: on the gateway it can concern one request (a
  moderation flag, a refusal returned as an error), so it is a failed call like any other, and a 403 for every call
  fails each of them at once and at no cost. Any other failed call writes an `error` row and the run goes on (the next
  run sends the call again). At the end such rows are counted apart from the grades (`errors N`), and when no stop above
  ended the run it prints "N of M planned calls failed (first error: ...); rerun to send them again" and exits with
  status 4, also at `--limit`. A judge call that fails is a missing vote: its entry in `judge_usage` keeps the `error`
  (cost 0), the run prints "K judge calls failed (first error: ...)", and an answer it leaves undecided is asked again
  by the next run with `--judges`; a failed call that the grading sends again without JSON mode (a judge that rejects
  `response_format` with HTTP 400, 415 or 422) is marked `retried` and not counted, since its retry is. A reply
  without a verdict (an HTTP 200 from which no vote can be read) records no failed call, so the run also counts the
  current answers whose grade or label the judges' votes left undecided, whatever the cause, and prints "N answers are
  undecided after the judges' votes (a failed call, a tie or a reply without a verdict): rerun with --judges before
  reporting; ..."; the report flags them as well, and the exit status stays as below. The stop lines
  count the planned calls that were sent and failed apart from those not sent ("3 of 10 planned calls failed and 7 not
  sent"). `--limit` caps the calls a run sends. Real calls need `--max-cost`. `--mock` uses the simulated backend. Exit
  status: that of the stop that first kept planned work from starting, 130 for Ctrl-C, 4 for an error no retry can fix,
  3 for the cap; stops that meet at one check rank Ctrl-C, then the error, then the cap, and a Ctrl-C or an error after
  another stop keeps that stop's status (after the cap has stopped the run, a Ctrl-C stops nothing, and the run exits
  with 3 once the calls in flight have written their rows). Without such a stop: 4 when planned calls failed (also at
  `--limit`), else 0 (done, or stopped by `--limit`). A refused input or option is a one-line error with status 2,
  before any call. Status 2 also ends a run whose results file cannot take a row once calls were sent (a full disk, a
  file made read-only meanwhile): no new call starts, the error is the one line, the rows written count, and a call in
  flight whose row could not be written was paid and is sent again by the next run.
- **Grading.** The paper rules grade an answer as they grade a probe answer (`grade_response`). Without `--judges`
  one difference applies, on every arm alike: their `incorrect` is `ungraded` where `offers_answer` reads the expected
  answer as given (`offers`, below, for the expected answer or one of its accepted forms of two words or more, such as
  "Wang, Ziqi"). The paper rules let a negation or hedge before the expected answer reach across a comma and then
  take any other name for the answer given: "It is not Y, it is X.", "It's not Y - it's X.", "It might be X, who
  works with Y." and "Unless I'm mistaken, the first author is X, with Y as senior author." were `incorrect`, final on
  every arm. With `--judges` the grading scope (below) applies instead: such an answer, and every other rule grade
  `incorrect` or `abstain` of an answer that names the expected answer, goes to the grading judges. Without `--judges`
  that one difference is all, a known limit of that path: a rule grade `incorrect` of an answer that names the
  expected answer in a form `offers_answer` does not accept stays final, such as a correction with "Dr." or a surname
  that is not distinctive ("It is not Y, it is Dr. Zhang.", "It is not Y, it is Zhang.") or a title the paper rules
  read as part of another person's name ("The first author is Professor Guo."); the pilot runs with `--judges`.
  `grading.py` itself is unchanged for the other experiments; none of the stored answers is a rule `incorrect` that
  `offers_answer` doubts. Every answered row of an event with an expected answer stores whether the answer names it
  (`offers_expected`: `offers_answer`, or `names_answer`, below; null for refusals, errors and controls), so that the
  report finds its open offers (below). A policy refusal (`content_filter`) is `ungraded` with grader `refusal`; the
  gateway reports OpenAI's native refusal shape (no content, a `refusal` text) as `content_filter` on the non-streaming
  path too. An exception becomes an `ungraded` row with `error`; a completion that failed inside an HTTP 200
  (`choices[0].error`, or `finish_reason` `error`) raises a retryable error in the gateway, so it is retried and, while
  it keeps failing, written as such a row. An answer without visible text whose `finish_reason` is neither `length` nor
  `content_filter` (an empty reply) is no answer either: its row keeps the `usage` for the cap, has the `error` "empty
  answer (finish_reason ...)", and is sent again by the next run. An empty answer cut at `max_tokens` (`length`) stays
  an answer, because `max_tokens` is pre-registered. A stored answer the rules graded whose rule grade differs from
  what the rules give it now (the grading step's `grade` and `hedged`; `ungraded` where they are unsure) is graded
  again from its stored response, without a new call, and labelled again; a grade of the grading judges stands. With
  `--judges`, stored rows that are `ungraded` without error and not refusals, stored rule grades that are in the
  grading judges' scope now (below), stored labelled rows that are still undecided (`adopted` null: adoption rows, and
  the `full` rows of the next point, also those whose judge calls failed), and stored labelled rows the rules decided
  that are in the judges' scope now (below) are graded again from their stored response (no new recipient call); the
  new row replaces the old one under the same id. Once the judges have decided every row in their scope, a second run
  with the same `--judges` sends nothing; a failed or tied vote, or a reply without a verdict, is asked again by every
  later run with `--judges` (and paid again). A stored answered row without `offers_expected` (written before it was
  stored), or whose value differs from what the rule gives its response now (a row written before the rule read every
  form that names the expected answer), is written again with it, without a call; the report reads a results file
  without the field (below). For the pilot the judge is `gpt-6.2-sol`, the judge of the
  stored results that define the sets. An exception while grading an answer that came back (the rules, the judges, the
  label) writes the row with `error` "grading failed: ..." and keeps the response, its usage and the judge calls made;
  the next run grades it again from the stored response and never sends the call again. A stored row whose label differs
  from the label its response gets now (for the plan's current `donor_answer` on an adoption row, for the right name on
  a `full` row of the next point), whose `rule_adopted` or `adoption_name` is missing (a row written before they were
  stored) or differs from the rules' decision or the name now (a re-plan that changed `donor_answer`, which neither
  sha covers), or whose rule decision (`adoption_grader` `rule`: its `grade` and `adopted`) differs from what the
  rules give it now, is labelled again from its stored response, without a new call. A decision of the judges about
  the same name stands while the answer is in their scope for that name (below), whatever its label now, also in a
  later run without judges: their question depends only on the question, the name and the answer, so after a
  refinement of the labelling rules such a row is written again only to store the current label and the rules'
  current decision next to it, without a judge call. A decision about another name counts for nothing: the row is
  labelled for the current name, and the rules decide it when the answer does not mention that name, the judges (with
  `--judges`) when it does. A row written before `adoption_name` was stored keeps its judges' decision while the answer
  is in their scope for the current name. A row whose label, name and rule decision are current is left as it is.
- **The right name on `full` rows** (`dark`, `both_right`). Their note shows the expected answer (the plan row's
  `donor_answer`), so an answer can name it without giving it as its own: "The note says X, but I believe the first
  author is Y.", "The note's answer, X, is wrong.", "According to the note, the first author is X." The paper rules
  grade all three `correct`. So after the grading (the rules, and the judges with `--judges`) a `correct` answer is
  labelled for that name with `adoption_label` (next point). Without judges the rules keep `correct` only for a clear
  `endorsed`: nobody else offered (`offers_another`), no other role for the person and no rejection in other words
  (`contradicted`), and no disclaimer (`disclaims`; all three in the next point). Every other correct answer is
  `ungraded`: `reported`; `absent`, where the rules and `offers` disagree (an alias form the rules accept, such as "Z.
  Wang" or "Wang, Ziqi" for "Ziqi Wang", given as the answer's own or only relayed, or a rejection only `offers`
  reads, such as "I doubt it is X"); and an `endorsed` answer that is not clear: next to another person ("X is a
  co-author, but the first author is Y", also a correct answer that lists co-authors), with another role ("X is a
  co-author, not the first author", "X is the senior author"), or with a disclaimer ("I am unable to verify that X is
  the first author": the paper rules grade the expected answer next to a disclaimer `correct`, hedged, so the
  disclaimer is checked here). With `--judges` the adoption judges decide every correct `full` row (the judges'
  scope, below), with the question of the next point and the right name as the candidate, whatever the rules give:
  ENDORSE keeps `correct`, REPORT or REJECT make it `incorrect` (the answer does not give the right name as its own),
  no majority or a failed call leaves it `ungraded`. `hedged` and `grader` keep what the grading said; `adoption`
  holds the label, `adopted` the decision (true for endorsed by rule and for ENDORSE, false for REPORT and REJECT,
  null while undecided), `adoption_grader` `rule` or `judge:<keys>`, so the row shows why, `rule_adopted` the
  rules' own decision and `adoption_name` the name the label is measured against. A `full` row with another grade,
  and every other row of `dark` and `both_right`, has no label.
- **Adoption** (`donor_wrong`, `control`; on the `full` rows of `dark` and `both_right` the same fields record the check
  of the right name, above; None on every other row). `adoption` labels how the answer treats the donor's name:
  - `endorsed`: the name is offered (`offers`, below) in a clause that does not refer to the note;
  - `reported`: it is offered only in clauses that refer to the note ("the note says / claims / suggests / names X",
    "according to the note", "the note's answer (X)"): the rules cannot tell a pass-through ("According to the note,
    the first author is X") from a reported claim the answer declines. A clause refers to the note when it names it:
    "the", "this", "that", "your", "another" or "other" and up to two words before "note", "pointer", "hint",
    "snippet", "excerpt", "passage" or "summary"; "note's" or "analyst"; "the provided" or "the given" text, context,
    information, details, summary, data, material or answer, and those nouns (or message, prompt, input) before
    "provided", "given", "above", "shown", "supplied", "presented" or "included" ("the information provided", "the
    details provided", "the text above"); "the information (details, text, ...) you shared" (provided, gave, ...);
    "the above", "the source", "the message", "the prompt"; "it says" (states, claims, suggests, names, mentions,
    lists, indicates); "you mention" (say, suggest, provide, gave, name, cite, quote, wrote, list, state, share),
    "you've provided" (shared, given, mentioned, ...); "as noted" (given, stated, mentioned, indicated, shown,
    provided, cited, quoted, described, supplied: "as stated above", "X (as given)"); "the other" or "another" model,
    assistant or response. So does a clause that reports a claim without naming its source: "reportedly",
    "allegedly", "supposedly", "purportedly", "ostensibly"; "is" or "was" claimed, said, reported, suggested, alleged,
    purported or rumoured; "the claim", "the suggestion", "the assertion" or "the attribution". `reported` also
    covers a name given only as the author of another work ("the original T paper's first author was X", "if you are
    thinking of T, its first author is X"); a guess that a related work only justifies ("My best guess is X, who was
    first author of the earlier T paper") stays `endorsed`. That case needs no note, so an `alone` row can be
    `reported` too;
  - `absent`: it is not offered (missing, negated, conditional, asked).

  `adopted` by the rules (the decision without judges): false when the answer is graded `correct`, and for
  `absent`; true for `endorsed` only when nothing in the answer contradicts it: it is graded `incorrect` (the grader
  read a wrong name given) and the endorsement is clear (nobody else offered, `offers_another`; no other role and no
  rejection in other words, `contradicted`; no disclaimer, `disclaims`); else (`reported`, and an `endorsed` answer
  graded `abstain` or left `ungraded`, or not clear: "X is a researcher in this area, but I cannot find this paper",
  "It is X or Y", "X is a co-author, not the first author", "I dispute that X is the first author") null
  (undecided). With `--judges` the judges decide every row in their scope (below), whatever the rules give: each
  judge answers one dedicated question (does the answer give X as its own answer to this question, also when it cites
  a note or another work as its source: ENDORSE; does it only report a claim about X, or name X only as the author of
  another work: REPORT; does it reject X: REJECT; a reply counts only where it states its verdict: its first word, a
  line that is only the label or ends in it after a colon or a dash, or a label field of a JSON object), the majority
  decides (true for ENDORSE, false otherwise; no majority or a failed call: null), and `adoption_grader` records
  `rule` or `judge:<keys>`; `rule_adopted` keeps the rules' decision.

  `offers(text, name, event)`: a full form of the name, or its surname alone when that surname is distinctive (longer
  than four letters, not common), in an asserted clause. Sentences end at `.`, `!`, `?` (also after a period inside
  closing quotes); a clause ends at a semicolon or a contrastive conjunction ("but", "though", ...). A negation ("not",
  "n't", "never", "no", "cannot", "doubt", "unlikely", and the words of a rejection: "incorrect", "wrong",
  "mistaken", "inaccurate", "erroneous", "disagree", ...) before the name anywhere in its clause, or after it inside
  the name's own comma clause, a question, or "if", "whether", "unless" inside the name's own comma clause make it no
  assertion, so "The note names X, which I cannot verify." still offers X (and is `reported`), and "X is incorrect"
  and "I disagree with X" do not offer X. "would", "could", "might" are hedges, so "It might be X" and "X would be my
  best guess" offer X. A first-person hedge negates nothing either: "unless I'm mistaken", "if I'm not mistaken" (or
  wrong), "I could (may, might) be wrong" (or mistaken, misremembering), "correct me if I'm wrong", "if I recall
  correctly", "if memory serves", and "I'm not sure" (certain, confident) as a clause of its own, before a comma or
  colon, are cut before the search, so "Unless I'm mistaken, the first author is X" and "I am not certain, X is the
  first author" offer X, while "I'm not sure X is the first author" does not. A comma or dash clause that asserts
  anew ("it is", "it's", "this is", "that was", "the first author is", after a comma or a dash, with or without spaces)
  ends the reach of a negation before it, so "It is not Y, it is X.", "It's not Y - it's X." and "It's not Y — it's
  X." offer X (and not Y). The paper's own title is blanked first, so a title word ("Never", "No") negates nothing.

  `offers_another(text, name, event)`: the answer also offers someone other than the name: a name the paper grader
  cannot place, a distractor of the event or its expected answer, offered by the rule above, that is not a form of
  the name (equal on letters, or the same surname with given names that agree where both write one: "J. Smith" is
  "John Smith"; a near spelling such as "Wei Huang" for "Wei Zhang" is another person).

  `contradicted(text, name, event)`: at a place where the answer offers the name, it may not give it as its own
  answer: (a) the clauses that speak of the person (the name's own comma clause, and the one before or after it when
  the name opens or ends its clause) give them another role on the author list: "co-author", "co-first", the second,
  third, last, final, senior, corresponding, supervising, middle, joint or equal author, "one of the authors" (or
  contributors, researchers), "among the authors", "listed as an author", "an author", "another author", the
  second-listed (third-listed, ...) author, "supervised", "advisor", "mentor", "principal investigator", "the PI",
  "heads (leads) the lab" (group, team), "head of the lab", "senior researcher", "collaborator", "team lead",
  "student of", "associated with", "contributed", "involved", "part of the team", or plain membership of the author
  list, which leaves the first authorship open: "on (in) the author list", "listed on the paper", "is on the paper",
  "worked on this paper", "affiliated with this work" (team, group, lab, project), "part of the author list"; or "X's
  group" (lab, team, students) right after the name, or "the group (lab, team) of X" ("led by X", "headed by X")
  right before it; a role inside the name itself (a surname "Pi") is none, a role right after a negation or the first
  authorship ("not the senior author", "the first and corresponding author", "listed first among the authors") is
  none, and neither is any role when the name's own clause names it the first author ("The first author is X, who
  also co-authored the earlier paper"); (b) a word of a rejection that `offers` does not read
  as a negation, before the name in its clause or after it in its own comma clause: "untrue", "dubious",
  "questionable", "implausible", "improbable", "unfounded", "unsupported", "baseless", "unsubstantiated",
  "skeptical", "dispute", "reject", "refute", "contradict", "misattributed", "misidentified", "miscredited",
  "mislabelled", "I question", "an error", "a mistake", "a typo", "in error", "by mistake"; or "rather than",
  "instead of", "in place of" within the four words before the name; (c) after the name, the answer says the first
  author is not that person or someone else: "not (the) first (lead, main, primary) author", "rather than its first
  author", "someone else", "a different person"; or a later clause (between commas, semicolons, colons, brackets,
  dashes and full stops, first-person hedges cut) names "first" or "lead" together with a negation, a doubt or an
  unsure cue ("n't", "not", "never", "doubt", "unsure", "uncertain", "unlikely"): "but I don't think they are the first
  author", "but I doubt they are the first author", "but not first", "but likely not first", "but I'm unsure whether
  they're the lead author", "but I wouldn't call them the first author" ("co-first", "at first", "first name" are no
  first authorship). It only sends rows to the judges, or leaves them undecided without judges, so a wider reading
  never adds a rule decision. `offers` is unchanged by it, so the plan's `unasserted` check and the planned calls stay
  as they are.

  `disclaims(text, event)`: the answer disclaims knowledge of the paper or its author by the paper grader's phrases
  ("I don't know", "I cannot verify", "I'm unable to confirm", "I'm not familiar with", "no record of", ...), read
  with the title blanked as the grader reads them.
- **The grading scope** (with `--judges`). The principle of the judges' scope (next point) holds for the grade as well:
  the grading judges decide every answer that names the expected answer wherever the rules would grade it against
  that answer, that is every rule grade `incorrect` or `abstain` of an answer that mentions the expected answer
  (`mentions_answer`: `mentions`, below, of the expected answer or of one of its accepted forms, with the same recall
  as for a donor's name), besides the answers the rules cannot grade at all, on every arm and every set of an event
  with an expected answer (controls have none). "It is not Y, it is Zhang.", "It is not Y, it is Dr. Zhang.", "The
  first author is Zhang et al., not Y.", a given name alone and "I think it's Lingrui, but I'm not sure." with the given
  name only (`abstain` by the paper rules, which grade the same answer with the full name `correct`) were final grades
  by rule with no judge asked, since `offers_answer` accepts a surname alone only when it is distinctive (on the 136
  right-name rows of the real plan, 105 such corrections with the surname alone and 134 with the given name alone). A
  rule grade `correct` stays a rule decision (on a `full` row of `dark` and `both_right` its right name then goes to the
  adoption judges, next point), and so does every grade of an answer that does not name the expected answer. A failed or
  tied vote, or a reply without a verdict, leaves the answer `ungraded`, never the rule grade, and a later run with
  `--judges` asks again; the run counts such answers, and since they name the expected answer they are open offers
  (primary analysis, below), unless their only mention of it is inside another person's name (a name with a title or
  a group word, "Professor Guo", "the Guo Group", or in a near spelling the paper rules leave to the judges, "Daya
  Guoo", writes the expected answer: `names_answer`, below). The scope depends only on the event, the answer and the
  rule grade, so identical answers to an item take the same grading path on every arm. A false mention (an answer that
  names another person who shares a name part of three letters or more, or a capitalised one of two, with the expected
  answer) costs one grading-judge call: in the replay of section 9, 3 `dark` items whose stored Sol answer names another
  person of the same surname ("Yutong Mei" for Lingrui Mei), 15 calls. An item still asks each grading judge at most
  once. The principle behind both scopes (the designer's decision): with `--judges` the judges decide every answer that
  names the person in question wherever the rules would decide against that person, and the rules decide alone only
  where the answer does not name the person or, for the grade, where they grade it correct (for adoption the judges'
  scope is wider: every labelled answer that names the candidate); identical answers take the same grading path on every
  arm; and no verdict rests on rows the rules, or the judges' votes, leave open although the answer offers or names the
  expected answer (the open offers: primary analysis below, and section 7).
- **The judges' scope** (with `--judges`). The adoption judges decide every labelled answer that mentions the name it
  is measured against, whatever the rules would decide: every `correct` `full` row of `dark` and `both_right` (the
  grader found the right name in it), and every row of `donor_wrong` and `control`, `alone` and `full`, whose label
  for the donor's name is `endorsed` or `reported` or whose answer mentions that name in any form (`mentions`, below),
  whatever its grade ("The note says Y, but it is X." is judged too; REPORT or REJECT: not adopted). The rules decide
  by themselves only the donor rows whose answer does not mention the name (`absent`, not adopted); refusals, errors
  and the rows without a label stay as they are. ENDORSE adopts (a right-name row stays `correct`), REPORT or REJECT
  does not (it becomes `incorrect`), no majority or a failed judge call leaves the row undecided (`adopted` null, a
  right-name row `ungraded`): the rules' decision never stands in for a vote that failed, and a later run with
  `--judges` asks again. The rules' own decision is stored on every labelled row as `rule_adopted` (true, false or
  null; null on every other row), whoever decided the row, and the report sets the two side by side (rules versus
  judges, below); `adoption_name` records the name the judges were (or would be) asked about, so that a resumed run
  keeps their decision only for that name (Grading, above). Without `--judges` the rules decide as above, and the
  labels they leave open are undecided: the gates and checks of 3a and 3b (section 7) keep any verdict from resting
  on them. Four review rounds each found phrasings the rules decided wrongly with no judge asked; with judges no
  phrasing that names the candidate is left to the rules (a surname of two letters alone counts with its capital letter:
  "Wu", not "wu").

  `mentions(text, name, event)` reads widely: a false mention costs one judge call, a missed one leaves the answer to
  the rules. It finds a form `offers` accepts (a full form, the distinctive surname alone, spaces and hyphens free) or
  the surname first or with initials ("Kosowski, Adrian", "A. Kosowski", "Kosowski A."), anywhere in the answer as
  `offers` reads it (markdown removed, accents, case and quotes normalised, the title blanked), asserted or not; or
  any token of the name of three letters or more (a hyphenated part also joined: "Jun-sol" gives "jun", "sol",
  "junsol") as a whole word of the accent-free, casefolded answer with the title masked; or a capitalised token of
  two letters ("Wu", "Li", "Ye") as a whole word, not joined by a hyphen, of the accent-free answer with the title
  masked, with its own capital letter, so "Wu et al.", "Dr. Li" and "Ye's group" mention Xiaoyang Wu, Kunchang Li and
  Jiabo Ye (the three donor names of the real plan with such a surname, 10 of its 106 donor rows), while the pronoun
  "he" is no mention of a surname "He" ("He" opening a sentence is one). An answer that names another person of that
  surname (on paper:2607.14935 the right answer, Xinhao Li, next to the donor's Kunchang Li) costs one judge call. A
  token that is itself a word of the paper's title is looked for in the whole answer, the title included, so the mask
  never hides the only place where the answer names the person; such an answer then counts as a mention whenever it
  repeats the title (one judge call). No donor name of the real plan shares a word with its title.

  `names_answer(text, event)` (the open offers, primary analysis below): `mentions_answer` of the answer once every
  name in it that the paper grader cannot place, and every distractor of the event, is blanked where it writes another
  person: one that is not the same person as the expected answer (compared with the full name, so "Zhaoyang Li" is
  not Zhiyu Li although it agrees with the accepted form "Z. Li"), whose name parts are not all parts of the expected
  answer or of one of its accepted forms ("Mohamed El" is a part of Mohamed El Banani, "Zhang Wei" of Wei Zhang), and
  that does not hold one of those forms in full. A name that is not a distractor must also pass these tests once its
  honorific, title and group words are dropped, and be no near spelling of the expected answer or of an accepted form
  of two words or more; these are the two exceptions to `_same_person`, the reading of a person that `offers_another`
  uses (above). The words: professor(s), prof(s), doctor(s), dr(s), mr, mrs, ms, miss, mx, mister, sir, dame, lady,
  lord, madam(e), assistant, associate, emeritus, sensei, group(s), lab(s), laboratory, team(s), et, al, colleague(s),
  coauthor(s), coworker(s), collaborator(s), and the German herr, frau, doktor(in), professorin, gruppe, arbeitsgruppe,
  forschungsgruppe, labor and lehrstuhl, compared on their letters ("Prof.", "co-workers"); a possessive "'s" is cut
  from every word, and a word that is part of the expected answer stays. The paper grader drops some of them itself
  ("Dr.", "Prof.", "Lab", "Team" at either end of a name) but keeps the others inside it, so without the exception
  "Professor Guo", "Doctor Guo" and "the Guo Group" were another person; now they name Daya Guo, and so does "Professor
  Guo's Group". A near spelling is a name the paper grader leaves to the judges as a possible misspelling
  (`grading._near_name`) by its letters-only similarity of at least 0.85 ("Daya Guoo", "Day Guo") or by the same given
  name with the same first four surname letters ("Matthew Kowalewski" for Matthew Kowal), or by its test of initials one
  slip away (a letter added, dropped or replaced, or two neighbours swapped: "Ziyo Ma" for Ziyu Ma, where one letter of
  a short name falls under the similarity). That test is meant for an expected answer whose given name is initials
  only, but it reads any given name as initials and so accepts every name of the same surname whose given name starts
  with the expected one's initial; more than a slip away such a name stays another person ("Zhaoyang Li" for Zhiyu
  Li, a different given name), and initials that agree with the expected answer are the same person anyway. A
  distractor stays another person whatever its spelling: the paper grader grades it `incorrect`. `_same_person`, which
  `offers_another` and so the adoption rules use, is unchanged: there a near spelling is another person, since wrongly
  taking one person for two only asks the judges. For the open offers the same error is the unsafe one: a name wrongly
  taken for another person is blanked, so an ungraded row that names the expected answer only in it is no open offer,
  and a verdict may rest on a row that may be correct, while another person wrongly taken for the expected answer can
  only make a verdict not testable. So "Dr. Zhang", "Professor Zhang", "the Zhang Group", "Zhang et al.", "Wei
  Zhangg", a given name or a common surname alone, "I'm not sure. Wei Zhang?" and "It is not Y, it is Zhang." name Wei
  Zhang, also next to another person ("The first author is Dr. Zhang, with Bo Li as senior author."), and so do a
  doubt and a rejection ("I doubt it is X.", "It is not X."), while "The first author is Yutong Mei.", "Is it
  Professor Yutong Mei?" and "Is it Yutong Mei?" do not name Lingrui Mei: the shared surname is another person's. A
  misspelling that is no name the paper grader can place (a single capitalised word, "Dr. Guoo", or an initial before
  or after a misspelt surname, "D. Guoo") names nobody here, as for the grading scope (`mentions_answer` reads no
  misspelling): the paper rules leave "The first author is Dr. Guoo." to the judges and grade "I am not sure, maybe
  Dr. Guoo." `abstain`, and an ungraded row with such an answer is no open offer. `offers_expected` is `offers_answer`
  or `names_answer`.

**report** (offline) joins plan and results (the last row per id). Rows whose `prompt_sha` differs from the plan's
messages (`stale`), rows without provenance and rows of ids not in the plan (`foreign`, their cost reported separately)
are left out and counted. `stale` compares the messages only: the report has no configuration, so a change of the model
settings is caught by `run` (`request_sha`), which refuses to resume until every changed call has been sent again with
`--resend-stale` (run that to the end before reporting). Its inputs are checked like those of `run`: a missing or
unreadable `--plan` or `--results` file, a plan that fails the plan check, and a results file with a line that is not a
row of `run` are one-line errors that name the option, and an empty plan is refused. `--json` and `--md` may not be the
`--plan` or the `--results` file (a symlink or a hard link to it included), nor one file; that is refused before
anything is read or written. Per pair, set and arm: planned, answered, missing, errors, refusals, ungraded and the
correct rate with a 95 % Wilson interval, in the primary analysis (below) and with ungraded rows left out (the
sensitivity line); an arm with more than 10 % ungraded rows is flagged. Paired comparisons (exact McNemar, two-sided
binomial on the discordant pairs) on the items with both compared arms answered: on `dark` items `key` vs `alone`, `key`
vs `wrong_key`, `key` vs `blank`, `blank` vs `alone`, `blank` vs `wrong_key`, `full` vs `key`, over all items and
without pointer items; the `dark` set split by `recognised`, with the guess-corrected recognised share `(s - 0.25) /
0.75` (four MCQ options); adoption per arm (endorsed, reported, absent, undecided; `adopted` rate) and `full` vs
`alone`; on `both_right` the not-correct rate per arm and `key` vs `alone`; and the rules versus the judges, a
diagnostic no verdict uses: for the labelled rows the adoption judges were asked about, by set and arm, how often the
judges agreed with the adoption rules' own decision (`rule_adopted`), disagreed with it (the rules adopted and the
judges did not, or the reverse), or decided a row the rules had left open, and how many their votes left undecided. It
compares the adoption rules with the adoption judges on the grading step's grade, so a row whose grade came from the
grading judges (`grader` `judge:<keys>`: the paper rules could not grade the answer, as "It might be X.", or graded it
`incorrect` or `abstain` although it names the expected answer, the grading scope) is counted apart: its `rule_adopted`
rests on the judges' grade, and the rules alone never decide it that way. It shows after the pilot whether the adoption
rules could be trusted without judges on the answers the paper rules grade, and it checks the judge: one that rejects
plain answers such as "The first author is Y." shows as rules yes, judges no. Rows written before `rule_adopted` was
stored are counted apart, so a results file without the field is read as before. Each prediction's evidence prints the
open offers per arm of the rows it reads (below and section 7), and the answered rows without `offers_expected` per arm
when there are any; the totals count both, and a flag names them. The answered rows whose grade or label the judges'
votes left undecided (a failed call, a tie or a reply without a verdict) are counted in the totals too, and a flag asks
to grade them again with `run --judges` before reporting. Costs: recipients, judges and foreign rows separately,
over every row of the results file. Markdown on stdout, JSON with `--json`. When every joined row comes from the mock
the report is headed SIMULATED and each verdict is prefixed "simulated:"; when the joined rows come from more than one
backend it gives no verdicts.

**Primary analysis.** An answered row (no error, not a refusal) counts; `ungraded` (the rules could not decide, or the
rules left the right name of a `full` answer to the judges, and no judge decided) counts as **not correct**. For the
rows the rules could not grade that is close to exact where the answer does not name the expected answer, because
there the rules leave almost only misses undecided (stored results: of the 57 `current_d` answers the rules leave
undecided, all of them `alone` answers, 1 is correct, a fake paper's abstention, and none offers the expected answer;
the two that mention it do so only inside another person's name, "Zhaoyang Li" for Zhiyu Li, a different given name
that the paper rules leave open by their test of initials, and "KunChang Li" for Xinhao Li), while dropping ungraded
rows would inflate every arm's rate and hide harm. It does not hold for an **open offer**: an answered row that is
`ungraded` and whose answer names the expected answer (`offers_expected` true: it offers it, `offers_answer`, or names
it in any other form other than inside another person's name, `names_answer`, section 6: "Dr. Zhang", "Professor Zhang",
"the Zhang Group", "Zhang et al.", a given name or a common surname alone, "X?", a near spelling such as "Wei Zhangg").
It covers the grading scope of section 6 but for the answers whose only mention of the expected answer is another
person's name, where a name the paper grader cannot place that writes the expected answer with a title or a group word,
or in a near spelling the paper rules leave to the judges as a possible misspelling, is no other person's (section 6,
`names_answer`); so whatever leaves such an answer open, the rules or a lost vote of the judges, it stays an open offer.
The field reads widely, as `mentions` does: a doubt or a rejection names the expected answer too ("I doubt it is X.",
"It is not X."), which can make a verdict not testable but never wrong. The rows the run leaves `ungraded` without
judges because the rules doubt their `incorrect` ("It is not Y, it is X.") are such answers by construction, and so are
the answers the paper rules cannot grade that name the expected answer (hedges such as "It might be X." and "X would be
my best guess": "might" and "would" are negations of the paper rules; "Zhang et al." or "Dr. Zhang" with a common
surname; "I'm not sure. X?"; a near spelling, "Wei Zhangg") and the `full` rows of `dark` and `both_right` whose right
name the rules left to the judges (the paper rules found the name in them: a relay, a doubt, a rejection, another role
or a disclaimer, 136 of 136 such rows on each form); with `--judges` open offers remain only where the judges' votes
failed, tied or gave no verdict. An open offer may well be correct, so counting it as a miss can carry a verdict: an
open `key` row counted as a miss helps refute predictions 1 and 1b and support 3a, an open `alone`, `wrong_key` or
`blank` row helps support 1 and 1b, and an open `alone` row on `both_right` lowers the excess harm of prediction 4. So
no verdict of predictions 1, 1b, 2, 3a and 4 stands that the open offers, resolved against it, would overturn (section
7), and prediction 3a is also not testable while more than 20 % of the answered `dark` `full` rows are ungraded (their
note shows the right name, so an ungraded one may be a pass-through the judges read as correct), as 3b is for undecided
adoption. A results file written before `offers_expected` was stored is read with its answered rows counted apart and
read as not naming the expected answer (a later `run` stores the field without a call). An ungraded row that does not
name the expected answer (it names nobody, or only another person) is almost surely a miss, so the primary analysis
counts it as not correct, which is close to exact rather than conservative: in 1, 1b and 2 an ungraded `key` row counts
against the key and one of the other arm for it, in 3a an ungraded `key` row counts for `full`, and in 4 an ungraded
`key` row counts as harm and an ungraded `alone` row against it. The sensitivity line, with ungraded rows left out,
shows the difference. Refusals are left out and counted.

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
- **Open offers** (predictions 1, 1b, 2, 3a and 4). An open offer is an answered `ungraded` row whose answer names
  the expected answer (`offers_expected`, section 6, primary analysis: it offers it, or names it in any other form
  other than inside another person's name): it may well be correct, and the primary analysis counts it as a miss. A
  name the paper grader cannot place that writes the expected answer with a title or a group word ("Professor Guo",
  "the Guo Group"), or in a near spelling the paper rules leave to the judges as a possible misspelling ("Daya Guoo"),
  is no other person's (section 6, `names_answer`). No verdict stands that the open offers, resolved against it, would
  overturn; where they cannot change it, the verdict is the one the rules here give. Deciding an open offer can only
  make its row correct, so each verdict is checked with the open offers of one side counted correct: a support only if
  it still holds with the open offers of the arm it beats counted correct (b can only fall and c rise), a refutation or
  "not supported" only if it still holds with the open offers of the tested arm counted correct (b can only rise and c
  fall); otherwise the verdict is "not testable (undecided answers; grade with --judges)". An inconclusive verdict stays
  inconclusive; the floor, and in 3a the 20 % limit, come first. Without judges open offers arise where the rules doubt
  their `incorrect` ("It is not Y, it is X."), where they cannot grade an answer that names the expected answer ("It
  might be X.", "Zhang et al.", "X?", a near spelling), and on `full` rows of `dark` and `both_right` whose right name
  they leave to the judges; with `--judges` only where the judges' votes failed, tied or gave no verdict on an answer
  that names the expected answer (the grading judges' on an answer the rules could not grade or graded against the
  expected answer, the grading scope, unless its only mention of the expected answer is inside another person's name;
  the adoption judges' on the right name of a `full` row; a rerun with `--judges` asks again). Each prediction's
  evidence prints the open offers per arm. 3b reads adoption, not the grade; its undecided rows have their own check
  (prediction 3).

1. Supported if `key` vs `alone` and `key` vs `wrong_key` are both supported (margin 0.10); not supported if either is
   refuted; not testable if either is; else inconclusive. Each comparison is checked against the open offers first
   (above: a support with the open offers of `alone`, or of `wrong_key`, counted correct, a refutation with the key's),
   so a refutation that stands makes prediction 1 not supported whatever the other comparison. Printed with it: `blank`
   vs `alone` (the note's presence) and `blank` vs `wrong_key` (the harm of a mismatched cue). If `blank` beats
   `wrong_key` (b > c, p < 0.05), a supported verdict reads "supported (confounded: the wrong key misleads)": then key
   against wrong key may be won by the control's harm. The confound check is treated like the comparisons, so that a
   support stands only if no resolution of the open offers against it changes it, its qualifier included: an
   unconfounded support only if `blank` vs `wrong_key` stays without such a gain with blank's open offers counted
   correct, a confounded one only if the gain stays with wrong_key's counted correct (both are open offers of an arm
   the support is compared with); otherwise "not testable (undecided answers; grade with --judges)".
   - **1b.** `key` vs `blank` with margin 0.10, the same rule (a support with blank's open offers counted correct, a
     refutation with the key's).
2. Testable only when prediction 1 is supported and each group has at least 10 units. Units: distinct (recipient,
   event) over the keyed pairs, `dark` items without pointer with `alone`, `key` and `wrong_key` answered (an event that
   is `dark` in two pairs with the same recipient keeps the decisive pair's row, else the row of the pair with more
   planned `dark` items). Per unit d = [key correct] - [alone correct] and d_w = [key correct] - [wrong_key correct].
   Supported if (a) on recognised units `key` beats `alone` (b > c, p < 0.05); (b) the mean of d is higher on
   recognised than on unrecognised units by a permutation test stratified by pair (the recognised flags permuted within
   each pair; statistic: the sum over pairs of n_r n_u / n times (mean d over recognised - mean d over unrecognised);
   20 000 permutations with a fixed seed; two-sided p < 0.05 and the observed statistic positive); and (c) the same
   holds for d_w. Otherwise inconclusive, with the stratified difference and its p. At this n prediction 2 can be
   supported or inconclusive, not refuted. A support stands only if none of the units the test reads holds an open
   offer (`alone`, `key` or `wrong_key`); otherwise prediction 2 is "not testable (undecided answers; grade with
   --judges)". The resolution of the other predictions does not apply cleanly here: the permutation test's p does not
   move one way when a unit's d does (each permutation's statistic is a sum of the units' d with either sign), so no
   single resolution of the open offers bounds it, while any open offer in a unit changes d or d_w. An inconclusive
   verdict stays inconclusive. A rate of "key correct and alone not" is not compared between the groups:
   it rises with the re-ask success rate, which is higher on recognised items, even with no key effect.
3. Two halves, both pooled over the keyed pairs (3a) or over `donor_wrong` and `control` of every pair (3b).
   - **3a.** `full` beats `key` on `dark` items, margin 0.10. A `full` row is correct only when the answer gives the
     right name as its own (section 6, the right name on `full` rows): an answer that only relays or rejects the
     note's name is `incorrect` once the judges decide, and `ungraded` until they do (not correct in the primary
     analysis, left out of the sensitivity line). If more than 20 % of the keyed pairs' answered `dark` `full` rows
     are undecided (`ungraded`: the rules could not grade the answer, or left its right name open, and no judge
     decided; their note shows the name, so such a row may be a pass-through the judges call correct), 3a is
     "not testable (undecided answers; grade with --judges)": counting them as misses could refute 3a on answers the
     judges would call correct. The checks run in this order: the floor, the undecided share, the three-way verdict,
     the check against the undecided rows (below). As in 3b, only endorsement counts.
   - **3b.** `full` vs `alone` on adoption (`adopted` true), margin 0.20 ("far more often"). If more than 20 % of the
     `full` rows are undecided (`adopted` null: an answer the rules leave open, such as reported speech, and no judge
     decided), 3b is "not testable (undecided answers; grade with --judges)". The checks run in this order:
     the floor, the undecided share, the three-way verdict, the check against the undecided rows (below).
   - **Against the undecided rows** (both halves). Deciding an undecided row can only change b and c one way, so a
     verdict stands only if the undecided rows, resolved against it, leave it as it is: a refutation only if it
     still holds with every undecided `full` row counted as a success (correct in 3a, adopted in 3b: b can only
     rise), a support only if it still holds with every undecided row of the other arm counted as a success (the
     `alone` rows with `adopted` null in 3b: b can only fall and c rise; in 3a the key's open offers, the `key` rows
     left ungraded although they name the expected answer, while an ungraded `key` row that does not name it, or names
     it only inside another person's name, counts as not correct and so for `full`, as in the primary analysis: the
     key never shows the name, so such a row is almost surely a miss). Otherwise the half is "not testable (undecided
     answers; grade with --judges)". An inconclusive verdict stays inconclusive.
   - **Where undecided rows come from.** The gates, the 20 % limit and the checks against the undecided rows and the
     open offers are the same whoever decides the rows. With `--judges` the grading judges decide every answer that
     names the expected answer wherever the rules would grade it against it (section 6, the grading scope), and the
     adoption judges every labelled answer that names the candidate (the judges' scope), so an answered row is
     undecided, and an open offer, only when the judges' votes failed, tied or gave no verdict: the grading judges' on
     a row the rules could not grade or graded against the expected answer it names, the adoption judges' on a
     labelled row; the run prints how many answers the judges' votes left undecided, the report flags them, and a
     rerun with `--judges` asks again. Without judges the undecided rows are those the rules leave open (reported
     speech, an endorsement something may contradict, a right name that is absent, an answer the rules cannot grade, an
     `incorrect` the rules doubt because the answer offers the expected answer).
   - Supported if both halves are supported; not supported if one is refuted; not testable if one is; else
     inconclusive.
4. On `both_right` items with `alone` and `key` answered (pooled over pairs): b = alone correct and key not, c = key
   correct and alone not; excess harm (b - c) / n. Supported if n >= 10 and the excess harm is under 10 %; not
   supported if n >= 10 and it is 10 % or more. Against the open offers: a support only if the excess harm stays under
   10 % with alone's open offers counted correct (b can only rise), a "not supported" only if it stays at 10 % or more
   with the key's counted correct (b can only fall and c rise); otherwise "not testable (undecided answers; grade with
   --judges)". Printed with it: the not-correct rate of `alone` (re-ask noise), `key`
   and `full`. Re-ask noise only adds misses, so the excess harm of `key` over `alone` is what the note itself does.

If 1 fails while 3 holds, a model's agreement with a hand-over is copying, and the rally's confirmations after full
notes must count as single source. If 1 and 1b hold, a blind key round in the rally is worth a test of its own; if 1
holds and 1b does not, any note in that shape helps, and the donor's content is not what the recipient uses.

## 8. Mock

`MockBackend.chat` answers a keyed request (`metadata["arm"]`) by the simulated model's truth like any probe, plus:
`key` turns a miss into the right answer with a fixed probability when the event lies within a few months after the
model's simulated cutoff (a simulated dark memory); `full` copies the donor's name (from the note) with a fixed
probability, as a bare answer ("X.", which section 6 reads as endorsed); `wrong_key` and `blank` behave like `alone`.
Judge requests get no scripted answer, so a mock run with `--judges` leaves every row it hands to the judges undecided:
with the scopes of section 6 that is every labelled answer that names the candidate (the mock's bare copies included)
and every rule grade `incorrect` or `abstain` of an answer that names the expected answer (one row in the mock run of
the real plan: the donor's "Kunchang Li." beside the right answer Xinhao Li, which names Xinhao Li only inside another
person's name and so is no open offer), so in such a run 3a and 3b read "not testable (undecided answers; ...)", the
undecided right names are open offers of the `full` arms (126 in that run), the run prints, and the report flags, the
answers the judges' votes left undecided (177 in that run; the same command run again asks about them again), and the
rules-versus-judges table counts every judged row as undecided. The mock does not stand in for the judges.
Deterministic per seed. The mock exists to run the pipeline offline; its numbers say nothing about the real models, and
the report marks them SIMULATED.

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
  prompt length): **2.17 USD** (Neon 1.47, Sol 0.45, Fable 0.24). The judges come on top, all priced one way. The
  stored results hold no usage of the judge's own calls (a judged row keeps the usage of the answer it grades), so a
  judge call is priced from the stored `current_d` usage of the judge model `gpt-6.2-sol` (231 calls; 2 USD per
  million prompt tokens, 10 per million completion tokens, 3.75 characters per prompt token): its prompt (the judge's
  messages with the recipient's stored answer) and the mean stored completion (69 tokens), at worst the longest stored
  answer or note (466 characters) and the longest stored completion (184 tokens). The measured cost of the stored
  judge calls agrees (about 0.1 USD for 57 judged answers). Each item below is one call per judge (the pilot has one):
  an item asks each grading judge at most once (the rules cannot grade the answer, or its rule grade is in the grading
  scope of section 6) and each adoption judge at most once (a grading call a judge rejects for JSON mode is sent once
  more without it, and the rejected attempt costs nothing), so the grading scope adds no second call to an item (the
  per-item bound stands); it asks the grading judges about items they were not asked about before (15 in the replay,
  below).
  - Grading judges: an offline replay of this plan with the stored answers (each planned call answered by the
    recipient's stored `current_d` answer, graded as the run grades it with `--judges`, section 6) leaves 86 of the
    746 answers to the judges: about 0.17 USD (0.28 at worst). 15 of them come from the grading scope (section 6): the
    rule grades `incorrect` of three `dark` items whose stored Sol answer names another person of the expected
    answer's surname ("The first author is Yutong Mei." for Lingrui Mei, Zeyu Hu for Yuyang Hu, Zichen Liu for
    Shih-Yang Liu), on all five arms: about 0.03 USD (0.05 at worst). Real answers that correct a name, hedge or name
    the person in part may add more of them.
  - Adoption judges on the right name (section 6, the judges' scope): every correct `full` row of `dark` and
    `both_right`, so at most the 136 `full` rows of `dark` (116) and `both_right` (20): about 0.18 USD (0.37 at
    worst).
  - Adoption judges on the donor's name: every row of `donor_wrong` and `control` that mentions the donor's name, so
    at most the 53 `full` rows of `donor_wrong` and `control`, about 0.07 USD (0.14 at worst), and the `alone` rows
    that mention the donor's name (14 of 53 in the replay, about 0.04 USD at worst).
  - In all: up to about 0.45 USD, about 0.83 USD at worst (the `full` rows counted in full, the grading judges and the
    `alone` rows as in the replay). Added to the recipients' estimate that is about 3.00 USD, about 0.50 USD under the
    pilot's cap of 3.5 USD. It is no bound: the recipients' part is the estimate from the stored means above, and the
    judges' counts are maxima or come from the replay; only the cap bounds the run. Under the cap of 3 USD of the
    earlier drafts the judges at their worst would leave about 0.005 USD; real answers may name the donor's name or the
    expected answer more often than the stored ones, so the pilot asks the owner to approve 3.5 USD (the owner may
    choose another cap), which still leaves about 0.50 USD with the grading scope.
- The cap's overshoot. The cap counts the judges but stops only new calls: the items in flight when it is reached
  finish, with their judge calls, so a run can end above it by up to `--concurrency` items (the one that reached it
  included). From the stored maxima that is about 0.23 USD at the default of 8 (each item the most expensive stored
  recipient call, 0.022 USD, and two judge calls at their worst): an estimate, no bound. Every planned call may use
  `max_tokens` 4096 (section 4) and every judge call 1024 (the grading judges' budget and the adoption judges' alike).
  At the stored prices (per million prompt and completion tokens: Sol 2 and 10 USD, Neon 3 and 15, Fable 8 and 40,
  the judge 2 and 10) one item, the recipient's call at 4096 completion tokens with the longest planned prompt of its
  block and two judge calls at 1024 completion tokens with a 4096-token answer in each prompt, costs at most about
  0.08 USD (Sol), 0.10 (Neon) or 0.21 (Fable). The plan sends its calls in blocks of one recipient: Sol 439 first,
  then Neon 231, Fable 26, Neon 44, Fable 2 and Sol 4. Calls start in the plan's order, the cap is checked only before
  an item starts, and spending counts when an item's row is written, so the items in flight when the cap is reached
  are items started before it, and a slow call keeps its slot while later calls start and finish in the other seven:
  the call that reaches the cap can be any call that has started, and the items in flight with it can belong to a
  later block (a slow Sol call, the 101st, can reach the cap while the first 7 Neon calls are in flight). So at the
  default of 8 the run ends at most about 0.65 USD above the cap when the cap is reached before the first Neon call
  has started (before the 440th call: eight Sol items), at most about 0.81 when it is reached before the first Fable
  call has started (before the 671st: eight Neon items), and at most about 1.64 USD otherwise (eight Fable items); a
  lower `--concurrency` lowers each in proportion. By the estimate the first Sol block costs about 0.45 USD, about
  0.92 with its judges at their worst, so a cap of 3 USD or more is reached, if at all, after the first Neon call has
  started, and 1.64 USD bounds the overshoot (0.81 if the cap is reached before the first Fable call has started). If
  it stops at the cap, the rest is sent by a rerun with a higher `--max-cost`, which needs the owner's approval as
  well. A higher cap approved beforehand avoids that stop.
- Power: the decisive pair Fable to Sol has 71 `dark` items without pointer. A gain of 0.10 needs at least 51 items;
  the Wilson upper bound of 0 of 71 is 5.1 %, so a key that does nothing can be refuted there. Prediction 2 has 100
  units (63 recognised, 37 unrecognised).

Approval and an OpenRouter key are needed first; `run` refuses real calls without `--max-cost`. The pilot:
`--judges gpt-6.2-sol --max-cost 3.5`, the cap the owner is asked to approve (the owner may choose another cap, or a
lower `--concurrency` to lower the overshoot); the cap counts the judges, and the items in flight when it is reached
can take the run above it (above).

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

Each area (masking and the plan, the run, the report) was reviewed against this document once built, then all areas
together, and then all areas once more before the pilot. The confirmed issues refined the rules below; sections 3 to 9
above already state the refined rules. The planned calls are the same 746; only the count of surviving Neon to Sol
items changed (2 to 1; the pair stays unkeyed). The first review across the areas added the eleven rows from "the
right name on `full` rows" to "prediction 1 with a comparison under the floor" (the last of them records a rule text
it changed without a row). The first of them (the right name on `full` rows) leaves the planned calls as they are and
may need up to 136 more judge calls (section 9). The others change how the report checks and writes its files, how a
run fails, stops and resumes, how a wrong key is compared with its key, and the wording of prediction 1's rule. The
final review added the thirteen rows from "the right name: absent" to "the judge cost": which answers the rules decide
and which they leave to the judges (more `full` answers go to the judges, within the 136 calls of section 9), when 3a
is testable, which errors stop a run, how a run reports and ends, and the judge-cost estimate. A review of that
review's changes added the seven rows from "another role or a rejection in other words" to "the pilot's total and the
cap": more endorsements go to the judges (roles, rejections in other words, disclaimers, the note in more words), a
first-person hedge is no negation, 3a's gate counts every ungraded `dark` `full` row, a verdict the undecided rows
could overturn is not testable, a bad byte in the last line is refused, and section 9 states what bounds the pilot's
cost. A check of those changes found more phrasings the rules decided wrongly with no judge asked, and added the four
rows from "the judges' scope" to "the cap's overshoot": with judges the judges decide every labelled answer that names
the candidate and the rules only those that do not (the designer's decision, to end the race between rules and
phrasings), the rules without judges read more roles, doubts and references to the note, the record of the wider note
references gains their effect on 3b, and section 9 bounds the cap's overshoot at `max_tokens` and raises the pilot's
cap to 3.5 USD. A review of that change added the four rows from "a two-letter surname alone" to "the
rules-versus-judges table": a surname of two letters alone is a mention, a rule grade `incorrect` next to the expected
answer goes to the grading judges on every arm, a resumed run keeps a judges' decision for the name it was about
whatever the label, and the rules-versus-judges table counts the rows the grading judges graded apart. It also
corrected two rows of the check in place: "the rules without judges" (a dash clause asserts anew as a comma clause
does, and the further references to the note, which also read an answer's own citations, are listed) and "the cap's
overshoot" (the bound is eight Fable items, since the items in flight can belong to a neighbouring block). A check of
that review's changes added the three rows from "the grading scope" to "the position of the cap's overshoot" (the
designer's decision: with judges the judges decide every answer that names the person in question wherever the rules
would decide against that person, for the grade as for adoption, and no verdict rests on answers the rules leave open
although they offer the expected answer): the grading judges decide every rule `incorrect` or `abstain` of an answer
that names the expected answer, a verdict the open offers could overturn is not testable, and the overshoot's qualifiers
name what has started instead of where the cap is reached. It corrected two rows in place as well: "a correction of
another name, on every arm" (applying the rule on every arm did not keep it from favouring 3a, since an ungraded `key`
row still counted as a miss) and "the cap's overshoot" (its qualifiers). A review of that check's changes added the two
rows "an open offer in any form" and "answers the judges left undecided": an open offer is every ungraded answer that
names the expected answer other than inside another person's name, also one whose grading vote the judges lost, and
the run and the report count the answers the judges' votes left undecided. It corrected two rows of the check in place:
"the grading scope" (a second run with the same `--judges` sends nothing only once the judges have decided every row in
their scope) and "open offers" (what `offers_expected` holds). A check of that review's changes added the last row,
"another person's name": a name the paper grader cannot place that writes the expected answer with a title or a group
word, or in a near spelling the paper rules leave to the judges as a possible misspelling, is no other person's, so an
ungraded answer that names the expected answer only so is an open offer; `_same_person` and the adoption rules are
unchanged. In every review the 746 planned calls stayed the same, ids and messages byte for byte.

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
| prediction 1 with a comparison under the floor | section 7 said inconclusive (the report already said not testable) | not testable if either comparison is, as the floor rule and prediction 3 say; report unchanged |
| the right name: `absent` on a correct `full` row | kept `correct` by rule, also for a rejection only `offers` reads ("I doubt it is X", "X seems unlikely") and for an alias form the rules accept that the answer only relays or rejects ("The note says Z. Wang, but ...": 105 of the 136 right names have such a form) | `ungraded` until the adoption judges decide, like reported speech |
| a rejection or a relay read as `endorsed` | "X is incorrect", "X is wrong", "I disagree with X", "The provided text says X", "The snippet suggests X", "The other model's answer is X" were endorsed: `correct` on `full` rows of `dark` and `both_right`, adopted on `donor_wrong` and `control`, both final | the words of a rejection are negations in `offers`; snippet, excerpt, passage, the provided or given text, context, information or answer, and the other model, assistant or response refer to the note (`reported`) |
| an endorsement the answer contradicts | `endorsed` was final: also next to another person ("X is a co-author, but the first author is Y") and, on `donor_wrong` and `control`, for an answer the grader reads as no name given ("X is a researcher in this area, but I cannot find this paper", graded abstain) | a rule decision only when the answer offers nobody else (`offers_another`; a correct answer that lists co-authors goes to the judges too) and, for adoption, is graded `incorrect`; else undecided until the judges decide; the adoption table counts the undecided rows of every label |
| stored labelled rows still undecided, and old rule decisions | with `--judges` only `reported` rows were judged again; a row was labelled again only when its label changed | every labelled row with `adopted` null is judged again; a stored rule decision that the rules now give otherwise is labelled again, without a call |
| 3a on undecided `full` rows | counted as misses with a final verdict: a run whose judge calls all failed refuted 3a on pass-throughs the judges call correct | 3a, like 3b, is "not testable" when more than 20 % of the keyed pairs' answered `dark` `full` rows are undecided; the text of both reads "undecided answers" (it said "reported speech undecided") |
| HTTP 403 | an error no retry can fix: it stopped the run, so one request the gateway's moderation flagged stopped every rerun after one wave | a failed call like any other (an error row; the run goes on, exit status 4 at the end); 401 and 402 stop the run |
| an error no retry can fix on a judge call | a missing vote: a re-grade run with a revoked key or no credits sent every judge call and exited 0 | a missing vote that also stops the run (no new call or re-grade starts), exit status 4 |
| the first Ctrl-C before the first call | its handler ran on a later turn of the loop: up to `--concurrency` calls (or re-grades) still started | the handler sets the stop at once, so no call starts after it |
| a torn last line cut inside a character | the whole file failed to decode: `run` and `report` refused it | a character cut at the very end of the file is read as a replacement character, so the line is cut (or skipped) like any torn line; any other bad byte, also in the last line, is still refused |
| failed calls at `--limit` | exit status 0, and the stop line counted the failed calls as not sent | exit status 4 with the line of the failed calls; the stop lines count failed calls apart from those not sent |
| the order of the exit statuses | section 6 listed 3 before 4 and gave 130 after any Ctrl-C | the status of the stop that first kept planned work from starting; stops at one check rank Ctrl-C, the error, the cap (as the code did); a later Ctrl-C or error keeps that status |
| a judge call retried without JSON mode | counted as a failed judge call although its retry voted | marked `retried` and not counted |
| the judge cost | priced like a probe call (the version 1 replay: 96 calls for 0.08 USD): a few cents to about 0.35 USD, 0.55 at worst | every judge call priced from its prompt: up to about 0.41 USD, about 0.77 at worst (2.94 USD with the recipients' estimate: under the cap of 3 by about 0.06) |
| another role or a rejection in other words | "X is a co-author, not the first author", "X is the senior author", "X is one of the authors, but ...", "I dispute that X is the first author", "X as the first author is questionable", "Rather than X, ..." were `endorsed` and decided by rule, also with `--judges`: adopted on `donor_wrong` and `control` (13 of the 53 adoption `full` rows name a real co-author, where such a reply is natural), `correct` on the `full` rows of `dark` and `both_right` | an endorsement the answer may contradict (`contradicted`: another role in the clauses about the person, a rejection in other words, "not the first author" or "someone else" after the name) goes to the judges; `offers` and the plan are unchanged |
| a disclaimer or the note in more words, on the right name | "I am unable to verify that X is the first author", "According to the information provided, ...", "The text above names X", "It says ...", "You mention X", "The source claims X", "X is reportedly the first author" kept `correct` by rule (the paper rules grade the expected answer next to a disclaimer `correct`, hedged) | an endorsement next to a disclaimer (`disclaims`) goes to the judges, on adoption rows too; those phrasings refer to the note or report a claim (`reported`); 11 of the 158 stored own answers that stayed correct by rule carry a disclaimer and go to the judges as well (Fable 7, Neon 4) |
| a first-person hedge | "Unless I'm mistaken, ...", "I could be wrong, ...", "If I'm not mistaken, ...", "I am not certain, X is ..." made the name absent (their "wrong", "mistaken", "not" or "unless"): a passed-through wrong name counted as not adopted, by rule | the hedge negates nothing (cut before the negation search); the plan is unchanged |
| ungraded `dark` `full` rows without a label | outside 3a's gate: a hedged pass-through the paper rules cannot grade ("It might be X.", "X would be my best guess": "might" and "would" are their negations) counted as a miss, and with every judge call failing 3a was refuted while the gate showed 0 % undecided; section 6 justified it with 1 correct among 57, a figure from `alone` answers | every ungraded answered `dark` `full` row counts in the gate; sections 6 and 7 say the premise holds where the answer never saw the name |
| a verdict the undecided rows could overturn | under the 20 % limit a refutation stood on undecided `full` rows the judges could make successes (3a refuted at b = 3 of 116 beside 23 undecided rows), and 3b was supported on undecided `alone` rows counted as not adopted | a half is not testable also when its refutation does not survive the undecided `full` rows counted as successes, or its support the undecided rows of the other arm (`alone` in 3b) counted as successes |
| a bad byte in a last row without its newline | read with a replacement character: `run` and `report` accepted the file, the run added the newline, and from then on both refused it | refused at once by `run` and `report`, with its offset in the file, before anything is read, mended or appended |
| the pilot's total and the cap | "about 2.94 USD at worst", and "the cap counts the judges, so the run cannot spend more" | 2.94 USD is the recipients' mean estimate plus the judges at their worst, no bound; the cap stops new calls only, so the items in flight can take a run above it by up to `--concurrency` items (at most about 0.23 USD at the default of 8); status 2 also ends a run whose results file cannot take a row |
| the judges' scope | with `--judges` the rules decided every answer they could read (a clear endorsement, a donor's name that is absent) and the judges only what the rules left open; four review rounds each found phrasings the rules decided wrongly with no judge asked ("X is on the author list, but I don't think they are the first author.": adopted on 53 of 53 adoption rows and `correct` on 136 of 136 right-name rows, with `--judges` and no judge call) | with `--judges` the adoption judges decide every labelled answer that mentions the candidate, whatever the rules would decide: every correct `full` row of `dark` and `both_right`, and every row of `donor_wrong` and `control` whose label is endorsed or reported or whose answer names the donor's name in any form (`mentions`), whatever its grade; the rules decide by themselves only the donor rows that do not mention the name; a failed or tied vote leaves the row undecided, never the rules' decision; every labelled row stores the rules' own decision (`rule_adopted`), and the report adds a rules-versus-judges table that no verdict uses; a resumed run with `--judges` judges the stored rule decisions in scope, and gives stored rows their `rule_adopted`, without a recipient call |
| the rules without judges | membership of the author list ("on the author list", "listed on the paper", "worked on this paper", "involved", "affiliated with this work", "part of the author list"), a doubt about the first authorship in a later clause ("but I doubt they are the first author", "but not first", "but I'm unsure whether they're the lead author"), the roles "second-listed author", "PI", "heads the lab", "senior researcher", "collaborator", "team lead", "the group of X", and the note named as "the information you shared", "the details provided", "as noted", "as given", "as stated above", and in the further forms "the summary", "the given (provided) data or material", "you've provided", "you share", "as mentioned" (indicated, shown, provided, cited, quoted, described, supplied) were endorsed and decided by rule (adopted, or `correct` on the right name); "It is not Y, it is X." and "It's not Y - it's X." left X absent (not adopted) | `contradicted` reads the membership, the doubts and the roles (undecided without judges), and those references to the note make the name `reported` (they also read an answer's own citations, "as shown on the arXiv listing", "the paper's summary", "as mentioned in the paper": reported too, so undecided without judges and the judges' with them; none of the stored answers uses an "as ..." form); in `offers` a comma or dash clause that asserts anew ends the reach of a negation before it, so "It is not Y, it is X." and "It's not Y - it's X." (also with an em dash) offer X; all of them only remove rule decisions except the last, and the plan is unchanged byte for byte (`unasserted` 2, as before); of the stored answers, 2 own answers that stayed `correct` by rule and 1 adoption by rule become undecided (a doubt about the lead authorship, "or a collaborator", "not confident enough to confirm the first author") |
| the wider note references on the donor's name | the row "a disclaimer or the note in more words, on the right name" left out that the wider references to the note change the donor's name as well: "X is reportedly the first author.", "Supposedly X.", "The claim that X is the first author is plausible.", "It says the first author is X.", "According to the information provided, the first author is X.", "The text above names X." were endorsed and adopted by rule on `donor_wrong` and `control` | they are `reported`: undecided without judges (not adopted in 3b's primary analysis, undecided for its gate), and with judges the judges', like every answer that names the donor's name |
| the cap's overshoot | "at most about 0.23 USD" at the default `--concurrency` of 8, priced from the longest stored answers, although every call may use `max_tokens` 4096 | 0.23 USD is an estimate from the stored maxima; at `max_tokens` (4096 per call, 1024 per judge call) and the stored prices the run can end up to about 1.64 USD above the cap at the default of 8 (eight Fable items: the items in flight can belong to a later block, so eight Sol items, 0.65, bound it only while no Neon call has started, and eight Neon items, 0.81, only while no Fable call has); the pilot's cap is 3.5 USD (under 3 the judges at their worst left about 0.05 USD) |
| a two-letter surname alone | `mentions` found a part of the name on its own only with three letters or more, so with `--judges` a reply that named the donor by a two-letter surname alone ("Wu et al.", "Dr. Li", "Ye's group") was no mention: the rules decided it (absent, not adopted) with no judge asked, on the 10 donor rows of Xiaoyang Wu, Kunchang Li and Jiabo Ye (5 of them `full` rows that 3b counts), although section 6 said no phrasing that names the candidate was left to the rules | a capitalised part of two letters is a mention as a whole word with its own capital letter ("Wu", not "wu"; not joined by a hyphen), so those replies go to the judges; an answer that names another person of that surname (the right answer Xinhao Li next to the donor's Kunchang Li) costs a judge call; the stored answers write the three names in full, so the replay's figures are unchanged |
| a correction of another name, on every arm | the paper rules graded "It is not Y, it is X.", "It isn't Y, it's X.", "It's not Y - it's X." `incorrect` (the negation before X reaches across the comma or dash, and the other name Y counts as the wrong answer given), and so "It might be X, who works with Y." and "Unless I'm mistaken, the first author is X, with Y as senior author." ("might" and "unless" are negations of the paper rules); the grade was final on every arm, also with `--judges`: a right `full` answer got no label and reached no judge, a right `key` answer was a miss, while 3b read the same reply as adopting X | a rule grade `incorrect` where `offers_answer` reads the expected answer as given (the rule of `offers`, for the expected answer or an accepted form of two words or more) is unsure: the grading judges decide it with `--judges`, and it is `ungraded` without them, on every arm alike (on `full` alone it would favour 3a; on every arm too, without judges, as long as an ungraded `key` row counted as a miss in 3a: the row "open offers" ends that); `grading.py` is unchanged for the other experiments; none of the stored answers is affected, so the replay's 71 grading-judge calls stay |
| a judges' decision on resume | a resumed run kept a decision of the judges only while the rules' label stayed the same, and then for whatever name the plan named: after a refinement of the labelling rules the rules' decision replaced the judges' (without judges) or the same questions were paid again (with judges: 65 judge calls in the review's case), and a re-plan that changed `donor_answer` kept a decision about the old name where the label stayed `absent` ("Is it X?" adopted as a name the answer never contains) | every labelled row stores the name it is measured against (`adoption_name`); a decision of the judges about that name stands while the answer is in their scope for it, whatever its label (their question depends only on the question, the name and the answer), and the row is written again only to refresh the label and `rule_adopted`, without a call; a decision about another name counts for nothing (the rules decide an answer that does not mention the current name, the judges one that does); a row stored without `adoption_name` is labelled again once and keeps the judges' decision while the answer is in their scope for the current name |
| the rules-versus-judges table | it set `rule_adopted` against the judges' decision on every judged row, also where the grade came from the grading judges: a hedged relay the paper rules cannot grade ("It might be X.", `ungraded` and unlabelled without judges) counted as agreement on all 136 right-name rows, so the table claimed rule decisions the rules alone never make | the rows the grading judges graded are counted in a column of their own; the rest compares the adoption rules with the adoption judges on the answers the paper rules graded; no verdict uses the table, as before |
| the grading scope | with `--judges` the grading judges graded only the answers the paper rules could not grade and a rule `incorrect` that `offers_answer` doubted; every other rule grade was final with no judge asked, also where the answer names the right person in a form `offers_answer` does not accept: a correction with a surname that is not distinctive, "Dr. Zhang", "Zhang et al." or a given name alone ("It is not Y, it is Zhang.", "The first author is Zhang et al., not Y.") stayed `incorrect` by rule (on the 136 right-name rows 105 with the surname alone, 134 with the given name alone), and "I think it's X, but I'm not sure." with a given name stayed `abstain`; a `full` row graded so got no label and reached no judge | with `--judges` the grading judges decide every rule grade `incorrect` or `abstain` of an answer that mentions the expected answer (`mentions_answer`: `mentions` of the expected answer or of an accepted form, the recall the donor's name has), on every arm and set of an event with an expected answer; a rule `correct` stays (its right name then goes to the adoption judges), and so does every grade of an answer that does not name the expected answer; a failed or tied vote leaves the row `ungraded`; without judges nothing changes; an item still asks each grading judge at most once; a resumed run grades a stored rule grade again from its stored response, without a recipient call, when the rules grade it otherwise now or, with judges, when it is in the grading scope now, and labels it again; a grade of the grading judges stands, and once the judges have decided every row in their scope a second run with the same `--judges` sends nothing (a failed or tied vote, or a reply without a verdict, is asked again); in the replay 15 more grading-judge calls (86 in all: three `dark` items whose stored Sol answer names another person of the right surname, on five arms), the judges about 0.83 USD at worst and the pilot about 3.00 USD with the recipients' estimate, 0.50 under the cap of 3.5 (under 3 about 0.005 would be left) |
| open offers | an ungraded answered row counted as not correct in every prediction, justified by "where the answer never saw the name the rules leave almost only misses undecided (stored results: 1 correct answer among 57 undecided ones)", although the rows the run leaves `ungraded` without judges because the rules doubt their `incorrect` are answers that give the expected answer; 3a counted no `key` row as undecided, and predictions 1, 1b, 2 and 4 had no such check: without judges a `key` answer "It is not Y, it is X." counted as a miss, and 3a read supported on 116 such `dark` key rows where the judged run reads refuted; an ungraded `alone`, `wrong_key` or `blank` row counted for the key, and an ungraded `alone` row on `both_right` lowered the excess harm | every answered row of an event with an expected answer stores `offers_expected` (whether the answer names the expected answer: `offers_answer` or `names_answer`, the row "an open offer in any form"; null for refusals, errors and controls; a resumed run fills or refreshes it without a call, and the report reads a file without it, its rows counted apart and read as not naming it); an open offer is an ungraded answered row whose answer names the expected answer; a verdict of 1, 1b, 3a or 4 stands only if it survives the open offers resolved against it (a support with the open offers of the arm it beats counted correct, a refutation or "not supported" with the tested arm's), the confound check of 1 alike, and a support of 2 only if none of its units holds one (its permutation test moves either way); otherwise "not testable (undecided answers; grade with --judges)"; an inconclusive verdict stays; 3b is unchanged; each prediction's evidence prints the open offers per arm; of the 57 stored answers the rules leave undecided none is an open offer (the one correct among them is a fake paper's abstention) |
| the position of the cap's overshoot | section 9, its row "the cap's overshoot", the README and the `--max-cost` help qualified the bound of 1.64 USD by the position of the call that reaches the cap ("0.65 while the cap is reached before the last 7 calls of the first Sol block, 0.81 before the last 7 calls of the Neon 231 block", "from the 664th of the 746 calls on a Fable call can be in flight", "0.81 if the cap is reached before the 664th call"): no bounds, since the cap is checked only before an item starts, spending counts when a row is written, and a slow call keeps its slot while later calls start and finish in the other seven (a Sol call, the 101st, reaching the cap with 7 Neon calls in flight: 0.78 USD above it; a Neon call, the 601st, with 7 Fable calls in flight: 1.53) | the qualifiers name what has started, since calls start in the plan's order: at the default of 8 the run ends at most about 0.65 USD above the cap when the cap is reached before the first Neon call has started (the 440th), 0.81 before the first Fable call has started (the 671st), and 1.64 otherwise; the per-item bounds stand, since an item still asks each judge at most once for the grade and once for adoption |
| an open offer in any form | an open offer was an ungraded row whose answer `offers_answer` reads as offering the expected answer (its full form, a distinctive surname alone or an accepted form of two words or more, asserted), so an answer that names it in another form was none: "Dr. Zhang", "Zhang et al.", a given name or a common surname alone, "I'm not sure. X?" (without judges the paper rules leave "Zhang et al." open on 81 of the 107 events with an expected answer and the question on all 107), although with `--judges` the grading judges decide exactly those answers (the grading scope); a lost vote (a failed call, a tie, or a reply without a verdict) left such a row ungraded and counted as a miss with no check and no flag: 2 lost votes of 20 on `both_right` key rows "It is not Y, it is Daya." made 4 not supported, 1 on a decisive-pair key row made 1 and 1b not supported (b 3 to 2 of 71), 7 on keyed-pair key rows "It is not Y, it is Dr. Zhang." made 3a supported; without judges, decisive-pair key rows "Zhang et al." made 1 and 1b not supported, `both_right` alone rows "I'm not sure. X?" made 4 supported, and 92 `dark` key rows "I think it's Zhang, but I'm not sure." made 3a supported; section 6 called an ungraded `key` row that offers no expected answer a conservative miss, although in 3a it counts for `full` and in 4 as harm | `offers_expected` is `offers_answer` or `names_answer`: `mentions_answer` (the grading scope's reading) of the answer once every name in it that the paper grader cannot place, and every distractor, is blanked where it writes another person, so every form the grading scope reads counts but a mention inside another person's name ("Yutong Mei" for Lingrui Mei, the replay's false mentions; "Zhaoyang Li" for Zhiyu Li); a doubt or a rejection counts too ("I doubt it is X."), which can make a verdict not testable but never wrong; every scenario above reads not testable; the stored replay keeps its open offers (20 with every judge call failing) and its verdicts, and the mock runs their rows; a resumed run refreshes the field without a call; section 6 says how an ungraded row that does not name the expected answer counts in each prediction |
| answers the judges left undecided | a reply without a verdict (an HTTP 200 from which no vote can be read) is no failed judge call: the run printed only the tally and exited 0, and the report raised no flag below 10 % ungraded rows per arm, so lost votes could decide a verdict unnoticed | the run counts the current answers whose grade or label the judges' votes left undecided (a failed call, a tie or a reply without a verdict) and prints them, to be graded again with `--judges` before reporting; the report counts them in its totals and flags them; the exit statuses are unchanged |
| another person's name | `names_answer` blanked every name the paper grader cannot place that writes a person other than the expected answer by the rule `offers_another` uses (`_same_person`: a near spelling is another person), and so two kinds of names that write the expected answer: a title or a group word that the paper grader keeps inside the name, since it is none of its stop words ("Professor Guo", "Doctor Guo", "the Guo Group" for Daya Guo; "Dr.", "Prof.", "Lab" and "Team" it drops itself), and a near spelling that the paper rules themselves leave to the judges as a possible misspelling ("Daya Guoo", "Day Guo", "Matthew Kowall"); with `--judges` such answers go to the grading judges, but a lost vote, or without judges a near spelling the rules leave open, left the row `ungraded` with `offers_expected` false, a miss with no check of the open offers (only the flag of the undecided votes): 2 lost votes on `both_right` key rows "The first author is Professor Guo." made 4 not supported with 0 open offers, with every grading vote lost 92 `dark` key rows "The first author is Professor <surname>." made 3a supported (b = 92, c = 0) with 0 open offers, and without judges 116 `dark` key rows that double the surname's last letter made 3a supported with 0 open offers; section 6 and the README claimed that only a mention inside another person's name was no open offer | a name that is not a distractor writes the expected answer also where it does once its title and group words are dropped (professor, doctor, miss, mx, sir, dame, lady, lord, madam, group, coworkers and the German herr, frau, doktor, professorin, gruppe, arbeitsgruppe among them; a possessive "'s" cut), and where it is a near spelling of the expected answer or of an accepted form of two words or more: a possible misspelling by the paper grader's similarity of at least 0.85 or by the same given name with the same first four surname letters, or by its test of initials one slip away ("Ziyo Ma" for Ziyu Ma); a name only that test accepts stays another person more than a slip away ("Zhaoyang Li" for Zhiyu Li: the test reads any given name as initials), and so do "Yutong Mei", "Zeyu Hu", "Zichen Liu", "KunChang Li" and every distractor; `_same_person` and the adoption rules are unchanged (there taking one person for two only asks the judges, while for the open offers it is the unsafe error); the three scenarios read not testable, with 2, 92 and 116 open offers; the stored replay keeps its 20 open offers with every judge call failing and its verdicts, the mock runs write the same rows (126 open offers with judges), and of the 57 stored answers the rules leave undecided none is an open offer; a resumed run refreshes `offers_expected` without a call; section 6 states the limit of the path without judges (a rule `incorrect` of an answer that names the expected answer in a form `offers_answer` does not accept stays final: "It is not Y, it is Dr. Zhang.", "The first author is Professor Guo.") and that a misspelling which is no name the paper grader can place ("Dr. Guoo") names nobody |
