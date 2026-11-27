# Key, not value: does another model's cue unlock a model's own memory?

Status: design contract and pre-registered predictions (2026-11-27). Built and tested offline; the pilot with the real
models has **not** run (it needs an OpenRouter key and the owner's approval, about 2 to 3 dollars).

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
| key: donor hidden state at the subject token, rotated | the donor's own answer text with the answer masked |
| value: what the recipient retrieves from its weights | the name the recipient writes |
| self-swap reference | the recipient asked alone again (noise floor of re-asking) |
| controls: mismatched facts, random maps | wrong key: the masked note of another event |
| dark memory: stored but not retrieved | the recipient misses the open question but passes the multiple-choice question |

Why it matters for the relay: notes are handed between models. If a masked cue recovers B's own memory, then B's
agreement after a key-only hand-over is an independent confirmation, and the rally preset could use a blind key round
(keys first, values after). If B only improves by copying, a `Confirm:` after a full note is an echo and the answer
must treat it as a single source. The same numbers are a text-level transfer matrix between the three models: how well
each one uses another's notes, which is what switching between Sol, Neon and Fable rests on.

## 2. Data (v1)

The AI-paper probe set (`data/events/papers_events.jsonl`: 231 events, 21 of them fake-paper controls, question "Who is
the first author of the arXiv paper titled ...?") and its graded results of 25 November 2026
(`runs/papers_results_final.jsonl`: probes `current_d` and `mcq` for all three models). The stored `current_d` rows give
every (event, model) its alone outcome and its response text, so the donor notes cost nothing. Counts from that file
(donor correct, recipient not correct, real papers): Fable to Neon 50, Fable to Sol 96, Neon to Sol 47; every other
direction 1. The knowledge is nested in this topic (Fable knows almost everything the others know), so the reverse
directions cannot be measured with this set; the report lists them with their counts. The Wikidata events have no
stored results; they need a donor run first (later step).

## 3. Item sets per directed pair (donor A, recipient B)

| set | selection (stored `current_d` grades) | arms |
|---|---|---|
| `dark` | real paper, A `correct`, B not `correct` | alone, key, full, wrong_key |
| `donor_wrong` | real paper, A `incorrect` (a wrong name), B not `correct` | alone, full |
| `both_right` | real paper, A and B `correct`; seeded sample of `--both-right` (default 10) per pair | key, full |
| `control` | fake paper, A `incorrect` (it named an author for a paper that does not exist) | alone, full |

`recognised` (dark items only): B's stored `mcq` grade for the event is `correct`. A pair with fewer than `--min-items`
(default 5) `dark` items is planned for no set and reported as too small.

## 4. Prompts

System prompt: `current_d_system(today)` for every arm (the probe the index is built on), `today` = the run's date.

- `alone`: user = the event's `question_current` (the stored probe, asked again).
- `key`, `full`, `wrong_key`: user = `KEY_USER_TEMPLATE`, identical for the three arms, so only the note differs:

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
- `wrong_key`: the donor's stored correct response for a **different** real paper (seeded draw among the donor's correct
  responses, never the same event, never one whose expected answer equals this item's), masked with that event's own
  answer forms and title. It has the shape of a key and points elsewhere.

Calls: `ChatRequest(messages, max_tokens=4096, temperature=0.0, metadata={"event_id", "probe": "current_d", "model",
"arm", "set", "donor", "item"})`, the same request shape as `bench run`.

## 5. Masking and leak check (`mask_note`)

- Forms of the answer: `expected_current`, every entry of `aliases`, and each name token of `expected_current`
  (surname and given names, at least 2 letters) as a whole word. Matching is case-insensitive, accents folded (the
  text is matched in both its own and its accent-free spelling), spaces and hyphens inside a name interchangeable.
  Every match becomes `[withheld]`; adjacent placeholders collapse into one.
- The title (`subject`) becomes `[title]` (case-insensitive, punctuation between its words free).
- Leak check: the masked note must not be graded `correct` by the paper rule grader (`grade_paper(event, "current_d",
  note)`), and no answer token of 3 or more letters may remain. An item whose key leaks is excluded from `dark` and
  `both_right` (counted in the plan summary as `leaks`); its wrong-key note may still serve other items once masked
  with its own forms and checked.

## 6. Plan, run, report

```
cutoffmix bench keys plan   --events E.jsonl --results R.jsonl --out runs/keys_plan.jsonl
                            [--models K,K,K] [--pairs A:B,...] [--both-right 10] [--min-items 5] [--seed 0] [--today D]
cutoffmix bench keys run    --plan runs/keys_plan.jsonl --out runs/keys_results.jsonl
                            [--mock] [--concurrency 8] [--max-cost USD] [--limit N] [--judges KEY ...]
cutoffmix bench keys report --plan runs/keys_plan.jsonl --results runs/keys_results.jsonl [--json OUT] [--md OUT]
```

- `plan` (offline, deterministic for its inputs and seed) writes one row per call: `id` (stable hash of set, donor,
  recipient, event, arm), `set`, `donor`, `recipient`, `event_id`, `arm`, `recognised`, `donor_grade`, `donor_answer`
  (the name the donor asserted, for `donor_wrong` and `control`), `note`, `note_event` (the wrong key's source event),
  `messages`. It prints the counts per pair, set and arm, the excluded pairs and leaks, and a cost estimate (stored mean
  cost per `current_d` call per model, scaled by prompt length).
- `run` sends the planned calls (resumable: an id that already has a row without `error` is skipped), grades each answer
  with the existing graders (`grade_response`: the paper rules, the judges only when `--judges` is given) and appends
  rows `{id, set, donor, recipient, event_id, arm, response, grade, hedged, grader, adopted, latency_s, usage, error}`.
  `adopted` (sets `donor_wrong` and `control`): the response asserts the donor's wrong name. `--max-cost` stops sending
  once the summed `usage.cost` reaches it; `--limit` caps the calls. `--mock` uses the simulated backend.
- `report` (offline) joins plan and results: per pair and set, per arm `n`, correct, rate with a 95 % Wilson interval;
  on the items that have all arms graded, paired comparisons with the exact McNemar test (two-sided binomial on the
  discordant pairs): `key` vs `alone`, `key` vs `wrong_key`, `full` vs `key`; the `dark` set split by `recognised`;
  adoption rates (`donor_wrong`, `control`: `full` vs `alone`); harm in `both_right` (share not `correct`); calls,
  ungraded rows and cost. Markdown on stdout, JSON with `--json`.

## 7. Pre-registered predictions (written before any real call)

Taken from the paper and stated so the pilot can refute them.

1. On `dark` items, `key` beats both `alone` and `wrong_key` (paired McNemar, two-sided, p < 0.05 on at least the
   largest pair, Fable to Sol). Effect size expected from the paper: some, not most, of the misses recovered (there: 18 %
   of dark memories); we expect 10 to 30 % of `dark` items correct under `key` against under 10 % under `wrong_key`.
2. The recovery is concentrated on `recognised` items: the `key` rate on `recognised` items is higher than on
   unrecognised ones.
3. `full` beats `key` (the value is copied), and `full` makes the recipient adopt a wrong donor answer (`donor_wrong`,
   `control`) far more often than `alone` does.
4. The notes rarely break a correct recall (`both_right`, `key`): harm under 10 %.

If 1 fails while 3 holds, a model's agreement with a hand-over is copying, and the rally's confirmations after full
notes must count as single source. If 1 holds, a blind key round in the rally is worth a test of its own.

## 8. Mock

`MockBackend.chat` answers a keyed request (`metadata["arm"]`) by the simulated model's truth like any probe, plus:
`key` turns a miss into the right answer with a fixed probability when the event lies within a few months after the
model's simulated cutoff (a simulated dark memory); `full` copies the donor's name (from the note) with a fixed
probability; `wrong_key` behaves like `alone`. Deterministic per seed. The mock exists to run the pipeline offline; its
numbers say nothing about the real models.

## 9. Costs

Planned by default: `dark` 193 items x 4 arms, `donor_wrong` and `control` x 2, `both_right` 3 pairs x 10 x 2: about
1,000 calls. Mean stored cost per `current_d` call: Fable 0.84 ct, Neon 0.51 ct, Sol 0.09 ct; the recipients are Neon
and Sol, so the estimate is about 2 to 3 dollars (the plan prints the exact figure). Approval and an OpenRouter key are
needed first.
