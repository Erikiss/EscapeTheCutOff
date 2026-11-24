## Knowledge-cutoff benchmark deep-dive (apoorvumang/knowledge-cutoff), as of 2026-11-24

### Main points
1. **The GitHub repo is newer than the HF dataset.** The HF `results.jsonl` (lastModified 2026-07-10) has only 12 models. The repo's `report_answers.json` (HEAD a7bef1e, 2026-11-24) has 34 models, and all three targets are in it with both probes, 330 events each. They were added in commit 77e48c9 on 2026-11-23: `claude-fable-5.2`, `gpt-6.2-sol`, `gemini-4-neon`.
   - Name check: the "6.2 Sol" model in this benchmark is OpenAI `openai/gpt-6.2-sol`. There is no "Gemini 6.2".
2. **The event window stops at Jun 2026.** It covers 2024-01..2026-06 and the events file has not changed since 2026-07-10. On the direct probe all three targets score 0/10 in 2026-06, and on the MCQ probe Fable 5.2 and GPT-6.2 Sol never fall below threshold. So the benchmark says nothing about Jul–Nov 2026. You would need new events for those months.
3. **GPT-6.2 Sol is ahead everywhere; per-topic routing has almost no measurable gain on this data.**
   - Direct crossover: Fable 5.2 = 2026-02; Neon = 2026-03; GPT-6.2 Sol = no_tail (last month ≥0.5 is 2026-05).
   - An oracle that always picks the right model per event beats always-GPT-6.2-Sol by only +2 of 280 eligible events on direct (both 2026-03, both Neon, both politics) and +1 on MCQ.
   - GPT-6.2 Sol is the only correct model on 16 direct events; Neon on 2; Fable on 1.
4. **Per-topic × quarter estimation is too sparse.** Details and counts are in section 6.

### 1. Files fetched
- `git clone --depth 1` worked, then `--unshallow` (29 commits).
- Repo files: `kc/{__init__,cli,grade,prompts,providers,run,schema,score}.py`, `models.yaml`, `data/events.jsonl` (330 rows, byte-identical to the HF `events.jsonl`), `data/controls.jsonl` (18), `data/events.sample.jsonl`, `report_answers.json` (identical to `docs/report_answers.json`; format `{model:{probe:{event_id:{l:c|w|a|u, r:response capped at 700 chars}}}}`), `docs/index.html` (contains the published summary as JSON), `report_template.html`, `scripts/{export_viz,build_hf,build_page,assemble}.py`, `run_*.sh`.
- `runs/` and `graded/` are gitignored and not published. So there is no per-row meta (sampling, finish_reason, reasoning tokens) anywhere.
- Events file history: last changed 4f6fcdb (2026-07-10). `score.py`/`grade.py` last changed 61b54d4 (2026-08-21).

### 2. The `kc/` package

**Schema (`schema.py`)**
- Required fields: id, date, month (YYYY-MM), category, region (US|International), predictability (low|medium|high), subject, fact, question_direct, expected_direct, mcq_question, mcq_choices (exactly 4 strings, "A) …"), mcq_answer (A–D). `source` is optional.
- `REAL_CATEGORIES = {death, office_change}`.
- `counts_for_curve = category in REAL and predictability in {low, medium}`.
- Counts: death 201, office_change 111, control_alive 10, fake_event 8. Predictability: low 102, medium 178, high 50. Region: US 169, Intl 161.
- Curve-eligible events: 280, 7–11 per month.

**Prompts (`prompts.py`), exact strings**
- Direct system: `"Answer the question directly and factually based on what you know. If you are not sure, say so, but give your best answer."`
- Direct user: `question_direct`, verbatim.
- MCQ system: `"Answer the multiple-choice question. Respond with ONLY the single letter (A, B, C, or D) of the best answer, nothing else."`
- MCQ user: `f"{mcq_question}\n\n{'\n'.join(mcq_choices)}\n\nAnswer with a single letter:"`
- No date is ever injected.

**Judge (`grade.py`)**
- `JUDGE_SYSTEM`, exact: `"You are a strict grader. You are given a QUESTION, the GROUND TRUTH answer, and a MODEL ANSWER. Classify the MODEL ANSWER into exactly one label:\n  correct   - it asserts the ground-truth fact (minor wording/date-precision differences are fine; the key fact must match).\n  incorrect - it asserts something that contradicts the ground truth (e.g. says a person is alive/active when the truth is they died, names the wrong office-holder, gives a clearly wrong outcome).\n  abstain   - it declines to answer, says it does not know, says its information may be out of date, or hedges without committing to either the true or a false claim.\n\nJudge ONLY against the provided ground truth. Do not use your own knowledge of current events. Output STRICT JSON: {\"label\": \"...\", \"reason\": \"...\"}"`
- Judge user: `f"QUESTION:\n{question_direct}\n\nGROUND TRUTH:\n{expected_direct}\n(context: {fact})\n\nMODEL ANSWER:\n{answer}"`
- Judge model: `judge-opus-4-8` = OpenRouter `anthropic/claude-opus-4.8`, max_tokens=300, temperature 0, 5 attempts with exponential backoff.
- Parsing: regex `\{.*\}` (DOTALL), then `json.loads`. A label outside {correct, incorrect, abstain} becomes `"ungraded"`.
- API error or empty response becomes `ungraded`, never `abstain`.

**MCQ grading**
- Letter extraction tries, in order: `^\s*\(?([A-D])\)?[\.\):]?\b`, then `answer\s*(?:is|:)?\s*\(?([A-D])\)?` (case-insensitive), then the first isolated `\b[A-D]\b`.
- No letter found → `abstain`. Empty response → `ungraded`.

**Scoring (`score.summarize`, threshold default 0.5)**
- Ungraded rows are dropped from every denominator.
- Per month over curve-eligible rows: `known_rate = correct/n`; `wrong_rate` and `abstain_rate` are computed the same way. No smoothing and no confidence intervals.
- `last_above` = latest month with known_rate ≥ thr.
- `crossover` = latest month m with known_rate ≥ thr such that at least `MIN_TAIL = 2` months follow it and all of them are < thr.
- `cutoff_status`:
  - `no_data`
  - `before_window`: never ≥ thr.
  - `never_below`: all months ≥ thr; crossover is set to `"2026-06+"`.
  - `no_tail`: dips and recovers, or the tail is shorter than 2 months; crossover = None.
  - `pinned`: otherwise.
- Controls: `alive_correct/alive_n` and `fake_confirmed` (label `incorrect`)`/fake_n`.
- The page uses `crossover` as "the cutoff". `export_viz.py` skips a probe if more than 5% of rows are ungraded.

**`models.yaml` and OpenRouter calls**
- Format: `providers: {name: {kind: anthropic|openai, base_url, api_key_env}}` and `models: {key: {provider, model_id, params?}}`. `KC_REGISTRY_EXTRA` overlays a private YAML.
- Target entries:
  - `claude-fable-5.2`: `model_id: "anthropic/claude-fable-5.2"`, `params: {provider: {only: [anthropic], allow_fallbacks: false}}`
  - `gpt-6.2-sol`: `model_id: "openai/gpt-6.2-sol"`, `only: [openai]`
  - `gemini-4-neon`: `model_id: "google/gemini-4-neon"`, `only: [google-ai-studio]`
- Call path: `openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=$OPENROUTER_API_KEY).chat.completions.create(model, messages=[system, user], temperature=0, max_completion_tokens=2048, extra_body={"provider":{...}})`. Parameters other than top_p, presence/frequency_penalty, seed and stop go into `extra_body`.
- On *any* exception it retries without temperature, then with `max_tokens`. The row meta still records temperature 0, so this is a minor reproducibility hole.
- Only `message.content` is kept. No `reasoning` or `reasoning_effort` is sent, so the targets ran with OpenRouter's default reasoning setting. That default is UNVERIFIED.
- Runner: 8 threads, resumable by event_id, retries on 429/5xx/timeout.

### 3. Results files
- **HF `results.jsonl`:** 7,920 rows; columns `model, probe, event_id, month, category, predictability, label, response`. 12 models × 2 probes × 330: claude-fable-5, claude-opus-4-8, claude-sonnet-5, deepseek-v4-pro, gemini-3.1-pro, gemini-3.5-flash, glm-5.2, gpt-4o, gpt-5.4, gpt-5.5, gpt-5.6-sol, grok-4.5.
  - It is stale: it was built before the 2026-08-11 "ungraded" fix. 31 MCQ labels differ from the repo (glm-5.2 24, deepseek-v4-pro 3, claude-fable-5 2, gpt-5.5 1), all HF `abstain` that the repo now labels incorrect, correct or ungraded.
- **Repo `report_answers.json`:** 34 models.
  - Present: claude-fable-5.2, claude-fable-5, gpt-6.2-sol, gpt-5.6-sol, gemini-4-neon, gemini-4-carbon, gemini-4-argon.
  - Absent: claude-fable-5.1 (in `models.yaml` but no results), gpt-6.1-sol, gpt-6-sol.
- My recomputation with the repo's exact definition matches the published page for all 68 model/probe pairs (cutoff, status and monthly rates).

**Leaderboard (crossover / status / overall known_rate)**

| model | advertised | direct | mcq |
|---|---|---|---|
| claude-fable-5.2 | Jun 2026 | 2026-02 pinned, 0.87 | 2026-06+ never_below, 0.96 |
| gpt-6.2-sol | Jun 2026 | None no_tail (last_above 2026-05), 0.93 | 2026-06+ never_below, 0.98 |
| gemini-4-neon | Mar 2026 | 2026-03 pinned, 0.87 | None no_tail (last_above 2026-05), 0.95 |
| gemini-4-carbon | Mar 2026 | 2026-02, 0.65 | 2026-04, 0.89 |
| gpt-5.6-sol | Feb 2026 | 2025-12, 0.79 | 2026-06+ never_below, 0.93 |
| claude-fable-5 | Jan 2026 | 2025-10, 0.72 | 2025-11, 0.82 |
| muse-spark-1.2 | n/p | 2025-11 | 2026-01 |
| ox-alpha | n/p | 2025-10 | 2026-01 |
| gpt-5.5 / claude-opus-4-8 | | 2025-09 | 2025-11 |
| inkling / kimi-k3 | | 2025-09 | 2026-04 |
| glm-5.3-flash | | 2025-09 | 2026-01 |
| inkling-2-mini | | 2025-09 | no_tail |
| claude-sonnet-5 | | 2025-08 | 2025-11 |
| grok-4.6 | | 2025-08 | 2025-09 |
| muse-glimmer-30b | | 2025-08 | 2025-09 |
| hy4-preview | | 2025-08 | 2026-02 |
| gpt-5.4 | | 2025-06 | no_tail |
| glm-5.3 | | 2025-06 | 2025-11 |
| grok-4.5 | | 2025-03 | 2025-10 |
| deepseek-v4-flash-0731 | | 2025-03 | 2025-04 |
| gemini-4-argon | | 2025-01 | 2025-09 |
| gemini-3.x, qwen3.8-max/27b, mercury-2.5, glm-5.2, deepseek-v4-pro | | 2024-12 | various |
| gpt-4o | | before_window | before_window |

Controls: all three targets 10/10 alive and 0/8 fake confirmed. Neon has 3 ungraded MCQ rows, all in 2026-03.

**Target tail, direct, correct/n**

| month | n | Fable 5.2 | GPT-6.2 Sol | Neon |
|---|---|---|---|---|
| 2025-09..2026-01 | | ≈1.0 | ≈1.0 | ≈1.0 |
| 2026-02 | 9 | 6 | 9 | 9 |
| 2026-03 | 10 | 4 | 8 | 5 |
| 2026-04 | 10 | 2 | 9 | 0 |
| 2026-05 | 10 | 0 | 5 | 0 |
| 2026-06 | 10 | 0 | 0 | 0 |

**Target tail, MCQ, correct/n** (25% guessing floor)

| month | Fable 5.2 | GPT-6.2 Sol | Neon |
|---|---|---|---|
| 2026-02 | 7/9 | 9/9 | 9/9 |
| 2026-03 | 10/10 | 10/10 | 7/7 |
| 2026-04 | 10/10 | 10/10 | 9/10 |
| 2026-05 | 7/10 | 9/10 | 5/10 |
| 2026-06 | 5/10 | 5/10 | 3/10 |

### 4. How the targets behave past their horizon (useful router signals)
Direct probe, 2026 eligible events (n=58 each):
- **GPT-6.2 Sol:** 40 correct, 18 incorrect, 0 abstain (0 abstains in all 330 direct rows). It never mentions a cutoff and confidently confabulates "alive".
- **Neon:** 23 correct, 9 incorrect, 26 abstain. It self-reports "As of … March 2026" in 39 responses.
- **Fable 5.2:** 21 correct, 23 incorrect, 14 abstain.

The judge is not consistent on the abstain/incorrect boundary. Neon's "As of March 2026, Kharazi is alive" was graded abstain, while Fable's "As far as I know … is alive" was graded incorrect. So use only `known_rate`, as the repo does.

### 5. Oracle headroom over the three targets
File: `targets_oracle_union.json`.
- Direct, 2025-09..2026-06: the union beats the best single model only in 2026-03 (10 vs 8).
- Events only one target got right: GPT-6.2 Sol 16 (entertainment 6, business 5, sports 3, politics-intl 2); Neon 2 (Hanabusa, politics-US; Oddsson, politics-intl); Fable 5.2 1 (José Jerí, 2025-10).
- MCQ: union 10 vs best 9 in 2026-05; uniquely correct: GPT-6.2 Sol 2.

### 6. Topic tags and sparsity
Tagging uses ordered regexes on fact+subject: an "other" override (religion, Harvard, journalism, Epstein, gang), then business (CEO / chief executive / …), then science, then whichever of sports / entertainment / politics matches earliest. office_change events with a politics keyword go to politics, split US/intl by `region`. I spot-checked all 330 by eye and the result is plausible; Hulk Hogan → sports and Sophia Hutchins → business are judgment calls.

Totals over all 330: entertainment/arts 130, politics-intl 94, sports 38, politics-US 28, business/finance 24, other 9, science/tech 7.

Curve-eligible events per topic × quarter:

| topic | 24Q1 | 24Q2 | 24Q3 | 24Q4 | 25Q1 | 25Q2 | 25Q3 | 25Q4 | 26Q1 | 26Q2 |
|---|---|---|---|---|---|---|---|---|---|---|
| politics-US | 3 | 0 | 3 | 3 | 0 | 1 | 4 | 2 | 4 | 0 |
| politics-intl | 8 | 14 | 12 | 12 | 7 | 7 | 8 | 7 | 8 | 6 |
| business/finance | 2 | 2 | 3 | 3 | 0 | 0 | 2 | 0 | 1 | 7 |
| sports | 4 | 5 | 4 | 3 | 4 | 1 | 4 | 2 | 1 | 6 |
| entertainment/arts | 11 | 7 | 9 | 8 | 11 | 12 | 9 | 14 | 14 | 10 |
| science/tech | 1 | 0 | 0 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| other | 1 | 1 | 0 | 1 | 1 | 3 | 0 | 0 | 0 | 1 |

**Verdict: per-topic × quarter estimation is not feasible from this data.**
- Only 4 science/tech events in total. politics-US has 0 in 2026Q2. Most cells hold 0–7 events.
- At n=8 the binomial SE is about ±0.18.
- The targets only differ in 2026-02..05, about 40 events.
- 2026H1 direct, correct/n:

| topic | Fable 5.2 | GPT-6.2 Sol | Neon |
|---|---|---|---|
| entertainment | 8/24 | 17/24 | 10/24 |
| politics-intl | 7/14 | 9/14 | 7/14 |
| business | 2/8 | 7/8 | 1/8 |
| sports | 1/7 | 4/7 | 1/7 |
| politics-US | 3/4 | 3/4 | 4/4 |

- None of the gaps in Neon's favour are significant.
- To get per-topic estimates you would need new low/medium events for Jul–Nov 2026, plus denser 2026-01..06 coverage, with roughly ≥20 per topic per quarter (SE ≈ 0.11). Compare models with paired tests on the same events (McNemar), or fit a pooled logistic model of P(correct | model, topic, month) with per-topic cutoff offsets shrunk toward the model's overall cutoff. That model is my design suggestion; it is not in the repo. Neither the README nor the dataset card cites a paper or arXiv ID.

### 7. OpenRouter metadata
Fetched from the public `/api/v1/models` endpoint (473 models). All three IDs exist. `knowledge_cutoff` is null for all three.

| model | created | context | price per M tokens (in/out) | supported_parameters |
|---|---|---|---|---|
| anthropic/claude-fable-5.2 | 2026-11-09 | 1M | $8 / $40 | include_reasoning, max_completion_tokens, max_tokens, reasoning, reasoning_effort, response_format, stop, structured_outputs, tool_choice, tools, verbosity |
| openai/gpt-6.2-sol | 2026-11-17 | 1.05M | $2 / $10 | same as Fable minus `stop`, plus `seed` |
| google/gemini-4-neon | 2026-11-18 | 1.05M | $3 / $15 | include_reasoning, max_tokens, reasoning, reasoning_effort, response_format, seed, stop, structured_outputs, temperature, tool_choice, tools, top_p |

GPT-6.2 Sol does not list `stop`, so segment boundaries for switching cannot rely on stop sequences across all three; use `max_tokens` instead.

### 8. License
- HF dataset card and HF tags: `license: cc-by-4.0` (dataset).
- The GitHub README shows a CC BY 4.0 badge, but the repo has no LICENSE file (none in `git ls-files`). So the code (`kc/`, `scripts/`) has no formal software license grant; reimplement it or ask the author before vendoring.
- Events and results can be reused with attribution; the card asks users to cite the dataset. Per-row `source` URLs are Wikipedia or news.

### Files (all under /tmp/claude-0/-home-user-EscapeTheCutOff/95d45c65-31f1-5b9f-b0f4-53e755166f71/scratchpad)
- (scratchpad)/kc-repo/ — full clone, HEAD a7bef1e
- (scratchpad)/kc-data/events.jsonl — HF copy
- (scratchpad)/kc-data/results.jsonl — HF copy
- (scratchpad)/kc-data/summary.json — curves and cutoffs for both sources, HF-vs-repo consistency, topic tables, per-topic half-year rates for the targets and siblings, definitions
- (scratchpad)/kc-data/curves_all_models.csv — 34 models × 2 probes × 30 months
- (scratchpad)/kc-data/results_repo_all.jsonl — repo answers for all 34 models in long form (responses capped at 700 chars)
- (scratchpad)/kc-data/results_repo_targets.jsonl — the three targets only
- (scratchpad)/kc-data/events_topics.jsonl — topic tag per event
- (scratchpad)/kc-data/targets_oracle_union.json
- (scratchpad)/kc-data/report_data_from_docs.json — the published summary extracted from the page
- (scratchpad)/kc-data/openrouter/models.json
- Scripts: (scratchpad)/kc-scripts/analyze.py (exact port of `score.summarize` plus the topic tagger) and (scratchpad)/kc-scripts/oracle.py. Run with `python3 -I <script> <scratchpad>`.