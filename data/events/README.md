# Event sets

## `wikidata_events.jsonl`

961 template probe events dated 2024-01-01 to 2026-11-21, one `cutoffmix.types.Event` per line (`origin: "wikidata"`).

**Provenance**
- Built on 2026-11-24 (17:55 UTC, as-of date 2026-11-24) with
  `cutoffmix.bench.sources.wikidata.build_wikidata_events("2024-01-01", "2026-11-24")`.
- Source: the Wikidata Query Service (`query.wikidata.org`). The run sent its queries one at a time: 16 date-range
  windows, then follow-up `VALUES` queries in chunks of 200 ids.
  - the date-range queries, in yearly windows: `heads`, `elections`, `awards` (from 2023-01-01, so that 2024 awards
    have a previous recipient), `sports`, `sports_series`;
  - the follow-up queries: `election_context`, `sports_context`.

  The queries are in `cutoffmix.bench.sources.wikidata.QUERIES`. No window timed out, so no narrowing was needed.
  The build took 206 s.
- No LLM was used. Questions, answers and multiple-choice options are templates filled from Wikidata labels.
- Each event's `source` field is the Wikidata item it came from: the person, election, award or competition edition.

**License and attribution.** Wikidata content is CC0 1.0 (public domain dedication), so attribution is not required.
Please still credit "Data from Wikidata (wikidata.org), CC0". The repository code that built the file is MIT.

**Reproduce.** Run `cutoffmix bench build-wikidata --out FILE` (CLI, WP5), or
`python -c "from cutoffmix.bench.sources.wikidata import build_wikidata_events as b; from cutoffmix.bench.io import save_events as s; s('FILE', b('2024-01-01', '2026-11-24'))"`.
Wikidata keeps changing, so a rebuild will give slightly different events.

### What an event contains

| Family | Category / topic | `question_current` (no date) | `question_recall` (names month and year) | Stale answer (`reject_aliases`, MCQ option) |
|---|---|---|---|---|
| Heads of state and government (P39 with P580, offices from P1906/P1313) | `office_change` / government | "Who is the President of Serbia?" | "Who became President of Serbia in November 2026?" | P1365 "replaces" and earlier holders. If the direct predecessor was only acting, the substantive holder they stood in for |
| Elections with one successful candidate (P991) | `election` / elections | "Who won the most recent Brazilian presidential election?" | "Who won the 2026 Brazilian presidential election, held in October 2026?" | Winner of the previous election (P155), or the previous election in the data |
| Recipients of 49 curated awards (P166 with P585) | `award` / per award (Nobel sciences → science_research, Booker → arts_entertainment, …) | "Who won the most recent Booker Prize?" | "Who won the Booker Prize in November 2026?" | The previous recipients |
| Editions of recurring competitions with one winner (P1346) | `sports_result` / sports | "Who won the most recent edition of the Dutch Grand Prix?" | "Who won the 2025 Dutch Grand Prix, which concluded in August 2025?" | Winner of the previous edition (P155), or the previous edition in the data |

- `question_current` is left empty when something later in the data supersedes the answer. That happens when:
  - a later holder or edition exists;
  - the office-holding statement ends on or before the as-of date;
  - two different winners appear under one competition label within 30 days.

  The MCQ of a superseded event uses the dated question instead.
- The MCQ has four options:
  - the correct answer;
  - the stale answer, if one exists;
  - same-type candidates taken from the same query results, in this order of preference: candidates of the same
    election, holders from the same country, earlier winners of the same competition, the same sport and team kind,
    the same award family.

  The correct letter is `stable_hash(event id) % 4`. Answers are spread A 249, B 242, C 211, D 259.
- `aliases` holds the label plus simple variants: without diacritics, without a parenthetical, without "Jr.".
- When the winner equals the previous winner (re-election, repeat champion), the event gets `predictability: "high"`
  and no stale option. Every other event is `medium`. Neither label has been checked against old models yet: run the
  external predictability check before fitting the index.
- Filters drop:
  - bare-QID labels;
  - deprecated statements;
  - dates less precise than a month (one exception, below);
  - acting holders, collective or co-held offices, and multi-winner items;
  - elections and sports editions with fewer than 3 sitelinks;
  - competitions without a sport (P641);
  - duplicate items describing the same result.

  The exception to the date rule: year-precision award dates are mapped to the award's usual announcement month
  (Nobel → October, Booker → November, …; see `AWARDS`). Such events carry day `01`.
- Every event passes `cutoffmix.bench.probegen.validate_event`. That check rejects:
  - date or month tokens in `question_current`;
  - an answer that appears in a question;
  - an MCQ that does not have 4 unique options with a valid letter.

### Counts

Topic × quarter (2026Q4 covers 2026-10-01 to 2026-11-24):

| topic | 2024Q1 | 2024Q2 | 2024Q3 | 2024Q4 | 2025Q1 | 2025Q2 | 2025Q3 | 2025Q4 | 2026Q1 | 2026Q2 | 2026Q3 | 2026Q4 | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| international_relations | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 3 |
| government | 22 | 14 | 17 | 16 | 25 | 12 | 12 | 15 | 10 | 11 | 11 | 4 | 169 |
| elections | 9 | 10 | 7 | 22 | 2 | 5 | 1 | 17 | 7 | 5 | 6 | 5 | 96 |
| business_economy | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 3 |
| science_research | 4 | 3 | 1 | 3 | 3 | 2 | 1 | 3 | 2 | 0 | 0 | 3 | 25 |
| sports | 49 | 88 | 32 | 48 | 41 | 104 | 52 | 48 | 32 | 82 | 22 | 12 | 610 |
| arts_entertainment | 8 | 3 | 2 | 3 | 12 | 4 | 2 | 3 | 10 | 3 | 2 | 3 | 55 |
| **all** | 92 | 118 | 59 | 94 | 83 | 127 | 68 | 88 | 61 | 101 | 41 | 29 | 961 |

| category | events | with `question_current` | high predictability | with stale answer |
|---|---:|---:|---:|---:|
| award | 93 | 38 | 1 | 86 |
| election | 96 | 96 | 23 | 40 |
| office_change | 169 | 127 | 0 | 151 |
| sports_result | 603 | 356 | 112 | 275 |

531 events can feed the `current`/`current_d` index: they have a `question_current` and are not high-predictability.
By topic: sports 296, government 127, elections 73, arts_entertainment 22, science_research 11,
international_relations 1, business_economy 1.

### Caveats

- **Coverage is uneven.** Wikidata covers only 7 of the 15 topics, and sports makes up 63% of the events. Wikidata
  has nothing for conflict, law, accidents, natural disasters, health, AI/ML, computing or space. Those topics need the
  Wikipedia Current-events items (`cutoffmix.bench.sources.wikipedia` plus `cutoffmix.bench.probegen`, which needs an
  API key) and the kc import.
- **The last months are thin.** Wikidata lags real events: 2026Q3 has 41 events and 2026Q4 has 29, against about 90
  per quarter earlier. Only 15 events fall after 2026-10-25. Events less than 14 days old are provisional: rebuild
  before use.
- **The ground truth is Wikidata's.** Data errors pass straight into the events: wrong dates, a successor that has not
  been entered yet, and so on. If Wikidata has not yet recorded a change, `question_current` can be out of date. Run
  the 5% human spot check.
- **Award months are partly inferred.** Year-precision award dates use the award's usual announcement month, so their
  month is approximate.
- **MCQ distractors can be weak.** In thin sports, candidates come from other sports, which makes the MCQ easier. Treat
  the MCQ as an upper bracket only.
- **Labels are English** and come straight from Wikidata. Some are long ("Spain men's national football team"). Each
  answer's accepted surface forms are listed in `aliases`.
