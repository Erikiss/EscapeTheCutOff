# Event sets

## `wikidata_events.jsonl`

953 template probe events dated 2024-01-01 to 2026-11-21, one `cutoffmix.types.Event` per line (`origin: "wikidata"`).

**Provenance**
- Rebuilt on 2026-11-24 (19:29 UTC, as-of date 2026-11-24) after the bench review fixes (BENCH-1/3/5/6/8/9), with
  `cutoffmix.bench.sources.wikidata.build_wikidata_events("2024-01-01", "2026-11-24")`.
- Source: the Wikidata Query Service (`query.wikidata.org`). The run sent its queries one at a time: 15 date-range
  windows, then follow-up `VALUES` queries in chunks of 200 ids.
  - the date-range queries, in yearly windows: `heads`, `elections`, `awards` (from 2023-01-01, so that 2024 awards
    have a previous recipient), `sports`, `sports_series`;
  - the follow-up queries: `election_context`, `sports_context` (both now also fetch the dates of the P155/P156
    neighbour editions) and `comp_editions` (every dated edition of each competition since 2020, with or without a
    winner).

  The queries are in `cutoffmix.bench.sources.wikidata.QUERIES`. No window timed out, so no narrowing was needed.
  The build took 233 s.
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
| Recipients of 49 curated awards (P166 with P585) | `award` / per award (Nobel sciences → science_research, Booker → arts_entertainment, …) | "Who won the most recent Booker Prize?"; for work awards (best film, series, album, song, festival prizes) "What won the most recent Golden Lion?" | "Who won the Booker Prize in November 2026?" | The previous recipients |
| Editions of recurring competitions with one winner (P1346) | `sports_result` / sports | "Who won the most recent edition of the Dutch Grand Prix?" | "Who won the 2025 Dutch Grand Prix, which concluded in August 2025?" | Winner of the previous edition (P155), or the previous edition in the data |

- `question_current` is left empty when something later supersedes the answer. That happens when:
  - a later holder or edition exists in the data (a competition is identified by its QID or label; an edition
    belongs to its most specific competition: the one named in the edition label, then the one with the fewest
    editions, then the P179 series);
  - the election's or edition's P156 successor, or any later edition of the competition (`comp_editions`, winner or
    not), is dated on or before the as-of date;
  - the next edition is overdue: the series recurs at most every 400 days (median gap between known editions,
    "2024–25" season labels, every curated award) and date + period + 30 days is on or before the as-of date;
  - the office-holding statement ends on or before the as-of date;
  - two different winners appear under one competition within 30 days.

  The MCQ of a superseded event uses the dated question instead.
- The MCQ has four options:
  - the correct answer;
  - the stale answer, if one exists;
  - same-type candidates taken from the same query results, in this order of preference: candidates of the same
    election, then winners from the same country; holders of the same office kind; earlier winners of the same
    competition, then winners of the same sport and team kind (women's, national and youth markers equal);
    recipients of the same award, then of the same award family and recipient type (person or work). There is no
    fallback to other sports, countries or award types: an event with too few such candidates has no MCQ.

  The correct letter is `stable_hash(event id) % 4`. Answers are spread A 234, B 210, C 177, D 234.
- `aliases` holds the label plus simple variants: without diacritics, without a parenthetical, without "Jr.", and
  for teams the short names people answer with ("Manchester United F.C." → "Manchester United", "Spain men's
  national football team" → "Spain", "Spain national team", "2023 Kansas City Chiefs season" → "Kansas City
  Chiefs"). A short form that occurs in a question is left out. Work awards answer with the work; people credited
  for it (P1686 "for work") are extra aliases.
- When the winner equals the previous winner (re-election, repeat champion), the event gets `predictability: "high"`
  and no stale option. Every other event is `medium`. Neither label has been checked against old models yet: run the
  external predictability check before fitting the index.
- Filters drop:
  - bare-QID labels and blank nodes ("unknown value", `.../.well-known/genid/...`);
  - predecessors (P1365) that are not people (e.g. "2025 Malagasy protests");
  - human recipients of work awards without a P1686 work (honorary prizes such as a lifetime Golden Lion);
  - deprecated statements;
  - dates less precise than a month (one exception, below);
  - acting holders, collective or co-held offices, and multi-winner items;
  - elections and sports editions with fewer than 3 sitelinks;
  - competitions without a sport (P641);
  - duplicate items describing the same result (same competition, date and winner, or the same winner within 3 days
    in a shared sport, e.g. "UEFA Euro 2024" and "UEFA Euro 2024 final").

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
| sports | 49 | 88 | 31 | 46 | 41 | 103 | 52 | 48 | 32 | 82 | 22 | 12 | 606 |
| arts_entertainment | 7 | 2 | 2 | 3 | 11 | 4 | 2 | 3 | 10 | 3 | 1 | 3 | 51 |
| **all** | 91 | 117 | 58 | 92 | 82 | 126 | 68 | 88 | 61 | 101 | 40 | 29 | 953 |

| category | events | with `question_current` | high predictability | with stale answer | with MCQ |
|---|---:|---:|---:|---:|---:|
| award | 89 | 26 | 1 | 81 | 85 |
| election | 96 | 93 | 23 | 40 | 74 |
| office_change | 169 | 127 | 0 | 151 | 166 |
| sports_result | 599 | 145 | 111 | 274 | 517 |

341 events can feed the `current`/`current_d` index: they have a `question_current` and are not high-predictability.
By topic: government 127, sports 119, elections 71, arts_entertainment 17, science_research 5,
international_relations 1, business_economy 1. (Before the review fixes there were 531; most of the 190 dropped
sports and award questions asked for a "most recent" result that a later edition had already replaced.)

### Caveats

- **Coverage is uneven.** Wikidata covers only 7 of the 15 topics, and sports makes up 64% of the events (119 of the
  341 current-question events). A paid run on this file alone mostly measures sports, government and elections;
  add the topic-stratified Current-events set (or kc events) and cap sports before using it for topic reversals. Wikidata
  has nothing for conflict, law, accidents, natural disasters, health, AI/ML, computing or space. Those topics need the
  Wikipedia Current-events items (`cutoffmix.bench.sources.wikipedia` plus `cutoffmix.bench.probegen`, which needs an
  API key) and the kc import.
- **The last months are thin.** Wikidata lags real events: 2026Q3 has 40 events and 2026Q4 has 29, against about 90
  per quarter earlier. Only 14 events fall after 2026-10-25. Events less than 14 days old are provisional: rebuild
  before use.
- **The ground truth is Wikidata's.** Data errors pass straight into the events: wrong dates, a successor that has not
  been entered yet, and so on. If Wikidata has not yet recorded a change, `question_current` can be out of date. Run
  the 5% human spot check.
- **Award months are partly inferred.** Year-precision award dates use the award's usual announcement month, so their
  month is approximate.
- **"Most recent" can still be stale** when Wikidata has neither a later edition nor enough history to see the
  period. Of 145 sports `question_current` events, 14 are more than 13 months old as of 2026-11-24; most of them are
  biennial (U19 World Cups, Ryder Cup), but about 5 annual ones (Coupe de France 2024, Bulgarian Supercup 2024,
  Champions Hockey League 2025, UEFA Youth League 2025, Liga Nacional de Hockey Hielo 2025) are stale.
- **MCQ distractors are same-type only**, so 111 events have no MCQ rather than a padded one. Person-award
  families do not know the recipient's gender beyond the award itself (actor vs actress categories are separate).
- **Labels are English** and come straight from Wikidata. Some are long ("Spain men's national football team"). Each
  answer's accepted surface forms are listed in `aliases`.
