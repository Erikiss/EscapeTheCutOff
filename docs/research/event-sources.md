## Topic-stratified event set for per-topic knowledge cutoffs: data sources tested on 2026-11-24

All sources below were tested with curl or Python from the container, using User-Agent "cutoff-mix-research/0.1 (research prototype; generic client)". Output files are under `(scratchpad)/event-sources/`, called `ES/` from here on (`ES/scripts`, `ES/sparql`, `ES/data`, `ES/samples`, `ES/ext`).

**Summary**
- **Wikipedia Current events portal (CE):** all 1,058 day pages from 2024-01-01 to 2026-11-23 were fetched in 22 batched requests. Parsing them gave **17,718 dated leaf events**, 99.8% with a news-source URL, at roughly 270–800 per month.
- **Wikipedia "Deaths in <Month> <Year>" lists:** 789–1,047 entries per month. Joining them to Wikidata matches 98.6% of entries.
- **Wikidata Query Service (WDQS):** 8 SPARQL queries ran and returned rows. Queries that scan a date range across a whole property time out.
- **Model IDs:** the three model IDs exist on OpenRouter. The user said "Gemini 6.2 Sol", but there is no Gemini "Sol" model. The listing has `openai/gpt-6.2-sol` ("OpenAI: GPT-6.2 Sol", created 2026-11-17), `google/gemini-4-neon` (2026-11-18) and `anthropic/claude-fable-5.2` (2026-11-09). All three also have `:batch` variants at half price.
- **Thin topics:** CE alone gives too few events for AI/ML, technology, science and climate. These topics need year-in-topic pages, Wikidata and deaths sliced by occupation.

### 1. Source test results

| Source / endpoint | HTTP | Result / note |
|---|---|---|
| `https://en.wikipedia.org/w/api.php?action=parse&page=Portal:Current_events/2026_March_3&prop=wikitext&format=json&formatversion=2` | 429, then 200 | Wikimedia's front proxy rate-limits clients without a token (envoy `x-envoy-ratelimited`, `Retry-After` 4–54 s). Retrying and honoring `Retry-After` works. The 429 body links https://www.mediawiki.org/wiki/Wikimedia_APIs/Rate_limits |
| Batched CE fetch: `action=query&prop=revisions&rvprop=content\|timestamp\|ids&rvslots=main&format=json&formatversion=2&maxlag=5&titles=Portal:Current events/2026 November 1\|…` (≤50 titles per request) | 200 | 50 pages per call. The whole range took 22 calls and about 6 minutes, including five 429 backoffs |
| `titles=Deaths in October 2026` (same query API) | 200 | 220 KB of wikitext. The current month redirects: "Deaths in November 2026" goes to "Deaths in 2026#November", so pass `redirects=1` and parse the `==November==` section |
| "<YEAR> in <topic>" pages (`prop=info`) | 200 | Pages exist for 2025 and 2026 "in" science, artificial intelligence, spaceflight, video games, film, music, sports, climate change, aviation, rail transport, archaeology, paleontology, literature, association football, plus "2026 in the United States", "List of large language models" and "Timeline of artificial intelligence". "<YEAR> in economics" and "<YEAR> in Formula One" do not exist |
| `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/Patrick_Stewart/daily/20261020/20261105` | 429, then 200 | Daily views, e.g. 3,461 on 2026-10-20 |
| `https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/2026/10/29` | 200 | Daily top 1000. Rank 2 is Patrick_Stewart with 2,356,552 views (a death spike). One request per day makes a cheap salience filter |
| `https://query.wikidata.org/sparql` (POST, `Accept: application/sparql-results+json`) | 200 / 502 / 504 | Selective queries take 1–44 s. Unbounded date-range scans return 502 or 504 after about 30–60 s |
| `https://www.wikidata.org/w/api.php?action=query&prop=revisions&rvdir=newer&rvlimit=1&rvprop=timestamp&format=json&formatversion=2&titles=Q141507603` | 200 (one 429) | Item creation date. Calibration: Q131318777 → 2024-11-24, Q137880562 → 2026-01-24, Q141507603 → 2026-09-20, Q142270299 → 2026-11-24, about 0.45–0.5M QIDs per month. The QID number is therefore a free proxy for "this entity is new" |
| `https://api.gdeltproject.org/api/v2/doc/doc?...` | 000 | Unusable from here: curl exit 35, TLS failure through the proxy |
| `https://data.gdeltproject.org/gdeltv2/lastupdate.txt` | 200 | Lists 15-minute files under http:// links; change them to https://. `20261124163000.export.CSV.zip` has 1,575 rows and 61 columns (CAMEO codes and SOURCEURL). These are machine-coded events with no question/answer ground truth, so use them only for salience, not as truth |
| `https://huggingface.co/api/models?author=Qwen&sort=createdAt&direction=-1&limit=5` | 200 | e.g. `Qwen/Qwen-Image-2.1-Turbo` with createdAt 2026-10-09. createdAt is when the repo was created, which can be before it went public, so these dates need care |
| `https://pypi.org/pypi/torch/json` | 200 | e.g. torch 2.15.0 uploaded 2026-10-28T15:16:30. Exact timestamps, but version bumps are predictable |
| `https://huggingface.co/datasets/apoorvumang/knowledge-cutoff-benchmark/resolve/main/events.jsonl` | 200 | 330 events, 11 per month, covering **only 2024-01 to 2026-06**. Categories: 201 death, 111 office_change, 10 control_alive, 8 fake_event. Predictability: 102 low, 178 medium, 50 high |
| `https://openrouter.ai/api/v1/models` | 200 | 470 models (pricing in section 7) |

**robots.txt and User-Agent.** en.wikipedia.org/robots.txt has `User-agent: *` with `Disallow: /w/` and `Disallow: /api/`. query.wikidata.org/robots.txt disallows `/sparql`. These rules target crawlers; programmatic API use falls under Wikimedia's API etiquette and User-Agent policy. For production:
- Use a descriptive User-Agent with a project URL (not personal data) and keep concurrency at 1.
- Send `maxlag=5`, honor `Retry-After`, and back off exponentially.
- Batch 50 titles per request.
- For bulk work, use dumps.wikimedia.org, which returned 200.
- Optionally use a Wikimedia API token for higher limits. The current token and limit details are UNVERIFIED.

### 2. Current events portal: format and parser (`ES/scripts/fetch_ce.py`, `ES/scripts/parse_ce.py`)

**Format seen in the wikitext:**
- Content sits inside `{{Current events|year=2026|month=03|day=3|content= … }}`. Newer pages instead use separate `{{Current events|…|top=yes}}` and `…|bottom=yes}}` lines.
- Category headings appear as a bold line, e.g. `'''Armed conflicts and attacks'''`. Sometimes the heading is glued to a comment: `<!-- All news items below this line -->'''Armed conflicts and attacks'''`.
- Items are nested bullets: `*[[Story]]` / `**[[Sub-story]]` / `***Leaf sentence. [https://… (Reuters)]`. Source links take the form `[URL (Outlet)]`.
- One non-canonical heading appeared in the whole range (`'''Music'''`).

**Parsing algorithm:**
1. Strip `<!-- -->` comments, replacing each with a newline.
2. Remove the wrapper templates, matching only single-line patterns `\{\{Current events\|[^{}\n]*?content=` and `\{\{Current events\|[^{}\n]*?\}\}`. A multi-line regex here deleted whole pages and cut the event count from 17.7k to 7.5k.
3. Match heading lines with `^\s*(?:'''(.+?)'''|;\s*(.+?))\s*$` and normalize them to the 10 canonical CE categories using an alias table plus `difflib` with cutoff 0.8.
4. Keep a bullet-depth stack. A node is a **leaf** if the next line's depth is ≤ its own depth. Emit leaves that have a source link or more than 60 characters of text.
5. The `story_path` is the first wikilink of each ancestor bullet.
6. Extract the wikilink targets (entities) and the `[url (outlet)]` sources, and record `rev_ts`/`revid` so the revision can be pinned.

**Output:** `ES/data/ce_events.jsonl`, 17,718 records. Each record has `date, ce_category, story_path, text, links, sources[{url,outlet}], depth, rev_ts, revid`.
- Category totals: Armed conflicts 5,930; Disasters 2,879; Law & crime 2,784; Politics & elections 2,101; International relations 1,330; Business 779; Sports 712; Health & environment 541; Science & technology 353; Arts & culture 302; unassigned 7.
- Events per month: 266–615 from 2024-01 to 2026-02, 727–801 from 2026-03 to 2026-07, 543–638 from 2026-08 to 2026-10, and 417 for 2026-11-01 to 2026-11-23.
- Top outlets: Reuters 3,399, AP 2,373, Al Jazeera 1,488, BBC 1,203.
- Ongoing-story clustering is heavy: "Middle Eastern crisis (2023–present)" alone is 1,403 leaves.

**Settling.** CE pages keep being edited for 1–2 weeks: pages for Nov 1–17 were last edited Nov 8–23. Treat events less than 14 days old as provisional and re-fetch them at T+14 days.

**Sample record (2026-03-03, Armed conflicts):** "An Iranian drone strike damages Al Minhad Air Base, the headquarters of the Australian military's Joint Task Force 633…", source reuters.com/…/australia-says-defence-personnel-safe-after-drone-attack-near-dubai-air-base-2026-03-03/.

**First-pass topic rules** (`ES/scripts/topic_rules.py` → `ES/data/ce_events_topics.jsonl`) use keyword regexes over text, story_path and links. Mean events per month over the last 12 months:

| Topic | Mean / month |
|---|---|
| conflict | 217 |
| law | 85 |
| accidents | 69 |
| diplomacy | 43 |
| government | 33 |
| elections | 32 |
| sports | 29 |
| natural disasters | 26 |
| business | 25 |
| health | 24 |
| arts | 11 |
| climate | 5.1 |
| macro / trade | 5.2 |
| science | 4.6 |
| space | 3.9 |
| AI/ML | 3.8 |
| tech | 2.8 |

The rules make clear errors. "G7 releases oil reserves" was tagged business instead of macro, and "Captains Regent sworn in" was tagged elections instead of government. Use the rule labels only as weak supervision for an LLM or embedding classifier. The 17.7k labelled records can also train the reasoning-window → topic classifier.

### 3. Wikidata SPARQL: queries that ran, with row counts

Files are `ES/sparql/*.rq`, raw results are in `ES/data/wd/`, and samples are in `ES/samples/wd_*_sample.json`. Runner: `python3 -I ES/scripts/run_sparql.py Q.rq OUT.json`, which POSTs to WDQS and retries on 429/503.

**Q2: new heads of state and government, 124 rows / 118 people (78 in 2025, 46 in 2026), 7.7 s.**
```sparql
SELECT DISTINCT ?person ?personLabel ?office ?officeLabel ?country ?countryLabel ?start ?replaces ?replacesLabel WHERE {
  ?country wdt:P31 wd:Q6256 .
  { ?country wdt:P1906 ?office } UNION { ?country wdt:P1313 ?office }
  ?person p:P39 ?st . ?st ps:P39 ?office ; pq:P580 ?start .
  FILTER(?start >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?start < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  OPTIONAL { ?st pq:P1365 ?replaces }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?start)
```
Sample rows:
- Aleksandar Čupić, President of Serbia, 2026-11-19, replaces Ana Brnabić
- Ülle Madise, President of Estonia, 2026-10-12, replaces Alar Karis
- Fatima Ezzahra El Mansouri, Prime Minister of Morocco, 2026-09-29, replaces Aziz Akhannouch
- Haakon VIII, Monarch of Norway, 2026-08-28, replaces Harald V

The P1365 "replaces" qualifier gives the stale-state distractor for free.

**Q3: sports winners, 2,683 rows / 2,067 editions / 1,676 with an English label, 44 s.** Labelled editions per month: median 65, minimum 26.
```sparql
SELECT DISTINCT ?ed ?edLabel ?comp ?compLabel ?winner ?winnerLabel ?date ?sport ?sportLabel WHERE {
  { ?ed wdt:P3450 ?comp } UNION { ?ed wdt:P179 ?comp . ?comp wdt:P31/wdt:P279* wd:Q15275719 }
  ?ed wdt:P1346 ?winner .
  { ?ed wdt:P582 ?date } UNION { ?ed wdt:P585 ?date }
  FILTER(?date >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?date < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  OPTIONAL { ?comp wdt:P641 ?sport }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?date) LIMIT 5000
```
Sample rows:
- 2026 CME Group Tour Championship: Atthaya Thitikul, 2026-11-22, golf
- 2026 F1 Academy season: Alisha Palmowski, 2026-11-21

Noise to filter:
- Unlabelled Ligue 2/3 match items: require an English label and minimum sitelinks.
- A suspicious date: the 2026 IIHF Women's World Championship is dated 2026-11-16. Cross-validate dates.

Sports breakdown: football 375, basketball 276, ice hockey 112, baseball 83.

**Q4: software releases, 5,000 rows (hit the LIMIT; covers 2026-11-24 back to 2025-06-23), 12.3 s.**
```sparql
SELECT ?sw ?swLabel ?clsLabel ?version ?date ?sitelinks WHERE {
  VALUES ?cls { wd:Q9143 wd:Q9135 wd:Q6368 wd:Q131669 wd:Q271680 wd:Q188860 wd:Q7889 wd:Q218616 }
  ?sw wdt:P31 ?cls ; wikibase:sitelinks ?sitelinks . FILTER(?sitelinks >= 10)
  ?sw p:P348 ?st . ?st ps:P348 ?version ; pq:P577 ?date .
  FILTER(?date >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?date < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?date) LIMIT 5000
```
- The class labels returned confirm the QIDs: programming language, operating system, web browser, Linux distribution, software framework, software library, video game, proprietary software.
- Sample rows: Firefox 161.0 on 2026-11-24, Chrome 158.0.8127.89 on 2026-11-23, Selenium 4.54.0 on 2026-11-23.
- Keeping only major or minor versions without pre-release tags (`^\d+\.\d+(\.0)?$`, excluding a/b/rc/esr) leaves 1,013 rows across 237 programs, a median of 56 per month.
- These dates are mostly **high predictability**, so use them only for "what feature or version" questions.
- Without the `VALUES ?cls` entry point, the query timed out (504).

**Q5a: films, 1,843 rows / 725 films (213 with ≥20 sitelinks), 11.5 s.**
```sparql
SELECT ?w ?label ?d ?sitelinks WHERE {
  ?w wdt:P31 wd:Q11424 ; wdt:P577 ?d .
  FILTER(?d >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?d < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  ?w wikibase:sitelinks ?sitelinks . FILTER(?sitelinks >= 8)
  OPTIONAL { ?w rdfs:label ?label FILTER(LANG(?label)="en") } } ORDER BY DESC(?d)
```
- Sample row: The Hunger Games: Sunrise on the Reaping (Q126390739), 2026-11-20, 29 sitelinks.
- Films can have several P577 dates (by region or festival). Take the minimum P577 per film on the client side and drop films with a low QID (old items with re-release dates, e.g. Q768223 "The Secret Heart" dated 2026-11-21).
- Querying films and albums together with GROUP BY/MIN timed out (504).

**Q5b: albums, 194 rows / 176 albums, 27 s.** Use P31 Q482994 (album). With P31 Q208569 (studio album) the query returned 0 rows, because "studio album" is now recorded in P7937 (form of creative work).
```sparql
SELECT ?w ?label ?d ?sitelinks ?perfLabel WHERE {
  ?w wdt:P31 wd:Q482994 ; wdt:P577 ?d .
  FILTER(?d >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?d < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  ?w wikibase:sitelinks ?sitelinks . FILTER(?sitelinks >= 5)
  OPTIONAL { ?w rdfs:label ?label FILTER(LANG(?label)="en") }
  OPTIONAL { ?w wdt:P175 ?perf . ?perf rdfs:label ?perfLabel FILTER(LANG(?perfLabel)="en") } } ORDER BY DESC(?d)
```
Sample row: "Grand Theft Auto VI: The Album" (Q141489183), 2026-11-19.

**Q7: elections with a winner, 227 rows / 73 elections (28 with ≥10 sitelinks), 1.3 s.**
```sparql
SELECT DISTINCT ?e ?eLabel ?d ?winner ?winnerLabel ?country ?countryLabel ?sitelinks WHERE {
  ?e wdt:P991 ?winner ; wdt:P585 ?d .
  FILTER(?d >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?d < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  ?e wikibase:sitelinks ?sitelinks . FILTER(?sitelinks >= 3)
  OPTIONAL { ?e wdt:P17 ?country }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?d)
```
Sample rows: 2026 Brazilian presidential election, 2026-10-25, Lula; 2026 US Senate election in Georgia, 2026-11-03, Jon Ossoff.

**Q8: major awards, 45 rows, 14.3 s.** Award labels returned confirm the QIDs.
```sparql
SELECT ?award ?awardLabel ?who ?whoLabel ?d WHERE {
  VALUES ?award { wd:Q38104 wd:Q37922 wd:Q44585 wd:Q35637 wd:Q80061 wd:Q47170 wd:Q166177 wd:Q160082 wd:Q185667 wd:Q28444 wd:Q41254 wd:Q103360 wd:Q103916 wd:Q131520 }
  ?who p:P166 ?st . ?st ps:P166 ?award ; pq:P585 ?d .
  FILTER(?d >= "2025-01-01T00:00:00Z"^^xsd:dateTime && ?d < "2026-11-25T00:00:00Z"^^xsd:dateTime)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". } } ORDER BY DESC(?d)
```
Sample rows: Booker Prize, M. John Harrison, 2026-11-09; Academy Award for Best Actor, Michael B. Jordan, 2026-03-15.

**Q1: deaths enriched by occupation, using a list of titles as the entry point.** Template: `ES/sparql/q1_deaths_enrich_TEMPLATE.rq`; runs use chunks of ≤200 titles and take 3–9 s each.
```sparql
SELECT ?name ?person ?dod ?sitelinks ?occL ?ctyL WHERE {
  VALUES ?name { "Patrick Stewart"@en "Yasuo Fukuda"@en … }
  ?article schema:about ?person ; schema:isPartOf <https://en.wikipedia.org/> ; schema:name ?name .
  ?person wikibase:sitelinks ?sitelinks .
  OPTIONAL { ?person wdt:P570 ?dod }
  OPTIONAL { ?person wdt:P106 ?occ . ?occ rdfs:label ?occL FILTER(LANG(?occL)="en") }
  OPTIONAL { ?person wdt:P27 ?cty . ?cty rdfs:label ?ctyL FILTER(LANG(?ctyL)="en") } }
```
Results for October 2026:
- 789 list entries; 778 matched a QID; 603 have P570; **553 have P570 equal to the list date (91.7%)**. Use agreement between the two sources as a validation gate.
- Notability by sitelinks: ≥5: 334, ≥10: 112, ≥20: 32, ≥40: 13.
- Occupation groups at ≥10 sitelinks: arts/media 43, sports 37, politics 21, science 7, religion 3.
- Top rows: Patrick Stewart Q16296, died 2026-10-28, 93 sitelinks; Yasuo Fukuda Q101137, 2026-10-22; Claude Cohen-Tannoudji Q190697, 2026-10-22; William Nordhaus Q562481, 2026-10-30.
- The plain date-range version (any person with P570 in a given month) timed out every time, even at one week (502/504).

**Q6: ownership changes, 24 rows only, 14.5 s.** The query takes instances of Q891723/Q4830453/Q6881511 with ≥15 sitelinks and checks P127/P749 with a P580 start time. Sample rows: Smartwings ← Pegasus Airlines (2026-10-03); Starz Entertainment ← Allen Media Group (2026-03-06). **Wikidata is too sparse for M&A**; use CE Business and economy instead. The unrestricted version timed out (504).

**WDQS rules learned from these runs:**
1. Always start from a selective set: VALUES of classes, offices, awards, or enwiki titles via `schema:name`.
2. Never scan a date range across a whole property.
3. Prefer `rdfs:label` with a language filter. The label service sometimes returned a bare QID, e.g. "Q42" in a smoke test, and items with no English label also come back as QIDs.

### 4. Other Wikipedia sources

**Deaths lists** (`ES/scripts/parse_deaths.py`). Entry pattern: `*[[Name]], age, Nationality occupation (works), cause.<ref>[URL title]</ref>`, under `===day===` headings. Parsed entry counts: 2024-01: 1,047; 2025-03: 904; 2026-10: 789; 2026-11-01 to 11-24: 566. Age and cause feed a predictability rule: violent or accidental death, or age under 70 → low; under 85 → medium; otherwise high. For October 2026 at ≥10 sitelinks this gives 26 low, 39 medium, 47 high.

**Year-in-topic pages** (`ES/scripts/parse_yearpage.py`). The parser handles `===Month===` sections with bullets of the form `* 7 January &ndash; text` or `* 1 January` followed by `** text`.
- 2026 in science: 118 events, 6–17 per month.
- 2026 in artificial intelligence: 39 events, 1–10 per month.
- 2025 in artificial intelligence: 45 events.
- "2026 in video games" (day/text wikitables) and "2026 in climate change" (prose sections) yielded 0 events with this parser. They need a table parser, e.g. mwparserfromhell, or `action=parse&prop=text` plus HTML table parsing. That tooling is UNTESTED here.

### 5. Proposed taxonomy: 16 topics, plus a separate region tag

Region values: US, Europe, East Asia, South Asia, MENA, Sub-Saharan Africa, LatAm, Oceania. Derive region from P17 on linked entities, or from an LLM tag.

| # | Topic | Main sources | Measured raw per month (last 12 months) | Usable per month after filters (estimate) |
|---|---|---|---|---|
| T01 | armed_conflict_security | CE Armed conflicts | 217 | 40–60 (keep discrete new strikes and attacks; drop "death toll rises" updates) |
| T02 | international_relations | CE International relations | 43 | 15–25 |
| T03 | government_officeholders | CE Politics (non-election), WD Q2 (~5/mo), CEO changes, deaths of politicians (21) | ~60 | 20–35 |
| T04 | elections_referendums | CE elections (32), WD Q7 (~3/mo), seasonal | ~35 | 10–20 |
| T05 | law_courts_crime | CE Law & crime | 85 | 25–40 |
| T06 | accidents_humanmade_disasters | CE Disasters (non-natural) | 69 | 25–35 |
| T07 | natural_disasters_climate_env | CE natural (26) and climate (5), climate year-page | ~33 | 12–20 |
| T08 | health_medicine | CE Health (24) | 24 | 8–15 |
| T09 | business_economy | CE Business (25), macro (5), WD Q6, deaths of business people | ~32 | 12–20 |
| T10 | ai_ml | CE (4), "YYYY in AI" (~3–4), "List of large language models", HF Hub | ~10–15 | 5–10 (pool into 2-month bins) |
| T11 | computing_software_internet | CE tech (3), WD Q4 majors (56, mostly predictable), PyPI | ~60 | 8–15 |
| T12 | space_astronomy | CE (4), "YYYY in spaceflight" launch tables, science page | ~30 | 8–15 |
| T13 | science_research | CE (5), "YYYY in science" (~10), WD Q8 Nobel, deaths of scientists (7) | ~25 | 8–15 |
| T14 | sports | CE (29), WD Q3 (65), deaths of athletes (37) | ~130 | 30–50 |
| T15 | arts_entertainment | CE (11), WD films (~70 raw / ~9 with ≥20 sitelinks), albums (~8), WD Q8, video games page, deaths of artists (43) | ~140 | 30–50 |
| T16 | notable_deaths (also split by occupation into T03/T09/T13/T14/T15) | Deaths lists | ~110 at ≥10 sitelinks | 60–100, about 55% low or medium predictability |

### 6. Automatic pipeline

**1. Fetch (per day or month, idempotent, revision pinned)**
- CE batches (22 requests for the whole range).
- Deaths lists: one page per closed month; the current month comes from "Deaths in YYYY#Month".
- Year-in-topic pages, with `prop=pageprops&ppprop=wikibase_item` to get QIDs.
- WDQS Q1–Q8.
- Daily pageview top-1000 lists (1,058 requests for the whole range).
- Map linked titles to QIDs with the `schema:name` VALUES join (chunks of 200), not with per-title API calls.

**2. Normalize** into one record schema:
`{event_id, date, date_precision, topic, region, source_kind, fact, subject_entities[{qid,title}], prior_state, post_state, source_urls, wp_revid, wd_ids, notability{sitelinks, max_daily_views_±3d, in_top1000}, novelty{qid_created_at}, predictability{rule, oldmodel_acc}, probes{}, validation{}}`

**3. Filter for notable, low-predictability events**
- Notability: a linked non-generic entity is in the daily top 1000 within ±3 days, or has ≥10 sitelinks.
- Novelty: the story-leaf entity's QID was created within ±14 days of the event (use the QID-to-date calibration above). This marks a genuinely new event rather than an ongoing-story update.
- Drop text matching `death toll|rises to|continues|ongoing` when the event sits deep in a big ongoing story.
- Deduplicate: same story leaf and entity Jaccard > 0.5 within 7 days → keep the earliest.
- Predictability tag: rules plus an LLM tag (low/medium/high). Then apply an **external predictability check**: run the MCQ and direct probes on old-cutoff models available on OpenRouter (`meta-llama/llama-3.1-70b-instruct`, `qwen/qwen-2.5-72b-instruct`, `openai/gpt-4o-2024-05-13`; their pre-2024 cutoffs come from vendor docs and are UNVERIFIED here). If ≥2 of 3 answer correctly, mark the event high-predictability or distractor-weak, and regenerate it or exclude it from the cutoff curve. This gives a predictability index based on evaluation, not self-report.

**4. Generate probe triplets with an LLM.** The generator gets the fact text, prior and post state, and same-type candidate entities pulled from Wikidata (candidates in the election via P726, the predecessor via P1365, rival teams, and so on). It returns JSON:
`{subject, q_dated, a_dated, q_current, a_current, stale_answer, mcq_question, mcq_choices[4], mcq_answer, distractor_sources, predictability, reason}`

Rules:
- `q_dated` gives the month and year but never the answer.
- `q_current` has no date or recency hint, matching the benchmark's leakage rule.
- MCQ: one correct option, one stale/pre-event option, and two plausible same-type options taken from real candidates.
- Letter position is set by hash(event_id) so answers are balanced.
- Use a generator from outside the three subject models, or rotate generators, to avoid style bias.

Illustrative triplet, written by hand from Q2 row Ülle Madise:
- q_dated: "Who became President of Estonia in October 2026?"
- q_current: "Who is the President of Estonia?" (stale answer: Alar Karis)
- MCQ: A) Alar Karis, B) Ülle Madise, C) Kersti Kaljulaid, D) Kaja Kallas

**5. Controls, about 15% per topic**
- **fake_event**: perturb a real event by swapping in a same-type entity or outcome, then check it is false against Wikidata and the CE/Deaths lists (for example, the person has no P570 and is in no Deaths list from 2024 to 2026).
- **control_alive / unchanged**: people with ≥60 sitelinks, no P570 and not in any Deaths list; incumbents whose P39 has no P582 end date.
- **future_scheduled**: events after 2026-11-24, such as elections or films with a later P585/P577. Ask in past tense; the correct answer is "has not happened". This catches confabulation.

**6. Validation gates**
- V1 cross-source date agreement within ±1 day (Wikidata vs list vs CE).
- V2 settling: only use events at least 14 days old, and re-fetch to drop items that were deleted.
- V3 format checks: the answer is grounded in the fact; no answer string in the question; no year or month tokens in `q_current`; 4 unique options with balanced lengths.
- V4 a verifier model *with* the fact must answer all three probes correctly.
- V5 distractors must be false according to Wikidata.
- V6 human spot-check of 5% of each topic-month.

**7. Scoring**
- Grade MCQs deterministically. Grade direct probes three ways (correct / incorrect / abstain), as the benchmark does, using a judge that is given the ground truth. Use non-subject judges or two judges from different families. `anthropic/claude-opus-4.8` (the benchmark's judge) is on OpenRouter at $5/$25 per million tokens, but it is from the same family as Fable.
- Per model and topic, fit a logistic curve of P(correct) against month over low and medium predictability events only, with hierarchical pooling across topics. Output a topic cutoff with a confidence interval, using the benchmark's `last_above`/`crossover` definitions.

### 7. Sample sizes and cost

**Event budget.** Use 25 events per topic-month in the 12 critical months (2025-12 to 2026-11) and 8 per topic-month in the 23 earlier months. That is 16 × (300 + 184) ≈ **7.7k events**, about 8.9k with controls.
- At 25 MCQs and p ≈ 0.5, the standard error per bin is about 0.1. The logistic fit pools across months.
- T10, T11, T12 and T13 cannot reach 25 per month. Use 2-month bins for these, or accept wider intervals.

**Probing cost.** About 26.7k prompts per model, assuming ~120 input and ~250 output tokens with low reasoning effort. OpenRouter prices per million tokens (input/output):

| Model | Standard | `:batch` | Estimated cost (batch) |
|---|---|---|---|
| `anthropic/claude-fable-5.2` | $8 / $40 | $4 / $20 | ≈ $150 |
| `openai/gpt-6.2-sol` | $2 / $10 | $1 / $5 | ≈ $40 |
| `google/gemini-4-neon` | $3 / $15 | $1.5 / $7.5 | ≈ $55 |

The judge adds about $30–60. **Do not enable web search**: no `:online` suffix and no web plugin, because web search is priced separately per request.

### 8. Relevant papers (all checked via the export.arxiv.org API)
- **2403.12958**, "Dated Data: Tracing Knowledge Cutoffs in Large Language Models". Defines an *effective cutoff* that applies separately to sub-resources and topics. This is the closest prior work to the "virtual, topic-specific cutoff" idea.
- **2410.09870**, "ChroKnowledge: … Chronological Knowledge of Language Models in Multiple Domains" (ChroKnowBench, multi-domain).
- **2411.08324**, "Are LLMs Prescient? A Continuous Evaluation using Daily News as the Oracle" (Daily Oracle: QA generated automatically from daily news).
- **2005.10070**, a multi-document summarization dataset built from the Wikipedia Current Events Portal (prior use of the CE portal).
- **2605.21521**, "Same Pipeline, Opposite Conclusions: Sample-Surface Effects in Breaking-News Latency". Samples CE events ranked by pageviews, which supports using pageviews for salience.
- **2605.03762**, OracleProto (forecasting evaluation using knowledge cutoffs and temporal masking).
- **2510.02340**, "Can Prompts Rewind Time for LLMs?"
- **2601.13717**, "Simulated Ignorance Fails…"
- **2310.03214** FreshLLMs/FreshQA; **2207.13332** RealTime QA; **2205.11388** StreamingQA; **2204.14211** TemporalWiki; **2311.08106** "Carpe Diem" (EvolvingQA); **2106.15110** TempLAMA; **2402.16797** "Set the Clock"; **2312.13401** "Time is Encoded in the Weights…"; **2102.01951** "Mind the Gap"; **2206.15474** Autocast.

### 9. Caveats and UNVERIFIED items
- The Wikimedia rate-limit and token tiers have not been checked against current documentation. Only the observed behavior is known: 429 with `Retry-After` 4–54 s.
- HF `createdAt` may be earlier than the public release date.
- Wikidata data quality issues seen: the suspicious IIHF date, 8.3% P570 mismatches against the Deaths list, and regional P577 dates for films. Every fact needs agreement from two sources.
- The cutoffs of the old baseline models are vendor claims and have not been checked here.
- GDELT's DOC API cannot be reached from this container.

### Files
All under `(scratchpad)/event-sources/`:
- **Scripts:** `scripts/fetch_ce.py`, `scripts/parse_ce.py`, `scripts/fetch_titles.py`, `scripts/parse_deaths.py`, `scripts/parse_yearpage.py`, `scripts/run_sparql.py`, `scripts/topic_rules.py`
- **SPARQL:** `sparql/q2_heads.rq`, `sparql/q3_sports.rq`, `sparql/q4_software.rq`, `sparql/q5_films.rq`, `sparql/q5_albums.rq`, `sparql/q6_acquisitions.rq`, `sparql/q7_elections.rq`, `sparql/q8_awards.rq`, `sparql/q1_deaths_enrich_TEMPLATE.rq`
- **Data:**
  - `data/ce_raw/` (22 raw batches)
  - `data/ce_events.jsonl` (17,718 events)
  - `data/ce_events_topics.jsonl`
  - `data/deaths_2024-01.jsonl`, `data/deaths_2025-03.jsonl`, `data/deaths_2026-10.jsonl`, `data/deaths_2026-11.jsonl`
  - `data/year_events.jsonl`
  - `data/wd/*.json`
- **Samples:** `samples/ce_events_topics_sample.jsonl`, `samples/deaths_2026-10_enriched.json`, `samples/wd_q2_heads_sample.json` (and the other `samples/wd_q*_sample.json` files), `samples/yearpage_events_sample.jsonl`
- **External downloads:** `ext/kcb/events.jsonl` (benchmark), `ext/misc/openrouter_models.json`, `ext/misc/pageviews_*.json`, `ext/gdelt/export.zip`, `ext/arxiv/*.xml`