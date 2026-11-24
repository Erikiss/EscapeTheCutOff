# Test fixtures (WP3 data side)

Small offline samples for `tests/test_kc_import.py`, `tests/test_wikidata.py` and `tests/test_wikipedia.py`.
Tests never touch the network.

| File | Content | Source and license |
|---|---|---|
| `kc_events_sample.jsonl` | 14 events (deaths, office changes, one alive control, one fake event) | apoorvumang/knowledge-cutoff-benchmark `events.jsonl`, CC BY 4.0 |
| `kc_report_answers_sample.json` | Answers of `claude-fable-5.2`, `gpt-6.2-sol` and `gpt-4o` to those events (responses cut to 300 chars) | apoorvumang/knowledge-cutoff `report_answers.json`, CC BY 4.0 |
| `ce_2026_March_3.wikitext` | Portal:Current events/2026 March 3, trimmed to 14 leaf items | English Wikipedia, CC BY-SA 4.0 |
| `deaths_october_2026.wikitext` | "Deaths in October 2026", lead plus the first entries of days 1 and 2 | English Wikipedia, CC BY-SA 4.0 |
| `deaths_2026_year.wikitext` | "Deaths in 2026", the `==November==` section (first entries) and the start of "Previous months" | English Wikipedia, CC BY-SA 4.0 |
| `wd_*.json` | WDQS result documents (SPARQL 1.1 JSON) shaped like the answers to the queries in `cutoffmix.bench.sources.wikidata.QUERIES` | Hand-written from Wikidata rows (CC0); `Example…`, `Team Alpha`, `Second Regent` and QIDs `Q9000xx`/`Q1000xxx` are synthetic edge cases |
