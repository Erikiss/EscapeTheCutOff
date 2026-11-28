# Messergebnisse

## `papers_probes_2026-11-25.jsonl`

1323 bewertete Antworten (`cutoffmix.types.ProbeResult`, eine pro Zeile) der drei Modelle auf das AI-Paper-Probe-Set
`data/events/papers_events.jsonl` (210 Paper + 21 Kontrollen), Sonden `current_d` (Erstautor, offene Frage mit
Datum im Systemprompt) und `mcq` (4 Optionen; keine MCQ für Kontrollen). Erhoben am 25.11.2026 über OpenRouter mit
`reasoning: {effort: low}`, `max_tokens 1500`.

- Bewertung: Regelbewerter (`cutoffmix bench grade`), danach 57 unklare Antworten per Judge (`gpt-6.2-sol`,
  `grader: judge:gpt-6.2-sol`). 0 Zeilen blieben unbewertet.
- Kosten: ≈ 5,9 $ (Fable 3,2 $, Neon 2,2 $, Sol 0,3 $, Judge ≈ 0,1 $).
- `usage` enthält die Token-Zahlen und Kosten pro Aufruf, `latency_s` die Dauer.
- Reproduktion des Index: siehe README, Abschnitt „Gemessene Zelle ai_ml“.

Bekannte Einschränkungen: Fünf Antworten von Sol nennen ein Team („Wan Team“, „StepFun Team“ …) statt der auf
arXiv als Erstautor geführten Person und zählen als falsch; die Interpretability-Teilmenge (34 Paper) ist für eine
eigene Kurve zu klein; Paper, die nach v1 umbenannt wurden, sind nicht enthalten (siehe `../events/README_papers.md`).

## `rally_probe_2026-11-28/`

Live-Test des Rally-Presets (`docs/DESIGN_rally.md`, Abschnitt 4.1) am 28.11.2026: zwei Rallys Fable 5.2 ↔ Gemini 4
Neon mit der vollen Rally-Vorgabe für Fable, Frage „Welche ist die neueste stabile Python-Version, und wann ist sie
erschienen?“. Jeder Trace stammt von `cutoffmix run --preset rally --config <probe_*.yaml> --json` und enthält die
Notizen, die Antwort, die `rally`-Kennzahlen sowie Nutzung und Latenz pro Aufruf.

- `fable_first.json` (`probe_fable_first.yaml`): Ring Fable → Neon, 3 Aufrufe, 0,0837 $.
- `neon_first.json` (`probe_neon_first.yaml`): Ring Neon → Fable, 3 Aufrufe, 0,0833 $.

Beide Male gab es keine Sperre und keinen Fehler. Die Versions- und Datumsangaben in den Notizen sind Aussagen der
Modelle und nicht geprüft.
