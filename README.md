# cutoffmix

Eine gemeinsame Gedankenkette, die abschnittsweise von drei Modellen geschrieben wird. Vor jedem Abschnitt
entscheidet ein Router, welches Modell weiterschreibt: das, dessen **themenspezifischer effektiver Wissensstand**
(„virtueller Cutoff“) für das aktuelle Thema am nächsten an heute liegt. Die Cutoffs werden **extern gemessen**,
mit der Methodik des [Knowledge-Cutoff-Bench](https://apoorvumang.github.io/knowledge-cutoff/). Was die Modelle
über sich selbst sagen, spielt keine Rolle.

| Schlüssel | OpenRouter-ID | Provider (fest gepinnt) |
|---|---|---|
| `claude-fable-5.2` | `anthropic/claude-fable-5.2` | anthropic, azure |
| `gpt-6.2-sol` | `openai/gpt-6.2-sol` (*„Gemini 6.2 Sol“ ist ein OpenAI-Modell*) | openai, azure |
| `gemini-4-neon` | `google/gemini-4-neon` | google-vertex |

`docs/DESIGN.md` ist der vollständige Entwurf, inklusive der gegen die Live-Endpunkt-Metadaten geprüften API-Fakten.

Zwei Presets verändern den Ablauf (beide optional, mit `--preset`): **`explore`** schreibt strukturierte Notizen mit
fachfremden Blickwinkeln, und **`rally`** lässt zwei oder drei Modelle die Notizen im Ring weiterreichen, damit jedes
ergänzt, was es Neueres weiß (siehe unten, „Rally-Preset“).

## Ehrlicher Stand

- **Mit simulierten Modellen (`--mock`) läuft alles:** Relay, Router, Web-UI, OpenAI-kompatible API, Benchmark-Pipeline,
  Schätzer. Alle Tests laufen offline.
- **Live geprüft am 24.11.2026** über OpenRouter:
  - Alle drei Modelle wurden einzeln aufgerufen. Request-Format, Provider-Pinning und Stopp am Sentinel funktionieren,
    und die Kosten kommen pro Aufruf an.
  - Ein voller Relay-Lauf mit dem Prior-Index (Policy `softmax`): Erwartungsgemäß schrieb nur Sol, nach 17 s lag eine
    deutsche Antwort vor.
  - Ein Lauf mit Rotation über alle drei Modelle (Policy `round_robin`): 6 Abschnitte, 5 Wechsel, 57 s, 0,15 $. Die
    Modelle greifen die Notizen der anderen auf und korrigieren sie. Bei Artemis II (April 2026) brachte Fable Details
    ein, die Sol und Neon zunächst als unbelegt markiert hatten.
  - Die echten SSE-Mitschnitte liegen als Test-Fixtures in `tests/fixtures/live_sse/`.
  - Ein Lauf mit dem **Explore-Preset** (`--policy round_robin`, Frage zu Lithium-Ionen-Akkus bei Kälte): 11 Abschnitte,
    139 s, 0,29 $. Fable brachte einen weiten Blickwinkel ein (Erzähltheorie: der „unzuverlässige Erzähler“ als Bild
    für die Reichweitenanzeige bei Kälte), Sol prüfte ihn und behielt ihn mit einer konkreten Testidee, Sol korrigierte
    später eine zu starre 0-°C-Grenze für Lithium-Plating, Neon schrieb die Antwort mit `Basis:` und `Not resolved:`.
  - **Beobachtung zu Fable im Explore-Modus:** Anthropics Eingangsfilter sperrte 2 von 4 Fable-Aufrufen mit der
    Kategorie „reverse engineering or duplicating model outputs“. Das Relay hat jeweils umgeroutet und Fable nach der
    zweiten Verweigerung für den Lauf deaktiviert. 17 Cent-Aufrufe zur Eingrenzung zeigen: Der Standard-Prompt geht
    immer durch, auch mit derselben Frage; die Explore-Vorgabe (beschriftete Punkte, Liste der noch fehlenden Punkte)
    löst die Sperre fast immer aus, mit vorhandenen Notizen seltener. Welches Wort genau, ließ sich nicht isolieren;
    es wirkt wie ein Schwellenwert-Klassifikator, den die Schema-Anweisung überschreitet. Deshalb bekommt Fable im
    Explore-Modus eine abgeschwächte Vorgabe (siehe unten).
- **Rally-Preset (Schaukel zwischen den Modellen): nur offline geprüft.** Ablauf, Ring-Reihenfolge, Abbruchregel,
  Kennzahlen, CLI, Server und UI laufen gegen die simulierten Modelle; mit den echten drei Modellen wurde das Preset
  **noch nicht** gelaufen (kein Schlüssel in der Entwicklungsumgebung). Offen sind vor allem, ob Fables Eingangsfilter
  die Rally-Vorgabe durchlässt (beim Explore-Preset sperrte er) und ob die Modelle `Nothing newer:` ehrlich schreiben.
- **Latenz (live gemessen):**

  | Modell | Zeit bis zum ersten Token | Dauer pro Abschnitt | Kosten pro Abschnitt |
  |---|---|---|---|
  | GPT-6.2 Sol | ≈ 1,1 s | ≈ 4,7 s | ≈ 0,4–0,8 ct |
  | Claude Fable 5.2 | ≈ 6,3 s | ≈ 9,8 s | ≈ 1,7–2,2 ct |
  | Gemini 4 Neon | ≈ 7,2 s | ≈ 9,2 s | ≈ 0,9–1,6 ct |

  Alle drei denken vor jeder Ausgabe verdeckt; die niedrigste mögliche Stufe ist `low`. Ein Wechsel kostet deshalb
  jedes Mal diese Anlaufzeit. Schneller als ein Wechsel pro Abschnitt geht über diese APIs nicht.
- **Der mitgelieferte Index ist für 14 Themen ein Prior und für `ai_ml` eine Messung.** Der Prior-Teil stammt aus
  den öffentlichen Daten des Knowledge-Cutoff-Bench (bis **Juni 2026**), dort liegt GPT-6.2 Sol in jedem Thema vorn.
  Die Zelle `ai_ml` wurde am 25.11.2026 mit 210 AI-Papern gemessen: dort liegt **Fable** klar vorn. Für KI-Fragen
  schreibt also Fable, für Nachrichten-Themen Sol.
- **Themen-Umkehrungen** (z. B. Fable bei KI-Themen, Neon bei Raumfahrt) lassen sich erst mit einem neuen,
  themenstratifizierten Benchmark-Lauf über Juli–November 2026 messen. Die 953 Wikidata-Ereignisse dafür liegen in
  `data/events/`. Der Lauf kostet Geld (siehe unten) und wird nur mit Freigabe gestartet.
- **Wechsel pro Abschnitt, nicht pro Token.** Die Modelle teilen über diese APIs keine nativen Thinking-Tokens: Jeder
  Aufruf denkt verborgen, und dieses Denken wird verworfen. Geteilt wird ein **sichtbarer Notizstrom**. Jeder Abschnitt
  ist standardmäßig etwa 128 Token lang, und nach jedem Abschnitt kann das Modell wechseln. Gemäß deiner Prämisse
  setzen die Modelle die Notizen des jeweils anderen fort.

## So funktioniert es

```
Frage ──► [Abschnitt 1: Modell A] ─► [Abschnitt 2: Modell B] ─► … ─► FINAL ANSWER ─► Antwort (geroutetes Modell)
               ▲                         ▲
               └── Router: Themen-Posterior des Fensters × Cutoff-Index × Zeitsensitivität → Sampling
```

1. **Relay** (`cutoffmix/relay.py`). Jeder Abschnitt ist ein eigener Streaming-Aufruf über OpenRouter.
   - Die Notizen werden append-only mitgeschickt, damit Prompt-Caches treffen.
   - Ein Abschnitt endet am Sentinel `<<END_STEP>>`, nach ca. K Token oder spätestens nach 600 Zeichen.
   - Verweigert ein Modell, geht der Abschnitt an ein anderes Modell. Es wird nie umformuliert und erneut versucht.
   - Mit `--strategy speculative` schreiben bei zeitkritischen Fenstern alle drei Modelle parallel einen Entwurf, und
     der Router wählt mit Vorausschau. Das beseitigt die eine Abschnittsverzögerung der Standardstrategie, kostet aber
     etwa 3× Token.
2. **Zeitfenster.** Geroutet wird nach den letzten 3 Abschnitten (exponentiell geglättet) plus der Frage. Ein Fenster
   von etwa 300–400 Token reicht, damit das Thema stabil erkannt wird, und ist kurz genug, damit ein Themenwechsel nach
   spätestens einem Abschnitt greift.
3. **Router** (`cutoffmix/router/`).
   - Ein Themenklassifikator (15 Themen, deutsch und englisch, ohne Training) liefert den Themen-Posterior π.
   - Ein Zeitsensitivitäts-Gate liefert r(W). Bei zeitlosen Fragen („Warum schwimmt Eis?“) ist das Routing nahezu egal.
   - Aus dem Index wird pro Modell ein Score berechnet, der Unsicherheit und Halluzinationsneigung bestraft:

     `U_m = Σ_z π_z·τ_mz − κ·√(Σ_z π_z²·Var_mz) − λ·Σ_z π_z·H_mz`

     Dabei ist τ der virtuelle Cutoff in Monaten, Var dessen Unsicherheit und H die Rate veralteter oder falscher
     Behauptungen nach dem Cutoff. Standardwerte sind κ = 1 und λ = 0,5. Der H-Term entscheidet damit nur bei fast
     gleichen Cutoffs und überstimmt keinen echten Vorsprung.
4. **Sampling** für die feinen relativen Unterschiede:
   - Die Standard-Policy `softmax` zieht mit P ∝ exp((U_m + β·[aktuelles Modell]) / T), mit T = T₀ / max(r, r_min).
     Bei hoher Zeitsensitivität entscheidet sie also scharf und bei zeitlosen Fenstern locker, und der Bonus β
     verhindert sinnloses Hin- und Herspringen.
   - `thompson` zieht stattdessen einen Cutoff aus den Bootstrap-Draws des Index. Damit wird jeder Abstand von einem
     Monat oder weniger proportional zur echten Unsicherheit ausgenutzt.
   - Weitere Policies: `argmax` (mit Hysterese), `round_robin`, `random` und `fixed:<key>` als Vergleichsbaselines.
5. **Index** (`cutoffmix/bench/estimate.py`). Es ist ein hierarchisches 4PL-Modell der Trefferquote über die
   Ereigniszeit:

   `P(richtig | t) = g + (a − g)·σ((τ − t)/s)`

   - Der Ratefloor g ist 0 bei offenen Fragen und 0,25 bei Multiple Choice.
   - Gefittet wird zuerst pro Modell, dann pro (Modell, Thema) mit Themen-Offset und Shrinkage zum Modellmittel.
   - Pro Zelle entstehen Bootstrap-Draws und P(best).
   - Die primäre Sonde ist `current_d`: die Frage nach dem aktuellen Stand mit heutigem Datum im Systemprompt, also so,
     wie das Modell im Relay tatsächlich gefragt wird. Diese Sonde ist der „virtuelle Cutoff“.
   - Mit `cutoffmix simulate` kann man prüfen, ob der Schätzer bekannte Cutoffs wiederfindet.

## Mitgelieferter Index: Prior für Weltereignisse, gemessene Zelle `ai_ml`

Die Weltereignis-Themen stammen aus dem Prior (unten), die Zelle `ai_ml` aus der eigenen Messung (nächster Abschnitt).

### Prior-Teil (Knowledge-Cutoff-Bench, Daten bis Juni 2026)

Sonde `current`, also die direkte Frage des Knowledge-Cutoff-Bench. Datenbasis: 280 Ereignisse, 1677 bewertete
Antworten, Ereignisse von 2024-01 bis 2026-06. In jeder Zelle steht τ als `JJJJ-MM.Zehntel` und dahinter P(best);
`*` markiert das wahrscheinlich beste Modell.

| Thema | Fable 5.2 | GPT-6.2 Sol | Gemini 4 Neon |
|---|---|---|---|
| Regierung / Amtsinhaber | 2026-03.3 · 0 % | **2026-04.7 · 99 %*** | 2026-03.7 · 1 % |
| Wahlen | 2026-03.9 · 0 % | **2026-05.6 · 100 %*** | 2026-03.8 · 0 % |
| Wirtschaft | 2026-03.5 · 0 % | **2026-05.8 · 100 %*** | 2026-03.5 · 0 % |
| Sport | 2026-03.3 · 0 % | **2026-05.4 · 100 %*** | 2026-03.5 · 0 % |
| Kunst & Kultur | 2026-02.7 · 0 % | **2026-05.0 · 100 %*** | 2026-03.1 · 0 % |
| KI / Software / Wissenschaft / Raumfahrt | ~2026-03.2 · 0 % | **~2026-05.2 · ≈100 %*** | ~2026-03.5 · 0–2 % |

Die globalen τ (Fable 2026-03.2, Sol 2026-05.2, Neon 2026-03.5) liegen jeweils etwa 0,5 Monate über den
veröffentlichten Werten des Bench. Das entspricht unserer tagesgenauen Zeitachse (Monatsmitte = +0,5). In Themen
mit nur 0–4 Ereignissen steht einfach der Modellwert. `cutoffmix index show` zeigt die vollständige Tabelle.

## Gemessene Zelle `ai_ml`: AI-Paper-Probe-Set (25.11.2026)

Für Research-Fragen zählt die Zelle `ai_ml`, und die stand im Prior-Index auf einem einzigen Ereignis. Deshalb wurde
ein eigenes Probe-Set gemessen: **210 AI-Paper** (Januar 2024 bis November 2026, dicht ab Oktober 2025 mit 12 pro
Monat), ausgewählt nach Bekanntheit über die Hugging-Face-Daily-Papers-Liste, Metadaten exakt von arXiv (v1), dazu
**21 erfundene Titel** als Kontrollen. Sonde: „Wer ist Erstautor des arXiv-Papers mit dem Titel ‚…‘?“ (aus dem Titel
nicht zu raten, per Regel bewertbar) plus 4er-Multiple-Choice. 34 Paper gehören zur Interpretability-Teilmenge.
Aufbau: `cutoffmix bench build-papers`; Daten: `data/events/papers_events.jsonl`, Antworten:
`data/results/papers_probes_2026-11-25.jsonl`; Kosten des Laufs ≈ 5,9 $.

**Anteil richtiger Erstautoren je Monat** (offene Frage, Datum im Systemprompt; n = 12 pro Monat ab 2025-10):

| Monat | Fable 5.2 | Gemini 4 Neon | GPT-6.2 Sol |
|---|---|---|---|
| 2025-07 | 1,00 | 0,83 | 0,00 |
| 2025-10 | 1,00 | 0,83 | 0,08 |
| 2025-11 | 1,00 | 0,75 | 0,17 |
| 2025-12 | 1,00 | 0,58 | 0,00 |
| 2026-01 | 1,00 | 0,33 | 0,17 |
| 2026-02 | 0,75 | 0,00 | 0,00 |
| 2026-03 | 0,83 | 0,17 | 0,00 |
| 2026-04 | 0,67 | 0,08 | 0,17 |
| 2026-05 | 0,25 | 0,00 | 0,00 |
| 2026-06 | 0,08 | 0,00 | 0,00 |
| 2026-07 … 2026-11 | 0,00–0,17 | 0,00–0,08 | 0,00–0,08 |

- **Umkehrung gegenüber dem Prior:** Bei ML-Literatur liegt Fable klar vorn, Neon folgt, Sol kennt Erstautoren
  selbst von 2025er-Papern kaum (Plateau ≈ 0,5 schon 2024/25). Der Prior-Index hatte Sol in jedem Thema vorn.
- **Multiple Choice** (Wiedererkennen): Fable bis Juli 2026 ≥ 0,92, bis November 2026 noch über Zufall (0,25);
  Neon bis Juli 2026 ≥ 0,67; Sol meist 0,2–0,5. Fable hat also auch Titel aus Herbst 2026 gesehen, kann aber die
  Autoren nicht mehr abrufen.
- **Kontrollen (erfundene Titel):** Fable 0 von 21 konfabuliert, Neon 0 von 21, Sol 4 von 21. Bei echten Papern nennt
  Sol in 26 % der Fälle einen falschen Autor, Fable in 4 %, Neon in 1 %. Das geht als Halluzinationsterm H ins
  Routing ein.
- **Interpretability-Teilmenge:** Fable bis März 2026 alle richtig, Neon und Sol ab Ende 2025 fast nichts; für eine
  eigene Kurve ist die Teilmenge (0–3 Paper pro Monat) zu klein.

**Gefitteter Index** (hierarchisches 4PL, Modellniveau auf den Bench-Weltereignissen verankert, Zelle `ai_ml` aus
den Paper-Daten; `cutoffmix index show`):

| Sonde | Fable 5.2 | Gemini 4 Neon | GPT-6.2 Sol |
|---|---|---|---|
| `current` (offene Frage), Zelle `ai_ml` | **2026-04.6 · P(freshest) 100 %** | 2026-01.3 | 2025-09.9 |
| `mcq` (Wiedererkennen), Zelle `ai_ml` | **2026-09.8 · 100 %** | 2026-04.5 | 2025-10.0 |
| `current`, Weltereignis-Themen (Prior, unverändert) | ≈ 2026-03.2 | ≈ 2026-03.5 | **≈ 2026-05.2** |

Dieser Index ist jetzt der **Standard** (`cutoffmix/data/index_prior.json`). Für KI-Fragen schreibt damit Fable,
für Nachrichten-Themen weiterhin Sol; der Themenklassifikator entscheidet pro Abschnitt. Sols `ai_ml`-Wert hängt
davon ab, wie die flache Kurve seines Paper-Wissens gelesen wird (Halbwert 2025-04 oder Nullpunkt 2025-10); an
der Reihenfolge ändert das nichts. Reproduktion:

```bash
cutoffmix bench import-kc --out runs/kc
cutoffmix bench fit --events runs/kc/kc_events.jsonl data/events/papers_events.jsonl \
    --results runs/kc/kc_results.jsonl data/results/papers_probes_2026-11-25.jsonl \
    --probe-map current_d=current --global-origins kc --probes current mcq \
    --priors claude-fable-5.2=25.8 gpt-6.2-sol=27.75 gemini-4-neon=26.0 --prior-sd 3 --out runs/index.json
```

Einschränkungen: Die Sonde misst Erstautor-Wissen, nicht Verständnis des Inhalts; Paper, die nach v1 umbenannt
wurden, fehlen; die Monate Juli bis November 2026 liegen für alle drei Modelle hinter dem Cutoff, und 12 Paper pro
Monat geben pro Zelle eine Unsicherheit von etwa einem Monat.

## Installation

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[server]"   # Laufzeit: numpy, httpx, pyyaml; UI/API: fastapi, uvicorn
pip install -e ".[dev]"      # zusätzlich pytest
```

## Ausprobieren ohne API-Key

`--mock` simuliert die drei Modelle mit bekannten Themen-Cutoffs: Sol meist vorn, Fable bei KI/Software, Neon bei
Wissenschaft/Raumfahrt. So sieht man das Umschalten ohne Netz.

```bash
cutoffmix run --mock "Wer ist aktuell Präsident von Estland und was weiß man über neue KI-Modelle?"
cutoffmix run --mock "…" --strategy speculative --policy thompson
cutoffmix serve --mock          # Web-UI auf http://127.0.0.1:8000/, OpenAI-API unter /v1
cutoffmix simulate              # Schätzer an simulierten Antworten prüfen
```

## Mit den echten Modellen

```bash
export OPENROUTER_API_KEY=…     # nie in Dateien oder ins Repo schreiben
cutoffmix run "Wer ist aktuell Bundeskanzler von Deutschland?"
cutoffmix serve
```

`cutoffmix serve` bietet `POST /v1/chat/completions` mit diesen Modell-IDs:

- `cutoffmix`: das Relay;
- `cutoffmix-speculative`: das Relay mit parallelen Entwürfen;
- `cutoffmix-fixed-<key>`: nur ein Modell, als Vergleich;
- `cutoffmix-explore`: das Relay mit dem Explore-Preset (siehe unten);
- `cutoffmix-rally`: das Relay mit dem Rally-Preset (siehe unten).

Die Notizen kommen als `delta.reasoning` (dort, wo OpenAI-kompatible Clients Thinking anzeigen), die Antwort als
`delta.content`. Pro Anfrage lassen sich `policy`, `strategy`, `segment_tokens`, `preset` und `ring` (nur Rally) im
Request-Body setzen.

Konfiguration: `--config datei.yaml` wird über `cutoffmix/data/default.yaml` gemischt. Beispiel für einen neu
gemessenen Index:

```yaml
router:
  index_path: runs/index.json
  policy: softmax      # oder thompson, argmax, fixed:gpt-6.2-sol …
relay:
  segment_tokens: 128
```

## Explore-Preset: Querverbindungen und ausführlichere Notizen

```bash
cutoffmix run --preset explore "Warum verlieren Lithium-Ionen-Akkus bei Kälte an Kapazität?"
cutoffmix run --preset explore --policy round_robin "…"      # alle drei Modelle sichtbar, nicht nur das freshste
cutoffmix serve --preset explore                              # UI-Auswahl „Preset“; API-Modell-ID cutoffmix-explore
```

Hintergrund: Der Wunsch war, die Modelle mit hoher Temperatur laufen zu lassen, damit die Kette über andere
Fachgebiete wandert, und sie mehr Zwischenschritte ausschreiben zu lassen. Drei Literaturrecherchen mit über 140
direkt geprüften Quellen (`docs/research/explore-*.md`, Synthese in `docs/DESIGN_explore.md`) haben ergeben:

- **Temperatur ist bei keinem der drei Modelle wirksam.** Fable 5.2 und GPT-6.2 Sol haben den Parameter nicht.
  Bei Gemini 4 Neon nimmt OpenRouter ihn an, aber laut Googles Doku ignoriert der Server Werte im erlaubten Bereich
  und lehnt andere mit 400 ab. Das Preset schickt deshalb keine Temperatur. Die Studien sprechen auch inhaltlich
  dagegen: Höhere Temperatur bringt wenig Neuheit und messbar mehr Inkohärenz, lange Ketten brechen ab T ≈ 1,5
  zusammen, und selbst bei T = 2 bleiben über 60 % der Antworten fast gleich.
- **„Mehr schreiben“ verbessert bei Frontier-Modellen die Antwort nicht,** eher im Gegenteil (umgekehrte U-Kurve
  über die Länge; erzwungenes Weiterdenken erzeugt Varianz). Es verbessert aber Nachvollziehbarkeit und Breite.
- **Was nachweislich hilft, ist „anderes schreiben“:** explizit verworfene Alternativen, Prinzipien aus anderen
  Feldern, Zwischenstände, Impulse von außen.

Das Preset setzt deshalb auf Protokoll statt Sampling:

| Mechanismus | Was passiert |
|---|---|
| **Fachfremder Blickwinkel** (`lens`) | Vor einzelnen mittleren Abschnitten zieht das Relay per Seed ein Fachgebiet aus 23 Feldern (Thermodynamik, Immunologie, Spieltheorie, Musiktheorie, …), meist mittelweit oder weit vom Thema entfernt. Das schreibende Modell muss zwei bis drei funktionale Entsprechungen nennen, eine prüfbare Folgerung ableiten und zur Frage zurückkehren. |
| **Prüfabschnitt** (`check`) | Direkt danach prüft ein **anderes** Modell: Welche Entsprechung hält, wo bricht die Analogie, `Lens kept:` oder `Lens dropped:`. Blickwinkel-Abschnitte zählen nicht zum Routing-Fenster, damit der Router nicht dem Analogiefeld hinterherwechselt. |
| **Strukturierte Notizen** | Jeder Abschnitt besteht aus 2–4 beschrifteten Punkten (`Claim:`, `Evidence:` mit Quelle, `Inference:`, `Principle:`, `Alternative set aside:`, `Open question:`, `Next:`) und endet mit `Where we stand:` (beste Antwort, Konfidenz, schwächste Annahme). Abschnitte sind 192 statt 128 Token lang. |
| **Abschluss-Sperre** | `FINAL ANSWER` wird erst akzeptiert, wenn mindestens 6 Hauptabschnitte vorliegen und das Protokoll `Evidence:`, `Alternative set aside:` und `Open question:` enthält. |
| **Fortsetzung statt „Wait“** (`continue`) | Schließt ein Modell zu früh ab, bleibt sein Notizteil erhalten, die Antwort wird verworfen, und ein anderes Modell setzt mit einem inhaltlichen Einstieg fort („Alternatively, …“, „The strongest objection to the current conclusion is …“). Höchstens dreimal; bei Wiederholung wird die Antwort akzeptiert. |
| **Sol** | `verbosity: medium` für Notizen, `low` für die Antwort, fester `seed` pro Abschnitt. |
| **Fable: abgeschwächte Vorgabe** (`light_schema_models`) | Fable bekommt den Standard-Systemprompt und einen Schlussblock ohne die Liste beschrifteter Punkte; Nummer der Notiz, Blickwinkel-/Prüf-/Fortsetzungsanweisung und die Freigabe zum Abschluss bleiben. Beides ist live nachgewiesen durchgelassen (erster Abschnitt und Blickwinkel-Abschnitt mit Notizen). Fables Notizen tragen deshalb meist keine Labels; die Checkliste wird von Sol und Neon erfüllt. |
| **Optional** `vs_menu: stagnation` | Bei festgefahrener Kette fragt ein Nebenaufruf an Sol fünf Richtungen mit Wahrscheinlichkeiten ab (Verbalized Sampling) und der Client wählt eine unwahrscheinliche. Standardmäßig aus. |

**Messwerte** stehen in `done.explore` bzw. im Trace: besuchte Themen und ihre Entropie, Blickwinkel
(angeboten/behalten/verworfen, Felder, Quote), Label-Abdeckung, erzwungene Fortsetzungen, Antwortwechsel in den
`Where we stand:`-Zeilen, Überlappung der letzten Abschnitte. CLI-Zusammenfassung und UI zeigen sie an.

**Grenzen:** Mit dem Prior-Index schreibt auch hier meist Sol, Querverbindungen wechseln also das Modell nicht
(`--policy round_robin` zeigt alle drei). Das Preset verbessert Antworten nicht, es macht sie nachvollziehbarer und
breiter. Kosten: etwa das 1,5- bis 2,5-Fache eines Standardlaufs, ca. 100–140 s mit `round_robin`.

## Rally-Preset: Schaukel zwischen den Modellen

```bash
cutoffmix run --preset rally "Was ist der aktuelle Stand bei KI-Modellen?"
cutoffmix run --ring claude-fable-5.2,gemini-4-neon,gpt-6.2-sol "…"   # drei Modelle im Ring (--ring schaltet das Preset ein)
cutoffmix serve --preset rally          # UI-Auswahl „Preset“ = rally; API-Modell-ID cutoffmix-rally
curl … -d '{"question": "…", "ring": ["claude-fable-5.2", "gemini-4-neon"]}' http://127.0.0.1:8000/api/relay
```

**Idee.** Das normale Relay lässt den Router pro Abschnitt *ein* Modell wählen, meist das frischeste. Die Rally macht
das Gegenteil mit Absicht: Zwei oder drei Modelle mit **verschiedenen Trainingsdaten** reichen die Notizen im festen
Ring weiter (Standard: Fable, Neon, Fable, Neon, …).

1. Modell A schreibt, was es weiß, jeden Punkt mit Datum (`Known:`), und den neuesten Monat, den es wirklich kennt
   (`Newest I know:`).
2. Modell B bekommt As Notiz und die Anweisung: bestätigen (`Confirm:`), Überholtes oder Falsches korrigieren
   (`Update:`, mit Datum), Neueres nach As letztem Datum nennen (`Newer:`), Zweifel markieren (`Unverified:`) oder
   ehrlich sagen, dass nichts Neueres bekannt ist (`Nothing newer:`). Erfinden ist ausdrücklich verboten.
3. Modell A liest Bs Notiz. Bs Neueres ist für A Stichwort und Prüfstein: A erinnert sich vielleicht an Zusammenhängendes,
   das es von allein nicht geschrieben hätte. Und so weiter.
4. Ende: wenn ein voller Umlauf nichts Neueres bringt (`min_turns` = 2 × Ringgröße, danach `quiet_turns` = Ringgröße
   ruhige Notizen in Folge) oder bei `max_turns` (Standard 8). Dann schreibt ein Aufruf die Antwort aus den Notizen:
   was ein zweites Modell bestätigt hat, steht ohne Einschränkung, was nur eines gesagt hat, trägt `(single source)`.

Das Preset ändert nur den Ablauf. Die Reihenfolge bestimmt der Ring, nicht der Router. Der Router wählt weiterhin das
Antwortmodell unter den Ring-Modellen und meldet Thema und Scores wie immer. Mit `rally.answer_from_ring: false`
darf auch ein Modell außerhalb des Rings antworten. Nicht kombinierbar mit `--preset explore`.

**Kennzahlen** stehen in `done.rally` (CLI-Zusammenfassung und UI-Streifen „Rally“): Notizen und Stopp-Grund
(`converged`, `max_turns`, `no_partner`, `max_segments`, `empty`, `error`), pro Notiz die Label-Zahlen und der
angegebene neueste Monat, die **Frontier-Kurve** (laufendes Maximum dieser Monate) mit `gain_months` (wie weit sie über
den besten ersten Eintrag hinausgeht) und pro Modell, wie viel es korrigiert (`update`), Neues gebracht (`newer`)
und bestätigt hat.

**Grenzen, ehrlich:**

- **Die Monatsangaben sind Selbstauskünfte der Modelle.** Das Projekt zählt Selbstauskünfte sonst nicht. Die Kurve
  zeigt, was der Austausch getan hat, nicht, dass er hilft.
- **Unbewiesen ist, dass die Rally den effektiven Cutoff über das beste Einzelmodell hinaushebt.** Das ist eine
  Hypothese. Gegenrisiko: Modelle können Fehler voneinander übernehmen oder etwas „Neueres“ erfinden, um zu gefallen.
  Die Prompts verbieten das, die Antwort markiert Einzelquellen, mehr ist offline nicht prüfbar.
- **Gemessen wird das später mit dem Benchmark:** dieselben Aktuell-Fragen nach den Cutoffs der Modelle, drei Arme
  (jedes Modell allein, Router-Relay, Rally), Trefferquote nach Ereignismonat (`docs/DESIGN_rally.md`, Abschnitt 9).
  Das ist noch nicht gebaut und kostet echtes Geld, also nur mit Freigabe.
- **Mit echten Modellen nicht gelaufen** (siehe „Ehrlicher Stand“). Die simulierten Modelle konvergieren, weil sie so
  gebaut sind. Das sagt nichts darüber, wie die echten schreiben.

## Den Index neu messen (themenstratifiziert)

```bash
cutoffmix bench import-kc --out runs/kc                                   # öffentliche Bench-Daten
cutoffmix bench build-wikidata --out runs/wikidata_events.jsonl           # optional; 953 Ereignisse liegen bereits in data/events/
cutoffmix bench run --events data/events/wikidata_events.jsonl --out runs/results.jsonl \
    --probes current_d mcq --dry-run                                      # zuerst nur Kosten schätzen
cutoffmix bench run --events data/events/wikidata_events.jsonl --out runs/results.jsonl --probes current_d mcq
cutoffmix bench fit --events runs/kc/kc_events.jsonl data/events/wikidata_events.jsonl \
    --results runs/kc/kc_results.jsonl runs/results.jsonl --out runs/index.json
```

Der Lauf kann unterbrochen und fortgesetzt werden. Danach `router.index_path` auf `runs/index.json` setzen.

**Kosten und Abdeckung:**

- Für 953 Ereignisse × 3 Modelle × 2 Sonden schätzt `--dry-run` **≈ 51 $**. Die Obergrenze liegt bei ≈ 330 $, wenn
  jeder Aufruf `max_tokens` ausschöpft; Judge-Aufrufe sind nicht eingerechnet.
- Erst einen Pilotlauf mit `--limit 20` machen und die echten Token-Zahlen ansehen.
- Die Wikidata-Ereignisse decken nur 7 der 15 Themen ab, mit Schwerpunkt Sport (606), Regierung (169) und Wahlen (96).
- Für KI, Software, Raumfahrt, Gesundheit, Konflikte und Unglücke braucht es zusätzlich das Wikipedia-Current-events-Set
  (`bench build-wikipedia --probe-model …`). Dessen Fragen erzeugt ein Modell, was ebenfalls Geld kostet.

## Optionales Training (≤ 1 h)

Ohne Training geht alles. Der Themenklassifikator arbeitet mit Schlüsselwörtern, und der Index ist reine Statistik.
Optional:

- `cutoffmix router train`: ein linearer Themenklassifikator und ein linearer Gate, auf der CPU in Minuten.
- `torchrun --nproc_per_node=N -m cutoffmix.router.train_gpu --events … --results …`: ein Encoder (mmBERT bzw.
  ModernBERT) mit Köpfen für Thema, Zeitsensitivität und „Aktualität pro Modell“. Das Zeitbudget ist hart auf
  55 Minuten begrenzt und für eine B200/B300 ausgelegt; mehrere GPUs per torchrun.

  Sinnvoll ist dieses Training erst, wenn ein neuer Benchmark-Lauf mit Themen-Umkehrungen vorliegt.

## Daten und Lizenzen

- Daten des Knowledge-Cutoff-Bench (apoorvumang/knowledge-cutoff): **CC BY 4.0**. Sie fließen in den Prior-Index ein
  (nur aggregierte Statistiken) und werden von `bench import-kc` verwendet. Der Code des Bench steht unter keiner
  Lizenz und wurde deshalb nicht übernommen, sondern neu implementiert.
- Wikidata: **CC0 1.0** (`data/events/wikidata_events.jsonl`, siehe `data/events/README.md`).
- Wikipedia: **CC BY-SA 4.0**, für `bench build-wikipedia`.
- Code: MIT.

## Tests

```bash
python -m pytest -q    # offline, kein Test greift aufs Netz zu
```
