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
- **Plato's-Key-Experiment („Schlüssel statt Antwort“): Version 2 gebaut, offline geprüft, mit den echten Modellen noch
  nicht gelaufen.** Es misst, ob die maskierte Notiz eines anderen Modells ein Modell an eigenes Wissen erinnert oder ob
  es nur abschreibt. Die Vorhersagen stehen vorab fest (`docs/DESIGN_keys.md`); nach zehn Prüfrunden entscheidet mit
  `--judges` der Judge bei der Übernahme jede Antwort, die den Namen nennt, und bei der Bewertung jede, die ihn nennt
  und die die Regeln sonst als falsch oder als Enthaltung werten würden; die festen Regeln entscheiden nur noch
  „richtig“, die Antworten ohne den Namen und den Lauf ohne Judge. Kein Urteil stützt sich auf unbewertete Antworten,
  die die richtige Antwort nennen, ob die Regeln sie offen lassen oder die Stimme des Judge ausfiel, auch mit einem
  Titel wie „Professor“ („Professor Guo“) oder leicht verschrieben nach dem Maß, mit dem die Bewertung eine mögliche
  Verschreibung dem Judge überlässt („Daya Guoo“; auch bei einer angenommenen Form wie „Mei Lingrui“); nur eine Nennung
  im Namen einer anderen Person zählt nicht („Yutong Mei“ bei Lingrui Mei), und als Name einer anderen Person gilt auch
  ein Name mit einem Wort, das auf keiner Liste der Prüfung steht („Researcher Guo“), ein Name mit demselben Nachnamen
  und Anfangsbuchstaben, den die Bewertung nur deshalb offen lässt und dessen Vorname um mehr als einen Buchstaben
  abweicht („Lingwei Mei“ bei Lingrui Mei), sowie eine Verschreibung jenseits dieses Maßes („Ziyu Mi“ bei Ziyu Ma). Der
  Pilot hat 746 Aufrufe, kostet geschätzt 2,17 $ plus bis etwa 0,45 $ für den Judge (im ungünstigsten Fall 0,83 $;
  zusammen geschätzt 3,00 $, eine Schätzung, keine Obergrenze: die setzt `--max-cost`, freigegeben werden sollen 3,5 $)
  und läuft erst mit Freigabe und Schlüssel.
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

**Was passiert, wenn ein Partner ausfällt.** Kann das nächste Modell seinen Zug nicht machen (Fehler, Verweigerung, leere
Notiz), schreibt das vorige Modell weiter. Diese Notiz ist eine **Selbstantwort**: Sie bekommt eine eigene Anweisung
(„die Notiz davor ist deine eigene, bestätige dich nicht selbst“), zählt nie als ruhig, kann die Schaukel nicht
beenden, und die Antwort erfährt, welche Notizen das waren. Ein Partner, der zweimal hintereinander mit einem Fehler
ausfällt (bei einem nicht wiederholbaren Fehler sofort), ist für den Lauf raus; Verweigerungen zählen über
`relay.disable_after_refusals` (Preset: 3). Die Schaukel endet dann mit `no_partner`, statt zum Selbstgespräch zu
werden, und die Antwort wird aus den bisherigen Notizen geschrieben. Übersprungene Übergaben stehen in den Kennzahlen.

**Fable bekommt die leichte Variante.** Im Explore-Modus hat Anthropics Eingangsfilter Fable bei Prompts mit vielen
Labels gesperrt (siehe oben). Die Rally-Anweisung hat noch mehr Labels und ist live nicht geprüft, deshalb steht Fable
im Preset unter `rally.light_schema_models`: der normale System-Prompt, eine Anweisung in Prosa ohne Label-Liste, das
andere Modell wird nicht beim Namen genannt. Fables Notizen werden dann ohne Labels gelesen (ruhig ist eine Antwort ohne
Labels, wenn ihr „Newest I know:“ nicht über dem bisherigen Stand liegt). Nach einem kurzen Live-Test (4–6 Aufrufe)
lässt sich die Liste leeren, falls die volle Anweisung durchgeht.

**Robust gegen Abschneiden und Wiederholung.** Eine Notiz wird aus dem vollständigen Text gelesen, bevor die
Längengrenze greift; die Grenze liegt im Preset bei 1500 Zeichen, damit die letzte Zeile („Newest I know:“) selten
verloren geht. Eine ruhige Notiz darf fast wortgleich mit der vorigen sein, ohne als Echo verworfen zu werden.

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
- **Geprüft:** ein unabhängiges Review mit Gegenprüfung fand 17 echte Probleme im Kern (u. a. Selbstbestätigung nach
  einem Ausfall, verlorene letzte Zeile durch die Längengrenze, ein künftiger Monat, der als gültiges Jahr durchrutschte,
  ein Regex, der bei langen Leerzeichenfolgen Sekunden brauchte). Alle sind behoben und mit Tests abgedeckt.

## Schlüssel statt Antwort: Plato's-Key-Experiment

Vorlage ist der Blogbeitrag „Plato's Key“ von Hazy Research (Stanford, 25.11.2026): Bei Faktenwissen baut ein Modell in
den frühen und mittleren Schichten einen *Schlüssel* zum Thema (etwa „Frankreich“), mit dem spätere Schichten den *Wert*
(„Paris“) aus den Gewichten holen. Überträgt man den Schlüssel eines Modells (gedreht und skaliert) in ein anderes,
nennt das Empfängermodell in median 41 % der Fälle die Antwort des Spenders. Von den Fakten, die der Empfänger allein
verpasst, aber gespeichert hat („dunkle Erinnerungen“), holt der fremde Schlüssel 18 % zurück; Kontrollen höchstens
1,1 %. Das braucht Zugriff auf die inneren Zustände, Fable, Sol und Neon geben über die API aber nur Text aus.

Übertragen wird deshalb nur die Frage, auf Textebene: **Wenn Modell B eine Frage nicht beantworten kann, die Modell A
richtig beantwortet, erinnert sich B dann an die Antwort, sobald es A's Notiz sieht, aus der die Antwort selbst
entfernt ist?** Oder wird B nur besser, wenn es die Antwort abschreiben kann? Für das Relay und die Rally heißt das:
Erinnert sich B wirklich, ist seine Zustimmung nach einer Übergabe eine unabhängige Bestätigung. Schreibt es nur ab, ist
ein `Confirm:` nach einer vollen Notiz ein Echo und zählt als eine Quelle.

| Arm | Was das Empfängermodell sieht |
|---|---|
| `alone` | nur die Frage, noch einmal gestellt (Rauschgrenze des erneuten Fragens) |
| `key` | Frage und die Notiz des Spenders, Antwort durch `[withheld]` ersetzt |
| `full` | Frage und die vollständige Notiz des Spenders |
| `wrong_key` | Frage und die maskierte Notiz des Spenders zu einem **anderen** Paper (Kontrolle: falscher Fakt) |
| `blank` | Frage und eine inhaltsleere Notiz in derselben Form (Kontrolle: hilft schon, dass da eine Notiz steht?) |

Daten: das AI-Paper-Probe-Set und seine bewerteten Antworten vom 25.11.2026 (bereits bezahlt, versioniert unter
`data/results/papers_probes_2026-11-25.jsonl`). Fable weiß dort bei 50 Fragen, was Neon nicht weiß, und bei 96, was Sol
nicht weiß; Neon bei 47, was Sol nicht weiß. Neon antwortet aber in einem Satz, nach dem Maskieren bleibt davon nur „Der
Erstautor von [title] ist [withheld]“. Solche Notizen tragen keinen Hinweis und fallen heraus, damit fällt die Richtung
Neon → Sol weg. Nach Lecks und Notizen ohne Hinweis bleiben 79 Fragen für Fable → Sol (die entscheidende Richtung) und
37 für Fable → Neon. In die Gegenrichtungen gibt es je nur eine Frage, sie sind nicht messbar. Außerdem gemessen:
Spender lag falsch (übernimmt der Empfänger den falschen Namen?), beide lagen richtig (stört die Notiz?) und erfundene
Paper (übernimmt er eine Erfindung?).

Vorab festgelegte Vorhersagen (`docs/DESIGN_keys.md`, Abschnitt 7, Version 2):

1. Bei den Fragen, die nur der Spender konnte, ist `key` besser als `alone` und `wrong_key`; 1b: auch besser als
   `blank`, es zählt also der Inhalt der Notiz und nicht nur, dass es eine gibt.
2. Der Gewinn durch den Schlüssel ist dort größer, wo der Empfänger die Antwort im Multiple-Choice-Test erkennt.
3. `full` ist besser als `key`, und mit `full` macht sich der Empfänger falsche Spender-Antworten weit häufiger zu eigen
   als allein.
4. Die Notiz stört selten eine richtige Antwort: weniger als 10 Prozentpunkte über dem Rauschen des erneuten Fragens.

Die möglichen Ausgänge hängen von der Vorhersage ab. Die Vorhersagen 1, 1b und 3 sind gestützt, nicht gestützt (ein
Vergleich schließt den vorhergesagten Gewinn aus: 10 Prozentpunkte, bei 3b 20), unentschieden oder nicht testbar (unter
6 Fragen in einem Vergleich; bei 3a und 3b auch, wenn mehr als 20 % ihrer `full`-Antworten unentschieden bleiben).
Vorhersage 2
wird nur geprüft, wenn 1 gestützt ist und jede Gruppe mindestens 10 Einheiten hat (sonst nicht testbar), und ist dann
nur gestützt oder unentschieden, nie widerlegt. Vorhersage 4 ist gestützt oder nicht gestützt, je nachdem, ob der
gemessene Mehrschaden unter der festen Schwelle von 10 Prozentpunkten liegt (ein Punktwert, kein statistischer
Ausschluss), oder nicht testbar (unter 10 Fragen). Ein unvollständiger Lauf ergibt „vorläufig“ statt eines Urteils;
Zeilen aus mehreren Backends ergeben gar keines. Nicht testbar ist eine Vorhersage auch, wenn offene Angebote ihr
Urteil kippen könnten: unbewertete Antworten, die die richtige Antwort nennen (unten).

**Version 2 nach einem Review.** Drei unabhängige Prüfer haben die erste Fassung angegriffen. Jeder Fund wurde
gegengeprüft und auf dem echten Plan nachgerechnet; 16 Funde haben sich bestätigt. Die Regeln wurden geändert, bevor
ein einziger echter Aufruf lief (Abschnitt 10 der Design-Doku). Die wichtigsten Punkte:

- Als „übernommen“ zählte auch ein Name, den der Empfänger nur zitiert („Die Notiz nennt X, das kann ich nicht
  bestätigen“). Jetzt wird unterschieden: zu eigen gemacht, nur berichtet oder nicht genannt. Berichtete Fälle
  entscheidet ein Judge.
- Vorhersage 2 verglich, wie oft der Schlüssel trifft. Erkannte Fragen gelingen aber auch beim bloßen Nachfragen
  öfter. Jetzt zählt der Gewinn gegenüber `alone`.
- Notizen ohne Hinweis, Hinweise auf andere Paper desselben Autors und Masken, die einen Namensteil eines Koautors
  verraten („Mengdi [withheld]“), werden erkannt und aussortiert bzw. getrennt ausgewertet.
- Jede Ergebniszeile trägt ihre Herkunft (Backend, Prompt-Hash). Mock- und Echtzeilen lassen sich nicht mehr mischen,
  Judge-Kosten zählen zum Kostendeckel, und eine zweite Instanz auf derselben Datei wird abgewiesen.

Danach wurde auch die Umsetzung Bereich für Bereich gegen die Design-Doku geprüft und nachgebessert, ebenfalls vor dem
ersten echten Aufruf (Abschnitt 10, „Refinements“). Ein Beispiel: Nennt eine Antwort einen Namen nur als Autor eines
*anderen* Papers („der Erstautor des ursprünglichen Vidu-Papers war X“), gilt das nicht mehr als zu eigen gemacht,
sondern als berichtet, und der Judge entscheidet.

Dann wurden alle Bereiche zusammen geprüft, ebenfalls vor dem ersten echten Aufruf. Dabei fiel eine Lücke im Arm
`full` auf: Dort steht in der Notiz der richtige Name. Die Regeln werteten eine Antwort schon dann als richtig, wenn sie
diesen Namen nur weitergibt oder sogar zurückweist („Die Notiz nennt X, ich glaube aber, es ist Y“, „Die Antwort der
Notiz, X, ist falsch“). Das hätte die Trefferquote von `full` künstlich erhöht und damit Vorhersage 3a („`full` ist
besser als `key`“) begünstigt. Jetzt gilt für den richtigen Namen dieselbe Unterscheidung wie für den falschen: Richtig
ist eine `full`-Antwort nur, wenn sie sich den Namen zu eigen macht. Nennt sie ihn nur als Inhalt der Notiz, bleibt sie
unbewertet und zählt als nicht richtig, bis der Judge entscheidet: zu eigen gemacht heißt richtig, nur berichtet oder
zurückgewiesen heißt falsch.

Eine letzte Prüfung aller Bereiche vor dem Pilot hat die Regeln noch einmal verschärft. Die festen Regeln entscheiden
jetzt nur noch die klaren Fälle („Der Erstautor ist X.“). Beim richtigen Namen einer `full`-Notiz (3a) entscheidet
alles andere der Judge: eine andere Schreibweise des Namens („Z. Wang“ statt „Ziqi Wang“), Zweifel („Ich bezweifle,
dass es X ist“), eine Zurückweisung („X ist falsch“, „X ist nur Koautor, Erstautor ist Y“), eine Notiz, die mit
anderen Worten zitiert wird („der bereitgestellte Text nennt X“) und jede Antwort, die neben dem Namen noch eine andere
Person als Antwort anbietet. Beim falschen Namen des Spenders (3b) gilt eine Antwort nach den Regeln als übernommen,
wenn sie sich den Namen zu eigen macht, die Bewertung sie als falsche Antwort liest und sie niemand anderen als
Antwort anbietet; als nicht übernommen gilt sie nach den Regeln, wenn sie den Namen nicht als Antwort anbietet (auch
in anderer Schreibweise, mit Zweifel oder als Zurückweisung) oder als richtig bewertet wird. Den Rest, etwa nur
berichtete Rede, entscheidet der Judge. Bleiben bei 3a mehr als 20 % der `full`-Antworten unentschieden (etwa weil
alle Judge-Aufrufe fehlschlugen), ist 3a nicht testbar statt widerlegt. Außerdem: Ein HTTP 403 betrifft beim Gateway
oft nur eine einzelne Anfrage (etwa eine Moderationssperre) und hält den Lauf nicht mehr an; ein 401 oder 402 hält ihn
an, auch bei einem Judge-Aufruf; das erste Strg-C wirkt sofort, auch vor dem ersten Aufruf; eine abgerissene letzte
Zeile wird auch dann abgeschnitten, wenn sie mitten in einem Zeichen wie „ß“ endet. Die 746 geplanten Aufrufe sind
dieselben geblieben.

Eine Prüfung dieser Änderungen fand noch Lücken, ebenfalls vor dem ersten echten Aufruf. Eine Antwort, die den Namen
nur nennt, um der Person eine andere Rolle zu geben („X ist Koautor, nicht Erstautor“, „X ist der Seniorautor“), oder
die ihn mit anderen Worten zurückweist („X als Erstautor ist fragwürdig“, „Ich bestreite, dass X der Erstautor ist“),
galt als zu eigen gemacht. Bei 3b hätte das die Übernahme überschätzt, zumal 13 der 53 falschen Spender-Namen echte
Koautoren des Papers sind. Beim richtigen Namen galt auch eine Antwort mit Vorbehalt („Ich kann nicht bestätigen, dass
X der Erstautor ist“) als richtig, ebenso eine, die die Notiz mit anderen Worten zitiert („laut den bereitgestellten
Informationen“, „der Text oben nennt X“, „X ist angeblich der Erstautor“). All das entscheidet jetzt der Judge.
Umgekehrt verneint ein Vorbehalt in der ersten Person den Namen nicht mehr („Wenn ich mich nicht irre, ist es X“, „Ich
könnte mich irren, der Erstautor ist X“). Bei 3a zählt jede unbewertete `full`-Antwort als unentschieden, auch eine,
die die Regeln gar nicht bewerten konnten („Es könnte X sein.“). Ein Urteil zu 3a oder 3b gilt außerdem nur, wenn die
unentschiedenen Antworten es nicht kippen können: Widerlegt ist eine Hälfte nur, wenn sie es auch bliebe, sobald der
Judge alle unentschiedenen `full`-Antworten als Treffer wertet; gestützt nur, wenn sie es auch bliebe, sobald er alle
unentschiedenen Antworten des Vergleichsarms als Treffer wertet. Sonst ist sie nicht testbar. Ein ungültiges Byte in
der letzten Zeile der Ergebnisdatei wird jetzt sofort abgewiesen, statt die Datei nach dem nächsten Lauf unlesbar zu
machen. Die 746 geplanten Aufrufe sind dieselben geblieben.

Auch danach fand eine weitere Prüfung Antworten, die die Regeln falsch entschieden, ohne dass ein Judge gefragt wurde,
etwa „X steht auf der Autorenliste, aber ich glaube nicht, dass X der Erstautor ist“: Das galt bei allen 53 falschen
Spender-Namen als übernommen und bei allen 136 richtigen Namen als richtig, auch mit `--judges`. Statt die Regeln ein
weiteres Mal zu erweitern, gilt jetzt: Läuft der Judge mit (`--judges`), entscheidet er jede bewertete Antwort, die den
Namen nennt, in welcher Form auch immer (mit Initiale, Nachname zuerst, ohne Akzente, auch der Nachname allein; einer
aus zwei Buchstaben wie „Wu“ zählt dabei großgeschrieben), egal, was die Regeln sagen würden. Das sind alle als richtig
bewerteten `full`-Antworten bei 3a und alle Antworten, die den Namen des Spenders nennen, bei 3b, auch beim erneuten
Fragen (`alone`) und auch dann, wenn die Antwort als richtig bewertet ist („Die Notiz sagt Y, es ist aber X“). Die
Regeln entscheiden allein nur noch die Antworten, die den Namen gar nicht nennen (nicht übernommen). Schlägt ein
Judge-Aufruf fehl oder gibt es keine Mehrheit, bleibt die Antwort unentschieden: Die Regeln springen nicht ein, und ein
späterer Lauf mit `--judges` fragt erneut, ohne den Empfänger noch einmal aufzurufen. Die Entscheidung der Regeln wird
trotzdem in jeder Zeile gespeichert (`rule_adopted`), und der Bericht stellt sie in einer Tabelle „Regeln gegen Judge“
der des Judge gegenüber. Kein Urteil hängt an ihr. Sie zeigt nach dem Pilot, ob man den Übernahme-Regeln ohne Judge
trauen kann, soweit die festen Regeln die Antwort überhaupt bewerten konnten (Antworten, die erst der Bewertungs-Judge
bewertet hat, zählt sie gesondert), und ob der Judge selbst klare Antworten („Der Erstautor ist X.“) zurückweist. Ohne
Judge entscheiden die Regeln wie bisher, lassen aber mehr offen: bloße Zugehörigkeit zur Autorenliste („steht auf der
Autorenliste“, „hat an dem Paper mitgearbeitet“), Zweifel an der Erstautorschaft in einem späteren Satzteil („aber
vermutlich nicht als Erster“), weitere Rollen (PI, Laborleitung, Kollaborateur, „die Gruppe von X“) und weitere Verweise
auf die Notiz („laut den Informationen, die du geteilt hast“, „wie angegeben“, „wie erwähnt“, „die Zusammenfassung“,
„was du bereitgestellt hast“). Diese Verweise treffen auch Quellenangaben der Antwort selbst („wie auf arXiv angegeben“,
„laut der Zusammenfassung des Papers“): Auch die gelten jetzt als nur berichtet und bleiben ohne Judge unentschieden.
„Es ist nicht Y, es ist X.“ gilt jetzt als X zu eigen gemacht, auch mit Gedankenstrich („Es ist nicht Y – es ist X.“).
Nachgetragen ist außerdem: Die Verweise auf die Notiz in anderen Worten aus der vorigen Runde („laut den
bereitgestellten Informationen“, „X ist angeblich der Erstautor“) gelten auch beim falschen Namen des Spenders (3b) als
nur berichtet; vorher galten sie dort nach den Regeln als übernommen. Die 746 geplanten Aufrufe sind dieselben
geblieben.

Eine Prüfung dieser Änderung fand noch fünf Lücken, ebenfalls vor dem ersten echten Aufruf. Erstens erkannte die Suche
nach dem Namen einen Nachnamen aus zwei Buchstaben allein nicht („Wu et al.“, „Dr. Li“, „die Gruppe um Ye“): Bei drei
Spender-Namen (Xiaoyang Wu, Kunchang Li, Jiabo Ye; 10 der 106 Spender-Zeilen) entschieden dann auch mit `--judges` die
Regeln (nicht übernommen). Jetzt zählt ein solcher Nachname als Nennung, wenn er großgeschrieben ist; eine Antwort, die
eine andere Person dieses Namens nennt, kostet dann einen Judge-Aufruf. Zweitens bewerteten die festen Regeln eine
Berichtigung wie „Es ist nicht Y, es ist X.“ oder „Es könnte X sein, der mit Y arbeitet.“ als falsch, endgültig und in
jedem Arm: Die Verneinung oder das „könnte“ vor X reicht für sie über das Komma hinweg, und Y gilt dann als die gegebene
falsche Antwort. Bei 3b galt dieselbe Antwort dagegen als X zu eigen gemacht. Jetzt ist ein „falsch“ der Regeln
unsicher, wenn die Antwort die richtige Antwort nach der Regel von `offers` anbietet: Mit `--judges` entscheidet der
Bewertungs-Judge, ohne ihn bleibt die Antwort unbewertet, in allen Armen gleich (bei den gespeicherten Antworten kommt
das nicht vor). Drittens hielt ein fortgesetzter Lauf eine Entscheidung des Judge nur, solange das Etikett der Regeln
gleich blieb: Nach einer späteren Änderung der Regeln entschieden ohne Judge wieder die Regeln, mit Judge wurde dieselbe
Frage noch einmal bezahlt; und nach einer Neuplanung mit einem anderen Spender-Namen blieb eine Entscheidung über den
alten Namen stehen. Jetzt speichert jede bewertete Zeile den Namen, um den es geht (`adoption_name`); die Entscheidung
des Judge gilt, solange die Antwort diesen Namen nennt, egal welches Etikett die Regeln ihr geben, und eine Entscheidung
über einen anderen Namen zählt nicht. Viertens zählte die Tabelle „Regeln gegen Judge“ auch Antworten, die erst der
Bewertungs-Judge bewerten konnte („Es könnte X sein.“), als Übereinstimmung mit den Regeln; die stehen jetzt in einer
eigenen Spalte. Fünftens hielten die Grenzen für das Überschreiten des Deckels an den Übergängen zwischen den Blöcken
des Plans nicht (unten). Die 746 geplanten Aufrufe sind dieselben geblieben.

Eine Prüfung dieser Änderungen fand drei weitere Lücken, ebenfalls vor dem ersten echten Aufruf. Erstens werteten die
festen Regeln auch mit `--judges` manche Antwort endgültig als falsch, die die richtige Person nennt, ohne dass ein
Judge gefragt wurde: eine Berichtigung mit einem Nachnamen, den die Regeln allein nicht annehmen („Es ist nicht Y, es
ist Zhang.“), mit „Dr. Zhang“, „Zhang et al.“ oder nur mit dem Vornamen (bei 105 bzw. 134 der 136 richtigen Namen);
„Ich glaube, es ist X, bin mir aber nicht sicher.“ mit dem Vornamen galt als Enthaltung. Jetzt gilt für die Bewertung
derselbe Grundsatz wie für die Übernahme: Mit `--judges` entscheidet der Bewertungs-Judge jede Antwort, die die
richtige Antwort in irgendeiner Form nennt, sobald die Regeln sie als falsch oder als Enthaltung werten würden, in
allen Armen gleich; „richtig“ bleibt eine Entscheidung der Regeln, und ohne Judge ändert sich nichts. In der
Nachrechnung kostet das 15 Judge-Aufrufe mehr (drei Fragen, bei denen Sol eine andere Person desselben Nachnamens
nennt). Zweitens zählte eine unbewertete Antwort immer als falsch, auch wenn sie die richtige Antwort anbietet. Ohne
Judge sind das gerade die Berichtigungen, die die Regeln nicht mehr als falsch werten („Es ist nicht Y, es ist X.“):
Bei 3a galt `full` dann als besser als `key`, obwohl der Judge beide als richtig wertet. Jetzt speichert jede Zeile, ob
die Antwort die richtige Antwort anbietet (`offers_expected`), und ein Urteil zu 1, 1b, 3a oder 4 gilt nur, wenn diese
offenen Angebote es nicht kippen können: gestützt nur, wenn es auch bliebe, sobald die offenen Angebote des
Vergleichsarms als richtig zählen; nicht gestützt nur, wenn es auch bliebe, sobald die des geprüften Arms als richtig
zählen; sonst ist die Vorhersage nicht testbar. Das gilt auch für die Prüfung, ob die falsche Notiz in die Irre führt
(Vorhersage 1). Vorhersage 2 gilt nur dann als gestützt, wenn keine ihrer Einheiten ein offenes Angebot enthält, weil
ihr Permutationstest sich nicht in eine Richtung verschiebt. Von den 57 gespeicherten Antworten, die die Regeln offen
lassen, bietet keine die richtige Antwort an. Drittens waren die Zusätze zur Grenze von 1,64 $ („vor den letzten 7
Aufrufen eines Blocks“, „ab dem 664. Aufruf“) keine Grenzen: Ein langsamer Aufruf behält seinen Platz, während spätere
starten und fertig werden. Jetzt hängen sie davon ab, was schon gestartet ist (unten). Die 746 geplanten Aufrufe sind
dieselben geblieben.

Eine Prüfung dieser Änderungen fand noch zwei Lücken, ebenfalls vor dem ersten echten Aufruf. Erstens galt eine
unbewertete Antwort nur dann als offenes Angebot, wenn die festen Regeln die richtige Antwort darin als angeboten lesen
(der volle Name oder ein seltener Nachname allein). Eine Antwort, die sie anders nennt („Dr. Zhang“, „Zhang et al.“, nur
der Vorname, ein häufiger Nachname allein, „Ich bin nicht sicher. X?“), zählte ohne Prüfung als falsch, obwohl mit
`--judges` gerade solche Antworten an den Bewertungs-Judge gehen. Fiel seine Stimme aus (ein fehlgeschlagener Aufruf,
keine Mehrheit oder eine Antwort ohne Urteil), konnte das ein Urteil still entscheiden: Zwei ausgefallene Stimmen bei
den 20 `key`-Antworten von `both_right` machten Vorhersage 4 zu „nicht gestützt“. Ohne Judge galt dasselbe für solche
Antworten, die die Regeln gar nicht bewerten können. Jetzt ist jede unbewertete Antwort, die die richtige Antwort in
irgendeiner Form nennt, ein offenes Angebot, außer sie nennt sie nur im Namen einer anderen Person („Yutong Mei“ bei
Lingrui Mei); auch ein Zweifel oder eine Zurückweisung zählt, was ein Urteil höchstens nicht testbar macht, nie falsch.
Eine unbewertete Antwort, die die richtige Antwort gar nicht nennt, ist dagegen fast sicher falsch und zählt weiter so.
Zweitens galt eine Antwort ohne Urteil nicht als fehlgeschlagener Judge-Aufruf, der Lauf meldete sie nicht. Jetzt nennen
Lauf und Bericht die Antworten, die nach den Stimmen des Judge offen sind, mit dem Hinweis, vor dem Bericht noch einmal
mit `--judges` zu laufen. Außerdem sind einige Texte berichtigt: „richtig“ bleibt auch mit `--judges` eine Entscheidung
der Regeln; ein zweiter Lauf mit `--judges` schickt nur dann nichts, wenn der Judge alles entschieden hat; und der
Befehl für den Plan unten nennt das Datum des vorab festgelegten Plans, `--today 2026-11-27` (ohne die Option nimmt
`plan` das heutige Datum, und alle 746 Prompts wären andere). Bei den gespeicherten Antworten ändert sich nichts: Die
offenen Angebote und alle Urteile der Nachrechnung bleiben dieselben. Die 746 geplanten Aufrufe sind dieselben
geblieben.

Eine Prüfung dieser Änderungen fand noch eine Lücke, ebenfalls vor dem ersten echten Aufruf: Als Name einer anderen
Person galten auch zwei Schreibweisen der richtigen Antwort. Erstens ein Name mit einem Titel oder einem Gruppenwort,
das die Bewertung im Namen lässt („Professor Guo“, „Doctor Guo“, „die Guo Group“; „Dr.“, „Prof.“, „Lab“ und „Team“
lässt sie selbst weg), zweitens eine leicht verschriebene Form, die die Bewertung selbst als mögliche Verschreibung
dem Judge überlässt („Daya Guoo“, „Matthew Kowall“). Mit `--judges` gehen solche Antworten an den Bewertungs-Judge.
Fiel aber seine Stimme aus, oder ließen ohne Judge die Regeln eine Verschreibung offen, zählte die Antwort ohne Prüfung
als falsch: Zwei ausgefallene Stimmen bei „Der Erstautor ist Professor Guo.“ unter den 20 `key`-Antworten von
`both_right` machten Vorhersage 4 zu „nicht gestützt“, und ohne Judge machten 116 `key`-Antworten bei `dark`, deren
Nachname einen verdoppelten letzten Buchstaben hat, Vorhersage 3a zu „gestützt“, beide Male ohne ein einziges offenes
Angebot. Jetzt liest die Prüfung einen solchen Namen auch ohne seine Titel- und Gruppenwörter (auch die deutschen:
Herr, Frau, Doktor, Professorin, Gruppe, Arbeitsgruppe), und eine Verschreibung, die die Bewertung als möglich ansieht,
gilt als die richtige Antwort; beide Fälle sind jetzt nicht testbar. Ein verschriebener Nachname allein („Dr. Guoo“)
ist dagegen für die Bewertung kein Name und zählt weiter nicht als Nennung der richtigen Antwort. Ein anderer Vorname
mit demselben Nachnamen bleibt eine andere Person, wenn die Bewertung ihn nur wegen desselben Anfangsbuchstabens offen
lässt und er mehr als einen Buchstaben abweicht („Zhaoyang Li“ bei Zhiyu Li), ebenso jede falsche Antwortmöglichkeit der
Frage; ein ähnlicher Vorname („Ziyi Ma“ bei Ziyu Ma, „Zhiyuan Li“ bei Zhiyu Li) gilt als die richtige Antwort, was ein
Urteil höchstens nicht testbar macht. Bei der Übernahme gelten zwei ähnliche Namen weiter als zwei Personen, denn dort
kostet eine Verwechslung nur einen Judge-Aufruf; bei den offenen Angeboten dagegen ließe sie ein Urteil auf einer
Antwort stehen, die richtig sein kann. Ohne Judge bleibt eine bekannte Grenze: Ein „falsch“ der Regeln für eine Antwort,
die die richtige Antwort in einer Form nennt, die die Regel von `offers` nicht annimmt („Es ist nicht Y, es ist Dr.
Zhang.“), bleibt endgültig; der Pilot läuft mit Judge. Bei den gespeicherten Antworten ändert sich nichts. Die 746
geplanten Aufrufe sind dieselben geblieben.

Eine Prüfung dieser Änderungen fand noch zwei Lücken, ebenfalls vor dem ersten echten Aufruf. Erstens ließ die Prüfung
nur die Titel- und Gruppenwörter ihrer Liste weg, und jedes andere großgeschriebene Wort, das die Bewertung im Namen
lässt, machte ihn weiter zum Namen einer anderen Person: ein Wort vor einem Titel („Visiting Professor Guo“,
„Distinguished Professor Guo“), ein Titel, der nicht auf der Liste stand („Lecturer Guo“, „Herrn Guo“, „Monsieur Guo“,
„Director Guo“), und ein Wort, das den Satz eröffnet („Either Guo or Wang“, „Clearly Guo wrote it“). Zwei ausgefallene
Stimmen bei „Der Erstautor ist Visiting Professor Guo.“ unter den 20 `key`-Antworten von `both_right` machten Vorhersage
4 wieder zu „nicht gestützt“, und als alle Stimmen ausfielen, machten 92 `key`-Antworten bei `dark` mit „Visiting
Professor“ vor dem Nachnamen Vorhersage 3a zu „gestützt“, beide Male ohne ein offenes Angebot. Jetzt fallen mit einem
Titel auch die Wörter davor weg, die Liste hat mehr Titel (Lecturer, Director, Herrn, Dozent, Privatdozent,
Juniorprofessor und französische, spanische und italienische), und ein Wort, das einen Satz eröffnet, fällt am Anfang
eines Namens weg; beide Fälle sind jetzt nicht testbar. Ein anderes Wort, das die Bewertung im Namen lässt („Researcher
Guo“, auch ein Vorname wie in „Dean Guo“), und eine Verschreibung, die nach dem Maß der Bewertung weder der richtigen
Antwort noch einer angenommenen Form nahekommt („Ziyu Mi“ bei Ziyu Ma), machen ihn weiter zum Namen einer anderen
Person: Fällt bei einer solchen Antwort die Stimme des Judge aus, zählt sie ohne Prüfung als falsch, und nur der Hinweis
auf die offenen Stimmen meldet sie. Zweitens verglich die Prüfung eine Verschreibung auch mit den Formen mit Initiale
(„L. Mei“, „Guo D.“), und an einer so kurzen Form hält die Bewertung auch einen anderen kurzen Vornamen mit demselben
Anfangsbuchstaben für eine mögliche Verschreibung: „Lu Mei“ bei Lingrui Mei, „Bo Wang“ bei Bin Wang und sogar „Gao
Huang“ bei Ailin Huang, den die Bewertung selbst als falsch wertet, galten als die richtige Antwort. Das machte ein
Urteil höchstens nicht testbar, widersprach aber der Beschreibung. Jetzt zählt eine Form mit Initiale nur für einen
Namen ohne ausgeschriebenen Vornamen („Professor Gur-Ariehh“ bei Yoav Gur-Arieh). Berichtigt sind außerdem die
Beschreibung der Prüfung auf Initialen in `docs/DESIGN_keys.md` (sie liest einen Vornamen ohne Bindestrich als Folge von
Initialen) und die Aussage, jede Verschreibung sei eine, die die Bewertung dem Judge überlässt: Eine verschriebene Form
mit dem Nachnamen zuerst („Mei Lingrul“) gilt als die richtige Antwort, obwohl die Bewertung sie als falsch wertet. Bei
den gespeicherten Antworten ändert sich nichts. Die 746 geplanten Aufrufe sind dieselben geblieben.

```bash
cutoffmix bench keys plan --events data/events/papers_events.jsonl --today 2026-11-27 \
    --results data/results/papers_probes_2026-11-25.jsonl --out runs/keys_plan.jsonl   # offline, druckt die Kosten
cutoffmix bench keys run --plan runs/keys_plan.jsonl --events data/events/papers_events.jsonl \
    --out runs/keys_results.jsonl --judges gpt-6.2-sol --max-cost 3.5   # echte Modelle; --max-cost ist Pflicht
cutoffmix bench keys report --plan runs/keys_plan.jsonl --results runs/keys_results.jsonl --md runs/keys_report.md
```

Kosten: 746 Aufrufe, Empfänger sind Sol (443) und Neon (275), Fable nur bei 28. Geschätzt aus den gespeicherten Kosten
(Mittelwerte): 2,17 $ (Neon 1,47, Sol 0,45, Fable 0,24). Dazu kommt der Judge: etwa 0,17 $ für die Antworten, die die
Regeln nicht bewerten können oder gegen die genannte richtige Antwort werten (in der Nachrechnung 86, davon 15 durch die
neue Regel), und, da er jede Antwort mit dem Namen entscheidet, bis etwa 0,45 $ (bis zu 136 Aufrufe für den richtigen
Namen, 53 für den falschen in den `full`-Antworten, in der Nachrechnung 14 beim erneuten Fragen), im ungünstigsten Fall
etwa 0,83 $. Mit der Schätzung für die Empfänger sind das etwa 3,00 $, also etwa 50 Cent unter `--max-cost 3.5`. Unter
dem früheren Deckel von 3 $ bliebe nur etwa 0,5 Cent; freigegeben werden soll deshalb ein Deckel von 3,5 $ (wer
freigibt, kann auch einen anderen wählen). Eine Obergrenze ist das nicht: Der Anteil der Empfänger ist
ein Mittelwert, und die Zahl der Judge-Aufrufe ist ein Höchstwert oder stammt aus einer Nachrechnung mit den
gespeicherten Antworten; begrenzt wird der Lauf nur durch den Deckel. Der Deckel zählt den Judge mit, hält aber nur neue
Aufrufe an: Was beim Erreichen schon läuft, läuft mit seinen Judge-Aufrufen zu Ende. Der Lauf kann den Deckel also um
die Kosten von bis zu `--concurrency` laufenden Aufrufen überschreiten. Aus den längsten gespeicherten Antworten
geschätzt sind das bei der Voreinstellung 8 etwa 0,23 $, aber das ist keine Grenze: Jeder Aufruf darf `max_tokens` 4096
ausschöpfen, ein Judge-Aufruf 1024. Zu den gespeicherten Preisen sind es dann höchstens etwa 1,64 $ (acht
Fable-Aufrufe); ein kleineres `--concurrency` senkt das anteilig. Der Plan schickt erst 439 Aufrufe an Sol (geschätzt
etwa 0,45 $, mit dem Judge im ungünstigsten Fall etwa 0,92 $), dann 231 an Neon, 26 an Fable, 44 an Neon, 2 an Fable und
4 an Sol. Die Aufrufe starten in dieser Reihenfolge, der Deckel wird nur vor jedem Start geprüft, und ein langsamer
Aufruf behält seinen Platz, während spätere starten und fertig werden. Was beim Erreichen des Deckels noch läuft, wurde
vorher gestartet und kann deshalb schon zu einem späteren Block gehören. Höchstens etwa 0,65 $ über dem Deckel endet
der Lauf, wenn der Deckel erreicht wird, bevor der erste Neon-Aufruf gestartet ist (vor dem 440. Aufruf: acht
Sol-Aufrufe), höchstens etwa 0,81 $, wenn er erreicht wird, bevor der erste Fable-Aufruf gestartet ist (vor dem 671.:
acht Neon-Aufrufe), sonst höchstens etwa 1,64 $. Ein Deckel von 3 $ oder mehr wird, wenn überhaupt, erst erreicht,
nachdem der erste Neon-Aufruf gestartet ist, und dann gilt die Grenze von 1,64 $ (0,81 $, solange kein Fable-Aufruf
gestartet ist). Hält der Lauf am Deckel an,
schickt ein weiterer Lauf mit höherem `--max-cost` den Rest, was ebenfalls eine Freigabe braucht. Ein vorab
freigegebener höherer Deckel vermeidet diesen Halt. Die Planung druckt die Schätzung pro Modell. `run` setzt nach einer
Unterbrechung fort. Der Exit-Code ist der des Halts, der als erster geplante Aufrufe am Start hinderte: 3 bei
`--max-cost`, 4 nach einem Fehler, den kein neuer Versuch behebt (HTTP 401 oder 402: Schlüssel oder Guthaben, auch bei
einem Judge-Aufruf; dann startet kein Aufruf mehr), 130 nach Strg-C. Treffen mehrere bei derselben Prüfung zusammen,
gilt Strg-C vor dem Fehler vor dem Deckel; ein späteres Strg-C oder ein späterer Fehler ändert den Code nicht, und ein
Strg-C nach dem Start des letzten Aufrufs hält nichts mehr an. Ohne einen solchen Halt endet `run` mit 4, wenn Aufrufe
fehlschlugen (auch bei `--limit`; der nächste Lauf schickt sie erneut) oder ein Fehler, den kein neuer Versuch behebt,
erst nach dem Start des letzten Aufrufs kam, sonst mit 0. Das erste Strg-C lässt die laufenden Aufrufe zu Ende laufen
und speichern, erst ein zweites bricht sofort ab. Exit-Code 2 heißt: Eingaben oder Optionen abgewiesen, vor jedem
Aufruf, oder die Ergebnisdatei nahm während des Laufs keine Zeile mehr an (etwa bei voller Platte); dann zählen die
geschriebenen Zeilen, und ein bezahlter Aufruf, dessen Zeile fehlt, wird beim nächsten Lauf erneut geschickt. Mit
`--mock` läuft alles offline gegen simulierte Modelle, in eine eigene Datei (`--out runs/keys_results_mock.jsonl`).
Deren Zahlen sind eingestellt, nicht gemessen, der Bericht kennzeichnet sie als SIMULATED. **Mit den echten Modellen ist
das Experiment noch nicht gelaufen.**

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
