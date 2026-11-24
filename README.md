# cutoffmix

One chain of working notes, written segment by segment by several models. Each segment goes to the model whose
**topic-specific effective knowledge cutoff** is the most recent for what the notes are currently about. The models
are `anthropic/claude-fable-5.2`, `openai/gpt-6.2-sol` and `google/gemini-4-neon` on OpenRouter. The cutoffs come
from external evaluation, never from the models' self-report. `docs/DESIGN.md` is the full design.

## Install

```bash
pip install -e ".[server]"        # runtime: numpy, httpx, pyyaml; server: fastapi, uvicorn
pip install -e ".[dev]"           # + pytest, pytest-asyncio
```

## Try it without an API key

`--mock` uses simulated models with known per-topic cutoffs (Sol freshest on most topics, Fable on AI and
software, Neon on science and space), so routing is visible without network access.

```bash
cutoffmix run --mock "Wer ist aktuell Präsident von Estland und was weiß man über neue KI-Modelle?"
cutoffmix run --mock "..." --strategy speculative --policy thompson
cutoffmix serve --mock            # web UI on http://127.0.0.1:8000/ and an OpenAI-compatible API under /v1
cutoffmix simulate --out runs/sim # fit the estimator on simulated answers and report how well it recovers the truth
```

## Real models

Set `OPENROUTER_API_KEY`, then use the same commands without `--mock`. `cutoffmix serve` exposes
`POST /v1/chat/completions` with the model ids `cutoffmix`, `cutoffmix-speculative` and `cutoffmix-fixed-<key>`.
The notes stream as `delta.reasoning`, the answer as `delta.content`.

## The cutoff index

`cutoffmix/data/index_prior.json` is the packaged default. It is a **prior**, fitted from the
apoorvumang/knowledge-cutoff benchmark, whose events end in 2026-06. On its routing probe (`current`, the
benchmark's direct question) it ranks GPT-6.2 Sol freshest on every topic, so always using Sol is the baseline that
topic routing has to beat. Most topic cells hold only a few events, and its
`meta` records the sources, the data range and these limits. `cutoffmix index show` prints it.

To measure topic reversals, build a topic-stratified probe set and fit a new index:

```bash
cutoffmix bench import-kc --out runs/kc
cutoffmix bench build-wikidata --out runs/wikidata_events.jsonl
cutoffmix bench run --events runs/wikidata_events.jsonl --out runs/results.jsonl --probes current_d mcq --dry-run
cutoffmix bench fit --events runs/kc/kc_events.jsonl runs/wikidata_events.jsonl \
    --results runs/kc/kc_results.jsonl runs/results.jsonl --out runs/index.json
```

Then point `router.index_path` in a config file (`--config FILE`) at the new index.

## Data attribution

- apoorvumang/knowledge-cutoff benchmark data: CC BY 4.0. It is used for the packaged prior index (aggregate
  statistics only) and by `bench import-kc`.
- Wikidata (CC0 1.0) and Wikipedia (CC BY-SA 4.0) are sources for `bench build-wikidata` and `bench build-wikipedia`.

## Tests

```bash
python -m pytest -q               # offline: no test touches the network
```
