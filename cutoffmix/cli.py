"""The ``cutoffmix`` command line (DESIGN.md §6).

::

    cutoffmix run "question" [--mock] [--policy P] [--strategy S] [--segments N] [--json OUT] [--config FILE]
    cutoffmix serve [--host H] [--port P] [--mock]
    cutoffmix index show [--path FILE] [--probe P]
    cutoffmix bench import-kc --out DIR
    cutoffmix bench build-wikidata --out FILE
    cutoffmix bench build-wikipedia --out FILE --start DATE --end DATE [--probe-model KEY]
    cutoffmix bench run --events FILE --out FILE --models KEY... --probes P... [--dry-run] [--batch] [--limit N]
    cutoffmix bench grade --events FILE --results FILE
    cutoffmix bench fit --events FILE... --results FILE... --out FILE [--models KEY...] [--probes P...]
                        [--priors KEY=MONTH...] [--note TEXT]
    cutoffmix router train [options of cutoffmix.router.train]
    cutoffmix simulate [--out DIR]

``--mock`` uses the simulated models of :mod:`cutoffmix.backends.mock` everywhere. Commands that call the real
models stop with a clear message when the API key variable (``OPENROUTER_API_KEY``) is not set. Benchmark modules
are imported only by the subcommands that need them.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from cutoffmix import __version__
from cutoffmix.relay import RelayRuntime, to_jsonable
from cutoffmix.timeutil import month_label
from cutoffmix.types import ModelSpec, RelayEvent

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig
    from cutoffmix.router.index import CutoffIndex

__all__ = ["CLIError", "TerminalPrinter", "build_parser", "format_tau", "main", "use_color"]

PROBE_CHOICES = ("current", "current_d", "recall", "mcq")


class CLIError(Exception):
    """A user-facing error: printed without a traceback, exit status ``code``."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------------------
# Terminal output
# --------------------------------------------------------------------------------------


def use_color(stream: TextIO, env: Mapping[str, str] | None = None) -> bool:
    """ANSI colours only on a terminal and only when ``NO_COLOR`` is unset or empty (https://no-color.org)."""
    env = os.environ if env is None else env
    if env.get("NO_COLOR"):
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def format_tau(x: float | None) -> str:
    """A month-axis value as ``YYYY-MM.f`` (tenths of the month), e.g. ``27.75 -> '2026-04.8'``."""
    if x is None or not math.isfinite(x):
        return "n/a"
    base, tenth = divmod(round(x * 10), 10)
    return f"{month_label(base)}.{tenth}"


def _rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return 136, 136, 136


class TerminalPrinter:
    """Prints relay events: one coloured block per segment with a model chip, then the answer and run stats.

    ``live=True`` (a terminal) streams deltas as they arrive and, if the committed text differs from what was
    streamed, redraws the block with the authoritative text. ``live=False`` (a pipe or file) prints every block
    once, when it is committed. ``color`` switches ANSI colours on (truecolor, one colour per model).
    """

    def __init__(self, models: Sequence[ModelSpec], *, out: TextIO, err: TextIO | None = None, color: bool = False,
                 live: bool = False, width: int | None = None) -> None:
        self.models = {m.key: m for m in models}
        self.out = out
        self.err = err if err is not None else out
        self.color = color
        self.live = live
        self.width = width
        self._start: RelayEvent | None = None
        self._shown = ""  # visible characters of the open block (header + streamed text) in live mode
        self._streamed = ""

    # ---- styling ----

    def _sgr(self, text: str, *codes: str) -> str:
        return f"\033[{';'.join(codes)}m{text}\033[0m" if self.color and codes else text

    def _fg(self, model: str | None) -> tuple[str, ...]:
        if model not in self.models:
            return ()
        r, g, b = _rgb(self.models[model].color)
        return (f"38;2;{r};{g};{b}",)

    def _chip(self, model: str | None) -> str:
        label = f" {self.models[model].label() if model in self.models else model} "
        if not self.color or model not in self.models:
            return f"[{label.strip()}]"
        r, g, b = _rgb(self.models[model].color)
        return self._sgr(label, f"48;2;{r};{g};{b}", "38;2;255;255;255", "1")

    def _header(self, ev: RelayEvent) -> tuple[str, str]:
        """``(styled header, visible header)`` for a segment or answer start."""
        data = ev.data or {}
        bits = [str(ev.decision or "")]
        if "r" in data:
            bits.append(f"r={data['r']:.2f}")
        if ev.topic_posterior:
            topic, p = max(ev.topic_posterior.items(), key=lambda kv: kv[1])
            bits.append(f"{topic} {p:.0%}")
        if ev.phase == "answer":
            bits.insert(0, "answer")
        meta = " · ".join(b for b in bits if b)
        chip = self._chip(ev.model)
        plain_chip = f" {self.models[ev.model].label()} " if self.color and ev.model in self.models else chip
        return f"{chip} {self._sgr(meta, '2')}", f"{plain_chip} {meta}"

    # ---- output ----

    def _write(self, text: str) -> None:
        self.out.write(text)
        self.out.flush()

    def _rows(self, visible: str) -> int:
        cols = self.width or shutil.get_terminal_size((100, 24)).columns
        return sum(max(1, math.ceil(len(line) / cols)) for line in visible.split("\n"))

    def _open_block(self, ev: RelayEvent) -> None:
        self._start = ev
        self._streamed = ""
        if self.live:
            styled, visible = self._header(ev)
            self._write(styled + "\n")
            self._shown = visible + "\n"

    def _close_block(self, ev: RelayEvent) -> None:
        """Commit (or drop) the open block with the authoritative text of ``ev``."""
        start = self._start or ev
        data = ev.data or {}
        dropped = bool(data.get("dropped"))
        if self.live:
            if dropped or ev.text != self._streamed:
                rows = self._rows(self._shown)
                height = shutil.get_terminal_size((100, 24)).lines
                if rows < height:  # erase the block; taller blocks cannot be reached any more
                    self._write("\r" + (f"\033[{rows - 1}A" if rows > 1 else "") + "\033[J")
                    if not dropped:
                        self._write(self._header(start)[0] + "\n" + self._sgr(ev.text, *self._fg(ev.model)))
                elif not dropped:
                    self._write("\n" + self._sgr("(committed) ", "2") + self._sgr(ev.text, *self._fg(ev.model)))
        elif not dropped:
            self._write(self._header(start)[0] + "\n" + self._sgr(ev.text, *self._fg(ev.model)))
        if dropped:
            reason = data.get("reason", "failed")
            detail = f": {data['error']}" if data.get("error") else ""
            label = self.models[ev.model].label() if ev.model in self.models else ev.model
            self._write(self._sgr(f"  ↳ {label} {reason}{detail} - re-routing", "2") + "\n")
        else:
            self._write("\n\n")
        self._start, self._shown, self._streamed = None, "", ""

    def handle(self, ev: RelayEvent) -> None:
        """Render one relay event."""
        if ev.type in ("segment_start", "phase"):
            self._open_block(ev)
        elif ev.type == "delta":
            self._streamed += ev.text
            if self.live:
                self._shown += ev.text
                self._write(self._sgr(ev.text, *self._fg(ev.model)))
        elif ev.type == "segment_end":
            self._close_block(ev)
        elif ev.type == "error":
            self.err.write(self._sgr(f"error: {ev.text}", "31", "1") + "\n")
            self.err.flush()
        elif ev.type == "done":
            self._write(self._sgr(self.summary(ev.data or {}), "2") + "\n")

    def summary(self, data: Mapping[str, Any]) -> str:
        """One-line run statistics from the ``done`` data."""
        segments = [s for s in data.get("segments", []) if s.get("phase") == "think"]
        latency = data.get("latency") or {}
        usage = data.get("usage_total") or {}
        parts = [f"segments {len(segments)}", f"switches {data.get('switches', 0)}"]
        share = data.get("share_by_model") or {}
        if share:
            parts.append("share " + ", ".join(
                f"{self.models[m].label() if m in self.models else m} {v:.0%}" for m, v in share.items()))
        if latency.get("first_delta_s") is not None:
            parts.append(f"first token {latency['first_delta_s']:.2f}s")
        if latency.get("total_s") is not None:
            parts.append(f"total {latency['total_s']:.2f}s")
        if usage.get("total_tokens"):
            parts.append(f"tokens {usage['total_tokens']}")
        if usage.get("cost"):
            parts.append(f"cost ${usage['cost']:.4f}")
        return " · ".join(parts)


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _load_cfg(args: argparse.Namespace, overrides: dict[str, Any] | None = None) -> AppConfig:
    from cutoffmix.config import load_config

    try:
        return load_config(getattr(args, "config", None), overrides or None)
    except (ValueError, OSError) as e:
        raise CLIError(f"invalid configuration: {e}") from e


def _require_key(cfg: AppConfig, what: str) -> None:
    if not os.environ.get(cfg.api_key_env):
        raise CLIError(f"{cfg.api_key_env} is not set, so {what} cannot call the models on OpenRouter.\n"
                       f"Set it (export {cfg.api_key_env}=...) or add --mock to use the simulated models.")


def _model_spec(cfg: AppConfig, key: str) -> ModelSpec:
    """A configured model, or an ad-hoc spec for any other OpenRouter id (judges, probe generators)."""
    if key in cfg.model_keys():
        return cfg.model(key)
    if "/" not in key:
        raise CLIError(f"unknown model {key!r}: use a configured key ({', '.join(cfg.model_keys())}) or an "
                       "OpenRouter id such as 'deepseek/deepseek-v4-pro'")
    return ModelSpec(key=key, provider_id=key, display=key)


def _write_json(path: str, data: Any) -> None:
    p = Path(path)
    if p.parent != Path():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _parse_priors(items: Sequence[str]) -> dict[str, float | str]:
    priors: dict[str, float | str] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise CLIError(f"--priors expects KEY=VALUE (a month index or YYYY-MM), got {item!r}")
        try:
            priors[key] = float(value)
        except ValueError:
            priors[key] = value
    return priors


def _parse_month(value: str) -> int:
    from cutoffmix.timeutil import parse_month

    try:
        return parse_month(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def _progress(stream: TextIO) -> Callable[[int, int, Any], None]:
    def report(done: int, total: int, _result: Any) -> None:
        stream.write(f"\r  {done}/{total}")
        if done == total:
            stream.write("\n")
        stream.flush()

    return report


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    relay_over: dict[str, Any] = {}
    if args.strategy:
        relay_over["strategy"] = args.strategy
    if args.segments is not None:
        relay_over["max_segments"] = args.segments
    if args.segment_tokens is not None:
        relay_over["segment_tokens"] = args.segment_tokens
    if args.answer_mode:
        relay_over["answer_mode"] = args.answer_mode
    overrides: dict[str, Any] = {}
    if relay_over:
        overrides["relay"] = relay_over
    router_over: dict[str, Any] = {}
    if args.policy:
        router_over["policy"] = args.policy
    if args.seed is not None:
        router_over["seed"] = args.seed
    if router_over:
        overrides["router"] = router_over
    if args.today:
        overrides["today"] = args.today
    cfg = _load_cfg(args, overrides)
    if not args.mock:
        _require_key(cfg, "'cutoffmix run'")
    printer = TerminalPrinter(cfg.models, out=sys.stdout, err=sys.stderr, color=use_color(sys.stdout),
                              live=sys.stdout.isatty())
    done = asyncio.run(_relay_once(cfg, args.question, mock=args.mock, seed=args.seed or 0, printer=printer))
    if args.json and done is not None:
        _write_json(args.json, done)
        print(f"trace written to {args.json}", file=sys.stderr)
    return 0 if done is not None and done.get("status") == "ok" else 1


async def _relay_once(cfg: AppConfig, question: str, *, mock: bool, seed: int,
                      printer: TerminalPrinter) -> dict[str, Any] | None:
    runtime = RelayRuntime(cfg, mock=mock, seed=seed)
    done: dict[str, Any] | None = None
    try:
        async with contextlib.aclosing(runtime.relay().run(question)) as events:
            async for event in events:
                printer.handle(event)
                if event.type == "done":
                    done = event.data
    finally:
        await runtime.aclose()
    return done


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as e:
        raise CLIError("serving needs uvicorn and FastAPI: pip install 'cutoffmix[server]'") from e
    from cutoffmix.server import create_app

    cfg = _load_cfg(args)
    if not args.mock:
        _require_key(cfg, "'cutoffmix serve'")
    app = create_app(cfg, mock=args.mock)
    print(f"cutoffmix {'(simulated models) ' if args.mock else ''}on http://{args.host}:{args.port}/", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def _index_provenance(meta: Mapping[str, Any]) -> list[str]:
    """Short provenance parts for the index header: its source(s) and the date range of the events behind it."""
    parts: list[str] = []
    sources = meta.get("source") or meta.get("sources")
    if isinstance(sources, str):
        parts.append(sources)
    elif sources:
        parts.append(", ".join(map(str, sources)))
    span = meta.get("data_range")
    if isinstance(span, Mapping) and span.get("first") and span.get("last"):
        parts.append(f"events {span['first']} .. {span['last']}")
    return parts


def render_index(index: CutoffIndex, probe: str, *, color: bool = False) -> str:
    """The index as a topic x model text table: ``tau`` as ``YYYY-MM.f`` and P_best, best model per topic marked."""
    rows = index.table(probe)
    cells = {(r["topic"], r["model"]): r for r in rows}
    topic_w = max(len("topic"), *(len(t) for t in index.topics))
    col_w = max(16, *(len(m) + 2 for m in index.models))
    lines = [" · ".join(["cutoff index", f"probe {probe}", *_index_provenance(index.meta)]),
             "topic".ljust(topic_w) + "".join(m.rjust(col_w) for m in index.models)]
    for topic in index.topics:
        row = [cells[(topic, m)] for m in index.models]
        best = max(range(len(row)), key=lambda i: row[i]["p_best"])
        line = topic.ljust(topic_w)
        for i, r in enumerate(row):
            text = f"{format_tau(r['tau'])} {r['p_best']:4.0%}{'*' if i == best else ' '}"
            padded = text.rjust(col_w)
            line += f"\033[1m{padded}\033[0m" if color and i == best else padded
        lines.append(line)
    lines.append("tau = estimated virtual cutoff (YYYY-MM.tenths); % = P(best on this topic); * = most likely best")
    return "\n".join(lines)


def _cmd_index_show(args: argparse.Namespace) -> int:
    from cutoffmix.router.index import load_index

    try:
        index = load_index(args.path)
    except (OSError, ValueError) as e:
        raise CLIError(f"cannot load the index: {e}") from e
    if index is None:
        raise CLIError("no cutoff index: pass --path FILE (written by 'cutoffmix bench fit'), or ship "
                       "cutoffmix/data/index_prior.json", code=1)
    probe = args.probe or index.primary().probe
    if probe not in index.probes:
        raise CLIError(f"probe {probe!r} not in the index; available: {', '.join(index.probes)}")
    print(render_index(index, probe, color=use_color(sys.stdout)))
    return 0


def _cmd_import_kc(args: argparse.Namespace) -> int:
    from cutoffmix.bench.kc_import import import_kc

    events_path, results_path = import_kc(args.out, models=args.models or None, force=args.force)
    print(f"events  -> {events_path}\nresults -> {results_path}")
    return 0


def _cmd_build_wikidata(args: argparse.Namespace) -> int:
    from cutoffmix.bench.io import save_events
    from cutoffmix.bench.sources.wikidata import BUILDERS, build_wikidata_events

    events = build_wikidata_events(args.start, args.end, kinds=args.kinds or BUILDERS)
    save_events(args.out, events)
    print(f"{len(events)} events -> {args.out}")
    return 0


def _cmd_build_wikipedia(args: argparse.Namespace) -> int:
    import datetime as dt

    from cutoffmix.bench.io import save_events, write_jsonl
    from cutoffmix.bench.sources.wikipedia import (
        ce_item_topic,
        fetch_ce_days,
        parse_ce_wikitext,
    )

    cfg = _load_cfg(args)
    if args.probe_model and not args.mock:
        _require_key(cfg, "probe generation")
    start, end = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    pages = fetch_ce_days(start, end)
    items = [item for day in sorted(pages) for item in parse_ce_wikitext(pages[day], day)]
    for item in items:
        item["topic"] = ce_item_topic(item)
    if not args.probe_model:
        write_jsonl(args.out, items)
        print(f"{len(items)} Current-events items (no probes; pass --probe-model to generate them) -> {args.out}")
        return 0
    from cutoffmix.backends import make_backend
    from cutoffmix.bench.probegen import generate_probes

    backend = make_backend(cfg, mock=args.mock)
    model = _model_spec(cfg, args.probe_model)

    async def go() -> list[Any]:
        try:
            return await generate_probes(backend, model, items, concurrency=args.concurrency)
        finally:
            await _close(backend)

    events = asyncio.run(go())
    save_events(args.out, events)
    print(f"{len(events)} events from {len(items)} items -> {args.out}")
    return 0


async def _close(backend: Any) -> None:
    close = getattr(backend, "aclose", None)
    if close is not None:
        await close()


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from cutoffmix.backends import make_backend
    from cutoffmix.bench.io import load_events
    from cutoffmix.bench.runner import estimate_cost, run_probes

    cfg = _load_cfg(args)
    events = load_events(args.events)
    if args.limit is not None:
        events = events[: args.limit]
    models = [_model_spec(cfg, k) for k in (args.models or cfg.model_keys())]
    judges = [_model_spec(cfg, k) for k in args.judges] or None
    if args.dry_run:
        print(json.dumps(estimate_cost(models, events, args.probes, batch=args.batch), indent=2))
        return 0
    if not args.mock:
        _require_key(cfg, "'cutoffmix bench run'")
    backend = make_backend(cfg, mock=args.mock, events=events)

    async def go() -> list[Any]:
        try:
            return await run_probes(backend, models, events, args.probes, args.out, today=cfg.today,
                                    concurrency=args.concurrency, use_batch_variant=args.batch, judges=judges,
                                    progress=_progress(sys.stderr))
        finally:
            await _close(backend)

    results = asyncio.run(go())
    _print_grades(results)
    print(f"{len(results)} results -> {args.out}")
    return 0


def _print_grades(results: Sequence[Any]) -> None:
    counts: dict[tuple[str, str], dict[str, int]] = {}
    for r in results:
        row = counts.setdefault((r.model, r.probe), {})
        row[r.grade] = row.get(r.grade, 0) + 1
    for (model, probe), grades in sorted(counts.items()):
        print(f"  {model:<24} {probe:<10} " + "  ".join(f"{g} {n}" for g, n in sorted(grades.items())))


def _cmd_bench_grade(args: argparse.Namespace) -> int:
    from cutoffmix.backends import make_backend
    from cutoffmix.bench.grading import grade_results
    from cutoffmix.bench.io import load_events, load_results, save_results

    cfg = _load_cfg(args)
    events = [e for path in args.events for e in load_events(path)]
    results = load_results(args.results)
    judges = [_model_spec(cfg, k) for k in args.judges]
    backend = None
    if judges:
        if not args.mock:
            _require_key(cfg, "judge grading")
        backend = make_backend(cfg, mock=args.mock, events=events)

    async def go() -> list[Any]:
        try:
            return await grade_results(results, events, backend=backend, judges=judges or None,
                                       concurrency=args.concurrency, regrade_all=args.regrade_all)
        finally:
            if backend is not None:
                await _close(backend)

    graded = asyncio.run(go())
    out = args.out or args.results
    save_results(out, graded)
    _print_grades(graded)
    print(f"{len(graded)} results -> {out}")
    return 0


def _cmd_bench_fit(args: argparse.Namespace) -> int:
    from cutoffmix.bench.estimate import fit_index
    from cutoffmix.bench.io import load_events, load_results

    events = [e for path in args.events for e in load_events(path)]
    results = [r for path in args.results for r in load_results(path)]
    index = fit_index(events, results, models=args.models or None, probes=args.probes or None,
                      priors=_parse_priors(args.priors) or None, prior_sd=args.prior_sd, bootstrap=args.bootstrap,
                      seed=args.seed, meta={"note": args.note} if args.note else None)
    index.save(args.out)
    print(render_index(index, index.primary().probe, color=use_color(sys.stdout)))
    print(f"index -> {args.out}")
    return 0


def _cmd_router_train(args: argparse.Namespace, extra: Sequence[str]) -> int:
    from cutoffmix.router.train import main as train_main

    return int(train_main(list(extra)) or 0)


def _cmd_simulate(args: argparse.Namespace) -> int:
    from cutoffmix.backends.mock import default_mock_truths
    from cutoffmix.bench.estimate import fit_index
    from cutoffmix.bench.io import save_events, save_results
    from cutoffmix.bench.simulate import (
        recovery_report,
        simulate_results,
        synthetic_events,
    )
    from cutoffmix.taxonomy import TOPIC_IDS

    if args.last < args.first:
        raise CLIError("--to must not be before --from")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    truths = default_mock_truths()
    events = synthetic_events(TOPIC_IDS, range(args.first, args.last + 1), args.per_cell, seed=args.seed)
    results = simulate_results(truths, events, args.probes, seed=args.seed)
    index = fit_index(events, results, bootstrap=args.bootstrap, seed=args.seed)
    reports = {probe: recovery_report(index, truths, probe) for probe in index.probes}
    save_events(out / "events.jsonl", events)
    save_results(out / "results.jsonl", results)
    index.save(out / "index.json")
    _write_json(str(out / "recovery.json"), reports)
    print(f"{len(events)} synthetic events, {len(results)} simulated results -> {out}")
    for probe, rep in reports.items():
        print(f"  {probe:<10} RMSE {rep['rmse']:.2f} months · argmax agreement {rep['argmax_agreement']:.0%} · "
              f"P_best Brier {rep['p_best_brier']:.3f} · 90% coverage {rep['coverage90']:.0%}")
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def _add_config(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", metavar="FILE", help="YAML config merged over the packaged defaults")


def build_parser() -> argparse.ArgumentParser:
    """The ``cutoffmix`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="cutoffmix",
        description="Multi-model chain-of-thought relay routed by topic-specific knowledge cutoffs.")
    parser.add_argument("--version", action="version", version=f"cutoffmix {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress (INFO) to stderr")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    run = sub.add_parser("run", help="answer a question with the relay, printing coloured segments")
    run.add_argument("question")
    run.add_argument("--mock", action="store_true", help="use the simulated models (no API key needed)")
    run.add_argument("--policy", help="softmax | thompson | argmax | round_robin | random | fixed:<model key>")
    run.add_argument("--strategy", choices=("sequential", "speculative"))
    run.add_argument("--segments", type=int, metavar="N", help="maximum number of note segments")
    run.add_argument("--segment-tokens", type=int, metavar="K", help="visible tokens asked per segment")
    run.add_argument("--answer-mode", choices=("routed", "inline"))
    run.add_argument("--seed", type=int, help="seed for routing (and the simulated models)")
    run.add_argument("--today", metavar="YYYY-MM-DD", help="date given to the models (default: today)")
    run.add_argument("--json", metavar="OUT", help="write the trace (with usage and latency) as JSON")
    _add_config(run)
    run.set_defaults(func=_cmd_run)

    serve = sub.add_parser("serve", help="serve the web UI and the OpenAI-compatible API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--mock", action="store_true", help="use the simulated models")
    _add_config(serve)
    serve.set_defaults(func=_cmd_serve)

    index = sub.add_parser("index", help="inspect the cutoff index")
    index_sub = index.add_subparsers(dest="index_command", required=True, metavar="ACTION")
    show = index_sub.add_parser("show", help="print the topic x model cutoff table")
    show.add_argument("--path", help="index JSON (default: the packaged prior)")
    show.add_argument("--probe", choices=PROBE_CHOICES)
    show.set_defaults(func=_cmd_index_show)

    bench = sub.add_parser("bench", help="build and run the topic-stratified cutoff benchmark")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True, metavar="ACTION")

    kc = bench_sub.add_parser("import-kc", help="import apoorvumang/knowledge-cutoff events and answers")
    kc.add_argument("--out", required=True, metavar="DIR")
    kc.add_argument("--models", nargs="*", help="only these models' answers")
    kc.add_argument("--force", action="store_true", help="download again")
    kc.set_defaults(func=_cmd_import_kc)

    wd = bench_sub.add_parser("build-wikidata", help="template events from Wikidata (office holders, elections, ...)")
    wd.add_argument("--out", required=True, metavar="FILE")
    wd.add_argument("--start", default="2024-01-01", metavar="YYYY-MM-DD")
    wd.add_argument("--end", metavar="YYYY-MM-DD", help="default: today")
    wd.add_argument("--kinds", nargs="*", help="heads elections awards sports (default: all)")
    wd.set_defaults(func=_cmd_build_wikidata)

    wp = bench_sub.add_parser("build-wikipedia", help="events from the Wikipedia Current-events portal")
    wp.add_argument("--out", required=True, metavar="FILE")
    wp.add_argument("--start", required=True, metavar="YYYY-MM-DD")
    wp.add_argument("--end", required=True, metavar="YYYY-MM-DD")
    wp.add_argument("--probe-model", metavar="KEY",
                    help="model that writes the probes (config key or OpenRouter id); without it the parsed items "
                         "are written instead of events")
    wp.add_argument("--concurrency", type=int, default=4)
    wp.add_argument("--mock", action="store_true", help="use the simulated models for probe generation")
    _add_config(wp)
    wp.set_defaults(func=_cmd_build_wikipedia)

    br = bench_sub.add_parser("run", help="probe the models on benchmark events")
    br.add_argument("--events", required=True, metavar="FILE")
    br.add_argument("--out", required=True, metavar="FILE", help="results JSONL (resumed if it exists)")
    br.add_argument("--models", nargs="*", metavar="KEY", help="default: every configured model")
    br.add_argument("--probes", nargs="+", choices=PROBE_CHOICES, default=list(PROBE_CHOICES))
    br.add_argument("--dry-run", action="store_true", help="only print the call and cost estimate")
    br.add_argument("--batch", action="store_true", help="use OpenRouter's discounted :batch variants")
    br.add_argument("--limit", type=int, metavar="N", help="only the first N events")
    br.add_argument("--judges", nargs="*", default=[], metavar="KEY", help="judge models for open answers")
    br.add_argument("--concurrency", type=int, default=8)
    br.add_argument("--mock", action="store_true", help="probe the simulated models")
    _add_config(br)
    br.set_defaults(func=_cmd_bench_run)

    bg = bench_sub.add_parser("grade", help="(re-)grade stored results, with judges for unsure open answers")
    bg.add_argument("--events", required=True, nargs="+", metavar="FILE")
    bg.add_argument("--results", required=True, metavar="FILE")
    bg.add_argument("--out", metavar="FILE", help="default: overwrite --results")
    bg.add_argument("--judges", nargs="*", default=[], metavar="KEY")
    bg.add_argument("--regrade-all", action="store_true")
    bg.add_argument("--concurrency", type=int, default=8)
    bg.add_argument("--mock", action="store_true")
    _add_config(bg)
    bg.set_defaults(func=_cmd_bench_grade)

    bf = bench_sub.add_parser("fit", help="fit the per-topic cutoff index")
    bf.add_argument("--events", required=True, nargs="+", metavar="FILE")
    bf.add_argument("--results", required=True, nargs="+", metavar="FILE")
    bf.add_argument("--out", required=True, metavar="FILE")
    bf.add_argument("--models", nargs="+", metavar="KEY", help="only these models (default: all in the results)")
    bf.add_argument("--probes", nargs="+", choices=PROBE_CHOICES, help="only these probes (default: all present)")
    bf.add_argument("--priors", nargs="*", default=[], metavar="KEY=MONTH",
                    help="prior centres per model: month index (27.75) or YYYY-MM")
    bf.add_argument("--prior-sd", type=float, default=3.0)
    bf.add_argument("--bootstrap", type=int, default=200)
    bf.add_argument("--seed", type=int, default=0)
    bf.add_argument("--note", help="free text stored as meta.note (what the index is, what it may be used for)")
    bf.set_defaults(func=_cmd_bench_fit)

    router = sub.add_parser("router", help="train the router's classifiers")
    router_sub = router.add_subparsers(dest="router_command", required=True, metavar="ACTION")
    train = router_sub.add_parser("train", add_help=False,
                                  help="train the topic classifier and gate (options: cutoffmix router train --help)")
    train.set_defaults(func=_cmd_router_train, passthrough=True)

    sim = sub.add_parser("simulate", help="recover known cutoffs from simulated probe results")
    sim.add_argument("--out", default="sim_out", metavar="DIR")
    sim.add_argument("--from", dest="first", type=_parse_month, default=_parse_month("2025-01"), metavar="YYYY-MM")
    sim.add_argument("--to", dest="last", type=_parse_month, default=_parse_month("2026-11"), metavar="YYYY-MM")
    sim.add_argument("--per-cell", type=int, default=4, help="events per topic x month")
    sim.add_argument("--probes", nargs="+", choices=PROBE_CHOICES, default=["current_d", "mcq"])
    sim.add_argument("--bootstrap", type=int, default=50)
    sim.add_argument("--seed", type=int, default=0)
    sim.set_defaults(func=_cmd_simulate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``cutoffmix`` command; returns the exit status."""
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    passthrough = getattr(args, "passthrough", False)
    if extra and not passthrough:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return int(args.func(args, extra) if passthrough else args.func(args))
    except CLIError as e:
        print(f"cutoffmix: {e}", file=sys.stderr)
        return e.code
    except KeyboardInterrupt:
        print("cutoffmix: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:  # the reader went away (e.g. `cutoffmix index show | head`): stop quietly
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())  # so the interpreter's final flush cannot fail again
        return 141  # 128 + SIGPIPE, as a shell reports it


if __name__ == "__main__":
    raise SystemExit(main())
