"""The ``cutoffmix`` command line (DESIGN.md §6).

::

    cutoffmix run "question" [--mock] [--policy P] [--strategy S] [--segments N] [--json OUT] [--config FILE]
                             [--preset NAME] [--ring KEY,KEY[,KEY]]
    cutoffmix serve [--host H] [--port P] [--mock] [--allow-origin ORIGIN...] [--allow-host HOST...] [--preset NAME]
                    [--ring KEY,KEY[,KEY]]
    cutoffmix index show [--path FILE] [--probe P] [--config FILE]
    cutoffmix bench import-kc --out DIR
    cutoffmix bench build-wikidata --out FILE
    cutoffmix bench build-wikipedia --out FILE --start DATE --end DATE [--probe-model KEY]
    cutoffmix bench build-papers --out FILE [--dense-from 2025-10 --dense-to 2026-11 --per-month 12
                        --anchor-from 2024-01 --anchor-to 2025-09 --anchor-every 3 --anchor-per-month 6
                        --interp-share 0.5 --controls-share 0.1 --seed 0 --meta FILE --cache DIR --candidates 300]
    cutoffmix bench run --events FILE --out FILE --models KEY... --probes P... [--dry-run] [--batch] [--limit N]
    cutoffmix bench grade --events FILE... --in FILE --out FILE [--judges KEY...] [--regrade-all] [--mock]
    cutoffmix bench fit --events FILE... --results FILE... --out FILE [--models KEY...] [--probes P...]
                        [--probe-map FROM=TO...] [--global-origins ORIGIN...]
                        [--priors KEY=MONTH...] [--note TEXT]
    cutoffmix bench keys plan --events FILE --results FILE --out FILE [--models K,K,K] [--pairs A:B,...]
                        [--both-right N] [--min-items N] [--seed N] [--today YYYY-MM-DD] [--force]
    cutoffmix bench keys run --plan FILE --events FILE --out FILE [--mock] [--max-cost USD] [--concurrency N]
                        [--limit N] [--judges KEY...] [--resend-stale]
    cutoffmix bench keys report --plan FILE --results FILE [--json OUT] [--md OUT]
    cutoffmix router train [options of cutoffmix.router.train]
    cutoffmix simulate [--out DIR]

``--mock`` uses the simulated models of :mod:`cutoffmix.backends.mock` everywhere. Commands that call the real
models stop with a clear message when the API key variable (``OPENROUTER_API_KEY``) is not set. Benchmark modules
are imported only by the subcommands that need them. ``bench grade`` re-grades a results file into a *new* file
(never in place; ``usage`` and ``latency_s`` are kept): offline with the rule grader alone, with ``--judges`` for
the rows the rules cannot decide; it prints the grade counts per model and probe before and after. ``--preset explore`` (``run`` and ``serve``) merges the packaged
explore preset (DESIGN.md §7) before ``--config``; the segment headers then tag lens, check and continuation notes
and the summary line adds the exploration metrics of the ``done`` event. ``--preset rally`` merges the rally preset
(DESIGN_rally.md): two or three models hand one record to each other in a fixed ring; the segment headers show the kind
and the label counts of each note, and a summary after the run lists the turns, the stop reason and the (self-reported)
frontier. ``--ring KEY,KEY[,KEY]`` sets the ring (and switches the rally on when no preset is given).
``bench keys`` is the key-not-value experiment (DESIGN_keys.md, after Hazy Research's "Plato's Key"): ``plan`` builds
the calls offline from stored graded results (does a donor's note with the answer masked make a recipient recall what
it missed alone?) and prints the cost estimate (it does not replace a plan whose calls differ unless ``--force``,
and never a file that is not a plan);
``run`` sends them (``--max-cost`` is required for the real models and counts the judges; it stops new calls, and the
calls in flight finish, so a run can end above it; with ``--judges`` the judges decide every rule grade incorrect or
abstain of an answer that names the expected answer and every labelled answer that names the candidate, the rules only
a grade correct and the answers that do not name them, and the answers the judges' votes leave undecided are counted,
to be asked again by a rerun with ``--judges``). Its exit status is that of the stop that first kept planned calls
from starting: 3 for the cap, 4 for an error no retry can fix (HTTP 401 or 402, on a recipient's call or a judge's),
130 for Ctrl-C (the first lets the calls in flight finish); stops met at one check rank Ctrl-C, then the error, then
the cap, and a later Ctrl-C or error keeps the first stop's status. Without such a stop it is 4 when planned calls
failed (also at ``--limit``) or an error no retry can fix came after the last call had started, else 0; 2 is a refused
input before any call, or a results file that cannot take a row during the run. A results file holds one backend's
rows, so a mock run writes to its own file. ``report`` joins plan and results offline into rates, paired tests and
adoption rates (it never writes its ``--json`` or ``--md`` over its inputs); no verdict stands that the open offers
(ungraded answers that name the expected answer) could overturn.
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
import signal
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from cutoffmix import __version__
from cutoffmix.relay import RelayRuntime, to_jsonable
from cutoffmix.taxonomy import argmax_topic
from cutoffmix.timeutil import month_label
from cutoffmix.types import ModelSpec, RelayEvent

if TYPE_CHECKING:
    from cutoffmix.config import AppConfig
    from cutoffmix.router.index import CutoffIndex

__all__ = ["CLIError", "TerminalPrinter", "build_parser", "explore_summary", "format_tau", "main", "rally_note_tag",
           "rally_summary", "use_color"]

PROBE_CHOICES = ("current", "current_d", "recall", "mcq")
_ANY_ADDRESS = ("0.0.0.0", "::", "")  # --host values that bind every interface (no host name to allow)


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

    @staticmethod
    def explore_tag(data: Mapping[str, Any]) -> str:
        """The explore tag of a ``segment_start`` (DESIGN.md §7.6: ``data["explore"]`` with ``kind`` lens, check or
        continue, ``field``, ``tier`` and ``cue``; the same keys flat in ``data`` are accepted too), else ``""``."""
        nested = data.get("explore")
        if isinstance(nested, Mapping):
            data = nested
        kind = data.get("kind")
        if kind == "lens":
            tier = data.get("tier")
            return f"lens: {data.get('field') or '?'}" + (f" ({tier})" if tier else "")
        if kind == "check":
            return f"check: {data.get('field') or 'the previous lens'}"
        if kind == "continue":
            cue = str(data.get("cue") or "").strip()
            return f"continue: '{cue}'" if cue else "continue"
        return ""

    @staticmethod
    def rally_tag(data: Mapping[str, Any], end: Mapping[str, Any] | None = None) -> str:
        """The rally tag of a segment: ``rally: reply`` from the ``segment_start`` (``data["rally"]`` with ``kind``,
        ``turn``, ``ring``), extended with the label counts and the declared newest month once the ``segment_end``
        data carries the note record (``end["rally"]``, see :func:`rally_note_tag`)."""
        note = (end or {}).get("rally")
        if isinstance(note, Mapping):
            return "rally: " + rally_note_tag(note)
        nested = data.get("rally")
        if isinstance(nested, Mapping) and nested.get("kind"):
            turn = nested.get("turn")
            return f"rally: {nested['kind']}" + (f" (note {turn})" if turn else "")
        return ""

    def _header(self, ev: RelayEvent, end: RelayEvent | None = None) -> tuple[str, str]:
        """``(styled header, visible header)`` for a segment or answer start; ``end`` (the matching ``segment_end``)
        adds what is only known once the note is written (the rally's label counts)."""
        data = ev.data or {}
        bits = [str(ev.decision or "")]
        if "r" in data:
            bits.append(f"r={data['r']:.2f}")
        if ev.topic_posterior:
            topic = argmax_topic(ev.topic_posterior)  # None: flat posterior, no topic to name
            bits.append(f"{topic} {ev.topic_posterior[topic]:.0%}" if topic else "topic ?")
        if ev.phase == "answer":
            bits.insert(0, "answer")
        meta = " · ".join(b for b in bits if b)
        tag = self.explore_tag(data) or self.rally_tag(data, end.data if end is not None else None)
        chip = self._chip(ev.model)
        plain_chip = f" {self.models[ev.model].label()} " if self.color and ev.model in self.models else chip
        styled = f"{chip} {self._sgr(meta, '2')}" + (f" {self._sgr(f'[{tag}]', '1')}" if tag else "")
        return styled, f"{plain_chip} {meta}" + (f" [{tag}]" if tag else "")

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
                        self._write(self._header(start, ev)[0] + "\n" + self._sgr(ev.text, *self._fg(ev.model)))
                elif not dropped:
                    self._write("\n" + self._sgr("(committed) ", "2") + self._sgr(ev.text, *self._fg(ev.model)))
            elif not dropped and isinstance(data.get("rally"), Mapping):  # streamed as it is: the counts come last
                self._write("\n" + self._sgr("  ↳ " + rally_note_tag(data["rally"]), "2"))
        elif not dropped:
            self._write(self._header(start, ev)[0] + "\n" + self._sgr(ev.text, *self._fg(ev.model)))
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
            rally = (ev.data or {}).get("rally")
            if isinstance(rally, Mapping):
                self._write(self._sgr("\n".join(rally_summary(rally, self.models)), "2") + "\n")

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
        partial = "" if data.get("usage_complete", True) else " (partial)"  # totals are a lower bound
        if usage.get("total_tokens"):
            parts.append(f"tokens {usage['total_tokens']}{partial}")
        if usage.get("cost"):
            parts.append(f"cost ${usage['cost']:.4f}{partial}")
        explore = data.get("explore")
        if isinstance(explore, Mapping):
            parts.extend(explore_summary(explore))
        return " · ".join(parts)


def _count(value: Any) -> int | None:
    """A count from a metric that may be a number, a list (its length) or a mapping (its ``all`` or first value)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, (list, tuple, set)):
        return len(value)
    if isinstance(value, Mapping) and value:
        return _count(value.get("all", next(iter(value.values()))))
    return None


def _share(value: Any) -> float | None:
    """A share in [0, 1] from a coverage metric that may be a number or a mapping of shares (the first one)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    if isinstance(value, Mapping) and value:
        return _share(next(iter(value.values())))
    return None


def explore_summary(explore: Mapping[str, Any]) -> list[str]:
    """Summary parts for the explore metrics of a ``done`` event (DESIGN.md §7): lenses kept/dropped, labels
    coverage, topics visited, forced continuations. Every key is optional."""
    parts: list[str] = []
    lens = explore.get("lens")
    if isinstance(lens, Mapping):
        written = _count(lens.get("written", lens.get("offered")))
        kept, dropped = _count(lens.get("kept")), _count(lens.get("dropped"))
        if written is not None or kept is not None or dropped is not None:
            verdicts = ", ".join(f"{n} {word}" for n, word in ((kept, "kept"), (dropped, "dropped")) if n is not None)
            parts.append(f"lenses {written if written is not None else (kept or 0) + (dropped or 0)}"
                         + (f" ({verdicts})" if verdicts else ""))
    labels = explore.get("labels")
    coverage = _share(labels.get("coverage")) if isinstance(labels, Mapping) else None
    if coverage is not None:
        parts.append(f"labels {coverage:.0%}")
    topics = _count(explore.get("topics_visited"))
    if topics is not None:
        parts.append(f"topics {topics}")
    forced = _count(explore.get("forced_continuations"))
    if forced is not None:
        parts.append(f"forced {forced}")
    wws = explore.get("wws")  # notes without 'Where we stand:' (skipped / cut) and gate waivers; only when non-zero
    if isinstance(wws, Mapping):
        bits = [f"{n} {word}" for word in ("missing", "cut", "waived") if (n := _count(wws.get(word)))]
        if bits:
            parts.append("wws " + ", ".join(bits))
    return parts


def rally_note_tag(note: Mapping[str, Any]) -> str:
    """``reply · +1 update · +2 newer · 1 confirm · newest 2026-03`` for one rally note record
    (``segment_end.data["rally"]``, DESIGN_rally.md section 6); every key is optional."""
    labels = note.get("labels") if isinstance(note.get("labels"), Mapping) else {}
    parts = [str(note.get("kind") or "note")]
    if labels.get("known"):
        parts.append(f"{labels['known']} known")
    for name in ("update", "newer"):
        if labels.get(name):
            parts.append(f"+{labels[name]} {name}")
    for name, word in (("confirm", "confirm"), ("unverified", "unverified"), ("nothing_newer", "nothing newer")):
        if labels.get(name):
            parts.append(word if name == "nothing_newer" else f"{labels[name]} {word}")
    if note.get("kind") == "reply" and note.get("quiet"):
        parts.append("quiet")
    if note.get("self_reply"):
        parts.append("self-reply")
    if note.get("cut"):
        parts.append("cut")
    newest = note.get("newest")
    parts.append(f"newest {newest}" if newest else "newest not stated")
    return " · ".join(parts)


def rally_summary(rally: Mapping[str, Any], models: Mapping[str, ModelSpec] | None = None) -> list[str]:
    """The lines printed after a rally run (DESIGN_rally.md section 6): turns and stop reason, the frontier curve
    (the newest month each note declared, a running maximum; **self-reported**) and what each model added."""
    def name(key: str) -> str:
        spec = (models or {}).get(key)
        return spec.label() if spec is not None else key

    ring = " → ".join(name(k) for k in rally.get("ring") or [])
    head = f"rally · {rally.get('turns', 0)} notes · {rally.get('stop_reason') or 'ended'}"
    head += f" · ring {ring}" if ring else ""
    frontier = rally.get("frontier") if isinstance(rally.get("frontier"), Mapping) else {}
    curve = " → ".join(m or "?" for m in frontier.get("by_turn") or [])
    gain = frontier.get("gain_months")
    rough = frontier.get("gain_months_approx")
    lines = [head]
    if curve:
        lines.append(f"  frontier (self-reported, newest month declared per note): {curve}"
                     + (f" · gain {gain:+d} month{'' if abs(gain) == 1 else 's'} beyond the best opening note"
                        if isinstance(gain, int) and not isinstance(gain, bool) else
                        f" · gain about {rough:+d} month{'' if abs(rough) == 1 else 's'} (approximate or missing dates)"
                        if isinstance(rough, int) and not isinstance(rough, bool) else ""))
    replies = rally.get("self_replies")
    if isinstance(replies, int) and replies:
        lines.append(f"  {replies} note{'' if replies == 1 else 's'} written by the model that wrote the note before "
                     "(its partner could not take the turn); not counted as confirmation")
    skipped = rally.get("skipped_hand_overs")
    if isinstance(skipped, list) and skipped:
        lines.append("  skipped hand-overs: " + ", ".join(
            f"{name(str(row.get('model')))} ({row.get('reason')})" for row in skipped if isinstance(row, Mapping)))
    adds = rally.get("adds_by_model") if isinstance(rally.get("adds_by_model"), Mapping) else {}
    for key, row in adds.items():
        if isinstance(row, Mapping):
            words = (("update", "update"), ("newer", "newer"), ("confirm", "confirm"), ("unverified", "unverified"),
                     ("nothing_newer", "nothing newer"))
            bits = [f"{row[label]} {word}" for label, word in words if row.get(label)]
            lines.append(f"  {name(key)}: " + (", ".join(bits) or "no labelled lines"))
    return lines


# --------------------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------------------


def _ring_overrides(args: argparse.Namespace, overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply ``--ring KEY,KEY[,KEY]``: it implies ``--preset rally`` when no preset is given and is refused with any
    other preset; the models themselves are validated by the configuration (known keys, at least two, no repeats)."""
    raw = getattr(args, "ring", None)
    if raw is None:
        return overrides
    ring = [part.strip() for part in raw.split(",")]
    if len(ring) < 2 or not all(ring):
        raise CLIError(f"--ring expects two or three model keys separated by commas, got {raw!r}")
    preset = getattr(args, "preset", None)
    if preset not in (None, "rally"):
        raise CLIError(f"--ring belongs to the rally preset and cannot be combined with --preset {preset}")
    args.preset = "rally"
    return {**overrides, "rally": {**overrides.get("rally", {}), "ring": ring, "enabled": True}}


def _load_cfg(args: argparse.Namespace, overrides: dict[str, Any] | None = None) -> AppConfig:
    from cutoffmix.config import load_config

    overrides = _ring_overrides(args, dict(overrides or {}))
    try:
        return load_config(getattr(args, "config", None), overrides or None, preset=getattr(args, "preset", None))
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
    if not args.question.strip():
        raise CLIError("the question must not be empty")
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
    from cutoffmix import server

    cfg = _load_cfg(args)
    if not args.mock:
        _require_key(cfg, "'cutoffmix serve'")
    hosts = list(args.allow_host)
    if args.host in _ANY_ADDRESS:
        if not hosts:
            print(f"cutoffmix: listening on all interfaces, but only {', '.join(server.LOOPBACK_HOSTS)} and the "
                  "hosts of --allow-origin are served; add --allow-host NAME for the name or address other "
                  "machines use", file=sys.stderr)
    else:
        hosts.append(args.host)
    app = server.create_app(cfg, mock=args.mock, allowed_origins=args.allow_origin, allowed_hosts=hosts,
                            config_path=args.config)
    modes = [*(["simulated models"] if args.mock else []), *([f"preset {args.preset}"] if args.preset else [])]
    print(f"cutoffmix{' (' + ', '.join(modes) + ')' if modes else ''} on http://{args.host}:{args.port}/",
          file=sys.stderr)
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

    cfg = _load_cfg(args)  # router.index_path and router.probe_preference, as routing and /api/index use them
    try:
        index = load_index(args.path or cfg.router.index_path)
    except (OSError, ValueError) as e:
        raise CLIError(f"cannot load the index: {e}") from e
    if index is None:
        raise CLIError("no cutoff index: pass --path FILE (written by 'cutoffmix bench fit'), or ship "
                       "cutoffmix/data/index_prior.json", code=1)
    probe = args.probe or index.primary(cfg.router.probe_preference).probe
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


def _cmd_build_papers(args: argparse.Namespace) -> int:
    from cutoffmix.bench.io import save_events, write_jsonl
    from cutoffmix.bench.sources.papers import build_paper_events

    errors: list[str] = []
    try:
        events, meta = build_paper_events(
            args.dense_from, args.dense_to, per_month=args.per_month, anchor_from=args.anchor_from,
            anchor_to=args.anchor_to, anchor_every=args.anchor_every, anchor_per_month=args.anchor_per_month,
            interp_share=args.interp_share, controls_share=args.controls_share, seed=args.seed,
            candidates_per_month=args.candidates, cache_dir=args.cache, errors=errors)
    except ValueError as e:
        raise CLIError(str(e)) from e
    save_events(args.out, events)
    meta_path = args.meta or str(Path(args.out).with_name("papers_meta.jsonl"))
    write_jsonl(meta_path, meta)
    by_id = {m["event_id"]: m for m in meta}
    controls = sum(e.control for e in events)
    interp = sum(1 for e in events if by_id[e.id].get("interp"))
    months = sorted({e.month for e in events})
    print(f"{len(events)} events ({len(events) - controls} papers, {interp} interp-tagged, {controls} controls) over "
          f"{len(months)} months {months[0] if months else '-'}..{months[-1] if months else '-'} -> {args.out}\n"
          f"meta -> {meta_path}")
    if errors:
        print(f"{len(errors)} daily list(s) could not be fetched (rerun fills them from the network):\n  "
              + "\n  ".join(errors), file=sys.stderr)
    return 0


async def _close(backend: Any) -> None:
    close = getattr(backend, "aclose", None)
    if close is not None:
        await close()


def _token_spec(values: Sequence[str], what: str) -> int | dict[str, int]:
    """``["600"]`` -> ``600``; ``["claude-fable-5.2=350", "default=60"]`` -> ``{"claude-fable-5.2": 350, "default":
    60}`` (a bare number among KEY=N entries is the default)."""
    out: dict[str, int] = {}
    for item in values:
        key, _, number = item.partition("=")
        if not number:
            key, number = "default", key
        try:
            n = int(number)
        except ValueError:
            raise CLIError(f"--{what.replace('_', '-')}: expected N or KEY=N, got {item!r}") from None
        if n < 0:
            raise CLIError(f"--{what.replace('_', '-')}: token counts must be >= 0, got {item!r}")
        out[key.strip()] = n
    if list(out) == ["default"]:
        return out["default"]
    return out


def _cmd_bench_run(args: argparse.Namespace) -> int:
    from cutoffmix.backends import make_backend
    from cutoffmix.bench.io import load_events
    from cutoffmix.bench.runner import BATCH_UNSUPPORTED, estimate_cost, run_probes

    if args.batch and not args.dry_run:
        raise CLIError(BATCH_UNSUPPORTED)
    cfg = _load_cfg(args)
    events = load_events(args.events)
    if args.limit is not None:
        events = events[: args.limit]
    models = [_model_spec(cfg, k) for k in (args.models or cfg.model_keys())]
    judges = [_model_spec(cfg, k) for k in args.judges] or None
    extra: dict[str, Any] = {}  # only what was given, so the estimator's defaults stay its own
    if args.max_tokens is not None:
        extra["max_tokens"] = args.max_tokens
    if args.dry_run:
        for name, spec in (("in_tokens", args.in_tokens), ("out_tokens", args.out_tokens)):
            if spec:
                extra[name] = _token_spec(spec, name)
        if args.out_tokens_measured:
            extra["measured"] = True
        print(json.dumps(estimate_cost(models, events, args.probes, batch=args.batch, **extra), indent=2))
        return 0
    if not args.mock:
        _require_key(cfg, "'cutoffmix bench run'")
    backend = make_backend(cfg, mock=args.mock, events=events)

    async def go() -> list[Any]:
        try:
            return await run_probes(backend, models, events, args.probes, args.out, today=cfg.today,
                                    concurrency=args.concurrency, use_batch_variant=args.batch, judges=judges,
                                    progress=_progress(sys.stderr), **extra)
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


_GRADE_ORDER = ("correct", "outdated", "incorrect", "abstain", "ungraded")


def _grade_table(before: Sequence[Any], after: Sequence[Any]) -> str:
    """Grade counts per model and probe (and per model over all probes, ``all``) before and after re-grading, one
    line each: a count that changed reads ``before->after``; the last column counts the rows whose grade or hedge
    changed. ``before`` and ``after`` hold the same rows in the same order."""
    counts: dict[tuple[str, str], dict[str, list[int]]] = {}  # (model, probe) -> grade -> [before, after]
    changed: dict[tuple[str, str], int] = {}
    for b, a in zip(before, after, strict=True):
        for key in ((b.model, b.probe), (b.model, "all")):
            row = counts.setdefault(key, {})
            row.setdefault(b.grade, [0, 0])[0] += 1
            row.setdefault(a.grade, [0, 0])[1] += 1
            changed[key] = changed.get(key, 0) + int(a.grade != b.grade or a.hedged != b.hedged)
    seen = {g for row in counts.values() for g in row}
    grades = [g for g in _GRADE_ORDER if g in seen] + sorted(seen - set(_GRADE_ORDER))
    lines = [f"  {'model':<24} {'probe':<10} " + " ".join(f"{g:>11}" for g in grades) + f" {'changed':>8}"]
    for key in sorted(counts, key=lambda k: (k[0], k[1] == "all", k[1])):
        cells = [f"{b}->{a}" if b != a else str(a) for b, a in (counts[key].get(g, [0, 0]) for g in grades)]
        lines.append(f"  {key[0]:<24} {key[1]:<10} " + " ".join(f"{c:>11}" for c in cells)
                     + f" {changed[key]:>8}")
    return "\n".join(lines)


def _cmd_bench_grade(args: argparse.Namespace) -> int:
    from cutoffmix.backends import make_backend
    from cutoffmix.bench.grading import grade_results
    from cutoffmix.bench.io import load_events, load_results, save_results

    src, out = Path(args.results), Path(args.out)
    if src.resolve() == out.resolve() or (src.exists() and out.exists() and src.samefile(out)):
        raise CLIError(f"--out must be a new file, not --in itself ({args.results}): results are never re-graded in "
                       "place")
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
    save_results(out, graded)
    print(_grade_table(results, graded))
    changed = sum(a.grade != b.grade or a.hedged != b.hedged for b, a in zip(results, graded, strict=True))
    print(f"{len(graded)} results ({changed} changed) -> {out}")
    return 0


def _cmd_bench_fit(args: argparse.Namespace) -> int:
    import dataclasses
    from typing import cast

    from cutoffmix.bench.estimate import fit_index
    from cutoffmix.bench.io import load_events, load_results
    from cutoffmix.types import ProbeKind

    probe_map: dict[str, str] = {}
    for item in args.probe_map:
        src, sep, dst = item.partition("=")
        if not sep or not src or dst not in PROBE_CHOICES:
            raise CLIError(f"--probe-map expects FROM=TO with TO one of {', '.join(PROBE_CHOICES)}, got {item!r}")
        if src in probe_map:
            raise CLIError(f"--probe-map maps {src!r} twice")
        probe_map[src] = dst
    events = [e for path in args.events for e in load_events(path)]
    results = [r for path in args.results for r in load_results(path)]
    if probe_map:  # relabel at fit time, e.g. current_d=current (the dated paper probes join the kc 'current' table)
        results = [dataclasses.replace(r, probe=cast(ProbeKind, probe_map[r.probe])) if r.probe in probe_map else r
                   for r in results]
    meta: dict[str, Any] = {}
    if args.note:
        meta["note"] = args.note
    if probe_map:
        meta["probe_map"] = probe_map
    index = fit_index(events, results, models=args.models or None, probes=args.probes or None,
                      priors=_parse_priors(args.priors) or None, prior_sd=args.prior_sd, bootstrap=args.bootstrap,
                      seed=args.seed, global_origins=args.global_origins or None, meta=meta or None)
    index.save(args.out)
    print(render_index(index, index.primary().probe, color=use_color(sys.stdout)))
    print(f"index -> {args.out}")
    return 0


# ---- bench keys (DESIGN_keys.md) --------------------------------------------------------------------------------


def _key_list(value: str) -> list[str]:
    keys = [k.strip() for k in value.split(",") if k.strip()]
    if not keys:
        raise argparse.ArgumentTypeError(f"expected KEY,KEY[,KEY], got {value!r}")
    return keys


def _pair_list(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in (part.strip() for part in value.split(",")):
        donor, sep, recipient = item.partition(":")
        if not item or not sep or not donor.strip() or not recipient.strip() or donor.strip() == recipient.strip():
            raise argparse.ArgumentTypeError(f"expected DONOR:RECIPIENT[,DONOR:RECIPIENT...] with two different keys, "
                                             f"got {item!r}")
        pairs.append((donor.strip(), recipient.strip()))
    return pairs


def _check_keys(cfg: AppConfig, keys: Sequence[str], option: str) -> None:
    unknown = [k for k in keys if k not in cfg.model_keys()]
    if unknown:
        raise CLIError(f"{option}: unknown model(s) {', '.join(unknown)}; configured: {', '.join(cfg.model_keys())}")


def _keys_input(load: Callable[[str], Any], path: str, option: str, what: str) -> Any:
    """``load(path)`` for an input file of ``bench keys``: a missing or unreadable file, or a file of another kind (a
    row the loader cannot take), becomes a one-line :class:`CLIError` that names the option."""
    try:
        return load(path)
    except FileNotFoundError:
        raise CLIError(f"{option} {path}: no such file") from None
    except OSError as e:
        raise CLIError(f"{option} {path}: cannot read it ({e.strerror or e})") from e
    except (ValueError, TypeError, KeyError) as e:
        raise CLIError(f"{option} {path} is not {what} ({type(e).__name__}: {' '.join(str(e).split())})") from e


def _plan_conflict(path: Path, rows: Sequence[Mapping[str, Any]] | None, *, force: bool = False,
                   inputs: Mapping[str, str] | None = None) -> str | None:
    """Why ``plan`` may not write its plan to ``path`` (DESIGN_keys.md section 6), or None. Refused even with
    ``force``: ``path`` is one of the ``inputs`` (option -> file: ``--results``, ``--events``), or an existing file
    that is not a plan (probe results, events, keys results, unreadable JSON, a directory); ``--force`` replaces only
    an earlier plan. Refused without ``force``: an earlier plan whose shared ids carry other messages or another
    ``donor_answer`` than the new ``rows`` (results answered under it would no longer match: a changed name is
    covered by neither sha, so a run could not tell). A plan with the same messages and donor answers for every
    shared id is replaced. ``rows=None`` checks only what does not depend on the new plan."""
    from cutoffmix.bench.io import read_jsonl
    from cutoffmix.bench.keys import prompt_sha

    for option, other in (inputs or {}).items():
        with contextlib.suppress(OSError):
            if os.path.samefile(path, other):
                return f"--out {path} is the {option} file: write the plan to a file of its own"
    if not path.exists():
        return None
    keep = "remove it first or write the plan to another --out (--force replaces only an earlier plan)"
    if not path.is_file():
        return f"--out {path} exists and is not a regular file; {keep}"
    try:
        old = read_jsonl(path)
    except (OSError, ValueError) as e:
        return f"--out {path} exists and is not a readable plan ({' '.join(str(e).split())}); {keep}"
    for n, row in enumerate(old, 1):
        if not row.get("id") or not isinstance(row.get("messages"), list):
            return f"--out {path} exists and is not a plan (row {n} has no id or messages); {keep}"
    if force or rows is None:
        return None
    new = {str(r["id"]): (prompt_sha(r["messages"]), r.get("donor_answer")) for r in rows}
    changed = [r for r in old
               if str(r["id"]) in new and (prompt_sha(r["messages"]), r.get("donor_answer")) != new[str(r["id"])]]
    if changed:
        named = sum(1 for r in changed if prompt_sha(r["messages"]) == new[str(r["id"])][0])
        return (f"--out {path} holds a plan whose calls differ: {len(changed)} of its {len(old)} ids carry other "
                f"messages or another donor answer ({named} only the donor answer; another --today, --seed, template, "
                "masking or name rule), so results answered under it would no longer match; pass --force to replace "
                "it, or write the new plan to another --out")
    return None


def _first_sigint_sets(stop: asyncio.Event, notice: Callable[[], None]) -> Callable[[], None]:
    """Make the first SIGINT (Ctrl-C) set ``stop`` instead of aborting (DESIGN_keys.md section 6: the run starts no new
    call, and the calls in flight finish and write their rows), and return the function that undoes it. That first
    SIGINT also puts the previous handler back, so a second one aborts at once, as before, and has the loop call
    ``notice``. Called inside the running loop. The handler is a Python signal handler, so it runs in the main thread
    right after the signal arrives, between two bytecodes, not on a later turn of the loop: ``stop`` is set before the
    run checks it again, also when the signal comes before the first call has started (the run then starts none).
    Nothing awaits ``stop`` (the run reads it before it starts each item), so setting it there is safe; ``notice``
    prints, so it runs on the loop instead. POSIX only, in the main thread, with SIGINT handled by Python; elsewhere
    nothing changes and a SIGINT aborts at once."""
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    if os.name != "posix" or not callable(previous):  # not POSIX; or SIGINT ignored or left to the system: keep it so
        return lambda: None
    armed = False

    def disarm() -> None:
        nonlocal armed
        if armed:
            armed = False
            signal.signal(signal.SIGINT, previous)  # from now on a SIGINT aborts as it would have

    def first(_signum: int, _frame: Any) -> None:
        disarm()
        stop.set()
        loop.call_soon_threadsafe(notice)  # wakes the loop; printing inside a signal handler could hit a busy stream

    try:
        signal.signal(signal.SIGINT, first)
    except (ValueError, OSError):  # not the main thread
        return lambda: None
    armed = True
    return disarm


def _keys_run_start(start: Any, limit: int | None) -> str:
    """The line ``bench keys run`` prints before its first call: what is answered, what it sends, what it re-grades
    (``start`` is the run's :class:`cutoffmix.bench.keys.RunStart`)."""
    line = f"{start.planned} planned calls: {start.answered} already answered, {start.pending} without a current answer"
    detail = [f"{n} {what}" for n, what in ((start.errors, "after an error"),
                                            (start.stale, "answered under another request")) if n]
    line += (f" ({', '.join(detail)})" if detail else "") + f"; sending {start.to_send} now"
    if start.to_send < start.pending:
        line += f" (--limit {limit})"
    if start.regrade:
        judges = start.regrade - start.failed - start.rule_grade - start.relabel - start.offer_flag
        why = [f"{n} {what}" for n, what in ((start.failed, "whose grading failed"),
                                             (start.rule_grade, "whose rule grade is out of date"),
                                             (start.relabel, "whose label, name or rule decision is out of date"),
                                             (judges, "for the judges"),
                                             (start.offer_flag, "whose offers_expected is missing or out of date"))
               if n]
        line += f"; grading {start.regrade} stored answers again first, without a new call ({', '.join(why)})"
    return line + f"; cost so far ${start.cost_usd:.4f} ({start.backend})"


def _cmd_bench_keys_plan(args: argparse.Namespace) -> int:
    import datetime as _dt

    from cutoffmix.bench import keys
    from cutoffmix.bench.io import load_events, load_results, write_jsonl

    cfg = _load_cfg(args)
    if args.models:
        _check_keys(cfg, args.models, "--models")
    if args.pairs:
        _check_keys(cfg, [k for pair in args.pairs for k in pair], "--pairs")
    if args.both_right < 0 or args.min_items < 1:
        raise CLIError("--both-right must be >= 0 and --min-items >= 1")
    today = args.today or cfg.today
    if today is not None:
        try:
            _dt.date.fromisoformat(today)
        except ValueError:
            raise CLIError(f"--today must be an ISO date (YYYY-MM-DD), got {today!r}") from None
    events = _keys_input(load_events, args.events, "--events", "an events file")
    results = _keys_input(load_results, args.results, "--results", "a probe results file")
    inputs = {"--results": args.results, "--events": args.events}
    if conflict := _plan_conflict(Path(args.out), None, inputs=inputs):  # before the work: --force does not help
        raise CLIError(conflict)
    try:
        rows, summary = keys.build_plan(events, results, models=args.models or None, pairs=args.pairs or None,
                                        both_right=args.both_right, min_items=args.min_items, seed=args.seed,
                                        today=today)
    except ValueError as e:
        raise CLIError(str(e)) from e
    if conflict := _plan_conflict(Path(args.out), rows, force=args.force, inputs=inputs):
        raise CLIError(conflict)
    try:
        write_jsonl(args.out, rows)
    except OSError as e:
        raise CLIError(f"cannot write --out {args.out} ({e.strerror or e})") from e
    print(json.dumps(to_jsonable(summary), ensure_ascii=False, indent=2))
    print(json.dumps(to_jsonable({"cost_estimate": keys.estimate_cost(rows, results)}), ensure_ascii=False, indent=2))
    print(f"{len(rows)} planned calls -> {args.out} (nothing was sent; 'cutoffmix bench keys run' sends them)")
    return 0


def _cmd_bench_keys_run(args: argparse.Namespace) -> int:
    from cutoffmix.backends import make_backend
    from cutoffmix.bench import keys
    from cutoffmix.bench.io import load_events, read_jsonl
    from cutoffmix.bench.keys.run import GRADING_FAILED, check_plan

    cfg = _load_cfg(args)
    if args.max_cost is not None and not args.max_cost > 0:
        raise CLIError(f"--max-cost must be a positive number of dollars, got {args.max_cost}")
    if args.limit is not None and args.limit < 0:
        raise CLIError(f"--limit must be >= 0, got {args.limit}")
    if args.concurrency < 1:
        raise CLIError(f"--concurrency must be >= 1, got {args.concurrency}")
    if not args.mock:
        _require_key(cfg, "'cutoffmix bench keys run'")
        if args.max_cost is None:
            raise CLIError("--max-cost USD is required when the real models are called ('bench keys plan' prints the "
                           "estimate); add --mock to run the simulated models")
    plan = _keys_input(read_jsonl, args.plan, "--plan", "a plan")
    events = _keys_input(load_events, args.events, "--events", "an events file")
    if not plan:
        raise CLIError(f"the plan {args.plan} has no calls")
    try:
        check_plan(plan)  # before the model keys are read from it: a file of another kind gets this message
    except ValueError as e:
        raise CLIError(f"--plan {args.plan}: {' '.join(str(e).split())}") from e
    recipients = sorted({str(row.get("recipient")) for row in plan})
    _check_keys(cfg, recipients, "the plan's recipients")
    models = [cfg.model(k) for k in recipients]
    judges = [_model_spec(cfg, k) for k in args.judges] or None
    backend = make_backend(cfg, mock=args.mock, events=events)
    line_open = False  # the progress line on stderr has no newline yet

    def end_line() -> None:
        nonlocal line_open
        if line_open:
            sys.stderr.write("\n")
            sys.stderr.flush()
            line_open = False

    def progress(done: int, total: int, _row: Any) -> None:
        nonlocal line_open
        sys.stderr.write(f"\r  {done}/{total}")
        line_open = done != total
        if not line_open:
            sys.stderr.write("\n")
        sys.stderr.flush()

    def on_start(start: Any) -> None:
        print(_keys_run_start(start, args.limit), flush=True)

    def on_interrupt() -> None:
        end_line()
        print("interrupted: no new calls are started; the calls in flight finish and are written (Ctrl-C again "
              "aborts them)", file=sys.stderr, flush=True)

    async def go() -> Any:
        stop = asyncio.Event()
        restore = _first_sigint_sets(stop, on_interrupt)
        try:
            return await keys.run_plan(backend, models, plan, args.out, events=events, concurrency=args.concurrency,
                                       max_cost=args.max_cost, limit=args.limit, judges=judges, progress=progress,
                                       backend_name="mock" if args.mock else "openrouter",
                                       resend_stale=args.resend_stale, on_start=on_start, stop=stop)
        finally:
            restore()
            await _close(backend)

    try:
        outcome = asyncio.run(go())
    except (OSError, ValueError) as e:  # the run's refusals, its argument checks, a file it cannot use: one line
        raise CLIError(" ".join(str(e).split())) from e
    finally:
        end_line()  # stopped before the last planned call: end the progress line
    def failed(row: Mapping[str, Any]) -> bool:  # a send error: no answer (a failed grading keeps its answer)
        return bool(row.get("error")) and not str(row["error"]).startswith(GRADING_FAILED)
    errors = [r for r in outcome.rows if failed(r)]
    grades = Counter(str(r.get("grade")) for r in outcome.rows if not failed(r))
    tally = [f"{g} {n}" for g, n in sorted(grades.items())] + ([f"errors {len(errors)}"] if errors else [])
    print("  " + "  ".join(tally))
    print(f"{len(outcome.rows)} rows -> {args.out} · sent {outcome.sent} calls, graded {outcome.regraded} stored "
          f"answers again · cost ${outcome.cost_usd:.4f} (recipients ${outcome.recipients_usd:.4f} + judges "
          f"${outcome.judges_usd:.4f}; this run ${outcome.run_cost_usd:.4f})"
          + (" (simulated models)" if args.mock else ""))
    if outcome.not_graded:
        first = next((str(r["error"])[len(GRADING_FAILED):] for r in outcome.rows
                      if str(r.get("error") or "").startswith(GRADING_FAILED)), "")
        print(f"{outcome.not_graded} answers could not be graded ({first}); they are kept and counted in the cost, "
              "and the next run grades them again from the stored responses, without a new call", file=sys.stderr)
    if outcome.judge_failed:
        print(f"{outcome.judge_failed} judge calls failed (first error: {outcome.judge_error}); an answer they leave "
              "undecided stays undecided, and the next run with --judges asks again", file=sys.stderr)
    if outcome.judges_undecided:  # also a tie or a reply without a verdict, which no failed call records
        print(f"{outcome.judges_undecided} answers are undecided after the judges' votes (a failed call, a tie or a "
              "reply without a verdict): rerun with --judges before reporting; it asks the judges again, without a "
              "new recipient call", file=sys.stderr)
    # the planned calls without a current answer: those whose last row is a send error were sent and failed
    never = outcome.not_sent - len(errors)
    left = f"{never} of {outcome.planned} planned calls not sent"
    if errors:
        left = f"{len(errors)} of {outcome.planned} planned calls failed and {never} not sent"
    if outcome.not_regraded:
        left += f", {outcome.not_regraded} stored answers not graded again"
    # the exit status of the stop that first kept planned work from starting (DESIGN_keys.md section 6), then 4 for
    # failed calls (also at --limit), else 0
    if outcome.stopped_by == "interrupted":
        print(f"interrupted: {left}; rerun to resume", file=sys.stderr)
        return 130
    if outcome.stopped_by == "max_cost":
        print(f"stopped at --max-cost ${args.max_cost:g}: {left}; rerun with a higher --max-cost to resume",
              file=sys.stderr)
        return 3
    if outcome.stopped_by == "error":
        print(f"stopped after an error that no retry can fix ({outcome.error}): {left}; fix its cause (the API key "
              "or the credits) and rerun to resume", file=sys.stderr)
        return 4
    if outcome.stopped_by == "limit":
        print(f"stopped at --limit {args.limit}: {left}; rerun to send the next ones", file=sys.stderr)
    if errors:
        print(f"{len(errors)} of {outcome.planned} planned calls failed (first error: {errors[0]['error']}); rerun "
              "to send them again", file=sys.stderr)
        return 4
    return 0


def _same_file(a: str, b: str) -> bool:
    """Whether the paths ``a`` and ``b`` name one file: :func:`os.path.samefile` when both exist (a symlink or a hard
    link to the file is the file), else their resolved paths."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return Path(a).resolve() == Path(b).resolve()


def _report_conflict(args: argparse.Namespace) -> str | None:
    """Why ``report`` may not write its ``--json`` or ``--md`` (DESIGN_keys.md section 6), or None: one of them is the
    ``--plan`` or ``--results`` file (the report would replace the plan or the paid results), or both are one file."""
    outputs = [(option, path) for option, path in (("--json", args.json), ("--md", args.md)) if path]
    for option, path in outputs:
        for source, other in (("--plan", args.plan), ("--results", args.results)):
            if _same_file(path, other):
                return f"{option} {path} is the {source} file: write the report to a file of its own"
    if len(outputs) == 2 and _same_file(args.json, args.md):
        return f"--json {args.json} and --md {args.md} are the same file: give each its own"
    return None


def _cmd_bench_keys_report(args: argparse.Namespace) -> int:
    from cutoffmix.bench import keys
    from cutoffmix.bench.io import read_jsonl
    from cutoffmix.bench.keys.run import check_plan, read_results

    if conflict := _report_conflict(args):  # before anything is read or written
        raise CLIError(conflict)
    plan = _keys_input(read_jsonl, args.plan, "--plan", "a plan")
    if not plan:
        raise CLIError(f"the plan {args.plan} has no calls")
    try:
        check_plan(plan)  # a file of another kind (events, results) gets a one-line error, not a report on it
    except ValueError as e:
        raise CLIError(f"--plan {args.plan}: {' '.join(str(e).split())}") from e
    results = _keys_input(read_results, args.results, "--results", "a results file of 'bench keys run'")
    report = keys.build_report(plan, results)
    text = keys.render_report(report)
    print(text)

    def write_json(target: str) -> None:
        _write_json(target, report)

    def write_md(target: str) -> None:
        path = Path(target)
        if path.parent != Path():
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text.rstrip() + "\n", encoding="utf-8")
    for option, target, write in (("--json", args.json, write_json), ("--md", args.md, write_md)):
        if target:
            try:
                write(target)
            except OSError as e:  # e.g. a directory: one line, no traceback
                raise CLIError(f"cannot write {option} {target} ({e.strerror or e})") from e
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


def _add_preset(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", metavar="NAME",
                   help="packaged preset merged before --config: 'explore' (structured notes, cross-field lenses, a "
                        "record gate before the answer; DESIGN.md section 7) or 'rally' (two or three models hand "
                        "the record back and forth, each adding what it knows that is newer; DESIGN_rally.md)")
    p.add_argument("--ring", metavar="KEY,KEY[,KEY]",
                   help="rally hand-over order, e.g. claude-fable-5.2,gemini-4-neon; implies --preset rally")


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
    run.add_argument("--seed", type=int,
                     help="seed for routing, the simulated models and the explore preset's client-side draws")
    run.add_argument("--today", metavar="YYYY-MM-DD", help="date given to the models (default: today)")
    run.add_argument("--json", metavar="OUT", help="write the trace (with usage and latency) as JSON")
    _add_preset(run)
    _add_config(run)
    run.set_defaults(func=_cmd_run)

    serve = sub.add_parser("serve", help="serve the web UI and the OpenAI-compatible API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--mock", action="store_true", help="use the simulated models")
    serve.add_argument("--allow-origin", action="append", default=[], metavar="ORIGIN",
                       help="a browser origin (scheme://host[:port]) besides the server's own that may POST to the "
                            "API, e.g. the public URL of a reverse proxy (its host is also served); repeatable")
    serve.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                       help="a host name or address clients may use in Host besides localhost and --host, e.g. a "
                            "LAN address or a proxy's public name ('*' for any: no DNS-rebinding protection); "
                            "repeatable")
    _add_preset(serve)
    _add_config(serve)
    serve.set_defaults(func=_cmd_serve)

    index = sub.add_parser("index", help="inspect the cutoff index")
    index_sub = index.add_subparsers(dest="index_command", required=True, metavar="ACTION")
    show = index_sub.add_parser("show", help="print the topic x model cutoff table")
    show.add_argument("--path", help="index JSON (default: router.index_path of the config, else the packaged prior)")
    show.add_argument("--probe", choices=PROBE_CHOICES, help="default: the first of router.probe_preference")
    _add_config(show)
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

    pp = bench_sub.add_parser("build-papers",
                              help="first-author probes from AI/ML papers (Hugging Face daily papers + arXiv API)")
    pp.add_argument("--out", required=True, metavar="FILE")
    pp.add_argument("--dense-from", default="2025-10", metavar="YYYY-MM", help="first month with --per-month papers")
    pp.add_argument("--dense-to", default="2026-11", metavar="YYYY-MM", help="last month with --per-month papers")
    pp.add_argument("--per-month", type=int, default=12)
    pp.add_argument("--anchor-from", default="2024-01", metavar="YYYY-MM", help="first anchor month")
    pp.add_argument("--anchor-to", default="2025-09", metavar="YYYY-MM", help="last anchor month")
    pp.add_argument("--anchor-every", type=int, default=3, help="every n-th month of the anchor range")
    pp.add_argument("--anchor-per-month", type=int, default=6)
    pp.add_argument("--interp-share", type=float, default=0.5,
                    help="share of a month's papers taken from the interpretability-tagged pool when available")
    pp.add_argument("--controls-share", type=float, default=0.1, help="fake-paper controls as a share of real papers")
    pp.add_argument("--seed", type=int, default=0, help="MCQ order, distractor draw and control titles")
    pp.add_argument("--meta", metavar="FILE", help="side table (default: papers_meta.jsonl next to --out)")
    pp.add_argument("--cache", metavar="DIR", help="raw-response cache (default: $CUTOFFMIX_PAPERS_CACHE or "
                                                  "~/.cache/cutoffmix/papers)")
    pp.add_argument("--candidates", type=int, default=300,
                    help="Hugging Face top-N per month before filtering (deep, so that the interp pool is not empty; "
                         "the non-interp picks are the top-upvoted anyway)")
    pp.set_defaults(func=_cmd_build_papers)

    br = bench_sub.add_parser("run", help="probe the models on benchmark events")
    br.add_argument("--events", required=True, metavar="FILE")
    br.add_argument("--out", required=True, metavar="FILE", help="results JSONL (resumed if it exists)")
    br.add_argument("--models", nargs="*", metavar="KEY", help="default: every configured model")
    br.add_argument("--probes", nargs="+", choices=PROBE_CHOICES, default=list(PROBE_CHOICES))
    br.add_argument("--dry-run", action="store_true", help="only print the call and cost estimate")
    br.add_argument("--batch", action="store_true",
                    help="not supported (OpenRouter serves :batch ids only through its Batch API); with --dry-run "
                         "it only notes that no batch discount applies")
    br.add_argument("--limit", type=int, metavar="N", help="only the first N events")
    br.add_argument("--judges", nargs="*", default=[], metavar="KEY", help="judge models for open answers")
    br.add_argument("--concurrency", type=int, default=8)
    br.add_argument("--max-tokens", type=int, metavar="N",
                    help="completion budget per call, hidden reasoning included (default 4096; ~1500 suffices for "
                         "first-author probes and caps the worst-case cost shown by --dry-run)")
    br.add_argument("--in-tokens", nargs="+", metavar="N|KEY=N", default=[],
                    help="--dry-run: prompt tokens per call, one number or per model key (default 200)")
    br.add_argument("--out-tokens", nargs="+", metavar="N|KEY=N", default=[],
                    help="--dry-run: completion tokens per call incl. hidden reasoning, one number or per model key "
                         "(e.g. claude-fable-5.2=350 gemini-4-neon=900 default=60; default 600)")
    br.add_argument("--out-tokens-measured", action="store_true",
                    help="--dry-run: mark the --out-tokens values as measured on a pilot")
    br.add_argument("--mock", action="store_true", help="probe the simulated models")
    _add_config(br)
    br.set_defaults(func=_cmd_bench_run)

    bg = bench_sub.add_parser("grade", help="re-grade stored results into a new file: the rules offline, judges for "
                                           "unsure open answers when given; prints the grades before and after")
    bg.add_argument("--events", required=True, nargs="+", metavar="FILE")
    bg.add_argument("--in", "--results", dest="results", required=True, metavar="FILE",
                    help="results JSONL to re-grade (read only)")
    bg.add_argument("--out", required=True, metavar="FILE",
                    help="new results JSONL (never --in itself); usage and latency are kept")
    bg.add_argument("--judges", nargs="*", default=[], metavar="KEY",
                    help="judge models for the rows the rules cannot decide (default: none, offline)")
    bg.add_argument("--regrade-all", action="store_true",
                    help="re-grade every row with a response, not only the ungraded ones")
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
    bf.add_argument("--global-origins", nargs="+", metavar="ORIGIN",
                    help="fit the model-level cutoff, slope and plateau from events of these origins only (e.g. kc, "
                         "the world events); topic cells, offsets, draws and hazard rates still use every row, so "
                         "topics without own data inherit this anchor (DESIGN.md §4)")
    bf.add_argument("--probe-map", nargs="+", default=[], metavar="FROM=TO",
                    help="relabel result probes before the fit, e.g. current_d=current: the paper probes were asked "
                         "with the dated deployment prompt (current_d) but join the kc direct probe 'current' so one "
                         "per-topic table exists; the paper-specific difference lands in the ai_ml topic cell")
    bf.add_argument("--note", help="free text stored as meta.note (what the index is, what it may be used for)")
    bf.set_defaults(func=_cmd_bench_fit)

    bk = bench_sub.add_parser("keys", help="key, not value (DESIGN_keys.md): does another model's note with the answer "
                                          "masked unlock a model's own memory, or does it only copy?")
    keys_sub = bk.add_subparsers(dest="keys_command", required=True, metavar="STEP")
    kp = keys_sub.add_parser("plan", help="build the calls offline from stored graded results and print the counts and "
                                          "the cost estimate (nothing is sent)")
    kp.add_argument("--events", required=True, metavar="FILE")
    kp.add_argument("--results", required=True, metavar="FILE",
                    help="stored graded results (current_d and mcq) of the donors and recipients")
    kp.add_argument("--out", required=True, metavar="FILE", help="plan JSONL, one row per call")
    kp.add_argument("--models", type=_key_list, metavar="K,K,K", help="models to pair (default: all configured)")
    kp.add_argument("--pairs", type=_pair_list, metavar="A:B,...",
                    help="directed donor:recipient pairs (default: every pair of --models)")
    kp.add_argument("--both-right", type=int, default=10, metavar="N",
                    help="items per pair where donor and recipient were both right (harm check)")
    kp.add_argument("--min-items", type=int, default=5, metavar="N", help="smallest dark set a pair is planned with")
    kp.add_argument("--seed", type=int, default=0)
    kp.add_argument("--today", metavar="YYYY-MM-DD", help="the date in the probe's system prompt (default: config or today)")
    kp.add_argument("--force", action="store_true",
                    help="replace an earlier plan even when its ids carry other messages (the results answered "
                         "under it turn stale) or other donor answers (the next run labels their adoption again); a "
                         "file that is not a plan, or the --results or --events file, is never replaced")
    _add_config(kp)
    kp.set_defaults(func=_cmd_bench_keys_plan)
    krun = keys_sub.add_parser("run", help="send the planned calls (resumable) and grade the answers",
                               epilog="exit status: that of the stop that first kept planned calls from starting: 3 "
                                      "--max-cost, 4 an error no retry can fix (HTTP 401 or 402, a recipient's or a "
                                      "judge's), 130 Ctrl-C (the first lets the calls in flight finish, a second "
                                      "aborts at once); stops met at one check rank Ctrl-C, then the error, then the "
                                      "cap, and a later Ctrl-C or error keeps the first stop's status. Without such a "
                                      "stop: 4 planned calls failed (also at --limit) or an error no retry can fix "
                                      "came after the last call had started, else 0 (done, or stopped by --limit). 2: "
                                      "refused before any call, or --out could not take a row during the run (the "
                                      "rows written count; a paid call whose row is missing is sent again by the next "
                                      "run)")
    krun.add_argument("--plan", required=True, metavar="FILE")
    krun.add_argument("--events", required=True, metavar="FILE")
    krun.add_argument("--out", required=True, metavar="FILE",
                      help="results JSONL (resumed if it exists; one backend per file: a mock run writes to its own)")
    krun.add_argument("--mock", action="store_true", help="the simulated models (no key, no cost)")
    krun.add_argument("--max-cost", type=float, metavar="USD",
                      help="stop sending once the summed cost of every row in --out, recipients and judges, reaches "
                           "this (required with the real models; the pilot asks for 3.5); the calls in flight finish "
                           "with their judge calls, so a run can end above it: by up to --concurrency items, at "
                           "max_tokens and the default of 8 up to about 1.64 USD (eight Fable items: the items in "
                           "flight can belong to a later block of the plan), 0.65 if the cap is reached before the "
                           "first Neon call has started and 0.81 before the first Fable call has "
                           "(docs/DESIGN_keys.md section 9); a run it stops exits with status 3")
    krun.add_argument("--limit", type=int, metavar="N", help="send at most N calls")
    krun.add_argument("--concurrency", type=int, default=8)
    krun.add_argument("--judges", nargs="*", default=[], metavar="KEY",
                      help="judge models for answers the rules cannot grade, for every rule grade incorrect or "
                           "abstain of an answer that names the expected answer in any form (every arm and set), and "
                           "for every labelled answer that names the candidate in any form, whatever the rules would "
                           "decide: the right name of a correct full answer (dark, both_right) and the donor's name on "
                           "donor_wrong and control (alone and full, whatever the grade); the rules decide alone only "
                           "a grade correct and the answers that do not name the expected answer, and for adoption "
                           "the answers that do not name the donor's name. A failed or tied vote, or a reply without "
                           "a verdict, leaves the answer undecided, never to the rules (the run prints how many are), "
                           "and a later run with --judges asks again; stored rows that are undecided, or that the "
                           "rules graded or decided and the judges decide now, are graded again first, without a new "
                           "call (default: none: the rules decide, what they leave open stays undecided, and the "
                           "report gives no verdict that the open offers, ungraded answers that name the expected "
                           "answer, could overturn)")
    krun.add_argument("--resend-stale", action="store_true",
                      help="send again the calls whose stored rows answered another request (the plan or the model "
                           "settings changed); the newest row counts (default: stop)")
    _add_config(krun)
    krun.set_defaults(func=_cmd_bench_keys_run)
    krep = keys_sub.add_parser("report", help="rates, paired tests and adoption rates from plan and results (offline)")
    krep.add_argument("--plan", required=True, metavar="FILE")
    krep.add_argument("--results", required=True, metavar="FILE")
    krep.add_argument("--json", metavar="OUT", help="also write the report as JSON")
    krep.add_argument("--md", metavar="OUT", help="also write the Markdown report to a file")
    krep.set_defaults(func=_cmd_bench_keys_report)

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
