"""JSONL storage for benchmark records (events and probe results).

One JSON object per line, UTF-8, non-ASCII kept as is. :func:`write_jsonl` replaces a file atomically;
:func:`append_jsonl` adds one line and flushes it, so a long probe run can resume after an interruption. A crash
during an append can leave a truncated last line without a newline; :func:`read_jsonl` skips exactly that case
(with a warning) and raises on any other malformed line.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

from cutoffmix.types import Event, ProbeResult

log = logging.getLogger(__name__)

PathLike = str | os.PathLike[str]

# Event.origin -> attribution required when redistributing anything derived from that source
ORIGIN_ATTRIBUTION: dict[str, str] = {
    "kc": "apoorvumang/knowledge-cutoff (CC BY 4.0)",
    "wikidata": "Wikidata (CC0 1.0)",
    "wikipedia": "Wikipedia (CC BY-SA 4.0)",
}


def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """Read all JSON objects from ``path``; blank lines are ignored.

    Raises ``ValueError`` naming the file and line for malformed JSON or a non-object line, except for a truncated
    final line without a trailing newline (an interrupted append), which is skipped with a warning.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    lines = text.split("\n")
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            if lineno == len(lines) and not text.endswith("\n"):
                log.warning("%s: skipping truncated last line %d (interrupted append)", p, lineno)
                continue
            raise ValueError(f"{p}:{lineno}: invalid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise ValueError(f"{p}:{lineno}: expected a JSON object, got {type(obj).__name__}")
        rows.append(obj)
    return rows


def _dumps(row: Mapping[str, Any]) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def write_jsonl(path: PathLike, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write ``rows`` to ``path`` (parent directories are created); the file is replaced atomically, keeping an
    existing file's permissions (new files get 0644)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = p.stat().st_mode & 0o777 if p.exists() else 0o644  # mkstemp would leave 0600
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(_dumps(row) + "\n")
        os.chmod(tmp, mode)
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_jsonl(path: PathLike, row: Mapping[str, Any]) -> None:
    """Append one object as a line to ``path`` (created with its parent directories if missing) and flush it."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(_dumps(row) + "\n")
        f.flush()


def load_events(path: PathLike) -> list[Event]:
    """Load :class:`~cutoffmix.types.Event` records; unknown keys are ignored, duplicate ids raise ``ValueError``."""
    events = [Event.from_dict(d) for d in read_jsonl(path)]
    seen: set[str] = set()
    for e in events:
        if e.id in seen:
            raise ValueError(f"{path}: duplicate event id {e.id!r}")
        seen.add(e.id)
    return events


def save_events(path: PathLike, events: Iterable[Event]) -> None:
    """Write events as JSONL (atomic replace)."""
    write_jsonl(path, (e.to_dict() for e in events))


def load_results(path: PathLike) -> list[ProbeResult]:
    """Load :class:`~cutoffmix.types.ProbeResult` records; unknown keys are ignored."""
    return [ProbeResult.from_dict(d) for d in read_jsonl(path)]


def save_results(path: PathLike, results: Iterable[ProbeResult]) -> None:
    """Write probe results as JSONL (atomic replace)."""
    write_jsonl(path, (r.to_dict() for r in results))
