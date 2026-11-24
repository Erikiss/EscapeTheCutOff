"""Tests for cutoffmix.bench.io (JSONL storage of events and probe results)."""

from __future__ import annotations

import json
import logging

import pytest

from cutoffmix.bench.io import (
    append_jsonl,
    load_events,
    load_results,
    read_jsonl,
    save_events,
    save_results,
    write_jsonl,
)
from cutoffmix.types import Event, ProbeResult


def _event(eid: str = "e1", **kw) -> Event:
    base = dict(id=eid, date="2026-03-03", month="2026-03", topic="government", category="office_change",
                subject="Ülle Madise", fact="Ülle Madise became President of Estonia.",
                mcq_choices=["A) x", "B) y", "C) z", "D) w"], aliases=["Ülle Madise"])
    base.update(kw)
    return Event(**base)


def test_events_roundtrip_and_parent_dirs(tmp_path):
    path = tmp_path / "deep" / "dir" / "events.jsonl"
    events = [_event("e1"), _event("e2", control=True, reject_aliases=["Alar Karis"])]
    save_events(path, events)
    assert load_events(path) == events
    assert "Ülle" in path.read_text(encoding="utf-8")  # non-ASCII kept as is


def test_results_roundtrip(tmp_path):
    path = tmp_path / "results.jsonl"
    results = [ProbeResult("e1", "gpt-6.2-sol", "current", "Ülle Madise.", "correct", usage={"total_tokens": 9}),
               ProbeResult("e1", "gemini-4-neon", "mcq", "B", "outdated", hedged=True, grader="mcq")]
    save_results(path, results)
    assert load_results(path) == results


def test_append_and_read_skip_blank_lines(tmp_path):
    path = tmp_path / "sub" / "rows.jsonl"
    append_jsonl(path, {"a": 1})
    with path.open("a", encoding="utf-8") as f:
        f.write("\n   \n")
    append_jsonl(path, {"a": 2})
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_truncated_last_line_is_skipped_with_warning(tmp_path, caplog):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n{"a": 3, "b', encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
    assert "truncated" in caplog.text


def test_invalid_line_in_the_middle_raises_with_location(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"rows\.jsonl:2"):
        read_jsonl(path)


def test_invalid_last_line_with_newline_raises(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": \n', encoding="utf-8")
    with pytest.raises(ValueError, match=":2"):
        read_jsonl(path)


def test_non_object_line_raises(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_jsonl(path)


def test_load_events_rejects_duplicate_ids_and_ignores_unknown_keys(tmp_path):
    path = tmp_path / "events.jsonl"
    row = {**_event().to_dict(), "raw": {"extra": 1}, "question_direct": "kc field"}
    write_jsonl(path, [row])
    (event,) = load_events(path)
    assert event == _event()
    write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="duplicate event id"):
        load_events(path)


def test_write_is_atomic_on_failure(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"keep": True}])

    def rows():
        yield {"new": 1}
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        write_jsonl(path, rows())
    assert read_jsonl(path) == [{"keep": True}]
    assert [p.name for p in tmp_path.iterdir()] == ["rows.jsonl"]


def test_write_keeps_permissions(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"a": 1}])
    assert path.stat().st_mode & 0o777 == 0o644
    path.chmod(0o664)
    write_jsonl(path, [{"a": 2}])
    assert path.stat().st_mode & 0o777 == 0o664


def test_one_object_per_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"text": "line\nbreak"}, {"n": 2}])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[0]) == {"text": "line\nbreak"}
