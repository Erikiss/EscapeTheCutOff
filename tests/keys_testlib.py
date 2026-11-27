"""Shared fixtures of the key-experiment tests (docs/DESIGN_keys.md): paths of the stored paper data, the three
model keys, and small events and stored probe rows for fixture plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cutoffmix.types import Event

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events" / "papers_events.jsonl"
# the graded probe results of 25 November 2026 (data/results/README.md), the input of every real-data plan
RESULTS = ROOT / "data" / "results" / "papers_probes_2026-11-25.jsonl"
FABLE, SOL, NEON = "claude-fable-5.2", "gpt-6.2-sol", "gemini-4-neon"
TODAY = "2026-11-27"


def forms(name: str) -> list[str]:
    given, surname = name.rsplit(" ", 1)
    return [name, f"{surname}, {given}", f"{given[0]}. {surname}", f"{surname} {given[0]}."]


def paper(i: int, name: str = "", *, subject: str | None = None, control: bool = False,
          rejects: tuple[str, ...] = (), aliases: list[str] | None = None) -> Event:
    subject = subject or f"Widget Paper {i}: Scaling Sparse Widgets Further"
    return Event(id=f"paper:{i}", date="2026-03-02", month="2026-03", topic="ai_ml",
                 category="fake_paper" if control else "paper", subject=subject, fact="",
                 question_current=f"Who is the first author of the arXiv paper titled '{subject}'?",
                 expected_current="" if control else name,
                 aliases=list(aliases if aliases is not None else ([] if control else forms(name))),
                 reject_aliases=list(rejects), control=control)


def result(event_id: str, model: str, grade: str, response: str = "", probe: str = "current_d",
           usage: dict | None = None) -> dict[str, Any]:
    return {"event_id": event_id, "model": model, "probe": probe, "response": response, "grade": grade,
            "hedged": False, "grader": "rule", "latency_s": 1.0, "usage": usage, "error": None}


NAMES = ["Alba Quist", "Bodo Rinne", "Cora Sell", "Dario Tamm", "Edda Ulm", "Femke Vos", "Gil Wendt"]
# one co-author per paper: every key carries a cue (the co-author's name) and no two keys are the same note, so each
# item finds a wrong key that differs from its key (design section 3)
COAUTHORS = ["Yara Okafor", "Ines Brandt", "Ugo Ferrante", "Mira Kovac", "Lene Haugen", "Omar Saleh"]


def fixture_data():
    """Events ``paper:1`` to ``paper:7`` (real) and ``paper:20`` (fake) and stored rows of two models: ``a`` answers
    papers 1 to 6 correctly (each note names another co-author), ``b`` misses papers 1 to 4 and knows 5 and 6; ``a``
    gives a wrong name for paper 7 and invents an author for the fake paper. With ``pairs=[("a", "b")]`` and
    ``min_items=3``: four ``dark`` items, one ``donor_wrong``, a ``both_right`` sample of papers 5 and 6, one
    ``control``."""
    events = [paper(i + 1, name) for i, name in enumerate(NAMES)]
    events.append(paper(20, control=True, subject="Quantum Llamas: Teaching Alpacas to Tunnel"))
    results = []
    for i, name in enumerate(NAMES[:6]):
        results.append(result(f"paper:{i + 1}", "a", "correct", f"The first author is {name}, with {COAUTHORS[i]}."))
    for i in range(4):
        results.append(result(f"paper:{i + 1}", "b", "abstain", "I don't know."))
    for i in (4, 5):
        results.append(result(f"paper:{i + 1}", "b", "correct", f"It is {NAMES[i]}."))
    results.append(result("paper:7", "a", "incorrect", "My best guess is Wrong Person."))
    results.append(result("paper:7", "b", "abstain", "No idea."))
    results.append(result("paper:20", "a", "incorrect", "I believe the first author is Made Upname."))
    results.append(result("paper:20", "b", "correct", "I cannot find that paper."))
    results += [result(f"paper:{i}", "b", "correct" if i % 2 else "incorrect", "A", probe="mcq") for i in range(1, 5)]
    return events, results
