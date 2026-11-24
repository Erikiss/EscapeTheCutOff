"""CPU training of the router's learned parts (``cutoffmix router train``).

* :func:`build_training_corpus` -- topic-labelled texts from benchmark events (facts and probe questions) plus the
  taxonomy seeds and a synthetic keyword corpus for topics the events miss.
* :func:`build_gate_corpus` -- time-sensitive positives (event questions/facts, present-state question templates)
  and time-insensitive negatives (math, logic, code and timeless-fact snippets, EN/DE).
* :func:`train_topic_classifier` / :func:`train_gate` -- fit and save ``.npz`` models.
* :func:`main` -- argparse entry point used by the CLI.

Both models train in seconds to minutes on a laptop CPU (pure numpy).
"""

from __future__ import annotations

import argparse
import json
import random
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from cutoffmix.router.features import DEFAULT_DIM
from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.topics import LinearTopicClassifier
from cutoffmix.taxonomy import TOPIC_IDS, TOPICS
from cutoffmix.types import Event

__all__ = [
    "build_gate_corpus",
    "build_training_corpus",
    "event_texts",
    "main",
    "read_events",
    "synthetic_time_insensitive",
    "synthetic_time_sensitive",
    "synthetic_topic_corpus",
    "train_gate",
    "train_topic_classifier",
]

PathLike = str | Path


def read_events(path: PathLike) -> list[Event]:
    """Read an events JSONL file (one :class:`~cutoffmix.types.Event` dict per line; blank lines skipped)."""
    events = []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                events.append(Event.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"{path}:{lineno}: not an Event record ({e})") from e
    return events


def _as_events(events_or_paths: Iterable[Event | PathLike] | PathLike) -> list[Event]:
    if isinstance(events_or_paths, (str, Path)):
        return read_events(events_or_paths)
    out: list[Event] = []
    for item in events_or_paths:
        out.extend([item] if isinstance(item, Event) else read_events(item))
    return out


def event_texts(event: Event) -> list[str]:
    """The texts an event contributes: fact, current/recall questions and the MCQ stem (non-empty, unique)."""
    texts = [event.fact, event.question_current, event.question_recall, event.mcq_question]
    return list(dict.fromkeys(t.strip() for t in texts if t and t.strip()))


# --------------------------------------------------------------------------------------------------------------------
# Synthetic corpora
# --------------------------------------------------------------------------------------------------------------------

_TOPIC_TEMPLATES = (
    "The notes so far mention {a} and {b}.",
    "Next we check the {a} question; it depends on {b}.",
    "Reports about {a}, {b} and {c} dominate this part of the problem.",
    "Evidence: the source links {a} to {b}.",
    "Die bisherigen Notizen erwähnen {a} und {b}.",
    "Als Nächstes prüfen wir {a}; das hängt von {b} ab.",
    "Laut Quelle steht {a} im Zusammenhang mit {b} und {c}.",
)


def synthetic_topic_corpus(per_topic: int = 40, seed: int = 0) -> tuple[list[str], list[str]]:
    """Topic-labelled texts from the taxonomy: each topic's seeds, its name + description, and ``per_topic``
    template sentences filled with random keywords of that topic (EN and DE templates)."""
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[str] = []
    for topic in TOPICS:
        keywords = [k.strip() for k in topic.keywords if k.strip()]
        base = [*topic.seeds, f"{topic.name}: {topic.description}"]
        generated = [
            rng.choice(_TOPIC_TEMPLATES).format(**dict(zip("abc", rng.sample(keywords, 3))))
            for _ in range(per_topic)
        ]
        texts.extend(base + generated)
        labels.extend([topic.id] * (len(base) + len(generated)))
    return texts, labels


def synthetic_time_insensitive(n: int = 300, seed: int = 0) -> list[str]:
    """``n`` math / logic / code / timeless-fact snippets (EN and DE) whose answer cannot go stale."""
    rng = random.Random(seed)

    def make() -> str:
        a, b, c = rng.randint(2, 9), rng.randint(1, 30), rng.randint(31, 99)
        n_, k, x, y = rng.randint(3, 60), rng.randint(2, 12), rng.randint(50, 999), rng.randint(3, 17)
        p = rng.choice((7, 11, 13, 17, 19, 23, 29, 31, 37, 41))
        d, t = rng.randint(60, 600), rng.randint(2, 8)
        return rng.choice((
            f"If {a}x + {b} = {c}, then x = ({c} - {b}) / {a}.",
            f"The sum of the first {n_} positive integers is {n_}*({n_}+1)/2 = {n_ * (n_ + 1) // 2}.",
            f"A train covers {d} km in {t} hours, so its average speed is {d}/{t} km/h.",
            f"Since {p} is prime, its only divisors are 1 and {p}.",
            f"Each of the {n_} boxes holds {k} balls, so there are {n_ * k} balls in total.",
            f"The remainder of {x} divided by {y} is {x % y}.",
            "Knights always tell the truth and knaves always lie, so if A calls B a knave they differ in type.",
            "The probability of rolling two sixes with two fair dice is 1/36.",
            "def area(r): return 3.14159 * r ** 2 computes the area of a circle of radius r.",
            "Water boils at 100 degrees Celsius at sea level.",
            f"A triangle with sides {a}, {a + 1} and {a + 2} satisfies the triangle inequality.",
            "Assume the statement is false; then we reach a contradiction, so it must be true.",
            f"Wenn {a}x + {b} = {c} gilt, dann ist x = ({c} - {b}) / {a}.",
            f"Die Summe der ersten {n_} natürlichen Zahlen beträgt {n_ * (n_ + 1) // 2}.",
            f"Ein Zug fährt {d} km in {t} Stunden, also beträgt die Geschwindigkeit {d}/{t} km/h.",
            f"Da {p} eine Primzahl ist, hat sie nur die Teiler 1 und {p}.",
            f"Jede der {n_} Kisten enthält {k} Kugeln, also sind es insgesamt {n_ * k} Kugeln.",
            f"Der Rest von {x} geteilt durch {y} ist {x % y}.",
            "Ritter sagen immer die Wahrheit und Schurken lügen immer.",
            "Wasser siedet auf Meereshöhe bei 100 Grad Celsius.",
        ))

    return [make() for _ in range(n)]


_ROLES_EN = ("CEO", "president", "prime minister", "head coach", "chancellor", "foreign minister", "chair",
             "world champion", "governor")
_ROLES_DE = ("Vorstandschef", "Präsident", "Bundeskanzler", "Trainer", "Ministerpräsident", "Außenminister",
             "Weltmeister")
_ENTITIES = ("France", "the company", "the club", "Brazil", "Japan", "the central bank", "the league", "Germany",
             "the party", "the lab")
_SOFTWARE = ("Python", "the Linux kernel", "iOS", "the model family", "React", "Android")


def synthetic_time_sensitive(n: int = 300, seed: int = 0) -> list[str]:
    """``n`` present-state questions and notes (EN and DE) whose answer changes over time."""
    rng = random.Random(seed)

    def make() -> str:
        role, role_de, ent = rng.choice(_ROLES_EN), rng.choice(_ROLES_DE), rng.choice(_ENTITIES)
        year = rng.randint(2024, 2026)
        return rng.choice((
            f"Who is the current {role} of {ent}?",
            f"Who is the {role} of {ent} right now?",
            f"What is the latest version of {rng.choice(_SOFTWARE)}?",
            f"Is the former {role} of {ent} still alive?",
            "Which team won the most recent final of the league?",
            f"As of {year}, the {role} of {ent} had just been replaced.",
            f"The notes must state who currently serves as {role} of {ent}.",
            f"Wer ist aktuell {role_de} von {ent}?",
            f"Wer ist derzeit {role_de}?",
            f"Was ist die neueste Version von {rng.choice(_SOFTWARE)}?",
            f"Seit {year} amtiert ein neuer {role_de}.",
        ))

    return [make() for _ in range(n)]


# --------------------------------------------------------------------------------------------------------------------
# Corpora from events
# --------------------------------------------------------------------------------------------------------------------


def build_training_corpus(
    events_paths: Iterable[Event | PathLike] | PathLike, *, synthetic_per_topic: int = 20, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Topic-labelled ``(texts, labels)``: event facts and probe questions (topic = ``event.topic``; events with a
    topic outside the taxonomy are skipped), the taxonomy seeds, and ``synthetic_per_topic`` keyword sentences per
    topic. Duplicate ``(text, label)`` pairs are dropped."""
    known = set(TOPIC_IDS)
    pairs: dict[tuple[str, str], None] = {}
    for event in _as_events(events_paths):
        if event.topic in known:
            for text in event_texts(event):
                pairs[(text, event.topic)] = None
    syn_texts, syn_labels = synthetic_topic_corpus(synthetic_per_topic, seed)
    for text, label in zip(syn_texts, syn_labels):
        pairs[(text, label)] = None
    return [t for t, _ in pairs], [label for _, label in pairs]


def build_gate_corpus(
    events_paths: Iterable[Event | PathLike] | PathLike, *, n_synthetic: int = 300, seed: int = 0
) -> tuple[list[str], list[int]]:
    """Gate ``(texts, labels)``: label 1 for non-control event questions/facts and synthetic present-state
    questions, label 0 for synthetic math/logic/code/timeless snippets (``n_synthetic`` of each kind plus as many
    negatives as there are event positives, to keep the classes balanced)."""
    positives: dict[str, None] = {}
    for event in _as_events(events_paths):
        if not event.control:
            positives.update(dict.fromkeys(event_texts(event)))
    positives.update(dict.fromkeys(synthetic_time_sensitive(n_synthetic, seed)))
    negatives = synthetic_time_insensitive(max(n_synthetic, len(positives)), seed + 1)
    texts = list(positives) + negatives
    return texts, [1] * len(positives) + [0] * len(negatives)


# --------------------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------------------


def train_topic_classifier(
    texts: Sequence[str], labels: Sequence[str], out_path: PathLike, **train_kwargs: Any
) -> LinearTopicClassifier:
    """Fit a :class:`LinearTopicClassifier` (keyword arguments go to its ``train``) and save it to ``out_path``."""
    clf = LinearTopicClassifier.train(texts, labels, **train_kwargs)
    clf.save(out_path)
    return clf


def train_gate(
    texts: Sequence[str], labels: Sequence[float | int | bool], out_path: PathLike, **train_kwargs: Any
) -> TimeSensitivityGate:
    """Fit a learned :class:`TimeSensitivityGate` (keyword arguments go to its ``train``) and save it."""
    gate = TimeSensitivityGate.train(texts, labels, **train_kwargs)
    gate.save(out_path)
    return gate


def _holdout(text: str, frac: float) -> bool:
    return zlib.crc32(text.encode("utf-8")) % 10_000 < frac * 10_000


def _read_gate_rows(path: PathLike) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if "text" not in row or "label" not in row:
                    raise ValueError(f"{path}:{lineno}: gate rows need 'text' and 'label'")
                texts.append(str(row["text"]))
                labels.append(1 if float(row["label"]) >= 0.5 else 0)
    return texts, labels


def main(argv: Sequence[str] | None = None) -> int:
    """``cutoffmix router train``: train the linear topic classifier and the learned gate on CPU.

    Writes ``<out-dir>/topic_linear.npz`` and ``<out-dir>/gate_linear.npz`` and prints held-out accuracies
    (a deterministic 10% split by text hash). Use the topic model with ``router.classifier: linear`` and
    ``router.classifier_path: <out-dir>/topic_linear.npz``, the gate with ``router.gate_path:
    <out-dir>/gate_linear.npz`` (blended with the lexical gate).
    """
    ap = argparse.ArgumentParser(prog="cutoffmix router train", description=main.__doc__)
    ap.add_argument("--events", nargs="*", default=[], help="events JSONL files (bench output)")
    ap.add_argument("--gate-data", nargs="*", default=[], help='extra gate JSONL rows {"text": ..., "label": 0|1}')
    ap.add_argument("--out-dir", default="router_models")
    ap.add_argument("--synthetic-per-topic", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--dim", type=int, default=DEFAULT_DIM)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", type=float, default=0.1, help="held-out share for the reported accuracy")
    ap.add_argument("--no-gate", action="store_true", help="train only the topic classifier")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    hyper = {"dim": args.dim, "epochs": args.epochs, "lr": args.lr, "l2": args.l2, "seed": args.seed}

    texts, labels = build_training_corpus(args.events, synthetic_per_topic=args.synthetic_per_topic, seed=args.seed)
    train = [(t, y) for t, y in zip(texts, labels) if not _holdout(t, args.holdout)]
    held = [(t, y) for t, y in zip(texts, labels) if _holdout(t, args.holdout)]
    clf = train_topic_classifier([t for t, _ in train], [y for _, y in train], out_dir / "topic_linear.npz", **hyper)
    acc = sum(clf.predict(t) == y for t, y in held) / max(len(held), 1)
    print(f"topic classifier: {len(train)} train / {len(held)} held-out texts, held-out accuracy {acc:.3f} "
          f"-> {out_dir / 'topic_linear.npz'}")

    if not args.no_gate:
        g_texts, g_labels = build_gate_corpus(args.events, seed=args.seed)
        for path in args.gate_data:
            extra_t, extra_y = _read_gate_rows(path)
            g_texts += extra_t
            g_labels += extra_y
        g_train = [(t, y) for t, y in zip(g_texts, g_labels) if not _holdout(t, args.holdout)]
        g_held = [(t, y) for t, y in zip(g_texts, g_labels) if _holdout(t, args.holdout)]
        gate = train_gate([t for t, _ in g_train], [y for _, y in g_train], out_dir / "gate_linear.npz", **hyper)
        g_acc = sum((gate.score(t) >= 0.5) == bool(y) for t, y in g_held) / max(len(g_held), 1)
        print(f"time-sensitivity gate: {len(g_train)} train / {len(g_held)} held-out texts, held-out accuracy "
              f"{g_acc:.3f} -> {out_dir / 'gate_linear.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
