import json

import pytest

from cutoffmix.router.gate import TimeSensitivityGate
from cutoffmix.router.topics import LinearTopicClassifier
from cutoffmix.router.train import (
    build_gate_corpus,
    build_training_corpus,
    event_texts,
    main,
    read_events,
    synthetic_time_insensitive,
    synthetic_time_sensitive,
    synthetic_topic_corpus,
    train_gate,
    train_topic_classifier,
)
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.types import Event


def _event(i, topic="sports", control=False, **kw):
    base = dict(id=f"e{i}", date="2026-05-10", month="2026-05", topic=topic, category="sports_result",
                subject="Club", fact=f"Club {i} won the cup final.", question_current=f"Who holds cup {i} now?",
                question_recall=f"Who won cup {i} in May 2026?", control=control)
    base.update(kw)
    return Event(**base)


def _write_events(path, events):
    path.write_text("".join(json.dumps(e.to_dict()) + "\n" for e in events) + "\n", encoding="utf-8")
    return path


def test_synthetic_corpora_cover_topics_and_languages():
    texts, labels = synthetic_topic_corpus(per_topic=6, seed=1)
    assert len(texts) == len(labels)
    assert set(labels) == set(TOPIC_IDS)
    assert texts == synthetic_topic_corpus(per_topic=6, seed=1)[0]  # deterministic
    neg = synthetic_time_insensitive(50, seed=2)
    pos = synthetic_time_sensitive(50, seed=2)
    assert len(neg) == len(pos) == 50
    assert any("Zahl" in t or "Zug" in t or "Kisten" in t or "Rest" in t or "Ritter" in t for t in neg)
    assert any(t.startswith("Wer ") or t.startswith("Was ") or t.startswith("Seit ") for t in pos)


def test_event_texts_and_read_events(tmp_path):
    e = _event(1, mcq_question="Which club won cup 1?")
    assert event_texts(e) == ["Club 1 won the cup final.", "Who holds cup 1 now?", "Who won cup 1 in May 2026?",
                              "Which club won cup 1?"]
    path = _write_events(tmp_path / "events.jsonl", [e, _event(2)])
    assert [x.id for x in read_events(path)] == ["e1", "e2"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        read_events(bad)


def test_build_training_corpus_from_events_and_files(tmp_path):
    events = [_event(1), _event(2, topic="government", fact="The minister resigned."), _event(3, topic="unknown")]
    path = _write_events(tmp_path / "events.jsonl", events)
    texts, labels = build_training_corpus(path, synthetic_per_topic=2)
    pairs = set(zip(texts, labels))
    assert ("Club 1 won the cup final.", "sports") in pairs
    assert ("The minister resigned.", "government") in pairs
    assert not any("cup 3" in t for t in texts)  # unknown topic skipped
    assert set(labels) == set(TOPIC_IDS)  # taxonomy seeds/synthetic sentences cover every topic
    assert len(pairs) == len(texts)  # no duplicates
    texts2, labels2 = build_training_corpus([events[0], path], synthetic_per_topic=2)
    assert len(texts2) == len(texts)  # the duplicate event is dropped


def test_build_gate_corpus_balances_classes_and_skips_controls():
    events = [_event(i) for i in range(40)] + [_event(99, control=True, fact="Control person is alive.")]
    texts, labels = build_gate_corpus(events, n_synthetic=20, seed=0)
    assert "Club 1 won the cup final." in texts and "Control person is alive." not in texts
    assert labels.count(0) == labels.count(1)


def test_train_helpers_write_models(tmp_path):
    texts, labels = synthetic_topic_corpus(per_topic=8)
    clf = train_topic_classifier(texts, labels, tmp_path / "m" / "topic.npz", dim=4096, epochs=3)
    assert isinstance(LinearTopicClassifier.load(tmp_path / "m" / "topic.npz"), LinearTopicClassifier)
    assert clf.topics == list(TOPIC_IDS)
    g_texts, g_labels = build_gate_corpus([], n_synthetic=40)
    gate = train_gate(g_texts, g_labels, tmp_path / "m" / "gate.npz", dim=4096, epochs=3)
    assert TimeSensitivityGate.load(tmp_path / "m" / "gate.npz").score("x") == pytest.approx(gate.score("x"))


def test_main_cli(tmp_path, capsys):
    events = _write_events(tmp_path / "events.jsonl", [_event(i) for i in range(10)])
    gate_rows = tmp_path / "gate.jsonl"
    gate_rows.write_text('{"text": "Who is the coach now?", "label": 1}\n{"text": "2 + 2 = 4", "label": 0}\n',
                         encoding="utf-8")
    out = tmp_path / "router_models"
    code = main(["--events", str(events), "--gate-data", str(gate_rows), "--out-dir", str(out), "--dim", "4096",
                 "--epochs", "2", "--synthetic-per-topic", "5"])
    assert code == 0
    assert (out / "topic_linear.npz").exists() and (out / "gate_linear.npz").exists()
    printed = capsys.readouterr().out
    assert "held-out accuracy" in printed and "time-sensitivity gate" in printed

    only_topic = tmp_path / "only_topic"
    assert main(["--out-dir", str(only_topic), "--dim", "4096", "--epochs", "1", "--no-gate"]) == 0
    assert (only_topic / "topic_linear.npz").exists() and not (only_topic / "gate_linear.npz").exists()
