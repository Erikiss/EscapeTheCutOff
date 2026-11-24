import subprocess
import sys
from pathlib import Path

import pytest

from cutoffmix.router import train_gpu
from cutoffmix.router.train_gpu import (
    DEFAULT_MODEL,
    IGNORE_INDEX,
    Example,
    build_examples,
    evaluation_metrics,
    parse_args,
    probe_text,
    split_by_event,
)
from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.types import Event, ProbeResult

REPO = Path(__file__).resolve().parents[1]


def test_module_imports_without_torch():
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        "import cutoffmix.router.train_gpu as m\n"
        "assert callable(m.main) and callable(m.build_examples)\n"
        "print('ok')\n"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"


def _event(eid, topic="ai_ml", control=False):
    return Event(id=eid, date="2026-08-01", month="2026-08", topic=topic, category="release", subject="Model",
                 fact="fact", question_current=f"What is the newest model of lab {eid}?",
                 question_recall=f"Which model did lab {eid} release in August 2026?",
                 mcq_question=f"Which is lab {eid}'s newest model?", control=control)


def _res(eid, model, probe, grade):
    return ProbeResult(event_id=eid, model=model, probe=probe, response="...", grade=grade)


def test_build_examples_targets():
    events = [_event("e1"), _event("e2", topic="sports"), _event("c1", control=True), _event("u1", topic="nope")]
    results = [
        # e1 / question_current: current_d preferred over current for fable; sol only has `current`
        _res("e1", "fable", "current_d", "correct"),
        _res("e1", "fable", "current_d", "outdated"),
        _res("e1", "fable", "current", "incorrect"),
        _res("e1", "sol", "current", "correct"),
        _res("e1", "sol", "current_d", "ungraded"),  # ignored
        _res("e1", "neon", "recall", "correct"),
        _res("e1", "neon", "current_d", "abstain"),
        _res("e2", "sol", "recall", "outdated"),
        _res("c1", "sol", "current_d", "correct"),
        _res("u1", "fable", "current_d", "correct"),
    ]
    examples, models = build_examples(events, results, negatives=["2 + 2 = 4"])
    assert models == ["fable", "neon", "sol"]
    by_text = {e.text: e for e in examples}
    assert not any("c1" in e.event_id for e in examples)  # controls skipped by default

    cur = by_text["What is the newest model of lab e1?"]
    assert cur.event_id == "e1" and cur.topic == TOPIC_IDS.index("ai_ml") and cur.time_sensitive == 1.0
    assert cur.currentness == [0.5, 0.0, 1.0]  # fable 1/2 on current_d, neon 0/1 (abstain), sol 1/1 on current

    rec = by_text["Which model did lab e1 release in August 2026?"]
    assert rec.currentness == [None, 1.0, None]

    e2 = by_text["What is the newest model of lab e2?"]
    assert e2.topic == TOPIC_IDS.index("sports") and e2.currentness == [None, None, None]
    assert by_text["Which model did lab e2 release in August 2026?"].currentness == [None, None, 0.0]

    unknown = by_text["What is the newest model of lab u1?"]
    assert unknown.topic == IGNORE_INDEX and unknown.currentness == [1.0, None, None]

    neg = by_text["2 + 2 = 4"]
    assert (neg.topic, neg.time_sensitive, neg.currentness) == (IGNORE_INDEX, 0.0, [None, None, None])
    assert neg.event_id.startswith("negative:")
    assert len(examples) == 3 * 2 + 1  # two texts per non-control event + one negative


def test_build_examples_options():
    events = [_event("e1"), _event("c1", control=True)]
    results = [_res("e1", "sol", "mcq", "correct"), _res("c1", "sol", "current", "correct")]
    examples, models = build_examples(events, results, models=["sol", "fable"], probes=["mcq", "current"],
                                      include_controls=True)
    assert models == ["sol", "fable"]
    mcq = [e for e in examples if e.text == "Which is lab e1's newest model?"][0]
    assert mcq.currentness == [1.0, None]
    assert any(e.event_id == "c1" and e.currentness == [1.0, None] for e in examples)
    with pytest.raises(ValueError, match="unknown probes"):
        build_examples(events, results, probes=["bogus"])
    with pytest.raises(ValueError):
        build_examples(events, [])
    assert probe_text(events[0], "current_d") == "What is the newest model of lab e1?"
    with pytest.raises(ValueError):
        probe_text(events[0], "nope")


def test_split_by_event_keeps_events_together():
    examples = [Example(f"t{i}-{j}", f"e{i}", 0, 1.0, [None]) for i in range(200) for j in range(3)]
    train, held = split_by_event(examples, 0.2, seed=0)
    assert len(train) + len(held) == len(examples)
    assert {e.event_id for e in train}.isdisjoint({e.event_id for e in held})
    assert 0.1 < len({e.event_id for e in held}) / 200 < 0.3
    assert split_by_event(examples, 0.2, seed=0) == (train, held)
    assert split_by_event(examples, 0.0)[1] == []
    with pytest.raises(ValueError):
        split_by_event(examples, 1.0)


def test_evaluation_metrics():
    m = evaluation_metrics(
        topic_pred=[0, 1, 2], topic_true=[0, 2, IGNORE_INDEX],
        time_prob=[0.9, 0.2, 0.7], time_true=[1.0, 0.0, 1.0],
        cur_prob=[[0.8, 0.1], [0.3, 0.6], [0.9, 0.5]], cur_true=[[1.0, None], [0.0, 1.0], [1.0, 0.0]],
        models=["a", "b"],
    )
    assert m["n"] == 3 and m["topic_accuracy"] == 0.5 and m["time_accuracy"] == 1.0 and m["time_auc"] == 1.0
    assert m["currentness"]["a"]["n"] == 3 and m["currentness"]["a"]["auc"] == 1.0
    assert m["currentness"]["a"]["brier"] == pytest.approx((0.04 + 0.09 + 0.01) / 3)
    assert m["currentness"]["b"]["auc"] == 1.0
    empty = evaluation_metrics([], [], [], [], [], [], ["a"])
    assert empty["topic_accuracy"] is None and empty["currentness"]["a"]["brier"] is None


def test_parse_args_defaults_and_budget_check():
    args = parse_args(["--events", "e.jsonl", "--results", "r.jsonl"])
    assert args.model_name == DEFAULT_MODEL and args.budget_min == 55.0 and args.bf16 is True
    assert args.out == "router_models/encoder" and args.probes == list(train_gpu.DEFAULT_PROBES)
    with pytest.raises(SystemExit):
        parse_args(["--events", "e", "--results", "r", "--budget-min", "3", "--reserve-min", "4"])


def test_launch_script():
    script = REPO / "scripts" / "train_router_gpu.sh"
    text = script.read_text(encoding="utf-8")
    assert "torchrun" in text and "--nproc_per_node" in text and "cutoffmix.router.train_gpu" in text
    assert 'BUDGET_MIN="${BUDGET_MIN:-55}"' in text
    assert "train_router_gpu.sh 1" in text and "train_router_gpu.sh 8" in text
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
