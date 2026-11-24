"""Optional v2 router: fine-tune a multilingual encoder on probe outcomes (GPU, torchrun DDP, bf16).

A mean-pooled encoder (default ``jhu-clsp/mmBERT-base``, fallback ``answerdotai/ModernBERT-base``) gets three heads:

* topic -- softmax over the taxonomy topics (cross-entropy; unlabelled rows ignored),
* time sensitivity -- one sigmoid (event texts = 1, math/logic/timeless snippets = 0),
* currentness -- one sigmoid per model: P(model answers this event's question correctly), trained on the share of
  correct graded probe answers (masked where a model has no graded answer).

Examples are built in pure Python (:func:`build_examples`, importable without torch); evaluation uses events held out
by event id (:func:`split_by_event`). A hard wall-clock guard stops training ``--reserve-min`` minutes before
``--budget-min`` (all ranks agree via all-reduce), then rank 0 evaluates and saves ``--out`` (encoder weights,
tokenizer, ``heads.pt``, ``router_config.json``, ``metrics.json``).

Single GPU::

    torchrun --standalone --nproc_per_node=1 -m cutoffmix.router.train_gpu --events E.jsonl --results R.jsonl

Eight GPUs: ``--nproc_per_node=8`` (see ``scripts/train_router_gpu.sh``). torch and transformers are imported inside
:func:`main` only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from cutoffmix.taxonomy import TOPIC_IDS
from cutoffmix.types import Event, ProbeResult

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROBES",
    "FALLBACK_MODEL",
    "IGNORE_INDEX",
    "Example",
    "build_examples",
    "evaluation_metrics",
    "main",
    "probe_text",
    "read_results",
    "split_by_event",
]

DEFAULT_MODEL = "jhu-clsp/mmBERT-base"
FALLBACK_MODEL = "answerdotai/ModernBERT-base"
DEFAULT_PROBES: tuple[str, ...] = ("current_d", "current", "recall")
IGNORE_INDEX = -100  # topic label of rows without a topic (torch cross-entropy convention)

_PROBE_FIELD = {"current_d": "question_current", "current": "question_current", "recall": "question_recall",
                "mcq": "mcq_question"}


# --------------------------------------------------------------------------------------------------------------------
# Data preparation (pure Python)
# --------------------------------------------------------------------------------------------------------------------


@dataclass
class Example:
    """One training row."""

    text: str
    event_id: str
    topic: int  # index into the topic list, IGNORE_INDEX if unlabelled
    time_sensitive: float  # 1.0 = answer depends on recent world knowledge
    currentness: list[float | None]  # per model: share of graded answers that were correct; None = no label


def probe_text(event: Event, probe: str) -> str:
    """The question text a probe kind asks about ``event`` (``""`` if the event has none)."""
    try:
        field = _PROBE_FIELD[probe]
    except KeyError:
        raise ValueError(f"unknown probe {probe!r}; expected one of {', '.join(_PROBE_FIELD)}") from None
    return str(getattr(event, field) or "").strip()


def build_examples(
    events: Iterable[Event],
    results: Iterable[ProbeResult],
    *,
    models: Sequence[str] | None = None,
    probes: Sequence[str] = DEFAULT_PROBES,
    topics: Sequence[str] = TOPIC_IDS,
    negatives: Sequence[str] = (),
    include_controls: bool = False,
) -> tuple[list[Example], list[str]]:
    """Build ``(examples, models)`` from events and graded probe results.

    One example per distinct question text of an event (``current`` and ``current_d`` share ``question_current``).
    Its currentness target for model ``m`` is ``correct / graded`` over ``m``'s results for that event and the
    first probe in ``probes`` (preference order) that maps to the text and has a graded result; ``ungraded``
    results are ignored. ``models`` defaults to the sorted models found in ``results``. Control events are skipped
    unless ``include_controls``. Each of ``negatives`` becomes a time-insensitive example without topic or
    currentness labels.
    """
    unknown = [p for p in probes if p not in _PROBE_FIELD]
    if unknown:
        raise ValueError(f"unknown probes {unknown}; expected some of {', '.join(_PROBE_FIELD)}")
    results = list(results)
    model_list = list(models) if models is not None else sorted({r.model for r in results})
    if not model_list:
        raise ValueError("no models: pass models= or results with graded answers")
    model_pos = {m: i for i, m in enumerate(model_list)}
    topic_pos = {t: i for i, t in enumerate(topics)}
    tally: dict[tuple[str, str, str], list[int]] = {}
    for r in results:
        if r.grade == "ungraded" or r.model not in model_pos:
            continue
        slot = tally.setdefault((r.event_id, r.probe, r.model), [0, 0])
        slot[0] += r.grade == "correct"
        slot[1] += 1

    examples: list[Example] = []
    for event in events:
        if event.control and not include_controls:
            continue
        by_text: dict[str, list[str]] = {}
        for p in probes:
            text = probe_text(event, p)
            if text:
                by_text.setdefault(text, []).append(p)
        for text, text_probes in by_text.items():
            currentness: list[float | None] = []
            for m in model_list:
                share = None
                for p in text_probes:
                    counts = tally.get((event.id, p, m))
                    if counts:
                        share = counts[0] / counts[1]
                        break
                currentness.append(share)
            examples.append(Example(text, event.id, topic_pos.get(event.topic, IGNORE_INDEX), 1.0, currentness))
    for i, text in enumerate(negatives):
        examples.append(Example(text, f"negative:{i}", IGNORE_INDEX, 0.0, [None] * len(model_list)))
    return examples, model_list


def split_by_event(examples: Sequence[Example], eval_frac: float, seed: int = 0) -> tuple[list[Example], list[Example]]:
    """Deterministic ``(train, eval)`` split by event id: all texts of one event land on the same side."""
    if not 0.0 <= eval_frac < 1.0:
        raise ValueError(f"eval_frac must lie in [0, 1), got {eval_frac}")
    cut = eval_frac * 10_000
    train, held = [], []
    for ex in examples:
        (held if zlib.crc32(f"{seed}:{ex.event_id}".encode()) % 10_000 < cut else train).append(ex)
    return train, held


def read_results(path: str | Path) -> list[ProbeResult]:
    """Read a probe-results JSONL file (one :class:`~cutoffmix.types.ProbeResult` dict per line)."""
    with Path(path).open(encoding="utf-8") as f:
        return [ProbeResult.from_dict(json.loads(line)) for line in f if line.strip()]


def _auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Area under the ROC curve (Mann-Whitney with average ranks); None if one class is missing."""
    pos = sum(labels)
    neg = len(labels) - pos
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, y in zip(ranks, labels) if y)
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def evaluation_metrics(
    topic_pred: Sequence[int], topic_true: Sequence[int], time_prob: Sequence[float], time_true: Sequence[float],
    cur_prob: Sequence[Sequence[float]], cur_true: Sequence[Sequence[float | None]], models: Sequence[str],
) -> dict[str, Any]:
    """Held-out metrics: topic accuracy, time-sensitivity accuracy/AUC, per-model currentness Brier score/AUC."""
    labelled = [(p, t) for p, t in zip(topic_pred, topic_true) if t != IGNORE_INDEX]
    metrics: dict[str, Any] = {
        "n": len(time_true),
        "topic_accuracy": sum(p == t for p, t in labelled) / len(labelled) if labelled else None,
        "time_accuracy": sum((p >= 0.5) == (t >= 0.5) for p, t in zip(time_prob, time_true)) / len(time_true)
        if time_true else None,
        "time_auc": _auc(time_prob, [int(t >= 0.5) for t in time_true]),
        "currentness": {},
    }
    for mi, m in enumerate(models):
        pairs = [(row_p[mi], row_t[mi]) for row_p, row_t in zip(cur_prob, cur_true) if row_t[mi] is not None]
        metrics["currentness"][m] = {
            "n": len(pairs),
            "brier": sum((p - t) ** 2 for p, t in pairs) / len(pairs) if pairs else None,
            "auc": _auc([p for p, _ in pairs], [int(t >= 0.5) for _, t in pairs]),
        }
    return metrics


# --------------------------------------------------------------------------------------------------------------------
# Training (torch / transformers imported lazily)
# --------------------------------------------------------------------------------------------------------------------


class _Collator:
    """Tokenise a batch of examples into tensors (module level so DataLoader workers can pickle it)."""

    def __init__(self, tokenizer: Any, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: list[Example]) -> tuple[Any, ...]:
        import torch

        enc = self.tokenizer([e.text for e in batch], padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        cur = [[0.0 if c is None else c for c in e.currentness] for e in batch]
        mask = [[0.0 if c is None else 1.0 for c in e.currentness] for e in batch]
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor([e.topic for e in batch], dtype=torch.long),
            torch.tensor([e.time_sensitive for e in batch], dtype=torch.float32),
            torch.tensor(cur, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
        )


def _build_model(torch: Any, encoder: Any, n_topics: int, n_models: int, dropout: float) -> Any:
    nn = torch.nn

    class RouterEncoder(nn.Module):
        """Encoder + mean pooling + topic / time-sensitivity / currentness heads."""

        def __init__(self) -> None:
            super().__init__()
            hidden = encoder.config.hidden_size
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.topic = nn.Linear(hidden, n_topics)
            self.time = nn.Linear(hidden, 1)
            self.current = nn.Linear(hidden, n_models)

        def forward(self, input_ids: Any, attention_mask: Any) -> tuple[Any, Any, Any]:
            states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            weights = attention_mask.unsqueeze(-1).to(states.dtype)
            pooled = self.dropout((states * weights).sum(1) / weights.sum(1).clamp(min=1.0))
            return self.topic(pooled), self.time(pooled).squeeze(-1), self.current(pooled)

        def heads_state_dict(self) -> dict[str, Any]:
            return {k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}

    return RouterEncoder()


def _loss(torch: Any, outputs: tuple[Any, Any, Any], batch: Sequence[Any], w_time: float, w_cur: float) -> Any:
    F = torch.nn.functional
    topic_logits, time_logit, cur_logits = (o.float() for o in outputs)
    _, _, topic, time_y, cur, cur_mask = batch
    n_topic = (topic != IGNORE_INDEX).sum().clamp(min=1)
    l_topic = F.cross_entropy(topic_logits, topic, ignore_index=IGNORE_INDEX, reduction="sum") / n_topic
    l_time = F.binary_cross_entropy_with_logits(time_logit, time_y)
    per = F.binary_cross_entropy_with_logits(cur_logits, cur, reduction="none")
    l_cur = (per * cur_mask).sum() / cur_mask.sum().clamp(min=1.0)
    return l_topic + w_time * l_time + w_cur * l_cur


def _freeze_unused(torch: Any, model: Any, batch: Sequence[Any], w_time: float, w_cur: float) -> list[str]:
    """Freeze parameters that get no gradient from the loss (e.g. a BERT pooler we never use).

    DDP requires every trainable parameter to receive a gradient each step; freezing them up front (identically on
    every rank, since it depends only on the architecture) avoids the slower ``find_unused_parameters`` search.
    """
    _loss(torch, model(batch[0], batch[1]), batch, w_time, w_cur).backward()
    unused = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    for name, p in model.named_parameters():
        if name in unused:
            p.requires_grad_(False)
    model.zero_grad(set_to_none=True)
    return unused


def _load_encoder(name: str, fallback: str | None) -> tuple[Any, Any, str]:
    from transformers import AutoModel, AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name), name
    except (OSError, ValueError) as e:
        if not fallback:
            raise
        print(f"[train_gpu] could not load {name} ({e}); falling back to {fallback}", flush=True)
        return AutoTokenizer.from_pretrained(fallback), AutoModel.from_pretrained(fallback), fallback


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command-line options of :func:`main`."""
    ap = argparse.ArgumentParser(prog="python -m cutoffmix.router.train_gpu", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--events", nargs="+", required=True, help="events JSONL file(s)")
    ap.add_argument("--results", nargs="+", required=True, help="graded probe-results JSONL file(s)")
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--fallback-model", default=FALLBACK_MODEL, help="used if --model-name fails to load ('' = none)")
    ap.add_argument("--budget-min", type=float, default=55.0, help="hard wall-clock budget for the whole run")
    ap.add_argument("--reserve-min", type=float, default=4.0, help="minutes kept free for evaluation and saving")
    ap.add_argument("--out", default="router_models/encoder")
    ap.add_argument("--batch-size", type=int, default=32, help="per GPU")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--probes", nargs="+", default=list(DEFAULT_PROBES))
    ap.add_argument("--eval-frac", type=float, default=0.15, help="share of events held out for evaluation")
    ap.add_argument("--negatives", type=int, default=2000, help="synthetic time-insensitive examples")
    ap.add_argument("--warmup-frac", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--w-time", type=float, default=0.5, help="loss weight of the time-sensitivity head")
    ap.add_argument("--w-cur", type=float, default=1.0, help="loss weight of the currentness head")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-bf16", dest="bf16", action="store_false", help="disable bf16 autocast")
    args = ap.parse_args(argv)
    if args.budget_min <= args.reserve_min:
        ap.error(f"--budget-min ({args.budget_min}) must exceed --reserve-min ({args.reserve_min})")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Train, evaluate on held-out events and save; returns a process exit code."""
    started = time.monotonic()
    args = parse_args(argv)
    deadline = started + (args.budget_min - args.reserve_min) * 60.0

    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from transformers import get_linear_schedule_with_warmup

    from cutoffmix.router.train import read_events, synthetic_time_insensitive

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if world > 1:
        dist.init_process_group(backend="nccl" if use_cuda else "gloo", timeout=timedelta(minutes=30))
    torch.manual_seed(args.seed)

    def log(msg: str) -> None:
        if is_main:
            print(f"[train_gpu {time.monotonic() - started:7.1f}s] {msg}", flush=True)

    events = [e for path in args.events for e in read_events(path)]
    results = [r for path in args.results for r in read_results(path)]
    negatives = synthetic_time_insensitive(args.negatives, args.seed)
    examples, models = build_examples(events, results, probes=args.probes, negatives=negatives)
    train_set, eval_set = split_by_event(examples, args.eval_frac, args.seed)
    if not train_set:
        raise SystemExit("no training examples (check --events/--results)")
    log(f"{len(events)} events, {len(results)} results, models {models}; {len(train_set)} train / "
        f"{len(eval_set)} held-out examples; world size {world}, device {device}")

    if world > 1 and not is_main:
        dist.barrier()  # rank 0 downloads the model into the shared cache first
    tokenizer, encoder, model_name = _load_encoder(args.model_name, args.fallback_model or None)
    if world > 1 and is_main:
        dist.barrier()
    model = _build_model(torch, encoder, len(TOPIC_IDS), len(models), args.dropout).to(device)
    collate = _Collator(tokenizer, args.max_length)
    probe_batch = tuple(t.to(device) for t in collate(train_set[: args.batch_size]))
    frozen = _freeze_unused(torch, model, probe_batch, args.w_time, args.w_cur)
    if frozen:
        log(f"froze {len(frozen)} parameters that the loss does not use: {', '.join(frozen)}")
    base = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)

    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, shuffle=True, seed=args.seed) \
        if world > 1 else None
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=sampler is None, sampler=sampler,
                        collate_fn=collate, num_workers=args.num_workers, pin_memory=use_cuda, drop_last=False)
    total_steps = max(1, args.epochs * len(loader))
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_frac * total_steps), total_steps)
    autocast_on = bool(args.bf16)  # bf16 autocast (CUDA; also valid on CPU for debugging runs)

    def out_of_time() -> bool:
        flag = torch.tensor([1.0 if time.monotonic() >= deadline else 0.0], device=device)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    step = 0
    stopped_by_budget = False
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        for batch in loader:
            if out_of_time():
                stopped_by_budget = True
                break
            batch = tuple(t.to(device, non_blocking=True) for t in batch)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_on):
                outputs = model(batch[0], batch[1])
            loss = _loss(torch, outputs, batch, args.w_time, args.w_cur)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_every == 0:
                log(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")
        if stopped_by_budget:
            log(f"wall-clock budget reached after {step} steps; stopping to evaluate and save")
            break

    if is_main:
        metrics = _evaluate(torch, base, eval_set, collate, args, device, autocast_on, models) if eval_set else {}
        metrics.update({"steps": step, "total_steps": total_steps, "stopped_by_budget": stopped_by_budget,
                        "elapsed_s": round(time.monotonic() - started, 1)})
        _save(torch, base, tokenizer, Path(args.out), model_name, models, args, metrics)
        log(f"saved to {args.out}: {json.dumps(metrics)}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _evaluate(torch: Any, model: Any, eval_set: list[Example], collate: _Collator, args: argparse.Namespace,
              device: Any, autocast_on: bool, models: list[str]) -> dict[str, Any]:
    model.eval()
    topic_pred: list[int] = []
    time_prob: list[float] = []
    cur_prob: list[list[float]] = []
    with torch.no_grad():
        for lo in range(0, len(eval_set), args.batch_size):
            chunk = eval_set[lo : lo + args.batch_size]
            ids, mask = (t.to(device) for t in collate(chunk)[:2])
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_on):
                topic_logits, time_logit, cur_logits = model(ids, mask)
            topic_pred += topic_logits.float().argmax(-1).tolist()
            time_prob += torch.sigmoid(time_logit.float()).tolist()
            cur_prob += torch.sigmoid(cur_logits.float()).tolist()
    return evaluation_metrics(topic_pred, [e.topic for e in eval_set], time_prob,
                              [e.time_sensitive for e in eval_set], cur_prob, [e.currentness for e in eval_set], models)


def _save(torch: Any, model: Any, tokenizer: Any, out: Path, model_name: str, models: list[str],
          args: argparse.Namespace, metrics: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out)
    tokenizer.save_pretrained(out)
    torch.save(model.heads_state_dict(), out / "heads.pt")
    config = {
        "base_model": model_name,
        "topics": list(TOPIC_IDS),
        "models": models,
        "probes": list(args.probes),
        "max_length": args.max_length,
        "pooling": "mean",
        "heads": {"topic": len(TOPIC_IDS), "time": 1, "current": len(models)},
    }
    (out / "router_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
