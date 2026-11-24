#!/usr/bin/env bash
# Fine-tune the optional encoder router (cutoffmix/router/train_gpu.py) within a hard 55-minute wall-clock budget.
#
# Needs: pip install -e '.[gpu]'  (torch, transformers, accelerate) on a CUDA machine with bf16 support.
# Inputs: an events JSONL and graded probe-results JSONL from `cutoffmix bench ...`.
#
# Usage:
#   scripts/train_router_gpu.sh 1   # one GPU
#   scripts/train_router_gpu.sh 8   # eight GPUs (torchrun DDP; the batch size below is per GPU)
#
# Environment overrides: EVENTS, RESULTS, OUT, MODEL, BUDGET_MIN, BATCH_SIZE, LR, EPOCHS.
set -euo pipefail

NGPUS="${1:-1}"
EVENTS="${EVENTS:-runs/bench/events.jsonl}"
RESULTS="${RESULTS:-runs/bench/results.jsonl}"
OUT="${OUT:-router_models/encoder}"
MODEL="${MODEL:-jhu-clsp/mmBERT-base}"
BUDGET_MIN="${BUDGET_MIN:-55}"
EPOCHS="${EPOCHS:-4}"

if [[ "$NGPUS" == "1" ]]; then
  BATCH_SIZE="${BATCH_SIZE:-32}"
  LR="${LR:-3e-5}"
else
  # 8 GPUs: keep the per-GPU batch, scale the learning rate roughly with sqrt(global batch).
  BATCH_SIZE="${BATCH_SIZE:-32}"
  LR="${LR:-6e-5}"
fi

for f in "$EVENTS" "$RESULTS"; do
  [[ -f "$f" ]] || { echo "missing input file: $f (set EVENTS/RESULTS)" >&2; exit 1; }
done

# The budget covers the whole run (model download, data prep, training, evaluation, saving); training stops
# --reserve-min minutes before it on every rank at the same step.
exec torchrun --standalone --nproc_per_node="$NGPUS" -m cutoffmix.router.train_gpu \
  --events "$EVENTS" \
  --results "$RESULTS" \
  --model-name "$MODEL" \
  --budget-min "$BUDGET_MIN" \
  --reserve-min 4 \
  --out "$OUT" \
  --batch-size "$BATCH_SIZE" \
  --lr "$LR" \
  --epochs "$EPOCHS"
