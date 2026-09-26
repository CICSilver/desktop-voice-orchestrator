#!/usr/bin/env bash
# Fine-tune, export and score the final decoder (run inside WSL, see
# docs/WSL_TRAINING_SETUP.md). Every epoch is exported next to its checkpoint
# and scored on the test split; pick the epoch by the dev CER in summary.json.
#
#   bash training/asr/run.sh <experiment-name> [finetune.py options...]
#
# The configuration used on 2026-09-26 (see docs/PERSONALIZATION.md):
#   bash training/asr/run.sh asr-ft1 --num-epochs 3
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate funasr

REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1
shift
EXP=$HOME/dvo/exp/$NAME
mkdir -p "$EXP"

python "$REPO/training/asr/finetune.py" --exp-dir "$EXP" "$@" 2>&1 | tee "$EXP/train.log"

for epoch in $(ls "$EXP" | sed -n 's/^epoch-\([0-9]*\)\.pt$/\1/p' | sort -n); do
  python "$REPO/training/asr/export.py" --checkpoint "$EXP/epoch-$epoch.pt" \
    --out "$EXP/export-epoch-$epoch" > "$EXP/export-$epoch.log" 2>&1
  echo "== $NAME epoch $epoch"
  python "$REPO/training/asr/evaluate.py" --model-dir "$EXP/export-epoch-$epoch" \
    --out "$EXP/eval-epoch-$epoch.json"
done
