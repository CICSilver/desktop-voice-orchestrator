#!/usr/bin/env bash
# Fine-tune, export and score the KWS model (run inside WSL, see
# docs/WSL_TRAINING_SETUP.md). Every epoch is exported to streaming ONNX
# (fp32 and int8) next to its checkpoint and scored on the test split.
#
#   bash training/kws/run.sh <experiment-name> [finetune.py options...]
#
# The configuration that worked on 2026-09-25 (see docs/PERSONALIZATION.md):
#   bash training/kws/run.sh kws-ft4 --num-epochs 6 --repeat 4 \
#     --pseudo-labels ~/dvo/data/pseudo_labels.jsonl --pseudo-repeat 5 --freeze-decoder
set -euo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate kws

REPO=$(cd "$(dirname "$0")/../.." && pwd)
NAME=$1
shift
EXP=$HOME/dvo/exp/$NAME
PRETRAINED=$HOME/dvo/models/icefall-kws-zipformer-zh-en-3M-2025-12-20
THRESHOLDS=${THRESHOLDS:-0.15 0.25 0.35}
mkdir -p "$EXP"

if [[ " $* " == *" --pseudo-labels "* ]] && [ ! -f "$HOME/dvo/data/pseudo_labels.jsonl" ]; then
  python "$REPO/training/kws/pseudo_label.py" --out "$HOME/dvo/data/pseudo_labels.jsonl"
fi

python "$REPO/training/kws/finetune.py" --exp-dir "$EXP" "$@" 2>&1 \
  | grep --line-buffered -v -E "FutureWarning|def forward|def backward|ScheduledFloat|Whitening|Clipping_scale" \
  | tee "$EXP/train.log"

# The model hyper-parameters below are those of the pretrained checkpoint.
cd "$HOME/dvo/icefall/egs/librispeech/ASR"
for epoch in $(ls "$EXP" | sed -n 's/^epoch-\([0-9]*\)\.pt$/\1/p' | sort -n); do
  python zipformer/export-onnx-streaming.py \
    --exp-dir "$EXP" --tokens "$PRETRAINED/data/lang_phone/tokens.txt" \
    --epoch "$epoch" --avg 1 --use-averaged-model 0 \
    --num-encoder-layers 1,1,1,1,1,1 --downsampling-factor 1,2,4,8,4,2 \
    --feedforward-dim 192,192,192,192,192,192 --num-heads 4,4,4,8,4,4 \
    --encoder-dim 128,128,128,128,128,128 --encoder-unmasked-dim 128,128,128,128,128,128 \
    --query-head-dim 32 --value-head-dim 12 --pos-head-dim 4 --pos-dim 48 \
    --cnn-module-kernel 15,15,15,15,15,15 --decoder-dim 320 --joiner-dim 320 \
    --causal 1 --chunk-size 16 --left-context-frames 64 --use-transducer 1 --use-ctc 1 \
    > "$EXP/export-$epoch.log" 2>&1
  echo "== $NAME epoch $epoch (int8)"
  # shellcheck disable=SC2086
  python "$REPO/training/kws/evaluate.py" --model-dir "$EXP" \
    --suffix=-epoch-$epoch-avg-1-chunk-16-left-64 --int8 \
    --thresholds $THRESHOLDS --out "$EXP/eval-epoch-$epoch-int8.json"
done
