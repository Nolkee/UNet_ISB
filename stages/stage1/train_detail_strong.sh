#!/bin/bash
set -euo pipefail
# Stage-1 调参组：更强细节约束

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc"
TRAIN_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm"
TRAIN_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/manifest_train.csv"
VAL_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc"
VAL_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm"
VAL_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/manifest_test.csv"
VAL_SAVE_COUNT="${VAL_SAVE_COUNT:-4}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints_stage1_detail_strong}"

CMD=(
  python "$REPO_ROOT/train_stage1_restoration.py"
  --train-input-dir "$TRAIN_INPUT"
  --train-target-dir "$TRAIN_TARGET"
  --train-manifest "$TRAIN_MANIFEST"
  --val-input-dir "$VAL_INPUT"
  --val-target-dir "$VAL_TARGET"
  --save-dir "$SAVE_DIR"
  --device cuda
  --epochs 50
  --batch-size 8
  --learning-rate 8e-5
  --in-channels 1
  --out-channels 1
  --image-size 320
  --crop-size 256
  --high-frequency-weight 1.0
  --patch-nce-weight 0.25
  --bro-weight 0.02
  --irc-weight 0.0
  --save-every 5
  --val-save-count "$VAL_SAVE_COUNT"
)

if [ -f "$VAL_MANIFEST" ]; then
  CMD+=(--val-manifest "$VAL_MANIFEST")
else
  echo "[WARN] Validation manifest not found: $VAL_MANIFEST"
  echo "[WARN] Falling back to filename-based pairing for validation"
fi

"${CMD[@]}"
