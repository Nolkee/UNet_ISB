#!/bin/bash
# Stage-1 配置：配对训练（noisy_esc -> rss_norm via manifest）

TRAIN_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc"
TRAIN_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm"
TRAIN_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/manifest_train.csv"
VAL_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc"
VAL_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm"
VAL_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/manifest_val.csv"

CMD=(
  python train_stage1_restoration.py
  --train-input-dir "$TRAIN_INPUT"
  --train-target-dir "$TRAIN_TARGET"
  --train-manifest "$TRAIN_MANIFEST"
  --val-input-dir "$VAL_INPUT"
  --val-target-dir "$VAL_TARGET"
  --save-dir ./checkpoints_stage1
  --device cuda
  --epochs 50
  --batch-size 8
  --learning-rate 2e-4
  --amp
  --in-channels 1
  --out-channels 1
  --image-size 320
  --crop-size 256
  --irc-weight 0.0
  --save-every 5
)

if [ -f "$VAL_MANIFEST" ]; then
  CMD+=(--val-manifest "$VAL_MANIFEST")
else
  echo "[WARN] Validation manifest not found: $VAL_MANIFEST"
  echo "[WARN] Falling back to filename-based pairing for validation"
fi

"${CMD[@]}"


