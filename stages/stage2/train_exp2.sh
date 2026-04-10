#!/bin/bash
set -euo pipefail
# Stage-2: PatchGAN 调参 Exp-2
# 更稳的 GAN：降低 adv-weight，延长 warmup，进一步降低 d_lr

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Data paths ----
TRAIN_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc"
TRAIN_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm"
TRAIN_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/manifest_train.csv"
VAL_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc"
VAL_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm"
VAL_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/manifest_test.csv"

# ---- Configurable paths ----
STAGE1_CKPT="${STAGE1_CKPT:-$REPO_ROOT/checkpoints_stage1_tuning_20260401_163223/detail_mild/best_stage1.pth}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints_stage2_exp2_20260411}"
VAL_SAVE_COUNT="${VAL_SAVE_COUNT:-4}"

CMD=(
  python "$REPO_ROOT/train_stage2_restoration.py"
  --load-stage1 "$STAGE1_CKPT"
  --train-input-dir "$TRAIN_INPUT"
  --train-target-dir "$TRAIN_TARGET"
  --train-manifest "$TRAIN_MANIFEST"
  --val-input-dir "$VAL_INPUT"
  --val-target-dir "$VAL_TARGET"
  --save-dir "$SAVE_DIR"
  --device cuda

  # Training hyperparameters
  --epochs 50
  --batch-size 8
  --g-lr 1e-4
  --d-lr 1e-4

  # Model
  --in-channels 1
  --out-channels 1
  --image-size 320
  --crop-size 256

  # Discriminator
  --ndf 64
  --n-layers 3

  # Content losses
  --reconstruction-weight 1.0
  --high-frequency-weight 0.8
  --patch-nce-weight 0.15
  --bro-weight 0.03
  --irc-weight 0.0

  # Adversarial (Exp-2)
  --adv-weight 0.7
  --warmup-epochs 10

  --save-every 5
  --val-save-count "$VAL_SAVE_COUNT"
  --amp
)

if [ -f "$VAL_MANIFEST" ]; then
  CMD+=(--val-manifest "$VAL_MANIFEST")
else
  echo "[WARN] Validation manifest not found: $VAL_MANIFEST"
  echo "[WARN] Falling back to filename-based pairing for validation"
fi

echo "=========================================="
echo " Stage-2: PatchGAN Tuning (Exp-2)"
echo " Stage-1 checkpoint: $STAGE1_CKPT"
echo " Save directory:     $SAVE_DIR"
echo "=========================================="

"${CMD[@]}"
