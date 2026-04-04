#!/bin/bash
set -euo pipefail
# Stage-3: Barycenter Regularization (DANN + GRL)
# GAN 保留自 Stage-2；新增 L_WB + re-enable IRC

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---- Data paths (same as Stage-1/2) ----
TRAIN_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc"
TRAIN_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm"
TRAIN_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/manifest_train.csv"
VAL_INPUT="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc"
VAL_TARGET="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm"
VAL_MANIFEST="/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/manifest_test.csv"

# ---- Configurable paths ----
STAGE2_CKPT="${STAGE2_CKPT:-$REPO_ROOT/checkpoints_stage2/best_stage2.pth}"
SAVE_DIR="${SAVE_DIR:-$REPO_ROOT/checkpoints_stage3}"
VAL_SAVE_COUNT="${VAL_SAVE_COUNT:-4}"

CMD=(
  python "$REPO_ROOT/train_stage3_restoration.py"
  --load-stage2 "$STAGE2_CKPT"
  --train-input-dir "$TRAIN_INPUT"
  --train-target-dir "$TRAIN_TARGET"
  --train-manifest "$TRAIN_MANIFEST"
  --val-input-dir "$VAL_INPUT"
  --val-target-dir "$VAL_TARGET"
  --save-dir "$SAVE_DIR"
  --device cuda

  # Training hyperparameters (LR halved from Stage-2)
  --epochs 50
  --batch-size 8
  --g-lr 5e-5
  --d-lr 1e-4

  # Model (match Stage-1/2)
  --in-channels 1
  --out-channels 1
  --image-size 320
  --crop-size 256

  # Discriminator (same as Stage-2)
  --ndf 64
  --n-layers 3

  # Content losses
  --reconstruction-weight 1.0
  --high-frequency-weight 0.8
  --patch-nce-weight 0.15
  --bro-weight 0.05
  --irc-weight 0.05

  # GAN adversarial (no re-warmup)
  --adv-weight 1.0
  --adv-warmup-epochs 0

  # Barycenter regularization (NEW)
  --wb-weight 0.1
  --wb-warmup-epochs 5
  --grl-max-lambda 1.0
  --grl-ramp-epochs 10
  --num-degradation-classes 0
  --wb-hidden-dim 256

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
echo " Stage-3: Barycenter Regularization"
echo " Stage-2 checkpoint: $STAGE2_CKPT"
echo " Save directory:     $SAVE_DIR"
echo "=========================================="

"${CMD[@]}"
