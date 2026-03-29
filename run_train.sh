#!/bin/bash

# Stage-1 SB-EQ U-Net Training Script
# Usage: bash run_train.sh

python train_stage1_restoration.py \
  --train-input-dir /path/to/train/degraded \
  --train-target-dir /path/to/train/clean \
  --val-input-dir /path/to/val/degraded \
  --val-target-dir /path/to/val/clean \
  --save-dir ./checkpoints_stage1 \
  --device cuda \
  --epochs 100 \
  --batch-size 8 \
  --learning-rate 2e-4 \
  --num-workers 4 \
  --amp \
  --image-size 256 \
  --crop-size 256 \
  --save-every 5 \
  --reconstruction-weight 1.0 \
  --high-frequency-weight 0.5 \
  --patch-nce-weight 0.1 \
  --bro-weight 0.05 \
  --irc-weight 0.05
