#!/bin/bash
set -euo pipefail
# 顺序运行三组 Stage-1 调参实验，并自动保存日志

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/checkpoints_stage1_tuning_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="$RUN_ROOT/logs"
VAL_SAVE_COUNT="${VAL_SAVE_COUNT:-4}"

mkdir -p "$LOG_DIR"

run_variant() {
  local name="$1"
  local script_path="$2"
  local save_dir="$RUN_ROOT/$name"
  local log_path="$LOG_DIR/${name}.log"

  echo "[INFO] Running $name"
  echo "[INFO] save_dir=$save_dir"
  echo "[INFO] log_path=$log_path"

  SAVE_DIR="$save_dir" VAL_SAVE_COUNT="$VAL_SAVE_COUNT" bash "$script_path" 2>&1 | tee "$log_path"
}

run_variant "detail_mild" "$SCRIPT_DIR/train_detail_mild.sh"
run_variant "detail_strong" "$SCRIPT_DIR/train_detail_strong.sh"
run_variant "patch_focused" "$SCRIPT_DIR/train_patch_focused.sh"

echo "[INFO] All Stage-1 tuning runs completed. Outputs saved under $RUN_ROOT"
python "$SCRIPT_DIR/summarize_tuning_runs.py" "$RUN_ROOT"
