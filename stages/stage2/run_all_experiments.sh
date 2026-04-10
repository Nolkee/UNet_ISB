#!/bin/bash
set -euo pipefail
# Sequential Stage-2 tuning runner
# Runs Exp-1 through Exp-4 one after another without overwriting prior results.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_experiment() {
  local script_name="$1"
  echo "=========================================="
  echo " Running ${script_name}"
  echo " Started at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
  bash "$SCRIPT_DIR/${script_name}"
  echo "=========================================="
  echo " Finished ${script_name}"
  echo " Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "=========================================="
}

run_experiment "train.sh"
run_experiment "train_exp2.sh"
run_experiment "train_exp3.sh"
run_experiment "train_exp4.sh"

echo "All Stage-2 experiments completed."
