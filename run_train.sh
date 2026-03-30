#!/bin/bash

# Compatibility wrapper for the canonical Stage-1 launcher.
# Usage: bash run_train.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/stages/stage1/train.sh"
