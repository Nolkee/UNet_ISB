#!/bin/bash
# Compatibility wrapper for the canonical Stage-2 launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

bash "$REPO_ROOT/stages/stage2/train.sh"
