#!/usr/bin/env python3
"""Compatibility wrapper for the canonical Stage-1 dataset verifier."""

from pathlib import Path
import runpy

SCRIPT_PATH = Path(__file__).resolve().parent / 'stages' / 'stage1' / 'verify_dataset.py'
runpy.run_path(str(SCRIPT_PATH), run_name='__main__')

