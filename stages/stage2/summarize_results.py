#!/usr/bin/env python3
"""Summarize Stage-2 experiment results from multiple output directories."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = [
    ('Exp-1', REPO_ROOT / 'checkpoints_stage2_exp1_20260409'),
    ('Exp-2', REPO_ROOT / 'checkpoints_stage2_exp2_20260411'),
    ('Exp-3', REPO_ROOT / 'checkpoints_stage2_exp3_20260411'),
    ('Exp-4', REPO_ROOT / 'checkpoints_stage2_exp4_20260411'),
]


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def format_metric(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return '-'
    return f'{value:.{digits}f}'


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, exp_dir in EXPERIMENTS:
        final_eval = load_json(exp_dir / 'final_eval_metrics.json') or {}
        train_args = load_json(exp_dir / 'train_args.json') or {}

        row = {
            'experiment': name,
            'dir': exp_dir.name,
            'exists': 'yes' if exp_dir.exists() else 'no',
            'checkpoint': str(final_eval.get('checkpoint', '-')),
            'psnr': format_metric(final_eval.get('psnr')),
            'ssim': format_metric(final_eval.get('ssim')),
            'lpips': format_metric(final_eval.get('lpips')),
            'samples': str(final_eval.get('sample_count', '-')),
            'adv_weight': format_metric(train_args.get('adv_weight'), digits=2),
            'warmup': str(train_args.get('warmup_epochs', '-')),
            'g_lr': format_metric(train_args.get('g_lr'), digits=6),
            'd_lr': format_metric(train_args.get('d_lr'), digits=6),
            'hf': format_metric(train_args.get('high_frequency_weight'), digits=2),
            'patch_nce': format_metric(train_args.get('patch_nce_weight'), digits=2),
        }
        rows.append(row)
    return rows


def print_table(rows: list[dict[str, str]]) -> None:
    headers = [
        'experiment', 'exists', 'checkpoint', 'psnr', 'ssim', 'lpips', 'samples',
        'adv_weight', 'warmup', 'g_lr', 'd_lr', 'hf', 'patch_nce', 'dir',
    ]
    widths = {
        header: max(len(header), *(len(row[header]) for row in rows))
        for header in headers
    }

    def render(row: dict[str, str]) -> str:
        return ' | '.join(row[header].ljust(widths[header]) for header in headers)

    print(render({header: header for header in headers}))
    print('-+-'.join('-' * widths[header] for header in headers))
    for row in rows:
        print(render(row))


def print_best(rows: list[dict[str, str]]) -> None:
    available = []
    for row in rows:
        try:
            available.append((float(row['lpips']), -float(row['psnr']), row))
        except ValueError:
            continue
    if not available:
        print('\nNo completed experiment results found yet.')
        return

    available.sort()
    best = available[0][2]
    print('\nBest LPIPS so far:')
    print(
        f"  {best['experiment']} | LPIPS={best['lpips']} | PSNR={best['psnr']} | "
        f"SSIM={best['ssim']} | checkpoint={best['checkpoint']}"
    )


def main() -> None:
    rows = collect_rows()
    print_table(rows)
    print_best(rows)


if __name__ == '__main__':
    main()
