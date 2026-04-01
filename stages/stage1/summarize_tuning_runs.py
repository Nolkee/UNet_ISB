#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_metrics(run_dir: Path) -> dict[str, object] | None:
    metrics_path = run_dir / 'final_eval_metrics.json'
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text(encoding='utf-8'))
    return {
        'name': run_dir.name,
        'checkpoint': data.get('checkpoint', '-'),
        'sample_count': data.get('sample_count', '-'),
        'psnr': data.get('psnr', None),
        'ssim': data.get('ssim', None),
        'lpips': data.get('lpips', None),
    }


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f'{value:.{digits}f}' if isinstance(value, float) else str(value)
    if value is None:
        return '-'
    return str(value)


def build_rows(run_root: Path) -> list[dict[str, object]]:
    rows = []
    for child in sorted(run_root.iterdir()):
        if not child.is_dir() or child.name == 'logs':
            continue
        metrics = read_metrics(child)
        if metrics is not None:
            rows.append(metrics)
    return rows


def print_table(rows: list[dict[str, object]], run_root: Path) -> None:
    print(f'[INFO] Stage-1 tuning summary: {run_root}')
    if not rows:
        print('[WARN] No final_eval_metrics.json files found.')
        return

    headers = ['variant', 'sample_count', 'psnr', 'ssim', 'lpips', 'checkpoint']
    table = [
        [
            row['name'],
            fmt(row['sample_count'], 0),
            fmt(row['psnr']),
            fmt(row['ssim']),
            fmt(row['lpips']),
            fmt(row['checkpoint'], 0),
        ]
        for row in rows
    ]
    widths = [max(len(header), *(len(str(r[i])) for r in table)) for i, header in enumerate(headers)]

    def render(cols: list[str]) -> str:
        return ' | '.join(str(col).ljust(widths[i]) for i, col in enumerate(cols))

    print(render(headers))
    print('-+-'.join('-' * width for width in widths))
    for row in table:
        print(render(row))

    best_psnr = max((row for row in rows if isinstance(row['psnr'], (int, float))), key=lambda item: item['psnr'], default=None)
    best_ssim = max((row for row in rows if isinstance(row['ssim'], (int, float))), key=lambda item: item['ssim'], default=None)
    best_lpips = min((row for row in rows if isinstance(row['lpips'], (int, float))), key=lambda item: item['lpips'], default=None)

    if best_psnr is not None:
        print(f"[INFO] Best PSNR : {best_psnr['name']} ({fmt(best_psnr['psnr'])})")
    if best_ssim is not None:
        print(f"[INFO] Best SSIM : {best_ssim['name']} ({fmt(best_ssim['ssim'])})")
    if best_lpips is not None:
        print(f"[INFO] Best LPIPS: {best_lpips['name']} ({fmt(best_lpips['lpips'])})")


if __name__ == '__main__':
    run_root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
    print_table(build_rows(run_root), run_root)
