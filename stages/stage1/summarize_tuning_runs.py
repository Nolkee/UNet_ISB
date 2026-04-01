#!/usr/bin/env python3
from __future__ import annotations

import csv
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


def rank_rows(rows: list[dict[str, object]], key: str, reverse: bool) -> list[dict[str, object]]:
    valid = [row for row in rows if isinstance(row.get(key), (int, float))]
    return sorted(valid, key=lambda item: item[key], reverse=reverse)


def choose_recommended_variant(rows: list[dict[str, object]]) -> dict[str, object] | None:
    valid = [row for row in rows if all(isinstance(row.get(key), (int, float)) for key in ('psnr', 'ssim', 'lpips'))]
    if not valid:
        return None

    psnr_rank = {row['name']: idx + 1 for idx, row in enumerate(rank_rows(valid, 'psnr', reverse=True))}
    ssim_rank = {row['name']: idx + 1 for idx, row in enumerate(rank_rows(valid, 'ssim', reverse=True))}
    lpips_rank = {row['name']: idx + 1 for idx, row in enumerate(rank_rows(valid, 'lpips', reverse=False))}

    scored = []
    for row in valid:
        total_rank = psnr_rank[row['name']] + ssim_rank[row['name']] + lpips_rank[row['name']]
        scored.append((total_rank, -row['psnr'], -row['ssim'], row['lpips'], row))
    scored.sort()
    winner = scored[0][-1]
    return {
        'variant': winner['name'],
        'reason': 'lowest combined rank across PSNR↑, SSIM↑, LPIPS↓',
        'ranks': {
            'psnr': psnr_rank[winner['name']],
            'ssim': ssim_rank[winner['name']],
            'lpips': lpips_rank[winner['name']],
            'total': scored[0][0],
        },
        'metrics': {
            'psnr': winner['psnr'],
            'ssim': winner['ssim'],
            'lpips': winner['lpips'],
        },
    }


def build_summary(rows: list[dict[str, object]], run_root: Path) -> dict[str, object]:
    best_psnr = max((row for row in rows if isinstance(row['psnr'], (int, float))), key=lambda item: item['psnr'], default=None)
    best_ssim = max((row for row in rows if isinstance(row['ssim'], (int, float))), key=lambda item: item['ssim'], default=None)
    best_lpips = min((row for row in rows if isinstance(row['lpips'], (int, float))), key=lambda item: item['lpips'], default=None)
    rankings = {
        'psnr_desc': [row['name'] for row in rank_rows(rows, 'psnr', reverse=True)],
        'ssim_desc': [row['name'] for row in rank_rows(rows, 'ssim', reverse=True)],
        'lpips_asc': [row['name'] for row in rank_rows(rows, 'lpips', reverse=False)],
    }
    return {
        'run_root': str(run_root),
        'runs': rows,
        'best': {
            'psnr': {'variant': best_psnr['name'], 'value': best_psnr['psnr']} if best_psnr is not None else None,
            'ssim': {'variant': best_ssim['name'], 'value': best_ssim['ssim']} if best_ssim is not None else None,
            'lpips': {'variant': best_lpips['name'], 'value': best_lpips['lpips']} if best_lpips is not None else None,
        },
        'rankings': rankings,
        'recommended_next_variant': choose_recommended_variant(rows),
    }


def write_summary_files(rows: list[dict[str, object]], run_root: Path) -> tuple[Path, Path]:
    summary = build_summary(rows, run_root)
    json_path = run_root / 'summary.json'
    csv_path = run_root / 'summary.csv'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['name', 'sample_count', 'psnr', 'ssim', 'lpips', 'checkpoint'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return json_path, csv_path


def print_table(rows: list[dict[str, object]], run_root: Path) -> None:
    print(f'[INFO] Stage-1 tuning summary: {run_root}')
    if not rows:
        print('[WARN] No final_eval_metrics.json files found.')
        return

    json_path, csv_path = write_summary_files(rows, run_root)

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

    summary = build_summary(rows, run_root)
    best = summary['best']
    if best['psnr'] is not None:
        print(f"[INFO] Best PSNR : {best['psnr']['variant']} ({fmt(best['psnr']['value'])})")
    if best['ssim'] is not None:
        print(f"[INFO] Best SSIM : {best['ssim']['variant']} ({fmt(best['ssim']['value'])})")
    if best['lpips'] is not None:
        print(f"[INFO] Best LPIPS: {best['lpips']['variant']} ({fmt(best['lpips']['value'])})")

    print(f"[INFO] Ranking PSNR↑ : {', '.join(summary['rankings']['psnr_desc']) or '-'}")
    print(f"[INFO] Ranking SSIM↑ : {', '.join(summary['rankings']['ssim_desc']) or '-'}")
    print(f"[INFO] Ranking LPIPS↓: {', '.join(summary['rankings']['lpips_asc']) or '-'}")

    recommended = summary['recommended_next_variant']
    if recommended is not None:
        print(
            f"[INFO] Recommended next variant: {recommended['variant']} "
            f"(PSNR={fmt(recommended['metrics']['psnr'])}, SSIM={fmt(recommended['metrics']['ssim'])}, "
            f"LPIPS={fmt(recommended['metrics']['lpips'])}; total_rank={recommended['ranks']['total']})"
        )
    print(f'[INFO] Wrote summary files: {json_path} {csv_path}')


if __name__ == '__main__':
    run_root = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else Path.cwd()
    print_table(build_rows(run_root), run_root)
