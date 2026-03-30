#!/usr/bin/env python3
"""验证数据集路径、manifest 和可配对样本数"""

import csv
from pathlib import Path

TRAIN_INPUT = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc')
TRAIN_TARGET = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm')
TRAIN_MANIFEST = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/manifest_train.csv')
VAL_INPUT = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc')
VAL_TARGET = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm')
VAL_MANIFEST = Path('/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/manifest_test.csv')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.npy', '.pt', '.pth'}


def count_images(path: Path) -> int:
    if not path.exists():
        return -1
    return sum(1 for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES)


def resolve_manifest_keys(fields: list[str], input_root: Path, target_root: Path) -> tuple[str | None, str | None]:
    field_map = {field.strip().lower(): field for field in fields if field}
    input_candidates = ['input_path', f'{input_root.name.lower()}_path', input_root.name.lower()]
    target_candidates = ['target_path', f'{target_root.name.lower()}_path', target_root.name.lower()]
    input_key = next((field_map[candidate] for candidate in input_candidates if candidate in field_map), None)
    target_key = next((field_map[candidate] for candidate in target_candidates if candidate in field_map), None)
    return input_key, target_key


def check_manifest(manifest: Path, input_root: Path, target_root: Path) -> tuple[bool, str]:
    if not manifest.exists():
        return False, f'缺少 manifest: {manifest}'

    with manifest.open('r', newline='', encoding='utf-8-sig') as handle:
        delimiter = '\t' if manifest.suffix.lower() == '.tsv' else ','
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = reader.fieldnames or []
        input_key, target_key = resolve_manifest_keys(fields, input_root, target_root)
        if input_key is None or target_key is None:
            return False, f'manifest 缺少路径列，当前列为: {fields}'

        total = 0
        usable = 0
        missing = 0
        for row in reader:
            total += 1
            raw_input = (row.get(input_key) or '').strip()
            raw_target = (row.get(target_key) or '').strip()
            if not raw_input or not raw_target:
                continue
            input_path = Path(raw_input)
            target_path = Path(raw_target)
            if not input_path.is_absolute():
                input_path = input_root / input_path
            if not target_path.is_absolute():
                target_path = target_root / target_path
            if not input_path.exists() or not target_path.exists():
                missing += 1
                continue
            usable += 1

    if usable == 0:
        return False, f'manifest 共 {total} 条，但可用配对样本为 0'
    if missing > 0:
        return False, f'manifest 共 {total} 条，可用 {usable} 条，其中 {missing} 条路径不存在'
    return True, f'manifest 可用，共 {usable} 对样本（原始 {total} 条）'


print('=== 数据集验证 ===\n')
for name, path in [
    ('train_noisy', TRAIN_INPUT),
    ('train_clean', TRAIN_TARGET),
    ('val_noisy', VAL_INPUT),
    ('val_clean', VAL_TARGET),
]:
    count = count_images(path)
    if count >= 0:
        print(f'✅ {name}: {count} 张图像')
    else:
        print(f'❌ {name}: 路径不存在 -> {path}')

print('\n=== manifest 检查 ===\n')
ok, msg = check_manifest(TRAIN_MANIFEST, TRAIN_INPUT, TRAIN_TARGET)
print(('✅ ' if ok else '❌ ') + f'train_manifest: {msg}')
ok, msg = check_manifest(VAL_MANIFEST, VAL_INPUT, VAL_TARGET)
print(('✅ ' if ok else '❌ ') + f'val_manifest: {msg}')

