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


def check_manifest(manifest: Path, input_root: Path, target_root: Path) -> tuple[bool, str]:
    if not manifest.exists():
        return False, f'缺少 manifest: {manifest}'

    with manifest.open('r', newline='') as handle:
        delimiter = '\t' if manifest.suffix.lower() == '.tsv' else ','
        reader = csv.DictReader(handle, delimiter=delimiter)
        fields = reader.fieldnames or []
        if 'input_path' not in fields or 'target_path' not in fields:
            return False, f'manifest 缺少必要列，当前列为: {fields}'

        total = 0
        missing = 0
        for row in reader:
            total += 1
            input_path = Path(row['input_path'])
            target_path = Path(row['target_path'])
            if not input_path.is_absolute():
                input_path = input_root / input_path
            if not target_path.is_absolute():
                target_path = target_root / target_path
            if not input_path.exists() or not target_path.exists():
                missing += 1

    if missing > 0:
        return False, f'manifest 共 {total} 条，其中 {missing} 条路径不存在'
    return True, f'manifest 可用，共 {total} 对样本'


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

