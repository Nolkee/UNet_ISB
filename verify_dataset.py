#!/usr/bin/env python3
"""验证数据集路径和样本数量"""

from pathlib import Path

# 数据集路径
PATHS = {
    'train_noisy': '/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/noisy_esc',
    'train_clean': '/home/cavin/CJX/fastMRIdata/MRI_dataset/train_dataset/train_slices/rss_norm',
    'val_noisy': '/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/noisy_esc',
    'val_clean': '/home/cavin/CJX/fastMRIdata/MRI_dataset/test_dataset/test_slices/rss_norm',
}

print("=== 数据集验证 ===\n")

for name, path in PATHS.items():
    p = Path(path)
    if p.exists():
        files = list(p.glob('*'))
        images = [f for f in files if f.suffix.lower() in {'.png', '.jpg', '.npy', '.pt'}]
        print(f"✅ {name}: {len(images)} 张图像")
    else:
        print(f"❌ {name}: 路径不存在")

print("\n预期数量：")
print("  训练集: noisy_esc=2420, rss_norm=2420")
print("  验证集: noisy_esc=1000, rss_norm=1000")
