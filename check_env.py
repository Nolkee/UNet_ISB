#!/usr/bin/env python3
"""Environment and dependency checker for SB-EQ U-Net training."""

import sys


def check_environment():
    print("=== Environment Check ===\n")

    # Python version
    print(f"Python: {sys.version}")

    # PyTorch
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    Memory: {mem:.1f} GB")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        return False

    basic_dependencies = [
        ('numpy', 'numpy'),
        ('Pillow', 'PIL'),
        ('tqdm', 'tqdm'),
    ]
    for display_name, module_name in basic_dependencies:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"{display_name}: {ver}")
        except ImportError:
            print(f"{display_name}: NOT INSTALLED")
            return False

    try:
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )
        print(f"torchmetrics: PSNR={peak_signal_noise_ratio.__module__} SSIM={structural_similarity_index_measure.__module__}")
    except ImportError:
        print("torchmetrics: NOT INSTALLED")
        return False

    try:
        import lpips
        ver = getattr(lpips, '__version__', 'unknown')
        print(f"lpips: {ver}")
    except ImportError:
        print("lpips: NOT INSTALLED")
        return False

    print("\n✅ All dependencies satisfied")
    return True


if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
