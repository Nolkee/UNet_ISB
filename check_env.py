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

    # Other dependencies
    deps = ['numpy', 'PIL', 'tqdm']
    for dep in deps:
        try:
            mod = __import__(dep)
            ver = getattr(mod, '__version__', 'unknown')
            print(f"{dep}: {ver}")
        except ImportError:
            print(f"{dep}: NOT INSTALLED")
            return False

    print("\n✅ All dependencies satisfied")
    return True

if __name__ == '__main__':
    success = check_environment()
    sys.exit(0 if success else 1)
