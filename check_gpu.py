#!/usr/bin/env python3
"""
Preflight check: verify that a CUDA-capable GPU is available and visible to PyTorch.
Exits with code 0 if GPU is OK, 1 otherwise. Used at the start of the test suite
so we do not run long training on CPU by mistake.
"""
import sys

def main():
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed. Install it before running GPU checks.")
        return 1

    if not torch.cuda.is_available():
        print("ERROR: No CUDA-capable GPU is visible to PyTorch.")
        print("  - Ensure a GPU is attached and drivers/CUDA are installed.")
        print("  - If using CUDA_VISIBLE_DEVICES, ensure it is set to a valid device (e.g. 0).")
        return 1

    ndev = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if ndev else "unknown"
    print(f"OK: CUDA available. Device count: {ndev}. Device 0: {name}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
