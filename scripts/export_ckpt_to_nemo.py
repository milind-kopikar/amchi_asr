#!/usr/bin/env python3
"""
Export a PyTorchLightning checkpoint into a .nemo artifact by loading weights trustfully
(tries load_from_checkpoint, prefix-stripping, then filtered matching). Intended for "post-mortem"
exports from existing checkpoints (no retraining).
"""
import os
import argparse
import shutil
import torch
import yaml
import nemo
import nemo.collections.asr as nemo_asr

from finetune_eval import load_model as _load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--min-free-gb', type=float, default=5.0)
    args = parser.parse_args()

    # Ensure ckpt exists
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(args.ckpt)

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Check free space
    st = os.statvfs(out_dir or '.')
    free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
    if free_gb < args.min_free_gb:
        raise RuntimeError(f"Not enough free space to export .nemo: {free_gb:.2f} GB available, require {args.min_free_gb} GB")

    print('Loading model from checkpoint (trustful path) ...')
    model = _load_model(args.ckpt)
    # Save to .nemo
    print('Saving .nemo to', args.out)
    try:
        model.save_to(args.out)
    except Exception as e:
        # Try fallback: use nemo serialization API if available
        try:
            import nemo.collections.asr as na
            model.save_to(args.out)
        except Exception as e2:
            raise RuntimeError(f"Failed to save .nemo: {e} | {e2}")
    print('Saved .nemo')


if __name__ == '__main__':
    main()
