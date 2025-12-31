Issue: Numba/RNNT GPU backend failed with nvJitLink errors on RunPod (CUDA 12.9 environment).

Fix: Switched loss_name to ctc (Connectionist Temporal Classification) in configs/konkani_finetune.yaml.

Patch: Applied `patches/conv_asr_fixed.py` to handle string Language IDs ('kok') correctly.

Environment: Downgraded Numba to 0.62.1 (though CTC bypasses this mostly).

Command: Use `APPLY_CONV_PATCH=1 python scripts/fine_tune.py --config configs/konkani_finetune.yaml` to run.
