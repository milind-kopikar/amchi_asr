#!/usr/bin/env bash
set -euo pipefail

echo "Running tokenizer round-trip check..."
python scripts/tokenizer_roundtrip.py --model_path models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo

echo "\nRunning smoke inference check..."
python scripts/run_smoke_and_check_deva.py --model_path models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo --manifest data/test/manifest.jsonl --output_dir results/AI4Bharat_amchi_konkani

echo "\nRunning overfit batch sanity check (single training step)..."
python scripts/check_overfit_batch.py

echo "\nAll tests completed."