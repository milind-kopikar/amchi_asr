#!/bin/bash
# Run both Amchi Konkani experiments sequentially.
# Checkpoints and results are saved to separate directories.
set -e
cd /workspace/amchi_asr
export APPLY_CONV_PATCH=1

RUN_C_DIR="results/run_c_story_split"
RUN_S_DIR="results/run_c_stratified_split"
mkdir -p "$RUN_C_DIR" "$RUN_S_DIR"

echo "================================================"
echo "$(date) — Starting Run C (freeze encoder, story-based split)"
echo "  Output: $RUN_C_DIR"
echo "================================================"
python3 scripts/fine_tune.py \
  --config configs/amchi_konkani_run_c.yaml \
  --output_dir "$RUN_C_DIR"
echo "$(date) — Run C training complete"

echo ""
echo "================================================"
echo "$(date) — Evaluating Run C on test set"
echo "================================================"
BEST_CKPT_C=$(ls -t "$RUN_C_DIR"/checkpoints/*.ckpt 2>/dev/null | grep -v last | head -1)
if [ -z "$BEST_CKPT_C" ]; then
  echo "ERROR: No Run C checkpoint found in $RUN_C_DIR/checkpoints/"
  exit 1
fi
echo "Best Run C checkpoint: $BEST_CKPT_C"
mkdir -p results/experiments/run_c_story_split
python3 scripts/evaluate.py \
  --checkpoint "$BEST_CKPT_C" \
  --manifest data/amchi/test/manifest.jsonl \
  --output results/experiments/run_c_story_split/final_test_results.json
echo "$(date) — Run C evaluation complete"

echo ""
echo "================================================"
echo "$(date) — Starting Run S (freeze encoder, speaker-stratified split)"
echo "  Output: $RUN_S_DIR"
echo "================================================"
python3 scripts/fine_tune.py \
  --config configs/amchi_konkani_run_c_stratified.yaml \
  --output_dir "$RUN_S_DIR"
echo "$(date) — Run S training complete"

echo ""
echo "================================================"
echo "$(date) — Evaluating Run S on stratified test set"
echo "================================================"
BEST_CKPT_S=$(ls -t "$RUN_S_DIR"/checkpoints/*.ckpt 2>/dev/null | grep -v last | head -1)
if [ -z "$BEST_CKPT_S" ]; then
  echo "ERROR: No Run S checkpoint found in $RUN_S_DIR/checkpoints/"
  exit 1
fi
echo "Best Run S checkpoint: $BEST_CKPT_S"
mkdir -p results/experiments/run_c_stratified_split
python3 scripts/evaluate.py \
  --checkpoint "$BEST_CKPT_S" \
  --manifest data/amchi_stratified/test/manifest.jsonl \
  --output results/experiments/run_c_stratified_split/final_test_results.json
echo "$(date) — Run S evaluation complete"

echo ""
echo "================================================"
echo "$(date) — ALL EXPERIMENTS DONE"
echo "Results:"
echo "  Run C: results/experiments/run_c_story_split/final_test_results.json"
echo "  Run S: results/experiments/run_c_stratified_split/final_test_results.json"
echo "================================================"
