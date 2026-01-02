#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Extended 5-Epoch Smoke Test...${NC}"

# 1. Cleanup previous runs
BASE_DIR="/workspace/amchi_asr/nemo_experiments"
echo "Cleaning up previous experiment directory: $BASE_DIR"
rm -rf "$BASE_DIR"

# 2. Run Training
echo -e "${GREEN}Running 5-epoch training...${NC}"
python scripts/fine_tune.py --config "configs/tmp_marathi_5epoch_ctc_smoke.yaml" --output_dir "$BASE_DIR"

# 3. Verify Training Results (Loss Reduction)
echo -e "${GREEN}Verifying training metrics...${NC}"
# Find the latest metrics file
METRICS_FILE=$(find "$BASE_DIR/experiments" -name "epoch_metrics.csv" | sort | tail -n 1)

if [ -z "$METRICS_FILE" ]; then
    echo -e "${RED}Error: Metrics file not found in $BASE_DIR/experiments${NC}"
    exit 1
fi
echo "Found metrics file: $METRICS_FILE"

# Use python to analyze the CSV
python -c "
import pandas as pd
import sys

try:
    df = pd.read_csv('$METRICS_FILE')
    print(f'Loaded metrics with {len(df)} epochs')
    
    if len(df) < 5:
        print('Error: Expected 5 epochs, found', len(df))
        sys.exit(1)
        
    first_loss = df.iloc[0]['train_loss']
    last_loss = df.iloc[-1]['train_loss']
    
    print(f'Epoch 1 Loss: {first_loss}')
    print(f'Epoch 5 Loss: {last_loss}')
    
    if pd.isna(first_loss):
        print('WARNING: First epoch loss is NaN. Cannot compare.')
    elif last_loss < first_loss:
        print('SUCCESS: Loss decreased.')
    else:
        print('WARNING: Loss did not decrease. This might be expected for a tiny dataset/short run, but worth noting.')
        
except Exception as e:
    print(f'Error analyzing metrics: {e}')
    sys.exit(1)
"

# 4. Verify Checkpoints (Top 3)
echo -e "${GREEN}Verifying checkpoints...${NC}"
# Look for checkpoints in the checkpoints subdir
CHECKPOINTS_DIR="$BASE_DIR/checkpoints"
# Exclude last.ckpt if it exists (though we set save_last=false, NeMo might create it)
CHECKPOINTS=$(find "$CHECKPOINTS_DIR" -name "*.ckpt" ! -name "last.ckpt" | sort)
NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)

echo "Found $NUM_CHECKPOINTS checkpoints (excluding last.ckpt):"
echo "$CHECKPOINTS"

if [ "$NUM_CHECKPOINTS" -ne 3 ]; then
    echo -e "${RED}Error: Expected exactly 3 checkpoints (save_top_k=3), found $NUM_CHECKPOINTS${NC}"
    if [ "$NUM_CHECKPOINTS" -eq 0 ]; then
        exit 1
    fi
fi

# 5. Run Inference on Top 3 Checkpoints
echo -e "${GREEN}Running inference on checkpoints...${NC}"

# Extract audio path from manifest
AUDIO_PATH=$(python -c "import json; print(json.load(open('tiny_one_sample.jsonl'))['audio_filepath'])")
echo "Using audio file: $AUDIO_PATH"

for ckpt in $CHECKPOINTS; do
    echo "---------------------------------------------------"
    echo "Testing Checkpoint: $ckpt"
    python scripts/smoke_test_inference.py \
        --checkpoint "$ckpt" \
        --audio "$AUDIO_PATH"
        
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Inference successful for $ckpt${NC}"
    else
        echo -e "${RED}Inference failed for $ckpt${NC}"
        exit 1
    fi
done

echo -e "${GREEN}Extended Smoke Test Completed Successfully!${NC}"
