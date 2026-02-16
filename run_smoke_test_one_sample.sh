#!/usr/bin/env bash
# One-sample smoke test: train/dev/test on the same single sample for 5 epochs.
# Success: validation loss and CER must both improve (decrease) over epochs; otherwise exit 1.
# Output: results/smoke_tests/

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${GREEN}=== One-sample smoke test (5 epochs, same sample for train/dev/test) ===${NC}"

# Ensure manifest and config exist
SMOKE_MANIFEST="results/smoke_tests/smoke_one_sample.jsonl"
if [ ! -f "$SMOKE_MANIFEST" ]; then
    echo "Creating $SMOKE_MANIFEST (single sample from Amchi train)..."
    mkdir -p results/smoke_tests
    # Use first line from amchi train manifest if available
    if [ -f "data/amchi/train/manifest.jsonl" ]; then
        head -1 "data/amchi/train/manifest.jsonl" > "$SMOKE_MANIFEST"
    else
        echo -e "${RED}Error: data/amchi/train/manifest.jsonl not found. Run data download first.${NC}" >&2
        exit 1
    fi
fi

CONFIG="configs/smoke_tests_5epoch.yaml"
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: $CONFIG not found.${NC}" >&2
    exit 1
fi

# Run 5-epoch fine-tuning (output to results/smoke_tests)
echo -e "${YELLOW}Running 5-epoch training...${NC}"
export APPLY_CONV_PATCH="${APPLY_CONV_PATCH:-1}"
python3 scripts/fine_tune.py --config "$CONFIG" --output_dir results/smoke_tests

# Validate: CER and val_loss must improve
echo -e "${YELLOW}Validating: CER and validation loss must improve over epochs...${NC}"
if python3 scripts/validate_smoke_one_sample.py; then
    echo -e "${GREEN}✅ One-sample smoke test PASSED.${NC}"
    exit 0
else
    echo -e "${RED}❌ One-sample smoke test FAILED (CER or val_loss did not improve).${NC}" >&2
    exit 1
fi
