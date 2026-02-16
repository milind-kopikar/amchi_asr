#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   AMCHI ASR: MASTER PREFLIGHT CHECK & SMOKE TEST SUITE   ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "User: $(whoami)"
echo "PWD:  $(pwd)"
echo ""

# 0. GPU check first — do not run heavy tests on CPU
run_step() {
    local step_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}>>> Running Step: $step_name${NC}"
    echo "Command: $command"
    
    if eval "$command"; then
        echo -e "${GREEN}>>> Step '$step_name' PASSED${NC}\n"
    else
        echo -e "${RED}>>> Step '$step_name' FAILED${NC}\n"
        exit 1
    fi
}

run_step "Check GPU (CUDA visible and available)" "python3 scripts/check_gpu.py"

# Function to run a step and check status
run_step() {
    local step_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}>>> Running Step: $step_name${NC}"
    echo "Command: $command"
    
    if eval "$command"; then
        echo -e "${GREEN}>>> Step '$step_name' PASSED${NC}\n"
    else
        echo -e "${RED}>>> Step '$step_name' FAILED${NC}\n"
        exit 1
    fi
}

# 1. Environment & Library Checks
run_step "Check Python Libraries" "python3 check_libs.py"

# 2. Data & Model Existence Checks
run_step "Check Data Files" "python3 check_data.py"
run_step "Check Model Files" "python3 check_model.py"

# 3. Audio Properties Check
# Ensure audio is 16kHz mono as expected
run_step "Check Audio Properties" "python3 check_audio_properties.py"

# 4. Basic 1-Epoch Smoke Test (Functional Check)
# This verifies the pipeline runs from start to finish without crashing
run_step "1-Epoch Functional Smoke Test" "./scripts/robust_smoke_test.sh"

# 5. Extended 5-Epoch Smoke Test (Learning & Checkpointing Check)
# This verifies that loss decreases and top-k checkpointing works
run_step "5-Epoch Learning & Checkpointing Test" "./scripts/extended_smoke_test.sh"

# 6. One-Sample Smoke Test (5 epochs, same sample train/dev/test)
# Success: validation loss and CER must both improve over epochs (see results/smoke_tests/README.md)
run_step "One-Sample Smoke Test (CER + val_loss improvement)" "./scripts/run_smoke_test_one_sample.sh"

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}   ALL PREFLIGHT CHECKS & SMOKE TESTS PASSED SUCCESSFULLY   ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo "You are now ready to proceed with full-scale training."
