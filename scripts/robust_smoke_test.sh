
# 5. Run Inference Smoke Test
echo "Running Inference Smoke Test on a dev sample..."
# Find the best checkpoint (assuming only one or taking the first one found)
CHECKPOINT=$(find results/smoke_test_robust/checkpoints -name "*.ckpt" ! -name "last.ckpt" | head -n 1)
# Fallback to last.ckpt if no other
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(find results/smoke_test_robust/checkpoints -name "last.ckpt" | head -n 1)
fi

TEST_AUDIO="data/dev/audio/417.wav"

if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}Error: No checkpoint found in results/smoke_test_robust/checkpoints${NC}"
    exit 1
fi

if [ ! -f "$TEST_AUDIO" ]; then
    echo -e "${RED}Error: Test audio not found at $TEST_AUDIO${NC}"
    # Fallback to any wav file if specific one missing
    TEST_AUDIO=$(find data/dev/audio -name "*.wav" | head -n 1)
    if [ -z "$TEST_AUDIO" ]; then
         echo "No audio files found in data/dev/audio. Skipping inference test."
         exit 0
    fi
    echo "Using fallback audio: $TEST_AUDIO"
fi

python3 scripts/smoke_test_inference.py \
    --checkpoint "$CHECKPOINT" \
    --audio "$TEST_AUDIO"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Inference smoke test passed!${NC}"
else
    echo -e "${RED}❌ Inference smoke test failed.${NC}"
    exit 1
fi
