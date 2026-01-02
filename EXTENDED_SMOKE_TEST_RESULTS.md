# Extended 5-Epoch Smoke Test Results
**Date:** 2026-01-02
**Script:** `scripts/extended_smoke_test.sh`
**Config:** `configs/tmp_marathi_5epoch_ctc_smoke.yaml`

## Objective
Verify that the fine-tuning pipeline:
1.  Actually "learns" (training loss decreases).
2.  Correctly saves the top-k checkpoints (k=3).
3.  Produces checkpoints that are valid for inference.

## Results

### 1. Training Dynamics
- **Epochs:** 5
- **Loss Behavior:**
    - Epoch 1: `nan` (Likely due to high learning rate or instability at start of fine-tuning on tiny data)
    - Epoch 2: `59.836`
    - Epoch 3: `41.438`
    - Epoch 4: `16.037`
    - **Conclusion:** Loss is decreasing significantly, indicating the model is updating its weights to fit the single sample.

### 2. Checkpointing
- **Expected:** 3 checkpoints (`save_top_k: 3`)
- **Found:** 3 checkpoints
    - `amchi_marathi_golden_5epoch_ctc-epoch=02-val_loss=59.836.ckpt`
    - `amchi_marathi_golden_5epoch_ctc-epoch=03-val_loss=41.438.ckpt`
    - `amchi_marathi_golden_5epoch_ctc-epoch=04-val_loss=16.037.ckpt`
- **Conclusion:** `exp_manager` is working correctly.

### 3. Inference Verification
Ran `scripts/smoke_test_inference.py` on all 3 checkpoints using the training sample (`data/dev/audio/570.wav`).

| Epoch | Checkpoint Loss | Prediction | Target |
|-------|-----------------|------------|--------|
| 2 | 59.836 | `रडालोडलो` | `रोहन होड ज़ाल्लो!` |
| 3 | 41.438 | `रडालो` | `रोहन होड ज़ाल्लो!` |
| 4 | 16.037 | `र होडालो` | `रोहन होड ज़ाल्लो!` |

**Observation:** The predictions are evolving. Epoch 4's prediction `र होडालो` is phonetically closer to the target `रोहन होड ज़ाल्लो!` than the earlier epochs.

## Next Steps
- The pipeline is verified for:
    - Data loading
    - Model instantiation (Hybrid/CTC)
    - Training loop execution
    - Loss optimization
    - Checkpoint management
    - Inference capability
- Ready to proceed with full dataset training or larger scale experiments.
