#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Robust 1-Epoch Smoke Test...${NC}"

# 1. Cleanup previous runs
BASE_DIR="/workspace/amchi_asr/results/smoke_test_robust"
echo "Cleaning up previous experiment directory: $BASE_DIR"
rm -rf "$BASE_DIR"

# 2. Run Training (1 epoch)
echo -e "${GREEN}Running 1-epoch training...${NC}"
# Create a temporary config for 1 epoch
cat > configs/tmp_smoke_1epoch.yaml <<EOC
# Temporary 1-epoch smoke test
model:
  nemo_model: "models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo"
  tokenizer:
    dir: "tokenizers"
    model_path: "tokenizers/marathi_tokenizer.model"
    type: "bpe"
  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    sample_rate: 16000
    normalize: "per_feature"
    window_size: 0.025
    window_stride: 0.01
    window: "hann"
    features: 80
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001
    pad_to: 16
    stft_conv: false
  encoder:
    _target_: nemo.collections.asr.modules.ConformerEncoder
    feat_in: 80
    feat_out: -1
    n_layers: 18
    d_model: 512
  loss:
    loss_name: "default"
  decoder_type: "ctc"

data:
  train_ds:
    manifest_filepath: "tiny_one_sample.jsonl"
    sample_rate: 16000
    batch_size: 1
    shuffle: true
    num_workers: 1
    pin_memory: true
    max_duration: 30.0
    min_duration: 0.1
    trim_silence: true
    load_audio: true
    use_start_end_token: false
  validation_ds:
    manifest_filepath: "tiny_one_sample.jsonl"
    sample_rate: 16000
    batch_size: 1
    shuffle: false
    num_workers: 1
    pin_memory: true
    max_duration: 30.0
    min_duration: 0.1
    trim_silence: true
    load_audio: true
    use_start_end_token: false
  test_ds:
    manifest_filepath: "tiny_one_sample.jsonl"
    sample_rate: 16000
    batch_size: 1
    shuffle: false
    num_workers: 1
    pin_memory: true
    max_duration: 30.0
    min_duration: 0.1
    trim_silence: true
    load_audio: true
    use_start_end_token: false

trainer:
  devices: 1
  accelerator: "gpu"
  max_epochs: 1
  max_steps: -1
  num_nodes: 1
  accumulate_grad_batches: 1
  enable_checkpointing: true
  logger: false
  log_every_n_steps: 1
  check_val_every_n_epoch: 1
  strategy: null

optim:
  name: adamw
  lr: 0.001
  weight_decay: 0.001

exp_manager:
  exp_dir: "$BASE_DIR"
  name: "smoke_test_robust"
  create_tensorboard_logger: false
  create_checkpoint_callback: true
  checkpoint_callback_params:
    monitor: "val_loss"
    mode: "min"
    save_top_k: 1
EOC

python3 scripts/fine_tune.py --config "configs/tmp_smoke_1epoch.yaml" --output_dir "$BASE_DIR"

# 3. Run Inference Smoke Test
echo "Running Inference Smoke Test on a dev sample..."
# Find the best checkpoint
CHECKPOINT=$(find "$BASE_DIR/checkpoints" -name "*.ckpt" ! -name "last.ckpt" | head -n 1)
if [ -z "$CHECKPOINT" ]; then
    CHECKPOINT=$(find "$BASE_DIR/checkpoints" -name "last.ckpt" | head -n 1)
fi

if [ -z "$CHECKPOINT" ]; then
    echo -e "${RED}Error: No checkpoint found in $BASE_DIR/checkpoints${NC}"
    exit 1
fi

TEST_AUDIO="data/dev/audio/570.wav"

python3 scripts/smoke_test_inference.py \
    --checkpoint "$CHECKPOINT" \
    --audio "$TEST_AUDIO"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Robust smoke test passed!${NC}"
else
    echo -e "${RED}❌ Robust smoke test failed.${NC}"
    exit 1
fi
