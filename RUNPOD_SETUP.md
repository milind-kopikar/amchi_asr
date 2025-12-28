# RunPod Setup Guide for Konkani ASR

This guide will help you set up and run the Konkani ASR fine-tuning on RunPod GPU instances.

## Prerequisites

- RunPod account with GPU pod (recommended: RTX 4090, A100, or T4)
- GitHub repository access
- Railway deployment URL for data download

## Step 1: Start RunPod Pod

1. Go to [RunPod.io](https://runpod.io)
2. Create a new pod:
   - **Template**: PyTorch 2.0+ (or RunPod PyTorch)
   - **GPU**: RTX 4090 / A100 / A6000 (recommended)
   - **Container Disk**: 50GB minimum
   - **Volume**: Optional (for persistent storage)

3. Once pod is running, click "Connect" → "Start Jupyter Lab" or "Start SSH"

## Step 2: Clone Repository

```bash
# Via HTTPS (recommended for RunPod)
cd ~
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr/konkani_asr

# Or via SSH (if you have SSH key configured)
git clone git@github.com:milind-kopikar/amchi_asr.git
cd amchi_asr/konkani_asr
```

## Step 3: Run Setup Script

```bash
# Make setup script executable
chmod +x setup_runpod.sh

# Run setup (creates venv, installs dependencies)
./setup_runpod.sh

# Activate virtual environment
source venv/bin/activate
```

## Step 4: Verify GPU Access

```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output:
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4090 (or your GPU model)
```

## Step 5: Download AI4Bharat Marathi Model

```bash
# Download the base model from Hugging Face
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large \
    --output_path models/indicconformer_mr
```

## Step 6: Download Konkani Audio Data

```bash
# Download recordings from Railway deployment
python scripts/download_data_from_railway.py \
    --base_url https://konkanicollector-production.up.railway.app \
    --output_dir data/train \
    --train_split 0.8
```

This will:
- Fetch all approved recordings
- Download audio files to `data/train/audio/` and `data/dev/audio/`
- Create NeMo-compatible manifest files

## Step 7: Verify Data

```bash
# Check downloaded files
ls -lh data/train/audio/ | head
cat data/train/manifest.jsonl | head -3

# Count recordings
echo "Train: $(wc -l < data/train/manifest.jsonl) recordings"
echo "Dev: $(wc -l < data/dev/manifest.jsonl) recordings"
```

## Step 8: Start Fine-tuning

```bash
# Run fine-tuning with default config
python scripts/fine_tune.py \
    --config configs/konkani_finetune.yaml

# Or with custom parameters
python scripts/fine_tune.py \
    --config configs/konkani_finetune.yaml \
    --trainer.max_epochs 30 \
    --data.train_ds.batch_size 16
```

## Monitor Training

### Option 1: TensorBoard (in RunPod)

```bash
# Start TensorBoard
tensorboard --logdir results/ --port 6006

# In RunPod, click "Connect" → "HTTP Service" → Port 6006
```

### Option 2: Watch Logs

```bash
# Real-time log monitoring
tail -f results/konkani_asr_finetune/lightning_logs/version_0/metrics.csv
```

## Troubleshooting

### Out of Memory Error

```bash
# Reduce batch size in config
sed -i 's/batch_size: 8/batch_size: 4/g' configs/konkani_finetune.yaml
```

### Model Download Failed

```bash
# Set Hugging Face token if needed
export HF_TOKEN="your_token_here"
python scripts/download_model.py --auth_token $HF_TOKEN
```

### CUDA Not Available

```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### "Error: Your SSH client doesn't support PTY"

- **Symptom**: VS Code Remote-SSH installer fails or an `ssh <host> 'tty'` command prints: `Error: Your SSH client doesn't support PTY`.
- **Cause**: Often seen when connecting via RunPod's proxy (`ssh.runpod.io`), which doesn't support allocating a PTY; VS Code installer requires a PTY to run installer commands and detect platform.
- **Workarounds**:
  - Use the pod's public IP and port (direct TCP endpoint) instead of the proxy host — this reliably supports PTY allocation.
  - Verify PTY: `ssh -vvv -tt <host> 'tty'` should return `/dev/pts/0` on success.
  - Ensure your SSH config points to your private key (e.g., `~/.ssh/runpod_ed25519`) for the host.

- **Automated helper**: use `python scripts/setup_runpod_remote.py --host <ip> --port <port> --user runpod --pubkey ~/.ssh/runpod_ed25519.pub --identity ~/.ssh/runpod_ed25519 --workspace konkani_asr.code-workspace --host-alias runpod-large` to add a host entry, try to copy your public key, run the PTY test, and update the workspace settings to show installer logs.

- **VS Code tips**: add `"remote.SSH.remotePlatform": { "runpod-large": "linux" }` and `"remote.SSH.showLoginTerminal": true` to your `.code-workspace` file to force Linux detection and capture installer output for debugging.

## After Training

### Download Trained Model

```bash
# Find best checkpoint
ls -lht results/konkani_asr_finetune/checkpoints/

# Download via RunPod Web Terminal or SCP
# In local machine:
scp -P <pod_ssh_port> root@<pod_ip>:~/amchi_asr/konkani_asr/results/konkani_asr_finetune/checkpoints/*.ckpt ./
```

### Test Inference

```bash
# Test with a sample audio file
python scripts/test_inference.py \
    --checkpoint results/konkani_asr_finetune/checkpoints/best.ckpt \
    --audio data/dev/audio/5.wav
```

## Cost Optimization

- **Stop pod when not training** - RunPod charges by the hour
- **Use spot instances** - 3-5x cheaper than on-demand
- **Download checkpoints regularly** - In case pod gets terminated
- **Use persistent storage** - For model files that don't change

## Estimated Training Time

With 30 minutes of audio (~1800 recordings):

| GPU | Batch Size | Time/Epoch | Total (50 epochs) |
|-----|-----------|-----------|------------------|
| RTX 4090 | 16 | ~5 min | ~4 hours |
| A100 | 32 | ~3 min | ~2.5 hours |
| T4 | 8 | ~12 min | ~10 hours |

## Next Steps

Once training completes:
1. Evaluate model on test set
2. Export model for deployment
3. Integrate with konkani_collector web interface
4. Compare with baseline Marathi model

## Questions?

Check the main README.md or open an issue on GitHub.
