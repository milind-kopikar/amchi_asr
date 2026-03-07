# Module: Inference — Loading a Checkpoint and Transcribing Audio

**Self-contained guide.** Read this to run inference with any trained checkpoint.
For which checkpoint to use, see `docs/CHECKPOINTS_REGISTRY.md`.

---

## What this module does

Loads a fine-tuned `.ckpt` file, transcribes WAV audio using the CTC decoder,
and optionally runs Gemini post-processing to clean up garbled output.

---

## Key files

| File | Purpose |
|---|---|
| `scripts/deaf_speech_inference.py` | End-to-end inference: load checkpoint → ASR → Gemini post-process → print result |
| `scripts/postprocess_asr.py` | Gemini post-processing module (can be used standalone) |
| `scripts/evaluate.py` | Batch evaluation on a manifest; outputs per-sample WER JSON |

---

## Quick start

```bash
# Set your Gemini API key (or pass --gemini_key directly)
source .env

python3 scripts/deaf_speech_inference.py \
  --checkpoint results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt \
  --audio data/deaf_speech/audio/131.wav \
  --gemini_key "$GEMINI_API_KEY"
```

Expected output:
```
Raw ASR   : किती आहे ⁇
Corrected : हे किती आहे?  [FILL]
Latency   : ASR 0.3s | Post-process 1.5s | Total 1.8s
```

---

## CRITICAL: How to load a checkpoint (the non-obvious pattern)

The hybrid CTC/RNNT model **cannot** be loaded with `model.load_from_checkpoint()` because:
1. The config saved inside the checkpoint has `loss_name: ctc` — which the RNNT validator rejects
2. The dataset paths in the checkpoint config don't exist in the inference environment

**The correct loading pattern** (from `REPRODUCTION_NOTES.md §9`):

```python
import torch
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

ckpt_path = "results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt"

# Step 1: Load raw checkpoint dict (don't use PyTorch Lightning's load_from_checkpoint)
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
cfg = ckpt['hyper_parameters']['cfg']

# Step 2: Patch config — remove problematic fields
cfg.loss.loss_name = 'default'
del cfg.train_ds, cfg.validation_ds, cfg.test_ds

# Step 3: Instantiate model from patched config (no trainer needed for inference)
model = EncDecHybridRNNTCTCBPEModel(cfg=cfg, trainer=None)

# Step 4: Add back empty dataset configs (transcribe() requires them to exist)
cfg.validation_ds = OmegaConf.create({})
cfg.test_ds = OmegaConf.create({})

# Step 5: Load weights (strict=False because some RNNT params not in CTC checkpoint)
model.load_state_dict(ckpt['state_dict'], strict=False)

# Step 6: Switch to CTC decoding (faster, more accurate for our use case)
model.change_decoding_strategy(decoder_type='ctc')
model.eval()

# Step 7: Transcribe
results = model.transcribe(audio=['path/to/audio.wav'])
raw_text = results[0]  # may contain ⁇ markers
```

---

## Audio requirements

- Format: WAV, 16kHz, mono (16-bit PCM)
- Duration: up to 30 seconds (configurable in config `max_duration`)
- The `⁇` marker in output means "token the model could not decode" — strip before display

---

## Post-processing with Gemini

`scripts/postprocess_asr.py` classifies each word in the ASR output and calls Gemini
to fill gaps or reconstruct heavily garbled predictions.

```bash
# Batch post-processing on a results JSON file
python3 scripts/postprocess_asr.py \
  --input results/experiments/deaf_speech_dsd/final_test_results.json \
  --output /tmp/postprocessed.json \
  --report /tmp/report.txt
  # (reads GEMINI_API_KEY from environment)
```

**Modes:**
- `SKIP` — WER=0, just strip `⁇`
- `FILL` — ≥1 trusted anchor word present; Gemini fills in `[___]` slots
- `RECONSTRUCT` — no trusted words; Gemini reconstructs from phonetic fragments

**Safety valve:** If Gemini's output has higher WER than the original, the script
reverts to the stripped original. This prevents Gemini from corrupting good predictions.

**Model to use:** `gemini-2.5-flash` (gemini-2.0-flash is deprecated for new users).
**Package to use:** `google-genai` (the older `google-generativeai` is deprecated).

---

## Downloading a checkpoint from R2 (for serverless / local inference)

```bash
pip install boto3

python3 - << 'EOF'
import boto3
from botocore.config import Config

s3 = boto3.client("s3",
    endpoint_url="https://c90f9011c5a59d5bf40c808f40e3e34b.r2.cloudflarestorage.com",
    aws_access_key_id="YOUR_ACCESS_KEY",
    aws_secret_access_key="YOUR_SECRET_KEY",
    config=Config(signature_version="s3v4"),
    region_name="auto",
)
s3.download_file(
    "asr-checkpoints",
    "results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt",
    "/local/path/to/save.ckpt"
)
EOF
```

Or simply use the public URL directly (no auth required):
```bash
wget "https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt"
```

See `docs/CHECKPOINTS_REGISTRY.md` for all public URLs.

---

## Latency expectations

| Step | Typical latency |
|---|---|
| Model load (first time) | 15–30s (downloads ~499MB base model config) |
| ASR transcription (1 utterance) | 0.2–0.5s |
| Gemini post-processing (1 utterance) | 1–2s (includes 0.5s rate-limit delay) |
| End-to-end per utterance | ~2s |
