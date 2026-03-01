# Amchi Konkani ASR — Fine-Tuning Guide
## Build Guide for Claude Agent (Next Session)

**Read this first. This is the complete guide for fine-tuning the Amchi Konkani ASR model on a fresh RunPod session.**

---

## 0. What this track is and how it differs from deaf speech

| | Deaf Speech (done) | Amchi Konkani (this guide) |
|--|-------------------|---------------------------|
| **Speakers** | Deaf speakers (atypical Marathi) | Normal Konkani speakers |
| **Data source** | `deafspeechcollector-production.up.railway.app` | `konkanicollector-production.up.railway.app` |
| **Data size** | 124 recordings (1 story) | ~548 recordings (5 stories) |
| **Data split** | Same 124 for train/dev/test (overfit test) | Stories 1,2,3 → train / Story 4 → dev / Story 5 → test |
| **Base model** | Marathi IndicConformer | Marathi IndicConformer (same) |
| **Tokenizer** | `tokenizers/marathi_tokenizer.model` (1024 tokens) | Same |
| **Training config** | `configs/deaf_speech_story4_50epoch.yaml` | `configs/amchi_konkani_50epoch.yaml` ← already in repo |
| **Best result so far** | val_WER 72% (epoch 21) | val_WER 65.4%, test_WER 35.1% (20-epoch pilot, Jan 2026) |
| **Target** | Demonstrate hearing of deaf speech | Better than 35.1% test WER with 50 epochs |
| **Post-processing** | Gemini FILL/RECONSTRUCT (Marathi trusted words) | TBD — ask user before building |

**Goal for this session:** Run a 50-epoch fine-tuning run on all available Amchi Konkani data (stories 1–5), beat the existing 35.1% test WER baseline.

---

## 1. Current status

| Item | Status |
|------|--------|
| Repo + scripts | ✅ Ready (`scripts/fine_tune.py`, `scripts/download_data_from_railway.py`) |
| Training config | ✅ Ready (`configs/amchi_konkani_50epoch.yaml`) |
| Base model (Marathi) | ✅ On RunPod persistent storage (or re-download from HF — see §4) |
| Marathi tokenizer | ✅ In git (`tokenizers/marathi_tokenizer.model`) |
| conv_asr patch | ✅ In git (`patches/conv_asr_fixed.py`) |
| Amchi Konkani audio data | ❌ Not in git — download from Railway (§3) |
| Data manifests | ❌ Not in git — generated during download (§3) |
| Training run | ❌ Not started |
| Post-processing for Konkani | ❌ Design TBD — ask user before building (§8) |

---

## 2. Credentials the agent needs to ask for

| Credential | Where to find it | Used in step |
|------------|-----------------|--------------|
| **HuggingFace token** (`hf_…`) | huggingface.co → Settings → Access Tokens | §4 (model download, if model missing) |
| **GitHub PAT** (`ghp_…`) | github.com → Settings → Developer settings → Tokens | `git push` at end of session |
| **Gemini API key** (`AIzaSy…`) | Google AI Studio | §8 (post-processing, if built this session) |

All go in `.env` on the pod. **Never commit `.env` to git.**

```bash
# On the RunPod pod, create .env:
cat > /workspace/.env << 'EOF'
HF_TOKEN=hf_...
GITHUB_PAT=ghp_...
GEMINI_API_KEY=AIzaSy...
EOF
source /workspace/.env
```

---

## 3. RunPod pod setup

### 3.1 Start a RunPod pod

**Recommended spec:**
- GPU: RTX 4000 Ada (20GB) or A40 — same as deaf speech session
- Disk: 50GB container volume (the `.nemo` model is ~500MB, audio data ~2GB)
- Template: RunPod PyTorch or any Ubuntu 22.04 + CUDA 11.8

### 3.2 Clone repo and install NeMo

```bash
cd /workspace

# Clone (or pull if repo already exists)
git clone https://github.com/milind-kopikar/amchi_asr.git
# OR: cd amchi_asr && git pull

cd amchi_asr

# Install NeMo ASR (Python 3.11, upstream — NOT AI4Bharat fork)
# Use [asr] only — NOT [all] — to keep disk usage under 20GB
pip install "nemo_toolkit[asr]" --ignore-installed blinker -q

# Apply the conv_asr patch (REQUIRED for hybrid CTC/RNNT loading)
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)")
cp /workspace/amchi_asr/patches/conv_asr_fixed.py "$NEMO_FILE"
echo "Patch applied to $NEMO_FILE"

# Verify GPU
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## 4. Download base model (if not already on pod)

The Marathi IndicConformer model should be at:
```
models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo
```

Check first:
```bash
ls -lh models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/*.nemo 2>/dev/null \
  && echo "Model already present" \
  || echo "Model missing — need to download"
```

If missing, download from HuggingFace:
```bash
source .env
huggingface-cli login --token "$HF_TOKEN"

python3 scripts/download_model_from_hf.py \
  --repo ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
  --outdir models

# Verify
ls -lh models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/*.nemo
```

Also verify the tokenizer is present (it's in git so should always be there):
```bash
ls -lh tokenizers/marathi_tokenizer.model
# Expected: ~360KB SentencePiece model (1024 tokens)
```

---

## 5. Download Amchi Konkani data from Railway

The Amchi Konkani audio data is NOT in git (too large). Download it fresh each session.

**Important:** Use `--use_story_split` to get the correct train/dev/test split:
- Stories 1, 2, 3 → `data/train/` (~472 samples)
- Story 4 → `data/dev/` (~38 samples)
- Story 5 → `data/test/` (~38 samples)

```bash
cd /workspace/amchi_asr

python3 scripts/download_data_from_railway.py \
  --base_url https://konkanicollector-production.up.railway.app \
  --output_dir data \
  --use_story_split

# Verify manifests were created
echo "Train samples: $(wc -l < data/train/manifest.jsonl)"
echo "Dev samples:   $(wc -l < data/dev/manifest.jsonl)"
echo "Test samples:  $(wc -l < data/test/manifest.jsonl)"
```

Expected counts (may be higher if new recordings were added since Jan 2026):
```
Train samples: 472
Dev samples:   38
Test samples:  38
```

Also verify audio was downloaded:
```bash
ls data/train/audio/ | wc -l   # should match train count
ls data/dev/audio/ | wc -l
ls data/test/audio/ | wc -l
```

---

## 6. Run fine-tuning (50 epochs)

### 6.1 Set environment variables

```bash
export APPLY_CONV_PATCH=1   # applies the conv_asr patch at runtime (belt + suspenders)
```

### 6.2 Optional: 1-epoch smoke test first

Run this to catch any config or data errors before the full 50-epoch run:

```bash
python3 scripts/fine_tune.py \
  --config configs/amchi_konkani_50epoch.yaml \
  --trainer.max_epochs 1 \
  --exp_manager.name amchi_konkani_smoke_1epoch
```

Expected: runs 1 epoch (~2 minutes), val_WER logged, no crashes.

### 6.3 Full 50-epoch training run

```bash
python3 scripts/fine_tune.py \
  --config configs/amchi_konkani_50epoch.yaml
```

**Expected training time:** ~45–90 minutes on RTX 4000 Ada (472 train samples, batch_size=4, 50 epochs ≈ 5,900 steps)

**What to watch:**
- val_WER should start around 60–70% and drop steadily
- Target: below 35.1% (beat the 20-epoch pilot)
- Best checkpoint is saved automatically (monitor: `val_wer`, save_top_k=3)
- Checkpoints go to `/workspace/nemo_experiments/amchi_konkani_50epoch/checkpoints/`

**Expected log output:**
```
Epoch 1/50: val_wer=0.62 val_loss=...
Epoch 5/50: val_wer=0.54 ...
Epoch 20/50: val_wer=0.38 ...   ← previous 20-epoch result was 0.351
Epoch 50/50: val_wer=0.XX ...   ← target < 0.35
```

---

## 7. Evaluate on test set

After training completes, run the test evaluation:

```bash
# Find the best checkpoint (lowest val_wer in filename)
ls /workspace/nemo_experiments/amchi_konkani_50epoch/checkpoints/*.ckpt

# Run test evaluation (replace epoch=XX-val_wer=Y.YYY with actual best checkpoint name)
python3 scripts/fine_tune.py \
  --config configs/amchi_konkani_50epoch.yaml \
  --trainer.max_epochs 0 \
  --exp_manager.name amchi_konkani_test_eval \
  --model.init_from_nemo_model "/workspace/nemo_experiments/amchi_konkani_50epoch/checkpoints/amchi_konkani_50epoch-epoch=XX-val_wer=Y.YYY.ckpt"
```

> **Note:** If the fine_tune.py script doesn't have a standalone test eval mode, use the inference smoke test approach from REPRODUCTION_NOTES.md §9 — load the checkpoint with config-patch + strict=False and run `model.transcribe()` on the test manifest.

Save results to:
```
nemo_experiments/amchi_konkani_50epoch/experiments/<timestamp>/final_test_results.json
```

---

## 8. Post-processing for Amchi Konkani (DISCUSS WITH USER FIRST)

**Do NOT build post-processing without asking the user.** The deaf speech post-processing was designed for garbled Marathi output from deaf speech. Konkani is different.

**Ask the user which approach they want:**

**Option A: No post-processing**
- The Konkani model already achieves ~35% WER (much better than deaf speech's 75%)
- At this WER, transcriptions are partially readable without correction
- Simpler, no Gemini API cost
- Best for: demo where raw ASR is the point

**Option B: Gemini cleanup (simpler than deaf speech)**
- No FILL/RECONSTRUCT modes — just ask Gemini to "clean up" the transcription
- Prompt: "The following is an ASR transcription of Konkani speech. Clean up any obvious errors while keeping the meaning. Output only the corrected Konkani text: {asr_output}"
- No trusted-word classification needed (WER is already decent)
- Best for: improving readability of partial transcriptions

**Option C: Reuse deaf speech post-processor with Konkani trusted words**
- Extend `postprocess_asr.py` with a Konkani `TRUSTED_WORDS` set
- Replace Marathi question words with Konkani equivalents
- More complex but consistent with existing architecture
- Best for: if the Konkani model still produces `⁇` markers frequently

**After discussing with user**, build the chosen approach as a new function in `scripts/postprocess_asr.py` or a new `scripts/postprocess_konkani.py`.

---

## 9. Compare results and commit

### 9.1 Results to commit

```bash
cd /workspace/amchi_asr

# Stage experiment results (NOT checkpoints — too large)
git add nemo_experiments/amchi_konkani_50epoch/experiments/*/
git add nemo_experiments/amchi_konkani_50epoch/experiments/*/epoch_metrics.csv
git add nemo_experiments/amchi_konkani_50epoch/experiments/*/final_test_results.json
git add nemo_experiments/amchi_konkani_50epoch/experiments/*/samples_epoch_*.json

# Don't add:
# - nemo_experiments/**/checkpoints/*.ckpt  (too large, in .gitignore)
# - data/  (audio files, in .gitignore)

git commit -m "Add Amchi Konkani 50-epoch training results (WER: XX.X% test)"
```

### 9.2 Push to GitHub

```bash
source .env
git remote set-url origin https://${GITHUB_PAT}@github.com/milind-kopikar/amchi_asr.git
git push
```

### 9.3 Upload best checkpoint to R2 (for future serverless endpoint)

```bash
pip install boto3 -q
export R2_ACCOUNT_ID="c90f9011c5a59d5bf40c808f40e3e34b"
export R2_ACCESS_KEY_ID="..."   # ask user for current R2 credentials
export R2_SECRET_ACCESS_KEY="..."
export R2_BUCKET_NAME="asr-checkpoints"

# Upload the best checkpoint (replace XX and Y.YYY with actual values)
python3 scripts/upload_checkpoint_to_r2.py \
  --file "nemo_experiments/amchi_konkani_50epoch/checkpoints/amchi_konkani_50epoch-epoch=XX-val_wer=Y.YYY.ckpt" \
  --public-url
```

The R2 bucket and public base URL are the same as the deaf speech checkpoint:
```
https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/<object-key>
```

---

## 10. Key differences from deaf speech (important gotchas)

### Data source URL is different
- Deaf speech: `https://deafspeechcollector-production.up.railway.app`
- Amchi Konkani: `https://konkanicollector-production.up.railway.app`

Do NOT mix these up when calling `download_data_from_railway.py`.

### Data split is different
- Deaf speech: Same 124 samples for train/dev/test (intentional overfit test)
- Amchi Konkani: Proper split by story ID. **Always use `--use_story_split`** flag.
  - Story 4 → dev (NOT test)
  - Story 5 → test (NOT dev)
  - Swapping Story 4 and Story 5 was a previous mistake — see DATA_SNAPSHOT_AMCHI_KONKANI.md.

### Data manifests not in git
- Deaf speech manifests ARE in git (explicit .gitignore exceptions for `data/deaf_speech/*/manifest.jsonl`)
- Amchi Konkani manifests are NOT in git (`data/train/`, `data/dev/`, `data/test/` are excluded)
- Must re-download from Railway each session

### Model loading for inference (same pattern as deaf speech)
The same config-patch + strict=False pattern from REPRODUCTION_NOTES.md §9 applies to Konkani too. The existing `scripts/deaf_speech_inference.py` can be adapted:
- Change `CHECKPOINT_PATH` to the Konkani checkpoint
- The `load_model()` function is identical
- Post-processing will differ (see §8)

### ⁇ markers in output
NeMo outputs `⁇` for any token it can't decode. The frequency of `⁇` in Konkani output depends on how well the model has learned the vocabulary. At 35% WER the model is producing mostly intelligible output, so `⁇` should be rarer than in deaf speech.

---

## 11. Quick reference

| Item | Value |
|------|-------|
| Training config | `configs/amchi_konkani_50epoch.yaml` |
| Training script | `scripts/fine_tune.py` |
| Data download | `scripts/download_data_from_railway.py` |
| Railway URL | `https://konkanicollector-production.up.railway.app` |
| Data split flag | `--use_story_split` |
| Base model | `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo` |
| Tokenizer | `tokenizers/marathi_tokenizer.model` |
| conv_asr patch | `patches/conv_asr_fixed.py` |
| Training env var | `export APPLY_CONV_PATCH=1` |
| Checkpoint output | `/workspace/nemo_experiments/amchi_konkani_50epoch/checkpoints/` |
| Baseline to beat | val_WER 65.4%, test_WER 35.1% (20-epoch pilot, Jan 2026) |
| R2 bucket | `asr-checkpoints` (public base URL: `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev`) |

---

## 12. Key docs to read before starting

| Doc | What it covers |
|-----|---------------|
| `AGENT_HANDOFF.md` | Full project context and session history |
| `REPRODUCTION_NOTES.md` §9 | Inference loading pattern (config-patch + strict=False) |
| `LEARNINGS.md` | Hard-won lessons: NeMo version, disk space, pip cache |
| `DATA_SNAPSHOT_AMCHI_KONKANI.md` | Exact data split from Jan 2026 pilot |
| `MASTER_REPRODUCTION_GUIDE.md` | Environment setup from scratch |
| `configs/amchi_konkani_50epoch.yaml` | The training config (ready to use) |
