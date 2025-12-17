# Model Download Guide - Two Approaches

## Quick Answer

You have **TWO options** for getting the base models:

### Option 1: Auto-Download During Training (Recommended) ⚡
NeMo downloads models automatically from Hugging Face when you call `.from_pretrained()`. **We need to update our code to use this.**

### Option 2: Manual Pre-Download (Current Code) 📥
Download models manually first, then reference local files. **This is what our current code expects.**

---

## 🔍 Current State of Our Code

**Right now, our code expects pre-downloaded models:**

```python
# nemo_train.py line 126
model_path = self.config.model.nemo_model  
# Points to: "models/indicconformer_mr/indicconformer_stt_mr_hybrid_ctc_rnnt_large.nemo"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Model file not found: {model_path}\n"
        f"Please download the model first using scripts/download_model.py"
    )

self.model = ModelClass.restore_from(model_path, strict=False)
```

**This means you MUST download models manually before training!**

---

## 📥 Option 1: Manual Download (Works Now)

### Step 1: Authenticate with Hugging Face

```bash
# On RunPod, login first
huggingface-cli login
# Paste your token when prompted
```

### Step 2: Download Models Using Our Script

```bash
# Download Marathi base model
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_mr

# Download Konkani base model (if testing both)
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_kok
```

### Step 3: Verify Downloaded Files

```bash
# Check Marathi model
ls -lh models/indicconformer_mr/

# Should see files like:
# - indicconformer_stt_mr_hybrid_ctc_rnnt_large.nemo (main model file)
# - tokenizer.model
# - tokenizer.vocab
# - config.yaml
```

### Where Are Files Stored?

```
konkani_asr/
├── models/
│   ├── indicconformer_mr/           # Marathi model
│   │   ├── indicconformer_stt_mr_hybrid_ctc_rnnt_large.nemo  (~500MB)
│   │   ├── tokenizer.model
│   │   ├── tokenizer.vocab
│   │   └── config.yaml
│   └── indicconformer_kok/          # Konkani model (if downloaded)
│       ├── indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo (~500MB)
│       ├── tokenizer.model
│       └── ...
```

### Step 4: Train as Usual

```bash
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi
```

---

## ⚡ Option 2: Auto-Download (Better, But Needs Code Changes)

**This is how NeMo is MEANT to work**, but we need to update our code.

### How It Works

Instead of downloading manually, NeMo downloads models on-the-fly:

```python
# NeMo downloads automatically from Hugging Face
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
)

# Model cached at: ~/.cache/huggingface/hub/
```

### Where Auto-Downloaded Models Go

```
~/.cache/huggingface/hub/
├── models--ai4bharat--indicconformer_stt_mr_hybrid_ctc_rnnt_large/
│   └── snapshots/
│       └── <commit_hash>/
│           ├── *.nemo
│           ├── tokenizer.model
│           └── ...
└── models--ai4bharat--indicconformer_stt_kok_hybrid_ctc_rnnt_large/
    └── snapshots/
        └── <commit_hash>/
            └── ...
```

**NeMo handles all caching automatically - you never need to worry about paths!**

### Why This Is Better

✅ **No manual download step** - training script handles everything
✅ **Automatic caching** - downloads only once, reuses cached models
✅ **Version control** - Hugging Face tracks model versions
✅ **Less disk space** - shared cache across projects
✅ **Cleaner code** - no hardcoded local paths

### What We Need to Change

We need to update `nemo_train.py` to use `.from_pretrained()` instead of `.restore_from()`:

```python
# CURRENT CODE (Manual download required):
model = ModelClass.restore_from("models/indicconformer_mr/model.nemo", strict=False)

# BETTER CODE (Auto-download):
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
)
```

---

## 🎯 Recommendation: What Should You Do?

### For First RunPod Session (Use Manual Download)

**Since our code currently expects manual download**, stick with Option 1 for now:

```bash
# 1. Login to Hugging Face
huggingface-cli login

# 2. Download model
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_mr

# 3. Run smoke tests
python scripts/download_smoke_test_data.py
python tests/test_e2e_pipeline.py --model marathi --test all

# 4. If tests pass, train
python scripts/download_data_from_railway.py
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi
```

### Future Improvement (Switch to Auto-Download)

After your first successful training, we can update the code to use auto-download for cleaner workflow.

---

## 🔍 How to Check Model Location

### If Using Manual Download:

```bash
# Check if model exists locally
ls -lh models/indicconformer_mr/*.nemo

# Get file size
du -sh models/indicconformer_mr/
```

### If Using Auto-Download:

```bash
# Check Hugging Face cache
ls -lh ~/.cache/huggingface/hub/

# Find specific model
find ~/.cache/huggingface/hub/ -name "*.nemo"

# Check cache size
du -sh ~/.cache/huggingface/
```

---

## 📊 Storage Requirements

### Per Model:
- **Marathi model**: ~500MB
- **Konkani model**: ~500MB

### For Full Training Session:
```
models/                      ~500MB  (base model)
data/                        ~10MB   (44 audio samples)
output/checkpoints/          ~500MB  (fine-tuned checkpoints)
logs/                        ~10MB   (training logs)
-------------------------------------------
TOTAL:                       ~1GB
```

**RunPod recommendation**: Use at least **50GB disk space** pod template to be safe.

---

## 🚨 Common Issues

### Error: "Repository not found"
**Solution**: Accept model conditions on Hugging Face website first
- Marathi: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
- Konkani: https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large

### Error: "Authentication required"
**Solution**: Run `huggingface-cli login` or set `HF_TOKEN` environment variable

### Error: "Model file not found"
**Solution**: Download model first using `scripts/download_model.py`

### Error: "No space left on device"
**Solution**: 
- Clean up old models: `rm -rf models/old_*`
- Clean cache: `rm -rf ~/.cache/huggingface/hub/`
- Use larger RunPod disk

---

## 🔄 Quick Reference Commands

### Manual Download Approach:
```bash
# Marathi
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_mr

# Konkani
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_kok

# Verify
ls -lh models/indicconformer_mr/*.nemo
```

### Check What's Downloaded:
```bash
# Local models
find models/ -name "*.nemo" -exec ls -lh {} \;

# Hugging Face cache
du -sh ~/.cache/huggingface/

# Total disk usage
df -h
```

### Clean Up:
```bash
# Remove local models
rm -rf models/indicconformer_mr/
rm -rf models/indicconformer_kok/

# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/hub/
```

---

## 📝 Summary

**Current Setup (Manual Download):**
1. ✅ You have `download_model.py` script
2. ✅ Config points to `models/indicconformer_mr/*.nemo`
3. ✅ Training expects pre-downloaded files
4. ⚠️ You must download before training

**What You Need to Do:**
```bash
# Before training, run:
huggingface-cli login
python scripts/download_model.py \
    --model_name ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large \
    --output_path models/indicconformer_mr

# Then train normally:
python scripts/nemo_train.py --config configs/konkani_finetune.yaml --model marathi
```

**Models stored at**: `models/indicconformer_mr/` (or `models/indicconformer_kok/`)
