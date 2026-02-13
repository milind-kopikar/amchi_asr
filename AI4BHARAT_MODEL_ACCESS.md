# AI4Bharat Model Access Guide

## Models Are On Hugging Face!

**AI4Bharat** is an organization that publishes their models on Hugging Face Hub. The models we're using are:

### 🇮🇳 Marathi Base Model
- **Full Name**: `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`
- **Link**: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
- **Size**: 120M parameters (Conformer-Large)
- **Downloads**: ~312/month
- **Language ID**: `mr` (for inference)

### 🇮🇳 Konkani Base Model
- **Full Name**: `ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large`
- **Link**: https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large
- **Size**: 120M parameters (Conformer-Large)
- **Downloads**: ~15/month
- **Language ID**: `kok` (for inference)

---

## 🔒 IMPORTANT: Access Requirements

### You MUST Accept Model Conditions First

Both models require you to:

1. **Log into Hugging Face**: https://huggingface.co/login
2. **Navigate to the model page**:
   - Marathi: https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large
   - Konkani: https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large
3. **Accept the conditions** to access model files
4. **Create a Hugging Face token** (if you don't have one):
   - Go to: https://huggingface.co/settings/tokens
   - Create new token with "Read" access
   - Save it securely

### On RunPod, Set Your Token

Before running training, authenticate with:

```bash
# Method 1: Login interactively
huggingface-cli login

# Method 2: Set token as environment variable
export HF_TOKEN="your_token_here"

# Method 3: Pass token in Python code
from huggingface_hub import login
login(token="your_token_here")
```

---

## 🛠️ NeMo Version Requirement

### This Repo: Python 3.11 + Upstream NeMo

**We use Python 3.11 and upstream (standard) NVIDIA NeMo.** Do not use the AI4Bharat NeMo fork for normal setup—it requires Python 3.9 and fails on 3.11 (e.g. `llvmlite==0.38.1` has no wheel for 3.11). See `SETUP_ENV.md` and `MASTER_REPRODUCTION_GUIDE.md`.

```bash
# Recommended (Python 3.11 venv)
python3.11 -m venv venv_py311
source venv_py311/bin/activate
pip install "nemo_toolkit[all]" pynini librosa
# Then apply conv_asr patch (see SETUP_ENV.md)
```

**If you must use the AI4Bharat fork** (e.g. Python 3.9 debugging), the model cards recommend their fork; see `AI4BHARAT_SETUP_GUIDE.md`. For RunPod and normal training, stick with **Python 3.11 + upstream NeMo**.

### Quick Test on RunPod

```python
import nemo.collections.asr as nemo_asr

# Test loading Marathi model
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
)
print(f"✅ Marathi model loaded: {type(model)}")

# Test loading Konkani model
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large"
)
print(f"✅ Konkani model loaded: {type(model)}")
```

If this works → continue with upstream NeMo. If you see decoder/load errors, ensure you use the conv_asr patch and correct tokenizer paths (see SETUP_ENV.md).

---

## 📥 Model Download Process

### Automatic Download (Our Code)

When you run:
```bash
python scripts/nemo_train.py --model marathi
```

NeMo will automatically:
1. Check if you have Hugging Face credentials
2. Download model from `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large`
3. Cache it locally in `~/.cache/huggingface/`
4. Load it for fine-tuning

### Manual Pre-Download (Optional)

If you want to download models before training:

```python
import nemo.collections.asr as nemo_asr

# Download and cache Marathi model
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large"
)
model.save_to("models/marathi_base.nemo")

# Download and cache Konkani model
model = nemo_asr.models.ASRModel.from_pretrained(
    "ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large"
)
model.save_to("models/konkani_base.nemo")
```

---

## 🔍 Model Architecture Details

Both models use **Hybrid CTC-RNNT** architecture:

- **Encoder**: Conformer-Large (17 blocks, 512 dimensions, 120M parameters)
- **Decoder**: Hybrid CTC + RNNT
  - CTC: Fast, greedy decoding
  - RNNT: Beam search, better accuracy
- **Input**: 16kHz mono WAV files
- **Output**: Devanagari text

### Inference Modes

When testing/validation, you can choose decoder:

```python
# CTC decoder (faster, slightly lower accuracy)
model.cur_decoder = "ctc"
text = model.transcribe(['audio.wav'], batch_size=1, language_id='mr')[0]

# RNNT decoder (slower, better accuracy)
model.cur_decoder = "rnnt"
text = model.transcribe(['audio.wav'], batch_size=1, language_id='kok')[0]
```

**For our training**: We fine-tune both decoders simultaneously (hybrid training)

---

## 🎯 Quick Start Checklist

Before running on RunPod:

- [ ] Log into Hugging Face account
- [ ] Accept conditions for Marathi model
- [ ] Accept conditions for Konkani model (optional, if testing both)
- [ ] Create Hugging Face token with Read access
- [ ] Save token securely
- [ ] On RunPod: `huggingface-cli login` or set `HF_TOKEN` env variable
- [ ] Test model loading with quick Python snippet
- [ ] If standard NeMo works, proceed with training
- [ ] If errors occur, confirm Python 3.11 + upstream NeMo and conv_asr patch (see SETUP_ENV.md)

---

## 💡 Expected Model Paths

After download, models will be cached at:

```
~/.cache/huggingface/hub/
├── models--ai4bharat--indicconformer_stt_mr_hybrid_ctc_rnnt_large/
│   └── snapshots/
│       └── <commit_hash>/
│           └── *.nemo files
└── models--ai4bharat--indicconformer_stt_kok_hybrid_ctc_rnnt_large/
    └── snapshots/
        └── <commit_hash>/
            └── *.nemo files
```

NeMo handles this automatically - you don't need to worry about paths!

---

## 🚨 Troubleshooting

### Error: "Repository Not Found"
**Solution**: Accept model conditions on Hugging Face website

### Error: "Authentication Required"
**Solution**: Run `huggingface-cli login` or set `HF_TOKEN`

### Error: "Decoder not compatible" or language_id issues
**Solution**: Use Python 3.11 and upstream NeMo; apply conv_asr patch. See SETUP_ENV.md. (AI4Bharat fork is only for Python 3.9.)

### Error: "Out of memory during download"
**Solution**: Each model is ~500MB - ensure you have 2GB free disk space

---

## 📊 Model Comparison

| Feature | Marathi Model | Konkani Model |
|---------|--------------|---------------|
| **Speakers** | Native Marathi | Goan Konkani |
| **Training Data** | Large corpus | Smaller corpus |
| **Downloads/Month** | ~312 | ~15 |
| **Linguistic Similarity** | Similar to Konkani | Target language |
| **Expected WER (untrained)** | High (different vocab) | Lower (same language) |
| **Expected WER (fine-tuned)** | ? (test both!) | ? (test both!) |

**We'll test both to see which gives better results for your Konkani data!**
