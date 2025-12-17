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

### AI4Bharat Fork vs Standard NeMo

**The model cards recommend AI4Bharat's NeMo fork:**

```bash
git clone https://github.com/AI4Bharat/NeMo.git
cd NeMo
git checkout nemo-v2
bash reinstall.sh
```

**However, our code uses standard NVIDIA NeMo:**

```bash
pip install nemo_toolkit['all']
```

### Which Should We Use?

**Testing Recommendation:**
1. **Start with standard NVIDIA NeMo** (already in requirements.txt)
2. **If you get compatibility errors** (e.g., decoder issues, language_id parameter problems), switch to AI4Bharat NeMo
3. **Most likely scenario**: Standard NeMo will work fine because:
   - Both models use standard NeMo architecture (Conformer + Hybrid CTC-RNNT)
   - AI4Bharat fork is based on NVIDIA NeMo v1.20.0
   - We're only fine-tuning, not training from scratch

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

If this works → use standard NeMo
If this fails → switch to AI4Bharat NeMo fork

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
- [ ] If errors occur, switch to AI4Bharat NeMo fork

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
**Solution**: Switch to AI4Bharat NeMo fork

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
