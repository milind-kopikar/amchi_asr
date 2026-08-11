# Amchi ASR — Automatic Speech Recognition for Konkani and Deaf Speech

Two research projects applying AI to under-resourced speech recognition problems:

1. **Amchi Konkani ASR** — speech recognition for Amchigale Konkani, an endangered dialect with no existing ASR support
2. **Deaf Speech ASR** — speech recognition for speakers who are deaf or hard-of-hearing, whose atypical speech patterns cause standard ASR to fail

Both projects fine-tune the same base model (AI4Bharat's IndicConformer, trained on Marathi) using NVIDIA NeMo, and both achieved **WER under 35%** — a substantial improvement over the respective baselines.

---

## Project 1: Amchi Konkani ASR

### The problem

Amchigale Konkani is spoken by approximately 2 million people along the west coast of Karnataka, India. It is critically endangered: unlike Goan Konkani, it has no Google Translate support, no standardized script, no digital dictionary, and no ASR system. The language has no written standard — speakers use either Devanagari or Kannada script. With each generation, usage declines.

Standard multilingual ASR models (Whisper, etc.) cannot transcribe spoken Konkani — they map it to the nearest Marathi or Hindi word, producing mostly incorrect output.

### Approach

**Base model:** AI4Bharat's IndicConformer (`indicconformer_stt_mr_hybrid_ctc_rnnt_large`) — a large Conformer trained on Marathi. Marathi is the linguistically closest language with a well-resourced ASR model: shared Sanskrit vocabulary, similar grammar, and the same Devanagari script.

**Fine-tuning strategy:** CTC-only fine-tuning using upstream NVIDIA NeMo v2.7.0. The RNNT head is disabled; only the CTC auxiliary head is trained. This avoids GPU JIT (Numba) issues on RunPod and significantly reduces training complexity.

**Data:** Recordings of Konkani speakers reading story sentences, collected via a community crowdsourcing web app. Audio is 16 kHz mono WAV. Transcriptions are in Devanagari script.

**Key experiments:**

| Experiment | Split strategy | Encoder | Epochs | Test WER |
|---|---|---|---|---:|
| Baseline | Story-based | Full fine-tune | 50 | 54.7% |
| Run C | Story-based | **Frozen** | 100 | 49.1% |
| **Run S** | **Speaker-stratified** | **Frozen** | 100 | **34.1%** |

**Key findings:**

- **Frozen encoder beats full fine-tune** on this dataset size. The Marathi encoder already captures the phoneme space well; retraining it causes overfitting.
- **Speaker-stratified split is critical.** A story-based split puts all recordings of some speakers entirely in the test set — the model has never heard those voices. Ensuring every speaker appears in training (stratified split) reduced WER by 15 percentage points.

**Post-processing:** A dictionary-based correction layer validates model output against a 4,381-word Konkani dictionary, then uses Gemini to correct words not found in the dictionary while preserving recognized Konkani words and cultural expressions.

### Results

Best model (**Run S**): **34.1% WER** (down from 54.7% baseline — a 20.6 percentage point improvement).

Checkpoint: `konkani_asr-epoch=88-val_wer=0.334.ckpt`

---

## Project 2: Deaf Speech ASR

### The problem

People who are deaf or hard-of-hearing produce speech that differs substantially from typical hearing speech: altered articulation, non-standard prosody, irregular rhythm, and atypical phoneme production. Standard ASR systems, trained exclusively on hearing speech, fail badly — producing largely unintelligible output or empty transcriptions.

This project targets a specific real-world task: transcribing deaf speakers reading everyday transactional sentences in Marathi (Story 4: "दैनंदिन कामे १" — daily tasks including shopping, transportation, and basic communication). 124 recordings were collected from deaf/hard-of-hearing speakers via a community crowdsourcing platform.

### Approach

**Base model:** Same IndicConformer Marathi model as the Konkani project.

**Data collection:** 124 approved recordings from the community web app at `deafspeechcollector-production.up.railway.app`. The same 124 samples are used for train/dev/test in most experiments — intentionally, to verify whether the model can learn the patterns of this specific population on this specific sentence set.

**Key experiments:**

| Experiment | Data | Encoder | Epochs | Test WER |
|---|---|---|---|---:|
| Baseline (DS-C) | 124 samples | Full fine-tune | 50 | 75.3% |
| DS-A | 124 samples | Frozen | 100 | 79.6% |
| DS-B | 188 samples (added OOD data) | Full fine-tune | 100 | 93.1% |
| **DS-D** | **372 samples (speed perturbation)** | **Full fine-tune** | 100 | **34.7%** |

**Key findings:**

- **Frozen encoder hurts deaf speech** — the opposite of Konkani. Deaf speech occupies a very different acoustic space from Marathi hearing speech. The encoder needs to be retrained, not preserved.
- **Out-of-distribution data makes things worse.** Adding 64 recordings from different speakers in a different acoustic environment (DS-B) increased WER to 93.1%. The model lost focus on the target population.
- **Speed perturbation is the winning strategy.** Applying 0.9×, 1.0×, and 1.1× speed shifts to the same 124 recordings produces 372 training samples from the same distribution. This tripled the effective dataset size without introducing new speakers or acoustic environments, yielding a 40.6 percentage point improvement over baseline.

**Gemini post-processing:** A three-mode LLM post-processing layer further improves output quality:
- **RECONSTRUCT mode** — when all ASR tokens are garbled/unknown, uses phonetic fragments and a curated list of 45 high-frequency Marathi words to reconstruct complete sentences
- **FILL mode** — when some trusted words exist but gaps remain, preserves recognized words and fills gaps contextually
- **PASSTHROUGH mode** — for high-quality utterances, strips unknown tokens without LLM intervention

Applied to the baseline model (DS-C), Gemini post-processing improved WER from 75.3% to 62.0% — a **13.3 percentage point improvement** (17.6% relative). Critically, a safety valve reverts to the original ASR output if Gemini's suggestion worsens WER, ensuring post-processing never degrades results.

### Results

Best model (**DS-D**): **34.7% WER** (down from 75.3% baseline — a 40.6 percentage point improvement).

Checkpoint: `konkani_asr-epoch=96-val_wer=0.269.ckpt`

---

## Shared Architecture

Both projects use the same training and inference infrastructure:

| Component | Details |
|---|---|
| Base model | AI4Bharat IndicConformer (Marathi, hybrid CTC-RNNT, 499MB) |
| Framework | NVIDIA NeMo v2.7.0 (upstream, not AI4Bharat fork) |
| Python | 3.11 |
| Training hardware | RunPod cloud GPU (RTX 4000 Ada 20GB / A40 48GB) |
| Checkpoint storage | Cloudflare R2 (public bucket) |
| Post-processing | Google Gemini API |
| Data collection | Railway-hosted web apps (PostgreSQL backend) |

**WER summary across both projects:**

| Project | Baseline WER | Best WER | Improvement |
|---|---:|---:|---:|
| Amchi Konkani | 54.7% | **34.1%** | −20.6 pp |
| Deaf Speech | 75.3% | **34.7%** | −40.6 pp |
| Deaf Speech + post-processing | 75.3% | **62.0%** | −13.3 pp (on baseline model) |

---

## Demo Webapps

### Deaf Speech Demo (`webapp-deaf/`)

A Next.js application showing all four experiments in sequence from worst to best, letting users record or upload audio, transcribe with each model, and see the effect of Gemini post-processing.

```bash
cd webapp-deaf
npm install
# Add GEMINI_API_KEY and GOOGLE_TTS_API_KEY to webapp-deaf/.env.local
npm run dev
# Open http://localhost:3000
```

Experiments shown (worst → best): DS-B (93.1%) → DS-A (79.6%) → DS-C / Baseline (75.3%) → **DS-D (34.7%)**

### Amchi Konkani Demo (`webapp-amchi/`)

A parallel web app for the Konkani ASR system with audio recording, transcription, and playback.

```bash
cd webapp-amchi
npm install
npm run dev
```

---

## Repository Navigation

Start with **[`AGENT_START_HERE.md`](AGENT_START_HERE.md)** — it maps every experiment, every checkpoint location, and every document in the repo.

| Document | What it covers |
|---|---|
| [`AGENT_START_HERE.md`](AGENT_START_HERE.md) | Project hub — current results, checkpoint URLs, navigation |
| [`docs/MODULE_TRAINING.md`](docs/MODULE_TRAINING.md) | How to train or retrain — RunPod environment, configs, scripts |
| [`docs/MODULE_INFERENCE.md`](docs/MODULE_INFERENCE.md) | How to load a checkpoint and transcribe audio |
| [`docs/MODULE_SERVERLESS.md`](docs/MODULE_SERVERLESS.md) | Docker build and RunPod serverless endpoint deployment |
| [`docs/MODULE_WEBAPP_DEAF.md`](docs/MODULE_WEBAPP_DEAF.md) | Deaf speech demo webapp — architecture, deployment, extension |
| [`docs/CHECKPOINTS_REGISTRY.md`](docs/CHECKPOINTS_REGISTRY.md) | All checkpoint R2 URLs and local paths |
| [`ENHANCED_POSTPROCESSING_METHOD.md`](ENHANCED_POSTPROCESSING_METHOD.md) | Gemini post-processing algorithm — RECONSTRUCT/FILL/PASSTHROUGH |
| [`LEARNINGS.md`](LEARNINGS.md) | Hard-won lessons from training runs (read before starting) |
| [`AGENT_HANDOFF.md`](AGENT_HANDOFF.md) | Session-by-session build history |

---

## Quick Start

### Run inference on a WAV file

```bash
# Deaf speech model
python scripts/deaf_speech_inference.py --audio path/to/audio.wav

# Amchi Konkani model
python scripts/amchi_inference.py --audio path/to/audio.wav
```

Audio must be 16 kHz mono WAV. See [`docs/MODULE_INFERENCE.md`](docs/MODULE_INFERENCE.md) for the checkpoint loading pattern (non-obvious — requires a manual config patch and `strict=False`).

### Train a new experiment

```bash
# After RunPod environment setup (see docs/MODULE_TRAINING.md):
python fine_tune.py configs/your_config.yaml
```

### Re-download audio data

```bash
# Deaf speech (124 WAV files from Railway API)
python scripts/download_data_from_railway.py

# Amchi Konkani audio
# See AGENT_START_HERE.md §5 for Konkani Railway API URL
```

---

## Environment Setup (RunPod)

After each pod restart (NeMo is not persistent):

```bash
pip install "nemo_toolkit[asr]" --ignore-installed blinker -q
pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128 -q

# Apply the required conv_asr patch
NEMO_FILE=$(python3 -c "import nemo.collections.asr.modules.conv_asr as m; print(m.__file__)" 2>&1 | tail -1)
cp patches/conv_asr_fixed.py "$NEMO_FILE"
export APPLY_CONV_PATCH=1
```

Copy `.env.example` to `.env` and fill in your API keys (Gemini, HuggingFace, GitHub PAT). Audio files and model checkpoints are not committed to git — see `AGENT_START_HERE.md` for R2 download URLs.
