# Amchi Konkani ASR - Automatic Speech Recognition for Konkani Language

## 🎯 Project Overview

This project implements Automatic Speech Recognition (ASR) for Konkani language, specifically targeting the "Amchi Konkani" dialect. The system is designed for a science fair demonstration and supports multiple ASR frameworks for flexibility and performance comparison.

### Goals
- Build functional Konkani ASR using Marathi base models
- Support framework switching (HuggingFace, NeMo, AI4Bharat)
- Enable easy migration between Windows/Mac environments
- Demonstrate working transcription with acceptable accuracy
- Compare performance across different frameworks

## 🏗️ Architecture

### Framework Architecture
The system uses a configurable framework approach with three supported ASR backends:

1. **HuggingFace Transformers** (✅ Working on Windows)
   - Base: `hriteshMaikap/marathi-asr-model` (Wav2Vec2-BERT)
   - Status: Fully functional with training and testing
   - Best for: Windows development, quick prototyping

2. **NVIDIA NeMo** (🔄 Planned)
   - Status: Windows compatibility issues (os.uname, signal.SIGKILL)
   - Best for: Production deployment, advanced features

3. **AI4Bharat** (🔄 Planned)
   - Status: Requires Linux/Mac environment
   - Best for: Indian language specialization, academic research

### Code Structure
```
konkani_asr/
├── configs/                    # Configuration files
│   ├── main_config.yaml       # Framework selector
│   ├── huggingface_config.yaml # HF training params
│   ├── nemo_config.yaml       # NeMo config (placeholder)
│   └── ai4bharat_config.yaml  # AI4Bharat config (placeholder)
├── scripts/                   # Main execution scripts
│   ├── fine_tune_hf.py       # HuggingFace training
│   ├── test_model.py         # Model testing
│   └── switch_framework.py   # Framework switching utility
├── data/                     # Training data
│   ├── audio/               # Audio files (.m4a, .wav)
│   └── test_run/            # Manifests (train.tsv, dev.tsv, test.tsv)
├── models/                  # Local model storage (C: drive)
└── D:/konkani_asr_models/   # Model checkpoints (D: drive - more space)
```

## 📊 Data Format

### Audio Data
- **Format**: M4A (primary), WAV (fallback)
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Duration**: 1-30 seconds per sample
- **Location**: `data/audio/`

### Text Data
- **Language**: Konkani (Devanagari script)
- **Encoding**: UTF-8
- **Format**: TSV manifests with columns: `audio_filepath`, `text`

### Manifest Format
```
audio_filepath	text
sentence_01.m4a	पाव वाट दाण्टुनु वत्ता म्हण्तना तिका एकु सिंहु मेऴ्ळो!
sentence_02.m4a	[konkani text here]
```

## 🤖 Models & Configurations

### Current Working Model
- **Framework**: HuggingFace
- **Base Model**: `hriteshMaikap/marathi-asr-model`
- **Architecture**: Wav2Vec2-BERT
- **Model Path**: `D:/konkani_asr_models/huggingface_konkani/checkpoint-5`
- **Training Data**: 1 sample (for testing)
- **Performance**: WER ~83% (expected to improve with more data)

### Configuration Files
- **Main Config**: `configs/main_config.yaml`
  - Controls active framework
  - Defines source/target languages
  - Sets model directories

- **Framework Configs**:
  - `configs/huggingface_config.yaml`: Training parameters, model settings
  - `configs/nemo_config.yaml`: NeMo-specific settings (placeholder)
  - `configs/ai4bharat_config.yaml`: AI4Bharat settings (placeholder)

## 🚀 Setup & Installation

### Prerequisites
- Python 3.13+
- FFmpeg (for M4A audio processing)
- 8GB+ RAM, 10GB+ disk space

### Installation Steps
1. **Clone and navigate**:
   ```bash
   cd /path/to/konkani_asr
   ```

2. **Install dependencies**:
   ```bash
   pip install torch transformers datasets librosa soundfile scipy evaluate
   ```

3. **Verify FFmpeg**:
   - FFmpeg binary located at: `ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe`
   - Used for M4A to WAV conversion

4. **Configure framework**:
   - Edit `configs/main_config.yaml` to set `framework: huggingface`
   - Adjust paths in config files as needed

---

### AI4Bharat + NeMo Golden Setup
If you plan to use the **AI4Bharat + NeMo** workflow (Linux/Mac), we provide a "golden" setup script that captures the environment and patches used during development. This is the recommended way to reproduce the working setup used for fine-tuning.

Bootstrap (fresh host):
```bash
# Clone + run the setup script (replace YOUR_USER):
git clone https://github.com/YOUR_USER/amchi_asr.git && cd amchi_asr && sudo ./setup_env.sh
```

Notes:
- **Python version**: Use Python 3.11 when possible (recommended). The AI4Bharat NeMo fork was historically tested on Python 3.9, so if you encounter dependency issues consider using a 3.9 virtualenv. `setup_env.sh` is the canonical source for the environment used in this repo.
- You must run `huggingface-cli login` (or set `HF_TOKEN`) before running tasks that download models from Hugging Face.
- The setup script now installs `Cython` before trying to install `pynini`/NeMo to reduce build failures on fresh hosts.
- **Run the preflight checks** after setup (this script validates tokenizer consistency and common failures):

```bash
python3 scripts/preflight_checks.py
```

Steps:
```bash
chmod +x setup_env.sh
sudo ./setup_env.sh
# Run the full preflight and unit tests before starting training
./scripts/run_preflight_tests.sh
```

Full details and caveats are in `SETUP_ENV.md` (this is the canonical reference — use it instead of older docs to avoid confusion).


## 🎯 Usage

### Framework Switching
```bash
python scripts/switch_framework.py list    # Show available frameworks
python scripts/switch_framework.py set huggingface  # Switch to HuggingFace
```

### Training
```bash
python scripts/fine_tune_hf.py
```
- Loads data from `data/test_run/`
- Trains model with parameters from `configs/huggingface_config.yaml`
- Saves checkpoint to `D:/konkani_asr_models/huggingface_konkani/`

### Testing
```bash
python scripts/test_model.py
```
- Loads model from `D:/konkani_asr_models/huggingface_konkani/checkpoint-5`
- Tests on `data/audio/sentence_06.m4a`
- Outputs transcription results

## 📈 Current Performance

### Metrics (1 training sample)
- **Training Loss**: 0.6335 → 0.2079 (5 steps)
- **Validation WER**: 83.3%
- **Test Transcription**: Good semantic understanding, minor character differences

### Sample Results
**Input Audio**: `sentence_06.m4a`  
**Expected Text**: `पाव वाट दाण्टुनु वत्ता म्हण्तना तिका एकु सिंहु मेऴ्ळो!`  
**Predicted Text**: `पाव वाट दाणटुनु वत्ता म्हणतना तिका एकु सिंहु मेळो`  
**Analysis**: Core words correct, understandable Konkani output

## 🔄 Framework Migration

### Windows → Mac/Linux Transition
1. **Copy project** to new environment
2. **Update paths** in config files
3. **Switch framework** in `main_config.yaml`
4. **Reinstall dependencies** (may need different versions)
5. **Retrain models** (framework-specific checkpoints)

### AI4Bharat Integration Plan
- **Environment**: Linux/Mac required
- **Authentication**: May need API tokens
- **Data Format**: Compatible with current TSV manifests
- **Expected Benefits**: Better Indian language support, potentially higher accuracy

## 🐛 Known Issues & Solutions

### Windows-Specific Issues
- **NeMo**: `os.uname` undefined, `signal.SIGKILL` missing
- **Solution**: Use WSL or migrate to Mac/Linux

### Audio Processing
- **M4A Support**: Requires FFmpeg for conversion
- **Solution**: FFmpeg binary included in project

### Disk Space
- **Issue**: Model saving fails with <4GB free space
- **Solution**: Save to D: drive (`D:/konkani_asr_models/`)

### Model Loading
- **Issue**: Missing `preprocessor_config.json`
- **Solution**: Copy from base model after training

## 🎯 Next Steps

### Immediate Tasks
1. **Scale training data** from 1 to full dataset (19 samples)
2. **AI4Bharat integration** on Mac/Linux
3. **Performance comparison** between frameworks
4. **Accuracy improvements** with more training data

### Long-term Goals
- Production deployment
- Real-time inference optimization
- Multi-speaker adaptation
- Extended Konkani dialect support

## 📚 Key Files for New Agents

When a new agent reviews this project, please have them examine:

1. **`configs/main_config.yaml`** - Framework selection and paths
2. **`scripts/fine_tune_hf.py`** - Main training logic
3. **`scripts/test_model.py`** - Testing and inference
4. **`data/test_run/train.tsv`** - Data format example
5. **This README** - Complete project overview

## 🤝 Contributing

For AI4Bharat integration or framework comparisons:
1. Review this README thoroughly
2. Test current HuggingFace pipeline
3. Set up target environment (Mac/Linux for AI4Bharat)
4. Follow framework-specific setup in config files
5. Compare performance metrics systematically

---

**Last Updated**: November 25, 2025  
**Status**: HuggingFace pipeline fully functional, AI4Bharat integration pending Mac environment setup

## **Work In Progress**

- **Date:** November 29, 2025
- **Current State:** Most of the HuggingFace pipeline runs in WSL. I created a fresh venv named `.venv_nemo` and installed a pinned NeMo stack (`nemo-toolkit==2.5.3` + matching `hydra-core`, `omegaconf`, `antlr4-python3-runtime`, `dill`, etc.). I also added an 8 GB swapfile to WSL so large pip builds won't OOM.

**Important (AI4Bharat / NeMo fork):** When using the AI4Bharat NeMo fork and AI4Bharat `.nemo` checkpoints, **use Python 3.9**. The fork is tested against specific dependency versions (notably `llvmlite`/`numba`) that may not be available on Python 3.10+; the fork also includes model options (e.g., `multisoftmax`) that are not in upstream NeMo. See `AI4BHARAT_SETUP_GUIDE.md` for the full rationale and setup steps.
- **Blocked On:** WSL is currently failing to start on this machine with `Error code: 6 (Wsl/Service/CreateInstance/E_FAIL)`. Until WSL is running again the final step (installing `lightning` and running the NeMo smoke test `scripts/nemo_finetune_smoke.py` in `.venv_nemo`) cannot be completed here.

### **What I tried (summary)**
- Created 8GB swap at `/swapfile` (persistent in `/etc/fstab`). Verified `free -h` shows Swap ≈ 11GiB.
- Built a clean venv `.venv_nemo` and installed CPU PyTorch, `nemo-toolkit==2.5.3` and required dependencies.
- Resolved a few package incompatibilities (`dill==0.3.6`, fixed `numpy`), but `lightning` (PyTorch Lightning) still needs installing before NeMo can import fully.
- During the final step WSL started failing (getpwuid/systemd errors) so the smoke test could not be executed.

### **Next steps to run after you reboot / fix WSL**
Run these commands from Windows PowerShell (normal or elevated). Prefer an elevated prompt if WSL had permission issues.

1) Quick check & start (recommended):

```powershell
# restart WSL host
wsl --shutdown

# try a simple WSL shell check
wsl -d Ubuntu-22.04 -e bash -lc "whoami; uname -a; free -h; swapon --show"
```

2) If that works, run the final NeMo step inside WSL (from the repo root):

```bash
cd ~/code/amchi_asr || cd /mnt/c/Users/Milind\ Kopikare/Code/amchi_konkani/konkani_asr
source .venv_nemo/bin/activate
pip install --no-cache-dir lightning
# set HF variables (already used earlier) and run smoke test
HF_TOKEN="<your HF token>" HF_HOME="/mnt/d/huggingface_cache" python scripts/nemo_finetune_smoke.py
```

3) If WSL still fails to start, try these host-side steps from an elevated PowerShell (one at a time):

```powershell
# restart LxssManager service
Restart-Service LxssManager -Force

# update WSL components
wsl --update
wsl --shutdown

# then retry the quick WSL check above
wsl -d Ubuntu-22.04 -e bash -lc "whoami; free -h; swapon --show"
```

4) If nothing fixes it, reboot Windows. If WSL continues to fail, collect the LxssManager events and paste them here:

```powershell
wsl -l -v > wsl_list.txt
wevtutil qe System /q:"*[System[Provider[@Name='LxssManager']]]" /f:text /c:200 > LxssManager_recent.txt
Get-Content LxssManager_recent.txt -Tail 60
Get-Content wsl_list.txt
```

### **If you want a faster verification now (optional)**
- If you prefer to validate manifests & WER quickly without NeMo, run the HuggingFace-based smoke test (faster, fewer heavy deps). I can run that immediately instead — say `run HF smoke` and I will execute it and report WER.

### **Where we left off for GitHub Copilot Agent**
- Repo root: `README.md`, `scripts/nemo_finetune_smoke.py`, and `.venv_nemo` were prepared. The next concrete command for the agent (after WSL is responsive) is the `pip install --no-cache-dir lightning` step followed by `python scripts/nemo_finetune_smoke.py` with `HF_TOKEN`/`HF_HOME` set.

---

**Last Updated**: November 29, 2025  
**WIP Status**: Waiting on WSL host restart; NeMo stack mostly installed in `.venv_nemo`.
For MMS reproducibility & recovery, see `docs/MMS_FINETUNE.md`.

## 🚀 RunPod Deployment Guide

This section documents the process of fine-tuning the Konkani ASR model on RunPod, including common issues and solutions for future deployments.

### Prerequisites
- **RunPod Instance**: RTX 4090 GPU, 40GB+ persistent storage, Ubuntu 22.04
- **Python Environment**: Python 3.9.25 with venv_py39
- **CUDA**: 12.4 compatible
- **Disk Space**: Minimum 40GB (increase from default 20GB to avoid space issues)

### Setup Steps
1. **Clone Repository**:
   ```bash
   git clone https://github.com/milind-kopikar/amchi_asr.git
   cd amchi_asr
   ```

2. **Install Dependencies**:
   ```bash
   pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install nemo_toolkit[asr]==1.19.0
   pip install huggingface_hub==0.19.4 transformers==4.24.0
   pip install jiwer librosa soundfile scipy
   ```

3. **Download Base Model**:
   ```bash
   HF_TOKEN=your_token python scripts/download_model.py
   ```

4. **Prepare Data**:
   ```bash
   python scripts/download_data_from_railway.py
   ```

5. **Fine-tune Model**:
   ```bash
   python scripts/nemo_train.py
   ```

6. **Evaluate Model**:
   ```bash
   python scripts/evaluate_nemo.py --model_path results/konkani_full/konkani_asr_final.nemo --test_manifest data/test/manifest.jsonl --output_file results/konkani_full/test_results_detailed.json --batch_size 8
   ```

### Common Issues & Solutions

#### 1. Disk Space Issues
- **Problem**: RunPod default 20GB storage insufficient for model downloads and training.
- **Solution**: Increase storage to 40GB+ in RunPod settings before starting.

#### 2. Dependency Version Conflicts
- **Problem**: `ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'`
- **Solution**: Downgrade huggingface_hub to 0.19.4 and transformers to 4.24.0 for NeMo 1.19.0 compatibility.

#### 3. Model Path Errors
- **Problem**: Scripts reference incorrect model paths (e.g., /tmp/ instead of models/).
- **Solution**: Update config files and scripts to use correct local paths.

#### 4. Evaluation Script Failures
- **Problem**: `transcribe()` returns None or KeyError due to missing language_id.
- **Solution**: Add `language_id='kok'` to all `model.transcribe()` calls for the multilingual IndicConformer model.

#### 5. Prediction Extraction Issues
- **Problem**: Hybrid RNNT-CTC models return tuple (ctc_list, rnnt_list), causing jiwer.wer() to fail on lists.
- **Solution**: Extract string predictions: `prediction = transcriptions[1][0] if len(transcriptions) > 1 and transcriptions[1] else transcriptions[0][0] if transcriptions[0] else ""`

#### 6. Indentation Errors in Scripts
- **Problem**: Manual edits to Python files introduce indentation issues.
- **Solution**: Use proper indentation (8 spaces for nested blocks) and test scripts after changes.

#### 7. ONNX Export Failures
- **Problem**: `ModuleNotFoundError: No module named 'einops'` during export.
- **Solution**: Skip ONNX export; deploy .nemo model directly via Hugging Face Spaces with custom inference.

### Performance Results
- **Training WER**: 38.72% on dev set (42 samples)
- **Test WER**: 0.2% on test set (42 samples)
- **Model Size**: ~523MB final .nemo file

### Deployment to Hugging Face
1. Create HF Space with GPU support.
2. Upload model, app.py (Gradio interface), and requirements.txt.
3. Deploy for API access.

### Key Learnings
- Always verify disk space before starting.
- Pin dependency versions to match NeMo requirements.
- Test evaluation scripts incrementally (single sample → batch).
- For multilingual models, specify language_id in inference.
- Handle hybrid model outputs carefully.
- Document all changes and issues for future runs.

---

**Last Updated**: December 22, 2025
**Status**: Full pipeline working on RunPod, deployed to HF Spaces
