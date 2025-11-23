# Konkani ASR Fine-tuning System Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Technology Stack](#technology-stack)
6. [API Interactions](#api-interactions)
7. [Training Pipeline](#training-pipeline)
8. [Inference Pipeline](#inference-pipeline)
9. [Configuration Management](#configuration-management)
10. [Error Handling & Logging](#error-handling--logging)

---

## System Overview

The Konkani ASR Fine-tuning System is a complete machine learning pipeline that adapts a pre-trained Marathi ASR model (IndicConformer) to recognize Konkani speech. The system follows a modular architecture with clear separation of concerns.

### Key Characteristics
- **Transfer Learning Approach**: Fine-tunes existing model instead of training from scratch
- **Modular Design**: Independent components for data prep, training, evaluation, inference
- **Production Ready**: Includes validation, monitoring, and deployment capabilities
- **Low Resource**: Optimized for scenarios with limited training data

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    KONKANI ASR SYSTEM                           │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   DATA      │    │  TRAINING   │    │ EVALUATION  │         │
│  │ COLLECTION  │───▶│  PIPELINE   │───▶│ & TESTING   │         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   MODEL     │    │ INFERENCE   │    │  CONFIG     │         │
│  │ MANAGEMENT  │    │  SERVICE    │    │ MANAGEMENT  │         │
│  │             │    │             │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 EXTERNAL DEPENDENCIES                       │ │
│  │  • NVIDIA NeMo Framework                                    │ │
│  │  • PyTorch & CUDA                                           │ │
│  │  • Hugging Face Hub                                         │ │
│  │  • Audio Processing Libraries                               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Data Management Layer
**Purpose**: Handle all data preparation and preprocessing

#### Components:
- **Audio Preprocessor**: Normalizes audio files (resampling, normalization)
- **Text Processor**: Cleans and normalizes Konkani transcripts
- **Manifest Generator**: Creates NeMo-compatible training manifests
- **Data Splitter**: Creates train/validation/test splits

#### Key Files:
- `scripts/prepare_data.py` - Main data preparation script
- `scripts/prepare_recording.py` - Audio recording utilities
- `data/` directory - Organized data storage

### 2. Model Management Layer
**Purpose**: Handle model downloading, storage, and versioning

#### Components:
- **Model Downloader**: Downloads IndicConformer from Hugging Face
- **Model Loader**: Loads models for training/inference
- **Checkpoint Manager**: Saves training checkpoints
- **Model Validator**: Verifies model integrity

#### Key Files:
- `scripts/download_model.py` - Model acquisition
- `models/` directory - Model storage
- `results/checkpoints/` - Training checkpoints

### 3. Training Pipeline
**Purpose**: Execute the fine-tuning process

#### Components:
- **Trainer**: Core training orchestration
- **Optimizer**: Learning rate and parameter optimization
- **Validator**: Validation during training
- **Logger**: Training metrics and progress tracking

#### Key Files:
- `scripts/fine_tune.py` - Main training script
- `configs/konkani_finetune.yaml` - Training configuration
- `results/logs/` - Training logs

### 4. Evaluation System
**Purpose**: Measure model performance and accuracy

#### Components:
- **WER Calculator**: Word Error Rate computation
- **CER Calculator**: Character Error Rate computation
- **BLEU Scorer**: Translation quality assessment
- **Comparative Analyzer**: Model comparison utilities

#### Key Files:
- `scripts/evaluate.py` - Evaluation script
- `scripts/validate_model.py` - Advanced validation
- `evaluation_results.json` - Results storage

### 5. Inference Service
**Purpose**: Transcribe new audio using trained models

#### Components:
- **Audio Processor**: Real-time audio preprocessing
- **Model Runner**: Execute transcription
- **Result Formatter**: Format output text
- **Batch Processor**: Handle multiple audio files

#### Key Files:
- `scripts/infer.py` - Inference script
- `scripts/minimal_test.py` - Quick testing

### 6. Configuration Management
**Purpose**: Centralized configuration handling

#### Components:
- **YAML Parser**: Load training configurations
- **Parameter Validator**: Validate configuration values
- **Environment Manager**: Handle system-specific settings

#### Key Files:
- `configs/*.yaml` - Configuration files
- `scripts/setup_environment.py` - Environment validation

---

## Data Flow Architecture

### End-to-End Data Flow

```
Raw Audio + Transcripts → Data Preparation → Training → Evaluation → Inference
       ↓                        ↓              ↓            ↓            ↓
   Audio Files (.wav)      Manifest Files   Checkpoints   Metrics     Transcriptions
   Text Files (.txt)       (.tsv format)    (.nemo)       (WER/CER)    (.txt/.json)
```

### Detailed Data Pipeline

#### Phase 1: Data Ingestion
```
Audio Files → Audio Validator → Format Converter → Normalized Audio
Transcripts → Text Cleaner → Unicode Normalizer → Clean Text
```

#### Phase 2: Data Preparation
```
Normalized Audio + Clean Text → Manifest Creator → Train/Val/Test Splits
                                    ↓
                             Training Manifest (.tsv)
                             Validation Manifest (.tsv)
                             Test Manifest (.tsv)
```

#### Phase 3: Training Data Flow
```
Training Manifest → Data Loader → Batch Creator → Model Input
Validation Manifest → Validator → Metrics Calculator → Training Adjustment
```

#### Phase 4: Model Training
```
Model Input → IndicConformer → Forward Pass → Loss Calculation → Backpropagation
       ↓            ↓              ↓            ↓              ↓
   Audio Features → Encoder → Decoder → Predictions → Gradient Update → Parameter Update
```

#### Phase 5: Evaluation Flow
```
Test Audio → Trained Model → Predictions → Reference Comparison → WER/CER Calculation
```

#### Phase 6: Inference Flow
```
New Audio → Preprocessor → Model → Transcription → Post-processor → Final Text
```

---

## Technology Stack

### Core Frameworks

#### NVIDIA NeMo (nemo-toolkit)
- **Purpose**: Primary ASR framework for training and inference
- **Key Components Used**:
  - `EncDecHybridRNNTCTCModel`: Main ASR model class
  - `AudioToTextDataLayer`: Data loading and preprocessing
  - `RNNTLoss`: Loss function for training
  - `GreedyRNNTInfer`: Inference engine
- **Version**: 1.20.0+
- **Installation**: `pip install nemo-toolkit[asr]`

#### PyTorch (torch)
- **Purpose**: Deep learning framework underlying NeMo
- **Key Components**:
  - `torch.nn`: Neural network modules
  - `torch.optim`: Optimization algorithms
  - `torch.utils.data`: Data loading utilities
  - `torch.cuda`: GPU acceleration
- **Version**: 1.12.0+
- **CUDA Support**: Required for GPU training

### Audio Processing Libraries

#### Librosa (librosa)
- **Purpose**: Audio feature extraction and preprocessing
- **Key Functions**:
  - `librosa.load()`: Audio file loading
  - `librosa.resample()`: Sample rate conversion
  - `librosa.util.normalize()`: Audio normalization
- **Version**: 0.9.0+

#### SoundFile (soundfile)
- **Purpose**: Audio file I/O operations
- **Key Functions**:
  - `soundfile.read()`: Read audio files
  - `soundfile.write()`: Write audio files
  - `soundfile.info()`: Get audio metadata
- **Version**: 0.10.0+

#### Torchaudio (torchaudio)
- **Purpose**: PyTorch audio processing utilities
- **Key Components**:
  - Audio I/O operations
  - Feature extraction
  - Data augmentation
- **Version**: 0.12.0+

### Data Processing & ML Libraries

#### Pandas (pandas)
- **Purpose**: Data manipulation and CSV/TSV handling
- **Key Uses**:
  - Manifest file creation
  - Data analysis and statistics
  - Results storage and export
- **Version**: 1.3.0+

#### NumPy (numpy)
- **Purpose**: Numerical computing foundation
- **Key Uses**:
  - Audio data arrays
  - Mathematical operations
  - Statistical calculations
- **Version**: 1.21.0+

#### SciPy (scipy)
- **Purpose**: Scientific computing utilities
- **Key Uses**:
  - Signal processing
  - Statistical functions
- **Version**: 1.7.0+

### Evaluation & Metrics

#### JIWER (jiwer)
- **Purpose**: ASR evaluation metrics
- **Key Functions**:
  - `jiwer.wer()`: Word Error Rate calculation
  - `jiwer.cer()`: Character Error Rate calculation
- **Version**: 3.0.0+

#### WERpy (werpy)
- **Purpose**: Advanced WER calculations
- **Key Features**:
  - Detailed error analysis
  - Multiple WER variants
- **Version**: 1.0.0+

### Configuration & Utilities

#### OmegaConf (omegaconf)
- **Purpose**: Configuration file parsing
- **Key Uses**:
  - YAML configuration loading
  - Parameter validation
  - Configuration merging
- **Version**: 2.1.0+

#### Hydra (hydra-core)
- **Purpose**: Configuration management
- **Key Features**:
  - Dynamic configuration
  - Command-line overrides
- **Version**: 1.1.0+

#### PyYAML (yaml)
- **Purpose**: YAML file processing
- **Key Uses**:
  - Configuration file reading/writing
- **Version**: 5.4.0+

### External APIs

#### Hugging Face Hub
- **Purpose**: Model repository and downloading
- **Key Components**:
  - `snapshot_download()`: Download model files
  - Authentication handling
  - Model versioning
- **Library**: `huggingface_hub`
- **Version**: Latest

### Development & Testing

#### pytest (pytest)
- **Purpose**: Unit testing framework
- **Key Uses**:
  - Component testing
  - Integration testing
- **Version**: 6.2.0+

#### tqdm (tqdm)
- **Purpose**: Progress bars for long operations
- **Key Uses**:
  - Training progress visualization
  - Data processing feedback
- **Version**: 4.62.0+

---

## API Interactions

### NVIDIA NeMo API Calls

#### Model Loading
```python
from nemo.collections.asr.models import EncDecHybridRNNTCTCModel

# Load pre-trained model
model = EncDecHybridRNNTCTCModel.from_pretrained("ai4bharat/indicconformer_marathi")

# Load fine-tuned model
model = EncDecHybridRNNTCTCModel.restore_from("path/to/model.nemo")
```

#### Training API
```python
from nemo.core import NeuralModule
from nemo.utils import logging

# Setup trainer
trainer = Trainer(
    gpus=1,
    max_epochs=config.model.max_epochs,
    accelerator="gpu"
)

# Training step
trainer.fit(model, train_dataloader, val_dataloader)
```

#### Inference API
```python
# Single audio transcription
transcriptions = model.transcribe([audio_tensor], batch_size=1)

# Batch transcription
transcriptions = model.transcribe(audio_batch, batch_size=8)
```

### PyTorch API Interactions

#### GPU Management
```python
import torch

# Check CUDA availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    model = model.to(device)
else:
    device = torch.device("cpu")
```

#### Data Loading
```python
from torch.utils.data import DataLoader

# Create data loader
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=4
)
```

### Audio Processing APIs

#### Librosa Integration
```python
import librosa

# Load and preprocess audio
audio, sr = librosa.load(audio_path, sr=16000)
audio = librosa.util.normalize(audio)
```

#### SoundFile Integration
```python
import soundfile as sf

# Read audio file
audio, samplerate = sf.read(audio_path)

# Write processed audio
sf.write(output_path, audio, samplerate)
```

### Configuration APIs

#### OmegaConf Usage
```python
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("configs/konkani_finetune.yaml")

# Access parameters
learning_rate = config.model.optim.lr
batch_size = config.model.train_ds.batch_size
```

---

## Training Pipeline

### Pipeline Flowchart

```
Start Training
      ↓
Load Configuration (YAML)
      ↓
Initialize Environment
      ↓
Download/Load Base Model
      ↓
Prepare Training Data
      ↓
Create Data Loaders
      ↓
Initialize Trainer
      ↓
Training Loop:
      ↓
    For each epoch:
        For each batch:
            Forward pass → Loss calculation → Backward pass → Parameter update
            ↓
        Validation → Metrics calculation → Checkpoint saving
        ↓
    Early stopping check
      ↓
Save Final Model
      ↓
Generate Training Report
      ↓
End Training
```

### Detailed Training Steps

#### 1. Configuration Loading
- Load YAML configuration files
- Validate parameters
- Set up experiment logging

#### 2. Environment Setup
- Check CUDA availability
- Set random seeds for reproducibility
- Initialize logging system

#### 3. Model Preparation
- Download IndicConformer model
- Load model weights
- Freeze/unfreeze appropriate layers
- Modify vocabulary if needed

#### 4. Data Pipeline
- Load training manifests
- Create data loaders
- Apply data augmentation
- Set up batch processing

#### 5. Training Loop
- **Forward Pass**: Audio → Model → Predictions
- **Loss Calculation**: Compare predictions vs ground truth
- **Backward Pass**: Calculate gradients
- **Optimization**: Update model parameters
- **Validation**: Periodic performance checking
- **Checkpointing**: Save model snapshots

#### 6. Training Monitoring
- Loss tracking
- WER/CER monitoring
- Learning rate scheduling
- Early stopping logic

---

## Inference Pipeline

### Inference Flowchart

```
New Audio Input
      ↓
Audio Validation
      ↓
Preprocessing:
  • Load audio
  • Resample to 16kHz
  • Normalize
  • Convert to tensor
      ↓
Model Loading
      ↓
Forward Pass:
  • Encoder processing
  • Decoder processing
  • CTC/RNN-T decoding
      ↓
Post-processing:
  • Text cleaning
  • Confidence scoring
      ↓
Output Transcription
```

### Inference Components

#### Audio Preprocessing
```python
def preprocess_audio(audio_path):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Normalize
    audio = librosa.util.normalize(audio)

    # Convert to tensor
    audio_tensor = torch.tensor(audio).unsqueeze(0)

    return audio_tensor
```

#### Model Inference
```python
def transcribe_audio(model, audio_tensor):
    with torch.no_grad():
        # Move to device
        audio_tensor = audio_tensor.to(model.device)

        # Transcribe
        transcriptions = model.transcribe([audio_tensor])

        return transcriptions[0][0]
```

#### Batch Processing
```python
def batch_transcribe(model, audio_files):
    results = []

    for audio_file in audio_files:
        audio_tensor = preprocess_audio(audio_file)
        transcription = transcribe_audio(model, audio_tensor)

        results.append({
            'audio_file': audio_file,
            'transcription': transcription
        })

    return results
```

---

## Configuration Management

### Configuration Hierarchy

```
Base Config (base_config.yaml)
      ↓
Task-Specific Config (konkani_finetune.yaml)
      ↓
Runtime Overrides (command line)
      ↓
Final Configuration (OmegaConf)
```

### Key Configuration Sections

#### Model Configuration
```yaml
model:
  tokenizertype: "bpe"
  encoder:
    n_layers: 12
    d_model: 256
  decoder:
    vocabulary: null
  joint:
    jointnet:
      d_model: 256
```

#### Training Configuration
```yaml
trainer:
  devices: 1
  max_epochs: 50
  accumulate_grad_batches: 4
  val_check_interval: 1.0
```

#### Data Configuration
```yaml
model:
  train_ds:
    manifest_filepath: "data/train.tsv"
    batch_size: 8
    shuffle: true
  val_ds:
    manifest_filepath: "data/val.tsv"
    batch_size: 8
  test_ds:
    manifest_filepath: "data/test.tsv"
    batch_size: 8
```

#### Optimization Configuration
```yaml
optim:
  name: adamw
  lr: 0.001
  weight_decay: 0.0001
  sched:
    name: CosineAnnealing
    warmup_steps: 1000
```

---

## Error Handling & Logging

### Error Types & Handling

#### Data Errors
- **Audio Loading Errors**: File corruption, unsupported formats
- **Text Encoding Errors**: Unicode issues, malformed transcripts
- **Manifest Errors**: Missing files, incorrect formatting

#### Model Errors
- **CUDA Out of Memory**: Batch size reduction, gradient accumulation
- **Model Loading Errors**: Corrupted checkpoints, version mismatches
- **Inference Errors**: Audio preprocessing failures

#### System Errors
- **Disk Space Issues**: Cleanup old checkpoints
- **Network Errors**: Retry logic for downloads
- **Permission Errors**: Path validation

### Logging Architecture

#### Logging Levels
- **DEBUG**: Detailed execution information
- **INFO**: General progress updates
- **WARNING**: Non-critical issues
- **ERROR**: Critical failures
- **CRITICAL**: System-stopping errors

#### Log Outputs
- **Console**: Real-time progress
- **File**: Persistent logs in `results/logs/`
- **TensorBoard**: Training metrics visualization
- **Weights & Biases**: Experiment tracking (optional)

### Monitoring & Alerting

#### Training Monitoring
- Loss convergence tracking
- Gradient explosion detection
- Learning rate scheduling
- Validation performance monitoring

#### System Monitoring
- GPU memory usage
- CPU utilization
- Disk space monitoring
- Network connectivity

---

## System Requirements & Dependencies

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended: 16GB+)
- **RAM**: 16GB system memory
- **Storage**: 50GB for models and datasets
- **CPU**: Multi-core processor for data preprocessing

### Software Dependencies
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **FFmpeg**: For audio processing
- **Git**: For repository management

### Environment Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install FFmpeg
conda install ffmpeg  # or download from ffmpeg.org

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Deployment & Scaling

### Development Environment
- Local GPU workstation
- Jupyter notebooks for experimentation
- VS Code with Python extensions

### Production Deployment
- Docker containers for reproducibility
- Cloud GPU instances (AWS, GCP, Azure)
- REST API for inference service
- Batch processing pipelines

### Scaling Considerations
- **Data Parallelism**: Multiple GPUs for training
- **Model Parallelism**: Large model distribution
- **Batch Processing**: High-throughput inference
- **Caching**: Model loading optimization

---

This architecture provides a complete, modular, and scalable solution for Konkani ASR fine-tuning, leveraging state-of-the-art deep learning frameworks while maintaining simplicity and maintainability.</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\ARCHITECTURE_GUIDE.md