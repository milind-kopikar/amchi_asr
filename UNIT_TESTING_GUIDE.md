# Unit Testing Guide for Konkani ASR System

## Building Block Breakdown & Unit Tests

### Block 1: Data Management Layer
**Purpose**: Handle all data preparation and preprocessing

#### Components & Responsibilities:
- **Audio Preprocessor**: Normalizes audio files (resampling, normalization)
- **Text Processor**: Cleans and normalizes Konkani transcripts
- **Manifest Generator**: Creates NeMo-compatible training manifests
- **Data Splitter**: Creates train/validation/test splits

#### Key Files:
- `scripts/prepare_data.py` - Main data preparation script
- `scripts/prepare_recording.py` - Audio recording utilities

---

## 🔬 Unit Test Cases for Data Management

### Test 1.1: Audio Preprocessing
```python
def test_audio_preprocessing():
    """Test audio loading, resampling, and normalization"""

    # Input: Raw audio file (44kHz, stereo)
    input_audio = "test_audio.wav"  # 44kHz, 2 channels, 3 seconds

    # Expected output: 16kHz mono normalized audio
    processed_audio = preprocess_audio(input_audio)

    # Assertions
    assert processed_audio.shape == (48000,)  # 3 seconds * 16000 Hz
    assert processed_audio.dtype == np.float32
    assert -1.0 <= processed_audio.min() <= processed_audio.max() <= 1.0
    assert librosa.get_samplerate(input_audio) == 16000

    print("✅ Audio preprocessing works correctly")
```

### Test 1.2: Text Processing
```python
def test_text_processing():
    """Test transcript cleaning and normalization"""

    # Input: Raw Konkani text with issues
    raw_text = "तुमी    कसो आसा??\n\nमाझे नाव  मिलिंद!"

    # Expected output: Clean normalized text
    clean_text = clean_transcript(raw_text)

    # Assertions
    assert clean_text == "तुमी कसो आसा? माझे नाव मिलिंद!"
    assert "\n" not in clean_text
    assert "  " not in clean_text  # No double spaces
    assert clean_text.endswith("!")

    print("✅ Text processing works correctly")
```

### Test 1.3: Manifest Generation
```python
def test_manifest_generation():
    """Test creation of training manifest files"""

    # Input: Audio files and transcripts
    audio_files = ["audio1.wav", "audio2.wav"]
    transcripts = ["तुमी कसो आसा?", "माझे नाव मिलिंद आहे."]

    # Expected output: TSV manifest file
    manifest_path = create_manifest(audio_files, transcripts, "test_manifest.tsv")

    # Read and verify manifest
    df = pd.read_csv(manifest_path, sep='\t')

    # Assertions
    assert len(df) == 2
    assert list(df.columns) == ['audio_filepath', 'text', 'duration']
    assert df['text'].iloc[0] == "तुमी कसो आसा?"
    assert df['duration'].iloc[0] > 0  # Valid duration

    print("✅ Manifest generation works correctly")
```

### Test 1.4: Data Splitting
```python
def test_data_splitting():
    """Test train/validation/test split creation"""

    # Input: List of audio-transcript pairs (100 samples)
    data_pairs = [("audio_{}.wav".format(i), "transcript_{}".format(i))
                  for i in range(100)]

    # Expected output: Three manifest files with correct proportions
    splits = split_data(data_pairs, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1)

    # Assertions
    assert len(splits['train']) == 70
    assert len(splits['val']) == 20
    assert len(splits['test']) == 10

    # Check no overlap between splits
    train_files = set(splits['train'])
    val_files = set(splits['val'])
    test_files = set(splits['test'])

    assert len(train_files & val_files) == 0
    assert len(train_files & test_files) == 0
    assert len(val_files & test_files) == 0

    print("✅ Data splitting works correctly")
```

---

## Block 2: Model Management Layer
**Purpose**: Handle model downloading, storage, and versioning

#### Components & Responsibilities:
- **Model Downloader**: Downloads IndicConformer from Hugging Face
- **Model Loader**: Loads models for training/inference
- **Checkpoint Manager**: Saves training checkpoints
- **Model Validator**: Verifies model integrity

#### Key Files:
- `scripts/download_model.py` - Model acquisition
- `models/` directory - Model storage

---

## 🔬 Unit Test Cases for Model Management

### Test 2.1: Model Downloading
```python
def test_model_download():
    """Test IndicConformer model download from Hugging Face"""

    # Input: Model identifier
    model_id = "ai4bharat/indicconformer_marathi"

    # Expected output: Local model directory with files
    model_path = download_model(model_id, "test_models/")

    # Assertions
    assert os.path.exists(model_path)
    assert os.path.exists(os.path.join(model_path, "model_config.yaml"))
    assert os.path.exists(os.path.join(model_path, "tokenizer.model"))

    # Check model size (should be substantial)
    model_size = get_directory_size(model_path)
    assert model_size > 100 * 1024 * 1024  # > 100MB

    print("✅ Model download works correctly")
```

### Test 2.2: Model Loading
```python
def test_model_loading():
    """Test loading ASR model for inference"""

    # Input: Downloaded model path
    model_path = "models/indicconformer_marathi/"

    # Expected output: Loaded NeMo model object
    model = load_asr_model(model_path)

    # Assertions
    assert model is not None
    assert hasattr(model, 'transcribe')
    assert hasattr(model, 'freeze')

    # Check model is on correct device
    if torch.cuda.is_available():
        assert next(model.parameters()).device.type == 'cuda'
    else:
        assert next(model.parameters()).device.type == 'cpu'

    print("✅ Model loading works correctly")
```

### Test 2.3: Checkpoint Saving/Loading
```python
def test_checkpoint_management():
    """Test saving and loading model checkpoints"""

    # Input: Trained model and checkpoint path
    model = create_test_model()  # Helper function
    checkpoint_path = "test_checkpoint.nemo"

    # Save checkpoint
    save_checkpoint(model, checkpoint_path)

    # Assertions for saved checkpoint
    assert os.path.exists(checkpoint_path)
    assert os.path.getsize(checkpoint_path) > 0

    # Load checkpoint
    loaded_model = load_checkpoint(checkpoint_path)

    # Assertions for loaded model
    assert loaded_model is not None

    # Compare model parameters (should be identical)
    original_params = list(model.parameters())
    loaded_params = list(loaded_model.parameters())

    for orig, loaded in zip(original_params, loaded_params):
        assert torch.equal(orig, loaded)

    print("✅ Checkpoint management works correctly")
```

---

## Block 3: Training Pipeline
**Purpose**: Execute the fine-tuning process

#### Components & Responsibilities:
- **Trainer**: Core training orchestration
- **Optimizer**: Learning rate and parameter optimization
- **Validator**: Validation during training
- **Logger**: Training metrics and progress tracking

#### Key Files:
- `scripts/fine_tune.py` - Main training script
- `configs/konkani_finetune.yaml` - Training configuration

---

## 🔬 Unit Test Cases for Training Pipeline

### Test 3.1: Configuration Loading
```python
def test_config_loading():
    """Test loading and validation of training configuration"""

    # Input: YAML configuration file
    config_path = "configs/konkani_finetune.yaml"

    # Expected output: Parsed configuration object
    config = load_training_config(config_path)

    # Assertions
    assert config.model.tokenizertype == "bpe"
    assert config.trainer.max_epochs > 0
    assert config.model.train_ds.batch_size > 0
    assert config.optim.lr > 0

    # Check required fields exist
    required_fields = ['model', 'trainer', 'optim']
    for field in required_fields:
        assert hasattr(config, field)

    print("✅ Configuration loading works correctly")
```

### Test 3.2: Data Loader Creation
```python
def test_data_loader_creation():
    """Test creation of training data loaders"""

    # Input: Training manifest and configuration
    manifest_path = "data/train.tsv"
    config = load_config("configs/konkani_finetune.yaml")

    # Expected output: PyTorch DataLoader
    train_loader = create_data_loader(manifest_path, config, mode='train')

    # Assertions
    assert train_loader is not None
    assert hasattr(train_loader, '__iter__')

    # Check batch structure
    batch = next(iter(train_loader))
    assert 'audio' in batch
    assert 'text' in batch
    assert 'audio_lengths' in batch

    # Check batch size
    assert batch['audio'].shape[0] == config.model.train_ds.batch_size

    print("✅ Data loader creation works correctly")
```

### Test 3.3: Single Training Step
```python
def test_single_training_step():
    """Test one forward/backward pass of training"""

    # Input: Model, batch of data, optimizer
    model = create_test_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # Create test batch
    batch = create_test_batch(batch_size=2)

    # Expected output: Loss value and updated parameters
    initial_params = [p.clone() for p in model.parameters()]

    loss = training_step(model, batch, optimizer)

    # Assertions
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0  # Loss should be positive
    assert loss.requires_grad == False  # Detached loss

    # Check parameters were updated
    final_params = list(model.parameters())
    params_changed = False
    for initial, final in zip(initial_params, final_params):
        if not torch.equal(initial, final):
            params_changed = True
            break
    assert params_changed, "Model parameters should have changed"

    print("✅ Single training step works correctly")
```

### Test 3.4: Validation Step
```python
def test_validation_step():
    """Test validation metrics calculation"""

    # Input: Model and validation batch
    model = create_test_model()
    val_batch = create_test_batch(batch_size=2)

    # Expected output: Validation metrics
    metrics = validation_step(model, val_batch)

    # Assertions
    assert 'wer' in metrics
    assert 'cer' in metrics
    assert isinstance(metrics['wer'], (int, float))
    assert isinstance(metrics['cer'], (int, float))
    assert 0 <= metrics['wer'] <= 1  # WER as percentage
    assert 0 <= metrics['cer'] <= 1  # CER as percentage

    print("✅ Validation step works correctly")
```

---

## Block 4: Evaluation System
**Purpose**: Measure model performance and accuracy

#### Components & Responsibilities:
- **WER Calculator**: Word Error Rate computation
- **CER Calculator**: Character Error Rate computation
- **BLEU Scorer**: Translation quality assessment
- **Comparative Analyzer**: Model comparison utilities

#### Key Files:
- `scripts/evaluate.py` - Evaluation script
- `scripts/validate_model.py` - Advanced validation

---

## 🔬 Unit Test Cases for Evaluation System

### Test 4.1: WER Calculation
```python
def test_wer_calculation():
    """Test Word Error Rate calculation"""

    # Input: Reference and hypothesis texts
    reference = "तुमी कसो आसा"
    hypothesis = "तुमी कसो आसा"  # Perfect match

    # Expected output: WER = 0.0
    wer = calculate_wer(reference, hypothesis)

    # Assertions
    assert wer == 0.0

    # Test with errors
    hypothesis_with_errors = "तुमी कसा आसा"  # 1 word different
    wer_with_errors = calculate_wer(reference, hypothesis_with_errors)

    assert wer_with_errors > 0.0
    assert wer_with_errors <= 1.0

    print("✅ WER calculation works correctly")
```

### Test 4.2: CER Calculation
```python
def test_cer_calculation():
    """Test Character Error Rate calculation"""

    # Input: Reference and hypothesis texts
    reference = "नमस्कार"
    hypothesis = "नमस्कार"  # Perfect match

    # Expected output: CER = 0.0
    cer = calculate_cer(reference, hypothesis)

    # Assertions
    assert cer == 0.0

    # Test with character-level errors
    hypothesis_with_errors = "नमस्ते"  # Different characters
    cer_with_errors = calculate_cer(reference, hypothesis_with_errors)

    assert cer_with_errors > 0.0
    assert cer_with_errors <= 1.0

    print("✅ CER calculation works correctly")
```

### Test 4.3: Batch Evaluation
```python
def test_batch_evaluation():
    """Test evaluation on multiple audio-transcript pairs"""

    # Input: Model, test audio files, reference transcripts
    model = load_test_model()
    test_audio_files = ["test1.wav", "test2.wav", "test3.wav"]
    reference_transcripts = {
        "test1.wav": "तुमी कसो आसा",
        "test2.wav": "माझे नाव मिलिंद आहे",
        "test3.wav": "कोकणी भाषा छान आहे"
    }

    # Expected output: Evaluation results dictionary
    results = evaluate_model_batch(model, test_audio_files, reference_transcripts)

    # Assertions
    assert len(results) == 3
    assert 'average_wer' in results
    assert 'average_cer' in results
    assert 'individual_results' in results

    # Check individual results structure
    for result in results['individual_results']:
        assert 'audio_file' in result
        assert 'reference' in result
        assert 'hypothesis' in result
        assert 'wer' in result
        assert 'cer' in result

    print("✅ Batch evaluation works correctly")
```

---

## Block 5: Inference Service
**Purpose**: Transcribe new audio using trained models

#### Components & Responsibilities:
- **Audio Processor**: Real-time audio preprocessing
- **Model Runner**: Execute transcription
- **Result Formatter**: Format output text
- **Batch Processor**: Handle multiple audio files

#### Key Files:
- `scripts/infer.py` - Inference script

---

## 🔬 Unit Test Cases for Inference Service

### Test 5.1: Audio Preprocessing for Inference
```python
def test_inference_audio_preprocessing():
    """Test audio preprocessing for inference"""

    # Input: Raw audio file
    audio_path = "test_audio.wav"

    # Expected output: Preprocessed audio tensor
    audio_tensor = preprocess_audio_for_inference(audio_path)

    # Assertions
    assert isinstance(audio_tensor, torch.Tensor)
    assert audio_tensor.dim() == 2  # [batch_size, audio_length]
    assert audio_tensor.shape[0] == 1  # Batch size 1
    assert audio_tensor.dtype == torch.float32

    # Check audio properties
    audio_length = audio_tensor.shape[1]
    expected_length = int(16000 * 2.5)  # 2.5 seconds at 16kHz
    assert abs(audio_length - expected_length) < 1000  # Allow some tolerance

    print("✅ Inference audio preprocessing works correctly")
```

### Test 5.2: Single Audio Transcription
```python
def test_single_transcription():
    """Test transcription of single audio file"""

    # Input: Model and audio tensor
    model = load_test_model()
    audio_tensor = create_test_audio_tensor()

    # Expected output: Transcribed text
    transcription = transcribe_single_audio(model, audio_tensor)

    # Assertions
    assert isinstance(transcription, str)
    assert len(transcription.strip()) > 0
    assert transcription.replace(" ", "").isalnum()  # Contains text

    # For test model, check if it produces reasonable output
    # (This will depend on your test model setup)

    print("✅ Single audio transcription works correctly")
```

### Test 5.3: Batch Transcription
```python
def test_batch_transcription():
    """Test transcription of multiple audio files"""

    # Input: Model and list of audio files
    model = load_test_model()
    audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]

    # Expected output: List of transcription results
    results = transcribe_batch_audio(model, audio_files)

    # Assertions
    assert len(results) == 3
    assert isinstance(results, list)

    for result in results:
        assert 'audio_file' in result
        assert 'transcription' in result
        assert isinstance(result['transcription'], str)
        assert len(result['transcription'].strip()) > 0

    print("✅ Batch transcription works correctly")
```

### Test 5.4: Result Formatting
```python
def test_result_formatting():
    """Test formatting of transcription results"""

    # Input: Raw transcription results
    raw_results = [
        {"audio_file": "test1.wav", "transcription": "  तुमी कसो आसा  "},
        {"audio_file": "test2.wav", "transcription": "माझे नाव मिलिंद आहे!!"}
    ]

    # Expected output: Formatted results
    formatted_results = format_transcription_results(raw_results)

    # Assertions
    assert len(formatted_results) == 2

    # Check text cleaning
    assert formatted_results[0]['transcription'] == "तुमी कसो आसा"
    assert formatted_results[1]['transcription'] == "माझे नाव मिलिंद आहे!"

    # Check structure preservation
    assert formatted_results[0]['audio_file'] == "test1.wav"

    print("✅ Result formatting works correctly")
```

---

## Block 6: Configuration Management
**Purpose**: Centralized configuration handling

#### Components & Responsibilities:
- **YAML Parser**: Load training configurations
- **Parameter Validator**: Validate configuration values
- **Environment Manager**: Handle system-specific settings

#### Key Files:
- `configs/*.yaml` - Configuration files
- `scripts/setup_environment.py` - Environment validation

---

## 🔬 Unit Test Cases for Configuration Management

### Test 6.1: YAML Configuration Loading
```python
def test_yaml_config_loading():
    """Test loading YAML configuration files"""

    # Input: YAML configuration file path
    config_path = "configs/test_config.yaml"

    # Expected output: Configuration object
    config = load_yaml_config(config_path)

    # Assertions
    assert config is not None
    assert hasattr(config, 'model')
    assert hasattr(config, 'trainer')
    assert hasattr(config, 'optim')

    # Check specific values
    assert config.trainer.max_epochs == 10
    assert config.model.train_ds.batch_size == 8

    print("✅ YAML configuration loading works correctly")
```

### Test 6.2: Configuration Validation
```python
def test_config_validation():
    """Test validation of configuration parameters"""

    # Input: Configuration object
    config = create_test_config()

    # Expected output: Validation result (True/False)
    is_valid, errors = validate_configuration(config)

    # Assertions
    assert is_valid == True
    assert len(errors) == 0

    # Test invalid configuration
    config.trainer.max_epochs = -1  # Invalid negative value
    is_valid, errors = validate_configuration(config)

    assert is_valid == False
    assert len(errors) > 0
    assert "max_epochs" in str(errors[0])

    print("✅ Configuration validation works correctly")
```

### Test 6.3: Environment Detection
```python
def test_environment_detection():
    """Test detection of system environment capabilities"""

    # Expected output: Environment information dictionary
    env_info = detect_environment()

    # Assertions
    assert 'python_version' in env_info
    assert 'cuda_available' in env_info
    assert 'gpu_count' in env_info
    assert 'total_memory' in env_info

    # Check data types
    assert isinstance(env_info['cuda_available'], bool)
    assert isinstance(env_info['gpu_count'], int)
    assert env_info['python_version'].startswith('3.')

    print("✅ Environment detection works correctly")
```

---

## 🧪 Running the Unit Tests

### Test Organization
```
tests/
├── test_data_management.py    # Tests 1.1-1.4
├── test_model_management.py   # Tests 2.1-2.3
├── test_training_pipeline.py  # Tests 3.1-3.4
├── test_evaluation_system.py  # Tests 4.1-4.3
├── test_inference_service.py  # Tests 5.1-5.4
├── test_config_management.py  # Tests 6.1-6.3
└── conftest.py                # Shared test fixtures
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Run specific test block
python -m pytest tests/test_data_management.py

# Run with verbose output
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_data_management.py::test_audio_preprocessing -v
```

### Test Dependencies
- `pytest` for test framework
- Test audio files and transcripts
- Pre-trained model for testing
- Mock objects for external dependencies

This comprehensive unit testing approach will help you verify each building block works correctly and make debugging much easier when issues arise.</content>
<parameter name="filePath">C:\Users\Milind Kopikare\Code\amchi_konkani\konkani_asr\UNIT_TESTING_GUIDE.md