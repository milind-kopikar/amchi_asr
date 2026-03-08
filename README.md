# RunPod inference for Amchi ASR

- **Serverless:** `handler.py` — use with RunPod Serverless. Full deploy and test: **[RUNPOD_SERVERLESS_DEPLOY.md](../RUNPOD_SERVERLESS_DEPLOY.md)**. See also [RUNPOD_INFERENCE_ENDPOINT.md](../RUNPOD_INFERENCE_ENDPOINT.md). Set `CHECKPOINT_PATH` in the environment to your `.ckpt` path.
- **Persistent pod:** `app_fastapi.py` — run on the same pod where you trained:  
  `uvicorn runpod.app_fastapi:app --host 0.0.0.0 --port 8000`  
  Then `POST /transcribe` with 16 kHz mono WAV (multipart file or `audio_base64` in JSON).

Both use the shared loader and transcription in `scripts/amchi_inference.py`.

## 🚀 Enhanced Post-Processing for Deaf Speech Recognition

**Major Achievement**: Our enhanced ASR post-processing algorithm achieved a **13.3 percentage point improvement** in Word Error Rate (WER) for Marathi deaf speech recognition, reducing WER from 75.3% to 62.0% (+17.6% relative improvement).

### Key Features
- **Three-mode algorithm**: RECONSTRUCT, FILL, and PASSTHROUGH modes based on word classification
- **High-frequency vocabulary guidance**: 45 domain-specific Marathi words for LLM reconstruction
- **Domain expertise**: Optimized for everyday transactional scenarios (shopping, transportation, daily routines)
- **Robust evaluation**: Tested on 124 deaf speech samples with incremental result saving

### Usage
```bash
# Process test results with enhanced post-processing
python scripts/postprocess_asr.py \
    --input path/to/test_results.json \
    --output enhanced_results.json \
    --report postprocess_report.txt

# Full evaluation with incremental saving
python robust_evaluation.py
```

📖 **Complete Documentation**: See [ENHANCED_POSTPROCESSING_METHOD.md](ENHANCED_POSTPROCESSING_METHOD.md) for detailed implementation, results, and usage instructions.
