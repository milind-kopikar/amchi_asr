#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Ensure model presence and optionally download from HF
# Usage: scripts/ensure_model_present.sh [--model marathi|konkani|path/to/custom.nemo] [--yes]

set -euo pipefail

MODEL="marathi"
AUTO_YES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --yes|-y)
      AUTO_YES=1
      shift
      ;;
    --help|-h)
      echo "Usage: $0 [--model marathi|konkani|/path/to/custom.nemo] [--yes]"
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# Resolve expected model path
case "$MODEL" in
  marathi)
    MODEL_DIR="models/indicconformer_mr"
    MODEL_FILE="$MODEL_DIR/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo"
    HF_SHORTHAND="marathi"
    ;;
  konkani)
    MODEL_DIR="models/konkani"
    MODEL_FILE="models/konkani_model.nemo"
    HF_SHORTHAND="konkani"
    ;;
  /*|~/*)
    # absolute path
    MODEL_FILE="$MODEL"
    MODEL_DIR="$(dirname "$MODEL_FILE")"
    HF_SHORTHAND="custom"
    ;;
  *)
    # assume user passed a path or a custom name
    MODEL_FILE="$MODEL"
    MODEL_DIR="$(dirname "$MODEL_FILE")"
    HF_SHORTHAND="custom"
    ;;
esac

echo "Checking model presence for: $MODEL_FILE"

if [[ -f "$MODEL_FILE" ]]; then
  echo "✅ Model file exists: $MODEL_FILE"
else
  echo "⚠️  Model file missing: $MODEL_FILE"
  echo "Looking for any .nemo under $MODEL_DIR..."
  if [[ -d "$MODEL_DIR" ]] && ls "$MODEL_DIR"/*.nemo >/dev/null 2>&1; then
    echo "Found alternate .nemo in $MODEL_DIR:" 
    ls -lh "$MODEL_DIR"/*.nemo
    echo "If you want to use a specific file, run: $0 --model /path/to/file.nemo"
    exit 0
  fi

  # Attempt to download if possible
  if [[ "$HF_SHORTHAND" == "custom" ]]; then
    echo "No automatic download available for custom model path. Please place the .nemo under models/ and ensure tokenizer files are present."
    exit 2
  fi

  echo "Attempting to download the model from Hugging Face hub (shorthand: $HF_SHORTHAND)"
  echo "If you haven't authenticated, run: huggingface-cli login or export HF_TOKEN=<token>"

  if [[ $AUTO_YES -eq 0 ]]; then
    read -p "Proceed to download ($HF_SHORTHAND) into $MODEL_DIR? [y/N] " yn
    case "$yn" in
      [Yy]* ) :;;
      * ) echo "Aborting download."; exit 3;;
    esac
  fi

  mkdir -p "$MODEL_DIR"

  # Use the existing download script
  if command -v python3 >/dev/null 2>&1; then
    echo "Running: python3 scripts/download_model.py --model ${HF_SHORTHAND} --output_path ${MODEL_DIR}"
    if python3 scripts/download_model.py --model ${HF_SHORTHAND} --output_path ${MODEL_DIR}; then
      echo "✅ Download completed."
    else
      echo "❌ Download failed. Check your HF token or network connectivity." >&2
      exit 4
    fi
  else
    echo "Python3 is not available; cannot run download script." >&2
    exit 5
  fi
fi

# Post-checks
if [[ -f "$MODEL_FILE" ]]; then
  echo "✔ Model file confirmed: $MODEL_FILE"
  # Check for tokenizer presence near models or tokenizers/
  TOKEN_FOUND=0
  if [[ -f "$MODEL_DIR/tokenizer.model" ]]; then TOKEN_FOUND=1; fi
  if [[ -f "tokenizers/konkani_tokenizer.model" ]]; then TOKEN_FOUND=1; fi
  if [[ $TOKEN_FOUND -eq 1 ]]; then
    echo "✔ Tokenizer file found (local tokenizers or model dir)"
  else
    echo "⚠️  No tokenizer.model found next to model or in tokenizers/ — please ensure you have a local tokenizer matching your training config."
  fi
  exit 0
else
  echo "❌ Model file still missing after attempted download: $MODEL_FILE" >&2
  exit 6
fi
