#!/usr/bin/env python3
"""
Evaluate fine-tuned Konkani ASR model
Calculate Word Error Rate (WER) and Character Error Rate (CER)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict
import torch
from tqdm import tqdm

# NeMo imports
import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils.exceptions import NeMoBaseException

# Evaluation metrics
try:
    from jiwer import wer, cer
except ImportError:
    print("jiwer not found. Install with: pip install jiwer")
    wer = cer = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path: str):
    """
    Load NeMo ASR model

    Args:
        model_path: Path to .nemo model file

    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading model: {model_path}")
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path)

    # Set model to evaluation mode
    model.eval()

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    else:
        logger.info("Using CPU for inference")

    return model

def load_manifest(manifest_path: str) -> List[Dict]:
    """
    Load manifest file

    Args:
        manifest_path: Path to manifest file

    Returns:
        List of manifest entries
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    entries = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(entries)} entries from {manifest_path}")
    return entries

def transcribe_audio(model, audio_path: str, batch_size: int = 1) -> str:
    """
    Transcribe audio file using the model

    Args:
        model: NeMo ASR model
        audio_path: Path to audio file
        batch_size: Batch size for inference

    Returns:
        Transcribed text
    """
    try:
        # Use NeMo's transcribe method
        transcriptions = model.transcribe([audio_path], batch_size=batch_size)
        return transcriptions[0] if transcriptions else ""
    except Exception as e:
        logger.warning(f"Transcription failed for {audio_path}: {e}")
        return ""

def calculate_wer_cer(predictions: List[str], references: List[str]):
    """
    Calculate Word Error Rate and Character Error Rate

    Args:
        predictions: List of predicted transcripts
        references: List of reference transcripts

    Returns:
        Dictionary with WER and CER scores
    """
    if wer is None or cer is None:
        logger.warning("jiwer not available. Install with: pip install jiwer")
        return {"wer": None, "cer": None}

    try:
        wer_score = wer(references, predictions)
        cer_score = cer(references, predictions)

        return {
            "wer": wer_score,
            "cer": cer_score
        }
    except Exception as e:
        logger.error(f"Error calculating WER/CER: {e}")
        return {"wer": None, "cer": None}

def evaluate_model(model, manifest_entries: List[Dict], batch_size: int = 8) -> Dict:
    """
    Evaluate model on manifest entries

    Args:
        model: NeMo ASR model
        manifest_entries: List of manifest entries
        batch_size: Batch size for inference

    Returns:
        Evaluation results
    """
    logger.info(f"Starting evaluation on {len(manifest_entries)} samples")

    predictions = []
    references = []
    audio_paths = []

    # Process entries
    for entry in tqdm(manifest_entries, desc="Transcribing"):
        audio_path = entry['audio_filepath']
        reference_text = entry['text']

        # Transcribe audio
        prediction = transcribe_audio(model, audio_path, batch_size)

        predictions.append(prediction)
        references.append(reference_text)
        audio_paths.append(audio_path)

    # Calculate metrics
    metrics = calculate_wer_cer(predictions, references)

    # Create detailed results
    results = {
        "summary": {
            "total_samples": len(manifest_entries),
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "model_path": str(model) if hasattr(model, '__str__') else "Unknown"
        },
        "samples": []
    }

    # Add individual sample results
    for i, (audio_path, reference, prediction) in enumerate(zip(audio_paths, references, predictions)):
        sample_result = {
            "index": i,
            "audio_path": audio_path,
            "reference": reference,
            "prediction": prediction,
            "wer": wer([reference], [prediction]) if wer else None,
            "cer": cer([reference], [prediction]) if cer else None
        }
        results["samples"].append(sample_result)

    logger.info(".2%")
    if metrics["wer"] is not None:
        logger.info(".2%")

    return results

def save_results(results: Dict, output_file: str):
    """
    Save evaluation results to JSON file

    Args:
        results: Evaluation results dictionary
        output_file: Output file path
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Konkani ASR model")
    parser.add_argument("--model_path", required=True, help="Path to .nemo model file")
    parser.add_argument("--test_manifest", required=True, help="Path to test manifest file")
    parser.add_argument("--output_file", default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")

    args = parser.parse_args()

    try:
        # Load model
        model = load_model(args.model_path)

        # Load manifest
        manifest_entries = load_manifest(args.test_manifest)

        # Limit samples if specified
        if args.max_samples:
            manifest_entries = manifest_entries[:args.max_samples]
            logger.info(f"Limited to {args.max_samples} samples")

        # Evaluate model
        results = evaluate_model(model, manifest_entries, args.batch_size)

        # Save results
        save_results(results, args.output_file)

        logger.info("Evaluation completed successfully!")

        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {args.model_path}")
        print(f"Test Samples: {summary['total_samples']}")
        if summary['wer'] is not None:
            print(".2%")
        if summary['cer'] is not None:
            print(".2%")
        print(f"Results saved to: {args.output_file}")

    except NeMoBaseException as e:
        logger.error(f"NeMo error during evaluation: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        exit(1)

if __name__ == "__main__":
    main()