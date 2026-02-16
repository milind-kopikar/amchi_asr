#!/usr/bin/env python3
"""
Validation script for Konkani ASR fine-tuning progress
Tests model performance at different training stages
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd
from typing import Dict, List

def load_transcriptions(json_file: str) -> Dict:
    """Load transcription results from JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def calculate_simple_wer(reference: str, hypothesis: str) -> float:
    """Calculate simple Word Error Rate"""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    # Simple WER calculation (not perfect but good enough for validation)
    errors = 0
    for i, ref_word in enumerate(ref_words):
        if i >= len(hyp_words) or ref_word != hyp_words[i]:
            errors += 1

    return errors / len(ref_words)

def calculate_simple_cer(reference: str, hypothesis: str) -> float:
    """Calculate simple Character Error Rate"""
    ref_chars = list(reference.strip().replace(' ', ''))
    hyp_chars = list(hypothesis.strip().replace(' ', ''))

    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0

    errors = sum(1 for a, b in zip(ref_chars, hyp_chars) if a != b)
    errors += abs(len(ref_chars) - len(hyp_chars))  # Account for length differences

    return errors / len(ref_chars)

def validate_model_performance(model_path: str, test_audio_dir: str, reference_transcripts: Dict[str, str]):
    """Validate model performance on test data"""

    print(f"🔍 Validating model: {model_path}")

    # Run inference on test data
    output_file = f"validation_{Path(model_path).stem}_results.json"
    cmd = f"python scripts/infer.py --model_path {model_path} --audio_dir {test_audio_dir} --output_file {output_file}"
    os.system(cmd)

    # Load results
    results = load_transcriptions(output_file)
    if not results:
        print("❌ No transcription results found")
        return None

    # Calculate metrics
    total_wer = 0.0
    total_cer = 0.0
    valid_samples = 0

    print("\n📊 Individual Sample Results:")
    print("-" * 60)

    for result in results:
        audio_file = Path(result['audio_file']).name
        transcription = result['transcription']

        # Find reference transcript
        base_name = Path(audio_file).stem
        reference = reference_transcripts.get(base_name, "")

        if not reference:
            print(f"⚠️ No reference transcript for {audio_file}")
            continue

        # Calculate metrics
        wer = calculate_simple_wer(reference, transcription)
        cer = calculate_simple_cer(reference, transcription)

        total_wer += wer
        total_cer += cer
        valid_samples += 1

        print(f"📝 {base_name}")
        print(".1f")
        print(".1f")
        print(f"   Ref: {reference[:50]}...")
        print(f"   Hyp: {transcription[:50]}...")
        print()

    if valid_samples == 0:
        print("❌ No valid samples for evaluation")
        return None

    avg_wer = total_wer / valid_samples
    avg_cer = total_cer / valid_samples

    print("📈 Overall Metrics:")
    print(".1f")
    print(".1f")

    # Save detailed results
    validation_results = {
        'model_path': model_path,
        'num_samples': valid_samples,
        'average_wer': avg_wer,
        'average_cer': avg_cer,
        'individual_results': results
    }

    result_file = f"validation_{Path(model_path).stem}_summary.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, ensure_ascii=False, indent=2)

    print(f"💾 Detailed results saved to: {result_file}")

    return validation_results

def compare_model_versions(model_paths: List[str], test_audio_dir: str, reference_transcripts: Dict[str, str]):
    """Compare performance across different model versions"""

    print("🔄 Comparing Model Versions")
    print("=" * 50)

    all_results = []

    for model_path in model_paths:
        if os.path.exists(model_path):
            result = validate_model_performance(model_path, test_audio_dir, reference_transcripts)
            if result:
                all_results.append(result)
        else:
            print(f"⚠️ Model not found: {model_path}")

    if not all_results:
        print("❌ No valid results to compare")
        return

    # Create comparison table
    print("\n📊 Model Comparison:")
    print("-" * 70)
    print("<30")
    print("-" * 70)

    for result in all_results:
        model_name = Path(result['model_path']).stem
        wer = result['average_wer'] * 100
        cer = result['average_cer'] * 100
        samples = result['num_samples']
        print("<30")

    # Save comparison
    comparison = {
        'models_compared': [r['model_path'] for r in all_results],
        'results': all_results,
        'best_model': min(all_results, key=lambda x: x['average_wer'])['model_path']
    }

    with open('model_comparison.json', 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Comparison saved to: model_comparison.json")
    best_model = comparison['best_model']
    print(f"🏆 Best performing model: {Path(best_model).name}")

def progressive_validation(base_model: str, checkpoints_dir: str, test_audio_dir: str, reference_transcripts: Dict[str, str]):
    """Validate model at different training checkpoints"""

    print("📈 Progressive Validation")
    print("=" * 50)

    # Get all checkpoints
    checkpoints = list(Path(checkpoints_dir).glob("*.nemo"))
    checkpoints.sort(key=lambda x: x.stat().st_mtime)  # Sort by modification time

    if not checkpoints:
        print("❌ No checkpoints found")
        return

    # Add base model
    model_paths = [base_model] + [str(cp) for cp in checkpoints]

    print(f"Found {len(checkpoints)} checkpoints + base model")

    # Compare all versions
    compare_model_versions(model_paths, test_audio_dir, reference_transcripts)

def load_reference_transcripts(transcript_dir: str) -> Dict[str, str]:
    """Load reference transcripts from directory"""

    transcripts = {}
    transcript_dir = Path(transcript_dir)

    if not transcript_dir.exists():
        print(f"❌ Transcript directory not found: {transcript_dir}")
        return transcripts

    for txt_file in transcript_dir.glob("*.txt"):
        base_name = txt_file.stem
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                transcripts[base_name] = f.read().strip()
        except Exception as e:
            print(f"⚠️ Error reading {txt_file}: {e}")

    print(f"✅ Loaded {len(transcripts)} reference transcripts")
    return transcripts

def main():
    parser = argparse.ArgumentParser(description="Validate Konkani ASR model performance")
    parser.add_argument("--model_path", help="Single model to validate")
    parser.add_argument("--model_dir", help="Directory with multiple models to compare")
    parser.add_argument("--checkpoints_dir", help="Directory with training checkpoints")
    parser.add_argument("--base_model", help="Base model path for comparison")
    parser.add_argument("--test_audio_dir", required=True, help="Directory with test audio files")
    parser.add_argument("--transcript_dir", required=True, help="Directory with reference transcripts")

    args = parser.parse_args()

    # Load reference transcripts
    reference_transcripts = load_reference_transcripts(args.transcript_dir)
    if not reference_transcripts:
        print("❌ No reference transcripts loaded")
        return

    if args.model_path:
        # Validate single model
        validate_model_performance(args.model_path, args.test_audio_dir, reference_transcripts)

    elif args.model_dir:
        # Compare models in directory
        model_paths = [str(p) for p in Path(args.model_dir).glob("*.nemo")]
        if model_paths:
            compare_model_versions(model_paths, args.test_audio_dir, reference_transcripts)
        else:
            print(f"❌ No .nemo files found in {args.model_dir}")

    elif args.checkpoints_dir and args.base_model:
        # Progressive validation
        progressive_validation(args.base_model, args.checkpoints_dir, args.test_audio_dir, reference_transcripts)

    else:
        print("❌ Please specify either:")
        print("   --model_path for single model validation")
        print("   --model_dir for model comparison")
        print("   --checkpoints_dir + --base_model for progressive validation")

if __name__ == "__main__":
    main()