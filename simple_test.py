#!/usr/bin/env python3
"""
Simple test of the enhanced post-processing algorithm
"""

import json
import sys
sys.path.append('scripts')

from postprocess_asr import postprocess_sample
import google.genai as genai

# Load test data
with open('nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json', encoding='utf-8') as f:
    data = json.load(f)

# Test on remaining samples (starting from sample 21)
samples = data['per_sample'][20:]

# Initialize Gemini
client = genai.Client(api_key="AIzaSyAwBr6FlR2nXTDyWqI8dBIhKBXeugND-Gw")

improved_count = 0
total_wer_before = 0
total_wer_after = 0

print("Testing enhanced post-processing on first 5 samples:")
print("=" * 80)

for i, sample in enumerate(samples):
    print(f"\nSample {i+1}:")
    print(f"Reference: {sample['reference']}")
    print(f"Prediction: {sample['prediction']}")
    print(f"Original WER: {sample['wer']:.2f}")

    # Run post-processing with error handling
    try:
        result = postprocess_sample(client, sample['prediction'], original_wer=sample['wer'])
        # Add delay to avoid rate limiting
        import time
        time.sleep(1)
    except Exception as e:
        print(f"Error processing sample: {e}")
        result = {'mode': 'ERROR', 'corrected': sample['prediction']}

    print(f"Mode: {result['mode']}")
    print(f"Corrected: {result['corrected']}")

    # Simple WER calculation (approximate)
    ref_clean = sample['reference'].replace("।", "").strip()
    pred_clean = sample['prediction'].replace("⁇", "").strip()
    corr_clean = result['corrected'].replace("।", "").strip() if result['corrected'] else pred_clean

    # Character-level similarity as proxy for WER
    ref_chars = set(ref_clean)
    pred_chars = set(pred_clean)
    corr_chars = set(corr_clean)

    pred_similarity = len(ref_chars & pred_chars) / len(ref_chars | pred_chars) if ref_chars or pred_chars else 1.0
    corr_similarity = len(ref_chars & corr_chars) / len(ref_chars | corr_chars) if ref_chars or corr_chars else 1.0

    wer_before = 1.0 - pred_similarity
    wer_after = 1.0 - corr_similarity

    print(f"WER before: {wer_before:.2f}, after: {wer_after:.2f}, delta: {wer_before - wer_after:+.2f}")

    total_wer_before += wer_before
    total_wer_after += wer_after

    if wer_after < wer_before:
        improved_count += 1
        print("✓ IMPROVED")
    elif wer_after > wer_before:
        print("✗ WORSENED")
    else:
        print("~ UNCHANGED")

print(f"\n{'='*80}")
print("SUMMARY (all samples):")
print(f"Improved: {improved_count}/{len(samples)}")
print(f"Average WER before: {total_wer_before/len(samples):.2f}")
print(f"Average WER after: {total_wer_after/len(samples):.2f}")
print(f"Average improvement: {(total_wer_before - total_wer_after)/len(samples):+.2f}")