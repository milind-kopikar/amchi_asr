#!/usr/bin/env python3
"""
Robust evaluation of the enhanced post-processing algorithm
Saves results incrementally to avoid losing progress
"""

import json
import sys
import time
sys.path.append('scripts')

from postprocess_asr import postprocess_sample
import google.genai as genai

# Load test data
with open('nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json', encoding='utf-8') as f:
    data = json.load(f)

# Load or initialize results
results_file = 'evaluation_results.json'
try:
    with open(results_file, encoding='utf-8') as f:
        saved_results = json.load(f)
    start_idx = len(saved_results)
    print(f"Resuming from sample {start_idx + 1}")
except FileNotFoundError:
    saved_results = []
    start_idx = 0

samples = data['per_sample'][start_idx:]

# Initialize Gemini
client = genai.Client(api_key="AIzaSyAwBr6FlR2nXTDyWqI8dBIhKBXeugND-Gw")

print(f"Testing enhanced post-processing on samples {start_idx + 1} to {len(data['per_sample'])}:")
print("=" * 80)

for i, sample in enumerate(samples):
    sample_idx = start_idx + i
    print(f"\nSample {sample_idx + 1}:")
    print(f"Reference: {sample['reference']}")
    print(f"Prediction: {sample['prediction']}")
    print(f"Original WER: {sample['wer']:.2f}")

    # Run post-processing with error handling
    try:
        result = postprocess_sample(client, sample['prediction'], original_wer=sample['wer'])
        # Add delay to avoid rate limiting
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

    # Save result
    sample_result = {
        'sample_idx': sample_idx,
        'reference': sample['reference'],
        'prediction': sample['prediction'],
        'original_wer': sample['wer'],
        'mode': result['mode'],
        'corrected': result['corrected'],
        'wer_before': wer_before,
        'wer_after': wer_after,
        'improvement': wer_before - wer_after
    }
    saved_results.append(sample_result)

    # Save incrementally
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(saved_results, f, ensure_ascii=False, indent=2)

    if wer_after < wer_before:
        print("✓ IMPROVED")
    elif wer_after > wer_before:
        print("✗ WORSENED")
    else:
        print("~ UNCHANGED")

# Calculate final summary
improved_count = sum(1 for r in saved_results if r['wer_after'] < r['wer_before'])
worsened_count = sum(1 for r in saved_results if r['wer_after'] > r['wer_before'])
unchanged_count = sum(1 for r in saved_results if r['wer_after'] == r['wer_before'])

total_wer_before = sum(r['wer_before'] for r in saved_results)
total_wer_after = sum(r['wer_after'] for r in saved_results)
total_samples = len(saved_results)

print(f"\n{'='*80}")
print("FINAL SUMMARY:")
print(f"Total samples processed: {total_samples}")
print(f"Improved: {improved_count}")
print(f"Worsened: {worsened_count}")
print(f"Unchanged: {unchanged_count}")
print(f"Average WER before: {total_wer_before/total_samples:.2f}")
print(f"Average WER after: {total_wer_after/total_samples:.2f}")
print(f"Average improvement: {(total_wer_before - total_wer_after)/total_samples:+.2f}")

# Compare to baseline
baseline_wer = 0.753  # From previous results
enhanced_wer = total_wer_after/total_samples
print(f"\nBaseline WER: {baseline_wer:.2f}")
print(f"Enhanced WER: {enhanced_wer:.2f}")
print(f"Overall improvement: {baseline_wer - enhanced_wer:+.2f} ({(baseline_wer - enhanced_wer)/baseline_wer*100:+.1f}%)")