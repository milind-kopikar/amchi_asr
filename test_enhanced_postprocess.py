#!/usr/bin/env python3
"""
Test the enhanced post-processing algorithm on a few samples
"""

import json
import os
import sys
sys.path.append('scripts')

from postprocess_asr import postprocess_sample
import google.genai as genai

# Test samples from the deaf speech results
test_samples = [
    {
        "audio": "data/deaf_speech/audio/130.wav",
        "reference": "दैनंदिन कामे १।",
        "prediction": "पि ⁇ ",
        "wer": 1.0
    },
    {
        "audio": "data/deaf_speech/audio/131.wav",
        "reference": "दूध किती आहे?।",
        "prediction": "ू किती ⁇ ",
        "wer": 0.6666666666666666
    },
    {
        "audio": "data/deaf_speech/audio/132.wav",
        "reference": "एक लिटर दूध द्या.।",
        "prediction": "हे दू द्या ⁇ ",
        "wer": 0.75
    }
]

def main():
    # Check for API key
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        return

    # Init Gemini
    client = genai.Client(api_key=api_key)

    print("Testing enhanced post-processing algorithm...")
    print("=" * 60)

    for i, sample in enumerate(test_samples):
        ref = sample["reference"]
        pred = sample["prediction"]
        original_wer = sample["wer"]

        print(f"\nSample {i+1}:")
        print(f"Reference: {ref}")
        print(f"ASR Prediction: {pred}")
        print(f"Original WER: {original_wer:.2f}")

        # Run post-processing
        result = postprocess_sample(client, pred, original_wer=original_wer)

        print(f"Mode: {result['mode']}")
        print(f"Corrected: {result['corrected']}")
        print(f"Word labels: {result['word_labels']}")

        # Calculate new WER (simplified)
        if result['corrected']:
            # Simple character-based similarity for demo
            ref_clean = ref.replace("।", "").strip()
            corr_clean = result['corrected'].replace("।", "").strip()
            similarity = len(set(ref_clean) & set(corr_clean)) / len(set(ref_clean) | set(corr_clean))
            print(f"Estimated improvement: {similarity:.2f}")

if __name__ == "__main__":
    main()