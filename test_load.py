#!/usr/bin/env python3
"""
Test loading the JSON data
"""

import json

# Load test data
with open('nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json', encoding='utf-8') as f:
    data = json.load(f)

print(f"Successfully loaded {len(data['per_sample'])} samples")
print("First sample:")
print(f"Reference: {data['per_sample'][0]['reference']}")
print(f"Prediction: {data['per_sample'][0]['prediction']}")
print(f"WER: {data['per_sample'][0]['wer']}")