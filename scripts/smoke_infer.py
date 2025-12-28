#!/usr/bin/env python3
"""
Inference smoke script
Usage: python scripts/smoke_infer.py --model facebook/mms-1b-all --manifest data_smoke/test/manifest.jsonl --lang kok
"""
import argparse
import json
from transformers import pipeline, AutoProcessor, AutoModelForCTC

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='HF model name or local path')
parser.add_argument('--manifest', required=True, help='JSONL manifest with audio_filepath and text')
parser.add_argument('--device', type=int, default=-1, help='Device index for pipeline; -1 for CPU')
parser.add_argument('--limit', type=int, default=5, help='Limit number of samples to transcribe')
parser.add_argument('--lang', type=str, default=None, help='Language code (optional)')
args = parser.parse_args()

print(f"Loading model: {args.model}")
# Use pipeline; Auto classes loaded internally
asr = pipeline('automatic-speech-recognition', model=args.model, device=args.device)

print(f"Reading manifest: {args.manifest}")
count = 0
with open(args.manifest, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        audio = rec.get('audio_filepath')
        expected = rec.get('text','')
        try:
            result = asr(audio)
            predicted = result.get('text','')
        except Exception as e:
            predicted = f"<ERROR: {e}>"
        print('---')
        print(f"Audio: {audio}")
        print(f"Expected: {expected}")
        print(f"Predicted: {predicted}")
        count += 1
        if args.limit and count >= args.limit:
            break

print('\nInference smoke finished')
