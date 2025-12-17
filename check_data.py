import json

print("="*80)
print("TRAIN MANIFEST - First 3 samples")
print("="*80)
with open("data/train/manifest.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        sample = json.loads(line)
        print(f"\n{i+1}. Audio: {sample['audio_filepath']}")
        print(f"   Text: {sample['text']}")
        print(f"   Duration: {sample['duration']}s")

print("\n" + "="*80)
print("DEV MANIFEST - First 3 samples")
print("="*80)
with open("data/dev/manifest.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        sample = json.loads(line)
        print(f"\n{i+1}. Audio: {sample['audio_filepath']}")
        print(f"   Text: {sample['text']}")
        print(f"   Duration: {sample['duration']}s")

print("\n" + "="*80)
print("FILE COUNTS")
print("="*80)
import os
train_audio = len([f for f in os.listdir("data/train/audio") if f.endswith(".wav")])
dev_audio = len([f for f in os.listdir("data/dev/audio") if f.endswith(".wav")])
print(f"Train audio files: {train_audio}")
print(f"Dev audio files: {dev_audio}")
