#!/usr/bin/env python3
"""
Download additional tnshenoy@gmail.com recordings (stories 19, 20, 21)
and create extended deaf speech manifests for DS-B experiment.

Existing data (story 22, 124 samples) remains unchanged in data/deaf_speech/.
New audio goes to data/deaf_speech_extended/audio/.

Output manifests:
  data/deaf_speech_extended/train/manifest.jsonl  — 124 + ~64 additional (85%)
  data/deaf_speech_extended/dev/manifest.jsonl    — 124 + ~11 additional (15%)
  data/deaf_speech_extended/test/manifest.jsonl   — 124 (story 22 only, unchanged)
"""
import os, sys, json, random, requests
from pathlib import Path
from tqdm import tqdm

BASE_URL = "https://deafspeechcollector-production.up.railway.app"
TARGET_USER = "tnshenoy@gmail.com"
SEED = 42
TRAIN_FRAC = 0.85

def download_file(url, path):
    r = requests.get(url, stream=True, timeout=30)
    r.raise_for_status()
    with open(path, 'wb') as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

def main():
    random.seed(SEED)

    # Fetch all approved recordings
    print("Fetching recordings from Railway...")
    r = requests.get(f"{BASE_URL}/api/recordings?limit=500", timeout=15)
    r.raise_for_status()
    all_recs = [x for x in r.json() if x.get('status') == 'approved']
    print(f"Total approved: {len(all_recs)}")

    # Additional recordings: tnshenoy, stories 19/20/21 only
    additional = [x for x in all_recs
                  if x.get('user_id') == TARGET_USER
                  and x.get('story_id') in (19, 20, 21)]
    print(f"Additional recordings (stories 19,20,21): {len(additional)}")

    # Download additional audio
    audio_dir = Path("data/deaf_speech_extended/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(additional)} additional audio files...")
    for rec in tqdm(additional, desc="Downloading"):
        out = audio_dir / f"{rec['id']}.wav"
        if out.exists():
            continue
        try:
            download_file(f"{BASE_URL}/api/recordings/{rec['id']}/audio", out)
        except Exception as e:
            print(f"  WARN: failed {rec['id']}: {e}")

    # Split additional into 85% train / 15% dev
    random.shuffle(additional)
    n_train = round(len(additional) * TRAIN_FRAC)
    add_train = additional[:n_train]
    add_dev   = additional[n_train:]
    print(f"Additional split: {len(add_train)} train / {len(add_dev)} dev")

    # Load existing story-22 manifests
    def load_manifest(path):
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    orig_train = load_manifest("data/deaf_speech/train/manifest.jsonl")
    orig_dev   = load_manifest("data/deaf_speech/dev/manifest.jsonl")
    orig_test  = load_manifest("data/deaf_speech/test/manifest.jsonl")

    # Build additional manifest entries
    def make_entry(rec, split, idx):
        return {
            "audio_filepath": f"data/deaf_speech_extended/audio/{rec['id']}.wav",
            "text": rec.get('sentence_text', ''),
            "duration": rec.get('duration', 0),
            "lang": "mr",
            "sample_id": f"{split}_ext_{idx:04d}",
            "speaker_id": rec.get('user_id', 'unknown'),
            "story_id": rec.get('story_id'),
        }

    ext_train = [make_entry(r, 'train', i) for i, r in enumerate(add_train)]
    ext_dev   = [make_entry(r, 'dev',   i) for i, r in enumerate(add_dev)]

    # Write combined manifests
    for split, base, extra in [('train', orig_train, ext_train),
                                ('dev',   orig_dev,   ext_dev),
                                ('test',  orig_test,  [])]:
        out_dir = Path(f"data/deaf_speech_extended/{split}")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "manifest.jsonl"
        combined = base + extra
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(json.dumps(e, ensure_ascii=False) for e in combined))
        print(f"  {split}: {len(combined)} samples → {out_path}")

    print("\nDone. Summary:")
    print(f"  DS-B train : {len(orig_train)} (orig) + {len(ext_train)} (new) = {len(orig_train)+len(ext_train)}")
    print(f"  DS-B dev   : {len(orig_dev)} (orig) + {len(ext_dev)} (new) = {len(orig_dev)+len(ext_dev)}")
    print(f"  DS-B test  : {len(orig_test)} (story 22, unchanged)")

if __name__ == "__main__":
    main()
