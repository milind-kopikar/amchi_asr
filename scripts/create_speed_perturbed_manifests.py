#!/usr/bin/env python3
"""
Generate speed-perturbed audio files and manifests for DS-C experiment.

Reads the extended train manifest (data/deaf_speech_extended/train/manifest.jsonl)
and creates 3 versions of each audio file at speeds 0.9x, 1.0x (original), 1.1x.

Output:
  data/deaf_speech_augmented/audio/<id>_sp0.9.wav
  data/deaf_speech_augmented/audio/<id>_sp1.0.wav  (copy of original)
  data/deaf_speech_augmented/audio/<id>_sp1.1.wav
  data/deaf_speech_augmented/train/manifest.jsonl   (3x entries)

Uses torchaudio.functional.speed (available in torchaudio >= 0.12).
"""
import json
import shutil
from pathlib import Path
import torch
import torchaudio
from tqdm import tqdm

SPEEDS = [0.9, 1.0, 1.1]
INPUT_MANIFEST = "data/deaf_speech_extended/train/manifest.jsonl"
OUTPUT_AUDIO_DIR = Path("data/deaf_speech_augmented/audio")
OUTPUT_MANIFEST = Path("data/deaf_speech_augmented/train/manifest.jsonl")
SAMPLE_RATE = 16000


def speed_perturb(waveform: torch.Tensor, orig_sr: int, speed: float) -> torch.Tensor:
    """Apply speed perturbation using torchaudio, resample back to orig_sr."""
    if abs(speed - 1.0) < 1e-6:
        return waveform
    # torchaudio.functional.speed changes sample rate proportionally
    out, new_sr = torchaudio.functional.speed(waveform, orig_freq=orig_sr, factor=speed)
    # Resample back to original sample rate so duration is correct
    out = torchaudio.functional.resample(out, orig_freq=new_sr, new_freq=orig_sr)
    return out


def main():
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_MANIFEST, encoding="utf-8") as f:
        entries = [json.loads(l) for l in f if l.strip()]

    print(f"Input entries: {len(entries)} → output: {len(entries) * len(SPEEDS)} (×{len(SPEEDS)} speeds)")

    out_entries = []
    for entry in tqdm(entries, desc="Speed-perturbing"):
        src_path = Path(entry["audio_filepath"])
        if not src_path.exists():
            print(f"  WARN: missing {src_path}, skipping")
            continue

        waveform, sr = torchaudio.load(str(src_path))
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        stem = src_path.stem  # e.g. "131" or "train_ext_0001"

        for speed in SPEEDS:
            tag = f"sp{speed:.1f}"
            out_name = f"{stem}_{tag}.wav"
            out_path = OUTPUT_AUDIO_DIR / out_name

            if not out_path.exists():
                perturbed = speed_perturb(waveform, SAMPLE_RATE, speed)
                torchaudio.save(str(out_path), perturbed, SAMPLE_RATE)

            # Compute actual duration from waveform
            if abs(speed - 1.0) < 1e-6:
                duration = entry.get("duration", 0)
            else:
                info = torchaudio.info(str(out_path))
                duration = info.num_frames / info.sample_rate

            out_entry = dict(entry)
            out_entry["audio_filepath"] = str(out_path)
            out_entry["duration"] = round(duration, 3)
            out_entry["sample_id"] = f"{entry.get('sample_id', stem)}_{tag}"
            out_entries.append(out_entry)

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(e, ensure_ascii=False) for e in out_entries))

    print(f"\nDone.")
    print(f"  Speed-perturbed audio : {OUTPUT_AUDIO_DIR}/ ({len(out_entries)} files)")
    print(f"  Augmented manifest    : {OUTPUT_MANIFEST} ({len(out_entries)} entries)")
    print(f"  Speeds applied        : {SPEEDS}")


if __name__ == "__main__":
    main()
