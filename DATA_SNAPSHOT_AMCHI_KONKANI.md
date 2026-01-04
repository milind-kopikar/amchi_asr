# Data Snapshot: Amchi Konkani (Marathi Base)
**Date:** January 4, 2026
**Project State:** Completed Marathi Pilot (WER 0.351)

This document records the data configuration used for the Amchi Konkani / Marathi Pilot experiments before switching to the Deaf Speech dataset.

## 1. Dataset Composition
The dataset consists of Marathi/Konkani stories split by story ID.

| Split | Source | Sample Count | Manifest Path |
|-------|--------|--------------|---------------|
| **Train** | Stories 1, 2, 3 | 472 | `data/train/manifest.jsonl` |
| **Dev** | Story 4 | 38 | `data/dev/manifest.jsonl` |
| **Test** | Story 5 | 38 | `data/test/manifest.jsonl` |

## 2. Manifest Format (NeMo JSONL)
Each entry follows this structure:
```json
{"audio_filepath": "data/train/audio/145.wav", "text": "चल रे भोपळा टुनुक टुनुक", "duration": 4.26, "lang": "mr", "sample_id": "train_0000"}
```

## 3. Audio Specifications
- **Format:** WAV (PCM 16-bit)
- **Sample Rate:** 16,000 Hz
- **Channels:** Mono
- **Location:** `data/[train|dev|test]/audio/`

## 4. Reproduction Steps for this Split
If the `data/` folder is overwritten, this specific split can be recreated by:
1. Collecting all story manifests.
2. Using `scripts/swap_manifests.py` to ensure Story 4 is assigned to `dev` and Story 5 is assigned to `test`.
3. Verifying that `max_duration` in the training config is set to at least `30.0` to avoid filtering the longer story segments.

## 5. Key Metrics Achieved
- **Model:** `indicconformer_stt_mr_hybrid_ctc_rnnt_large`
- **WER (Test):** 0.351
- **CER (Test):** 0.142
- **Epochs:** 20
