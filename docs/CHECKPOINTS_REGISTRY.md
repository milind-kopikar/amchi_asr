# Checkpoint Registry — Amchi ASR

**Single source of truth for all trained model checkpoints.**
All production checkpoints are stored in Cloudflare R2 (`asr-checkpoints` bucket).
The public base URL is: `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev`

---

## R2 Connection Details

| Field | Value |
|---|---|
| Account ID | `c90f9011c5a59d5bf40c808f40e3e34b` |
| Endpoint | `https://c90f9011c5a59d5bf40c808f40e3e34b.r2.cloudflarestorage.com` |
| Bucket | `asr-checkpoints` |
| Public base URL | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev` |
| Credentials | See `.env` → `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY` |

Upload script: `scripts/upload_checkpoint_to_r2.py` (uses `boto3`, S3-compatible API).

---

## Amchi Konkani Checkpoints

These use the AI4Bharat IndicConformer base model (Marathi, 499MB).

### Run C — Story-based split, frozen encoder (2026-03-07)
| Field | Value |
|---|---|
| Test WER | **49.1%** (104 samples, Story 5) |
| Val WER (best) | 50.4% (epoch 66) |
| Epochs | 100 |
| Encoder | Frozen (132K trainable params) |
| Train data | Stories 1,2,3,7 → train; Story 4 → dev; Story 5 → test |
| Config | `configs/amchi_konkani_run_c.yaml` |
| Local checkpoint | `results/run_c_story_split/checkpoints/konkani_asr-epoch=66-val_wer=0.504.ckpt` |
| R2 key | `results/run_c_story_split/checkpoints/konkani_asr-epoch=66-val_wer=0.504.ckpt` |
| R2 public URL | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_story_split/checkpoints/konkani_asr-epoch=66-val_wer=0.504.ckpt` |
| Results | `results/experiments/run_c_story_split/final_test_results.json` |

### Run S — Speaker-stratified split, frozen encoder (2026-03-07) ⭐ BEST KONKANI
| Field | Value |
|---|---|
| Test WER | **34.1%** (99 samples, stratified test set) |
| Val WER (best) | 33.4% (epoch 88) |
| Epochs | 100 |
| Encoder | Frozen (132K trainable params) |
| Train data | 70/15/15 per-speaker split; all 3 test speakers represented in train |
| Config | `configs/amchi_konkani_run_c_stratified.yaml` |
| Local checkpoint | `results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt` |
| R2 key | `results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt` |
| R2 public URL | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/run_c_stratified_split/checkpoints/konkani_asr-epoch=88-val_wer=0.334.ckpt` |
| Results | `results/experiments/run_c_stratified_split/final_test_results.json` |

---

## Deaf Speech Checkpoints

These also use the AI4Bharat IndicConformer base model (Marathi). All experiments use story_id=22 ("दैनंदिन कामे १"), 124 recordings from a single deaf speaker group.

### Baseline — Full fine-tune, 50 epochs (2026-03-01)
| Field | Value |
|---|---|
| Test WER | **75.3%** (124 samples, same as train) |
| Val WER (best) | 72.0% (epoch 21) |
| Epochs | 50 |
| Encoder | Full fine-tune |
| Train data | 124 story-4 samples (= dev = test) |
| Config | `configs/deaf_speech_story4_50epoch.yaml` |
| R2 key | `nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` |
| R2 public URL | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/nemo_experiments/deaf_speech_story4_50epoch/checkpoints/konkani_asr-epoch=21-val_wer=0.720.ckpt` |
| Results | `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/` |

### DS-A — Frozen encoder, 100 epochs (2026-03-07)
| Field | Value |
|---|---|
| Test WER | **79.6%** (worse than baseline — frozen encoder hurts deaf speech) |
| Val WER (best) | 76.6% (epoch 96) |
| Epochs | 100 |
| Encoder | Frozen |
| Config | `configs/deaf_speech_frozen_encoder_100epoch.yaml` |
| Results | `results/experiments/deaf_speech_dsa/final_test_results.json` |
| Note | Checkpoint not uploaded to R2 (not production-worthy) |

### DS-B — Full fine-tune, extended data (2026-03-07)
| Field | Value |
|---|---|
| Test WER | **93.1%** (worse — out-of-distribution extra data hurt) |
| Val WER (best) | 85.0% (epoch 75) |
| Train data | 188 samples: 124 story-4 + 63 tnshenoy stories 19/20/21 |
| Config | `configs/deaf_speech_full_finetune_extended_100epoch.yaml` |
| Results | `results/experiments/deaf_speech_dsb/final_test_results.json` |
| Note | Checkpoint not uploaded to R2 (not production-worthy) |

### DS-D — Full fine-tune, speed-perturbed baseline (2026-03-07) ⭐ BEST DEAF SPEECH
| Field | Value |
|---|---|
| Test WER | **34.7%** (40.6pp improvement over baseline) |
| Val WER (best) | 26.9% (epoch 96) |
| Epochs | 100 |
| Encoder | Full fine-tune |
| Train data | 372 samples = 124 story-4 × 3 speed factors (0.9×, 1.0×, 1.1×) |
| Config | `configs/deaf_speech_sp_baseline_100epoch.yaml` |
| Local checkpoint | `results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |
| R2 key | `results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |
| R2 public URL | `https://pub-9686b04ab1a94aad9688b9fb104d51ca.r2.dev/results/deaf_speech_dsd/checkpoints/konkani_asr-epoch=96-val_wer=0.269.ckpt` |
| Results | `results/experiments/deaf_speech_dsd/final_test_results.json` |

---

## Base Model

| Field | Value |
|---|---|
| Name | AI4Bharat IndicConformer STT Marathi Hybrid CTC-RNNT Large |
| HuggingFace | `ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large` |
| Local path | `models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo` |
| Size | ~499MB |
| Tokenizer | `tokenizers/marathi_tokenizer.model` (extracted from .nemo) |
| Download script | `scripts/download_model_from_hf.py` |

---

## How to Upload a New Checkpoint

```bash
cd /workspace/amchi_asr
source .env   # must have R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY

python3 scripts/upload_checkpoint_to_r2.py \
  --file results/<run_dir>/checkpoints/<best>.ckpt \
  --public-url
```

The R2 key mirrors the local path (e.g. `results/my_run/checkpoints/best.ckpt`).
Add the resulting public URL to this registry.
