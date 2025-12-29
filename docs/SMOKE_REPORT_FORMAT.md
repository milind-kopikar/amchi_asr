# Smoke Report Format (for publications)

This document specifies the output format produced by `scripts/generate_smoke_report.py` so results are consistent and ready to be included in papers.

Primary output files (in `--output_dir`):

- `smoke_report.json` (UTF-8, human readable) — top-level JSON with the following schema:

  {
    "model_id": "facebook/mms-1b-all",
    "architecture": "wav2vec2",
    "base_language": "kok",
    "overall_wer": 0.832,
    "average_latency_seconds": 0.245,
    "samples": [
      {
        "audio": "data_smoke/dev/audio/307.wav",
        "predicted": "पाव वाट ...",
        "reference": "पाव वाट ...",
        "wer": 0.00,
        "latency_seconds": 0.212
      },
      ...
    ]
  }

- `smoke_report.csv` — a CSV (UTF-8) with columns: `audio,predicted,reference,wer,latency_seconds`.

- `smoke_report.md` — a Markdown table ready for copy/paste into papers or README.

Recommendations for paper tables
- Use `smoke_report.json` to compute aggregate metrics (overall WER, median/95th percentile latency)
- For the per-sentence table in the paper include columns:
  - Audio link (file path or hosted URL)
  - Reference (Devanagari)
  - Predicted (Devanagari)
  - Per-sentence WER
  - Latency (ms)

Notes on latency
- The script measures wall-clock time per sample (audio load + model forward). For production latency estimates, measure the server end-to-end round-trip time (client → server → model → response) using the deployed endpoint under target load.

