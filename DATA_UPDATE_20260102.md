Railway data refresh — 2026-01-02

Summary:
- Pulled approved recordings from Railway using story-based split (stories 1/2/3 -> train, story 5 -> dev, story 4 -> test)
- Backup of previous `data/` directory saved as `data_backup_20260102_...` (timestamped)

Counts:
- Approved recordings found: 549
  - Train: 473
  - Dev: 38
  - Test: 38

Notes:
- User expected 519 approved audio files; the Railway API returned 549 approved recordings at time of download. Please confirm if we should filter to 519 samples or proceed with 549.
- `data/train/manifest.jsonl`, `data/dev/manifest.jsonl`, `data/test/manifest.jsonl` were created and contain paths like `data/dev/audio/<id>.wav` and transcription text.
- Updated smoke test manifest `tiny_one_sample.jsonl` to point to the first dev sample (`data/dev/audio/570.wav`).

Next steps (suggested):
1. Run preflight checks: `python scripts/preflight_checks.py`
2. Run a quick verification train+inference smoke: `./scripts/extended_smoke_test.sh` (this will use `tiny_one_sample.jsonl` for the tiny smoke run)
3. If you want us to filter the dataset down to 519 approved recordings, provide the filtering rule (e.g., remove recordings with known corrupted IDs) and I will apply it.

Artifacts:
- `data/` updated in place (not committed to git). Backup stored as `data_backup_*` in the workspace root.

Timestamp: 2026-01-02T22:30:00Z