Railway Data Manifest Summary — 2026-01-02

This file describes the offline metadata about the datasets that were pulled from the Railway API on 2026-01-02. The **audio files themselves are not stored in the repository**. This document records counts, manifest checksums, and backup information so that future agents can verify consistency or re-download the same dataset.

Summary
-------
- Total approved recordings returned by Railway API: 549
- Story-based split applied (stories 1,2,3 -> train; story 5 -> dev; story 4 -> test)

Counts (post-download)
----------------------
- Train manifests: 472 entries
- Dev manifests: 37 entries
- Test manifests: 37 entries
- Manifest total: 546 lines (some recordings may not have had duration or entries excluded; see `download_data_from_railway.py` logs)

Checksum (SHA-256)
------------------
- `data/train/manifest.jsonl`: 23bcee94f9cf9a4e593dd38e3267e023ecce1a6ee185436873ac4cd71f8b198a
- `data/dev/manifest.jsonl`: 9c04bf279d5ddfe6778547694a9012a1b796da9182c37dd66c86ea16400c8ab8
- `data/test/manifest.jsonl`: 6e0875b73acd34810e6341bd923cab301e6b22102d6665987467ebad83ba9dc6

Notes & Actions
---------------
- You previously expected **519** approved audio files; the Railway API returned **549** at the time of download. Please confirm whether you want us to (a) accept all 549 samples, (b) apply a filter to reduce to exactly 519 (provide filter criteria), or (c) remove known-corrupted IDs (if you can supply a list).

- A backup of the previous `data/` directory was saved before replacement; the backup directory is present in the workspace root with a timestamped name like `data_backup_20260102_XXXXXX`.

- The smoke-test tiny manifest (`tiny_one_sample.jsonl`) was updated to point to the first dev sample (`data/dev/audio/570.wav`). This manifest is committed to GitHub.

Verification Steps (to re-run locally)
--------------------------------------
1. Re-create the same dataset download (deterministic story-split):
   python3 scripts/download_data_from_railway.py --use_story_split --output_dir data

2. Verify manifest checksums match the values above:
   sha256sum data/train/manifest.jsonl data/dev/manifest.jsonl data/test/manifest.jsonl

3. Verify the smoke sample exists and plays:
   aplay data/dev/audio/570.wav  # or use ffplay

Commit & Location
-----------------
- This file: `DATA_MANIFEST_SUMMARY_20260102.md` (committed to repo)
- Backup location (not committed): workspace root (e.g., `data_backup_20260102_...`)

If you'd like I can also:
- Apply a deterministic filter to reduce the dataset to exactly 519 records (if you supply filtering rules or a list of IDs to exclude), or
- Produce a CSV listing of all audio IDs and their story_id / durations for easy inspection.

Timestamp: 2026-01-02T22:45:00Z