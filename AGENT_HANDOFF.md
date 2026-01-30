# AGENT HANDOFF — Resume instructions for next agent

Date: 2026-01-04 (Late Evening Update)

---

## 1) Short summary (one line)
Marathi Story Pilot (WER 0.351 -> **0.213** with Post-Processing) and Deaf Speech Multi-User Pilot (WER 0.948) are COMPLETE. MIT THINK Proposal drafted.

**Recent activity (2026-01-04):**
- **Post-Processing Breakthrough:** Implemented dictionary-based correction (`scripts/post_process_konkani.py`) which reduced Marathi Pilot v3 WER from 0.351 to **0.213** (39% improvement).
- **MIT THINK Proposal:** Drafted research proposal and preliminary analysis report. Generated plots (`wer_comparison.png`) for the application.
- **Marathi Pilot Success:** Achieved base WER 0.351. Post-processed to **WER 0.213 / CER 0.042** on Story 5.
- **Deaf Speech Track:**
  - **1-User Pilot:** Trained on 75 samples from a single deaf user (`tnshenoy@gmail.com`). Achieved WER 0.97.
  - **Multi-User Pilot:** Trained on 101 samples from all approved users on Railway. Achieved **WER 0.948**.
  - **Data Acquisition:** Created `scripts/download_deaf_speech.py` to pull and split data from the Railway API.
  - **Learnings:** Diversity (multi-user) is showing better generalization than single-user data, even at small scales.
- **Results Persisted:** 
  - Marathi Story: `nemo_experiments/marathi_pilot_v3/`
  - Deaf Speech (1-User): `nemo_experiments/marathi_deaf_1user_75samples/`
  - Deaf Speech (Multi-User): `nemo_experiments/marathi_deaf_multi_user_101samples/`

---

## 2) Current state & context
- **Success:** Both standard Marathi and Deaf Speech pilots are finished and synced to GitHub.
- **Data:** 
  - Standard: `data/train`, `data/dev`, `data/test`
  - Deaf (All Users): `data_all_users/train`, `data_all_users/dev`, `data_all_users/test`
- **Tokenizer:** Marathi tokenizer is at `tokenizers/marathi_tokenizer.model`.
- **Environment:** RunPod environment is stable. AI4Bharat NeMo fork is installed and patched.

## 3) Objective for you (next agent)
1. **Deaf Speech Scaling:** We need more data. The current trend suggests 500-1000 samples are needed for usable WER.
2. **Data Augmentation:** Experiment with speed/pitch perturbation in `configs/marathi_deaf_multi_user_50epoch.yaml` to simulate more deaf speech variations.
3. **Konkani Scaling:** Apply the same protocol to the Konkani model (`models/konkani_model.nemo`).

## 4) Files & locations you will use
- **Training Script:** `scripts/fine_tune.py`
- **Data Script:** `scripts/download_deaf_speech.py` (Pulls from Railway API)
- **Configs:** 
  - `configs/marathi_pilot_20epoch.yaml` (Standard)
  - `configs/marathi_deaf_multi_user_50epoch.yaml` (Deaf)
- **Post-Processing:**
  - `scripts/extract_predictions.py`
  - `scripts/post_process_konkani.py`
  - `scripts/evaluate_post_processed.py`
  - `post_process_metrics.json`
- **Proposal Artifacts:**
  - `mit_think_research_proposal.md`
  - `preliminary_metrics_report.md`
  - `wer_comparison.png`
- **Results:** `nemo_experiments/`

## 5) Verified Commands
```bash
# Download all approved deaf speech data
python3 scripts/download_deaf_speech.py --output_dir data_all_users

# Run Multi-User Deaf Speech Training
export APPLY_CONV_PATCH=1
python3 scripts/fine_tune.py --config configs/marathi_deaf_multi_user_50epoch.yaml
```

## 6) Recovery / Restart Guide 🆘
If the RunPod restarts:
1. Run `bash setup_env.sh`.
2. Ensure `export APPLY_CONV_PATCH=1` is set before running any NeMo scripts.
3. Results in `results/` are ignored by git; always move successful runs to `nemo_experiments/` for persistence.

---

import sentencepiece as spm
s=spm.SentencePieceProcessor()
s.load('tokenizers/konkani_tokenizer.model')
print('tokenizer size', s.get_piece_size())
PY
```
Expect a reasonable token size (e.g., 1024 or similar) and no load errors.

### C. Create Konkani Config
Create a new config `configs/konkani_1epoch_ctc.yaml` based on `configs/tmp_marathi_1epoch_ctc_golden.yaml`.
- Update `model.nemo_model` to `models/konkani_model.nemo`.
- Update `model.tokenizer.model_path` to `tokenizers/konkani_tokenizer.model`.
- Ensure `loss.loss_name` is `default` (or handled by `fine_tune.py`).

### D. Run Konkani Smoke Test
Use the same robust pipeline script (you may need to adapt it or run commands manually):
```bash
# 1. Verify Data
python3 scripts/verify_manifest_audio.py configs/konkani_1epoch_ctc.yaml

# 2. Run Fine-Tuning
python3 scripts/fine_tune.py --config configs/konkani_1epoch_ctc.yaml --output_dir results/smoke_test_konkani

# 3. Run Inference
# Find checkpoint
CHECKPOINT=$(find results/smoke_test_konkani/checkpoints -name "*.ckpt" | head -n 1)
# Use a Konkani audio file (story0.txt has text, need corresponding audio if available, or use any wav)
TEST_AUDIO="data/dev/audio/some_konkani_file.wav" 

python3 scripts/smoke_test_inference.py --checkpoint "$CHECKPOINT" --audio "$TEST_AUDIO"
```

## 8) Acceptance criteria (how to mark PASS)
- PASS if **train-loss** reduces by >= 50% over the run OR normalized **char-distance** ≤ 0.2
- The micro-overfit script currently enforces this and will exit non-zero on failure; see its logs and produced `samples_epoch_*.json` for predicted strings and distances.

## 9) Troubleshooting notes (common issues & fixes)
- If restore fails with errors about missing `multisoftmax`, `language_keys`, or unexpected kwargs:
  - Ensure the environment used the **AI4Bharat NeMo fork** (use `setup_env.sh` or `pip install -e /workspace/NeMo_ai4bharat` in a venv).
  - If the fork is installed and errors persist, try `EncDecHybridRNNTCTCBPEModel.restore_from(..., strict=False)` and inspect the stack trace to identify unsupported fields.
- If tokenizer errors show `KeyError: 'dir'` or cannot find tokenizer files:
  - Confirm `configs/konkani_finetune.yaml` points to `tokenizers/konkani_tokenizer.model` and that the file exists.
  - Alternatively set `MICRO_SKIP_MODEL_RESTORE=1` to instantiate model from a sanitized local `model_config.yaml` (not necessary for Konkani, but helpful when debugging other .nemo files).
- If the micro-overfit script fails early with missing manifest paths, edit the referenced `model_config.yaml` or the config to point to `tiny_train.jsonl` / `tiny_val.jsonl` local test manifests.

## 10) Logging & artifacts to collect
- `results/experiments/<timestamp>/` folder: copy `samples_epoch_*.json`, metrics, `hparams.yaml` and `lightning_logs` as artifacts.
- Save final model checkpoints (if any) and the run config used to `results/konkani_runs/` with a short metadata file (`who`, `what`, `why`, `nemo_sha256`) for reproducibility.

## 11) If all else fails — fallback options
- If the Konkani .nemo fails to restore (unexpected), you can:
  1. Try `MICRO_SKIP_MODEL_RESTORE=1` and instantiate from a sanitized `model_config.yaml` (patch local tokenizer paths and remove unsupported fields).
  2. As a last resort, recreate a tiny synthetic training pass (we already have a synthetic micro-overfit smoke) to validate infra.

## 12) Useful references in the repo
- `setup_env.sh` and `scripts/ensure_env.sh` — environment provisioning
- `scripts/run_micro_overfit.py` — driver for the overfit tests and acceptance checks
- `scripts/fine_tune.py` — actual training & restore logic
- `models/backups/` — storage of previous working .nemo backups
- `models/indicconformer_mr/unpacked/` — where tokenizer artifacts were extracted previously

---

### Checklist (mark when done)
- [ ] Downloaded Konkani model & tokenizer
- [ ] Verified tokenizer loads and token count
- [ ] Updated and validated `configs/konkani_finetune.yaml`
- [ ] Smoke micro-overfit (3 epochs) executed and passed
- [ ] Full 20‑epoch overfit executed and artifacts collected
- [ ] Summary added to PR/Issue with links to logs & artifacts

---

If you want, I can also create an automated `make` target or a small wrapper script that performs steps A→F to speed up repeating this work in future.

---

## Update: 6‑epoch run (in progress) — 2026-01-02

**Summary:** I launched a 6‑epoch run using `configs/full_6epoch_lr1e-4.yaml` with output dir `nemo_experiments/full_6epoch_lr1e-4`.

- Progress: Training completed epochs **0 → 4** (epoch numbering starts at 0). Epoch 4 hit a disk quota error while saving and the run aborted. The `exp_manager` was modified to `resume_if_exists: true` so the run can be resumed cleanly.
- Checkpoints currently present (local):
  - `amchi_marathi_full_6epoch_lr1e-4-epoch=02-val_loss=81.633.ckpt` (~1.4G)
  - `amchi_marathi_full_6epoch_lr1e-4-epoch=03-val_loss=55.738.ckpt` (~1.4G)
  - `amchi_marathi_full_6epoch_lr1e-4-epoch=04-val_loss=45.202.ckpt` (~1.0G)
  - `last.ckpt` (~1.4G)
- To free space so the run could continue, I **deleted** the older checkpoints for epoch 00 and 01 (they were ~1.4G each). This freed enough space for training to continue, but the user cancelled the resumed run; see logs for the disk quota error that caused abort.
- Metrics for the partial run are recorded here:
  - `nemo_experiments/full_6epoch_lr1e-4/experiments/<timestamp>/epoch_metrics.csv` (contains epochs 0–4 with `val_loss`, `val_wer`, `val_cer`)
  - Quick snapshot (from the run):
    - epoch 0: val_loss=158.876, val_wer=32.90
    - epoch 1: val_loss=130.277, val_wer=1.00
    - epoch 2: val_loss=81.633, val_wer=1.00
    - epoch 3: val_loss=55.738, val_wer=0.946
    - epoch 4: val_loss=45.202, val_wer=0.977

### Next-agent actions to finish & tidy (step-by-step)
1. **Pull repo & verify files**
```bash
# Pull latest changes (I committed the new config & this handoff update)
git pull origin master
```
2. **Verify disk space & optionally archive**
```bash
# Check free space
df -h .
# If disk near full, archive older experiments to a separate location or object storage
mkdir -p /workspace/amchi_asr/archives
# Example archiving of an experiment dir (change to the right path)
tar -czf /workspace/amchi_asr/archives/checkpoints_full6_$(date +%s).tar.gz nemo_experiments/full_6epoch_lr1e-4/checkpoints
sha256sum /workspace/amchi_asr/archives/checkpoints_full6_*.tar.gz
```
3. **Resume training** (config already has `resume_if_exists: true`, and I removed the two worst early ckpts to free space)
```bash
python3 scripts/fine_tune.py --config configs/full_6epoch_lr1e-4.yaml --output_dir nemo_experiments/full_6epoch_lr1e-4
```
- Watch the logs and ensure epochs 5 and 6 get executed and saved. If saving fails again, archive or delete older checkpoints (`rm`) to free space and retry.

4. **After the run finishes (or once you have >=6 epochs), select best 3 epochs to keep**
- Evaluate metrics: `nemo_experiments/full_6epoch_lr1e-4/experiments/<timestamp>/epoch_metrics.csv`
- Choose best 3 rows by `val_wer` (lowest) with `val_loss` as tie-breaker. Example helper:
```bash
python - <<'PY'
import csv
rows = list(csv.DictReader(open('nemo_experiments/full_6epoch_lr1e-4/experiments/`ls -1 nemo_experiments/full_6epoch_lr1e-4/experiments | tail -n1`/epoch_metrics.csv')))
rows = [r for r in rows if r.get('val_wer') and r['val_wer']!='']
rows_sorted = sorted(rows, key=lambda r: (float(r['val_wer']), float(r['val_loss'])))
best = rows_sorted[:3]
print('best_epochs=', [int(r['epoch']) for r in best])
PY
```
- Keep the ckpts for the three epoch numbers returned and `last.ckpt` if you prefer, then delete the others to free space. Example to delete others:
```bash
# Replace 2 3 4 with the chosen epochs
KEEP=(2 3 4)
for f in nemo_experiments/full_6epoch_lr1e-4/checkpoints/*.ckpt; do
  skip=0
  for e in "${KEEP[@]}"; do
    if echo "$f" | grep -q "epoch=$e-"; then skip=1; break; fi
done
  if [ "$skip" -eq 0 ]; then rm -v "$f"; fi
done
```

5. **Archive the kept checkpoints** (recommended)
```bash
mkdir -p /workspace/amchi_asr/archives
tar -czf /workspace/amchi_asr/archives/full6_best_checkpoints_$(date +%s).tar.gz nemo_experiments/full_6epoch_lr1e-4/checkpoints/amchi_marathi_full_6epoch_lr1e-4-epoch=0*_val_loss=*.ckpt
sha256sum /workspace/amchi_asr/archives/full6_best_checkpoints_*.tar.gz
```

6. **Run decoding experiments or quick inference** to validate the best checkpoint(s):
```bash
# Use smoke_test_inference.py; replace CHECKPOINT with path to the kept ckpt
python3 scripts/smoke_test_inference.py --checkpoint "nemo_experiments/full_6epoch_lr1e-4/checkpoints/amchi_marathi_full_6epoch_lr1e-4-epoch=03-val_loss=55.738.ckpt" --audio data/dev/audio/379.wav --device cuda
```

7. **Push relevant changes / artifacts to GitHub**
- I committed `configs/full_6epoch_lr1e-4.yaml` and this `AGENT_HANDOFF.md` update; make sure to `git pull` and `git push` any further changes. If you generate additional small metadata (e.g., `run_metadata.txt`), commit and push it as well for reproducibility.

---

## RunPod stop / resume checklist (cost-savings) ⚠️

If you need to stop the RunPod instance to save compute costs but preserve artifacts and state, follow these steps. Storage is persistent and inexpensive; compute is expensive — so stop compute when idle.

1. Before stopping the instance (on the RunPod UI or via the instance's SSH):
   - Ensure all important artifacts are on the persistent volume:
     - `nemo_experiments/experiments/<timestamp>/` — epoch metrics, samples, final_test_results.json
     - `nemo_experiments/checkpoints/` — saved `.ckpt` files
     - `results/` — any run artifacts
   - Optionally, create a small metadata file about the run:
     ```bash
     echo "best_checkpoint=$(ls nemo_experiments/checkpoints | sort | tail -n1)" > run_metadata.txt
     echo "completed_at=$(date -u --iso-8601=seconds)" >> run_metadata.txt
     git add run_metadata.txt && git commit -m "Add run metadata for runpod stop" || true
     ```
2. Stop compute (recommended via RunPod UI):
   - Use the RunPod control panel to **Stop / Power Off** the instance (this preserves attached storage).
   - Alternative (if you are on the machine and want to shutdown):
     ```bash
     sudo shutdown -h now
     ```
3. When you start the instance back up (or create a new RunPod instance and attach the same persistent volume):
   - Start the instance and `ssh` in.
   - Pull latest repo state: `git pull origin master`
   - Recreate the Python environment (`venv`) if required or activate the existing one.
   - Verify artifacts exist:
     ```bash
     ls -la nemo_experiments/checkpoints
     ls -la nemo_experiments/experiments
     cat run_metadata.txt || true
     ```
   - Run a quick verification (fast):
     ```bash
     # Verify environment (fast preflight steps)
     python scripts/preflight_checks.py
     # Sanity inference using best checkpoint and a dev sample
     python scripts/smoke_test_inference.py --checkpoint $(ls nemo_experiments/checkpoints | sort | tail -n1 | sed "s|^|nemo_experiments/checkpoints/|") --audio data/dev/audio/379.wav --device cuda
     ```
4. Next actions for a new agent on resume:
   - If the run was stopped intentionally: decide whether to resume training (change `configs/*.yaml` and re-run `python scripts/fine_tune.py`) or start a new experiment (create new output dir in `nemo_experiments`).
   - If disk space is low, archive older experiment dirs: `tar -czf archive_20260102.tar.gz nemo_experiments/experiments/20260102_*` and move to a separate storage bucket if you want.

**Note:** Leave `run_metadata.txt` and the most recent `nemo_experiments` directory intact so the next agent can verify the run and continue from the checkpoint.

---

## What the next agent should know (short recap)
- The critical fixes are in `scripts/fine_tune.py` (WER fix + trainer/Logger fix). The extended smoke test completed and produced checkpoints in `nemo_experiments/checkpoints`.
- Use the robust scripts for repeatability: `scripts/extended_smoke_test.sh` and `scripts/smoke_test_inference.py`.
- If you stop/restart a RunPod instance, run the quick verification step above (preflight + a single-sample inference) before attempting to resume training.

---

```}avascript