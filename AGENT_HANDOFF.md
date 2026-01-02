# AGENT HANDOFF — Resume instructions for next agent

Date: 2026-01-02

---

## 1) Short summary (one line)
Marathi smoke test (Training + Inference) is PASSING. Now switch to Konkani model using the verified pipeline.

## 2) Current state & context
- **Success:** We successfully ran a 1-epoch smoke test on the Marathi model (`models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo`).
- **Tokenizer Fix:** We identified the correct Marathi tokenizer (containing 'ळ') and integrated it.
- **Inference Fix:** We fixed `scripts/smoke_test_inference.py` to handle the Hybrid model checkpoint by:
  - Loading config from checkpoint.
  - Patching `loss_name` to `default`.
  - Removing data loader configs to avoid path errors.
  - Forcing CTC decoding.
- **Pipeline:** `scripts/robust_smoke_test.sh` runs the full end-to-end test (Verify Data -> Train -> Inference) and passes.

## 3) Objective for you (next agent)
1. **Switch to Konkani:** Apply the same pipeline to the Konkani model.
   - Download Konkani model and tokenizer.
   - Create a Konkani config similar to `configs/tmp_marathi_1epoch_ctc_golden.yaml`.
   - Run `scripts/fine_tune.py` and `scripts/smoke_test_inference.py`.
2. **Full Training:** Once Konkani smoke test passes, run the full fine-tuning (more epochs, full data).

## 4) Files & locations you will use
- **Smoke Test Runner:** `scripts/robust_smoke_test.sh` (Reference this for the workflow)
- **Inference Script:** `scripts/smoke_test_inference.py` (Robust logic for Hybrid checkpoints)
- **Training Script:** `scripts/fine_tune.py` (Handles tokenizer mismatch automatically)
- **Notes:** `REPRODUCTION_NOTES.md` (Detailed technical documentation of the fixes)

## 5) Verified Commands
```bash
# Run the robust smoke test (Marathi)
bash scripts/robust_smoke_test.sh
```

## 6) Recovery / Restart Guide 🆘
**If the environment is lost (e.g., Runpod restart), follow these exact steps to restore the current state:**

### A. Setup Environment
```bash
# 1. Clone the repo (if not already present)
git clone https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr

# 2. Install dependencies and patches
bash setup_env.sh
```

### B. Restore Marathi Model & Tokenizer
The smoke test expects the Marathi model and tokenizer to be present.
```bash
# 1. Create directories
mkdir -p models/indicconformer_stt_mr_hybrid_ctc_rnnt_large tokenizers

# 2. Download Marathi Model
wget https://huggingface.co/ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large/resolve/main/indicconformer_stt_mr_hybrid_rnnt_large.nemo \
  -O models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo

# 3. Extract & Install Correct Tokenizer
# We need the specific tokenizer file that supports Marathi (contains 'ळ')
# It is inside the .nemo archive with hash d8761317...
tar -xf models/indicconformer_stt_mr_hybrid_ctc_rnnt_large/indicconformer_stt_mr_hybrid_rnnt_large.nemo \
  -C tokenizers d8761317c86f47acb14f125a77ad359a_tokenizer.model

# 4. Rename it to what the scripts expect
mv tokenizers/d8761317c86f47acb14f125a77ad359a_tokenizer.model tokenizers/marathi_tokenizer.model
```

### C. Verify Restoration
Run the robust smoke test. It should pass immediately if the restore was successful.
```bash
bash scripts/robust_smoke_test.sh
```

---

## 7) Konkani Setup Instructions (Next Steps) 🚀
**Once the Marathi smoke test is verified (Section 5/6), proceed with Konkani:**

### A. Download Konkani Assets
```bash
mkdir -p models tokenizers
wget https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large/resolve/main/indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo -O models/konkani_model.nemo
wget https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large/resolve/main/tokenizer_spe_bpe_v1024/tokenizer.model -O tokenizers/konkani_tokenizer.model
```

### B. Validate Tokenizer
```bash
python - <<'PY'
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