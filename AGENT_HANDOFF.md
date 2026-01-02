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


### C. Validate tokenizer
```bash
python - <<'PY'
import sentencepiece as spm
s=spm.SentencePieceProcessor()
s.load('tokenizers/konkani_tokenizer.model')
print('tokenizer size', s.get_piece_size())
PY
```
Expect a reasonable token size (e.g., 256 or similar) and no load errors.

### D. Backup & edit config
```bash
cp configs/konkani_finetune.yaml configs/konkani_finetune.yaml.bak
# Edit and set model & tokenizer paths (examples below)
# model_path: models/konkani_model.nemo
# tokenizer_model: tokenizers/konkani_tokenizer.model
```
If the repo uses a different keypath (search for `konkani_finetune`), update whichever config the training uses. Keep the tiny test manifests (tiny_train/val/test) untouched for micro-overfit.

### E. Quick restore check
```bash
python - <<'PY'
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
p='models/konkani_model.nemo'
print('restoring...')
try:
    m=EncDecHybridRNNTCTCBPEModel.restore_from(p, strict=False)
    print('restore OK; decoder:', type(m.decoder), 'has vocab?', hasattr(m.decoder, 'vocabulary'))
except Exception as e:
    import traceback; traceback.print_exc()
    print('restore failed')
PY
```
If restore succeeds, great — move to running the smoke micro-overfit.

### F. Run smoke micro-overfit (3 epochs)
From repo root:
```bash
# Example env flags
export MAX_MICRO_EPOCHS=3
export SKIP_MICRO_PREFLIGHT=1   # optional to speed up
export MICRO_SKIP_MODEL_RESTORE=0 # set to 1 only if you need to instantiate from local config rather than .nemo
python scripts/run_micro_overfit.py
```
Check logs under `results/experiments/` for the run. If the script supports passing config names or model paths, use them as needed.

### G. Full 20-epoch test (when smoke passes)
```bash
export MAX_MICRO_EPOCHS=20
python scripts/run_micro_overfit.py
```

## 6) Acceptance criteria (how to mark PASS)
- PASS if **train-loss** reduces by >= 50% over the run OR normalized **char-distance** ≤ 0.2
- The micro-overfit script currently enforces this and will exit non-zero on failure; see its logs and produced `samples_epoch_*.json` for predicted strings and distances.

## 7) Troubleshooting notes (common issues & fixes)
- If restore fails with errors about missing `multisoftmax`, `language_keys`, or unexpected kwargs:
  - Ensure the environment used the **AI4Bharat NeMo fork** (use `setup_env.sh` or `pip install -e /workspace/NeMo_ai4bharat` in a venv).
  - If the fork is installed and errors persist, try `EncDecHybridRNNTCTCBPEModel.restore_from(..., strict=False)` and inspect the stack trace to identify unsupported fields.
- If tokenizer errors show `KeyError: 'dir'` or cannot find tokenizer files:
  - Confirm `configs/konkani_finetune.yaml` points to `tokenizers/konkani_tokenizer.model` and that the file exists.
  - Alternatively set `MICRO_SKIP_MODEL_RESTORE=1` to instantiate model from a sanitized local `model_config.yaml` (not necessary for Konkani, but helpful when debugging other .nemo files).
- If the micro-overfit script fails early with missing manifest paths, edit the referenced `model_config.yaml` or the config to point to `tiny_train.jsonl` / `tiny_val.jsonl` local test manifests.

## 8) Logging & artifacts to collect
- `results/experiments/<timestamp>/` folder: copy `samples_epoch_*.json`, metrics, `hparams.yaml` and `lightning_logs` as artifacts.
- Save final model checkpoints (if any) and the run config used to `results/konkani_runs/` with a short metadata file (`who`, `what`, `why`, `nemo_sha256`) for reproducibility.

## 9) If all else fails — fallback options
- If the Konkani .nemo fails to restore (unexpected), you can:
  1. Try `MICRO_SKIP_MODEL_RESTORE=1` and instantiate from a sanitized `model_config.yaml` (patch local tokenizer paths and remove unsupported fields).
  2. As a last resort, recreate a tiny synthetic training pass (we already have a synthetic micro-overfit smoke) to validate infra.

## 10) Useful references in the repo
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