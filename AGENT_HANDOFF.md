# AGENT HANDOFF — Resume instructions for next agent

Date: 2026-01-02

---

## 1) Short summary (one line)
We validated the offline-edit + strict=False transfer recipe on the Marathi `.nemo` and it succeeded. Decision: **Switch to Konkani**. Task: Run the 1‑Epoch CTC Smoke Test using the Konkani model (`models/konkani_model.nemo`) using the exact same steps (Offline Config Edit + strict=False + Real Decoder). Use FAST_FAIL=1 for a quick fail-fast run. Expect to see `FINAL: ctc_decoder type: ConvASRDecoder num_classes_with_blank: 257 -> Decoder: (256 classes)` and `Trainer.fit` reach `max_epochs=1` on GPU; collect logs and artifacts if successful.

Handoff for the next agent (explicit):
- Run the 1‑epoch smoke on **Konkani** using the exact commands under section **5) Exact commands** adjusted to point at `models/konkani_model.nemo`.
- Use `scripts/ensure_model_present.sh --model konkani --yes` to fetch the model if it's missing and then run:
  ```bash
  export APPLY_CONV_PATCH=1 FAST_FAIL=1
  unset USE_CTC_STUB
  python3 scripts/fine_tune.py --config configs/smoke_1sample_ctc.yaml 2>&1 | tee /tmp/smoke_konkani_1epoch.log
  ```
- On success, mark the checklist item and proceed to a 20‑epoch micro-overfit.
- If you hit errors, attach `/tmp/smoke_konkani_1epoch.log` to an issue; include the `FINAL:` line (if present) and any `KeyError` backtraces.

## 2) Current state & context
- We attempted to restore and micro-overfit a patched Marathi `.nemo` but hit config/API incompatibilities (fields like `multisoftmax`, `language_keys`, missing `tokenizer.dir`, etc.).
- Tokenizer artifacts were extracted from the Marathi `.nemo` and token counts validated, but the restore path remained brittle.
- The short-term plan approved by the reviewer is to switch to the official Konkani model from HF which has matching tokenizer & model and avoids needing to pin NeMo.
- Important repo scripts: `scripts/run_micro_overfit.py`, `scripts/fine_tune.py`, `scripts/run_smoke_and_check_deva.py`.

## 3) Objective for you (next agent)
1. Download the Konkani .nemo and its matching tokenizer
2. Update `configs/konkani_finetune.yaml` to reference the new model & tokenizer
3. Ensure the runtime uses the AI4Bharat NeMo fork (do NOT install upstream `nemo_toolkit==1.23.0`) — run env setup if needed
4. Run a short smoke micro-overfit (3 epochs) and validate acceptance criteria
5. If the smoke passes, run the full 20‑epoch overfit and collect artifacts

## 4) Files & locations you will use
- Repo root: `/workspace/amchi_asr`
- Models: `models/` (create `models/konkani_model.nemo`)
- Tokenizers: `tokenizers/` (create `tokenizers/konkani_tokenizer.model`)
- Config to edit: `configs/konkani_finetune.yaml` (backup before editing)
- Micro-overfit runner: `scripts/run_micro_overfit.py`
- Training code: `scripts/fine_tune.py`
- Tiny manifests (local test sets): `tiny_train.jsonl`, `tiny_val.jsonl`, `tiny_test.jsonl`
- Backups of earlier working .nemo (if needed): `models/backups/` (we have a few dated copies)

## 5) Exact commands (copy/paste)
### A. Download assets
```bash
mkdir -p models tokenizers
wget https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large/resolve/main/indicconformer_stt_kok_hybrid_ctc_rnnt_large.nemo -O models/konkani_model.nemo
wget https://huggingface.co/ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large/resolve/main/tokenizer_spe_bpe_v1024/tokenizer.model -O tokenizers/konkani_tokenizer.model
```

### B. Environment setup (use AI4Bharat fork)
Preferred: run the project's env provision script which will pull the AI4Bharat NeMo fork and compatible deps.
```bash
# from repo root
bash setup_env.sh
# or
bash scripts/ensure_env.sh
```
Note: do NOT install `nemo_toolkit==1.23.0` directly — that was the reviewer warning. The fork provides fork-specific features (e.g., `multisoftmax`).

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