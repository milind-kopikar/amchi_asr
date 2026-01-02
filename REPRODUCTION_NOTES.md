REPRODUCTION NOTES
==================

Purpose
-------
This document records the exact recipe used to successfully run a 1-epoch CTC-only smoke fine-tune on the IndicConformer Marathi model and the engineering decisions made to avoid validator and instantiation errors when switching to a small (256-class) CTC decoder.

1) Architecture Strategy 🔧
- Use decoder_type="ctc" (CTC-only) for transfer/fine-tune. This avoids RNNT-specific joint/decoder validation and GPU JIT complexity during smoke tests.

2) Loading Hack (Offline Config Edit + strict=False) 🧠
- Before instantiation, edit the extracted model config (`model_config.yaml`) or the in-memory config dictionary so that the auxiliary CTC decoder is defined as a small decoder compatible with the local tokenizer.
  - Specifically: set `aux_ctc.decoder.num_classes = 256` and remove any `vocabulary` entries under `aux_ctc.decoder` (clear `vocabulary` to avoid mismatch checks).
- Rationale: The upstream multilingual .nemo model decoder had `num_classes=5632` while the local tokenizer has 256 tokens. Editing the config to the target architecture avoids the `len(vocabulary) != num_classes` validation and lets us instantiate the smaller decoder directly.
- After instantiation, load pretrained weights from the .nemo checkpoint using `strict=False` (or selectively match shapes) so encoder weights are restored while mismatched decoder weights are skipped.

3) Loss Configuration Safety ⚖️
- Do NOT change the top-level `loss.loss_name` to `'ctc'` in the main model config. The RNNT loss resolver expects RNNT losses for the RNNT model and will raise `ValueError` if you set a top-level RNNT `loss_name` to `'ctc'`.
- Instead, place the CTC loss under the auxiliary block: `aux_ctc.loss.loss_name = 'ctc'`. This ensures RNNT remains consistent while exposing an auxiliary CTC loss for CTC-only training.

4) change_vocabulary API and Why It's Bypassed 🔀
- `change_vocabulary` requires the original (large) model to instantiate successfully. Because the multilingual decoder triggers early validation errors with a mismatched vocabulary, change_vocabulary often cannot be called.
- Bypassing strategy: perform the offline config edit (step 2) to instantiate the target architecture (small CTC decoder) and then load encoder weights with `strict=False`. This accomplishes the goals of `change_vocabulary` (a smaller decoder compatible with local tokenizer) without needing to first instantiate the large multilingual model.

5) Practical checklist for reproducing the smoke test ✅
- Environment: follow `SETUP_ENV.md` / `AI4BHARAT_SETUP_GUIDE.md` to ensure dependencies (NeMo fork, conv_asr patch optional) are present.
- Tokenizer: ensure local tokenizer directory `tokenizers/` contains `konkani_tokenizer.model` (or the correct tokenizer for the language); this example uses a 256-token SentencePiece model.
- Config: use `configs/smoke_1sample_ctc.yaml` (or equivalent) which sets data and trainer for a 1-sample, 1-epoch run.
- Execution steps (example):
  1. Export `APPLY_CONV_PATCH=1` (if using the runtime conv_asr fixes in `patches/conv_asr_fixed.py`).
  2. Ensure `USE_CTC_STUB` is unset/false for verification with a real decoder.
  3. Run: `python3 scripts/fine_tune.py --config configs/smoke_1sample_ctc.yaml`
  4. Confirm logs show:
     - `DEBUG: aux_ctc config: ... 'num_classes': 256 ...`
     - Model instantiation succeeds and `🔁 Matched N params, skipped M params` (indicating strict=False loading)
     - Training runs for 1 epoch (Trainer reaches `max_epochs=1`).

6) Where to look if something fails 🔎
- Key failure modes and their signatures:
  - KeyError: 'dir' → tokenizer config is missing expected keys; the `scripts` handle this by preferring `tokenizers/`.
  - ValueError: `Provided loss_name ctc not in list of available RNNT losses` → do not set top-level `loss.loss_name` to `ctc`.
  - ValueError: `num_classes` vs `len(vocabulary)` mismatch → ensure `aux_ctc.decoder.num_classes` matches tokenizer size or clear `vocabulary`.

7) Notes for future work
- After locking this flow for Marathi, follow the same recipe for the Konkani .nemo file (switch model path) and re-run the smoke test. If the Konkani .nemo has additional multilingual metadata, rely on the same sanitization strategy used here.

File links
----------
- `configs/smoke_1sample_ctc.yaml` — minimal 1-sample 1-epoch CTC test config used for verification.
- `scripts/fine_tune.py` — implements the offline config-edit + strict=False restore pattern.

Canonical .nemo models & exact operations ✅
-----------------------------------------
- Canonical Marathi base (used for verification in this repo):
  - Path: `models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo`
  - NOTE: Keep a backup copy in `models/backup/` before making edits.
- Konkani model (target later):
  - Path: `models/konkani_model.nemo` (or a subfolder `models/konkani/` if you prefer organization)

Exact patch & config recipe (copyable)
--------------------------------------
1) Ensure tokenizer present locally (example path `tokenizers/konkani_tokenizer.model`).
2) Apply runtime conv_asr patch (recommended):
   - `export APPLY_CONV_PATCH=1` will cause `scripts/fine_tune.py` to import `patches/conv_asr_fixed.py` and monkey-patch the installed NeMo `conv_asr` module at runtime.
   - Alternatively, copy `patches/conv_asr_fixed.py` to the installed NeMo location (not recommended unless you know what you are doing).
3) Offline config edit steps (automated in `scripts/fine_tune.py` but shown here as explicit steps):
   - Extract `model_config.yaml` and `model_weights.ckpt` from the `.nemo` (tar file). Example:
     ```bash
     mkdir -p /tmp/nemo_work && tar -xf models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo -C /tmp/nemo_work model_config.yaml model_weights.ckpt
     ```
   - Edit the config (`model_config.yaml`) or in-memory `conf` dict to set the aux CTC decoder and loss:
     ```python
     conf['aux_ctc'] = conf.get('aux_ctc', {})
     conf['aux_ctc'].setdefault('loss', {})['loss_name'] = 'ctc'
     conf['aux_ctc'].setdefault('decoder', {})
     conf['aux_ctc']['decoder'].pop('vocabulary', None)
     conf['aux_ctc']['decoder']['num_classes'] = 256
     ```
   - DO NOT set top-level `conf['loss']['loss_name'] = 'ctc'` (this breaks RNNT loss resolution).
4) Instantiate the edited config, then load weights with `strict=False` to ignore mismatched decoder parameters and load encoder weights:
   ```python
   model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_config_dict(conf, trainer=None)
   ckpt = torch.load(ckpt_path, map_location='cpu')
   state = ckpt.get('state_dict', ckpt)
   # filter state dict to matching shapes
   model.load_state_dict(filtered, strict=False)
   ```
5) Verify the instantiated model reports the CTC decoder size (we added debug prints):
   - Expect: `FINAL: ctc_decoder type: ConvASRDecoder num_classes_with_blank: 257 -> Decoder: (256 classes)`

Why this works
---------------
- Instantiating the model with the smaller CTC decoder avoids early validation checks that fail when the stored decoder has 5632 classes while the local tokenizer only has 256 tokens. Loading with `strict=False` restores encoder weights while ignoring decoder mismatches.

History & Rationale
-------------------
The offline-edit + strict=False loading approach was chosen after repeated failures to instantiate the multilingual model due to tokenizer / vocab mismatches and RNNT-specific loss semantics. This approach is conservative, reproducible, and avoids invasive library patches.

Maintainers: amchi_asr team
Date: 2026-01-02

---

Additional operational details (added 2026-01-02) 🧾

- Python version used: **3.11** (see `SETUP_ENV.md` for the canonical environment instructions used in this repo).

- Model storage and backup policy:
  - Store all downloaded `.nemo` checkpoint files in `models/` (for example `models/indicconformer_mr/...nemo` or `models/konkani_model.nemo`).
  - Keep a backup copy of any critical `.nemo` files (for example, copy to `models/backup/`) before deleting, moving, or re-downloading. Accidental deletion of `.nemo` files cost several hours of rework; we recommend **never** removing models/ files unless intentionally rotating versions.

- Re-downloading `.nemo` model (if missing):
  - Ensure Hugging Face auth: `huggingface-cli login` or export `HF_TOKEN` in environment.
  - Use the repo helper script to download the official AI4Bharat model (example):

    ```bash
    # Example (replace with actual HF path for your model)
    python3 scripts/download_model.py --repo_id ai4bharat/indicconformer_mr --out_dir models/indicconformer_mr
    # Or use huggingface_hub programmatic download
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai4bharat/indicconformer_mr', filename='indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo', local_dir='models/indicconformer_mr')"
    ```

- AUTO_DOWNLOAD_MODEL: If `AUTO_DOWNLOAD_MODEL=1` is set before running `setup_env.sh` the setup will attempt to fetch the `.nemo` into `models/` automatically.

- Tokenizer location and consistency:
  - Keep SentencePiece and vocab files in `tokenizers/` or `models/tokenizer/` with stable filenames (e.g., `tokenizers/konkani_tokenizer.model`).
  - The offline config edit expects the small tokenizer available locally in `tokenizers/` when it patches `conf['tokenizer']` to `{'type':'bpe','dir':'tokenizers','model_path':'tokenizers/konkani_tokenizer.model'}`.

- conv_asr runtime patch and env flags:
  - We include a vendored patch `patches/conv_asr_fixed.py` that fixes mask/language-id handling issues in `ConvASRDecoder`.
  - To apply the runtime patch (recommended during experiments), set: `export APPLY_CONV_PATCH=1` before running training/inference — `scripts/fine_tune.py` will import `patches/conv_asr_fixed` and monkey-patch the installed NeMo conv_asr module.

- Final practical recipe (copyable):

  ```bash
  # ensure hf token present
  huggingface-cli login  # or export HF_TOKEN

  # ensure our patch will be used
  export APPLY_CONV_PATCH=1

  # ensure we're using the local tokenizer
  ls tokenizers/konkani_tokenizer.model

  # run 1-epoch smoke test (strict=False offline aux_ctc edit is done by the script)
  python3 scripts/fine_tune.py --config configs/smoke_1sample_ctc.yaml
  ```

- Logs & artifacts:
  - Experiment outputs and logs are by default under `results/experiments/*` unless overridden by `exp_manager.exp_dir` in the config.
  - Save `/tmp/*.log` copies if you run CI/debug runs that may be ephemeral on cloud instances.

- If you must change the decoder size or tokenizer mapping later, follow the same offline-edit + strict=False pattern and *always* keep a copy of the original `.nemo` and `model_config.yaml` before editing.

---

Please read `SETUP_ENV.md` before attempting to rebootstrap a fresh host — `SETUP_ENV.md` is the canonical host setup doc for this project and contains the environment-specific notes (CUDA, Python version, AUTO_DOWNLOAD_MODEL, etc.).
