Models README
=============

Purpose
-------
This folder stores downloaded model artifacts (.nemo) used by experiments. Keep a stable, versioned copy of important models here and DO NOT delete them casually — recovery may be time-consuming.

Naming & structure
------------------
- Keep models in subdirectories under `models/` to avoid accidental overwrite. Examples:
  - `models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo`
  - `models/konkani_model.nemo` (preferred short name for local testing)
- Use descriptive filenames that include source and version when possible (HF repo id, tag).

Backup policy
-------------
- Always copy any `.nemo` you plan to modify or replace to `models/backup/` first.
- Recommended backup command (run from repo root):

```bash
mkdir -p models/backup
cp models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo models/backup/indicconformer_mr_$(date +%Y%m%d_%H%M%S).nemo
```

- Keep at least one backup copy of the last known-working `.nemo` file off-machine (S3, GCS, or other artifact storage) if you can.

Re-downloading models
---------------------
- You need a Hugging Face token to download private models: `huggingface-cli login` or `export HF_TOKEN="<your token>"`.
- Use `scripts/download_model.py` (repo helper) or `huggingface_hub.hf_hub_download` to fetch artifacts into `models/<model_name>/`.

Example:
```bash
python3 scripts/download_model.py --repo_id ai4bharat/indicconformer_mr --out_dir models/indicconformer_mr
# or programmatically
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='ai4bharat/indicconformer_mr', filename='indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo', local_dir='models/indicconformer_mr')"
```

Where to find models & config hooks
-----------------------------------
- Training and inference configs point to model files via `config.model.nemo_model` in YAML configs (for example `configs/smoke_1sample_ctc.yaml`).
- `setup_env.sh` supports `AUTO_DOWNLOAD_MODEL=1` to populate `models/` automatically during environment setup.

Notes
-----
- Do NOT commit `.nemo` files into git (they are large). Keep only pointers and helper scripts in the repo.
- Keep `models/README.md` up-to-date with any changes to naming conventions or backup policies.
