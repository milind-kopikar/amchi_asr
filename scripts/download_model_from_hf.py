#!/usr/bin/env python3
"""Download a .nemo model and extract tokenizer files from Hugging Face Hub.

Usage:
  python3 scripts/download_model_from_hf.py --repo ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large

It will download the snapshot, find the .nemo file, copy it into models/<repo_name>/ and extract tokenizer files into models/tokenizer/.
"""
import argparse
import os
from pathlib import Path
from huggingface_hub import snapshot_download
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', required=True, help='Hugging Face repo id, e.g. ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large')
    p.add_argument('--outdir', default='models', help='Base models directory')
    args = p.parse_args()

    repo = args.repo
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f'Downloading repo snapshot: {repo} ...')
    local = snapshot_download(repo_id=repo)
    print('Snapshot downloaded to', local)

    nemo_files = list(Path(local).rglob('*.nemo'))
    if not nemo_files:
        print('No .nemo files found in snapshot; listing files:')
        for p in Path(local).rglob('*'):
            if p.is_file():
                print('  ', p)
        raise SystemExit(1)

    nemo = nemo_files[0]
    repo_name = repo.split('/')[-1]
    dest_dir = outdir / repo_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_nemo = dest_dir / nemo.name
    if dest_nemo.exists():
        print('Model already present at', dest_nemo)
    else:
        print('Copying', nemo, '->', dest_nemo)
        dest_nemo.write_bytes(nemo.read_bytes())

    # Try to extract tokenizer from .nemo (it's a tar archive)
    try:
        with tarfile.open(dest_nemo, 'r') as tf:
            members = [m for m in tf.getmembers()]
            tokenizer_members = [m for m in members if 'tokenizer.model' in m.name or 'tokenizer.vocab' in m.name]
            if tokenizer_members:
                token_dir = outdir / 'tokenizer'
                token_dir.mkdir(parents=True, exist_ok=True)
                print('Extracting tokenizer files to', token_dir)
                for m in tokenizer_members:
                    tf.extract(m, path=token_dir)
                print('Tokenizer extraction complete')
            else:
                print('No tokenizer files found inside .nemo')
    except Exception as e:
        print('Failed to extract tokenizer from .nemo:', e)

    print('Done. Remember to update your config.model.nemo path to point to', dest_nemo)


if __name__ == '__main__':
    main()
