#!/usr/bin/env python3
"""
Smoke test for AI4Bharat IndicConformer Marathi model.

This script is intended to be run inside WSL/Linux (or any Unix-like env).
It loads the pretrained AI4Bharat Marathi model, runs inference on the
`data/manifests/dev_small.json` manifest, and prints transcriptions and WER.

Purpose: verify model download, loading, and inference locally before doing
any cloud GPU fine-tuning.
"""

import os
import sys
import json
import traceback

try:
    # Add uname shim for environments where os.uname is missing
    if not hasattr(os, "uname"):
        def _uname():
            import collections
            uname_result = collections.namedtuple('uname_result', ['sysname', 'nodename', 'release', 'version', 'machine'])
            return uname_result(sysname='Linux', nodename='localhost', release='0', version='0', machine='x86_64')
        os.uname = _uname

    import nemo.collections.asr as nemo_asr
    from jiwer import wer
except Exception:
    print("Failed to import prerequisites. Ensure you're running inside WSL/Linux with the correct Python env and that `nemo-toolkit`, `jiwer` are installed.")
    traceback.print_exc()
    sys.exit(1)


def load_manifest(path):
    examples = []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
                examples.append(j)
            except Exception:
                # not JSON -> try TSV/other
                continue
    return examples


def main():
    manifest = os.path.join(os.path.dirname(__file__), '..', 'data', 'manifests', 'dev_small.json')
    manifest = os.path.normpath(manifest)

    if not os.path.exists(manifest):
        print(f"Manifest not found at {manifest}")
        sys.exit(1)

    print("Loading AI4Bharat Marathi IndicConformer model (this may download files)...")
    # HF token should be in env var HF_TOKEN
    hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HF_TOKEN')
    try:
        model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_mr_hybrid_rnnt_large")
    except Exception as e:
        print("Failed to load model:", e)
        traceback.print_exc()
        sys.exit(1)

    model.freeze()

    examples = load_manifest(manifest)
    if not examples:
        print("No examples found in manifest.")
        sys.exit(1)

    total_wer = 0.0
    count = 0

    print(f"Running inference on {len(examples)} examples...")
    for ex in examples:
        audio = ex.get('audio_filepath') or ex.get('audio')
        ref = ex.get('text', '')
        if not audio:
            continue
        # If the path is relative, make it relative to repo root
        if not os.path.isabs(audio):
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            audio = os.path.normpath(os.path.join(repo_root, 'data', 'audio', os.path.basename(audio)))

        if not os.path.exists(audio):
            print(f"Audio file missing: {audio}")
            continue

        try:
            # Use CTC decoder for quick inference
            model.cur_decoder = 'ctc'
            transcription = model.transcribe([audio], batch_size=1, logprobs=False, language_id='mr')[0]
        except Exception as e:
            print(f"Transcription failed for {audio}: {e}")
            traceback.print_exc()
            continue

        print("\n---")
        print(f"Audio: {audio}")
        print(f"Reference: {ref}")
        print(f"Transcription: {transcription}")

        if ref:
            try:
                e_wer = wer(ref, transcription)
                print(f"WER: {e_wer:.2%}")
                total_wer += e_wer
                count += 1
            except Exception:
                print("Failed to compute WER")

    if count:
        print(f"\nAverage WER on manifest: {total_wer/count:.2%}")


if __name__ == '__main__':
    main()
