#!/usr/bin/env python3
"""
Run inference on validation manifest using the last checkpoint and save per-sample WER JSON similar to smoke test.
"""
import os
import json
import argparse

import torch
import nemo
import nemo.collections.asr as nemo_asr

OUT_DIR = os.path.join('results', 'AI4Bharat_amchi_konkani')
OUT_FILE = os.path.join(OUT_DIR, 'finetune_eval_devanagari.json')
CKPT_DIR = os.path.join('results', 'checkpoints')


def find_last_ckpt(ckpt_dir):
    files = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not files:
        raise FileNotFoundError('No checkpoints found')
    # prefer last.ckpt if present
    for name in ['last.ckpt', 'last-v1.ckpt']:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            return p
    # else return the newest by mtime
    return max(files, key=os.path.getmtime)


def load_model(ckpt_path):
    import yaml
    import torch
    print('Loading base .nemo model and applying checkpoint weights:', ckpt_path)
    # Load base .nemo model specified in config file so tokenizer and artifacts are present
    cfg = yaml.safe_load(open('configs/konkani_finetune.yaml'))
    nemo_model_path = cfg.get('model', {}).get('nemo_model')
    if not nemo_model_path or not os.path.exists(nemo_model_path):
        raise FileNotFoundError('Base .nemo model not found at ' + str(nemo_model_path))
    base = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(nemo_model_path, strict=False)

    # Load checkpoint state dict safely and apply matching weights
    # Robust checkpoint application: prefer class-level load_from_checkpoint, then try prefix stripping, else fallback to filtered matching
    try:
        ModelClass = base.__class__
        print('Attempting to load checkpoint via ModelClass.load_from_checkpoint(...)')
        loaded = ModelClass.load_from_checkpoint(ckpt_path, map_location='cpu')
        print('Loaded model via load_from_checkpoint successfully')
        return loaded
    except Exception as e_load:
        print(f'load_from_checkpoint failed: {e_load}; falling back to state_dict mapping')

    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        # Try an explicit trustful load (weights_only=False) if the checkpoint contains pickled objects
        try:
            print('Initial safe load failed; retrying with weights_only=False (trust required)')
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            raise RuntimeError(f'Failed to load checkpoint (both safe and trustful loads failed): {e} | {e2}')
    state = ckpt.get('state_dict', ckpt)
    base_sd = base.state_dict()

    # Try stripping prefixes like 'model.' and 'module.'
    def strip_prefixes(sd):
        new = {}
        for k, v in sd.items():
            kk = k
            for p in ('model.', 'module.'):
                if kk.startswith(p):
                    kk = kk[len(p):]
            new[kk] = v
        return new

    stripped = strip_prefixes(state)
    matched = 0
    for k, v in stripped.items():
        if k in base_sd and list(v.shape) == list(base_sd[k].shape):
            matched += 1
    total = len(base_sd)
    if matched >= max(1, int(0.6 * total)):
        print(f'Applying prefix-stripped state_dict: matched {matched}/{total} params; loading with strict=False')
        base.load_state_dict(stripped, strict=False)
        return base

    # Fallback: exact-match filtered mapping
    filtered = {k: v for k, v in state.items() if k in base_sd and list(v.shape) == list(base_sd[k].shape)}
    missing = set(base_sd.keys()) - set(filtered.keys())
    print(f'Applying filtered mapping: matched {len(filtered)} params, {len(missing)} missing; loading with strict=False')
    base.load_state_dict(filtered, strict=False)
    return base


def transcribe_and_save(model, manifest_path, out_file):
    import soundfile as sf
    from nemo.collections.asr.metrics.wer import word_error_rate as wer_fn
    from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(manifest_path)

    # Use CTC decoder for quick inference
    model.cur_decoder = 'ctc'

    # Create dataset and run inference per sample
    tokenizer_dir = model.cfg.get('tokenizer', {}).get('dir', 'models/tokenizer')
    dataset = AudioToBPEDataset(manifest_filepath=manifest_path, tokenizer=None, sample_rate=16000, int_values=False, max_duration=100, min_duration=0.0, trim=False, use_start_end_token=False, return_language_id=False)

    results = []
    total_wer = 0.0
    count = 0
    for ex in dataset:
        audio = ex.get('audio_filepath') or ex.get('audio')
        ref = ex.get('text','')
        try:
            hyp = model.transcribe([audio], batch_size=1, logprobs=False, language_id=0)[0]
        except Exception as e:
            hyp = ''
        try:
            e_wer = wer_fn(ref, hyp)
        except Exception:
            e_wer = 1.0
        results.append({'audio': audio, 'reference': ref, 'transcription': hyp, 'wer': e_wer})
        total_wer += e_wer
        count += 1
    avg = total_wer / count if count>0 else None
    out = {'average_wer': avg, 'count': count, 'entries': results}
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote', out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--manifest', type=str, default='data/dev/manifest.jsonl')
    args = parser.parse_args()

    ckpt = args.ckpt or find_last_ckpt(CKPT_DIR)
    model = load_model(ckpt)
    transcribe_and_save(model, args.manifest, OUT_FILE)
