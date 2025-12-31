#!/usr/bin/env python3
"""Run a small smoke inference on a manifest and verify outputs contain Devanagari characters"""
import argparse
import json
import os
from pathlib import Path
import time

import nemo
import nemo.collections.asr as nemo_asr
import torch
from jiwer import wer

DEVANAGARI_RANGE = (0x0900, 0x097F)


def has_devanagari(s: str) -> bool:
    return any(DEVANAGARI_RANGE[0] <= ord(c) <= DEVANAGARI_RANGE[1] for c in s)


def load_manifest(manifest_path: Path):
    entries = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def transcribe(model, audio):
    # Try language_id='kok' first (AI4Bharat models), fallback to no language_id on failure
    for kwargs in ({'language_id': 'kok'}, {}):
        try:
            out = model.transcribe([audio], batch_size=1, **kwargs)
            if isinstance(out, list) and out:
                if isinstance(out[0], list):
                    return out[0][0]
                return out[0]
            return str(out)
        except Exception as e:
            last_exc = e
            continue
    return f"<ERROR: {last_exc}>"


def run(args):
    model_path = args.model_path
    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    # Try a tolerant restore similar to the smoke script: try ASRModel.restore_from(strict=False),
    # otherwise perform a partial restore that loads matching-shape params only.
    from nemo.collections.asr.models import ASRModel as _ASRModel
    def partial_restore_from_nemo(nemo_path):
        import tarfile, tempfile, torch, yaml
        print(f"🔧 Attempting partial restore from: {nemo_path}")
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(nemo_path, 'r') as tar:
                members = {m.name: m for m in tar.getmembers()}
                if 'model_config.yaml' not in members or 'model_weights.ckpt' not in members:
                    raise RuntimeError('model_config.yaml or model_weights.ckpt missing in .nemo')
                tar.extract('model_config.yaml', path=td)
                tar.extract('model_weights.ckpt', path=td)

            config_path = os.path.join(td, 'model_config.yaml')
            ckpt_path = os.path.join(td, 'model_weights.ckpt')
            with open(config_path, 'r', encoding='utf-8') as f:
                conf = yaml.safe_load(f)

            try:
                model_instance = _ASRModel.from_config_dict(conf, trainer=None)
            except Exception:
                model_instance = nemo_asr.models.ASRModel.from_config_dict(conf, trainer=None)

            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            model_sd = model_instance.state_dict()
            filtered = {}
            matched = skipped = 0
            for k, v in state.items():
                if k in model_sd and list(v.shape) == list(model_sd[k].shape):
                    filtered[k] = v
                    matched += 1
                else:
                    skipped += 1
            print(f"🔁 Matched {matched} params, skipped {skipped} params")
            model_instance.load_state_dict(filtered, strict=False)
            return model_instance

    try:
        model = _ASRModel.restore_from(model_path, strict=False)
    except Exception as e:
        print('Restore_from failed with:', e)
        print('Falling back to partial state dict restore...')
        model = partial_restore_from_nemo(model_path)

    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    entries = load_manifest(manifest)
    results = {"summary": {"total": len(entries), "devanagari_ok": 0}, "samples": []}

    start = time.time()
    for i, e in enumerate(entries):
        audio = e["audio_filepath"]
        ref = e.get("text", "")
        pred = transcribe(model, audio)
        sample_has_deva = has_devanagari(pred)
        if sample_has_deva:
            results["summary"]["devanagari_ok"] += 1
        results["samples"].append({"index": i, "audio": audio, "ref": ref, "pred": pred, "deva_ok": sample_has_deva})
    duration = time.time() - start

    # overall WER (if jiwer available)
    hyps = [s["pred"] for s in results["samples"]]
    refs = [s["ref"] for s in results["samples"]]
    try:
        overall_wer = wer(refs, hyps)
    except Exception:
        overall_wer = None

    results["summary"].update({"time_s": duration, "overall_wer": overall_wer})

    out_file = out_dir / "smoke_eval_devanagari.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_file}")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .nemo model file")
    parser.add_argument("--manifest", required=True, help="Manifest to run")
    parser.add_argument("--output_dir", default="results/AI4Bharat_amchi_konkani")
    args = parser.parse_args()
    run(args)
