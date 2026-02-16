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
            # If a string language_id like 'kok' is provided, try to map it to an integer index
            if 'language_id' in kwargs and isinstance(kwargs['language_id'], str):
                lid = kwargs['language_id']
                mapped = None
                try:
                    # Preferred: model.joint.language_keys
                    if hasattr(model, 'joint') and hasattr(model.joint, 'language_keys'):
                        mapped = list(model.joint.language_keys).index(lid)
                except Exception:
                    mapped = None
                if mapped is None:
                    try:
                        # fallback: look for language_keys in model.config
                        if hasattr(model, 'cfg') and 'language_keys' in getattr(model.cfg, 'joint', {}):
                            mapped = list(model.cfg.joint.language_keys).index(lid)
                    except Exception:
                        mapped = None
                if mapped is not None:
                    print(f"DEBUG: Mapped language_id '{lid}' -> {mapped}")
                    kwargs['language_id'] = mapped
                else:
                    print(f"DEBUG: Could not map language_id '{lid}' to integer index; passing as-is")

            out = model.transcribe([audio], batch_size=1, **kwargs)
            if isinstance(out, list) and out:
                if isinstance(out[0], list):
                    return out[0][0]
                return out[0]
            return str(out)
        except Exception as e:
            # Print full traceback for debugging
            import traceback, os
            print(f"DEBUG: transcribe with kwargs={kwargs} failed with: {e}")
            traceback.print_exc()
            # If requested via env var, re-raise to get the full stack and abort
            if os.environ.get('RAISE_ON_ERROR') == '1':
                raise
            last_exc = e
            continue
    return f"<ERROR: {last_exc}>"


def run(args):
    model_path = args.model_path
    manifest = Path(args.manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    # Try restoring with specific model class first
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from nemo.collections.asr.models import ASRModel as _ASRModel
    def partial_restore_from_nemo(nemo_path):
        import tarfile, tempfile, torch, yaml
        print(f"🔧 Attempting partial restore from: {nemo_path}")
        
        def _find_member(tar, name):
            """Find member with or without ./ prefix"""
            if name in {m.name for m in tar.getmembers()}:
                return name
            alt_name = f"./{name}"
            if alt_name in {m.name for m in tar.getmembers()}:
                return alt_name
            return None
        
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(nemo_path, 'r') as tar:
                config_member = _find_member(tar, 'model_config.yaml')
                weights_member = _find_member(tar, 'model_weights.ckpt')
                tokenizer_member = _find_member(tar, 'tokenizer.model')
                
                if not config_member or not weights_member:
                    raise RuntimeError('model_config.yaml or model_weights.ckpt missing in .nemo')
                
                tar.extract(config_member, path=td)
                tar.extract(weights_member, path=td)
                if tokenizer_member:
                    tar.extract(tokenizer_member, path=td)
                
                # Handle ./ prefix in extracted paths
                config_path = os.path.join(td, config_member.lstrip('./'))
                ckpt_path = os.path.join(td, weights_member.lstrip('./'))

            with open(config_path, 'r', encoding='utf-8') as f:
                conf = yaml.safe_load(f)
            
            # Convert multilingual tokenizer to BPE format while preserving native vocab size
            # NeMo's standard install doesn't support type='multilingual', but the model
            # weights expect 5632 classes. Solution: convert tokenizer to BPE but keep
            # the full 5632-class decoder vocabularies intact.
            if 'tokenizer' in conf:
                tok_cfg = conf['tokenizer']
                if tok_cfg.get('type') == 'multilingual':
                    langs = tok_cfg.get('langs', {})
                    if isinstance(langs, dict) and 'kok' in langs:
                        print("DEBUG: Converting multilingual tokenizer to BPE (keeping 5632 vocab)")
                        # Extract kok tokenizer config and convert to monolingual BPE
                        kok_cfg = langs['kok']
                        conf['tokenizer'] = {
                            'type': 'bpe',
                            'dir': 'tokenizers',
                            'model_path': 'tokenizers/konkani_tokenizer.model'
                        }
                        # CRITICAL: Prevent the tokenizer from creating a 256-length vocabulary
                        # and injecting it into decoder config before we can change_vocabulary.
                        if 'decoder' in conf:
                            conf['decoder']['vocabulary'] = None
                        if 'aux_ctc' in conf and 'decoder' in conf['aux_ctc']:
                            conf['aux_ctc']['decoder']['vocabulary'] = None
                        print(f"DEBUG: Tokenizer converted; decoder vocab references cleared (weights remain 5632)")
                    else:
                        print(f"DEBUG: Multilingual tokenizer langs: {list(langs.keys()) if isinstance(langs, dict) else langs}")
                        # Fallback: still convert to BPE
                        conf['tokenizer'] = {
                            'type': 'bpe',
                            'dir': 'tokenizers'
                        }
            
            # Remove unsupported config keys that cause instantiation errors
            if 'decoder' in conf:
                conf['decoder'].pop('multisoftmax', None)
            if 'joint' in conf:
                conf['joint'].pop('language_keys', None)
                conf['joint'].pop('multilingual', None)
            
            try:
                model_instance = EncDecHybridRNNTCTCBPEModel.from_config_dict(conf, trainer=None)
            except Exception as e:
                print(f"EncDecHybridRNNTCTCBPEModel.from_config_dict failed: {e}")
                model_instance = _ASRModel.from_config_dict(conf, trainer=None)

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
        model = EncDecHybridRNNTCTCBPEModel.restore_from(model_path, strict=False)
    except Exception as e:
        print('Restore_from failed with:', e)
        print('Falling back to partial state dict restore...')
        model = partial_restore_from_nemo(model_path)

    model.eval()
    
    # Disable CUDA graphs to avoid CUDA failure with mismatched vocab sizes
    if hasattr(model, 'decoding') and hasattr(model.decoding, 'decoding_computer'):
        if hasattr(model.decoding.decoding_computer, 'use_cuda_graphs'):
            print("DEBUG: Disabling CUDA graphs for inference compatibility")
            model.decoding.decoding_computer.use_cuda_graphs = False
    
    if torch.cuda.is_available():
        model = model.cuda()

    entries = load_manifest(manifest)
    # Add model metadata
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception:
        param_count = None
    model_identity = os.path.basename(model_path)
    model_class = model.__class__.__name__ if hasattr(model, '__class__') else None

    results = {
        "summary": {
            "total": len(entries),
            "devanagari_ok": 0,
            "model_identity": model_identity,
            "model_class": model_class,
            "param_count": param_count,
            "language": "kok"
        },
        "samples": []
    }

    start = time.time()
    per_sample_latencies = []
    for i, e in enumerate(entries):
        audio = e["audio_filepath"]
        ref = e.get("text", "")
        t0 = time.time()
        pred = transcribe(model, audio)
        latency = time.time() - t0
        per_sample_latencies.append(latency)

        sample_has_deva = has_devanagari(pred)
        if sample_has_deva:
            results["summary"]["devanagari_ok"] += 1

        # per-sample WER (if prediction is valid)
        pswer = None
        if not (isinstance(pred, str) and pred.startswith("<ERROR")):
            try:
                pswer = wer([ref], [pred])
            except Exception:
                pswer = None

        # Normalize prediction to a single readable string
        def _normalize_pred(p):
            import ast
            # If it's a string representation of a Python object, try to parse
            if isinstance(p, str):
                try:
                    parsed = ast.literal_eval(p)
                    return _normalize_pred(parsed)
                except Exception:
                    # not a Python literal; return as-is
                    return p
            # If it's a list/tuple, drill down to first string
            if isinstance(p, (list, tuple)) and len(p) > 0:
                first = p[0]
                if isinstance(first, (list, tuple)) and len(first) > 0:
                    return _normalize_pred(first[0])
                return _normalize_pred(first)
            # Otherwise cast to str
            return str(p)

        pred_norm = _normalize_pred(pred)

        results["samples"].append({
            "index": i,
            "audio": audio,
            "ref": ref,
            "pred": pred_norm,
            "deva_ok": sample_has_deva,
            "pred_latency_s": latency,
            "wer": pswer
        })
    duration = time.time() - start

    # overall WER (if jiwer available)
    hyps = [s["pred"] for s in results["samples"] if not (isinstance(s.get("pred"), str) and s.get("pred").startswith("<ERROR"))]
    refs = [s["ref"] for s in results["samples"] if not (isinstance(s.get("pred"), str) and s.get("pred").startswith("<ERROR"))]
    try:
        overall_wer = wer(refs, hyps) if len(hyps) > 0 else None
    except Exception:
        overall_wer = None

    results["summary"].update({
        "time_s": duration,
        "avg_latency_s": sum(per_sample_latencies) / len(per_sample_latencies) if per_sample_latencies else None,
        "overall_wer": overall_wer
    })

    out_file = out_dir / "smoke_eval_devanagari.json"
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"Saved results to {out_file}")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to .nemo model file")
    parser.add_argument("--manifest", required=False, default="data/test/manifest.jsonl", help="Manifest to run (default: data/test/manifest.jsonl)")
    parser.add_argument("--output_dir", default="results/AI4Bharat_amchi_konkani")
    args = parser.parse_args()
    run(args)
