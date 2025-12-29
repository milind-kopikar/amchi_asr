#!/usr/bin/env python3
"""Generate a structured smoke-test report for ASR experiments.

Outputs:
 - JSON report with model metadata, overall WER, avg latency, and per-sample records
 - CSV file with one row per sample
 - Markdown table for inclusion in papers or READMEs

Usage example:
 python scripts/generate_smoke_report.py \
   --model_id facebook/mms-1b-all \
   --manifest data_smoke/dev/manifest.jsonl \
   --output_dir results/mms_smoke_report \
   --base_lang kok

Notes:
 - The script uses Hugging Face `pipeline('automatic-speech-recognition')` when possible; if not, it falls back to processor+model inference.
 - Latency is measured per sample (wall-clock) including audio loading time and model forward pass.
 - WER is computed with `jiwer` on a per-sentence basis.
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path

try:
    from transformers import AutoProcessor, AutoModelForCTC, pipeline
    import torch
    from jiwer import wer
except Exception as e:
    print("Missing dependencies. Please install: transformers torch jiwer soundfile huggingface_hub")
    raise

import soundfile as sf


def load_manifest(manifest_path, audio_root=None):
    examples = []
    cwd = os.getcwd()
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            audio = j.get('audio_filepath') or j.get('audio')
            if audio_root and not os.path.isabs(audio):
                # If the manifest stores a path that already includes directories (contains '/'),
                # interpret it relative to the repo root. Otherwise, interpret it relative to the manifest folder.
                if os.path.sep in audio:
                    audio = os.path.normpath(os.path.join(cwd, audio))
                else:
                    audio = os.path.normpath(os.path.join(audio_root, audio))
            examples.append({'audio': audio, 'text': j.get('text', '')})
    return examples


def transcribe_with_pipeline(pipe, audio_path):
    t0 = time.perf_counter()
    res = pipe(audio_path)
    t1 = time.perf_counter()
    # pipeline may return dict or list
    if isinstance(res, list):
        text = res[0]['text'] if 'text' in res[0] else str(res[0])
    else:
        text = res.get('text', '')
    return text, t1 - t0


def transcribe_manual(processor, model, audio_path, device):
    t0 = time.perf_counter()
    audio_arr, sr = sf.read(audio_path)
    inputs = processor(audio_arr, sampling_rate=sr, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)[0]
    text = processor.batch_decode(predicted_ids.unsqueeze(0))[0]
    t1 = time.perf_counter()
    return text, t1 - t0


def make_report(model_id, manifest, output_dir, base_lang='kok', device=None, checkpoint=None):
    os.makedirs(output_dir, exist_ok=True)

    # load processor from the base model id (if provided) and model from checkpoint or model_id
    print('Loading processor and model...')
    processor = None
    model = None

    # Load processor (prefer base model id if available)
    try:
        processor = AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"Warning: failed to load processor from '{model_id}': {e}. Will try loading from checkpoint if available.")
        processor = None

    # Load model weights
    model_load_target = checkpoint if checkpoint else model_id
    try:
        model = AutoModelForCTC.from_pretrained(model_load_target).to(device)
    except Exception as e:
        print(f"Failed to load model from '{model_load_target}': {e}")
        raise

    # If processor missing, try to load from the model path
    if processor is None:
        try:
            processor = AutoProcessor.from_pretrained(model_load_target)
        except Exception as e:
            print(f"Failed to load processor from model path '{model_load_target}': {e}")
            raise

    # try to init pipeline (may fail for some models)
    pipe = None
    try:
        pipe = pipeline('automatic-speech-recognition', model=model, processor=processor, device=0 if device.type=='cuda' else -1)
    except Exception as e:
        print('Pipeline init failed, will use manual inference fallback:', e)
        pipe = None

    # get architecture info (best-effort)
    arch = getattr(model.config, 'model_type', None) or getattr(model.config, 'architectures', [None])[0] or model.__class__.__name__

    rows = []
    per_sample_wers = []
    latencies = []

    for ex in manifest:
        audio = ex['audio']
        ref = ex['text']
        if not os.path.exists(audio):
            print('Warning: audio not found, skipping:', audio)
            continue
        try:
            if pipe:
                pred, latency = transcribe_with_pipeline(pipe, audio)
            else:
                pred, latency = transcribe_manual(processor, model, audio, device)
        except Exception as e:
            print('Transcription failed for', audio, e)
            pred = ''
            latency = None

        try:
            s_wer = wer(ref, pred) if ref else None
        except Exception:
            s_wer = None

        rows.append({
            'audio': audio,
            'predicted': pred,
            'reference': ref,
            'wer': s_wer,
            'latency_seconds': latency,
        })
        if s_wer is not None:
            per_sample_wers.append(s_wer)
        if latency is not None:
            latencies.append(latency)

    overall_wer = sum(per_sample_wers) / len(per_sample_wers) if per_sample_wers else None
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    report = {
        'model_id': model_id,
        'architecture': arch,
        'base_language': base_lang,
        'overall_wer': overall_wer,
        'average_latency_seconds': avg_latency,
        'samples': rows,
    }

    # write JSON
    out_json = os.path.join(output_dir, 'smoke_report.json')
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    # write CSV
    csv_path = os.path.join(output_dir, 'smoke_report.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['audio','predicted','reference','wer','latency_seconds'])
        for r in rows:
            writer.writerow([r['audio'], r['predicted'], r['reference'], r['wer'], r['latency_seconds']])

    # write markdown table
    md_path = os.path.join(output_dir, 'smoke_report.md')
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write('| audio | predicted (Devanagari) | reference (Devanagari) | wer | latency_s |\n')
        fh.write('|---|---|---|---:|---:|\n')
        for r in rows:
            audio_link = r['audio']
            pred = r['predicted'].replace('\n',' ')
            ref = r['reference'].replace('\n',' ')
            wer_s = f"{r['wer']:.3f}" if r['wer'] is not None else ''
            lat = f"{r['latency_seconds']:.3f}" if r['latency_seconds'] is not None else ''
            fh.write(f"| {audio_link} | {pred} | {ref} | {wer_s} | {lat} |\n")

    print('Report generated:', out_json, csv_path, md_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', required=True, help='HF model id or local checkpoint path')
    parser.add_argument('--manifest', required=True, help='Manifest jsonl to evaluate (one json per line with audio_filepath and text)')
    parser.add_argument('--output_dir', default='results/smoke_report', help='Where to write report files')
    parser.add_argument('--base_lang', default='kok', help='Base language id used for finetuning (e.g., kok, mar)')
    parser.add_argument('--device', default=None, help='torch device (cuda/cpu); default auto')
    parser.add_argument('--checkpoint', default=None, help='Optional local checkpoint directory (to load finetuned weights)')

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and (args.device is None or args.device=='cuda') else 'cpu')

    manifest = load_manifest(args.manifest, audio_root=os.path.dirname(args.manifest))
    make_report(args.model_id, manifest, args.output_dir, base_lang=args.base_lang, device=device, checkpoint=args.checkpoint)


if __name__ == '__main__':
    main()
