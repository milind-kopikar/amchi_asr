#!/usr/bin/env python3
"""Minimal MMS (facebook/mms-1b-all) smoke test:
- Downloads minimal smoke data (3 train,1 dev,1 test)
- Runs a very short fine-tune (few steps) on `kok` language
- Evaluates on dev set and prints WER
- Runs inference on test set and prints predictions

Designed for quick verification on a GPU host.
"""

import os
import sys
import json
import argparse
import shutil
import subprocess
from pathlib import Path

try:
    from datasets import load_dataset, Audio, Dataset
    from transformers import (
        AutoProcessor,
        AutoModelForCTC,
        TrainingArguments,
        Trainer,
    )
    import evaluate
    import torch
    import numpy as np
except Exception as e:
    print("Missing dependencies. Please run: pip install transformers datasets evaluate librosa soundfile huggingface_hub")
    print(e)
    sys.exit(1)


def download_smoke_data(output_dir='data_smoke'):
    # Use the bundle script if present
    print(f"Ensuring smoke data at {output_dir}...")
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print("Smoke data already present, skipping download.")
        return
    cmd = [sys.executable, 'scripts/download_smoke_test_data.py', '--n_train', '3', '--n_dev', '1', '--n_test', '1', '--output_dir', output_dir]
    print('Running:', ' '.join(cmd))
    subprocess.check_call(cmd)


def load_manifest_dataset(manifest_path, audio_root=None):
    # manifest is jsonl with audio_filepath and text
    examples = []
    with open(manifest_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            j = json.loads(line)
            audio = j.get('audio_filepath') or j.get('audio')
            # normalize relative paths
            if not os.path.isabs(audio) and audio_root:
                audio = os.path.normpath(os.path.join(audio_root, os.path.basename(audio)))
            examples.append({'audio': audio, 'text': j.get('text', '')})
    ds = Dataset.from_list(examples)
    # Avoid datasets.Audio decoding (torchcodec issues); we'll load audio manually in prepare step
    return ds


class DataCollatorCTC:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features):
        # features: list of dicts with 'audio' and 'text'
        input_values = [f['input_values'] for f in features]
        labels = [f['labels'] for f in features]
        batch = self.processor.pad({'input_values': input_values}, return_tensors='pt')
        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer(labels, padding=True, return_tensors='pt').input_ids
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels_batch[labels_batch == self.processor.tokenizer.pad_token_id] = -100
        batch['labels'] = labels_batch
        return batch


def prepare_dataset(ds, processor, text_column='text'):
    def prepare_batch(batch):
        # load audio from disk (avoid datasets audio decoding to prevent torchcodec issues)
        audio_path = batch['audio']
        import soundfile as sf
        audio_arr, sr = sf.read(audio_path)
        proc = processor(audio_arr, sampling_rate=sr, return_tensors=None)
        batch['input_values'] = proc['input_values'][0]
        batch['labels'] = batch['text']
        return batch

    ds_proc = ds.map(prepare_batch, remove_columns=['audio', 'text'])
    return ds_proc


def compute_wer(predictions, references):
    wer_metric = evaluate.load('wer')
    return wer_metric.compute(predictions=predictions, references=references)


def main(args):
    output_dir = args.output_dir
    download_smoke_data('data_smoke')

    # manifest paths
    train_manifest = os.path.join('data_smoke', 'train', 'manifest.jsonl')
    dev_manifest = os.path.join('data_smoke', 'dev', 'manifest.jsonl')
    test_manifest = os.path.join('data_smoke', 'test', 'manifest.jsonl')

    for p in (train_manifest, dev_manifest, test_manifest):
        if not os.path.exists(p):
            print('Missing manifest:', p)
            sys.exit(1)

    # load datasets
    audio_root = os.path.join('data_smoke')
    train_ds = load_manifest_dataset(train_manifest, audio_root=os.path.join('data_smoke','train','audio'))
    dev_ds = load_manifest_dataset(dev_manifest, audio_root=os.path.join('data_smoke','dev','audio'))
    test_ds = load_manifest_dataset(test_manifest, audio_root=os.path.join('data_smoke','test','audio'))

    # load processor and model
    model_id = 'facebook/mms-1b-all'
    print('Loading processor and model:', model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCTC.from_pretrained(model_id).to('cuda' if torch.cuda.is_available() else 'cpu')

    # optional: freeze encoder to speed up smoke training
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if 'lm_head' not in name:
                param.requires_grad = False
        print('Encoder frozen (only head will be trained)')

    # prepare datasets
    print('Preparing datasets (this may load audio files)')
    train_p = prepare_dataset(train_ds, processor)
    dev_p = prepare_dataset(dev_ds, processor)
    test_p = prepare_dataset(test_ds, processor)

    # Data collator
    data_collator = DataCollatorCTC(processor)

    # training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        max_steps=args.max_steps if args.max_steps>0 else -1,
        eval_strategy='steps',
        eval_steps=10,
        save_steps=10,
        logging_steps=5,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to='none'
    )

    # make a minimal Trainer - we need to provide dataset format with proper tensors
    def collate_fn(examples):
        # examples are dicts with 'input_values' (numpy array) and 'labels' (str)
        feats = []
        for ex in examples:
            feats.append({'input_values': ex['input_values'], 'labels': ex['labels']})
        return data_collator(feats)

    # Train
    print('Starting short training run...')
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_p,
        eval_dataset=dev_p,
        data_collator=collate_fn,
    )

    try:
        trainer.train()
    except Exception as e:
        print('Training failed:', e)
        print('Attempting a faster validation-only run using base model...')

    # Evaluate on dev
    print('Running inference on dev set...')
    dev_preds = []
    dev_refs = []
    pipe = None
    try:
        from transformers import pipeline
        pipe = pipeline('automatic-speech-recognition', model=model, processor=processor, device=0 if torch.cuda.is_available() else -1)
    except Exception:
        print('Pipeline failed to init; falling back to manual transcription')

    for ex in dev_ds:
        audio_path = ex['audio']
        ref = ex['text']
        try:
            if pipe:
                pred = pipe(audio_path)['text']
            else:
                # manual inference
                audio_arr, sr = (None, None)
                import soundfile as sf
                audio_arr, sr = sf.read(audio_path)
                inputs = processor(audio_arr, sampling_rate=sr, return_tensors='pt', padding=True)
                inputs = {k: v.to(model.device) for k,v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)[0]
                pred = processor.batch_decode(predicted_ids.unsqueeze(0))[0]
            print(f"Ref: {ref}\nPred: {pred}\n---")
            dev_preds.append(pred)
            dev_refs.append(ref)
        except Exception as e:
            print('Inference failed for', audio_path, e)

    try:
        wer = compute_wer(dev_preds, dev_refs)
        print(f'Average WER (dev): {wer:.2%}')
    except Exception as e:
        print('Failed to compute WER:', e)

    # Test inference on test set
    print('Running inference on test set...')
    for ex in test_ds:
        audio_path = ex['audio']
        ref = ex['text']
        try:
            if pipe:
                pred = pipe(audio_path)['text']
            else:
                import soundfile as sf
                audio_arr, sr = sf.read(audio_path)
                inputs = processor(audio_arr, sampling_rate=sr, return_tensors='pt', padding=True)
                inputs = {k: v.to(model.device) for k,v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)[0]
                pred = processor.batch_decode(predicted_ids.unsqueeze(0))[0]
            print(f"Test Ref: {ref}\nTest Pred: {pred}\n===")
        except Exception as e:
            print('Test inference failed for', audio_path, e)

    print('Smoke test complete. If you want an inference endpoint, run: python scripts/hf_inference_endpoint.py')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='results/mms_smoke', help='where to save checkpoints')
    parser.add_argument('--max_steps', type=int, default=20, help='max_steps for trainer (0 = use epochs)')
    parser.add_argument('--freeze_encoder', action='store_true', help='freeze encoder parameters (only head trained)')
    args = parser.parse_args()
    main(args)
