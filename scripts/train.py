#!/usr/bin/env python3
import os
import yaml
import torch
import json
import pandas as pd
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Trainer,
    TrainingArguments,
)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import numpy as np
import evaluate

# --- 1. Data Collator (Standard CTC) ---
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch

# --- 2. Metric Computation ---
wer_metric = evaluate.load("wer")
processor = None  # will be set in train()

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_label_ids = pred.label_ids.copy()
    pred_label_ids[pred_label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred_label_ids, skip_special_tokens=True)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 3. Main Training Logic ---
def train():
    import argparse

    # CLI overrides for safer production runs
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 model with robust preflight checks")
    parser.add_argument("--manifest-train", type=str, help="Path to train manifest (overrides config)")
    parser.add_argument("--manifest-val", type=str, help="Path to val manifest (overrides config)")
    parser.add_argument("--manifest-test", type=str, help="Path to test manifest (overrides config)")
    parser.add_argument("--auto-download-missing", action='store_true', help="Attempt to fetch missing audio files from Railway if RAILWAY_URL is set")
    parser.add_argument("--preflight-only", action='store_true', help="Run preflight checks only and exit (no training)")
    args = parser.parse_args()

    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Processor (must exist from previous step)
    processor_path = config["model"].get("processor_path", "data/processor_devanagari")
    print(f"Loading processor from {processor_path}...")
    global processor
    processor = Wav2Vec2Processor.from_pretrained(processor_path)

    # If smoke_test, prefer data_smoke manifests when available (quick verification)
    if config.get('experiment', {}).get('smoke_test', False):
        if os.path.exists('data_smoke/train/manifest.jsonl'):
            print('Smoke test: overriding train/val/test manifests with data_smoke manifests')
            config['data']['train_manifest'] = 'data_smoke/train/manifest.jsonl'
            config['data']['val_manifest'] = 'data_smoke/dev/manifest.jsonl'
            config['data']['test_manifest'] = 'data_smoke/test/manifest.jsonl'

    # Allow CLI manifest overrides (useful for production runs)
    if args.manifest_train:
        print(f"Overriding train manifest with {args.manifest_train}")
        config['data']['train_manifest'] = args.manifest_train
    if args.manifest_val:
        print(f"Overriding val manifest with {args.manifest_val}")
        config['data']['val_manifest'] = args.manifest_val
    if args.manifest_test:
        print(f"Overriding test manifest with {args.manifest_test}")
        config['data']['test_manifest'] = args.manifest_test

    # Load Data
    def load_manifest(json_path):
        data = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    print("Loading manifests...")
    df_train = load_manifest(config["data"]["train_manifest"]) if os.path.exists(config["data"].get("train_manifest")) else pd.DataFrame([])
    df_val = load_manifest(config["data"].get("val_manifest")) if os.path.exists(config["data"].get("val_manifest")) else pd.DataFrame([])
    df_test = load_manifest(config["data"].get("test_manifest")) if os.path.exists(config["data"].get("test_manifest")) else pd.DataFrame([])

    # SMOKE TEST LOGIC
    if config["experiment"].get("smoke_test", False):
        print("\n🔥 SMOKE TEST MODE ACTIVATED 🔥")
        print(f"Slicing data to {config['experiment']['max_train_samples_smoke']} samples.")
        df_train = df_train.head(config['experiment']['max_train_samples_smoke'])
        df_val = df_val.head(config['experiment']['max_val_samples_smoke'])

    # Resolve audio paths (support Windows-style paths and fallback to data_smoke/data audio dirs)
    def resolve_audio_path(p):
        import glob
        if not isinstance(p, str):
            return None
        p2 = p.replace('\\\\', os.path.sep).replace('\\', os.path.sep)
        p2 = os.path.normpath(p2)
        if not os.path.isabs(p2):
            p2 = os.path.normpath(os.path.join(os.getcwd(), p2))
        if os.path.exists(p2):
            return p2
        # Fallback: look for basename in common data directories
        b = os.path.basename(p)
        search_dirs = ['data_smoke/train/audio', 'data_smoke/dev/audio', 'data_smoke/test/audio', 'data/train/audio', 'data/dev/audio', 'data/test/audio', 'data/audio']
        for d in search_dirs:
            cand = os.path.join(os.getcwd(), d, b)
            if os.path.exists(cand):
                print(f"Note: resolved {p} -> {cand}")
                return cand
        return None

    def preflight_check_and_fix(df, split_name):
        """Validate that all audio files exist. If missing and --auto-download-missing is set, attempt to fetch them from Railway."""
        missing = []
        if df is None or df.empty:
            return missing
        for i, row in df.iterrows():
            resolved = resolve_audio_path(row['audio_filepath'])
            if resolved is None:
                missing.append(row['audio_filepath'])
        if missing:
            print(f"⚠️ {len(missing)} missing audio files detected in {split_name} manifest.")
            # If user requested auto-download and Railway is available, attempt to fetch
            if args.auto_download_missing and os.getenv('RAILWAY_URL'):
                print('Attempting to auto-download missing files from Railway...')
                try:
                    from scripts import download_data_from_railway as dl
                    base = os.getenv('RAILWAY_URL')
                    # fetch recordings list
                    recs = dl.fetch_recordings_list(base)
                    rec_map = {str(r['id']): r for r in recs}
                    # try to find matching ids by basename
                    for m in missing[:]:
                        bn = os.path.basename(m)
                        id_ = os.path.splitext(bn)[0]
                        if id_ in rec_map:
                            print(f"Downloading missing recording id {id_}...")
                            dl.download_recordings(base, [rec_map[id_]], os.path.join('data', split_name))
                            missing.remove(m)
                except Exception as e:
                    print('Auto-download failed:', e)
        return missing

    # Preflight: check missing and optionally auto-download
    missing_train = preflight_check_and_fix(df_train, 'train')
    missing_val = preflight_check_and_fix(df_val, 'dev')
    missing_test = preflight_check_and_fix(df_test, 'test')

    if missing_train or missing_val or missing_test:
        # Provide detailed diagnostics and fail early (don't start a long training run with missing data)
        print('\n🚨 Preflight failed. Missing audio files detected:')
        if missing_train:
            print(f"  - train: {len(missing_train)} files (examples: {missing_train[:5]})")
        if missing_val:
            print(f"  - dev: {len(missing_val)} files (examples: {missing_val[:5]})")
        if missing_test:
            print(f"  - test: {len(missing_test)} files (examples: {missing_test[:5]})")
        print('Resolve missing files (place them in the expected paths) or run the downloader with RAILWAY_URL set, then retry.')
        raise RuntimeError('Preflight failed: missing audio files')

    # If user requested preflight-only, exit successfully here (all checks passed)
    if args.preflight_only:
        total_train = len(df_train.index) if not df_train.empty else 0
        total_dev = len(df_val.index) if not df_val.empty else 0
        total_test = len(df_test.index) if not df_test.empty else 0
        print('\n✅ Preflight checks passed.')
        print(f"  Train samples: {total_train}")
        print(f"  Dev samples:   {total_dev}")
        print(f"  Test samples:  {total_test}")
        print('Exiting due to --preflight-only')
        return

    # Apply resolution and filter missing (now we know files are present)
    if not df_train.empty:
        df_train['audio_resolved'] = df_train['audio_filepath'].apply(resolve_audio_path)
        df_train = df_train[df_train['audio_resolved'].notnull()].copy()
        df_train['audio_filepath'] = df_train['audio_resolved']
        df_train.drop(columns=['audio_resolved'], inplace=True)

    if not df_val.empty:
        df_val['audio_resolved'] = df_val['audio_filepath'].apply(resolve_audio_path)
        df_val = df_val[df_val['audio_resolved'].notnull()].copy()
        df_val['audio_filepath'] = df_val['audio_resolved']
        df_val.drop(columns=['audio_resolved'], inplace=True)

    if not df_test.empty:
        df_test['audio_resolved'] = df_test['audio_filepath'].apply(resolve_audio_path)
        df_test = df_test[df_test['audio_resolved'].notnull()].copy()
        df_test['audio_filepath'] = df_test['audio_resolved']
        df_test.drop(columns=['audio_resolved'], inplace=True)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(df_train) if not df_train.empty else Dataset.from_list([])
    val_dataset = Dataset.from_pandas(df_val) if not df_val.empty else Dataset.from_list([])

    # Do not rely on datasets' Audio decoding (torchcodec may be incompatible). We'll load audio files manually during mapping.

    def process_data(batch):
        audio_path = batch["audio_filepath"]
        # Normalize path (handle Windows backslashes) and make absolute relative to repo root
        # Normalize backslashes to current OS separator, then normalize path
        audio_path = audio_path.replace('\\', os.path.sep)
        audio_path = os.path.normpath(audio_path)
        if not os.path.isabs(audio_path):
            audio_path = os.path.normpath(os.path.join(os.getcwd(), audio_path))

        import soundfile as sf
        try:
            audio_arr, sr = sf.read(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file {audio_path}: {e}")
        try:
            batch["input_values"] = processor(audio_arr, sampling_rate=sr).input_values[0]
        except Exception as e:
            raise RuntimeError(f"Processor failed for {audio_path}: {e}")
        with processor.as_target_processor():
            batch["labels"] = processor(batch["text"]).input_ids
        return batch

    if len(train_dataset) > 0:
        print("Processing train dataset (manual audio loading)...")
        train_dataset = train_dataset.map(process_data, remove_columns=[c for c in train_dataset.column_names if c not in ["input_values","labels"]])
    if len(val_dataset) > 0:
        print("Processing val dataset (manual audio loading)...")
        val_dataset = val_dataset.map(process_data, remove_columns=[c for c in val_dataset.column_names if c not in ["input_values","labels"]])

    # Load Model (THE BRAIN TRANSPLANT)
    print("Loading Model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        config["model"]["base_model"],
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )

    # Freeze the heavy parts
    try:
        model.freeze_feature_encoder()
    except Exception:
        # fallback: freeze parameters manually
        for name, p in model.named_parameters():
            if "lm_head" not in name:
                p.requires_grad = False

    # Training Args (disable eval if no val dataset present)
    eval_strategy = "steps" if (len(val_dataset) > 0) else "no"
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        group_by_length=False,
        per_device_train_batch_size=config["training"]["batch_size"],
        eval_strategy=eval_strategy,
        num_train_epochs=config["training"]["num_epochs"],
        fp16=torch.cuda.is_available(),
        save_steps=config["training"].get("save_steps", 50),
        eval_steps=config["training"].get("eval_steps", 50),
        logging_steps=int(config["training"].get("logging_steps", 50)),
        learning_rate=float(config["training"].get("learning_rate", 1e-4)),
        warmup_steps=int(config["training"].get("warmup_steps", 100)),
        save_total_limit=int(config["training"].get("save_total_limit", 2)),
    )

    trainer = Trainer(
        model=model,
        data_collator=DataCollatorCTCWithPadding(processor=processor, padding=True),
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset if len(train_dataset) > 0 else None,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        tokenizer=processor.tokenizer,
    )

    print("Starting Training...")
    trainer.train()

    # Save Final
    out_dir = os.path.join(config["training"]["output_dir"], "final_model")
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    print("✅ Training Complete. Model saved.")


if __name__ == "__main__":
    train()