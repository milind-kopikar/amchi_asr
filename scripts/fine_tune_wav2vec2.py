#!/usr/bin/env python3
"""
Fine-tune wav2vec2 model for Konkani ASR using transformers
"""

import os
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_manifest_data(manifest_file):
    """Load data from manifest file"""
    data = []
    with open(manifest_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data

def prepare_dataset(train_manifest, test_manifest, processor):
    """Prepare dataset for training"""

    # Load data
    train_data = load_manifest_data(train_manifest)
    test_data = load_manifest_data(test_manifest)

    def process_sample(sample):
        # Load audio
        audio_path = sample['audio_filepath']
        if not os.path.isabs(audio_path):
            # Assume relative to data/audio
            audio_path = os.path.join('data/audio', audio_path)

        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Process audio
        input_values = processor(waveform.squeeze(), sampling_rate=16000).input_values[0]

        # Process text
        with processor.as_target_processor():
            labels = processor(sample['text']).input_ids

        return {
            'input_values': input_values,
            'labels': labels
        }

    # Create datasets
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)

    # Process datasets
    train_dataset = train_dataset.map(process_sample, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(process_sample, remove_columns=test_dataset.column_names)

    return DatasetDict({
        'train': train_dataset,
        'test': test_dataset
    })

def compute_metrics(pred):
    """Compute WER and CER metrics"""
    from jiwer import wer, cer

    pred_logits = pred.predictions
    pred_ids = torch.argmax(pred_logits, dim=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, skip_special_tokens=True)

    wer_score = wer(label_str, pred_str)
    cer_score = cer(label_str, pred_str)

    return {"wer": wer_score, "cer": cer_score}

def main():
    # Model and processor
    model_name = "models/wav2vec2_xlsr"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    # Freeze feature encoder
    model.freeze_feature_encoder()

    # Prepare dataset
    datasets = prepare_dataset(
        'data/test_run/train_wav.tsv',
        'data/test_run/test_wav.tsv',
        processor
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="results/wav2vec2_konkani",
        group_by_length=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=10,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['test'],
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save model
    trainer.save_model("models/wav2vec2_konkani_finetuned")
    processor.save_pretrained("models/wav2vec2_konkani_finetuned")

    logger.info("Fine-tuning completed!")

if __name__ == "__main__":
    main()