#!/usr/bin/env python3
"""
Konkani ASR Fine-tuning Script using Hugging Face
Fine-tunes a Marathi ASR model with Konkani data for better performance
"""

import os
import json
import torch
import yaml
from transformers import (
    Wav2Vec2BertProcessor,
    Wav2Vec2BertForCTC,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, Audio
import librosa
import numpy as np
from pathlib import Path

class KonkaniASRTrainer:
    def __init__(self, config_path="configs/main_config.yaml"):
        # Load main configuration
        with open(config_path, 'r') as f:
            self.main_config = yaml.safe_load(f)

        # Load framework-specific configuration
        framework = self.main_config['framework']
        framework_config_path = self.main_config['frameworks'][framework]['config_file']

        with open(framework_config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up device and model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.framework = framework
        self.base_model = self.main_config['frameworks'][framework]['base_model']
        self.model_dir = self.main_config['frameworks'][framework]['model_dir']

        print(f"Using framework: {framework}")
        print(f"Base model: {self.base_model}")
        print(f"Output directory: {self.model_dir}")
        print(f"Device: {self.device}")

        # Load processor and model
        print("Loading ASR model and processor...")
        self.processor = Wav2Vec2BertProcessor.from_pretrained(self.base_model)
        self.model = Wav2Vec2BertForCTC.from_pretrained(self.base_model)
        self.model.to(self.device)

        # Freeze feature encoder if specified
        if self.config['model'].get('freeze_feature_encoder', True):
            self._freeze_feature_encoder()

    def _freeze_feature_encoder(self):
        """Freeze the feature encoder layers for fine-tuning"""
        try:
            # Try different possible attribute names
            if hasattr(self.model, 'wav2vec2'):
                self.model.wav2vec2.feature_extractor.requires_grad_(False)
                if self.config['model'].get('freeze_feature_projection', True):
                    self.model.wav2vec2.feature_projection.requires_grad_(False)
            elif hasattr(self.model, 'feature_extractor'):
                self.model.feature_extractor.requires_grad_(False)
            else:
                print("Warning: Could not find feature extractor to freeze")
        except Exception as e:
            print(f"Warning: Could not freeze feature encoder: {e}")

    def load_konkani_dataset(self):
        """Load Konkani dataset from TSV manifest files"""
        print("Loading Konkani dataset...")

        data_dir = self.main_config['common']['data_dir']
        audio_dir = self.main_config['common']['audio_dir']

        def load_manifest(manifest_path):
            data = []
            with open(manifest_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line.strip())
                        # Keep the relative path for now, we'll load audio manually
                        data.append(item)
            return data

        # Load train, dev, test data - use small subset for testing
        train_data = load_manifest(os.path.join(data_dir, "train.tsv"))[:1]  # Just 1 sample for testing
        dev_data = load_manifest(os.path.join(data_dir, "dev.tsv"))[:1]      # Just 1 sample for testing
        test_data = load_manifest(os.path.join(data_dir, "test.tsv"))[:1]    # Just 1 sample for testing

        print(f"Loaded {len(train_data)} train, {len(dev_data)} dev, {len(test_data)} test samples")

        # Convert to Hugging Face datasets (without Audio casting)
        def create_dataset(data):
            return Dataset.from_list(data)

        train_dataset = create_dataset(train_data)
        dev_dataset = create_dataset(dev_data)
        test_dataset = create_dataset(test_data)

        return train_dataset, dev_dataset, test_dataset

    def prepare_dataset(self, dataset):
        """Prepare dataset for training"""
        def process_sample(batch):
            # Process text labels using tokenizer
            labels = self.processor.tokenizer(
                batch["text"],
                return_tensors="pt",
                padding=True
            ).input_ids

            result = {
                "audio_filepath": batch["audio_filepath"],  # Keep filepath for data collator
                "labels": labels.squeeze().tolist()  # Convert to list for collation
            }
            return result

        # Process dataset - keep audio_filepath column
        processed_dataset = dataset.map(process_sample, remove_columns=["text", "duration", "valid"])
        print(f"Processed dataset features: {list(processed_dataset.features.keys())}")
        print(f"First example keys: {list(processed_dataset[0].keys())}")

        return processed_dataset

    def fine_tune(self, train_dataset, dev_dataset):
        """Fine-tune the model on Konkani data"""
        print("Starting fine-tuning...")

        # Get training config
        training_config = self.config['training']

        # Prepare datasets
        train_dataset = self.prepare_dataset(train_dataset)
        dev_dataset = self.prepare_dataset(dev_dataset)

        # Training arguments from config
        training_args = TrainingArguments(
            output_dir=training_config['output_dir'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=float(training_config['learning_rate']),  # Ensure it's a float
            warmup_steps=training_config['warmup_steps'],
            max_steps=training_config['max_steps'],
            logging_steps=training_config['logging_steps'],
            save_steps=training_config['save_steps'],
            eval_steps=training_config['eval_steps'],
            eval_strategy=training_config['eval_strategy'],
            save_strategy=training_config['save_strategy'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            push_to_hub=training_config['push_to_hub'],
            remove_unused_columns=False,  # Keep audio_filepath for data collator
            report_to="none",  # Disable wandb and other reporting
        )

        # Simple data collator for batch size 1
        def data_collator(features):
            # For batch size 1, just process single sample
            feature = features[0]

            # Load audio - try to get filepath
            if "audio_filepath" in feature:
                audio_path = os.path.join(self.main_config['common']['audio_dir'], feature["audio_filepath"])
            else:
                # Fallback - this shouldn't happen
                print("ERROR: No audio_filepath found!")
                return None

            # Load audio - handle both M4A and WAV files
            try:
                audio_array, _ = librosa.load(audio_path, sr=self.config['data']['sampling_rate'])
            except Exception as e:
                # If librosa fails (e.g., M4A files), try converting with ffmpeg
                print(f"Librosa failed to load {audio_path}, trying ffmpeg conversion...")
                import subprocess
                import tempfile

                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_wav_path = temp_file.name

                # Convert M4A to WAV using ffmpeg
                try:
                    subprocess.run([
                        'ffmpeg/ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe', '-i', audio_path, '-acodec', 'pcm_s16le',
                        '-ar', str(self.config['data']['sampling_rate']), temp_wav_path,
                        '-y', '-loglevel', 'quiet'
                    ], check=True)

                    # Load the converted WAV file
                    audio_array, _ = librosa.load(temp_wav_path, sr=self.config['data']['sampling_rate'])

                    # Clean up temp file
                    os.unlink(temp_wav_path)

                except subprocess.CalledProcessError as ffmpeg_error:
                    print(f"FFmpeg conversion failed: {ffmpeg_error}")
                    return None

            labels = feature["labels"]

            # Process audio through feature extractor
            batch = self.processor.feature_extractor(
                audio_array,
                sampling_rate=self.config['data']['sampling_rate'],
                return_tensors="pt"
            )

            batch["labels"] = torch.tensor(labels).unsqueeze(0)  # Add batch dimension
            return batch

        data_collator_fn = data_collator

        # Compute WER metric
        from transformers import Wav2Vec2BertProcessor
        import evaluate

        wer_metric = evaluate.load("wer")

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id

            pred_str = self.processor.batch_decode(pred_ids)
            label_str = self.processor.batch_decode(pred.label_ids, skip_special_tokens=True)

            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            return {"wer": wer}

        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=data_collator_fn,
            compute_metrics=compute_metrics,
        )

        # Train
        trainer.train()

        # Save model
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        print(f"Model saved to {output_dir}")
        return trainer

    def test_model(self, test_dataset):
        """Test the fine-tuned model"""
        print("Testing fine-tuned model...")

        # Load fine-tuned model
        model_path = self.config['training']['output_dir']
        processor = Wav2Vec2BertProcessor.from_pretrained(model_path)
        model = Wav2Vec2BertForCTC.from_pretrained(model_path)
        model.to(self.device)

        # Create ASR pipeline
        from transformers import pipeline
        asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=0 if self.device == "cuda" else -1
        )

        # Test on configured number of samples
        num_test_samples = self.config['testing']['num_test_samples']
        test_samples = test_dataset.select(range(min(num_test_samples, len(test_dataset))))

        print("\n🔍 Testing Results:")
        print("=" * 50)

        for i, sample in enumerate(test_samples):
            audio_path = sample["audio_filepath"]
            expected_text = sample["text"]

            # Transcribe
            result = asr(audio_path)
            predicted_text = result["text"]

            print(f"\nTest Sample {i+1}:")
            print(f"Audio: {os.path.basename(audio_path)}")
            print(f"Expected: {expected_text}")
            print(f"Predicted: {predicted_text}")
            print("-" * 30)

def main():
    print("🎯 Konkani ASR Fine-tuning with Configurable Framework")
    print("=" * 60)

    # Load configuration
    config_path = "configs/main_config.yaml"
    with open(config_path, 'r') as f:
        main_config = yaml.safe_load(f)

    framework = main_config['framework']
    print(f"Using framework: {framework.upper()}")
    print(f"Source language: {main_config['common']['source_language']}")
    print(f"Target language: {main_config['common']['target_language']}")
    print()

    if framework == "huggingface":
        # Initialize trainer
        trainer = KonkaniASRTrainer()

        # Load datasets
        train_dataset, dev_dataset, test_dataset = trainer.load_konkani_dataset()

        # Fine-tune model
        trained_trainer = trainer.fine_tune(train_dataset, dev_dataset)

        # Test model
        trainer.test_model(test_dataset)

        print("\n✅ Fine-tuning complete!")
        print(f"Your {framework.upper()} Konkani ASR model is ready in: {trainer.model_dir}")

    elif framework == "nemo":
        print("NeMo framework selected.")
        print("Note: NeMo works best on Unix/Mac systems.")
        print("To use NeMo, run: python scripts/fine_tune_nemo.py")

    elif framework == "ai4bharat":
        print("AI4Bharat framework selected.")
        print("Note: May require authentication token.")
        print("To use AI4Bharat, run: python scripts/fine_tune_ai4bharat.py")

    else:
        print(f"Unknown framework: {framework}")
        print("Available frameworks: huggingface, nemo, ai4bharat")

if __name__ == "__main__":
    main()