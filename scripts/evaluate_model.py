#!/usr/bin/env python3
import torch
import yaml
import pandas as pd
import json
import os
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from datasets import Dataset, Audio
import evaluate


def evaluate_model():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load Model & Processor
    model_path = os.path.join(config["training"]["output_dir"], "final_model")
    print(f"Loading model from {model_path}...")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data (Story 3)
    data = []
    with open(config["data"]["test_manifest"], 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            # Extra safety check for Story 3
            if any(s in entry.get("audio_filepath", "") for s in config["data"].get("test_story_ids", [])):
                data.append(entry)
            elif config["experiment"].get("smoke_test", False):
                 # In smoke mode, accept whatever is in the test manifest if story ID parsing fails
                 data.append(entry)

    df_test = pd.DataFrame(data)
    
    if config["experiment"].get("smoke_test", False):
        df_test = df_test.head(config['experiment']['max_test_samples_smoke'])

    if df_test.empty:
        print('No test samples found matching story IDs or test manifest is empty.')
        return

    dataset = Dataset.from_pandas(df_test)

    wer_metric = evaluate.load("wer")
    results = []

    print(f"Running Inference on {len(dataset)} samples...")

    import soundfile as sf
    for i, batch in enumerate(dataset):
        audio_path = batch["audio_filepath"]
        # Normalize and resolve path (handle windows backslashes)
        audio_path = audio_path.replace('\\\\', '/').replace('\\', '/')
        audio_path = os.path.normpath(audio_path)
        if not os.path.isabs(audio_path):
            audio_path = os.path.normpath(os.path.join(os.getcwd(), audio_path))

        audio_arr, sr = sf.read(audio_path)
        input_values = processor(audio_arr, sampling_rate=sr, return_tensors="pt", padding="longest").input_values.to(model.device)

        # INFERENCE
        with torch.no_grad():
            logits = model(input_values).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_str = processor.batch_decode(pred_ids)[0]
        
        # Calculate individual WER
        local_wer = wer_metric.compute(predictions=[pred_str], references=[batch["text"]])

        results.append({
            "audio_file": audio_path,
            "ground_truth": batch["text"],
            "prediction": pred_str,
            "wer": local_wer
        })
        print(f"Sample {i}: GT='{batch['text']}' | Pred='{pred_str}'")

    # Save Report
    os.makedirs('results', exist_ok=True)
    df_results = pd.DataFrame(results)
    report_path = os.path.join("results", "evaluation_report.csv")
    df_results.to_csv(report_path, index=False)
    
    print(f"\n✅ Report Card generated: {report_path}")
    print(f"Average WER: {df_results['wer'].mean()}")

if __name__ == "__main__":
    evaluate_model()
