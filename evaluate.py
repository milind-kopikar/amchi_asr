import os
import csv
import yaml
from datasets import load_from_disk
from transformers import AutoProcessor
from jiwer import wer, cer

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    output_dir = config['project']['output_dir']
    dataset_dir = os.path.join(output_dir, 'hf_dataset')
    model_dir = os.path.join(output_dir, 'final_model')
    report_path = os.path.join(output_dir, 'reports', 'evaluation_report.csv')

    dataset = load_from_disk(dataset_dir)["test"]
    from transformers import MmsForCTC
    processor = AutoProcessor.from_pretrained(model_dir)
    model = MmsForCTC.from_pretrained(model_dir)
    model.eval()

    rows = []
    wers, cers = [], []
    for item in dataset:
        audio = item["audio"]
        inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        pred = processor.batch_decode(pred_ids)[0].strip()
        ref = item["transcript"].strip()
        w = wer(ref, pred)
        c = cer(ref, pred)
        wers.append(w)
        cers.append(c)
        status = "Pass" if w < 0.25 else "Fail"
        rows.append({
            "audio_filename": os.path.basename(audio["path"]),
            "original_transcript": ref,
            "predicted_transcript": pred,
            "wer_score": w,
            "status": status
        })
    # Write CSV
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        # Summary row
        writer.writerow({
            "audio_filename": "SUMMARY",
            "original_transcript": "",
            "predicted_transcript": "",
            "wer_score": sum(wers)/len(wers) if wers else 0,
            "status": f"Model: {config['model']['base_model']}, Adapter: {config['model']['adapter_lang']}, CER: {sum(cers)/len(cers) if cers else 0}"
        })
    print(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    import torch
    main()
