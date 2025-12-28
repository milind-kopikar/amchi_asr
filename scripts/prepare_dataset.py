import os
import re
import yaml
from datasets import Dataset, DatasetDict, Audio
from typing import List, Dict

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_story_id(filename: str) -> str:
    # Example: story1_sentence_05.wav -> story1
    match = re.match(r"(story\d+)_", os.path.basename(filename))
    if match:
        return match.group(1)
    # Fallback: use parent folder
    return os.path.basename(os.path.dirname(filename))

def collect_data(dataset_path: str) -> List[Dict]:
    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                audio_path = os.path.join(root, file)
                transcript_path = audio_path.replace('.wav', '.txt')
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r', encoding='utf-8') as t:
                        transcript = t.read().strip()
                    story_id = extract_story_id(file)
                    data.append({
                        'audio': audio_path,
                        'transcript': transcript,
                        'story_id': story_id
                    })
    return data

def split_data(data: List[Dict], test_ids: List[str], val_ids: List[str]) -> DatasetDict:
    train, val, test = [], [], []
    for item in data:
        if item['story_id'] in test_ids:
            test.append(item)
        elif item['story_id'] in val_ids:
            val.append(item)
        else:
            train.append(item)
    # Assert zero overlap
    assert not (set(test_ids) & set(val_ids)), "Test and Validation story IDs overlap!"
    train_ids = set(i['story_id'] for i in train)
    assert not (train_ids & set(test_ids)), "Train and Test story IDs overlap!"
    return DatasetDict({
        'train': Dataset.from_list(train),
        'validation': Dataset.from_list(val),
        'test': Dataset.from_list(test)
    })

def main():
    config = load_config('config.yaml')
    dataset_path = config['data']['dataset_path']
    test_ids = config['data']['test_story_ids']
    val_ids = config['data']['validation_story_ids']
    data = collect_data(dataset_path)
    dsdict = split_data(data, test_ids, val_ids)
    # Cast audio column to Audio type for HF datasets
    for split in dsdict:
        dsdict[split] = dsdict[split].cast_column('audio', Audio())
    output_dir = config['project']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    dsdict.save_to_disk(os.path.join(output_dir, 'hf_dataset'))
    print(f"Saved dataset splits to {os.path.join(output_dir, 'hf_dataset')}")

if __name__ == "__main__":
    main()
