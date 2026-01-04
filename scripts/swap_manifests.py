import json
import os

def swap_manifests(dev_path, test_path):
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_lines = [json.loads(line) for line in f if line.strip()]
    
    with open(test_path, 'r', encoding='utf-8') as f:
        test_lines = [json.loads(line) for line in f if line.strip()]
    
    # Update old dev lines (now test)
    for i, item in enumerate(dev_lines):
        item['audio_filepath'] = item['audio_filepath'].replace('data/dev/audio/', 'data/test/audio/')
        item['sample_id'] = f"test_{i:04d}"
    
    # Update old test lines (now dev)
    for i, item in enumerate(test_lines):
        item['audio_filepath'] = item['audio_filepath'].replace('data/test/audio/', 'data/dev/audio/')
        item['sample_id'] = f"dev_{i:04d}"
    
    # Write back
    with open(dev_path, 'w', encoding='utf-8') as f:
        for item in test_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    with open(test_path, 'w', encoding='utf-8') as f:
        for item in dev_lines:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    swap_manifests('data/dev/manifest.jsonl', 'data/test/manifest.jsonl')
    print("Manifests swapped and updated successfully.")
