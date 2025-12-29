import json
import os
import yaml
from collections import Counter

def build_vocab(config_path="config.yaml"):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. Identify Data Paths
    # Assuming data is split into train/test manifests or a single file
    # Adjust this list based on where your actual manifest.json files are located
    manifest_files = [
        "data/manifest_train.json", 
        "data/manifest_test.json",
        "data/manifest_val.json"
    ]
    
    # Check which files actually exist
    existing_files = [f for f in manifest_files if os.path.exists(f)]
    if not existing_files:
        # Fallback for the user's specific folder structure if needed
        if os.path.exists("final_corpus/manifest.json"):
            existing_files = ["final_corpus/manifest.json"]
        else:
            raise FileNotFoundError(f"Could not find manifest files. Checked: {manifest_files}")

    print(f"Building vocabulary from: {existing_files}")

    # 3. Extract Characters
    all_text = ""
    for json_file in existing_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                if "text" in entry:
                    all_text += " " + entry["text"]

    # Optionally read an extra plain-text corpus to expand Devanagari coverage
    extra_corpus_path = "data/extra_corpus.txt"
    if os.path.exists(extra_corpus_path):
        print(f"Reading extra corpus from: {extra_corpus_path}")
        with open(extra_corpus_path, 'r', encoding='utf-8') as ef:
            all_text += " " + ef.read()

    # 4. Filter and Create Vocab
    # Get unique characters
    chars = set(all_text)
    
    # Remove standard English punctuation if desired, but KEEP Devanagari specific punctuation
    # For now, we keep everything that appeared in the text to be safe
    
    # Create dictionary
    vocab_dict = {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2  # CTC Separator (Space)
    }
    
    # Sort chars for determinism
    sorted_chars = sorted(list(chars))
    
    # Start indexing from 3
    index = 3
    for c in sorted_chars:
        if c.strip() == "": continue # Skip empty/newlines
        if c == " ": continue # Space is handled by "|"
        vocab_dict[c] = index
        index += 1

    # 5. Save
    output_path = config.get("data", {}).get("vocab_file", "data/vocab.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print(f"✅ Vocabulary saved to {output_path}")
    print(f"✅ Total tokens: {len(vocab_dict)}")
    print(f"Sample tokens: {list(vocab_dict.keys())[:10]}...")

if __name__ == "__main__":
    build_vocab()