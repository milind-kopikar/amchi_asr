#!/usr/bin/env python3
"""
Fix the tokenizer config in the downloaded model to work with NeMo 1.23.0
"""

import os
import tarfile
import yaml
import tempfile
import shutil

def fix_model_tokenizer_config(model_path):
    """Extract, fix tokenizer config, and repack the .nemo model"""
    
    print(f"🔧 Fixing tokenizer config in: {model_path}")
    
    # Create temp directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract .nemo file (it's a tar archive)
        print("📦 Extracting model...")
        with tarfile.open(model_path, 'r') as tar:
            tar.extractall(temp_dir)
        
        # Load model config
        config_path = os.path.join(temp_dir, 'model_config.yaml')
        print(f"📄 Reading config from: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Attempt to detect tokenizer vocabulary size from extracted files (if present)
        detected_vocab_size = None
        for root, dirs, files in os.walk(temp_dir):
            for fname in files:
                if fname.endswith('.vocab') or fname.endswith('vocab.txt') or fname.endswith('.vocab.txt'):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, 'r', encoding='utf-8') as vf:
                            lines = [l for l in (l.strip() for l in vf) if l]
                            if lines:
                                detected_vocab_size = len(lines)
                                print(f"🔎 Detected tokenizer vocab file '{fname}' with {detected_vocab_size} entries")
                                break
                    except Exception:
                        continue
            if detected_vocab_size is not None:
                break
        
        # Fix tokenizer config - convert multilingual to monolingual (Marathi only)
        if 'tokenizer' in config:
            if config['tokenizer'].get('type') == 'multilingual':
                print("🔨 Converting multilingual tokenizer to monolingual (Marathi)...")
                # Extract just the Marathi tokenizer config
                mr_tokenizer = config['tokenizer']['langs']['mr'].copy()
                # Set it as the main tokenizer config
                config['tokenizer'] = mr_tokenizer
                print(f"✓ Tokenizer converted to monolingual Marathi")
            elif 'dir' not in config['tokenizer']:
                print("🔨 Adding missing 'dir' field to tokenizer config...")
                config['tokenizer']['dir'] = os.path.dirname(model_path)
                print(f"✓ Tokenizer config fixed")
        
        # Remove AI4Bharat custom parameters (recursively)
        print("🔨 Removing AI4Bharat custom parameters...")
        def remove_key_recursive(d, key_to_remove='multisoftmax'):
            if isinstance(d, dict):
                if key_to_remove in d:
                    del d[key_to_remove]
                    print(f"  - Removed '{key_to_remove}' from dict")
                for v in d.values():
                    remove_key_recursive(v, key_to_remove)
            elif isinstance(d, list):
                for item in d:
                    remove_key_recursive(item, key_to_remove)

        # Remove any 'multisoftmax' occurrences anywhere in the config
        remove_key_recursive(config, 'multisoftmax')

        # Fix potential vocabulary/num_classes mismatches in nested decoder configs.
        # If a tokenizer vocab file was detected, prefer using its size as the num_classes target.
        def fix_decoder_num_classes(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, dict):
                        # If decoder dict with vocabulary present
                        if k == 'decoder' and 'vocabulary' in v and 'num_classes' in v:
                            # Prefer detected tokenizer vocab size if available, otherwise use provided vocabulary length
                            vocab_len = detected_vocab_size if detected_vocab_size is not None else len(v.get('vocabulary', []))
                            if v['num_classes'] != vocab_len:
                                print(f"🔧 Fixing num_classes in decoder from {v['num_classes']} to {vocab_len}")
                                v['num_classes'] = vocab_len
                        else:
                            fix_decoder_num_classes(v)
                    elif isinstance(v, list):
                        for item in v:
                            fix_decoder_num_classes(item)
        fix_decoder_num_classes(config)

        # Remove auxiliary CTC decoder config to avoid size mismatch with tokenizer vocab
        if 'aux_ctc' in config:
            print("🔧 Removing 'aux_ctc' config to avoid decoder/tokenizer size mismatch")
            del config['aux_ctc']

        if 'joint' in config and isinstance(config['joint'], dict):
            # Remove language_keys and multilingual parameters
            if 'language_keys' in config['joint']:
                del config['joint']['language_keys']
                print("  - Removed 'language_keys' from joint")
            if 'multilingual' in config['joint']:
                del config['joint']['multilingual']
                print("  - Removed 'multilingual' from joint")
        
        # Remove return_language_id from data configs
        for data_key in ['train_ds', 'validation_ds', 'test_ds']:
            if data_key in config and isinstance(config[data_key], dict):
                if 'return_language_id' in config[data_key]:
                    del config[data_key]['return_language_id']
                    print(f"  - Removed 'return_language_id' from {data_key}")
        
        print("✓ Custom parameters removed")
        
        # Save fixed config
        print("💾 Saving fixed config...")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        # Repack as .nemo file
        print("📦 Repacking model...")
        backup_path = model_path + '.backup'
        shutil.move(model_path, backup_path)
        
        with tarfile.open(model_path, 'w') as tar:
            for item in os.listdir(temp_dir):
                tar.add(os.path.join(temp_dir, item), arcname=item)
        
        print(f"✅ Model fixed! Original backed up to: {backup_path}")
        print(f"✅ Fixed model saved to: {model_path}")

if __name__ == '__main__':
    model_path = 'models/indicconformer_mr/indicconformer_stt_mr_hybrid_ctc_rnnt_large.nemo'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        exit(1)
    
    fix_model_tokenizer_config(model_path)
