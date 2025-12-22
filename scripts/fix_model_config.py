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
        
        # Remove AI4Bharat custom parameters not in standard NeMo
        print("🔨 Removing AI4Bharat custom parameters...")
        if 'decoder' in config and isinstance(config['decoder'], dict):
            # Remove multisoftmax parameter
            if 'multisoftmax' in config['decoder']:
                del config['decoder']['multisoftmax']
                print("  - Removed 'multisoftmax' from decoder")
        
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
