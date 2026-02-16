#!/usr/bin/env python3
"""
Windows-Compatible Pre-Flight Checks

Tests what we CAN test on Windows before RunPod:
1. ✅ Model download (via Hugging Face API)
2. ✅ Data download (via Railway API)
3. ✅ Config file validation
4. ❌ Transcription (requires Linux + NeMo)

Usage:
    # Run all Windows-compatible tests
    python tests/test_preflight.py
    
    # Skip model download (if already downloaded)
    python tests/test_preflight.py --skip-download
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)

MODELS = {
    'marathi': {
        'name': 'ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large',
        'path': 'models/indicconformer_mr',
    },
    'konkani': {
        'name': 'ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large',
        'path': 'models/indicconformer_kok',
    }
}


class PreflightChecker:
    """Run pre-flight checks before RunPod deployment"""
    
    def __init__(self):
        self.results = {
            'huggingface_auth': False,
            'models_downloaded': {},
            'test_data_downloaded': False,
            'config_valid': False
        }
    
    def check_huggingface_auth(self) -> bool:
        """Check if Hugging Face authentication is set up"""
        logger.info("\n" + "="*80)
        logger.info("🔐 Checking Hugging Face Authentication")
        logger.info("="*80)
        
        try:
            from huggingface_hub import HfApi, HfFolder
            
            # Check for token
            token = HfFolder.get_token()
            
            if token:
                # Verify token works
                api = HfApi()
                user = api.whoami(token=token)
                logger.info(f"✅ Authenticated as: {user['name']}")
                self.results['huggingface_auth'] = True
                return True
            else:
                logger.warning("❌ Not authenticated with Hugging Face")
                logger.info("\n💡 To authenticate:")
                logger.info("   huggingface-cli login")
                logger.info("   (Paste your token from https://huggingface.co/settings/tokens)")
                self.results['huggingface_auth'] = False
                return False
                
        except ImportError:
            logger.error("❌ huggingface_hub not installed")
            logger.info("   pip install huggingface_hub")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking authentication: {e}")
            return False
    
    def check_model_download(self, model_key: str, force_download: bool = False) -> bool:
        """
        Check if model is downloaded, optionally download it
        
        Args:
            model_key: 'marathi' or 'konkani'
            force_download: If True, download even if exists
        """
        model_info = MODELS[model_key]
        model_path = model_info['path']
        
        logger.info(f"\n📦 Checking {model_key.capitalize()} Model")
        logger.info(f"   Path: {model_path}")
        
        # Check if already exists
        if os.path.exists(model_path) and not force_download:
            nemo_files = list(Path(model_path).glob("*.nemo"))
            if nemo_files:
                file_size = nemo_files[0].stat().st_size / (1024**3)  # GB
                logger.info(f"   ✅ Model already downloaded ({file_size:.2f} GB)")
                self.results['models_downloaded'][model_key] = True
                return True
        
        # Download model
        logger.info(f"   📥 Downloading from Hugging Face...")
        logger.info(f"   Model: {model_info['name']}")
        
        try:
            from huggingface_hub import snapshot_download
            
            downloaded_path = snapshot_download(
                repo_id=model_info['name'],
                local_dir=model_path,
                ignore_patterns=["*.md", "*.txt", "*.jpg", "*.png"]
            )
            
            # Verify download
            nemo_files = list(Path(model_path).glob("*.nemo"))
            if nemo_files:
                file_size = nemo_files[0].stat().st_size / (1024**3)  # GB
                logger.info(f"   ✅ Downloaded successfully ({file_size:.2f} GB)")
                self.results['models_downloaded'][model_key] = True
                return True
            else:
                logger.error("   ❌ No .nemo file found after download")
                self.results['models_downloaded'][model_key] = False
                return False
                
        except Exception as e:
            logger.error(f"   ❌ Download failed: {e}")
            logger.info("\n   💡 Make sure you:")
            logger.info("      1. Ran: huggingface-cli login")
            logger.info(f"      2. Accepted conditions: https://huggingface.co/{model_info['name']}")
            self.results['models_downloaded'][model_key] = False
            return False
    
    def check_test_data_download(self, n_samples: int = 3) -> bool:
        """Download test samples from Railway"""
        logger.info("\n" + "="*80)
        logger.info(f"📊 Downloading {n_samples} Test Samples from Railway")
        logger.info("="*80)
        
        try:
            import subprocess
            
            test_dir = "test_preflight_data"
            
            cmd = [
                sys.executable,
                "scripts/download_smoke_test_data.py",
                "--n_train", "0",
                "--n_dev", "0",
                "--n_test", str(n_samples),
                "--output_dir", test_dir,
                "--seed", "42"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            test_manifest = os.path.join(test_dir, "test", "manifest.jsonl")
            
            if os.path.exists(test_manifest):
                # Read and display samples
                with open(test_manifest, 'r', encoding='utf-8') as f:
                    samples = [json.loads(line) for line in f]
                
                logger.info(f"✅ Downloaded {len(samples)} samples")
                logger.info("\nSample data:")
                for i, sample in enumerate(samples, 1):
                    logger.info(f"  {i}. Audio: {os.path.basename(sample['audio_filepath'])}")
                    logger.info(f"     Text: {sample['text']}")
                    logger.info(f"     Duration: {sample['duration']:.2f}s")
                
                self.results['test_data_downloaded'] = True
                return True
            else:
                logger.error("❌ Test manifest not created")
                self.results['test_data_downloaded'] = False
                return False
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Download failed: {e.stderr}")
            self.results['test_data_downloaded'] = False
            return False
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            self.results['test_data_downloaded'] = False
            return False
    
    def check_config_files(self) -> bool:
        """Validate config files"""
        logger.info("\n" + "="*80)
        logger.info("⚙️  Checking Configuration Files")
        logger.info("="*80)
        
        configs = [
            'configs/konkani_finetune.yaml',
            'configs/smoke_test.yaml'
        ]
        
        all_valid = True
        
        for config_path in configs:
            logger.info(f"\n📄 {config_path}")
            
            if not os.path.exists(config_path):
                logger.error("   ❌ File not found")
                all_valid = False
                continue
            
            try:
                from omegaconf import OmegaConf
                
                config = OmegaConf.load(config_path)
                logger.info("   ✅ Valid YAML syntax")
                
                # Check key fields
                if hasattr(config, 'model') and hasattr(config.model, 'nemo_model'):
                    logger.info(f"   📂 Model path: {config.model.nemo_model}")
                
                if hasattr(config, 'trainer') and hasattr(config.trainer, 'max_epochs'):
                    logger.info(f"   🔄 Epochs: {config.trainer.max_epochs}")
                
            except ImportError:
                logger.warning("   ⚠️  omegaconf not installed, skipping validation")
                logger.info("      pip install omegaconf")
            except Exception as e:
                logger.error(f"   ❌ Invalid config: {e}")
                all_valid = False
        
        self.results['config_valid'] = all_valid
        return all_valid
    
    def print_summary(self):
        """Print test summary and next steps"""
        logger.info("\n" + "="*80)
        logger.info("📊 PRE-FLIGHT CHECK SUMMARY")
        logger.info("="*80)
        
        # Authentication
        if self.results['huggingface_auth']:
            logger.info("✅ Hugging Face: Authenticated")
        else:
            logger.info("❌ Hugging Face: Not authenticated")
        
        # Models
        for model_key in MODELS.keys():
            status = self.results['models_downloaded'].get(model_key, False)
            icon = "✅" if status else "❌"
            logger.info(f"{icon} {model_key.capitalize()} Model: {'Downloaded' if status else 'Not downloaded'}")
        
        # Test data
        if self.results['test_data_downloaded']:
            logger.info("✅ Test Data: Downloaded")
        else:
            logger.info("❌ Test Data: Not downloaded")
        
        # Config
        if self.results['config_valid']:
            logger.info("✅ Config Files: Valid")
        else:
            logger.info("❌ Config Files: Invalid")
        
        # Overall status
        all_passed = (
            self.results['huggingface_auth'] and
            all(self.results['models_downloaded'].values()) and
            self.results['test_data_downloaded'] and
            self.results['config_valid']
        )
        
        logger.info("\n" + "="*80)
        if all_passed:
            logger.info("🎉 ALL PRE-FLIGHT CHECKS PASSED!")
            logger.info("="*80)
            logger.info("\n✅ Ready for RunPod deployment!")
            logger.info("\nNext steps:")
            logger.info("  1. Deploy RunPod pod (see RUNPOD_DEPLOYMENT_CHECKLIST.md)")
            logger.info("  2. Clone repo and setup environment")
            logger.info("  3. Run on RunPod:")
            logger.info("     python tests/test_local_models.py --download --test")
            logger.info("  4. If tests look good, run full training")
        else:
            logger.info("⚠️  SOME CHECKS FAILED")
            logger.info("="*80)
            logger.info("\n❌ Fix the issues above before deploying to RunPod")
        
        logger.info("\n" + "="*80)
        logger.info("ℹ️  NOTE: Transcription testing requires Linux + NeMo")
        logger.info("="*80)
        logger.info("\nWindows cannot run NeMo, so actual ASR testing must be done on RunPod.")
        logger.info("These pre-flight checks verify everything else is ready:")
        logger.info("  ✅ Authentication works")
        logger.info("  ✅ Models can be downloaded")
        logger.info("  ✅ Data can be fetched from Railway")
        logger.info("  ✅ Config files are valid")
        logger.info("\nOnce on RunPod, use:")
        logger.info("  python tests/test_local_models.py --download --test")
        logger.info("This will test actual transcription with both models.")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-flight checks before RunPod deployment (Windows-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip model download (assume already downloaded)'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['marathi', 'konkani'],
        default=['marathi', 'konkani'],
        help='Models to check/download (default: both)'
    )
    
    parser.add_argument(
        '--n-samples',
        type=int,
        default=3,
        help='Number of test samples to download (default: 3)'
    )
    
    args = parser.parse_args()
    
    checker = PreflightChecker()
    
    # Run checks
    logger.info("="*80)
    logger.info("🚀 RUNNING PRE-FLIGHT CHECKS (Windows Compatible)")
    logger.info("="*80)
    
    # 1. Check authentication
    checker.check_huggingface_auth()
    
    # 2. Check/download models
    if not args.skip_download:
        logger.info("\n" + "="*80)
        logger.info("📦 Checking/Downloading Models")
        logger.info("="*80)
        
        for model_key in args.models:
            checker.check_model_download(model_key)
    else:
        logger.info("\n⏭️  Skipping model download (--skip-download)")
        # Just check if models exist
        for model_key in args.models:
            model_path = MODELS[model_key]['path']
            if os.path.exists(model_path):
                nemo_files = list(Path(model_path).glob("*.nemo"))
                checker.results['models_downloaded'][model_key] = bool(nemo_files)
    
    # 3. Download test samples
    checker.check_test_data_download(args.n_samples)
    
    # 4. Validate configs
    checker.check_config_files()
    
    # Print summary
    checker.print_summary()


if __name__ == "__main__":
    main()
