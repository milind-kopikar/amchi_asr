#!/usr/bin/env python3
"""
Local Model Testing - Test ASR models before RunPod deployment

Tests:
1. Download both Marathi and Konkani ASR models
2. Transcribe 3 test audio samples with each model
3. Calculate WER for each model
4. Generate comparison report

Usage:
    # Download models and test
    python tests/test_local_models.py --download --test
    
    # Just test (models already downloaded)
    python tests/test_local_models.py --test
    
    # Just download models
    python tests/test_local_models.py --download
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Test configuration
MODELS = {
    'marathi': {
        'name': 'ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large',
        'path': 'models/indicconformer_mr',
        'language_id': 'mr'
    },
    'konkani': {
        'name': 'ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large',
        'path': 'models/indicconformer_kok',
        'language_id': 'kok'
    }
}

class LocalModelTester:
    """Test ASR models locally before RunPod deployment"""
    
    def __init__(self, output_dir: str = "test_local_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if NeMo is available
        self.nemo_available = self._check_nemo()
        
    def _check_nemo(self) -> bool:
        """Check if NeMo is available (Linux only)"""
        try:
            import nemo
            import nemo.collections.asr as nemo_asr
            logger.info("✅ NeMo is available")
            return True
        except ImportError:
            logger.warning("⚠️  NeMo not available (Windows or not installed)")
            logger.info("💡 This script will work on RunPod/Linux with NeMo installed")
            return False
    
    def download_models(self, models_to_download: List[str] = None) -> Dict[str, bool]:
        """
        Download ASR models using download_model.py
        
        Args:
            models_to_download: List of model keys ('marathi', 'konkani') or None for all
            
        Returns:
            Dict mapping model name to success status
        """
        logger.info("="*80)
        logger.info("📥 Downloading ASR Models")
        logger.info("="*80)
        
        if models_to_download is None:
            models_to_download = list(MODELS.keys())
        
        results = {}
        
        for model_key in models_to_download:
            if model_key not in MODELS:
                logger.error(f"❌ Unknown model: {model_key}")
                results[model_key] = False
                continue
            
            model_info = MODELS[model_key]
            model_path = model_info['path']
            
            # Check if already downloaded
            if os.path.exists(model_path):
                nemo_files = list(Path(model_path).glob("*.nemo"))
                if nemo_files:
                    logger.info(f"✓ {model_key.capitalize()} model already exists: {model_path}")
                    results[model_key] = True
                    continue
            
            # Download using our download script
            logger.info(f"📥 Downloading {model_key.capitalize()} model...")
            logger.info(f"   Model: {model_info['name']}")
            logger.info(f"   Destination: {model_path}")
            
            try:
                cmd = [
                    sys.executable,
                    "scripts/download_model.py",
                    "--model", model_key
                ]
                
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                logger.info(f"✅ {model_key.capitalize()} model downloaded successfully")
                results[model_key] = True
                
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to download {model_key} model")
                logger.error(f"   Error: {e.stderr}")
                results[model_key] = False
            except Exception as e:
                logger.error(f"❌ Unexpected error downloading {model_key}: {e}")
                results[model_key] = False
        
        return results
    
    def download_test_samples(self, n_samples: int = 3) -> str:
        """
        Download test samples from Railway
        
        Args:
            n_samples: Number of test samples to download
            
        Returns:
            Path to test manifest
        """
        logger.info("="*80)
        logger.info(f"📥 Downloading {n_samples} Test Samples")
        logger.info("="*80)
        
        test_dir = os.path.join(self.output_dir, "test_audio")
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # Use download_smoke_test_data.py to get test samples
            cmd = [
                sys.executable,
                "scripts/download_smoke_test_data.py",
                "--n_train", "0",
                "--n_dev", "0",
                "--n_test", str(n_samples),
                "--output_dir", test_dir,
                "--seed", "42"
            ]
            
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            test_manifest = os.path.join(test_dir, "test", "manifest.jsonl")
            
            if os.path.exists(test_manifest):
                logger.info(f"✅ Downloaded {n_samples} test samples")
                logger.info(f"   Manifest: {test_manifest}")
                return test_manifest
            else:
                logger.error("❌ Test manifest not created")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to download test samples: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return None
    
    def test_model_transcription(
        self,
        model_key: str,
        test_manifest: str
    ) -> List[Dict]:
        """
        Test transcription with one model
        
        Args:
            model_key: 'marathi' or 'konkani'
            test_manifest: Path to test manifest
            
        Returns:
            List of test results
        """
        if not self.nemo_available:
            logger.error("❌ NeMo not available - cannot test transcription")
            logger.info("💡 Run this script on RunPod/Linux to test transcription")
            return []
        
        try:
            import nemo.collections.asr as nemo_asr
            from jiwer import wer, cer
        except ImportError as e:
            logger.error(f"❌ Required library not available: {e}")
            return []
        
        logger.info(f"🧪 Testing {model_key.capitalize()} Model")
        logger.info("-" * 80)
        
        model_info = MODELS[model_key]
        model_path = model_info['path']
        
        # Find .nemo file
        nemo_files = list(Path(model_path).glob("*.nemo"))
        if not nemo_files:
            logger.error(f"❌ No .nemo file found in {model_path}")
            return []
        
        nemo_file = str(nemo_files[0])
        logger.info(f"📂 Loading model: {nemo_file}")
        
        try:
            # Load model
            model = nemo_asr.models.ASRModel.restore_from(nemo_file)
            model.freeze()
            logger.info("✅ Model loaded successfully")
            
            # Load test samples
            test_samples = []
            with open(test_manifest, 'r', encoding='utf-8') as f:
                for line in f:
                    sample = json.loads(line)
                    test_samples.append(sample)
            
            logger.info(f"📊 Testing on {len(test_samples)} samples")
            
            # Transcribe each sample
            results = []
            for i, sample in enumerate(test_samples, 1):
                audio_path = sample['audio_filepath']
                ground_truth = sample['text']
                
                logger.info(f"\n  Sample {i}/{len(test_samples)}:")
                logger.info(f"    Audio: {os.path.basename(audio_path)}")
                
                # Transcribe
                prediction = model.transcribe([audio_path])[0]
                
                # Calculate WER and CER
                sample_wer = wer(ground_truth, prediction) * 100
                sample_cer = cer(ground_truth, prediction) * 100
                
                logger.info(f"    Ground Truth: {ground_truth}")
                logger.info(f"    Prediction:   {prediction}")
                logger.info(f"    WER: {sample_wer:.2f}% | CER: {sample_cer:.2f}%")
                
                results.append({
                    'audio': os.path.basename(audio_path),
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'wer': sample_wer,
                    'cer': sample_cer
                })
            
            # Calculate average WER/CER
            avg_wer = sum(r['wer'] for r in results) / len(results)
            avg_cer = sum(r['cer'] for r in results) / len(results)
            
            logger.info(f"\n  📊 {model_key.capitalize()} Model Summary:")
            logger.info(f"    Average WER: {avg_wer:.2f}%")
            logger.info(f"    Average CER: {avg_cer:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error testing {model_key} model: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_tests(self, test_manifest: str) -> Dict:
        """
        Run tests on all models
        
        Args:
            test_manifest: Path to test manifest
            
        Returns:
            Complete test results
        """
        logger.info("="*80)
        logger.info("🧪 Running Model Tests")
        logger.info("="*80)
        
        all_results = {}
        
        for model_key in MODELS.keys():
            model_path = MODELS[model_key]['path']
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"⚠️  {model_key.capitalize()} model not found, skipping")
                continue
            
            # Test model
            results = self.test_model_transcription(model_key, test_manifest)
            all_results[model_key] = results
        
        return all_results
    
    def generate_report(self, results: Dict, output_file: str = None) -> None:
        """
        Generate comparison report
        
        Args:
            results: Test results from run_tests()
            output_file: Output file path (default: test_local_results/report.txt)
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "report.txt")
        
        logger.info("="*80)
        logger.info("📊 Generating Comparison Report")
        logger.info("="*80)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LOCAL ASR MODEL TEST REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Summary section
            f.write("SUMMARY\n")
            f.write("-"*80 + "\n\n")
            
            for model_key, model_results in results.items():
                if not model_results:
                    f.write(f"{model_key.upper()}: No results\n\n")
                    continue
                
                avg_wer = sum(r['wer'] for r in model_results) / len(model_results)
                avg_cer = sum(r['cer'] for r in model_results) / len(model_results)
                
                f.write(f"{model_key.upper()} Model:\n")
                f.write(f"  Samples tested: {len(model_results)}\n")
                f.write(f"  Average WER: {avg_wer:.2f}%\n")
                f.write(f"  Average CER: {avg_cer:.2f}%\n\n")
            
            # Detailed results
            f.write("\n" + "="*80 + "\n")
            f.write("DETAILED RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for model_key, model_results in results.items():
                if not model_results:
                    continue
                
                f.write(f"\n{model_key.upper()} MODEL\n")
                f.write("-"*80 + "\n\n")
                
                for i, result in enumerate(model_results, 1):
                    f.write(f"Sample {i}: {result['audio']}\n")
                    f.write(f"  Ground Truth: {result['ground_truth']}\n")
                    f.write(f"  Prediction:   {result['prediction']}\n")
                    f.write(f"  WER: {result['wer']:.2f}% | CER: {result['cer']:.2f}%\n\n")
        
        logger.info(f"✅ Report saved to: {output_file}")
        
        # Also save JSON version
        json_file = os.path.join(self.output_dir, "results.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ JSON results saved to: {json_file}")
        
        # Print summary to console
        logger.info("\n" + "="*80)
        logger.info("📊 COMPARISON SUMMARY")
        logger.info("="*80)
        
        for model_key, model_results in results.items():
            if not model_results:
                logger.info(f"{model_key.upper()}: No results")
                continue
            
            avg_wer = sum(r['wer'] for r in model_results) / len(model_results)
            avg_cer = sum(r['cer'] for r in model_results) / len(model_results)
            
            logger.info(f"\n{model_key.upper()} Model:")
            logger.info(f"  Average WER: {avg_wer:.2f}%")
            logger.info(f"  Average CER: {avg_cer:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Test ASR models locally before RunPod deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download models and run tests
  python tests/test_local_models.py --download --test
  
  # Just download models
  python tests/test_local_models.py --download
  
  # Just run tests (models already downloaded)
  python tests/test_local_models.py --test
  
  # Download and test specific model
  python tests/test_local_models.py --download --test --models marathi
  
  # Test with more samples
  python tests/test_local_models.py --test --n_samples 5
        """
    )
    
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download ASR models'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run transcription tests'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['marathi', 'konkani'],
        default=['marathi', 'konkani'],
        help='Models to download/test (default: both)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=3,
        help='Number of test samples (default: 3)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_local_results',
        help='Output directory for results (default: test_local_results)'
    )
    
    args = parser.parse_args()
    
    # At least one action must be specified
    if not args.download and not args.test:
        parser.print_help()
        logger.error("\n❌ Please specify --download and/or --test")
        sys.exit(1)
    
    # Create tester
    tester = LocalModelTester(output_dir=args.output_dir)
    
    # Download models if requested
    if args.download:
        download_results = tester.download_models(args.models)
        
        # Check if any downloads failed
        failed = [k for k, v in download_results.items() if not v]
        if failed:
            logger.error(f"\n❌ Failed to download models: {', '.join(failed)}")
            logger.info("\n💡 Troubleshooting:")
            logger.info("  1. Run: huggingface-cli login")
            logger.info("  2. Accept model conditions on Hugging Face:")
            for model_key in failed:
                logger.info(f"     https://huggingface.co/{MODELS[model_key]['name']}")
            
            if args.test:
                logger.warning("\n⚠️  Cannot run tests without models")
                sys.exit(1)
    
    # Run tests if requested
    if args.test:
        # Download test samples
        test_manifest = tester.download_test_samples(n_samples=args.n_samples)
        
        if not test_manifest:
            logger.error("❌ Failed to download test samples")
            sys.exit(1)
        
        # Run tests
        results = tester.run_tests(test_manifest)
        
        if not results or all(not v for v in results.values()):
            logger.error("❌ No test results generated")
            sys.exit(1)
        
        # Generate report
        tester.generate_report(results)
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL TESTS COMPLETED")
    logger.info("="*80)
    logger.info(f"\n📂 Results saved to: {args.output_dir}/")
    logger.info("\nNext steps:")
    logger.info("  1. Review test results")
    logger.info("  2. If satisfied, deploy to RunPod")
    logger.info("  3. Run smoke tests on RunPod GPU")
    logger.info("  4. If smoke tests pass, run full training")


if __name__ == "__main__":
    main()
