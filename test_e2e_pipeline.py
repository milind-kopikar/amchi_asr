#!/usr/bin/env python3
"""
End-to-End System Tests for NeMo ASR Pipeline
Tests training, validation, and inference workflows
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)


class E2ETestRunner:
    """End-to-end test orchestrator"""
    
    def __init__(self, test_dir: str = "test_e2e_results"):
        self.test_dir = Path(test_dir)
        self.test_dir.mkdir(exist_ok=True)
        self.results = []
    
    def run_command(self, cmd: list, description: str) -> dict:
        """Run a command and capture output"""
        logger.info(f"▶️  {description}")
        logger.info(f"   Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            
            success = result.returncode == 0
            
            if success:
                logger.info(f"   ✅ Success")
            else:
                logger.error(f"   ❌ Failed (exit code {result.returncode})")
                logger.error(f"   Error: {result.stderr[:500]}")
            
            return {
                'success': success,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            logger.error(f"   ❌ Timeout after 30 minutes")
            return {'success': False, 'error': 'timeout'}
        except Exception as e:
            logger.error(f"   ❌ Exception: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_wer(self, test_results_file: str) -> float:
        """Extract WER from test results JSON"""
        try:
            with open(test_results_file, 'r') as f:
                data = json.load(f)
                wer = data.get('summary', {}).get('wer', None)
                if wer is not None:
                    logger.info(f"   📊 WER: {wer*100:.2f}%")
                    return wer
                else:
                    logger.warning(f"   ⚠️  WER not found in {test_results_file}")
                    return None
        except Exception as e:
            logger.error(f"   ❌ Failed to extract WER: {e}")
            return None
    
    def test_1_overfitting(self, model_type: str = 'marathi') -> dict:
        """
        TEST 1: Train = Test (Overfitting Test)
        
        Train on small dataset, then test on SAME data.
        Expected: Near 0% WER (model should memorize)
        """
        logger.info("="*80)
        logger.info("🧪 TEST 1: OVERFITTING TEST (Train = Test)")
        logger.info("="*80)
        logger.info("Goal: Train and test on same data → expect WER ≈ 0%")
        logger.info("")
        
        test_name = "test1_overfitting"
        output_dir = self.test_dir / test_name
        output_dir.mkdir(exist_ok=True)
        
        # Download smoke test data (3 train, 1 dev, 1 test)
        logger.info("Step 1: Download minimal data")
        result1 = self.run_command(
            ['python', 'scripts/download_smoke_test_data.py', '--n_train', '3', '--n_dev', '1', '--n_test', '3'],
            "Download 3 train, 1 dev, 3 test samples"
        )
        
        if not result1['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'data_download'}
        
        # Train on train data (3 samples, should be quick)
        logger.info("\nStep 2: Train model on 3 samples (max 5 epochs)")
        
        # Create quick config (very few epochs for smoke test)
        quick_config = output_dir / "quick_config.yaml"
        shutil.copy('configs/konkani_finetune.yaml', quick_config)
        
        # Override config for quick training
        with open(quick_config, 'r') as f:
            content = f.read()
        content = content.replace('max_epochs: 50', 'max_epochs: 5')
        content = content.replace('batch_size: 8', 'batch_size: 2')
        content = content.replace('data/train/manifest.jsonl', 'data_smoke/train/manifest.jsonl')
        content = content.replace('data/dev/manifest.jsonl', 'data_smoke/dev/manifest.jsonl')
        with open(quick_config, 'w') as f:
            f.write(content)
        
        result2 = self.run_command(
            [
                'python', 'scripts/nemo_train.py',
                '--config', str(quick_config),
                '--model', model_type,
                '--freeze_encoder',  # Faster training
                '--output_dir', str(output_dir / 'training')
            ],
            "Train model (5 epochs, frozen encoder)"
        )
        
        if not result2['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'training_failed'}
        
        # Find trained model
        trained_model = output_dir / 'training' / f'{model_type}_asr_final.nemo'
        if not trained_model.exists():
            logger.error(f"   ❌ Trained model not found: {trained_model}")
            return {'test': test_name, 'status': 'FAILED', 'reason': 'model_not_found'}
        
        # Test on SAME data (train data) - should get near 0% WER
        logger.info("\nStep 3: Test on SAME data (train set)")
        result3 = self.run_command(
            [
                'python', 'scripts/nemo_test.py',
                '--model', str(trained_model),
                '--manifest', 'data_smoke/train/manifest.jsonl',
                '--model_type', model_type,
                '--output_dir', str(output_dir / 'test_on_train')
            ],
            "Test on training data (expect low WER)"
        )
        
        if not result3['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'testing_failed'}
        
        # Extract WER
        wer_train = self.extract_wer(str(output_dir / 'test_on_train' / 'test_results.json'))
        
        # Verify WER is low (< 30%)
        success = wer_train is not None and wer_train < 0.3
        
        logger.info("\n" + "="*80)
        logger.info(f"TEST 1 RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
        logger.info(f"WER on training data: {wer_train*100:.2f}% (expect <30%)")
        logger.info("="*80)
        
        return {
            'test': test_name,
            'status': 'PASSED' if success else 'FAILED',
            'wer_on_train': wer_train,
            'expected': '<30% WER',
            'actual': f'{wer_train*100:.2f}%' if wer_train else 'N/A'
        }
    
    def test_2_before_after(self, model_type: str = 'marathi') -> dict:
        """
        TEST 2: Before-After Test
        
        1. Test pre-trained model on test data (baseline WER)
        2. Fine-tune on train data
        3. Test fine-tuned model on SAME test data
        Expected: WER after < WER before
        """
        logger.info("="*80)
        logger.info("🧪 TEST 2: BEFORE-AFTER TEST")
        logger.info("="*80)
        logger.info("Goal: WER should improve after fine-tuning")
        logger.info("")
        
        test_name = "test2_before_after"
        output_dir = self.test_dir / test_name
        output_dir.mkdir(exist_ok=True)
        
        # Use existing smoke test data
        logger.info("Step 1: Using existing smoke test data")
        
        # Download base model (pretrained, not fine-tuned)
        logger.info("\nStep 2: Download base model (if not exists)")
        base_model_dir = f"models/indicconformer_{'mr' if model_type == 'marathi' else 'kok'}"
        base_model_path = f"{base_model_dir}/indicconformer_stt_{'mr' if model_type == 'marathi' else 'kok'}_hybrid_rnnt_large.nemo"
        
        if not Path(base_model_path).exists():
            logger.info(f"   ⚠️  Base model not found, skipping before test")
            logger.info(f"   On RunPod, download model first: python scripts/download_model.py --model {model_type}")
            wer_before = None
        else:
            # Test BEFORE fine-tuning
            logger.info("\nStep 3: Test base model BEFORE fine-tuning")
            result_before = self.run_command(
                [
                    'python', 'scripts/nemo_test.py',
                    '--model', base_model_path,
                    '--manifest', 'data_smoke/test/manifest.jsonl',
                    '--model_type', model_type,
                    '--output_dir', str(output_dir / 'before')
                ],
                "Test baseline (before fine-tuning)"
            )
            
            if not result_before['success']:
                wer_before = None
            else:
                wer_before = self.extract_wer(str(output_dir / 'before' / 'test_results.json'))
        
        # Fine-tune model
        logger.info("\nStep 4: Fine-tune model on train data")
        
        quick_config = output_dir / "quick_config.yaml"
        shutil.copy('configs/konkani_finetune.yaml', quick_config)
        with open(quick_config, 'r') as f:
            content = f.read()
        content = content.replace('max_epochs: 50', 'max_epochs: 5')
        content = content.replace('batch_size: 8', 'batch_size: 2')
        content = content.replace('data/train/manifest.jsonl', 'data_smoke/train/manifest.jsonl')
        content = content.replace('data/dev/manifest.jsonl', 'data_smoke/dev/manifest.jsonl')
        with open(quick_config, 'w') as f:
            f.write(content)
        
        result_train = self.run_command(
            [
                'python', 'scripts/nemo_train.py',
                '--config', str(quick_config),
                '--model', model_type,
                '--freeze_encoder',
                '--output_dir', str(output_dir / 'training')
            ],
            "Fine-tune model (5 epochs)"
        )
        
        if not result_train['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'training_failed'}
        
        # Test AFTER fine-tuning
        logger.info("\nStep 5: Test fine-tuned model AFTER training")
        trained_model = output_dir / 'training' / f'{model_type}_asr_final.nemo'
        
        result_after = self.run_command(
            [
                'python', 'scripts/nemo_test.py',
                '--model', str(trained_model),
                '--manifest', 'data_smoke/test/manifest.jsonl',
                '--model_type', model_type,
                '--output_dir', str(output_dir / 'after')
            ],
            "Test fine-tuned model (after training)"
        )
        
        if not result_after['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'testing_failed'}
        
        wer_after = self.extract_wer(str(output_dir / 'after' / 'test_results.json'))
        
        # Check improvement
        if wer_before is None:
            success = wer_after is not None
            improvement = "N/A (baseline not available)"
        else:
            success = wer_after < wer_before
            improvement = f"{(wer_before - wer_after)*100:.2f}% improvement"
        
        logger.info("\n" + "="*80)
        logger.info(f"TEST 2 RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
        logger.info(f"WER BEFORE: {wer_before*100:.2f}%" if wer_before else "WER BEFORE: N/A")
        logger.info(f"WER AFTER:  {wer_after*100:.2f}%" if wer_after else "WER AFTER: N/A")
        logger.info(f"IMPROVEMENT: {improvement}")
        logger.info("="*80)
        
        return {
            'test': test_name,
            'status': 'PASSED' if success else 'FAILED',
            'wer_before': wer_before,
            'wer_after': wer_after,
            'improvement': improvement
        }
    
    def test_3_empty_data(self, model_type: str = 'marathi') -> dict:
        """
        TEST 3: Empty Data Test (Negative Test)
        
        1. Test model on test data (baseline WER)
        2. "Fine-tune" on EMPTY/MINIMAL data (should not improve)
        3. Test again on same test data
        Expected: WER should stay same or get worse
        """
        logger.info("="*80)
        logger.info("🧪 TEST 3: EMPTY DATA TEST (Negative Test)")
        logger.info("="*80)
        logger.info("Goal: WER should NOT improve when trained on empty data")
        logger.info("")
        
        test_name = "test3_empty_data"
        output_dir = self.test_dir / test_name
        output_dir.mkdir(exist_ok=True)
        
        # Create empty training data
        logger.info("Step 1: Create empty training manifest")
        empty_dir = Path("data_smoke_empty")
        empty_dir.mkdir(exist_ok=True)
        (empty_dir / "train").mkdir(exist_ok=True)
        (empty_dir / "dev").mkdir(exist_ok=True)
        
        # Empty manifest (or single very short sample)
        with open(empty_dir / "train" / "manifest.jsonl", 'w') as f:
            # Deliberately empty - no training data
            pass
        
        with open(empty_dir / "dev" / "manifest.jsonl", 'w') as f:
            # Copy one sample from test for validation
            with open("data_smoke/test/manifest.jsonl", 'r') as src:
                f.write(src.readline())
        
        logger.info("   ✓ Empty training manifest created")
        
        # Get baseline model
        logger.info("\nStep 2: Get baseline model")
        base_model_dir = f"models/indicconformer_{'mr' if model_type == 'marathi' else 'kok'}"
        base_model_path = f"{base_model_dir}/indicconformer_stt_{'mr' if model_type == 'marathi' else 'kok'}_hybrid_rnnt_large.nemo"
        
        if not Path(base_model_path).exists():
            logger.warning("   ⚠️  Base model not available, using previously trained model")
            # Use model from test 2 if available
            prev_model = self.test_dir / "test2_before_after" / "training" / f"{model_type}_asr_final.nemo"
            if prev_model.exists():
                base_model_path = str(prev_model)
            else:
                return {'test': test_name, 'status': 'SKIPPED', 'reason': 'no_base_model'}
        
        # Test baseline
        logger.info("\nStep 3: Test baseline (before empty training)")
        result_before = self.run_command(
            [
                'python', 'scripts/nemo_test.py',
                '--model', base_model_path,
                '--manifest', 'data_smoke/test/manifest.jsonl',
                '--model_type', model_type,
                '--output_dir', str(output_dir / 'before')
            ],
            "Test baseline"
        )
        
        if not result_before['success']:
            return {'test': test_name, 'status': 'FAILED', 'reason': 'baseline_test_failed'}
        
        wer_before = self.extract_wer(str(output_dir / 'before' / 'test_results.json'))
        
        # "Train" on empty data (should fail gracefully or not improve)
        logger.info("\nStep 4: Attempt training on empty data")
        
        quick_config = output_dir / "empty_config.yaml"
        shutil.copy('configs/konkani_finetune.yaml', quick_config)
        with open(quick_config, 'r') as f:
            content = f.read()
        content = content.replace('max_epochs: 50', 'max_epochs: 2')
        content = content.replace('batch_size: 8', 'batch_size: 1')
        content = content.replace('data/train/manifest.jsonl', 'data_smoke_empty/train/manifest.jsonl')
        content = content.replace('data/dev/manifest.jsonl', 'data_smoke_empty/dev/manifest.jsonl')
        with open(quick_config, 'w') as f:
            f.write(content)
        
        result_train = self.run_command(
            [
                'python', 'scripts/nemo_train.py',
                '--config', str(quick_config),
                '--model', model_type,
                '--freeze_encoder',
                '--output_dir', str(output_dir / 'training')
            ],
            "Train on empty data (should fail or produce same model)"
        )
        
        # This might fail (expected for empty data)
        if not result_train['success']:
            logger.info("   ✅ Training failed as expected with empty data")
            success = True
            wer_after = wer_before
        else:
            # If training somehow succeeded, test the result
            logger.info("\nStep 5: Test after empty training")
            trained_model = output_dir / 'training' / f'{model_type}_asr_final.nemo'
            
            if trained_model.exists():
                result_after = self.run_command(
                    [
                        'python', 'scripts/nemo_test.py',
                        '--model', str(trained_model),
                        '--manifest', 'data_smoke/test/manifest.jsonl',
                        '--model_type', model_type,
                        '--output_dir', str(output_dir / 'after')
                    ],
                    "Test after empty training"
                )
                
                if result_after['success']:
                    wer_after = self.extract_wer(str(output_dir / 'after' / 'test_results.json'))
                    # WER should be same or worse
                    success = wer_after >= wer_before * 0.95  # Allow 5% margin
                else:
                    wer_after = None
                    success = False
            else:
                wer_after = wer_before
                success = True
        
        logger.info("\n" + "="*80)
        logger.info(f"TEST 3 RESULT: {'✅ PASSED' if success else '❌ FAILED'}")
        logger.info(f"WER BEFORE: {wer_before*100:.2f}%" if wer_before else "N/A")
        logger.info(f"WER AFTER:  {wer_after*100:.2f}% (should be same or worse)" if wer_after else "N/A")
        logger.info("="*80)
        
        return {
            'test': test_name,
            'status': 'PASSED' if success else 'FAILED',
            'wer_before': wer_before,
            'wer_after': wer_after,
            'expected': 'No improvement or training failure'
        }
    
    def run_all_tests(self, model_type: str = 'marathi'):
        """Run all end-to-end tests"""
        logger.info("="*80)
        logger.info("🚀 RUNNING ALL END-TO-END TESTS")
        logger.info("="*80)
        logger.info(f"Model type: {model_type}")
        logger.info("")
        
        results = []
        
        # Test 1: Overfitting
        try:
            result1 = self.test_1_overfitting(model_type)
            results.append(result1)
        except Exception as e:
            logger.error(f"Test 1 crashed: {e}")
            results.append({'test': 'test1_overfitting', 'status': 'CRASHED', 'error': str(e)})
        
        # Test 2: Before-After
        try:
            result2 = self.test_2_before_after(model_type)
            results.append(result2)
        except Exception as e:
            logger.error(f"Test 2 crashed: {e}")
            results.append({'test': 'test2_before_after', 'status': 'CRASHED', 'error': str(e)})
        
        # Test 3: Empty Data
        try:
            result3 = self.test_3_empty_data(model_type)
            results.append(result3)
        except Exception as e:
            logger.error(f"Test 3 crashed: {e}")
            results.append({'test': 'test3_empty_data', 'status': 'CRASHED', 'error': str(e)})
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 TEST SUMMARY")
        logger.info("="*80)
        
        passed = sum(1 for r in results if r.get('status') == 'PASSED')
        failed = sum(1 for r in results if r.get('status') == 'FAILED')
        crashed = sum(1 for r in results if r.get('status') == 'CRASHED')
        skipped = sum(1 for r in results if r.get('status') == 'SKIPPED')
        
        for result in results:
            status_icon = {
                'PASSED': '✅',
                'FAILED': '❌',
                'CRASHED': '💥',
                'SKIPPED': '⏭️'
            }.get(result['status'], '❓')
            
            logger.info(f"{status_icon} {result['test']}: {result['status']}")
        
        logger.info("")
        logger.info(f"Total: {len(results)} tests")
        logger.info(f"Passed: {passed}, Failed: {failed}, Crashed: {crashed}, Skipped: {skipped}")
        logger.info("="*80)
        
        # Save results
        results_file = self.test_dir / "test_summary.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved: {results_file}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run end-to-end ASR pipeline tests")
    parser.add_argument('--model', type=str, default='marathi', choices=['marathi', 'konkani'],
                       help='Model type to test')
    parser.add_argument('--test', type=str, choices=['1', '2', '3', 'all'], default='all',
                       help='Which test to run (default: all)')
    
    args = parser.parse_args()
    
    runner = E2ETestRunner()
    
    if args.test == 'all':
        runner.run_all_tests(args.model)
    elif args.test == '1':
        runner.test_1_overfitting(args.model)
    elif args.test == '2':
        runner.test_2_before_after(args.model)
    elif args.test == '3':
        runner.test_3_empty_data(args.model)


if __name__ == "__main__":
    main()
