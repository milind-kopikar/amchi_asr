#!/usr/bin/env python3
"""
NeMo ASR Testing Module
Calculate Word Error Rate (WER) on test sets with detailed error analysis
"""

import sys
import os
import platform
import argparse
import logging
import json
from pathlib import Path
from typing import List, Dict, Tuple

# Only apply Windows patch if running on Windows
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(__file__))
    import windows_patch

from omegaconf import OmegaConf
import pytorch_lightning as pl
from jiwer import wer, cer, compute_measures

import nemo
import nemo.collections.asr as nemo_asr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRTester:
    """Modular ASR Testing with WER Analysis"""
    
    SUPPORTED_MODELS = {
        'marathi': nemo_asr.models.EncDecHybridRNNTCTCBPEModel,
        'konkani': nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    }
    
    def __init__(self, model_path: str, model_type: str = 'marathi'):
        """
        Initialize tester
        
        Args:
            model_path: Path to .nemo model file or checkpoint
            model_type: Model type ('marathi' or 'konkani')
        """
        logger.info("="*80)
        logger.info("🧪 Initializing NeMo ASR Tester")
        logger.info("="*80)
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        
    def load_model(self) -> None:
        """Load model from file"""
        logger.info(f"📂 Loading model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            ModelClass = self.SUPPORTED_MODELS[self.model_type]
            
            if self.model_path.endswith('.nemo'):
                self.model = ModelClass.restore_from(self.model_path)
            elif self.model_path.endswith('.ckpt'):
                self.model = ModelClass.load_from_checkpoint(self.model_path)
            else:
                raise ValueError(f"Unsupported model format: {self.model_path}")
            
            self.model.eval()
            self.model.freeze()
            
            logger.info(f"✓ Model loaded: {self.model_type}")
            
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            raise
    
    def load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load manifest file"""
        logger.info(f"📄 Loading manifest: {manifest_path}")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠ Skipping invalid JSON on line {line_num}: {e}")
        
        logger.info(f"✓ Loaded {len(samples)} samples")
        return samples
    
    def transcribe_samples(self, samples: List[Dict], batch_size: int = 8) -> List[str]:
        """
        Transcribe audio samples
        
        Args:
            samples: List of manifest entries
            batch_size: Batch size for inference
            
        Returns:
            List of transcriptions
        """
        logger.info("="*80)
        logger.info("🎤 Running Transcription")
        logger.info("="*80)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        audio_files = [sample['audio_filepath'] for sample in samples]
        logger.info(f"🎵 Transcribing {len(audio_files)} files (batch_size={batch_size})...")
        
        try:
            transcriptions = self.model.transcribe(audio_files, batch_size=batch_size)
            logger.info(f"✓ Transcription complete")
            return transcriptions
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise
    
    def calculate_wer(
        self,
        references: List[str],
        hypotheses: List[str],
        save_errors: bool = True,
        output_dir: str = "test_results"
    ) -> Dict:
        """
        Calculate WER and detailed error analysis
        
        Args:
            references: Ground truth texts
            hypotheses: Predicted texts
            save_errors: Whether to save error analysis
            output_dir: Directory to save results
            
        Returns:
            Dictionary with metrics and error analysis
        """
        logger.info("="*80)
        logger.info("📊 Calculating WER")
        logger.info("="*80)
        
        if len(references) != len(hypotheses):
            raise ValueError(f"Mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
        
        try:
            # Calculate overall WER and CER
            overall_wer = wer(references, hypotheses)
            overall_cer = cer(references, hypotheses)
            
            # Detailed measures
            measures = compute_measures(references, hypotheses)
            
            logger.info("="*80)
            logger.info("📈 Test Results")
            logger.info("="*80)
            logger.info(f"   Total samples: {len(references)}")
            logger.info(f"   Word Error Rate (WER): {overall_wer*100:.2f}%")
            logger.info(f"   Character Error Rate (CER): {overall_cer*100:.2f}%")
            logger.info("")
            logger.info(f"   Substitutions: {measures['substitutions']}")
            logger.info(f"   Deletions: {measures['deletions']}")
            logger.info(f"   Insertions: {measures['insertions']}")
            logger.info(f"   Hits: {measures['hits']}")
            
            # Per-sample analysis
            sample_results = []
            for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
                sample_wer = wer(ref, hyp)
                sample_cer = cer(ref, hyp)
                
                sample_results.append({
                    'index': i,
                    'reference': ref,
                    'hypothesis': hyp,
                    'wer': sample_wer,
                    'cer': sample_cer,
                    'match': ref == hyp
                })
            
            # Identify worst samples
            sorted_results = sorted(sample_results, key=lambda x: x['wer'], reverse=True)
            
            logger.info("")
            logger.info("🔴 Top 10 Worst Samples (by WER):")
            logger.info("="*80)
            for i, result in enumerate(sorted_results[:10], 1):
                logger.info(f"{i}. Sample {result['index']} - WER: {result['wer']*100:.1f}%")
                logger.info(f"   REF: {result['reference'][:100]}")
                logger.info(f"   HYP: {result['hypothesis'][:100]}")
                logger.info("")
            
            # Save detailed results
            if save_errors:
                os.makedirs(output_dir, exist_ok=True)
                
                results_file = os.path.join(output_dir, "test_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'summary': {
                            'total_samples': len(references),
                            'wer': overall_wer,
                            'cer': overall_cer,
                            'substitutions': int(measures['substitutions']),
                            'deletions': int(measures['deletions']),
                            'insertions': int(measures['insertions']),
                            'hits': int(measures['hits'])
                        },
                        'samples': sample_results
                    }, f, ensure_ascii=False, indent=2)
                
                logger.info(f"💾 Detailed results saved: {results_file}")
                
                # Save worst samples separately
                worst_file = os.path.join(output_dir, "worst_samples.json")
                with open(worst_file, 'w', encoding='utf-8') as f:
                    json.dump(sorted_results[:50], f, ensure_ascii=False, indent=2)
                
                logger.info(f"💾 Worst 50 samples saved: {worst_file}")
            
            return {
                'wer': overall_wer,
                'cer': overall_cer,
                'measures': measures,
                'sample_results': sample_results
            }
            
        except Exception as e:
            logger.error(f"❌ WER calculation failed: {e}")
            raise
    
    def test(
        self,
        manifest_path: str,
        batch_size: int = 8,
        output_dir: str = "test_results"
    ) -> Dict:
        """
        Run complete test pipeline
        
        Args:
            manifest_path: Path to test manifest
            batch_size: Batch size for inference
            output_dir: Output directory for results
            
        Returns:
            Dictionary with test metrics
        """
        logger.info("="*80)
        logger.info("🧪 Running Complete Test")
        logger.info("="*80)
        
        # Load samples
        samples = self.load_manifest(manifest_path)
        
        # Transcribe
        hypotheses = self.transcribe_samples(samples, batch_size=batch_size)
        
        # Extract references
        references = [sample['text'] for sample in samples]
        
        # Calculate WER
        results = self.calculate_wer(
            references=references,
            hypotheses=hypotheses,
            save_errors=True,
            output_dir=output_dir
        )
        
        logger.info("="*80)
        logger.info("✅ Test Complete!")
        logger.info(f"📁 Results saved to: {output_dir}")
        logger.info("="*80)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test NeMo ASR model and calculate WER",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on test set
  python nemo_test.py --model results/marathi_asr_final.nemo --manifest ../data/test/manifest.jsonl
  
  # Test checkpoint with Konkani model
  python nemo_test.py --model results/checkpoints/best.ckpt --manifest ../data/test/manifest.jsonl --model_type konkani
  
  # Test with custom batch size and output directory
  python nemo_test.py --model results/konkani_asr_final.nemo --manifest ../data/test/manifest.jsonl --batch_size 16 --output test_results_run1
"""
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .nemo model file or .ckpt checkpoint"
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to test manifest file"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="marathi",
        choices=['marathi', 'konkani'],
        help="Model type (default: marathi)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference (default: 8)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create tester
        tester = ASRTester(
            model_path=args.model,
            model_type=args.model_type
        )
        
        # Load model
        tester.load_model()
        
        # Run test
        results = tester.test(
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            output_dir=args.output_dir
        )
        
        # Print summary
        print("")
        print("="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        print(f"Word Error Rate (WER): {results['wer']*100:.2f}%")
        print(f"Character Error Rate (CER): {results['cer']*100:.2f}%")
        print("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
