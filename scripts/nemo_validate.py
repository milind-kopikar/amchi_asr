#!/usr/bin/env python3
"""
NeMo ASR Validation Module
Compute WER on validation/test sets
"""

import sys
import os
import platform
import argparse
import logging
from pathlib import Path

# Only apply Windows patch if running on Windows
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(__file__))
    import windows_patch

from omegaconf import OmegaConf
import pytorch_lightning as pl

import nemo
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.metrics.wer import WER

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRValidator:
    """Modular ASR Validation Manager"""
    
    SUPPORTED_MODELS = {
        'marathi': nemo_asr.models.EncDecHybridRNNTCTCBPEModel,
        'konkani': nemo_asr.models.EncDecHybridRNNTCTCBPEModel
    }
    
    def __init__(self, model_path: str, model_type: str = 'marathi'):
        """
        Initialize validator
        
        Args:
            model_path: Path to .nemo model file or checkpoint
            model_type: Model type ('marathi' or 'konkani')
        """
        logger.info("="*80)
        logger.info("🔍 Initializing NeMo ASR Validator")
        logger.info("="*80)
        
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model type '{model_type}' not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        
    def load_model(self) -> None:
        """Load model from file"""
        logger.info(f"📂 Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            ModelClass = self.SUPPORTED_MODELS[self.model_type]
            
            # Check if it's a .nemo file or checkpoint
            if self.model_path.endswith('.nemo'):
                logger.info("🔄 Loading .nemo file...")
                self.model = ModelClass.restore_from(self.model_path)
            elif self.model_path.endswith('.ckpt'):
                logger.info("🔄 Loading from checkpoint...")
                self.model = ModelClass.load_from_checkpoint(self.model_path)
            else:
                raise ValueError(f"Unsupported model file format: {self.model_path}")
            
            # Set to eval mode
            self.model.eval()
            self.model.freeze()
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"📊 Model type: {self.model_type}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def validate(self, manifest_path: str, batch_size: int = 8, num_workers: int = 4) -> dict:
        """
        Run validation on a manifest file
        
        Args:
            manifest_path: Path to validation manifest
            batch_size: Batch size for inference
            num_workers: Number of data loader workers
            
        Returns:
            Dictionary with validation metrics
        """
        logger.info("="*80)
        logger.info("🎯 Running Validation")
        logger.info("="*80)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"📄 Manifest: {manifest_path}")
        
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        try:
            # Setup validation data
            val_config = {
                'manifest_filepath': manifest_path,
                'sample_rate': 16000,
                'batch_size': batch_size,
                'shuffle': False,
                'num_workers': num_workers,
                'pin_memory': True,
                'max_duration': 16.7,
                'min_duration': 0.1,
                'trim_silence': True
            }
            
            self.model.setup_test_data(OmegaConf.create(val_config))
            logger.info(f"✓ Validation data configured (batch_size={batch_size})")
            
            # Create trainer for validation
            trainer = pl.Trainer(
                accelerator='gpu' if platform.system() != 'Windows' else 'cpu',
                devices=1,
                logger=False,
                enable_checkpointing=False
            )
            
            # Run validation
            logger.info("🏃 Running inference...")
            results = trainer.test(self.model, verbose=False)
            
            # Extract metrics
            metrics = results[0] if results else {}
            
            logger.info("="*80)
            logger.info("📊 Validation Results")
            logger.info("="*80)
            
            for key, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"   {key}: {value:.4f}")
                else:
                    logger.info(f"   {key}: {value}")
            
            # Extract WER specifically
            wer = metrics.get('test_wer', metrics.get('val_wer', None))
            if wer is not None:
                logger.info("")
                logger.info(f"🎯 Word Error Rate (WER): {wer*100:.2f}%")
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
    
    def transcribe_files(self, audio_files: list, output_file: str = None) -> list:
        """
        Transcribe audio files
        
        Args:
            audio_files: List of audio file paths
            output_file: Optional file to save transcriptions
            
        Returns:
            List of transcriptions
        """
        logger.info("="*80)
        logger.info("🎤 Transcribing Audio Files")
        logger.info("="*80)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        logger.info(f"📁 Files to transcribe: {len(audio_files)}")
        
        try:
            # Run transcription
            transcriptions = self.model.transcribe(audio_files, batch_size=4)
            
            # Display and save
            logger.info("="*80)
            logger.info("📝 Transcriptions")
            logger.info("="*80)
            
            results = []
            for audio_path, text in zip(audio_files, transcriptions):
                logger.info(f"🎵 {os.path.basename(audio_path)}")
                logger.info(f"   → {text}")
                results.append({'file': audio_path, 'transcription': text})
            
            # Save to file if requested
            if output_file:
                import json
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"💾 Saved transcriptions to: {output_file}")
            
            return transcriptions
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Validate NeMo ASR model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate on dev set
  python nemo_validate.py --model results/marathi_asr_final.nemo --manifest ../data/dev/manifest.jsonl
  
  # Validate checkpoint
  python nemo_validate.py --model results/checkpoints/last.ckpt --manifest ../data/dev/manifest.jsonl --model_type marathi
  
  # Transcribe audio files
  python nemo_validate.py --model results/konkani_asr_final.nemo --audio file1.wav file2.wav --model_type konkani
"""
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to .nemo model file or .ckpt checkpoint"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="marathi",
        choices=['marathi', 'konkani'],
        help="Model type (default: marathi)"
    )
    
    parser.add_argument(
        "--manifest",
        type=str,
        help="Path to validation manifest file"
    )
    
    parser.add_argument(
        "--audio",
        type=str,
        nargs='+',
        help="Audio files to transcribe"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for validation (default: 8)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for transcriptions (JSON)"
    )
    
    args = parser.parse_args()
    
    if not args.manifest and not args.audio:
        parser.error("Must provide either --manifest or --audio")
    
    try:
        # Create validator
        validator = ASRValidator(
            model_path=args.model,
            model_type=args.model_type
        )
        
        # Load model
        validator.load_model()
        
        # Run validation or transcription
        if args.manifest:
            metrics = validator.validate(
                manifest_path=args.manifest,
                batch_size=args.batch_size
            )
            
        if args.audio:
            transcriptions = validator.transcribe_files(
                audio_files=args.audio,
                output_file=args.output
            )
        
        logger.info("="*80)
        logger.info("✅ Done!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
