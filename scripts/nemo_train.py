#!/usr/bin/env python3
"""
Modular NeMo ASR Training Module
Supports AI4Bharat IndicConformer models for Konkani ASR
"""

import sys
import os
import platform
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

# Only apply Windows patch if running on Windows
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(__file__))
    import windows_patch

from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASRTrainer:
    """Modular ASR Training Manager"""
    
    # Supported model configurations
    SUPPORTED_MODELS = {
        'marathi': {
            'name': 'ai4bharat/indicconformer_stt_mr_hybrid_ctc_rnnt_large',
            'nemo_file': 'models/indicconformer_mr/indicconformer_stt_mr_hybrid_ctc_rnnt_large.nemo',
            'description': 'AI4Bharat Marathi IndicConformer (RNNT+CTC)',
            'model_class': nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        },
        'konkani': {
            'name': 'ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large',
            'nemo_file': 'models/indicconformer_kok/indicconformer_stt_kok_hybrid_rnnt_large.nemo',
            'description': 'AI4Bharat Goan Konkani IndicConformer (CTC+RNNT)',
            'model_class': nemo_asr.models.EncDecHybridRNNTCTCBPEModel
        }
    }
    
    def __init__(self, config_path: str, model_name: str = 'marathi'):
        """
        Initialize trainer
        
        Args:
            config_path: Path to YAML config file
            model_name: Model name ('marathi' or 'konkani')
        """
        logger.info("="*80)
        logger.info("🚀 Initializing NeMo ASR Trainer")
        logger.info("="*80)
        
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported. Choose from: {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_config = self.SUPPORTED_MODELS[model_name]
        
        logger.info(f"📦 Selected Model: {self.model_config['name']}")
        logger.info(f"📝 Description: {self.model_config['description']}")
        
        self.config = self._load_config(config_path)
        self.model = None
        self.trainer = None
        
    def _load_config(self, config_path: str) -> DictConfig:
        """Load and validate configuration"""
        logger.info(f"📄 Loading config from: {config_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        config = OmegaConf.load(config_path)
        
        # Override model path based on selected model
        config.model.nemo_model = self.model_config['nemo_file']
        
        # Validate required paths
        train_manifest = config.data.train_ds.manifest_filepath
        if not os.path.exists(train_manifest):
            raise FileNotFoundError(f"Train manifest not found: {train_manifest}")
        
        logger.info(f"✓ Train manifest: {train_manifest}")
        
        if hasattr(config.data, 'validation_ds'):
            val_manifest = config.data.validation_ds.manifest_filepath
            if os.path.exists(val_manifest):
                logger.info(f"✓ Validation manifest: {val_manifest}")
            else:
                logger.warning(f"⚠ Validation manifest not found: {val_manifest}")
        
        logger.info(f"✓ Config loaded successfully")
        return config
    
    def load_model(self, freeze_encoder: bool = False) -> None:
        """
        Load pre-trained model for fine-tuning
        
        Args:
            freeze_encoder: Whether to freeze encoder layers (faster training, less data needed)
        """
        logger.info("="*80)
        logger.info("🔧 Loading Pre-trained Model")
        logger.info("="*80)
        
        model_path = self.config.model.nemo_model
        logger.info(f"📂 Model path: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please download the model first using scripts/download_model.py"
            )
        
        try:
            # Restore model from .nemo file
            ModelClass = self.model_config['model_class']
            logger.info(f"🔄 Restoring {ModelClass.__name__}...")
            
            # Use map_location to ensure compatibility
            self.model = ModelClass.restore_from(
                model_path, 
                strict=False,
                map_location='cpu'  # Load to CPU first to avoid CUDA issues
            )
            
            # Optionally freeze encoder
            if freeze_encoder:
                logger.info("🔒 Freezing encoder layers...")
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                logger.info("✓ Encoder frozen (only decoder will be trained)")
            else:
                logger.info("🔓 All layers trainable (full fine-tuning)")
            
            # Update model config from YAML
            self.model.set_trainer(None)  # Clear any previous trainer
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"📊 Model details:")
            logger.info(f"   - Encoder layers: {len(list(self.model.encoder.parameters()))}")
            logger.info(f"   - Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            logger.info(f"   - Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def setup_trainer(self, output_dir: str = "results") -> None:
        """
        Setup PyTorch Lightning trainer with callbacks
        
        Args:
            output_dir: Directory for checkpoints and logs
        """
        logger.info("="*80)
        logger.info("⚙️  Setting up Trainer")
        logger.info("="*80)
        
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename=f"{self.model_name}_asr-{{epoch:02d}}-{{val_wer:.3f}}",
            monitor="val_wer",
            mode="min",
            save_top_k=3,
            save_last=True,
            verbose=True,
            every_n_epochs=1
        )
        callbacks.append(checkpoint_callback)
        logger.info(f"✓ Checkpoint callback configured (save top 3 by WER)")
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_monitor)
        logger.info(f"✓ Learning rate monitor enabled")
        
        # TensorBoard logger
        tb_logger = TensorBoardLogger(
            save_dir=os.path.join(output_dir, "logs"),
            name=f"{self.model_name}_asr_finetune"
        )
        logger.info(f"✓ TensorBoard logging to: {tb_logger.log_dir}")
        
        # Create trainer
        self.trainer = pl.Trainer(
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            max_epochs=self.config.trainer.max_epochs,
            max_steps=self.config.trainer.max_steps,
            num_nodes=self.config.trainer.num_nodes,
            accumulate_grad_batches=self.config.trainer.accumulate_grad_batches,
            enable_checkpointing=self.config.trainer.enable_checkpointing,
            logger=tb_logger,
            log_every_n_steps=self.config.trainer.log_every_n_steps,
            check_val_every_n_epoch=self.config.trainer.check_val_every_n_epoch,
            callbacks=callbacks,
            deterministic=False,
            gradient_clip_val=1.0
        )
        
        logger.info(f"✓ Trainer configured:")
        logger.info(f"   - Accelerator: {self.config.trainer.accelerator}")
        logger.info(f"   - Devices: {self.config.trainer.devices}")
        logger.info(f"   - Max epochs: {self.config.trainer.max_epochs}")
        logger.info(f"   - Batch size: {self.config.data.train_ds.batch_size}")
    
    def train(self, output_dir: str = "results") -> str:
        """
        Run training
        
        Args:
            output_dir: Directory for outputs
            
        Returns:
            Path to best checkpoint
        """
        logger.info("="*80)
        logger.info("🎯 Starting Training")
        logger.info("="*80)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if self.trainer is None:
            self.setup_trainer(output_dir)
        
        try:
            # Setup data config in model
            self.model.setup_training_data(self.config.data.train_ds)
            logger.info(f"✓ Training data configured")
            
            if hasattr(self.config.data, 'validation_ds'):
                self.model.setup_validation_data(self.config.data.validation_ds)
                logger.info(f"✓ Validation data configured")
            
            # Setup optimization (from config)
            self.model.setup_optimization(self.config.optim)
            logger.info(f"✓ Optimizer configured: {self.config.optim.name}")
            logger.info(f"   - Learning rate: {self.config.optim.lr}")
            logger.info(f"   - Weight decay: {self.config.optim.weight_decay}")
            
            # Start training
            logger.info("🏋️  Training started...")
            self.trainer.fit(self.model)
            
            logger.info("="*80)
            logger.info("✅ Training Complete!")
            logger.info("="*80)
            
            # Save final model
            final_model_path = os.path.join(output_dir, f"{self.model_name}_asr_final.nemo")
            self.model.save_to(final_model_path)
            logger.info(f"💾 Final model saved: {final_model_path}")
            
            # Get best checkpoint
            best_ckpt = self.trainer.checkpoint_callback.best_model_path
            logger.info(f"🏆 Best checkpoint: {best_ckpt}")
            logger.info(f"📊 Best WER: {self.trainer.checkpoint_callback.best_model_score:.4f}")
            
            return best_ckpt
            
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Train NeMo ASR model for Konkani",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with Marathi base model (default)
  python nemo_train.py --config ../configs/konkani_finetune.yaml
  
  # Train with Konkani base model
  python nemo_train.py --config ../configs/konkani_finetune.yaml --model konkani
  
  # Freeze encoder for faster training
  python nemo_train.py --config ../configs/konkani_finetune.yaml --freeze_encoder
  
  # Custom output directory
  python nemo_train.py --config ../configs/konkani_finetune.yaml --output_dir my_results
"""
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="marathi",
        choices=list(ASRTrainer.SUPPORTED_MODELS.keys()),
        help="Base model to use: 'marathi' or 'konkani' (default: marathi)"
    )
    
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freeze encoder layers (faster training, recommended for <1hr data)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for checkpoints and logs (default: results)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create trainer
        trainer = ASRTrainer(
            config_path=args.config,
            model_name=args.model
        )
        
        # Load model
        trainer.load_model(freeze_encoder=args.freeze_encoder)
        
        # Setup trainer
        trainer.setup_trainer(output_dir=args.output_dir)
        
        # Train
        best_checkpoint = trainer.train(output_dir=args.output_dir)
        
        logger.info("="*80)
        logger.info("🎉 All done!")
        logger.info(f"📁 Results saved to: {args.output_dir}")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
