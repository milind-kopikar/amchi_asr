#!/usr/bin/env python3
"""
Fine-tune AI4Bharat IndicConformer Marathi ASR model for Konkani
Using NVIDIA NeMo framework
"""

import sys
import os
import platform

# Only apply Windows patch if running on Windows
if platform.system() == 'Windows':
    sys.path.insert(0, os.path.dirname(__file__))
    import windows_patch

import argparse
import logging
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# NeMo imports
import nemo
import nemo.collections.asr as nemo_asr
from nemo.utils import exp_manager
from nemo.utils.exceptions import NeMoBaseException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> DictConfig:
    """
    Load and validate configuration

    Args:
        config_path: Path to YAML configuration file

    Returns:
        OmegaConf configuration object
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config = OmegaConf.load(config_path)
    logger.info(f"Loaded configuration from {config_path}")
    return config

def setup_model(config: DictConfig):
    """
    Setup the ASR model for fine-tuning

    Args:
        config: Configuration object

    Returns:
        NeMo ASR model
    """
    logger.info("Setting up ASR model...")

    # Load base model
    model_path = config.model.nemo_model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading base model: {model_path}")
    try:
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path, strict=False)
    except Exception as e:
        # Fallback: attempt partial restore loading only matching-shape params
        import tarfile, tempfile, torch, yaml
        print(f"🔧 Partial restore from {model_path} due to: {e}")
        with tempfile.TemporaryDirectory() as td:
            with tarfile.open(model_path, 'r') as tar:
                members = {m.name: m for m in tar.getmembers()}
                if 'model_config.yaml' not in members or 'model_weights.ckpt' not in members:
                    raise RuntimeError('model_config.yaml or model_weights.ckpt missing in .nemo')
                tar.extract('model_config.yaml', path=td)
                tar.extract('model_weights.ckpt', path=td)

            config_path = os.path.join(td, 'model_config.yaml')
            ckpt_path = os.path.join(td, 'model_weights.ckpt')
            with open(config_path, 'r') as f:
                conf = yaml.safe_load(f)

            try:
                model = nemo_asr.models.ASRModel.from_config_dict(conf, trainer=None)
            except Exception:
                from nemo.collections.asr.models import ASRModel as _ASRModel
                model = _ASRModel.from_config_dict(conf, trainer=None)

            ckpt = torch.load(ckpt_path, map_location='cpu')
            state = ckpt.get('state_dict', ckpt)
            model_sd = model.state_dict()
            filtered = {}
            matched = skipped = 0
            for k, v in state.items():
                if k in model_sd and list(v.shape) == list(model_sd[k].shape):
                    filtered[k] = v
                    matched += 1
                else:
                    skipped += 1
            print(f"🔁 Matched {matched} params, skipped {skipped} params")
            model.load_state_dict(filtered, strict=False)

    # Freeze encoder layers (optional - for faster training with less data)
    if hasattr(config, 'freeze_encoder') and config.freeze_encoder:
        logger.info("Freezing encoder layers...")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Update model configuration for fine-tuning
    if hasattr(config.model, 'decoder'):
        # Update decoder vocabulary if needed
        pass

    logger.info("Model setup completed")
    return model

def setup_data_module(config: DictConfig):
    """
    Setup data module for training

    Args:
        config: Configuration object

    Returns:
        Data module
    """
    logger.info("Setting up data module...")

    # Create data module
    data_module = nemo_asr.data.AudioToBPEDataset(
        manifest_filepath=config.data.train_ds.manifest_filepath,
        tokenizer_dir=config.model.tokenizer.dir,
        tokenizer_type=config.model.tokenizer.type,
        sample_rate=config.data.train_ds.sample_rate,
        batch_size=config.data.train_ds.batch_size,
        shuffle=config.data.train_ds.shuffle,
        num_workers=config.data.train_ds.num_workers,
        pin_memory=config.data.train_ds.pin_memory,
        max_duration=config.data.train_ds.max_duration,
        min_duration=config.data.train_ds.min_duration,
        trim_silence=config.data.train_ds.trim_silence,
        load_audio=config.data.train_ds.load_audio,
        use_start_end_token=config.data.train_ds.use_start_end_token
    )

    logger.info("Data module setup completed")
    return data_module

def setup_trainer(config: DictConfig, output_dir: str):
    """
    Setup PyTorch Lightning trainer

    Args:
        config: Configuration object
        output_dir: Output directory for checkpoints

    Returns:
        PyTorch Lightning trainer
    """
    logger.info("Setting up trainer...")

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, "checkpoints"),
        filename="konkani_asr-{epoch:02d}-{val_wer:.3f}",
        monitor="val_wer",
        mode="min",
        save_top_k=5,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, "logs"),
        name="konkani_asr_finetune"
    )

    # Create trainer
    trainer = pl.Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        num_nodes=config.trainer.num_nodes,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        enable_checkpointing=config.trainer.enable_checkpointing,
        logger=tb_logger,
        log_every_n_steps=config.trainer.log_every_n_steps,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        callbacks=callbacks,
        strategy=config.trainer.strategy if hasattr(config.trainer, 'strategy') else None
    )

    logger.info("Trainer setup completed")
    return trainer

def fine_tune_model(config: DictConfig, output_dir: str):
    """
    Main fine-tuning function

    Args:
        config: Configuration object
        output_dir: Output directory
    """
    try:
        # Setup components
        model = setup_model(config)
        data_module = setup_data_module(config)
        trainer = setup_trainer(config, output_dir)

        # Setup experiment manager
        if hasattr(config, 'exp_manager'):
            exp_manager_config = OmegaConf.to_container(config.exp_manager)
            exp_manager_config['exp_dir'] = output_dir
            exp_manager(exp_manager_config, trainer)

        # Start training
        logger.info("Starting fine-tuning...")
        trainer.fit(model, data_module)

        # Print final training metrics if available
        try:
            metrics = trainer.callback_metrics
            logger.info(f"Final callback metrics: {metrics}")
            # Try to extract training loss
            if 'train_loss' in metrics:
                logger.info(f"Final training loss: {metrics['train_loss']}")
        except Exception as e:
            logger.warning(f"Could not read final metrics: {e}")


        logger.info("Fine-tuning completed successfully!")

    except NeMoBaseException as e:
        logger.error(f"NeMo error during fine-tuning: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during fine-tuning: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndicConformer for Konkani ASR")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--output_dir", default="results", help="Output directory for results")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Load configuration
        config = load_config(args.config)

        # Override output directory in config
        config.exp_manager.exp_dir = args.output_dir

        # Start fine-tuning
        fine_tune_model(config, args.output_dir)

    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()