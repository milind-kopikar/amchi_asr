import argparse
import torch
import nemo.collections.asr as nemo_asr
import logging
import os
from omegaconf import OmegaConf, DictConfig

def main():
    parser = argparse.ArgumentParser(description="Single-sample inference smoke test")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--audio", required=True, help="Path to audio .wav file")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("smoke_infer")

    if not os.path.exists(args.checkpoint):
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        exit(1)

    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    try:
        # Load checkpoint with weights_only=False to allow OmegaConf
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        exit(1)

    if 'hyper_parameters' not in ckpt or 'cfg' not in ckpt['hyper_parameters']:
        logger.error("Checkpoint does not contain 'hyper_parameters.cfg'")
        exit(1)

    cfg = ckpt['hyper_parameters']['cfg']
    
    # Ensure it's a DictConfig
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    # Patch loss_name if needed
    if 'loss' in cfg:
        if cfg.loss.get('loss_name') == 'ctc':
             logger.info("Patching loss_name from 'ctc' to 'default' for initialization...")
             cfg.loss.loss_name = 'default'

    # Remove data loader configs to avoid path errors during init
    OmegaConf.set_struct(cfg, False)
    
    removed_configs = {}
    for key in ['train_ds', 'validation_ds', 'test_ds']:
        if key in cfg:
            logger.info(f"Temporarily removing {key} from config...")
            removed_configs[key] = cfg[key]
            del cfg[key]
            
    OmegaConf.set_struct(cfg, True)

    logger.info("Instantiating model shell...")
    try:
        # Instantiate the model shell using the config from the checkpoint
        model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel(cfg=cfg)
    except Exception as e:
        logger.error(f"Failed to instantiate model shell: {e}")
        # Try to print what keys are missing if possible
        logger.error(f"Config keys: {cfg.keys()}")
        exit(1)
        
    # Restore validation_ds/test_ds as empty dicts for transcribe
    # We don't restore the original because it has bad paths
    OmegaConf.set_struct(model.cfg, False)
    if 'validation_ds' in removed_configs:
        logger.info("Restoring empty validation_ds...")
        model.cfg.validation_ds = OmegaConf.create({})
    if 'test_ds' in removed_configs:
        logger.info("Restoring empty test_ds...")
        model.cfg.test_ds = OmegaConf.create({})
    OmegaConf.set_struct(model.cfg, True)

    logger.info("Injecting weights...")

    logger.info("Injecting weights...")
    try:
        # strict=False is KEY: it ignores the mismatch between the Hybrid Shell and our CTC-only checkpoint
        model.load_state_dict(ckpt['state_dict'], strict=False)
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        exit(1)

    model.to(args.device)
    model.eval()

    # SWITCH TO CTC (Crucial!)
    logger.info("Switching inference strategy to CTC...")
    try:
        model.change_decoding_strategy(decoder_type='ctc')
    except Exception as e:
        logger.warning(f"Could not switch decoding strategy: {e}")
        # Fallback: try setting it manually if the method fails or doesn't exist
        model.cfg.decoder_type = 'ctc'

    logger.info(f"Transcribing {args.audio}...")
    
    try:
        files = [args.audio]
        # EncDecHybridRNNTCTCBPEModel.transcribe uses 'audio' argument, not 'paths2audio_files'
        predictions = model.transcribe(audio=files)
        
        # Handle different return types (list of strings or list of Hypothesis)
        pred_text = ""
        if isinstance(predictions, list):
            if len(predictions) > 0:
                p = predictions[0]
                if isinstance(p, str):
                    pred_text = p
                else:
                    # Hypothesis object
                    pred_text = p.text
        
        logger.info(f"Prediction: {pred_text}")
        
        if not pred_text:
            logger.warning("Prediction was empty!")
        
        print(f"SMOKE_INFERENCE_RESULT: {pred_text}")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
