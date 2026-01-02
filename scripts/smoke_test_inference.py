import argparse
import torch
import nemo.collections.asr as nemo_asr
import logging
import os
import sys

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
        sys.exit(1)
    
    if not os.path.exists(args.audio):
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    logger.info(f"Loading model from {args.checkpoint}...")
    
    try:
        # Load the model
        # We use EncDecHybridRNNTCTCBPEModel as that's what we are training
        # NOTE: strict=False is crucial because the checkpoint might contain CTC-specific loss configs
        # that the base class restore doesn't expect if it defaults to RNNT loss.
        try:
            model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
                args.checkpoint, 
                map_location=args.device
            )
        except Exception as e:
            if "loss_name" in str(e) or "unexpected key" in str(e):
                logger.warning(f"Standard load failed ({e}), retrying with strict=False...")
                model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.load_from_checkpoint(
                    args.checkpoint, 
                    map_location=args.device,
                    strict=False
                )
            else:
                raise e

        model.eval()
        model.to(args.device)
        
        logger.info(f"Transcribing {args.audio}...")
        files = [args.audio]
        
        # transcribe returns a list of strings or Hypothesis objects
        hypotheses = model.transcribe(paths2audio_files=files, batch_size=1)
        
        text = hypotheses[0]
        # Handle different return types (list of texts, or list of Hypothesis)
        if isinstance(text, list): 
            text = text[0]
        if hasattr(text, 'text'): 
            text = text.text
            
        logger.info(f"Transcription: {text}")
        
        # Simple check for Devanagari
        if any(0x0900 <= ord(c) <= 0x097F for c in text):
            logger.info("✅ Output contains Devanagari characters.")
        else:
            logger.warning("⚠️ Warning: Output does not contain Devanagari characters.")
            
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
