#!/usr/bin/env python3
"""
Minimal data test for Konkani ASR fine-tuning
Tests the pipeline with just 1 minute of Konkani speech
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

def create_minimal_test_data(audio_file, transcript_text, output_dir="minimal_test"):
    """Create a minimal test dataset from a single audio file"""

    test_dir = Path(output_dir)
    test_dir.mkdir(exist_ok=True)

    # Create subdirectories
    audio_dir = test_dir / "audio"
    transcript_dir = test_dir / "transcripts"
    audio_dir.mkdir(exist_ok=True)
    transcript_dir.mkdir(exist_ok=True)

    # Copy audio file with standardized name
    audio_name = "test_001.wav"
    shutil.copy2(audio_file, audio_dir / audio_name)

    # Create transcript file
    transcript_name = "test_001.txt"
    with open(transcript_dir / transcript_name, 'w', encoding='utf-8') as f:
        f.write(transcript_text.strip())

    print(f"✅ Created minimal test data in {test_dir}")
    print(f"   Audio: {audio_dir / audio_name}")
    print(f"   Transcript: {transcript_dir / transcript_name}")

    return test_dir

def run_minimal_pipeline(test_data_dir, config_path="configs/minimal_test.yaml"):
    """Run the full pipeline with minimal data"""

    print("🚀 Running Minimal Konkani ASR Test")
    print("=" * 50)

    # Step 1: Prepare data
    print("\n📊 Step 1: Preparing minimal data...")
    cmd = f"python scripts/prepare_data.py --audio_dir {test_data_dir}/audio --transcript_dir {test_data_dir}/transcripts --output_dir minimal_test/data --val_split 0.3 --test_split 0.3"
    os.system(cmd)

    # Step 2: Create minimal config
    print("\n⚙️ Step 2: Creating minimal training config...")
    create_minimal_config(config_path)

    # Step 3: Fine-tune (minimal epochs)
    print("\n🎯 Step 3: Fine-tuning with minimal data...")
    cmd = f"python scripts/fine_tune.py --config {config_path}"
    os.system(cmd)

    # Step 4: Test transcription
    print("\n🗣️ Step 4: Testing transcription...")

    # Test on training data (should be very good)
    model_path = "minimal_test/results/konkani_asr_minimal.nemo"
    if os.path.exists(model_path):
        print("   Testing on training data...")
        cmd = f"python scripts/infer.py --model_path {model_path} --audio_dir {test_data_dir}/audio --output_file minimal_test/training_transcriptions.json"
        os.system(cmd)

        # Test on a held-out portion if available
        test_manifest = "minimal_test/data/test.tsv"
        if os.path.exists(test_manifest):
            print("   Testing on unseen data...")
            cmd = f"python scripts/evaluate.py --model_path {model_path} --test_manifest {test_manifest}"
            os.system(cmd)

    print("\n" + "=" * 50)
    print("🎉 Minimal test completed!")
    print("Check minimal_test/ directory for results")

def create_minimal_config(config_path):
    """Create a minimal training configuration for testing"""

    config_content = """
name: "Konkani_ASR_Minimal_Test"

model:
  tokenizertype: "bpe"
  # Use smaller model for testing
  encoder:
    n_layers: 6  # Reduced from 12
    d_model: 256  # Smaller hidden size

  decoder:
    vocabulary: null  # Will be set automatically
    d_model: 256

  joint:
    jointnet:
      d_model: 256

trainer:
  devices: 1
  max_epochs: 3  # Very few epochs for testing
  accumulate_grad_batches: 1
  enable_progress_bar: true
  log_every_n_steps: 1

  val_check_interval: 1.0

exp_manager:
  exp_dir: "minimal_test/results"
  name: "konkani_asr_minimal"
  create_checkpoint_callback: true

model_defaults:
  enc_hidden: 256
  pred_hidden: 256
  joint_hidden: 256
"""

    with open(config_path, 'w') as f:
        f.write(config_content)

    print(f"✅ Created minimal config: {config_path}")

def analyze_results():
    """Analyze the results of the minimal test"""

    print("\n📈 Analyzing Results...")

    # Check if results exist
    result_file = "minimal_test/training_transcriptions.json"
    if os.path.exists(result_file):
        print("✅ Training data transcriptions completed")
        # Could add more detailed analysis here
    else:
        print("❌ No transcription results found")

    eval_file = "minimal_test/evaluation_results.json"
    if os.path.exists(eval_file):
        print("✅ Evaluation on unseen data completed")
    else:
        print("⚠️ No evaluation results (expected with very small dataset)")

def main():
    parser = argparse.ArgumentParser(description="Test Konkani ASR with minimal data")
    parser.add_argument("--audio_file", required=True, help="Path to your Konkani audio file (1 minute)")
    parser.add_argument("--transcript", required=True, help="Exact transcript of the audio")
    parser.add_argument("--test_dir", default="minimal_test", help="Directory for test outputs")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.audio_file):
        print(f"❌ Audio file not found: {args.audio_file}")
        sys.exit(1)

    if not args.transcript.strip():
        print("❌ Please provide the transcript text")
        sys.exit(1)

    # Create minimal test data
    test_data_dir = create_minimal_test_data(args.audio_file, args.transcript, args.test_dir)

    # Run the pipeline
    config_path = f"{args.test_dir}/minimal_config.yaml"
    run_minimal_pipeline(test_data_dir, config_path)

    # Analyze results
    analyze_results()

    print("\n" + "=" * 60)
    print("🎯 INTERPRETING YOUR RESULTS:")
    print("=" * 60)
    print("1. If training data transcription is 90%+ accurate:")
    print("   ✅ Fine-tuning is working! Model learned your speech patterns.")
    print("")
    print("2. If unseen data transcription is poor:")
    print("   ⚠️ Expected with 1 minute of data. Need more diverse samples.")
    print("")
    print("3. If both are poor:")
    print("   🔍 Check: audio quality, transcript accuracy, model loading.")
    print("")
    print("4. Next steps:")
    print("   - Collect 10-30 minutes of diverse Konkani speech")
    print("   - Test with different speakers/accents")
    print("   - Gradually increase training data")

if __name__ == "__main__":
    main()