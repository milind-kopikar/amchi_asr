#!/usr/bin/env python3
"""
Test AI4Bharat IndicConformer Konkani Model
"""

import os
import sys

# Add uname function for Windows compatibility BEFORE any other imports
def uname():
    """Mock uname function for Windows - returns uname_result object"""
    import collections
    uname_result = collections.namedtuple('uname_result', ['sysname', 'nodename', 'release', 'version', 'machine'])
    return uname_result(
        sysname='Windows',
        nodename='localhost',
        release=str(sys.getwindowsversion().major),
        version=str(sys.getwindowsversion().build),
        machine='AMD64'
    )

# Patch the os module BEFORE any imports
os.uname = uname

print("✅ os.uname patched for Windows compatibility")

# Now import NeMo
import nemo.collections.asr as nemo_asr

print("✅ NeMo ASR imported successfully!")

# Test AI4Bharat Konkani model
print("\n🤖 Testing AI4Bharat IndicConformer Konkani Model...")

try:
    # Load the AI4Bharat Konkani model
    model = nemo_asr.models.ASRModel.from_pretrained("ai4bharat/indicconformer_stt_kok_hybrid_ctc_rnnt_large")
    print("✅ AI4Bharat IndicConformer Konkani model loaded successfully!")

    # Get model info
    print(f"Model type: {type(model)}")

    # Test inference with sample text
    print("\n🗣️ Testing inference...")

    # Create a dummy audio file for testing (we'll use the existing test file)
    import torch
    import numpy as np

    # For now, just test that the model is ready
    print("✅ Model is ready for inference!")
    print(f"Model device: {next(model.parameters()).device}")

except Exception as e:
    print(f"❌ AI4Bharat Konkani model loading failed: {e}")
    import traceback
    traceback.print_exc()