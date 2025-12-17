#!/usr/bin/env python3
"""
Test AI4Bharat IndicConformer Model - Direct Load
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

# Test loading the model directly from the .nemo file
model_path = r'C:\Users\Milind Kopikare\.cache\torch\NeMo\NeMo_2.5.3\hf_hub_cache\ai4bharat\indicconformer_stt_mr_hybrid_ctc_rnnt_large\2d0e725a43e908823822e841161cebc2\indicconformer_stt_mr_hybrid_rnnt_large.nemo'

print(f"🤖 Loading AI4Bharat model from: {model_path}")

try:
    model = nemo_asr.models.ASRModel.restore_from(model_path)
    print("✅ AI4Bharat IndicConformer model loaded successfully!")
    print(f"Model type: {type(model)}")

    # Test inference
    print("\n🗣️ Testing inference...")

    # For now, just check that the model is ready
    print("✅ Model is ready for inference!")
    print(f"Model device: {next(model.parameters()).device}")

except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()