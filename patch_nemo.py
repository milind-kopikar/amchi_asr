#!/usr/bin/env python3
"""
Patch os.uname for Windows compatibility with NeMo
"""

import os
import sys

# Add uname function for Windows compatibility
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

# Patch the os module
os.uname = uname

print("✅ os.uname patched for Windows compatibility")

# Test NeMo import
try:
    import nemo.collections.asr as nemo_asr
    print("✅ NeMo ASR imported successfully!")
except Exception as e:
    print(f"❌ NeMo import failed: {e}")