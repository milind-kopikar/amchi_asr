#!/usr/bin/env python3
"""
Windows compatibility patch for NeMo/lhotse os.uname issue
"""

import os
import sys
import platform

# Monkey patch os.uname for Windows compatibility
def uname():
    """Mock uname function for Windows"""
    class UnameResult:
        def __init__(self):
            self.sysname = "Windows"
            self.nodename = os.environ.get('COMPUTERNAME', 'localhost')
            self.release = sys.getwindowsversion().major.__str__()
            self.version = sys.getwindowsversion().service_pack or ""
            self.machine = os.environ.get('PROCESSOR_ARCHITECTURE', 'x86_64')

        # Add the attributes that platform.system() expects
        @property
        def system(self):
            return self.sysname

        @property
        def node(self):
            return self.nodename

        @property
        def processor(self):
            return self.machine

    return UnameResult()

# Apply the patch before importing NeMo
if not hasattr(os, 'uname'):
    os.uname = uname

# Also patch platform.uname if it exists
try:
    if hasattr(platform, 'uname'):
        platform.uname = uname
except:
    pass

# Fix ml_dtypes compatibility issue with ONNX
try:
    import ml_dtypes
    import numpy as np

    # Add missing float dtypes that ONNX expects
    missing_dtypes = [
        'float4_e2m1fn',
        'float8_e8m0fnu',
        'float8_e4m3b11fnuz',
        'float8_e5m2fnuz'
    ]

    for dtype_name in missing_dtypes:
        if not hasattr(ml_dtypes, dtype_name):
            # Create dummy dtypes using float32 as fallback
            setattr(ml_dtypes, dtype_name, np.dtype('float32'))
            print(f"Added missing ml_dtypes.{dtype_name}")

    print("Applied ml_dtypes compatibility patch")
except ImportError:
    pass

print("Applied Windows compatibility patch for os.uname")