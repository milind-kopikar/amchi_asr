"""Patches package for runtime monkey-patching of third-party libraries.

This package exists so we can import files under `patches.*` at runtime
(e.g., `patches.conv_asr_fixed`).
"""

__all__ = ["conv_asr_fixed"]
