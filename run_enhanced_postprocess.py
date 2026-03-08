#!/usr/bin/env python3
"""
Wrapper script to run enhanced post-processing with API key
"""

import os
import subprocess
import sys

# Set the API key
os.environ["GEMINI_API_KEY"] = "AIzaSyAwBr6FlR2nXTDyWqI8dBIhKBXeugND-Gw"

# Run the post-processing script
cmd = [
    "C:/Users/Milind Kopikare/AppData/Local/Programs/Python/Python313/python.exe",
    "scripts/postprocess_asr.py",
    "--input", "nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/final_test_results.json",
    "--output", "enhanced_postprocess_results.json",
    "--report", "enhanced_postprocess_report.txt"
]

print("Running enhanced post-processing on deaf speech test set...")
print("Command:", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, cwd="C:/Users/Milind Kopikare/Code/amchi_konkani/konkani_asr")

print("STDOUT:")
print(result.stdout)
print("STDERR:")
print(result.stderr)
print(f"Return code: {result.returncode}")