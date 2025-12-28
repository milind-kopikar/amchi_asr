#!/usr/bin/env python3
"""A tiny test script to verify remote editing and execution on RunPod."""
print("Hello from RunPod")
with open('runpod_test_output.txt', 'w', encoding='utf-8') as f:
    f.write('Hello from RunPod')
