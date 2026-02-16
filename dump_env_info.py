#!/usr/bin/env python3
"""Dump runtime environment info (versions and env vars) to results/env_specs/env_info-<timestamp>.json"""
import json
import os
import subprocess
import sys
from datetime import datetime

out_dir = os.path.join('results', 'env_specs')
os.makedirs(out_dir, exist_ok=True)

info = {'date': datetime.utcnow().isoformat() + 'Z', 'python': sys.version, 'env': {}}

# Key environment vars
for v in ['APPLY_CONV_PATCH', 'CUDA_VISIBLE_DEVICES', 'HF_TOKEN']:
    info['env'][v] = os.environ.get(v)

# Packages
pkgs = ['torch', 'torchvision', 'torchaudio', 'nemo', 'sentencepiece', 'librosa', 'pandas', 'jiwer', 'pynini']
info['packages'] = {}
for p in pkgs:
    try:
        mod = __import__(p)
        info['packages'][p] = getattr(mod, '__version__', getattr(mod, 'version', 'unknown'))
    except Exception as e:
        info['packages'][p] = f'NOT INSTALLED ({e})'

# System commands
try:
    info['ffmpeg'] = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True).stdout.splitlines()[0]
except Exception as e:
    info['ffmpeg'] = f'absent ({e})'

# Write file
fn = os.path.join(out_dir, f'env_info_{datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")}.json')
with open(fn, 'w', encoding='utf-8') as fh:
    json.dump(info, fh, indent=2, ensure_ascii=False)

print('Wrote env snapshot to', fn)
print(json.dumps(info, indent=2, ensure_ascii=False))
