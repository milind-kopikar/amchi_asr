#!/usr/bin/env python3
"""Run preflight checks to validate environment and key resources before training.

Checks performed:
- Python version
- ffmpeg availability
- torch import and CUDA availability
- nemo import and conv_asr patch presence
- tokenizer encoding for a sample Devanagari string
- disk free space
- model file existence referenced in config
"""

import os
import sys
import shutil
import json
import yaml
from pathlib import Path

SAMPLE_TEXT = "रोहन होड ज़ाल्लो!"
RECOMMENDED_PYTHON = (3, 11)
MIN_PYTHON = (3, 9)


def check_python_version():
    v = sys.version_info
    ok = v >= MIN_PYTHON
    note = f"python {v.major}.{v.minor}.{v.micro}"
    recommended = v >= RECOMMENDED_PYTHON
    return {'ok': ok, 'recommended': recommended, 'note': note}


def check_ffmpeg():
    ff = shutil.which('ffmpeg')
    return {'ok': bool(ff), 'path': ff}


def check_torch():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        return {'ok': True, 'torch_version': torch.__version__, 'cuda_available': cuda_avail}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def check_nemo_and_patch():
    try:
        import nemo
        import nemo.collections.asr.modules.conv_asr as conv
        p = Path(conv.__file__)
        text = p.read_text(errors='ignore')
        patched = ('_LanguageMaskList' in text) or ('LanguageMask' in text) or ('conv_asr_fixed' in text)
        return {'ok': True, 'conv_asr_path': str(p), 'patched': bool(patched)}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def find_local_tokenizer():
    # search models/tokenizer or tokenizers/
    candidates = list(Path('models').rglob('*tokenizer.model')) + list(Path('tokenizers').rglob('*.model'))
    return candidates[0] if candidates else None


def check_tokenizer():
    try:
        import sentencepiece as spm
    except Exception as e:
        return {'ok': False, 'error': f'sentencepiece not installed: {e}'}

    model_path = find_local_tokenizer()
    if not model_path:
        return {'ok': False, 'error': 'tokenizer model not found under models/ or tokenizers/'}

    try:
        sp = spm.SentencePieceProcessor(model_file=str(model_path))
    except Exception as e:
        return {'ok': False, 'error': f'failed to load tokenizer: {e}', 'model_path': str(model_path)}

    ids = sp.encode(SAMPLE_TEXT, out_type=int)
    unk = sp.unk_id()
    has_nonunk = any(i != unk for i in ids)
    decoded = sp.decode(ids)
    return {'ok': has_nonunk, 'model_path': str(model_path), 'ids': ids, 'unk_id': unk, 'decoded': decoded}


def check_disk_space(min_gb=10):
    usage = shutil.disk_usage('.')
    free_gb = usage.free / (1024 ** 3)
    ok = free_gb >= min_gb
    return {'ok': ok, 'free_gb': round(free_gb, 2)}


def check_model_in_config(config_path='configs/konkani_finetune.yaml'):
    if not Path(config_path).exists():
        return {'ok': False, 'error': f'config not found: {config_path}'}
    try:
        with open(config_path, 'r', encoding='utf-8') as fh:
            cfg = yaml.safe_load(fh)
        model_path = cfg.get('model', {}).get('nemo_model')
        if not model_path:
            return {'ok': False, 'error': 'nemo_model not set in config'}
        exists = Path(model_path).exists()
        return {'ok': exists, 'model_path': model_path}
    except Exception as e:
        return {'ok': False, 'error': str(e)}


def run_all():
    results = {
        'python': check_python_version(),
        'ffmpeg': check_ffmpeg(),
        'torch': check_torch(),
        'nemo_patch': check_nemo_and_patch(),
        'tokenizer': check_tokenizer(),
        'disk': check_disk_space(),
        'model_config': check_model_in_config(),
        'env': {'APPLY_CONV_PATCH': os.environ.get('APPLY_CONV_PATCH', None)}
    }
    ok = all(v.get('ok', False) for v in results.values())
    return ok, results


if __name__ == '__main__':
    ok, results = run_all()
    print(json.dumps(results, indent=2, ensure_ascii=False))
    if not ok:
        print('\nOne or more preflight checks failed. Please address the issues above.')
        sys.exit(2)
    print('\nAll preflight checks passed.')
    sys.exit(0)
