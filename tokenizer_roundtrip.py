#!/usr/bin/env python3
"""Simple tokenizer round-trip check using the model's tokenizer"""
import argparse
import sys

# We avoid importing heavy NeMo modules here to keep the script lightweight -
# we parse the unpacked model_config.yaml and use sentencepiece directly.

TEXT = "एकी गोम्टी काणी आय्कयाति"


def main(model_path):
    # Instead of restoring heavy model weights (which may fail for shape mismatches),
    # read tokenizer definition from the unpacked model config and load SentencePiece model directly.
    import yaml
    from pathlib import Path
    unpacked = Path(model_path).parent / 'unpacked' if model_path.endswith('.nemo') else Path(model_path)
    cfg_file = unpacked / 'model_config.yaml'
    if not cfg_file.exists():
        print('Could not find unpacked model_config.yaml at', cfg_file)
        sys.exit(2)

    cfg = yaml.safe_load(cfg_file.read_text(encoding='utf-8'))
    tk_model = None
    if 'tokenizer' in cfg and 'model_path' in cfg['tokenizer']:
        mpath = cfg['tokenizer']['model_path']
        # strip nemo: prefix
        if isinstance(mpath, str) and mpath.startswith('nemo:'):
            mname = mpath.replace('nemo:', '')
            candidate = unpacked / mname
            if candidate.exists():
                tk_model = str(candidate)
    if tk_model is None:
        print('Could not determine tokenizer.model from model_config.yaml')
        sys.exit(3)

    print('Loading SentencePiece model:', tk_model)
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tk_model)

    # Encode
    ids = sp.encode(TEXT, out_type=int)
    print('Encoded ids:', ids)

    # Decode
    text2 = sp.decode(ids)
    print('Decoded text:', text2)

    print('Encoded ids (up to 40):', ids[:40] if ids is not None else None)

    # Decode using SentencePiece
    text2 = sp.decode(ids)

    print('\nRound-trip result:')
    print('input :', TEXT)
    print('output:', text2)
    if text2 == TEXT:
        print('SUCCESS: round-trip matches exactly')
        return 0
    else:
        print('FAIL: round-trip mismatch')
        return 1


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', default='models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo')
    args = p.parse_args()
    sys.exit(main(args.model_path))
