#!/usr/bin/env python3
"""Create a Wav2Vec2Processor (feature extractor + tokenizer) using the Devanagari vocab.
Saves processor to `processor_path` for training to use.
"""
import os
import yaml
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor


def make_processor(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    vocab_file = cfg['data']['vocab_file']
    processor_path = cfg['model']['processor_path']
    os.makedirs(processor_path, exist_ok=True)

    print(f"Loading vocab from {vocab_file}...")
    with open(vocab_file, 'r', encoding='utf-8') as fh:
        vocab = json.load(fh)

    # Ensure word_delimiter_token exists (we used '|' for CTC separator)
    unk_token = '[UNK]'
    pad_token = '[PAD]'
    word_delimiter = '|'

    # Create tokenizer
    print('Creating tokenizer...')
    tokenizer = Wav2Vec2CTCTokenizer(vocab_file, unk_token=unk_token, pad_token=pad_token, word_delimiter_token=word_delimiter)

    # Create feature extractor (defaults suitable for wav2vec2)
    print('Creating feature extractor...')
    feature_extractor = Wav2Vec2FeatureExtractor(sample_rate=16000, return_attention_mask=True, do_normalize=True)

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    print(f"Saving processor to {processor_path}...")
    processor.save_pretrained(processor_path)
    print('✅ Processor saved. You can now point training to this processor path.')


if __name__ == '__main__':
    make_processor()
