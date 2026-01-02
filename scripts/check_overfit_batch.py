#!/usr/bin/env python3
"""Load model and a batch to compute training loss for debugging overfit step"""
import torch
import nemo.collections.asr as nemo_asr
from pathlib import Path

# Load model via partial restore to avoid strict weight mismatch
model_path = 'models/indicconformer_mr/indicconformer_stt_mr_hybrid_rnnt_large_patched.nemo'
from nemo.collections.asr.models import ASRModel as _ASRModel
try:
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(model_path, strict=False)
except Exception as e:
    print('Restore strict failed, attempting partial restore:', e)
    import tarfile, tempfile, yaml, torch
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(model_path, 'r') as tar:
            tar.extract('model_config.yaml', path=td)
            tar.extract('model_weights.ckpt', path=td)
        conf = yaml.safe_load(open(Path(td)/'model_config.yaml', 'r').read())
        model = _ASRModel.from_config_dict(conf, trainer=None)
        ckpt = torch.load(Path(td)/'model_weights.ckpt', map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        model_sd = model.state_dict()
        filtered = {k:v for k,v in state.items() if k in model_sd and list(v.shape)==list(model_sd[k].shape)}
        model.load_state_dict(filtered, strict=False)

model = model.cuda() if torch.cuda.is_available() else model

# Build dataset and dataloader (reuse overfit config paths)
from nemo.collections.asr.data.audio_to_text import AudioToBPEDataset
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
import os
unpacked = 'models/indicconformer_mr/unpacked'
sp_model = None
for f in os.listdir(unpacked):
    if f.endswith('_tokenizer.model'):
        sp_model = os.path.join(unpacked, f)
        break
if sp_model is None:
    raise SystemExit('No sp model found')

tokenizer = SentencePieceTokenizer(sp_model)

dataset = AudioToBPEDataset(manifest_filepath='data/train/manifest.jsonl', tokenizer=tokenizer, sample_rate=16000, return_language_id=True)
from torch.utils.data import DataLoader
collate = getattr(dataset, 'collate_fn', None)
loader = DataLoader(dataset, batch_size=3, collate_fn=collate)

batch = next(iter(loader))
print('Batch type:', type(batch))
# print a short summary of batch elements
if isinstance(batch, dict):
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            print(k, v.shape, v.dtype)
        else:
            print(k, type(v))
else:
    print('Batch repr (first 2 elems):', [type(x) for x in batch[:2]])

# The model expects a dict with named tensors; if batch is tuple (audio, text, ...), try to use dataset._process to convert
if not isinstance(batch, dict):
    # try dataset._process to get dict (internal helper on _AudioTextDataset)
    try:
        batch = dataset._process(batch)
        print('Converted batch via dataset._process')
    except Exception as e:
        print('Could not convert batch to dict:', e)

# Move tensors to model device
device = next(model.parameters()).device
if isinstance(batch, dict):
    batch = {k:(v.cuda() if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
else:
    # tuple: move tensor elements to device
    batch = tuple((v.cuda() if isinstance(v, torch.Tensor) else v) for v in batch)

# Call training_step
model.train()
# If batch is a tuple from DataLoader, manually construct the dict the model expects
if not isinstance(batch, dict):
    # Expected tuple layout: (audio_signal, audio_signal_length, transcript, transcript_length) - try to unpack
    try:
        audio_signal, audio_lengths, transcript, transcript_len = batch
    except Exception:
        # Fallback: try first two tensors for audio and length
        audio_signal = batch[0]
        audio_lengths = batch[1] if len(batch) > 1 else torch.full((audio_signal.shape[0],), audio_signal.shape[1], dtype=torch.long)
        transcript = batch[2] if len(batch) > 2 else None
        transcript_len = batch[3] if len(batch) > 3 else None

    # Move tensors to device
    audio_signal = audio_signal.to(device) if hasattr(audio_signal, 'to') else audio_signal
    audio_lengths = audio_lengths.to(device) if hasattr(audio_lengths, 'to') else audio_lengths
    if transcript is not None:
        transcript = transcript.to(device) if hasattr(transcript, 'to') else transcript
    if transcript_len is not None:
        transcript_len = transcript_len.to(device) if hasattr(transcript_len, 'to') else transcript_len

    # Determine kok index
    try:
        kok_idx = list(model.joint.language_keys).index('kok')
    except Exception:
        try:
            kok_idx = list(model.cfg.joint.language_keys).index('kok')
        except Exception:
            kok_idx = 0

    # Construct 6-element tuple expected by training_step:
    # (signal, signal_length, transcript, transcript_length, sample_ids, language_ids)
    bsize = audio_signal.shape[0]
    # transcripts may be None here; create placeholder tensors if missing
    if transcript is None:
        transcript = torch.zeros((bsize, 1), dtype=torch.long, device=device)
        transcript_len = torch.zeros((bsize,), dtype=torch.long, device=device)
    else:
        transcript = transcript.to(device) if hasattr(transcript, 'to') else transcript
        if transcript_len is None:
            # Try to infer lengths if transcript is 2D LongTensor
            try:
                transcript_len = torch.sum((transcript != 0).long(), dim=1)
            except Exception:
                transcript_len = torch.full((bsize,), transcript.shape[1] if transcript.dim()>1 else 1, dtype=torch.long, device=device)
        else:
            transcript_len = transcript_len.to(device) if hasattr(transcript_len, 'to') else transcript_len
    sample_ids = torch.arange(bsize, dtype=torch.long, device=device)
    # Use a list of language key strings (e.g., 'kok') since joint modules are keyed by language IDs
    try:
        lang_key = list(model.joint.language_keys)[kok_idx]
    except Exception:
        lang_key = 'kok'
    language_ids = [lang_key for _ in range(bsize)]

    batch = (audio_signal, audio_lengths, transcript, transcript_len, sample_ids, language_ids)

else:
    # Ensure tensors are on device and language_ids exist
    batch = {k:(v.to(device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
    if batch.get('language_ids') is None:
        try:
            kok_idx = list(model.joint.language_keys).index('kok')
        except Exception:
            kok_idx = 0
        b = batch.get('input_signal').shape[0]
        batch['language_ids'] = torch.full((b,), kok_idx, dtype=torch.long, device=device)

try:
    out = model.training_step(batch, 0)
    print('training_step output type:', type(out))
    if isinstance(out, dict) and 'loss' in out:
        loss = out['loss']
    else:
        # Some models return loss directly
        loss = out
    print('Loss:', loss.item() if hasattr(loss, 'item') else loss)
    loss.backward()
    print('Backward OK')
except Exception as e:
    import traceback
    traceback.print_exc()
    print('Error during training_step:', e)
