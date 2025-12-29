#!/usr/bin/env python3
"""Simple FastAPI server to serve speech transcription using a HF model.
Usage:
  python scripts/hf_inference_endpoint.py --model facebook/mms-1b-all --port 8000

POST /transcribe accepts multipart/form-data {"file": <audio file>}
Returns JSON: {"text": "transcription"}
"""

import argparse
import tempfile
import shutil
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import uvicorn
import torch

app = FastAPI()
model = None
processor = None
pipe = None

@app.on_event('startup')
async def load_model():
    global model, processor, pipe
    from transformers import pipeline, AutoProcessor, AutoModelForCTC
    args = getattr(app.state, 'args', None)
    model_id = None
    if args is not None:
        model_id = args.model
    else:
        model_id = os.environ.get('MODEL_ID', 'facebook/mms-1b-all')
    print('Loading model on startup:', model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    try:
        model = AutoModelForCTC.from_pretrained(model_id).to('cuda' if torch.cuda.is_available() else 'cpu')
        pipe = pipeline('automatic-speech-recognition', model=model, processor=processor, device=0 if torch.cuda.is_available() else -1)
    except Exception as e:
        print('Warning: failed to load model directly into pipeline:', e)
        pipe = None

@app.post('/transcribe')
async def transcribe(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        if pipe:
            res = pipe(tmp_path)
            text = res['text']
        else:
            # fallback manual inference
            import soundfile as sf
            audio_arr, sr = sf.read(tmp_path)
            inputs = processor(audio_arr, sampling_rate=sr, return_tensors='pt', padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)[0]
            text = processor.batch_decode(predicted_ids.unsqueeze(0))[0]
        return JSONResponse({'text': text})
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='facebook/mms-1b-all')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    app.state.args = args
    uvicorn.run('scripts.hf_inference_endpoint:app', host=args.host, port=args.port, log_level='info', reload=False)
