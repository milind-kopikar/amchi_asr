#!/usr/bin/env python3
import soundfile as sf
from pathlib import Path

def duration(path):
    data, sr = sf.read(path)
    return round(len(data)/sr, 4)

base=Path('data')
base.mkdir(exist_ok=True)
# train
train_files=['data/train/audio/145.wav','data/train/audio/146.wav','data/train/audio/147.wav']
train_texts=['माझे नाव अमची आहे','तुमी कसो आसा','आज चांगलो हवामान आहे']
train_manifest=Path('data/train/manifest.jsonl')
train_manifest.parent.mkdir(parents=True, exist_ok=True)
import json
with open(train_manifest,'w',encoding='utf-8') as f:
    for a,t in zip(train_files,train_texts):
        d=duration(a)
        f.write(json.dumps({"audio_filepath":a, "text":t, "duration":d}, ensure_ascii=False) + "\n")
print('Wrote',train_manifest)
# dev
dev_files=['data/dev/audio/148.wav']
dev_texts=['हे एक चाचणी वाक्य आहे']
dev_manifest=Path('data/dev/manifest.jsonl')
dev_manifest.parent.mkdir(parents=True, exist_ok=True)
import json
with open(dev_manifest,'w',encoding='utf-8') as f:
    for a,t in zip(dev_files,dev_texts):
        d=duration(a)
        f.write(json.dumps({"audio_filepath":a, "text":t, "duration":d}, ensure_ascii=False) + "\n")
print('Wrote',dev_manifest)
# test (three samples)
test_files=['data/test/audio/279.wav','data/train/audio/145.wav','data/train/audio/146.wav']
test_texts=['एक शेवटचा चाचणी','माझे नाव अमची आहे','तुमी कसो आसा']
test_manifest=Path('data/test/manifest.jsonl')
import json
with open(test_manifest,'w',encoding='utf-8') as f:
    for a,t in zip(test_files,test_texts):
        d=duration(a)
        f.write(json.dumps({"audio_filepath":a, "text":t, "duration":d}, ensure_ascii=False) + "\n")
print('Wrote',test_manifest)