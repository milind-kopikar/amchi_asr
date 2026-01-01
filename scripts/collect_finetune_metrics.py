#!/usr/bin/env python3
"""
Collect scalar metrics from TensorBoard event files and write epoch-level summary JSON
"""
import json
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = os.path.join('results', 'logs', 'konkani_asr_finetune')
OUT = os.path.join('results', 'AI4Bharat_amchi_konkani', 'finetune_epoch_metrics.json')

def find_latest_event_file(log_dir):
    versions = sorted(glob.glob(os.path.join(log_dir, 'version_*')))
    if not versions:
        raise FileNotFoundError('No versions found in log dir')
    latest = versions[-1]
    ev = glob.glob(os.path.join(latest, 'events.out.*'))
    if not ev:
        raise FileNotFoundError('No event files in latest version')
    return ev[0]

def collect_scalars(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    # pick common tags
    interesting = [t for t in tags if any(k in t.lower() for k in ('loss','wer','val_wer'))]
    # get all scalar events and group by their wall_time order
    records = []
    # We will iterate over steps in the scalar with the most points, and sample values at same step if present
    # Gather all scalar events by tag
    tag_series = {t: ea.Scalars(t) for t in interesting}
    # Create a sorted list of unique steps present across these tags
    steps = sorted({e.step for series in tag_series.values() for e in series})
    for epoch_idx, step in enumerate(steps):
        rec = {'epoch': epoch_idx, 'step': step, 'values': {}}
        for tag, series in tag_series.items():
            # find the last value with step <= target step
            val = None
            for e in series:
                if e.step <= step:
                    val = e.value
                else:
                    break
            if val is not None:
                rec['values'][tag] = val
        records.append(rec)
    return records

if __name__ == '__main__':
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    try:
        ev = find_latest_event_file(LOG_DIR)
    except Exception as e:
        print('No event file found:', e)
        ev = None
    out = {'event_file': ev, 'records': []}
    if ev:
        out['records'] = collect_scalars(ev)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=2)
    print('Wrote', OUT)
