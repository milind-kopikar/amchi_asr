# Deaf Speech Results Webapp — Data Files

This directory contains experiment results data files that are loaded by the deaf speech webapp.

## File Structure

Each JSON file follows this structure:

```json
{
  "best_checkpoint": "path/to/checkpoint.ckpt",
  "summary": {
    "total_samples": 124,
    "mean_wer": 0.347
  },
  "per_sample": [
    {
      "audio": "data/deaf_speech/audio/130.wav",
      "reference": "दैनंदिन कामे १।",
      "prediction": "दैनंदन कामे १ ⁇ ",
      "wer": 0.0,
      "postprocessed": {
        "prediction": "दैनंदिन कामे १।",
        "mode": "FILL",
        "wer": 0.0
      }
    },
    ...
  ]
}
```

### Fields

- **best_checkpoint**: Path to the trained checkpoint file
- **summary**: Overall statistics
  - `total_samples`: Number of test samples
  - `mean_wer`: Average Word Error Rate (0-1 scale)
- **per_sample**: Array of results for each sample
  - `audio`: Path to audio file
  - `reference`: Ground truth text
  - `prediction`: Raw ASR output
  - `wer`: WER for this sample (0-1 scale, can exceed 1.0)
  - `postprocessed` (optional): Post-processed results
    - `prediction`: Corrected text
    - `mode`: "RECONSTRUCT" | "FILL" | "PASSTHROUGH"
    - `wer`: WER after post-processing

## Generating Data Files

### Extract from Training Results

```bash
# From the konkani_asr project root:
python3 scripts/extract_results_to_webapp.py \
  --experiment deaf_speech_dsd \
  --output webapp/public/data/deaf_speech_results_dsd.json
```

### Add Post-Processing Results

If you have post-processed predictions:

```bash
python3 scripts/postprocess_asr.py \
  --experiment deaf_speech_dsd \
  --input results/experiments/deaf_speech_dsd/final_test_results.json \
  --output data/deaf_speech_results_dsd_postprocessed.json
```

Then merge the post-processing results into the main data file.

### Manual Population

1. Load your experiment's `final_test_results.json`
2. Copy its structure into the corresponding data file
3. (Optional) Add post-processing results by appending `postprocessed` field to each sample
4. The webapp will automatically load and display the data

## Current Files

| File | Experiment | Status | Samples |
|---|---|---|---|
| `deaf_speech_results_baseline.json` | Baseline (50ep) | Placeholder | 3 |
| `deaf_speech_results_dsa.json` | DS-A (frozen) | Placeholder | 3 |
| `deaf_speech_results_dsb.json` | DS-B (extended) | Placeholder | 3 |
| `deaf_speech_results_dsd.json` | DS-D (best) | Partial | 6 |

## Updating Data

To update with complete results:

1. Generate results JSON from training pipeline
2. Validate structure with: `python3 validate_results.py <file>`
3. Place in this directory with correct naming: `deaf_speech_results_<expid>.json`
4. Restart webapp: `npm run dev`
5. Data will auto-load in the browser

## Troubleshooting

### "Failed to load results"
- Check file exists at correct path
- Validate JSON syntax: `jq . deaf_speech_results_dsd.json`
- Check browser console for network errors

### Some samples show N/A
- Verify all samples have `audio`, `reference`, `prediction`, `wer` fields
- Check that WER values are numbers (not strings)

### Post-processing metrics don't show
- Ensure `postprocessed` field is present in samples
- Verify `postprocessed.mode` is one of: RECONSTRUCT, FILL, PASSTHROUGH
- Ensure `postprocessed.wer` is a number

## Adding Post-Processing Support

To enable post-processing comparison in the webapp:

1. Process experiment predictions with: `scripts/postprocess_asr.py`
2. Merge results into data files (add `postprocessed` field per sample)
3. The webapp will automatically:
   - Show post-processing impact metrics
   - Display side-by-side comparisons
   - Calculate improvement percentages

Example merger script:

```python
import json

# Load base results
with open('deaf_speech_results_dsd.json') as f:
    base = json.load(f)

# Load post-processed results
with open('deaf_speech_postprocessed.json') as f:
    postproc = json.load(f)

# Merge post-processing data
for i, sample in enumerate(base['per_sample']):
    if i < len(postproc['per_sample']):
        sample['postprocessed'] = postproc['per_sample'][i]['postprocessed']

# Save merged
with open('deaf_speech_results_dsd.json', 'w') as f:
    json.dump(base, f, indent=2, ensure_ascii=False)
```

---

**Data Source**: `../../../results/experiments/deaf_speech_*/final_test_results.json`  
**Webapp**: `../deaf-speech/page.tsx`  
**Config**: `../lib/deaf-speech-experiments.ts`
