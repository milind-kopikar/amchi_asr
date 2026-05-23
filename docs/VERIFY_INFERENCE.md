# Verify ASR inference before building the Docker image

Before spending 30 min building a Docker image and another 5 min pushing
it to Docker Hub, run this end-to-end inference check on a GPU host. If
the verifier passes, the Docker image will almost certainly work — the
inference path is the same.

## What gets verified

For each of 5 representative held-out test samples (per model):

1. The checkpoint downloads cleanly from R2 (User-Agent fix exercised)
2. The conv_asr patch applies + the model loads
3. The held-out audio file downloads from the Railway data API
4. The production inference function transcribes it
5. The fresh transcription matches the stored prediction with CER ≤ 0.05

A pass means **no regression in the inference pipeline since the official
test set was run** — safe to bake into Docker and deploy to RunPod.

## How to run on Google Colab (recommended — free GPU, ~30 min)

1. Open https://colab.research.google.com → **File → Open notebook → GitHub**
2. Enter `milind-kopikar/amchi_asr`, pick the branch
   `feat/live-asr-demo-recording`, then open `docs/verify_inference_colab.ipynb`.
3. **Runtime → Change runtime type → T4 GPU** (free).
4. **Runtime → Run all**. The first cell takes ~15 min (NeMo install +
   checkpoint downloads); each verification cell takes 1–2 min.
5. Look at the final two cells' output. Each prints something like:

   ```
   ==> Result: 5/5 samples within CER ≤ 0.05
      Inference pipeline verified. Safe to bake into Docker.
   ```

   If either model shows `samples drifted from stored predictions — pipeline regression`,
   read the diagnosis section at the bottom of the notebook before proceeding.

## How to run on a local GPU machine

```bash
# 1. Clone the repo and check out the feature branch
git clone --branch feat/live-asr-demo-recording \
    https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr

# 2. Install deps in a fresh virtualenv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "nemo_toolkit[asr]" librosa omegaconf jiwer google-genai runpod \
    --ignore-installed blinker

# 3. Apply the conv_asr patch
python3 -c "
import shutil, nemo.collections.asr.modules.conv_asr as m
shutil.copy('patches/conv_asr_fixed.py', m.__file__)
print('patched', m.__file__)
"

# 4. Build the Konkani dictionary (Amchi only)
python3 scripts/build_amchi_dict.py

# 5. Run the smoke checks first (catches install issues fast)
python3 scripts/runpod_smoke.py --variant deaf
python3 scripts/runpod_smoke.py --variant amchi

# 6. Run the inference verification — deaf first (canary)
python3 scripts/verify_inference.py --variant deaf
python3 scripts/verify_inference.py --variant amchi
```

The verifier exits 0 on full success, 1 on regression, 2 on missing
prereqs. The Colab cells call this same script.

## How to run on a RunPod GPU pod (not serverless)

If you'd rather use RunPod's GPU pods instead of Colab:

1. Spin up an RTX 4000 Ada pod (~$0.30/hr) with the PyTorch 2.0 + CUDA 11.8
   template.
2. SSH in and run the same commands as the local section above.
3. Stop the pod when done (~$0.50 total for the verification run).

## Choosing the right tolerance

Default is `--tolerance 0.05` (CER 5% drift between fresh and stored).
That's intentionally strict — anything above 5% suggests a real pipeline
change, not just stochastic CUDA noise.

Loosen to `--tolerance 0.08` if you've intentionally changed something
in the inference pipeline (e.g. decoding strategy, audio preprocessing)
and want to allow that drift.

Don't go above `0.10` — at that point you're effectively running a model
quality test, not a regression check.

## What to do if it fails

The verifier prints a side-by-side `stored:` vs `fresh:` for every sample,
which makes the failure mode obvious. Common causes:

| Symptom | Most likely cause | Fix |
|---|---|---|
| `fresh` is empty | Wrong decoding strategy or model not in eval mode | Verify `change_decoding_strategy(decoder_type='ctc')` runs in the inference function |
| `fresh` is garbled Devanagari noise | conv_asr patch didn't apply | Re-run the patch step; check `python scripts/runpod_smoke.py --check patch` |
| `fresh` and `stored` differ by 1-2 chars per sample | CUDA float ordering noise | Bump `--tolerance` to 0.08 |
| `fresh` reasonable but very different from `stored` | Different post-processing path | Check `GEMINI_API_KEY` is or isn't set the same way as when the golden ran |
| HTTP 403 downloading the checkpoint | User-Agent missing | Check `scripts/runpod_http.py` is being used (not raw `urllib.urlretrieve`) |
| HTTP 404 on the audio download | Recording id missing from Railway | Skip that sample with `--num-samples 4` or check the Railway DB |

## Why I trust the verifier

The verifier exercises **the same `load_model_from_ckpt` / `transcribe_audio_bytes`
functions** the RunPod handler calls. If the verifier passes on a GPU host,
the only thing that changes inside Docker is the build environment — not
the inference logic. A green verifier + green smoke test in the Dockerfile
gives high confidence the deployed endpoint will work.

What it does NOT cover:

- RunPod-specific job-event shape (we have unit tests for that)
- Gemini post-processing API quotas / errors (verifier exercises the path
  if `GEMINI_API_KEY` is set, but the Gemini call itself is stateless)
- Cold-start time on RunPod (use `scripts/prewarm_runpod.py` on demo day)
