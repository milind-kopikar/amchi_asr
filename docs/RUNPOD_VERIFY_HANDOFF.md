# Handoff: verify inference on a RunPod GPU pod

**Read this before doing anything.** This is a short, scoped task. Do not start a
broader refactor — the next thing after this passes is the GitHub Actions Docker
build (see [docs/RUNPOD_GITHUB_ACTIONS_SETUP.md](RUNPOD_GITHUB_ACTIONS_SETUP.md)).

## Task in one line

Run [scripts/verify_inference.py](../scripts/verify_inference.py) end-to-end for
both `--variant deaf` and `--variant amchi` on this pod and get a green
`5/5 samples within CER ≤ 0.05` result for each.

If both pass, the inference pipeline is verified and we can build the Docker
image with high confidence. If either fails, the failure mode is interesting —
see the diagnosis table at the bottom of [docs/VERIFY_INFERENCE.md](VERIFY_INFERENCE.md).

## Why we're on a pod (and not Colab)

Original plan was to run the verifier on a Colab T4 (free GPU). We spent ~2 hours
and 6 install-cell iterations on Colab and hit a hard `ResolutionImpossible` from
pip:

- NeMo 2.7.0 (training-matched) pins `numba-cuda` to a version < 0.22
- Colab's 2026-05 base image ships `numba 0.65.1` which only works with
  `numba-cuda >= 0.22`
- Colab forces Python 3.12; training was Python 3.11
- The conflict is fundamental, not a missing pin

Earlier Colab iterations also surfaced a torch 2.12 vs torch 2.10 kwarg
incompatibility (`Config(deprecated=True)`) and a `lightning.fabric._graveyard.tpu`
circular import. **None of these will reproduce on this pod** — the pod is
Python 3.11 with the actual training-era package matrix. If you find yourself
adding `lightning==2.4.0` or `numba-cuda` pins, stop — those were Colab-only.

## Training environment (target — match this)

Per [LEARNINGS.md](../LEARNINGS.md) §5–6 and [AGENT_START_HERE.md](../AGENT_START_HERE.md) §7:

- **Python 3.11**
- **`nemo_toolkit[asr]` v2.7.0** (upstream NVIDIA, NOT the AI4Bharat fork)
- **`torch` 2.10.0+cu128** (or whatever the pod template ships if it's torch ≥ 2.4 + CUDA ≥ 12.4 — NeMo 2.7.0 is flexible here on a clean Python 3.11)
- Install with `--ignore-installed blinker` (known Ubuntu 22.04 conflict)

Recommended pod template: **RunPod PyTorch 2.4** (Python 3.11, CUDA 12.4). RTX 4000
Ada is enough (~$0.30/hr) — the verifier is short.

## What's already done in the repo

You're on branch `feat/live-asr-demo-recording`. Don't switch branches.

| Asset | Status |
|---|---|
| [scripts/verify_inference.py](../scripts/verify_inference.py) | Written, 35 unit tests passing locally |
| [scripts/runpod_smoke.py](../scripts/runpod_smoke.py) | Staged smoke test, 29 tests passing |
| [scripts/build_amchi_dict.py](../scripts/build_amchi_dict.py) | Builds Konkani dict JSON, 34 tests passing |
| [scripts/runpod_http.py](../scripts/runpod_http.py) | UA-injected HTTP helpers (R2 needs UA), 21 tests passing |
| [scripts/prewarm_runpod.py](../scripts/prewarm_runpod.py) | Demo-day prewarm, 28 tests passing |
| [runpod/handler.py](../runpod/handler.py) | Amchi handler with Gemini post-proc, 25 tests passing |
| [runpod/handler_deaf.py](../runpod/handler_deaf.py) | Deaf handler with Gemini post-proc |
| [runpod/Dockerfile.serverless](../runpod/Dockerfile.serverless) | Ready but not yet built |
| [runpod/Dockerfile.deaf](../runpod/Dockerfile.deaf) | Ready but not yet built |
| [patches/conv_asr_fixed.py](../patches/conv_asr_fixed.py) | The conv_asr patch needed to load the hybrid CTC/RNNT checkpoint |
| Both checkpoints on R2 | Public URLs in `scripts/verify_inference.py::DEFAULT_CHECKPOINTS` |
| Final test results | `results/run_s_stratified_split/test/final_test_results.json` (amchi), `results/deaf_speech_dsd/test/final_test_results.json` (deaf) — these are the "stored predictions" the verifier compares against |

Latest commit on branch: `b5f5880` (Colab notebook pins — irrelevant on the pod).

## Exact runbook on the pod

```bash
# 1. Clone (the pod starts empty)
git clone --branch feat/live-asr-demo-recording \
  https://github.com/milind-kopikar/amchi_asr.git
cd amchi_asr

# 2. Fresh venv on Python 3.11 (the pod's default if you chose the right template)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install nemo + utilities. Unpinned nemo is fine on Python 3.11 — the
#    training environment was the latest available, and 2.7.0 was just what
#    the pod happened to install. If the unpinned install fails or the
#    smoke-test step below complains, pin nemo_toolkit[asr]==2.7.0.
pip install "nemo_toolkit[asr]" \
  librosa omegaconf jiwer google-genai runpod \
  --ignore-installed blinker

# 4. Apply the conv_asr patch (the hybrid CTC/RNNT checkpoint needs it)
python3 -c "
import shutil, nemo.collections.asr.modules.conv_asr as m
shutil.copy('patches/conv_asr_fixed.py', m.__file__)
print('patched', m.__file__)
"

# 5. Build the Konkani dictionary (for the Amchi handler's post-processor)
python3 scripts/build_amchi_dict.py

# 6. Smoke-check both variants before the slow step
python3 scripts/runpod_smoke.py --variant deaf
python3 scripts/runpod_smoke.py --variant amchi

# 7. Full verification — deaf first (canary), then amchi
python3 scripts/verify_inference.py --variant deaf --num-samples 5
python3 scripts/verify_inference.py --variant amchi --num-samples 5
```

Each verifier call:
1. Downloads the checkpoint from R2 (~480 MB amchi, ~1.4 GB deaf)
2. Downloads 5 held-out test audio files from the Railway data API
3. Loads the model with the conv_asr patch
4. Transcribes each sample with `transcribe_audio_bytes`
5. Computes CER vs the stored prediction in `final_test_results.json`
6. Prints a side-by-side table and exits 0 (pass), 1 (regression), or 2 (prereq missing)

## What "pass" looks like

Final line of each `verify_inference.py` run should print:

```
==> Result: 5/5 samples within CER ≤ 0.05
   Inference pipeline verified. Safe to bake into Docker.
```

If both `deaf` and `amchi` print that, you're done. Tell the user. Next step
is the GitHub Actions Docker build — they will trigger it; you do not need to
build Docker on the pod.

## If a sample fails

`verify_inference.py` prints `stored:` vs `fresh:` for each sample. Common
causes are in [docs/VERIFY_INFERENCE.md](VERIFY_INFERENCE.md) § "What to do if
it fails". Most likely real failure modes on a fresh pod:

| Symptom | Likely cause | Action |
|---|---|---|
| Pod can't reach R2 (`HTTP 403`) | UA missing | Confirm `scripts/runpod_http.py` is used (it is, by both `verify_inference.py` and the handlers). If still 403, the bucket policy may have changed — check R2 console. |
| `nemo.collections.asr` import raises | Wrong nemo or torch version on pod | Pin `nemo_toolkit[asr]==2.7.0` explicitly. |
| `conv_asr` import works but model load raises `KeyError` on a config field | Newer nemo renamed something | Try `nemo_toolkit[asr]==2.7.0` exact. Don't go newer. |
| Fresh transcription is empty | Decoding strategy not set | `change_decoding_strategy(decoder_type='ctc')` should run inside the inference function — check `scripts/deaf_speech_inference.py::transcribe_wav` and `scripts/amchi_inference.py`. |
| Off by 1–2 chars per sample | CUDA float ordering noise | Bump `--tolerance 0.08` (don't go higher than 0.10). |
| All 5 samples drift the same way | Gemini post-processing happened in one run and not the other | Check whether `GEMINI_API_KEY` is set on this pod and was/wasn't set when `final_test_results.json` was generated. |

## Things NOT to do

- **Do not** retry the Colab notebook fixes (lightning==2.4.0, numba-cuda pins, etc.). Those are Colab-only artifacts.
- **Do not** edit production code (`scripts/*`, `runpod/*`) to make the verifier pass. The verifier exercises the same `load_model_from_ckpt` / `transcribe_audio_bytes` functions the RunPod handler calls — if you need to change those to get a pass, you've found a real regression. Stop and tell the user.
- **Do not** skip the `conv_asr` patch step. Without it, the model loads but produces garbage Devanagari noise.
- **Do not** push code from the pod. The pod is for verification only. The user pushes from their laptop.
- **Do not** stop the pod when done unless the user asks — they may want to keep it warm for follow-up tests.

## When you're done

Report back to the user with:
1. Both verifier exit codes
2. The final `Result:` line from each
3. Cold-start time (first sample) and warm time (subsequent samples) — visible in the verifier's per-sample timing output
4. Anything unexpected

Then the user will trigger the GitHub Actions Docker build from their laptop.
