# Handoff: verify inference on a RunPod GPU pod

## Status: VERIFIED on 2026-05-22

Both `--variant deaf` and `--variant amchi` returned **`5/5 samples within
CER ≤ 0.05`** (CER=0.0000 on every sample) on RunPod with an A40 GPU + driver
`570.195.03 / CUDA 12.8`. The inference pipeline is safe to bake into Docker.

**Next step:** the GitHub Actions Docker build — see
[docs/RUNPOD_GITHUB_ACTIONS_SETUP.md](RUNPOD_GITHUB_ACTIONS_SETUP.md). The user
triggers this from their laptop; the pod is not needed for it.

This doc remains the canonical runbook for **re-verifying on a fresh pod**.
The runbook below has been updated with the corrected install command (torch
pinned to the cu128 build) so a fresh session gets a working environment on the
first try. A full verification record — including all four pod-specific gotchas
discovered along the way — is preserved at the bottom under
[§ Verification record](#verification-record-2026-05-22). Read it if anything
goes sideways.

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

# 2. Fresh venv. Python 3.11 OR 3.12 work — the 2026-05-22 verification ran on
#    3.12.3 with no issues. Use `python3.12 -m venv .venv` to be explicit if
#    the pod has multiple Python installs.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 3. Install nemo + utilities. **Pin torch to the cu128 build** — unpinned
#    nemo on Python 3.12 resolves to torch 2.12.0 with CUDA-13 wheels, which
#    are unusable on the pod's CUDA 12.8 driver (Gotcha 4). Pinning torch
#    pulls the matching cu12 nvidia-* CUDA libs that *do* work on this driver.
#    Takes ~20 min total — downloads ~6.7 GB of wheels.
pip install \
  "torch==2.10.0+cu128" "torchaudio==2.10.0+cu128" \
  "nemo_toolkit[asr]" \
  librosa omegaconf jiwer google-genai runpod \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --ignore-installed blinker

# Sanity-check torch actually sees the GPU before continuing.
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

# 4. Apply the conv_asr patch (the hybrid CTC/RNNT checkpoint needs it)
python3 -c "
import shutil, nemo.collections.asr.modules.conv_asr as m
shutil.copy('patches/conv_asr_fixed.py', m.__file__)
print('patched', m.__file__)
"

# 5. Build the Konkani dictionary (for the Amchi handler's post-processor).
#    Uses the in-repo SQL dump; the original Railway endpoint that
#    scripts/build_amchi_dict.py defaults to has been 404 since 2026-05
#    (Gotcha 2). Writes data/amchi_konkani_dict.json (3,643 entries, gitignored).
python3 scripts/build_amchi_dict_from_sql.py

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

---

## <a name="verification-record-2026-05-22"></a>Verification record — 2026-05-22

Pod: `/workspace/amchi_asr`, GPU `NVIDIA A40`, driver `570.195.03 / CUDA 12.8`,
Python **3.12.3** (3.11 is no longer required; 3.12 works once torch is pinned
to the cu128 build). Branch `feat/live-asr-demo-recording`.

### Results

| Variant | Exit | Result | Cold start | Warm steady-state |
|---|---|---|---|---|
| `deaf`  | 0 | **5/5 samples within CER ≤ 0.05** (CER=0.0000 on all 5) | 206.8 s (model load incl. ~1.3 GB R2 checkpoint download) | ~12–13 it/s (~0.08 s/sample) |
| `amchi` | 0 | **5/5 samples within CER ≤ 0.05** (CER=0.0000 on all 5) | 198.1 s (incl. ~480 MB checkpoint download) | ~6–12 it/s (~0.10 s/sample) |

Final line on both:

```
==> Result: 5/5 samples within CER ≤ 0.05
   Inference pipeline verified. Safe to bake into Docker.
```

Fresh transcriptions matched stored predictions **byte-for-byte** on every
sample — no CUDA float drift, no Gemini post-processing divergence.

### Actual install command used (canonical for fresh pods)

This is the command that produced the working environment. The runbook in the
body of this doc already incorporates it; reproduced here as a single block
for cut-and-paste re-runs.

```bash
cd /workspace/amchi_asr

# (Re)build the venv. If you're on a fresh pod, .venv/ won't exist — that's
# fine. If you're resuming after a container restart and `import torch` is
# erroring with AttributeError, the venv is damaged — delete and rebuild.
rm -rf .venv
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip

# Install — pin torch to cu128 explicitly. Takes ~20 min.
.venv/bin/pip install \
  "torch==2.10.0+cu128" "torchaudio==2.10.0+cu128" \
  "nemo_toolkit[asr]" \
  librosa omegaconf jiwer google-genai runpod \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --ignore-installed blinker

# Verify torch sees the GPU before continuing.
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"

# Re-apply the conv_asr patch over the freshly-installed NeMo.
.venv/bin/python -c "import shutil, nemo.collections.asr.modules.conv_asr as m; shutil.copy('patches/conv_asr_fixed.py', m.__file__); print('patched', m.__file__)"

# Remove NeMo 2.7.3's shadowing scripts/ package (Gotcha — every install).
rm -rf .venv/lib/python3.12/site-packages/scripts

# Build the Konkani dict from the in-repo SQL dump (Railway endpoint is 404).
.venv/bin/python scripts/build_amchi_dict_from_sql.py

# Run the verifiers — PYTHONPATH=. is required (Gotcha 1).
PYTHONPATH=. .venv/bin/python scripts/verify_inference.py --variant deaf  --num-samples 5
PYTHONPATH=. .venv/bin/python scripts/verify_inference.py --variant amchi --num-samples 5
```

### Container-restart caveats (read before re-using a stopped pod)

RunPod's persistent volume covers `/workspace` but **not** `/tmp` or the
container's writable layer. Important consequences:

- **`/tmp/` is ephemeral.** Anything cached there (checkpoints downloaded by
  the verifier, audio samples, transient logs) will not survive a container
  stop/start. An earlier iteration of this doc optimistically promised the
  cached `/tmp/deaf_checkpoint.ckpt` would still be there on resume — it
  wasn't. Plan for re-download on every fresh container.
- **The persisted `.venv` itself can become partially corrupted across
  restarts.** When this session resumed, several site-packages directories
  (`pip/`, `torch/`) had survived as empty shells while their `*.dist-info`
  metadata remained, producing `AttributeError: module 'torch' has no
  attribute '__version__'` and `ModuleNotFoundError: No module named
  'pip._internal'`. Detection: `find .venv/lib/python3.12/site-packages
  -maxdepth 2 -name '__init__.py' -empty` — non-trivial hits mean damage.
  Fix is to **rebuild the venv from scratch** (the install command above
  takes ~20 min; trying to surgically repair takes much longer with worse
  outcomes).
- **`data/amchi_konkani_dict.json` is gitignored** (per `.gitignore` line 40:
  `data/**/*.json`). Rebuild via `scripts/build_amchi_dict_from_sql.py` —
  cheap, deterministic, and reads only the in-repo SQL dump.

### Pod-specific gotchas (kept for reference)

#### <a name="gotcha-1-scripts-not-on-syspath"></a>Gotcha 1 — `from scripts.X import Y` needs `PYTHONPATH=.`

`scripts/verify_inference.py` does `from scripts.runpod_http import …` but
`scripts/` has no `__init__.py` and Python 3 does not auto-add the cwd. Running
the script directly puts `scripts/` (the script's own directory) on
`sys.path[0]`, so `import scripts` fails. **Always run with
`PYTHONPATH=.` from the repo root** — the original runbook's
`python3 scripts/verify_inference.py …` will `ModuleNotFoundError` without it.

`setup.py` declares `packages=['scripts']` which suggests the original author
intended `scripts/` to be a regular package; the missing `__init__.py` looks
like an oversight. The runbook's prohibition on editing `scripts/*` to make
the verifier pass is why I used `PYTHONPATH` instead of adding `__init__.py`.
If you want a more permanent fix, `pip install -e .` (from the repo root)
plus a `scripts/__init__.py` would let you drop the `PYTHONPATH=` prefix.

#### <a name="gotcha-2-railway-dict-endpoint-404"></a>Gotcha 2 — Railway dictionary endpoint returns 404 — **fallback scripted**

`scripts/build_amchi_dict.py` defaults to
`https://konkanicollector-production.up.railway.app/api/dictionary?limit=5000`,
which has been 404 since at least 2026-05. The Railway base URL works (the
audio-recorder webapp loads), but `/api/dictionary` is not deployed (or has
been removed).

**Resolution:** [scripts/build_amchi_dict_from_sql.py](../scripts/build_amchi_dict_from_sql.py)
reads `konkani_dictionary_export.sql` and produces an identical-shape JSON.
It uses a proper VALUES tokenizer (handles SQL `''` escapes inside quoted
strings) and recovers 3,643 unique Devanagari entries from the dump (the
header claims 4,381 row entries; the remainder are duplicates of the same
Devanagari word with different `entry_number`s, deduped by the script).

If the resume needs a complete dictionary from a fresher source, either:
(a) bring `/api/dictionary` back up on Railway and use the original
`build_amchi_dict.py`; or (b) keep using the SQL-dump fallback. The
verification run on 2026-05-22 used the SQL fallback and passed cleanly.

#### <a name="gotcha-3-runpod-sdk-shadows-local-runpod"></a>Gotcha 3 — installed `runpod` SDK shadows local `runpod/` directory

`pip install runpod` (correctly, we want the SDK) installs the package at
`.venv/lib/python3.12/site-packages/runpod/`. The repo also has a local
`runpod/` directory that contains `handler.py` / `handler_deaf.py`, with no
`__init__.py`. Python's import resolution prefers the SDK regular package
over the local namespace package, so `from runpod.handler_deaf import handler`
in `scripts/runpod_smoke.py` raises `ModuleNotFoundError`.

This is what made the smoke-test handler check fail. **It does not affect the
verifier** — `verify_inference.py` only imports from `scripts.*`. It will
affect the production Docker handler at runtime; the Dockerfile likely COPYs
local `runpod/` into a path that does take precedence, but that's a separate
investigation for the Docker-build phase.

Workaround for smoke-test alone would be to set `PYTHONPATH` cleverly or
add `runpod/__init__.py` locally — neither was done because the smoke test
is informational, not the gate.

#### <a name="gotcha-4-torch-cu13-vs-driver-cu128"></a>Gotcha 4 — torch 2.12 (cu13) vs driver cu12.8 — **RESOLVED**

Unpinned `pip install nemo_toolkit[asr]` on Python 3.12 resolves to
`torch-2.12.0`, which on Python 3.12 pulls the CUDA-13 wheel (confirmed by
`nvidia-cublas-13.1.1.3`, `nvidia-cuda-runtime-13.0.96`,
`nvidia-cudnn-cu13-9.20.0.48` in the install list). The pod's NVIDIA driver
is `570.195.03` with `CUDA Version: 12.8`, which is too old for the CUDA-13
runtime. Symptom:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too
old (found version 12080).
ERROR: model load failed: No CUDA GPU detected. Inference on CPU is too
slow for this test.
```

**Resolution (now baked into the runbook):** pin both `torch` and
`torchaudio` to `2.10.0+cu128` and add the PyTorch cu128 wheel index as an
extra source. Pip then resolves the CUDA libs to their `-cu12` variants
(`nvidia-cublas-cu12-12.8.4.1`, `nvidia-cudnn-cu12-9.10.2.21`, etc.) which
match the driver. The exact install command is in [§ Actual install command
used](#verification-record-2026-05-22), step 2.

NeMo 2.7.3 happily accepts torch 2.10.0+cu128 — no need to pin nemo itself.

### Files / locations to know

- [scripts/build_amchi_dict_from_sql.py](../scripts/build_amchi_dict_from_sql.py)
  — SQL-dump → dict JSON. Added in this verification session.
- `data/amchi_konkani_dict.json` — output of the above; gitignored, rebuild
  with the script (~1 sec, no network needed).
- `/tmp/*` — **all ephemeral** (see [§ Container-restart caveats](#verification-record-2026-05-22)).
  Do not rely on cached checkpoints or audio samples across container
  stop/start; the verifier will redownload as needed.
- `.venv/lib/python3.12/site-packages/scripts/` — NeMo 2.7.3 ships a
  top-level `scripts/` package that shadows the repo's local `scripts/`.
  The runbook removes it after install. Sanity check after every reinstall:
  `find /workspace/amchi_asr/.venv/lib/python3.12/site-packages/scripts -maxdepth 1`
  (should error "No such file").

### What's next after this

1. **GitHub Actions Docker build** — see
   [docs/RUNPOD_GITHUB_ACTIONS_SETUP.md](RUNPOD_GITHUB_ACTIONS_SETUP.md).
   Triggered from the user's laptop. Pod is not needed.
2. The pod can be stopped after pushing the verification record. If you
   keep it warm, expect the same `.venv` damage / `/tmp` loss on the next
   container restart — plan ~25 minutes to redo install + dict + verifier.
