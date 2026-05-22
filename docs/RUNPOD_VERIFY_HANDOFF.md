# Handoff: verify inference on a RunPod GPU pod

**Read this before doing anything.** This is a short, scoped task. Do not start a
broader refactor — the next thing after this passes is the GitHub Actions Docker
build (see [docs/RUNPOD_GITHUB_ACTIONS_SETUP.md](RUNPOD_GITHUB_ACTIONS_SETUP.md)).

## ⚠️ Resume note — 2026-05-22 mid-session

A previous Claude session got the pod through env setup and as far as launching
the deaf verifier (it was at "Loading model" when the user stepped away). See
[§ Resume state](#resume-state-2026-05-22) at the bottom of this doc for the exact
state, the four pod-specific gotchas that were already worked around, and the
two-command resume sequence. **Read that section first** — it will save you from
re-discovering those gotchas.

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

---

## <a name="resume-state-2026-05-22"></a>Resume state — 2026-05-22

Pod: `/workspace/amchi_asr`, GPU `NVIDIA A40`, driver `570.195.03 / CUDA 12.8`,
Python **3.12.3** (not 3.11 — the doc above expected 3.11; nothing else changed).
Branch `feat/live-asr-demo-recording`, commit `51d05a1`. Clean working tree.

### What's already done

| Step | State | Notes |
|---|---|---|
| `git clone` | ✅ pre-existing | Repo already at `/workspace/amchi_asr`. |
| `.venv` created on Python 3.12 | ✅ | At `.venv/`. Activate with `source .venv/bin/activate`. |
| `pip install "nemo_toolkit[asr]" librosa omegaconf jiwer google-genai runpod --ignore-installed blinker` | ✅ | **No ResolutionImpossible** on Python 3.12. Resolver landed on: `nemo_toolkit-2.7.3`, `torch-2.12.0`, `numba-0.65.1`, `lightning-2.4.0`, `pytorch-lightning-2.6.4`, `numpy-2.4.6`. `numba-cuda` was **not** pulled in. |
| `conv_asr_fixed.py` patch | ✅ | Copied over `.venv/lib/python3.12/site-packages/nemo/collections/asr/modules/conv_asr.py`. |
| Konkani dictionary build | ✅ via SQL workaround | See [§ Gotcha 2](#gotcha-2-railway-dict-endpoint-404) — Railway endpoint returned 404; built `data/amchi_konkani_dict.json` (3,643 entries) from the in-repo `konkani_dictionary_export.sql` dump. |
| `runpod_smoke.py --variant deaf` | ⚠️ 3/4 PASS | imports / patch / dictionary PASS. **Handler check FAILS** because the pip-installed `runpod` SDK shadows the local `runpod/` directory ([§ Gotcha 3](#gotcha-3-runpod-sdk-shadows-local-runpod)). This does **not** affect the verifier — verifier only imports from `scripts.*`. Smoke check for amchi was not run; same gotcha will apply. |
| `verify_inference.py --variant deaf` | ❌ **EXIT 2** | Got past checkpoint download (~1.3 GB downloaded to `/tmp/deaf_checkpoint.ckpt`), audio download, then **failed at model load** with `No CUDA GPU detected` — torch 2.12.0 is compiled against CUDA 13 but the pod driver is CUDA 12.8. This is the current blocker — see [§ Gotcha 4](#gotcha-4-torch-cu13-vs-driver-cu128). |
| `verify_inference.py --variant amchi` | ⏸️ not run | Blocked by Gotcha 4. |

### Resume commands (after fixing Gotcha 4)

```bash
cd /workspace/amchi_asr
source .venv/bin/activate

# Run from the repo root with PYTHONPATH set — required by Gotcha 1.
PYTHONPATH=. python3 scripts/verify_inference.py --variant deaf  --num-samples 5
PYTHONPATH=. python3 scripts/verify_inference.py --variant amchi --num-samples 5
```

The checkpoint is already cached at `/tmp/deaf_checkpoint.ckpt`; the verifier
will reuse it on the next run.

### Pod-specific gotchas (already worked around — read before doing anything)

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

#### <a name="gotcha-2-railway-dict-endpoint-404"></a>Gotcha 2 — Railway dictionary endpoint returns 404

`scripts/build_amchi_dict.py` defaults to
`https://konkanicollector-production.up.railway.app/api/dictionary?limit=5000`,
which currently 404s. The Railway base URL works (the audio-recorder webapp
loads), but `/api/dictionary` is not deployed (or has been removed).

Workaround used: extracted the Devanagari column from the in-repo SQL dump
`konkani_dictionary_export.sql` (which the dump's header claims has 4,381
entries — the simple regex extractor recovered 3,643 unique rows, skipping
~700 that contained embedded single-quotes the regex didn't escape). The
output JSON lives at `data/amchi_konkani_dict.json`.

If the resume needs a complete dictionary, either: (a) bring `/api/dictionary`
back up on Railway; or (b) parse the SQL dump with a proper tokenizer rather
than the regex shortcut used here.

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

#### <a name="gotcha-4-torch-cu13-vs-driver-cu128"></a>Gotcha 4 — **CURRENT BLOCKER**: torch 2.12 (cu13) vs driver cu12.8

`pip install nemo_toolkit[asr]` (unpinned) resolved to `torch-2.12.0`, which
on Python 3.12 pulls in the CUDA-13 wheel — confirmed by the
`nvidia-cublas-13.1.1.3`, `nvidia-cuda-runtime-13.0.96`,
`nvidia-cudnn-cu13-9.20.0.48` packages in the install list. The pod's NVIDIA
driver is `570.195.03` with `CUDA Version: 12.8`, which is too old for the
CUDA-13 runtime. PyTorch logs:

```
UserWarning: CUDA initialization: The NVIDIA driver on your system is too
old (found version 12080).
ERROR: model load failed: No CUDA GPU detected. Inference on CPU is too
slow for this test.
```

I did **not** auto-pin torch and re-install, because:

1. The user's standing instruction was "if pip install nemo_toolkit[asr]
   fails with ResolutionImpossible or any package fails to build from
   source, stop and report back." Pip succeeded; nothing built from source.
   But the resulting torch is unusable on this driver, which is a related
   class of failure I want explicit approval to fix.
2. The handoff doc itself recommends `torch 2.10.0+cu128`, which would
   resolve this. The likely fix is one of:
   - `pip install --upgrade-strategy only-if-needed torch==2.10.0+cu128 torchaudio==2.10.0+cu128 --index-url https://download.pytorch.org/whl/cu128`
   - Or upgrade the pod's NVIDIA driver / pick a pod template with a
     newer driver (570 → 580+).
   - Or `pip install nemo_toolkit[asr]==2.7.0` (note the doc's
     suggestion to pin exactly 2.7.0 if 2.7.3 misbehaves), which **may**
     bring in an older torch as a side effect, though that's speculative.

**Recommended next action when resuming:** explicitly install
`torch==2.10.0+cu128` (or whichever torch the training environment used)
into the existing venv from the PyTorch cu128 wheel index, then retry the
verifier. Do not delete the venv — the rest of the dep tree is fine, and
the conv_asr patch will need to be re-applied if `nemo_toolkit` is touched.

### Files / locations to know

- `data/amchi_konkani_dict.json` — built dict (3,643 entries from SQL dump).
- `/tmp/deaf_checkpoint.ckpt` — cached deaf checkpoint (1.3 GB), safe to reuse.
- `/tmp/deaf_audio/` — 5 cached test audio samples, safe to reuse.
- `/tmp/verify_deaf.log` — last verifier run's full stdout/stderr.
- `.venv/lib/python3.12/site-packages/scripts/` — **removed**; NeMo 2.7.3 ships a top-level `scripts/` package that shadowed the repo's local `scripts/`. If you reinstall `nemo_toolkit`, you'll need to `rm -rf` it again. (Filed under "NeMo packaging bug" — `find /workspace/amchi_asr/.venv/lib/python3.12/site-packages/scripts -maxdepth 1` to confirm it's gone, or to find it again after a reinstall.)

### Things explicitly NOT done

- Did not edit any file in `scripts/`, `runpod/`, or `patches/` — per the
  handoff's no-edit rule.
- Did not pin `lightning==2.4.0` or `numba-cuda` — per the user's standing
  rule for Python 3.12 (the Colab-only fixes).
- Did not retry / pin torch to fix Gotcha 4 — waiting for user okay.
- Did not run the amchi verifier — blocked by Gotcha 4.
- Did not push, build Docker, or stop the pod.
