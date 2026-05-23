# GitHub Actions setup for RunPod image builds

The `.github/workflows/build-runpod.yml` workflow builds both Docker images
(amchi + deaf) in parallel on GitHub Actions runners and pushes them to
Docker Hub. This runs on every push to `master` or any `feat/*` branch that
touches a relevant file.

One-time setup before the workflow can push successfully:

## 1. Create a Docker Hub access token

1. Log in to https://hub.docker.com → Account Settings → **Security** →
   **New Access Token**.
2. Description: `github-actions-amchi-asr` (or similar so you can revoke
   it later without touching other PATs).
3. Permissions: **Read, Write, Delete**. The Delete permission is needed
   only if you ever want the workflow to clean up old image tags;
   Read+Write is sufficient for the current workflow.
4. **Copy the generated token now** — Docker Hub will not show it again.

## 2. Create the two Docker Hub repos (optional but recommended)

You can let the first push auto-create the repos, but they default to
**public**. Pre-creating them lets you mark them private if you prefer.

- `milindkopigithub/amchi-asr-runpod` — RunPod worker for the Amchi Konkani ASR
- `milindkopigithub/deaf-speech-asr-runpod` — RunPod worker for the Deaf Speech ASR

Either way, the workflow pushes images tagged with the commit SHA and (on
master) with `:latest`.

## 3. Add the two repository secrets

GitHub repo (`milind-kopikar/amchi_asr`) → **Settings** → **Secrets and
variables** → **Actions** → **New repository secret**.

| Secret name           | Value                                                              |
|-----------------------|--------------------------------------------------------------------|
| `DOCKERHUB_USERNAME`  | Your Docker Hub username — typically `milindkopi`                  |
| `DOCKERHUB_TOKEN`     | The PAT you just created in step 1 (starts with `dckr_pat_`)       |

Once both secrets exist, the workflow can authenticate to Docker Hub on
the next push.

## 4. Trigger the first build

After the secrets are in place, push any commit that touches one of the
workflow's trigger paths (`runpod/**`, `scripts/runpod_smoke.py`, etc.).
The simplest way to trigger a first build manually:

1. Repo → **Actions** → **Build RunPod ASR images**
2. **Run workflow** → pick `master` or your feature branch → **Run**

The workflow will:

1. Run the Python unit tests (3 seconds)
2. In parallel, build both Docker images
3. Push both images on success

Watch the **deaf** job for the canary — the build log includes the staged
smoke checks at the end:

```
[1/3] imports     PASS
[2/3] patch       PASS
[3/3] handler     PASS
```

If any check fails, the build stops at that RUN line and you can read the
failure inline.

## 5. After both images are pushed

Per the rollout summary that the workflow emits:

1. Create / update the **deaf** RunPod endpoint pointing at
   `milindkopigithub/deaf-speech-asr-runpod:<sha-or-latest>`
2. Verify with `scripts/test_deaf_endpoint.py`
3. Verify in the webapp at `/demo/live`
4. *Then* create / update the **amchi** endpoint with
   `milindkopigithub/amchi-asr-runpod:<sha-or-latest>`

Don't do both endpoints at once — if both are broken in the same way
(e.g. shared NeMo install regression), the canary approach saves you
debugging two endpoints in parallel.

## Troubleshooting

### "denied: requested access to the resource is denied"

The Docker Hub repo doesn't exist OR the PAT lacks Write permission.
Recheck step 1 (PAT scopes) and step 2 (repo existence).

### "[N/M] <check> FAIL: ..."

The Dockerfile's staged smoke check failed. The Action log includes the
exact reason — e.g. `dictionary FAIL: /app/data/amchi_konkani_dict.json
not found` means the dict-building step in the workflow didn't run or
failed silently. Re-check the workflow logs for the `Build Amchi Konkani
dictionary JSON` step.

### Workflow does not trigger on push

The push must change one of the watched paths (`runpod/**`,
`scripts/runpod_smoke.py`, etc.) OR be on `master`. Use **Run workflow**
from the Actions tab to force a build on any branch.

### "huggingface-hub" version conflict in unit tests

Pre-existing issue in `tests/test_sample_logger.py` (unrelated to this
workflow). The workflow already excludes it; just leave it as-is.
