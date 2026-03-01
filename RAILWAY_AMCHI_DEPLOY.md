# Railway Deployment Guide — Amchi Konkani ASR Demo

**Read this first.** Complete, self-contained guide for deploying `webapp-amchi/`
to Railway as a separate service from the deaf speech demo.

---

## What you are deploying

A Next.js app (`webapp-amchi/`) that:
- Shows 29 pre-computed Amchi Konkani test samples from `public/samples.json`
- Proxies audio from the Konkani Collector via `/api/audio/[id]`
- Translates Amchi Konkani → English via Gemini, then speaks via Google TTS
- Requires **two** environment variables: `GOOGLE_TTS_API_KEY` and `GEMINI_API_KEY`

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Railway account | Free tier works; $5/month plan has more resources |
| GitHub repo pushed | `milind-kopikar/amchi_asr` — `webapp-amchi/` must be present |
| Google Cloud TTS API key | Cloud Text-to-Speech API enabled in Google Cloud Console |
| Gemini API key | From https://aistudio.google.com/apikey (for translation) |

---

## Step 1 — Push the latest code to GitHub

```bash
cd /path/to/konkani_asr
git push origin master
```

Confirm `webapp-amchi/` is visible in the GitHub repo.

---

## Step 2 — Create a new Railway service

1. Go to [railway.app](https://railway.app) and log in.
2. Open your existing project (or create a new one).
3. Click **+ New** → **GitHub Repo** → select **milind-kopikar/amchi_asr**.
4. **Do not deploy yet** — set the root directory first.

> This will be a **second** Railway service in the same project,
> separate from the deaf speech demo service.

---

## Step 3 — Set root directory to `webapp-amchi`

1. In the new service → **Settings** tab.
2. Under **Source** → **Root Directory**, type: `webapp-amchi`
3. Click **Save**.

Railway will run `npm install`, `npm run build`, and `npm start` from `webapp-amchi/`.

---

## Step 4 — Add environment variables

In the service → **Variables** tab, add:

```
GOOGLE_TTS_API_KEY = <your Google Cloud TTS API key>
GEMINI_API_KEY     = <your Gemini API key>
```

- `GOOGLE_TTS_API_KEY`: Must have **Cloud Text-to-Speech API** enabled.
  Get/verify at https://console.cloud.google.com/apis/credentials
- `GEMINI_API_KEY`: From https://aistudio.google.com/apikey
  Used for Amchi Konkani → English translation (Gemini 2.5 Flash).

> **Security:** Both keys are server-side only. Never add as `NEXT_PUBLIC_` variables.

---

## Step 5 — Deploy

1. Go to **Deployments** tab → click **Deploy**.
2. Railway runs:
   ```
   npm install
   npm run build   # Next.js production build
   npm start       # next start on PORT provided by Railway
   ```
3. Once green, Railway assigns a URL such as:
   `https://amchi-konkani-demo.up.railway.app`

---

## Step 6 — Verify

Open the Railway URL and check:

- [ ] Page loads with the sample dropdown (29 samples with WER ≤ 75%)
- [ ] Reference sentence shows in Devanagari + Roman alphabet
- [ ] Audio player loads and plays correctly
- [ ] Clicking **Transcribe** shows Raw ASR at ~1.5s, Post-processed at ~2.0s
- [ ] Clicking **🔊 Speak in English** shows English translation and plays audio
- [ ] No errors in Railway deployment logs

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Build fails with `ENOENT` | Root directory not set to `webapp-amchi` | Set Root Directory → `webapp-amchi` |
| Audio player shows error | CORS from Konkani Collector | Audio is proxied via `/api/audio/[id]` — check Railway logs |
| Speak button shows "Translation not configured" | `GEMINI_API_KEY` missing | Add in Variables tab |
| Speak button shows "TTS not configured" | `GOOGLE_TTS_API_KEY` missing | Add in Variables tab |
| Speak button shows "Translation failed: 400" | Gemini API key invalid | Verify key at aistudio.google.com |

---

## Re-deploying after code changes

Every `git push origin master` triggers an automatic redeploy (if auto-deploy enabled).

To update samples with new post-processing results:
```bash
python3 scripts/amchi_postprocess_asr.py --input ... --output ...
python3 scripts/generate_amchi_samples.py --api_key $GEMINI_API_KEY
git add webapp-amchi/public/samples.json
git commit -m "Update Amchi Konkani demo samples"
git push origin master
```

---

## Two-service summary

| Service | Root Directory | URL | Env Vars Needed |
|---------|---------------|-----|-----------------|
| Deaf Speech Demo | `webapp` | `amchi-asr-demo.up.railway.app` | `GOOGLE_TTS_API_KEY` |
| Amchi Konkani Demo | `webapp-amchi` | `amchi-konkani-demo.up.railway.app` | `GOOGLE_TTS_API_KEY` + `GEMINI_API_KEY` |

Both services deploy from the same GitHub repo (`milind-kopikar/amchi_asr`),
from different root directories.
