# Railway Deployment Guide — Demo Web App

**Read this first.** This is a complete, self-contained guide for deploying the
`webapp/` Next.js demo app to Railway.

---

## What you are deploying

A Next.js 16 app that:
- Shows pre-computed deaf speech transcription samples from `webapp/public/samples.json`
- Proxies audio from the Railway deaf speech collector via `/api/audio/[id]`
- Calls Google Cloud Text-to-Speech via a server-side route `/api/tts`
- Requires **one** environment variable: `GOOGLE_TTS_API_KEY`

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Railway account | Free tier works; $5/month plan has more resources |
| GitHub repo pushed | `milind-kopikar/amchi_asr` — the `webapp/` directory must be present |
| Google Cloud TTS API key | Created in Google Cloud Console with Cloud Text-to-Speech API enabled |

---

## Step 1 — Push the latest code to GitHub

If not already done:

```bash
cd /path/to/konkani_asr
git push origin master
```

Confirm `webapp/` is visible in the GitHub repo before proceeding.

---

## Step 2 — Create the Railway service

1. Go to [railway.app](https://railway.app) and log in.
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select **milind-kopikar/amchi_asr**.
4. Railway will detect the repo. **Do not deploy yet** — set the root directory first (Step 3).

---

## Step 3 — Set root directory to `webapp`

Railway must be told to build from `webapp/` not the repo root:

1. After the service is created, go to the service → **Settings** tab.
2. Under **Source** → **Root Directory**, type: `webapp`
3. Click **Save**.

Railway will now run `npm run build` and `npm start` from inside `webapp/`.

---

## Step 4 — Add the environment variable

1. In the service, go to the **Variables** tab.
2. Click **New Variable** and add:

```
GOOGLE_TTS_API_KEY = <your Google Cloud TTS API key>
```

The key must have the **Cloud Text-to-Speech API** enabled in Google Cloud Console.
To get/verify the key:
- Go to [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
- Confirm "Cloud Text-to-Speech API" is enabled for the project
- Use the API key from that project

> **Security:** This key is server-side only. It is never exposed to the browser.
> Do NOT add it as a `NEXT_PUBLIC_` variable.

---

## Step 5 — Deploy

1. Go to the **Deployments** tab and click **Deploy** (or push a new commit to
   trigger an automatic redeploy).
2. Railway runs:
   ```
   npm run build   # Next.js production build
   npm start       # next start on PORT provided by Railway
   ```
3. Once the deployment is green, Railway assigns a public URL such as:
   `https://amchi-asr-demo.up.railway.app`

---

## Step 6 — Verify

Open the Railway URL and check:

- [ ] Page loads with the sample dropdown (should show samples with AI-corrected WER ≤ 75%)
- [ ] Selecting a sample loads the audio player
- [ ] Clicking **Transcribe** shows Raw ASR at ~1.5 s, AI-Corrected at ~2.0 s
- [ ] Clicking **🔊 Speak Corrected Text** plays a male Marathi voice
- [ ] No errors in the Railway deployment logs

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Build fails with `ENOENT` | Root directory not set to `webapp` | Set Root Directory → `webapp` in Settings |
| Audio player loads but shows error | CORS from Railway deaf speech collector | Audio is already proxied via `/api/audio/[id]` — check the Railway logs |
| Speak button shows "TTS not configured" | `GOOGLE_TTS_API_KEY` env var missing | Add it in Variables tab |
| Speak button shows "TTS upstream error: 403" | API key doesn't have TTS API enabled | Enable Cloud Text-to-Speech API in Google Cloud Console for this key's project |
| Speak button shows "TTS upstream error: 400" | Text is empty or malformed | Check that `corrected` field in samples.json is non-empty |

---

## Re-deploying after code changes

Every `git push origin master` to the GitHub repo triggers an automatic
redeploy on Railway (if auto-deploy is enabled in Settings → Source).

To update `webapp/public/samples.json` with new samples:
```bash
cp demo_data/samples.json webapp/public/samples.json
git add webapp/public/samples.json
git commit -m "Update demo samples"
git push origin master
```

---

## Phase 2 extension (live RunPod inference)

When the RunPod serverless endpoint is ready (see `RUNPOD_SERVERLESS_DEAF.md`),
add these additional Railway environment variables:

```
NEXT_PUBLIC_RUNPOD_ENDPOINT_URL = https://api.runpod.ai/v2/<endpoint-id>/runsync
NEXT_PUBLIC_RUNPOD_API_KEY      = <your RunPod API key>
```

The `handleTranscribe` function in `webapp/src/app/page.tsx` will need to be
updated to call the RunPod endpoint instead of using the pre-computed JSON.
See `DEMO_WEBAPP_GUIDE.md` §9 for the Phase 2 code snippet.
