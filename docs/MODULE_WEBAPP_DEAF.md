# Module: Deaf Speech Demo Webapp

**Directory:** `webapp-deaf/`
**Framework:** Next.js 16 + TypeScript + Tailwind CSS (App Router, src-dir layout)
**Purpose:** Interactive demo of all four deaf speech ASR experiments

---

## What the app does

The webapp lets you browse all four deaf speech experiments (DS-A through DS-D), demo any individual recording, and see the full results. It is self-contained — no live inference endpoint is needed because all ASR transcriptions are pre-computed and stored as static JSON.

---

## Experiment overview

| ID   | Name                         | Speakers | Training data        | Encoder | Test WER |
|------|------------------------------|----------|----------------------|---------|----------|
| DS-C | Baseline                     | 1        | 124 samples, 50 ep   | Full FT | 75.3%    |
| DS-A | Frozen Encoder               | 1        | 124 samples, 100 ep  | Frozen  | 79.6%    |
| DS-B | Extended Data (Multi-Speaker)| Multiple | 188 samples, 100 ep  | Full FT | 93.1%    |
| DS-D | Speed Perturbation (BEST)    | 1        | 372 samples, 100 ep  | Full FT | 34.7%    |

The homepage lists all four experiments **worst → best** so the progression is immediately clear.

---

## URL structure

```
/                           Homepage — experiment index (worst to best)
/deaf-speech/[id]           Experiment detail page (id = dsc | dsa | dsb | dsd)
  → Demo tab                Interactive transcription demo
  → Results tab             WER stats, histogram, best/worst samples
```

---

## Demo tab — how it works

1. **Sample dropdown** — all 124 per-sample results, sorted best-first (lowest WER at top)
2. **Reference sentence** — shows the ground-truth Marathi sentence
3. **Audio player** — streams the original deaf speaker recording via `/api/audio/[id]`
4. **Transcribe button** — triggers a 1.5s simulated inference delay, then:
   - Shows the **raw ASR output** (with `⁇` tokens highlighted red)
   - Calls `/api/postprocess` (Gemini 2.5 Flash) to clean up the output
   - Shows the **post-processed output** with a mode badge (Filled / Reconstructed / Passthrough) and WER before → after
5. **Speak button** — sends the post-processed text to `/api/tts` (Google TTS, Marathi `mr-IN` voice) and plays it back

---

## API routes

| Route                  | Method | Purpose |
|------------------------|--------|---------|
| `/api/audio/[id]`      | GET    | Proxy to Railway recording (avoids browser CORS) |
| `/api/postprocess`     | POST   | Gemini 2.5 Flash: FILL/RECONSTRUCT/PASSTHROUGH on raw ASR |
| `/api/tts`             | POST   | Google TTS: synthesise Marathi text as MP3 |

### `/api/postprocess` logic

Counts `⁇` (undecodable) tokens in the raw prediction:
- **0 garbled tokens** → `PASSTHROUGH` (return as-is, no Gemini call)
- **≤40% garbled** → `FILL` (ask Gemini to fill the gaps)
- **>40% garbled** → `RECONSTRUCT` (ask Gemini to reconstruct from the legible fragments)

If Gemini is unavailable (no `GEMINI_API_KEY`), falls back to stripping `⁇` tokens.

---

## Static data files

All pre-computed ASR results live in `webapp-deaf/public/data/`:

| File                              | Contents |
|-----------------------------------|----------|
| `deaf_speech_results_dsc.json`    | DS-C (baseline) per-sample predictions + WER |
| `deaf_speech_results_dsa.json`    | DS-A per-sample predictions + WER |
| `deaf_speech_results_dsb.json`    | DS-B per-sample predictions + WER |
| `deaf_speech_results_dsd.json`    | DS-D per-sample predictions + WER |

Format per file:
```json
{
  "best_checkpoint": "...",
  "summary": { "total_samples": 124, "mean_wer": 0.347 },
  "per_sample": [
    { "audio": "data/deaf_speech/audio/130.wav", "reference": "...", "prediction": "...", "wer": 0.0 }
  ]
}
```

Audio IDs (the integer in the filename) map directly to Railway recording IDs.

---

## Key source files

```
webapp-deaf/
  src/
    app/
      page.tsx                      Homepage — experiment index
      deaf-speech/[id]/page.tsx     Experiment detail + Demo/Results tabs
      api/audio/[id]/route.ts       Audio proxy
      api/postprocess/route.ts      Gemini post-processing
      api/tts/route.ts              Google TTS (Marathi)
    lib/
      deaf-speech-experiments.ts    Experiment metadata + getResultsPath()
      deaf-speech-types.ts          TypeScript interfaces
      deaf-speech-utils.ts          WER helpers, distribution, statistics
  public/
    data/                           Pre-computed per-sample JSON results
```

---

## Running locally

```bash
cd webapp-deaf
npm install
npm run dev      # http://localhost:3000
```

### Required environment variables (`.env.local`)

```
GEMINI_API_KEY=...        # For /api/postprocess (Gemini 2.5 Flash)
GOOGLE_TTS_API_KEY=...    # For /api/tts (Google Cloud TTS)
```

Without `GEMINI_API_KEY`, the post-processing panel falls back to stripping `⁇` tokens.
Without `GOOGLE_TTS_API_KEY`, the Speak button returns a 503 error.

---

## Deployment (Railway)

- **Root directory:** `webapp-deaf`
- **Build command:** `npm run build`
- **Start command:** `npm run start`
- **Environment variables:** Set `GEMINI_API_KEY` and `GOOGLE_TTS_API_KEY` in the Railway service settings

> Note: if you previously had this deployed with root directory `webapp`, update it to `webapp-deaf` after the rename.

---

## Adding a new experiment

1. Run inference on the test set, produce a `final_test_results.json`
2. Copy it to `webapp-deaf/public/data/deaf_speech_results_<id>.json`
3. Add an entry to `DEAF_SPEECH_EXPERIMENTS` in `deaf-speech-experiments.ts`
4. Add the results path in `getResultsPath()`
5. Add a description block in `DESCRIPTIONS` in `deaf-speech/[id]/page.tsx`
6. Add a brief description in `BRIEF_DESCRIPTIONS` in `page.tsx`
