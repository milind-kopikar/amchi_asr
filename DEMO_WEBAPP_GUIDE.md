# Demo Web App — Build Guide for Claude Agent

**Read this first. This is a complete, self-contained spec for building the Phase 1 demo web app.**

---

## 0. Context

This is part of the "Amchi ASR" project: an ASR system fine-tuned to transcribe **deaf speech** (atypical Marathi speech patterns). The demo shows:

1. A user picks a pre-recorded audio sample from a deaf speaker.
2. They click "Transcribe".
3. The app shows **two outputs side-by-side**:
   - **Raw ASR**: what the speech model heard (often garbled — e.g. `ू किती ⁇`)
   - **AI-Corrected**: what Gemini post-processing inferred the speaker meant (e.g. `हे किती आहे?`)
4. (Optional, if time permits) A "Speak" button that calls Google TTS to read the corrected text aloud in Marathi.

**All inference is pre-computed.** There is no live model running. The results are in `demo_data/samples.json` (already in this repo). The app just reads that JSON and plays audio from a Railway API.

---

## 1. What you are building

**A single-page Next.js web app** that:

- Loads pre-computed transcription results from `demo_data/samples.json`
- Shows a list/dropdown of audio samples (sentence previews)
- Has an audio player so the user can hear the recording
- Has a "Transcribe" button that reveals the results with a short animated delay (~1.5s) to feel realistic
- Displays raw ASR vs AI-corrected text side-by-side
- Shows metadata: mode (FILL/RECONSTRUCT/SKIP), WER before/after
- Is mobile-friendly (the demo may be shown on a phone)
- Deploys to Railway via GitHub

---

## 2. Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Framework | **Next.js 14** (App Router) | Railway has a Next.js template; easy deployment |
| Styling | **Tailwind CSS** | Fast to style, looks professional |
| Language | **TypeScript** | Optional but recommended |
| Deployment | **Railway** (from GitHub) | User's existing $5/month plan |
| Audio | HTML5 `<audio>` tag | No dependency needed |
| Data | Static JSON bundled in `public/` | No backend needed |

---

## 3. Project structure

Create a `webapp/` subdirectory in this repo. Railway will be pointed at `webapp/` as the root directory.

```
webapp/
├── public/
│   └── samples.json          ← copy of demo_data/samples.json (see §4)
├── src/
│   └── app/
│       ├── layout.tsx         ← root layout with Tailwind + fonts
│       ├── page.tsx           ← main demo page (the whole app)
│       └── globals.css
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── next.config.ts
```

---

## 4. Data source

**Copy `demo_data/samples.json` from the repo root into `webapp/public/samples.json`.**

The JSON structure is:

```json
{
  "meta": {
    "total_samples": 124,
    "good_demo_samples": 48,
    "story": "दैनंदिन कामे १ (Daily Activities 1)",
    "model": "AI4Bharat IndicConformer fine-tuned on deaf speech (epoch 21, val_WER=72%)",
    "postprocessing": "Gemini 2.5 Flash (FILL/RECONSTRUCT modes)",
    "mode_legend": {
      "FILL": "Gemini filled in garbled slots using anchor words",
      "RECONSTRUCT": "Gemini reconstructed sentence from phonetic fragments",
      "FILL_REVERTED": "Gemini fill attempted but reverted (safety valve prevented regression)",
      "SKIP": "Perfect transcription, no post-processing needed"
    }
  },
  "samples": [
    {
      "id": 130,
      "audio_url": "https://deafspeechcollector-production.up.railway.app/api/recordings/130/audio",
      "reference": "दैनंदिन कामे १।",
      "raw_asr": "पि ⁇",
      "corrected": "पाणी पिऊ का?",
      "mode": "RECONSTRUCT",
      "wer_before": 1.0,
      "wer_after": 1.0,
      "is_good_demo": true
    },
    {
      "id": 131,
      "audio_url": "https://deafspeechcollector-production.up.railway.app/api/recordings/131/audio",
      "reference": "दूध किती आहे?।",
      "raw_asr": "ू किती ⁇",
      "corrected": "हे किती आहे?",
      "mode": "FILL",
      "wer_before": 0.6667,
      "wer_after": 0.3333,
      "is_good_demo": true
    }
    // ... 122 more samples
  ]
}
```

**Key fields:**
- `audio_url` — direct URL to stream the WAV from the Railway deaf speech collector API. Use this as the `src` of an `<audio>` element.
- `raw_asr` — what the model transcribed (may contain `⁇` garble markers)
- `corrected` — Gemini post-processed output (natural Marathi sentence)
- `reference` — the correct intended sentence (what the speaker was saying)
- `mode` — how post-processing worked: FILL, RECONSTRUCT, FILL_REVERTED, or SKIP
- `is_good_demo` — `true` for samples where corrected output is meaningful (48 of 124)

**CORS note:** The Railway deaf speech collector API (`deafspeechcollector-production.up.railway.app`) may not have CORS headers for cross-origin audio requests. If the `<audio>` tag fails to load audio in the browser, see §8 (CORS Proxy).

---

## 5. UI Design

### Layout (mobile-first)

```
┌─────────────────────────────────────────┐
│  🎙️ Deaf Speech Transcriber            │
│  दैनंदिन कामे – Daily Activities       │
├─────────────────────────────────────────┤
│                                         │
│  [ Show all (124) / Good demos (48) ]   │  ← toggle filter
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Select a recording:              │  │
│  │  [▼ दूध किती आहे? — #131      ]  │  │  ← dropdown
│  └───────────────────────────────────┘  │
│                                         │
│  ▶ ────────────────── 0:06              │  ← audio player
│                                         │
│  [ Transcribe ]                         │  ← big button
│                                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─  │
│                                         │
│  🔴 Raw ASR Output                      │
│  ┌───────────────────────────────────┐  │
│  │  ू किती ⁇                         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ✅ AI-Corrected Output                 │
│  ┌───────────────────────────────────┐  │
│  │  हे किती आहे?                      │  │
│  │                                   │  │
│  │  Mode: FILL   WER: 67%→33%        │  │
│  └───────────────────────────────────┘  │
│                                         │
│  📖 Reference: दूध किती आहे?           │  ← small, below results
│                                         │
│  [ 🔊 Speak Corrected Text ]            │  ← future TTS button (disabled for now)
│                                         │
└─────────────────────────────────────────┘
```

### Behavior

1. **On load**: fetch `/samples.json`, show dropdown with all samples. Default filter to "Good demos (48)" since those show the best transformation.
2. **Dropdown label**: use the `reference` sentence as the label (e.g. "दूध किती आहे? — #131"), so the user knows what they're picking.
3. **Audio player**: standard HTML5 `<audio controls>` with `src={sample.audio_url}`. Loads immediately when sample is selected.
4. **Transcribe button**: when clicked:
   - Disable the button and show a spinner/progress bar
   - After ~1.5 seconds (simulated delay), reveal the results
   - Results animate in (simple fade-in)
5. **Raw ASR display**: show `raw_asr` text. Highlight `⁇` markers in red/orange so they stand out as "garble indicators".
6. **AI-Corrected display**: show `corrected` text in a green-tinted card. Show mode badge and WER change.
7. **Reference**: show below results in smaller text as ground truth.
8. **TTS button**: include a "🔊 Speak" button but **leave it disabled/greyed out** with a tooltip "Coming soon". This way it's in the UI for Phase 2 without being functional yet.

### Mode badge colors
- `FILL` → blue badge ("Anchor-guided correction")
- `RECONSTRUCT` → orange badge ("Full reconstruction")
- `FILL_REVERTED` → grey badge ("Post-processing skipped (safety valve)")
- `SKIP` → green badge ("Perfect transcription")

---

## 6. Implementation notes

### Fetching samples.json

```typescript
// In page.tsx (or a server component)
import samplesData from '../../public/samples.json'
// OR fetch at runtime:
const res = await fetch('/samples.json')
const data = await res.json()
```

Prefer importing directly as a module (`import samplesData`) since the file is small (~150KB) and it avoids a runtime fetch.

### Audio streaming

```html
<audio controls src={selectedSample.audio_url} />
```

The `audio_url` in the JSON is already the correct URL. No transformation needed.

If CORS is an issue (see §8), proxy it through a Next.js API route.

### Simulated transcription delay

```typescript
const [isTranscribing, setIsTranscribing] = useState(false)
const [result, setResult] = useState(null)

const handleTranscribe = () => {
  setIsTranscribing(true)
  setResult(null)
  setTimeout(() => {
    setIsTranscribing(false)
    setResult(selectedSample)
  }, 1500)
}
```

### Highlighting ⁇ in raw ASR

```typescript
function formatRawAsr(text: string) {
  // Split on ⁇ and wrap in a red span
  return text.split('⁇').map((part, i) =>
    i === 0 ? part : <><span className="text-red-500 font-bold">⁇</span>{part}</>
  )
}
```

### WER display

Show WER as percentage and indicate improvement:
- `wer_before=0.6667` → "67%"
- `wer_after=0.3333` → "33%"
- If `wer_after < wer_before` → show green arrow "↓ improved"
- If equal → show grey "─ unchanged"

---

## 7. Railway Deployment

### Step 1: Initialize Next.js project

```bash
cd /path/to/amchi_asr    # the repo root
npx create-next-app@latest webapp --typescript --tailwind --app --no-src-dir --import-alias "@/*"
# When prompted:
#   Would you like to use ESLint? → Yes
#   Would you like to customize the default import alias? → No
```

Wait — create it WITH src dir (more organized):
```bash
npx create-next-app@latest webapp --typescript --tailwind --app --src-dir --import-alias "@/*"
```

### Step 2: Copy data file

```bash
cp demo_data/samples.json webapp/public/samples.json
```

### Step 3: Build and run locally

```bash
cd webapp
npm run dev
# Visit http://localhost:3000
```

### Step 4: Deploy to Railway

1. Push the repo (with `webapp/` directory) to GitHub.
2. In [Railway](https://railway.app) → New Project → Deploy from GitHub repo → select `milind-kopikar/amchi_asr`.
3. Railway will auto-detect Next.js. **Set the root directory to `webapp`** in the Railway service settings (Settings → Source → Root Directory → `webapp`).
4. Railway will run `npm run build` and `npm start` automatically.
5. Railway will assign a public URL (e.g. `https://amchi-asr-demo.up.railway.app`).

### Environment variables for Railway

None needed for Phase 1 (all data is static). For Phase 2 (live inference) or TTS, you'll add:
- `NEXT_PUBLIC_RUNPOD_ENDPOINT_URL` — RunPod serverless endpoint URL
- `NEXT_PUBLIC_GOOGLE_TTS_API_KEY` — Google Cloud TTS API key (use with caution; for demo only)

---

## 8. CORS Proxy (if needed)

If the `<audio>` tag cannot load audio from `deafspeechcollector-production.up.railway.app` due to CORS, add a simple proxy API route in Next.js:

```typescript
// webapp/src/app/api/audio/[id]/route.ts
export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const upstream = `https://deafspeechcollector-production.up.railway.app/api/recordings/${params.id}/audio`
  const response = await fetch(upstream)
  const audioBuffer = await response.arrayBuffer()
  return new Response(audioBuffer, {
    headers: {
      'Content-Type': 'audio/wav',
      'Cache-Control': 'public, max-age=86400',
    },
  })
}
```

Then change `audio_url` usage in the frontend from the Railway URL to `/api/audio/{id}`.

**Only add this proxy if CORS actually fails.** Try without it first.

---

## 9. Phase 2 Extension: Live Inference (do NOT build now)

When Phase 2 is ready, the "Transcribe" button will make a real API call instead of using pre-computed data:

```typescript
// Phase 2: replace the setTimeout with this
const handleTranscribe = async () => {
  setIsTranscribing(true)

  // Fetch audio bytes, encode to base64
  const audioResponse = await fetch(selectedSample.audio_url)
  const audioBuffer = await audioResponse.arrayBuffer()
  const base64 = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)))

  // Call RunPod serverless endpoint
  const result = await fetch(process.env.NEXT_PUBLIC_RUNPOD_ENDPOINT_URL!, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${process.env.NEXT_PUBLIC_RUNPOD_API_KEY}`
    },
    body: JSON.stringify({ input: { audio_base64: base64 } })
  })
  const data = await result.json()
  setResult({ ...selectedSample, raw_asr: data.output.raw, corrected: data.output.corrected })
  setIsTranscribing(false)
}
```

The UI does not need to change for Phase 2 — only the data source changes.

---

## 10. Phase 2 Extension: Google TTS (do NOT build now)

When ready, the "🔊 Speak" button calls Google Cloud TTS with Marathi (`mr-IN`):

```typescript
const speakText = async (text: string) => {
  const response = await fetch(
    `https://texttospeech.googleapis.com/v1/text:synthesize?key=${process.env.NEXT_PUBLIC_GOOGLE_TTS_API_KEY}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        input: { text },
        voice: { languageCode: 'mr-IN', ssmlGender: 'FEMALE' },
        audioConfig: { audioEncoding: 'MP3' },
      }),
    }
  )
  const { audioContent } = await response.json()
  const audio = new Audio(`data:audio/mp3;base64,${audioContent}`)
  audio.play()
}
```

---

## 11. Acceptance criteria (what "done" looks like)

- [ ] `webapp/` directory exists with a working Next.js app
- [ ] App loads 124 samples from `/samples.json`
- [ ] Default view shows "Good demos" filter (48 samples)
- [ ] Dropdown shows reference sentence + ID for each sample
- [ ] Selecting a sample loads its audio in the player
- [ ] Clicking "Transcribe" shows a ~1.5s spinner, then reveals results
- [ ] Raw ASR box: shows `raw_asr` with `⁇` highlighted in red
- [ ] AI-Corrected box: shows `corrected` with mode badge + WER change
- [ ] Reference text shown below results
- [ ] "🔊 Speak" button present but disabled (grey, "Coming soon")
- [ ] Works on mobile (375px width minimum)
- [ ] `npm run build` succeeds with no errors
- [ ] Deployed to Railway, accessible at a public URL

---

## 12. Key files in this repo (for context)

| File | What it contains |
|------|-----------------|
| `demo_data/samples.json` | 124 pre-computed samples — copy to `webapp/public/samples.json` |
| `nemo_experiments/deaf_speech_story4_50epoch/experiments/20260301_003725/postprocessed_results.json` | Original source for samples.json (full detail including word_labels) |
| `data/deaf_speech/test/manifest.jsonl` | Original test manifest with reference sentences and audio paths |
| `AGENT_HANDOFF.md` | Full project context (model, training, post-processing details) |
| `scripts/postprocess_asr.py` | Gemini post-processing module (FILL/RECONSTRUCT logic) |

---

## 13. Notes for the demo

- **Best samples to highlight**: IDs with `mode=FILL` where the corrected output reads naturally. E.g. ID 131 (`ू किती ⁇` → `हे किती आहे?`), ID 135 (`प कितीी येईल ⁇` → `बस लवकर येईल का`).
- **The ⁇ symbol** is NeMo's way of saying "I cannot decode this token" — it's a deliberate garble marker, not a display bug.
- **WER metric caveat**: WER requires exact word matches, so a corrected sentence that is semantically equivalent but uses different words still shows WER=1.0. The human readability improvement is the real story.
- **Story context**: All 124 samples are deaf speakers attempting to say sentences from "दैनंदिन कामे १" (Daily Activities 1) — everyday phrases like asking for milk price, bus arrival time, shop directions.
