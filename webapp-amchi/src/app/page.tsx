"use client";

import Link from "next/link";
import { EXPERIMENTS } from "@/lib/experiments";

// ─── Brief descriptions per experiment ────────────────────────────────────────

const BRIEF_DESCRIPTIONS: Record<string, string> = {
  baseline:
    "Full fine-tune of IndicConformer (115M params) for 50 epochs on story-based train/test split. " +
    "3 speakers in test set. Sets the comparison baseline.",
  runC:
    "Encoder frozen — only the CTC decoder head (132K params) is trained for 100 epochs. " +
    "Same story-based split as baseline. Slightly better than full fine-tune.",
  runS:
    "Frozen encoder + stratified split: test samples drawn proportionally across all 7 speakers " +
    "and all stories. Best result — in-domain diversity beats raw data volume.",
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

function werColor(wer: number) {
  if (wer < 0.4) return "text-green-600";
  if (wer < 0.55) return "text-amber-600";
  return "text-red-500";
}

function cardBorder(isLast: boolean) {
  return isLast
    ? "border-green-200 bg-gradient-to-br from-green-50 to-white"
    : "border-gray-200 bg-white";
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function Home() {
  // Sort worst → best (descending WER)
  const experiments = Object.values(EXPERIMENTS).sort((a, b) => b.wer - a.wer);
  const bestId = experiments[experiments.length - 1].id;

  return (
    <main className="min-h-screen bg-gray-50 py-10 px-4">
      <div className="max-w-2xl mx-auto space-y-8">

        {/* Header */}
        <header className="text-center space-y-2">
          <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
            Amchi Konkani ASR
          </h1>
          <p className="text-sm text-gray-500">
            Fine-tuned IndicConformer for GSB Konkani — experiment comparison
          </p>
          <p className="text-xs text-gray-400">
            Click any experiment to demo transcriptions and explore results
          </p>
        </header>

        {/* Experiment cards */}
        <div className="space-y-4">
          {experiments.map((exp, i) => {
            const isFirst = i === 0;
            const isLast = exp.id === bestId;
            return (
              <Link
                key={exp.id}
                href={`/konkani/${exp.id}`}
                className={`block rounded-2xl border p-5 shadow-sm hover:shadow-md transition-all ${cardBorder(isLast)}`}
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="flex-1 space-y-1.5">
                    <div className="flex items-center gap-2">
                      {isFirst && (
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-gray-100 text-gray-500 uppercase tracking-wide">
                          Worst
                        </span>
                      )}
                      {isLast && (
                        <span className="text-xs font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-700 uppercase tracking-wide">
                          Best
                        </span>
                      )}
                      <span className="text-base font-semibold text-gray-800">
                        {exp.name}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500">{exp.config}</p>
                    <p className="text-xs text-gray-400 leading-relaxed">
                      {BRIEF_DESCRIPTIONS[exp.id]}
                    </p>
                  </div>

                  <div className="text-right flex-shrink-0 space-y-0.5">
                    <p
                      className={`text-2xl font-bold tabular-nums ${werColor(exp.wer)}`}
                    >
                      {Math.round(exp.wer * 100)}%
                    </p>
                    <p className="text-xs text-gray-400">Test WER</p>
                    <p className="text-xs text-gray-400">
                      {exp.samples} samples
                    </p>
                  </div>
                </div>
              </Link>
            );
          })}
        </div>

        {/* Footer */}
        <footer className="text-center text-xs text-gray-400 pt-4 border-t border-gray-100 space-y-0.5">
          <p>IndicConformer fine-tuned on Amchi Konkani (GSB / Chitrapur Saraswat)</p>
          <p>Post-processing via Gemini 2.5 Flash · TTS via Google Cloud</p>
        </footer>
      </div>
    </main>
  );
}
