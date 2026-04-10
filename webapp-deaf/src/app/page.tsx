"use client";

import Link from "next/link";
import { DEAF_SPEECH_EXPERIMENTS } from "@/lib/deaf-speech-experiments";

const BRIEF_DESCRIPTIONS: Record<string, string> = {
  dsc: "Baseline: full fine-tune (all 115M params), single speaker, 124 samples, 50 epochs.",
  dsa: "Frozen encoder — only the decoder head trained, same single speaker, 100 epochs.",
  dsb: "Extended data — added 64 clips from multiple additional speakers across other stories.",
  dsd: "Speed perturbation — same single speaker, 0.9x/1.0x/1.1x speed variants, 3x training data.",
};

function werColor(wer: number) {
  if (wer >= 0.8) return "text-red-600";
  if (wer >= 0.6) return "text-orange-500";
  if (wer >= 0.4) return "text-yellow-600";
  return "text-emerald-600";
}

function cardBorder(isLast: boolean) {
  if (isLast) return "border-emerald-200 hover:border-emerald-500 bg-emerald-50/20";
  return "border-gray-200 hover:border-purple-400";
}

const APP_ITERATIONS = [
  {
    href: "/demo/iteration5",
    label: "Iteration 5",
    title: "Direct Speech",
    description: "Record → app speaks immediately. No transcript shown.",
    note: "Taranath rejected this — errors were spoken aloud without his approval.",
    badge: null,
    color: "border-amber-200 hover:border-amber-400",
    badgeColor: "",
  },
  {
    href: "/demo/iteration6",
    label: "Iteration 6",
    title: "Transcript First",
    description: "Record → see raw + corrected transcript → speak.",
    note: "Better, but Taranath wanted to edit errors before speaking.",
    badge: null,
    color: "border-blue-200 hover:border-blue-400",
    badgeColor: "",
  },
  {
    href: "/demo/final",
    label: "Final Product",
    title: "Editable Transcript",
    description: "Record → edit corrected transcript → speak in Marathi or English.",
    note: "The version Taranath uses today.",
    badge: "BEST ⭐",
    color: "border-emerald-200 hover:border-emerald-500 bg-emerald-50/20",
    badgeColor: "bg-emerald-100 text-emerald-700",
  },
];

export default function HomePage() {
  const experiments = Object.values(DEAF_SPEECH_EXPERIMENTS).sort(
    (a, b) => b.wer - a.wer
  );

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="border-b border-gray-200 bg-white">
        <div className="max-w-2xl mx-auto px-4 py-6">
          <h1 className="text-2xl font-bold text-gray-900">ResonAIte — Deaf Speech ASR</h1>
          <p className="text-sm text-gray-500 mt-1">
            IndicConformer (AI4Bharat) fine-tuned on Marathi deaf speech — experiments &amp; app demos
          </p>
        </div>
      </header>

      <main className="max-w-2xl mx-auto px-4 py-8 space-y-4">
        {/* ML experiments section */}
        <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest">
          ML Experiments — worst to best
        </p>

        {experiments.map((exp, i) => {
          const isBest = i === experiments.length - 1;
          const isWorst = i === 0;

          return (
            <Link key={exp.id} href={`/deaf-speech/${exp.id}`}>
              <div
                className={`group bg-white rounded-xl border-2 p-5 hover:shadow-md transition-all cursor-pointer ${cardBorder(isBest)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs text-gray-400 font-mono">
                        {i + 1}.
                      </span>
                      <h2 className="text-base font-semibold text-gray-900">
                        {exp.name}
                      </h2>
                      {isWorst && (
                        <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded-full font-medium">
                          WORST
                        </span>
                      )}
                      {isBest && (
                        <span className="text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded-full font-medium">
                          BEST ⭐
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500 ml-5">
                      {BRIEF_DESCRIPTIONS[exp.id]}
                    </p>
                  </div>
                  <div className="text-right ml-4 shrink-0">
                    <p
                      className={`text-2xl font-bold ${werColor(exp.wer)}`}
                    >
                      {(exp.wer * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-400">Test WER</p>
                  </div>
                </div>

                <div className="mt-3 ml-5 flex gap-4">
                  <span className="text-xs text-purple-500 font-medium group-hover:text-purple-700 transition-colors">
                    Demo →
                  </span>
                  <span className="text-xs text-blue-500 font-medium group-hover:text-blue-700 transition-colors">
                    Results →
                  </span>
                </div>
              </div>
            </Link>
          );
        })}

        {/* App iterations section */}
        <div className="pt-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-widest mb-4">
            App Design Iterations — live demo
          </p>
          <div className="space-y-3">
            {APP_ITERATIONS.map((item) => (
              <Link key={item.href} href={item.href}>
                <div
                  className={`group bg-white rounded-xl border-2 p-5 hover:shadow-md transition-all cursor-pointer ${item.color}`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-xs text-gray-400 font-mono">{item.label}</span>
                        <h2 className="text-base font-semibold text-gray-900">{item.title}</h2>
                        {item.badge && (
                          <span className={`text-xs px-2 py-0.5 rounded-full font-medium ${item.badgeColor}`}>
                            {item.badge}
                          </span>
                        )}
                      </div>
                      <p className="text-sm text-gray-500 ml-5">{item.description}</p>
                      <p className="text-xs text-gray-400 ml-5 mt-1 italic">{item.note}</p>
                    </div>
                    <div className="ml-4 shrink-0 self-center">
                      <span className="text-xs text-purple-500 font-medium group-hover:text-purple-700 transition-colors">
                        Try it →
                      </span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}
