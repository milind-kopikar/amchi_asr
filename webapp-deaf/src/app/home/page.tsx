"use client";

import Link from "next/link";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50">
      {/* Header */}
      <header className="border-b border-purple-200 bg-white/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-gray-900">Amchi Konkani</h1>
          <p className="text-gray-600 mt-1">ASR & Transcription Demo</p>
        </div>
      </header>

      {/* Main Navigation */}
      <main className="max-w-7xl mx-auto px-4 py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Demo Card */}
          <Link href="/demo">
            <div className="group h-full bg-white rounded-xl border-2 border-gray-200 p-8 hover:border-purple-500 hover:shadow-lg transition-all cursor-pointer">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-900">Demo</h2>
                <div className="text-4xl">🎤</div>
              </div>
              <p className="text-gray-600 mb-4">
                Experience live Konkani ASR transcription with post-processing corrections. Try it with audio samples.
              </p>
              <div className="inline-flex items-center text-purple-600 font-semibold group-hover:translate-x-2 transition-transform">
                View Demo →
              </div>
            </div>
          </Link>

          {/* Results Card */}
          <Link href="/results">
            <div className="group h-full bg-white rounded-xl border-2 border-gray-200 p-8 hover:border-blue-500 hover:shadow-lg transition-all cursor-pointer">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-2xl font-bold text-gray-900">Results</h2>
                <div className="text-4xl">📊</div>
              </div>
              <p className="text-gray-600 mb-4">
                Explore deaf speech experiment results ranked from worst to best performance. Compare 4 different training approaches.
              </p>
              <div className="inline-flex items-center text-blue-600 font-semibold group-hover:translate-x-2 transition-transform">
                View Results →
              </div>
            </div>
          </Link>
        </div>
      </main>
    </div>
  );
}
