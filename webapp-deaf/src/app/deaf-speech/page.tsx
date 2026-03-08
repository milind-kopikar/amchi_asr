"use client";

import Link from "next/link";
import { useState, useEffect, Suspense } from "react";
import {
  formatPercentage,
  computeErrorStatistics,
  computeWERDistribution,
  findExampleSamples,
  isValidExperiment,
  getComparisonMetrics,
} from "@/lib/deaf-speech-utils";
import {
  DEAF_SPEECH_EXPERIMENTS,
  getExperiment,
  getExperimentsByPerformance,
} from "@/lib/deaf-speech-experiments";
import type { DeafSpeechExperiment, DeafSpeechSample } from "@/lib/deaf-speech-types";

/**
 * Error boundary component for loading failures
 * Prevents entire page from breaking if data loading fails
 */
function SafeDataLoader({
  children,
  fallback = <div className="text-red-600 p-4">Failed to load results</div>,
}: {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}) {
  const [error, setError] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setError(true), 5000);
    return () => clearTimeout(timer);
  }, []);

  if (error) return fallback;
  return <>{children}</>;
}

/**
 * Statistics card component - reusable for metrics display
 */
function StatCard({
  label,
  value,
  subtext,
}: {
  label: string;
  value: string | number;
  subtext?: string;
}) {
  return (
    <div className="rounded-xl bg-white border border-gray-200 p-3 shadow-sm text-center">
      <p className="text-2xl font-bold text-gray-900">{value}</p>
      <p className="text-xs text-gray-500 mt-0.5">{label}</p>
      {subtext && <p className="text-xs text-gray-400">{subtext}</p>}
    </div>
  );
}

/**
 * Per-sample comparison component
 * Shows raw ASR vs post-processed output when available
 */
function SampleComparison({ sample, index }: { sample: DeafSpeechSample; index: number }) {
  const comparison = getComparisonMetrics(sample);
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-200 rounded-lg p-4 mb-3 bg-gray-50 hover:bg-white transition-colors cursor-pointer"
      onClick={() => setExpanded(!expanded)}
    >
      {/* Header - always visible */}
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-900">Sample #{index + 1}</p>
          <p className="text-sm text-gray-600">{sample.reference}</p>
        </div>
        <div className="text-right">
          <p className="text-lg font-bold text-gray-900">{formatPercentage(sample.wer)}</p>
          <p className="text-xs text-gray-500">WER</p>
        </div>
      </div>

      {/* Expandable details */}
      {expanded && (
        <div className="mt-4 pt-4 border-t border-gray-200 space-y-3">
          {/* Correct/Transcribed version */}
          <div>
            <p className="text-xs font-semibold text-gray-600 mb-1">Correct Transcription</p>
            <p className="text-sm bg-green-50 p-2 rounded border border-green-300 text-gray-800">
              {sample.reference}
            </p>
            <p className="text-xs text-gray-500 mt-1">Ground truth (correct)</p>
          </div>

          {/* Raw ASR output */}
          <div>
            <p className="text-xs font-semibold text-gray-600 mb-1">Raw ASR Output</p>
            <p className="text-sm bg-white p-2 rounded border border-gray-300 text-gray-800">
              {sample.prediction}
            </p>
            <p className="text-xs text-gray-500 mt-1">WER: {formatPercentage(sample.wer)}</p>
          </div>

          {/* Post-processed output (if available) */}
          {comparison && (
            <div>
              <div className="flex items-center gap-2 mb-1">
                <p className="text-xs font-semibold text-gray-600">Post-Processed Output</p>
                <span className="px-2 py-0.5 bg-blue-100 text-blue-700 rounded text-xs font-medium">
                  {comparison.processed.mode}
                </span>
              </div>
              <p className="text-sm bg-white p-2 rounded border border-blue-300 text-gray-800">
                {comparison.processed.prediction}
              </p>
              <p className="text-xs text-gray-500 mt-1">
                WER: {comparison.processed.wer} ({comparison.improvement})
              </p>
            </div>
          )}

          {/* Audio file (for reference) */}
          <div>
            <p className="text-xs font-semibold text-gray-600 mb-1">Audio File</p>
            <p className="text-xs text-gray-500 font-mono">{sample.audio}</p>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * WER distribution histogram component
 */
function WERDistributionChart({
  samples,
}: {
  samples: DeafSpeechSample[];
}) {
  const distribution = computeWERDistribution(samples);

  return (
    <div className="space-y-2">
      {distribution.ranges.map((range) => (
        <div key={range.label} className="flex items-center gap-2 text-sm">
          <span className="w-12 text-xs font-medium text-gray-600">{range.label}</span>
          <div className="flex-1 bg-gray-200 h-6 rounded overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all"
              style={{ width: `${range.percentage}%` }}
            />
          </div>
          <span className="w-16 text-right text-xs text-gray-600">
            {range.count} ({range.percentage}%)
          </span>
        </div>
      ))}
    </div>
  );
}

/**
 * Best/Worst samples display component
 */
function ExampleSamples({
  samples,
  label,
  type,
}: {
  samples: DeafSpeechSample[];
  label: string;
  type: "best" | "worst";
}) {
  if (samples.length === 0) return null;

  const color = type === "best" ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200";
  const textColor = type === "best" ? "text-green-700" : "text-red-700";

  return (
    <div className={`border rounded-lg p-3 ${color}`}>
      <p className={`text-xs font-semibold ${textColor} mb-2`}>{label}</p>
      <div className="space-y-2">
        {samples.map((sample, i) => (
          <div key={i} className="text-sm">
            <p className="font-medium text-gray-900 truncate">{sample.reference}</p>
            <p className="text-xs text-gray-600 truncate">{sample.prediction}</p>
            <p className="text-xs text-gray-500">WER: {formatPercentage(sample.wer)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Main page component
 * Orchestrates data loading, display, and user interactions
 */
export default function DeafSpeechResultsPage() {
  const [selectedExpId, setSelectedExpId] = useState("dsd");
  const [selectedExp, setSelectedExp] = useState<DeafSpeechExperiment | null>(null);
  const [samples, setSamples] = useState<DeafSpeechSample[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBestWorst, setShowBestWorst] = useState(false);

  /**
   * Load experiment data when selection changes
   * Includes error handling and validation
   */
  useEffect(() => {
    const loadExperiment = async () => {
      try {
        setLoading(true);
        setError(null);

        const exp = getExperiment(selectedExpId);
        if (!isValidExperiment(exp)) {
          throw new Error("Invalid experiment configuration");
        }

        setSelectedExp(exp);

        // In production, this would load from API or static files
        // For now, we set empty samples and they would be populated dynamically
        setSamples(exp.per_sample || []);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Unknown error";
        setError(`Failed to load experiment: ${message}`);
        setSelectedExp(getExperiment("dsd"));
      } finally {
        setLoading(false);
      }
    };

    loadExperiment();
  }, [selectedExpId]);

  if (!selectedExp) {
    return <div className="text-center p-8">Loading...</div>;
  }

  const errorStats = computeErrorStatistics(samples);
  const examples = findExampleSamples(samples, 5);

  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar Navigation */}
      <aside className="w-48 bg-white border-r border-gray-200 shadow-sm">
        <nav className="sticky top-0 space-y-1 p-4">
          <Link
            href="/"
            className="block px-4 py-2 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-100 transition-colors"
          >
            Home
          </Link>
          <div className="pt-2">
            <p className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Deaf Speech Experiments
            </p>
            {Object.values(DEAF_SPEECH_EXPERIMENTS).map((e) => (
              <button
                key={e.id}
                onClick={() => setSelectedExpId(e.id)}
                className={`w-full text-left px-4 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  selectedExpId === e.id
                    ? "bg-purple-50 text-purple-700 border-l-2 border-purple-600"
                    : "text-gray-600 hover:bg-gray-50"
                }`}
              >
                {e.name}
                {e.id === "dsd" && <span className="text-xs ml-1">⭐</span>}
              </button>
            ))}
          </div>
        </nav>
      </aside>

      {/* Main Content */}
      <main className="flex-1 py-6 px-4">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Header */}
          <header className="space-y-1">
            <h1 className="text-3xl font-bold text-gray-900 tracking-tight">
              Deaf Speech ASR — {selectedExp.name}
            </h1>
            <p className="text-sm text-gray-600">{selectedExp.config}</p>
            <p className="text-xs text-gray-500">{selectedExp.date}</p>
          </header>

          {/* Error message */}
          {error && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {error}
            </div>
          )}

          {/* Key insights */}
          {selectedExp.notes && (
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-900">{selectedExp.notes}</p>
            </div>
          )}

          {/* Summary statistics cards */}
          <div className="grid grid-cols-4 gap-3">
            <StatCard
              label="Test WER"
              value={formatPercentage(selectedExp.wer)}
              subtext="Lower is better"
            />
            <StatCard
              label="Test Samples"
              value={selectedExp.samples}
              subtext="Single speaker group"
            />
            <StatCard
              label="Best Epoch"
              value={selectedExp.bestEpoch}
              subtext={`Val WER: ${formatPercentage(selectedExp.valWer)}`}
            />
            <StatCard
              label="Ranking"
              value={
                Object.values(DEAF_SPEECH_EXPERIMENTS)
                  .sort((a, b) => a.wer - b.wer)
                  .findIndex((e) => e.id === selectedExp.id) + 1
              }
              subtext="of 4 experiments"
            />
          </div>

          {/* Post-processing metrics (if available) */}
          {errorStats.improved > 0 && (
            <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
              <h2 className="text-sm font-semibold text-gray-800">Post-Processing Impact</h2>
              <div className="grid grid-cols-3 gap-3">
                <StatCard
                  label="Improved"
                  value={errorStats.improved}
                  subtext={`${formatPercentage(errorStats.successRate)} success`}
                />
                <StatCard
                  label="Worsened"
                  value={errorStats.worsened}
                  subtext={`Avg: ${errorStats.avgWorsening}pp`}
                />
                <StatCard
                  label="Avg Improvement"
                  value={`+${errorStats.avgImprovement}pp`}
                  subtext="For improved samples"
                />
              </div>
            </section>
          )}

          {/* WER Distribution */}
          {samples.length > 0 && (
            <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm space-y-3">
              <h2 className="text-sm font-semibold text-gray-800">WER Distribution ({samples.length} samples)</h2>
              <WERDistributionChart samples={samples} />
            </section>
          )}

          {/* Best/Worst examples */}
          {samples.length > 0 && (
            <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm">
              <button
                onClick={() => setShowBestWorst(!showBestWorst)}
                className="flex items-center justify-between w-full text-sm font-semibold text-gray-800 hover:text-gray-900"
              >
                <span>Best & Worst Performing Samples</span>
                <span className="text-lg">{showBestWorst ? "−" : "+"}</span>
              </button>
              {showBestWorst && (
                <div className="grid grid-cols-2 gap-4 mt-4">
                  <ExampleSamples
                    samples={examples.best}
                    label="✓ Lowest WER (Best)"
                    type="best"
                  />
                  <ExampleSamples
                    samples={examples.worst}
                    label="✗ Highest WER (Worst)"
                    type="worst"
                  />
                </div>
              )}
            </section>
          )}

          {/* Per-sample details */}
          {samples.length > 0 && (
            <section className="rounded-xl bg-white border border-gray-200 p-4 shadow-sm">
              <h2 className="text-sm font-semibold text-gray-800 mb-4">Sample Details</h2>
              <div className="max-h-96 overflow-y-auto">
                {samples.map((sample, i) => (
                  <SampleComparison key={i} sample={sample} index={i} />
                ))}
              </div>
            </section>
          )}

          {/* Loading state */}
          {loading && (
            <div className="text-center p-8 text-gray-500">Loading results...</div>
          )}
        </div>
      </main>
    </div>
  );
}
