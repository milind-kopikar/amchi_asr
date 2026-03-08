/**
 * Utility functions for processing Deaf Speech ASR results
 * 
 * Provides data loading, processing, and analysis utilities for deaf speech experiments.
 * All functions use defensive error handling to prevent breaking the UI if data is malformed.
 * 
 * @module deaf-speech-utils
 */

import type {
  DeafSpeechSample,
  DeafSpeechExperiment,
  ErrorStatistics,
  WERDistribution,
  RawExperimentResults,
} from "./deaf-speech-types";

/**
 * Format a decimal fraction as a percentage string
 * 
 * @param n - The fraction (0-1)
 * @param decimals - Number of decimal places (default 1)
 * @returns Formatted percentage string (e.g., "34.7%")
 * @example formatPercentage(0.347) => "34.7%"
 */
export function formatPercentage(n: number, decimals = 1): string {
  if (typeof n !== "number" || isNaN(n)) return "N/A";
  return (n * 100).toFixed(decimals) + "%";
}

/**
 * Calculate WER (Word Error Rate) improvements between two predictions
 * 
 * Used to compare raw ASR vs post-processed output.
 * Negative values indicate improvement (lower WER is better).
 * 
 * @param originalWER - WER of raw ASR output
 * @param processedWER - WER of post-processed output
 * @returns Improvement in percentage points
 * @example getWERImprovement(0.347, 0.280) => 6.7
 */
export function getWERImprovement(
  originalWER: number,
  processedWER: number
): number {
  if (typeof originalWER !== "number" || typeof processedWER !== "number") {
    return 0;
  }
  return Math.round((originalWER - processedWER) * 1000) / 10; // Result in percentage points
}

/**
 * Compute error analysis statistics for an experiment
 * 
 * Determines how many samples improved/worsened with post-processing.
 * Only evaluates samples that have post-processing results available.
 * 
 * @param samples - Array of samples with optional post-processed results
 * @returns Statistics object with improvement metrics
 * @throws Returns zero-initialized stats object on error
 */
export function computeErrorStatistics(
  samples: DeafSpeechSample[]
): ErrorStatistics {
  if (!Array.isArray(samples) || samples.length === 0) {
    return {
      improved: 0,
      worsened: 0,
      unchanged: 0,
      avgImprovement: 0,
      avgWorsening: 0,
      successRate: 0,
    };
  }

  let improved = 0;
  let worsened = 0;
  let unchanged = 0;
  let totalImprovement = 0;
  let totalWorsening = 0;
  let processedCount = 0;

  samples.forEach((sample) => {
    if (!sample.postprocessed) return;

    const originalWER = sample.wer || 0;
    const processedWER = sample.postprocessed.wer || 0;
    const delta = originalWER - processedWER;

    if (delta > 0.01) {
      improved++;
      totalImprovement += delta;
    } else if (delta < -0.01) {
      worsened++;
      totalWorsening += Math.abs(delta);
    } else {
      unchanged++;
    }
    processedCount++;
  });

  const avgImprovement = improved > 0 ? totalImprovement / improved : 0;
  const avgWorsening = worsened > 0 ? totalWorsening / worsened : 0;
  const successRate = processedCount > 0 ? improved / processedCount : 0;

  return {
    improved,
    worsened,
    unchanged,
    avgImprovement: Math.round(avgImprovement * 1000) / 10,
    avgWorsening: Math.round(avgWorsening * 1000) / 10,
    successRate: Math.round(successRate * 1000) / 10,
  };
}

/**
 * Generate WER distribution histogram bins
 * 
 * Groups samples by WER range (0-10%, 10-20%, etc.)
 * 
 * @param samples - Array of samples
 * @param binSize - Size of each bin (default 0.1 = 10%)
 * @returns WER distribution data
 */
export function computeWERDistribution(
  samples: DeafSpeechSample[],
  binSize = 0.1
): WERDistribution {
  if (!Array.isArray(samples) || samples.length === 0) {
    return { ranges: [] };
  }

  const ranges: WERDistribution["ranges"] = [];
  const maxBins = Math.ceil(1 / binSize);

  for (let i = 0; i < maxBins; i++) {
    const min = i * binSize;
    const max = (i + 1) * binSize;
    const label = `${Math.round(min * 100)}–${Math.round(max * 100)}%`;

    const count = samples.filter(
      (s) => s.wer >= min && s.wer < max
    ).length;

    ranges.push({
      label,
      min,
      max,
      count,
      percentage: Math.round((count / samples.length) * 1000) / 10,
    });
  }

  return { ranges };
}

/**
 * Find best and worst performing samples
 * 
 * Useful for showing concrete examples in the UI.
 * 
 * @param samples - Array of samples
 * @param count - Number of samples to return (default 5)
 * @returns Object with best and worst samples
 */
export function findExampleSamples(
  samples: DeafSpeechSample[],
  count = 5
): { best: DeafSpeechSample[]; worst: DeafSpeechSample[] } {
  if (!Array.isArray(samples) || samples.length === 0) {
    return { best: [], worst: [] };
  }

  const sorted = [...samples].sort((a, b) => (a.wer || 0) - (b.wer || 0));

  return {
    best: sorted.slice(0, count),
    worst: sorted.slice(-count).reverse(),
  };
}

/**
 * Safely access a nested property with fallback
 * 
 * Prevents errors when accessing deeply nested optional properties.
 * 
 * @param obj - The object to access
 * @param path - Dot-separated path (e.g., "summary.mean_wer")
 * @param fallback - Value to return if path not found
 * @returns The value or fallback
 */
export function safeAccess(
  obj: any,
  path: string,
  fallback: any = null
): any {
  try {
    const result = path.split(".").reduce((acc, part) => acc?.[part], obj);
    return result !== undefined ? result : fallback;
  } catch {
    return fallback;
  }
}

/**
 * Validate that experiment data has required fields
 * 
 * Used in error boundaries to check if data is properly loaded.
 * 
 * @param exp - Experiment data to validate
 * @returns True if experiment has all required fields
 */
export function isValidExperiment(exp: any): boolean {
  return (
    exp &&
    typeof exp.id === "string" &&
    typeof exp.wer === "number" &&
    Array.isArray(exp.per_sample) &&
    exp.per_sample.length > 0
  );
}

/**
 * Type guard to check if sample has post-processed results
 * 
 * @param sample - Sample to check
 * @returns True if sample has post-processed prediction
 */
export function hasPostProcessed(sample: DeafSpeechSample): boolean {
  return !!(sample.postprocessed && sample.postprocessed.prediction);
}

/**
 * Extract comparison metrics between raw and post-processed
 * 
 * Useful for showing side-by-side comparison in UI.
 * Only returns data if post-processing is available.
 * 
 * @param sample - Sample with optional post-processing
 * @returns Comparison object or null if no post-processing
 */
export function getComparisonMetrics(
  sample: DeafSpeechSample
): {
  original: { wer: string; prediction: string };
  processed: { wer: string; prediction: string; mode: string };
  improvement: string;
} | null {
  if (!sample.postprocessed) return null;

  const originalWER = formatPercentage(sample.wer);
  const processedWER = formatPercentage(sample.postprocessed.wer);
  const improvement = getWERImprovement(sample.wer, sample.postprocessed.wer);

  return {
    original: {
      wer: originalWER,
      prediction: sample.prediction,
    },
    processed: {
      wer: processedWER,
      prediction: sample.postprocessed.prediction,
      mode: sample.postprocessed.mode,
    },
    improvement: `${improvement > 0 ? "+" : ""}${improvement}pp`,
  };
}
