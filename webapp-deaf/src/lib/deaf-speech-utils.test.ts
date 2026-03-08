/**
 * Unit Tests for Deaf Speech Utilities
 * 
 * Tests all utility functions with various edge cases and error conditions.
 * Ensures code doesn't break on malformed data.
 * 
 * @module deaf-speech-utils.test
 */

import {
  formatPercentage,
  getWERImprovement,
  computeErrorStatistics,
  computeWERDistribution,
  findExampleSamples,
  safeAccess,
  isValidExperiment,
  hasPostProcessed,
  getComparisonMetrics,
} from "./deaf-speech-utils";
import type { DeafSpeechSample, DeafSpeechExperiment } from "./deaf-speech-types";

/**
 * Test Suite: formatPercentage
 */
describe("formatPercentage", () => {
  test("formats simple percentage", () => {
    expect(formatPercentage(0.347)).toBe("34.7%");
  });

  test("handles different decimal places", () => {
    expect(formatPercentage(0.347, 0)).toBe("35%");
    expect(formatPercentage(0.347, 2)).toBe("34.70%");
  });

  test("handles edge cases", () => {
    expect(formatPercentage(0)).toBe("0.0%");
    expect(formatPercentage(1)).toBe("100.0%");
  });

  test("handles invalid input gracefully", () => {
    expect(formatPercentage(NaN)).toBe("N/A");
    expect(formatPercentage("invalid" as any)).toBe("N/A");
  });
});

/**
 * Test Suite: getWERImprovement
 */
describe("getWERImprovement", () => {
  test("calculates positive improvement", () => {
    expect(getWERImprovement(0.5, 0.3)).toBe(20.0); // 20pp improvement
  });

  test("calculates negative improvement (worsening)", () => {
    expect(getWERImprovement(0.3, 0.5)).toBe(-20.0); // 20pp worse
  });

  test("handles no improvement", () => {
    expect(getWERImprovement(0.5, 0.5)).toBe(0);
  });

  test("handles invalid input", () => {
    expect(getWERImprovement(NaN, 0.5)).toBe(0);
    expect(getWERImprovement(0.5, NaN)).toBe(0);
  });
});

/**
 * Test Suite: computeErrorStatistics
 */
describe("computeErrorStatistics", () => {
  test("returns zero stats for empty array", () => {
    const stats = computeErrorStatistics([]);
    expect(stats.improved).toBe(0);
    expect(stats.successRate).toBe(0);
  });

  test("counts improved/worsened samples", () => {
    const samples: DeafSpeechSample[] = [
      {
        audio: "test1.wav",
        reference: "test 1",
        prediction: "test 1",
        wer: 0.5,
        postprocessed: { prediction: "test 1", mode: "FILL", wer: 0.3 },
      },
      {
        audio: "test2.wav",
        reference: "test 2",
        prediction: "test 2",
        wer: 0.3,
        postprocessed: { prediction: "test 2", mode: "RECONSTRUCT", wer: 0.5 },
      },
      {
        audio: "test3.wav",
        reference: "test 3",
        prediction: "test 3",
        wer: 0.4,
      },
    ];

    const stats = computeErrorStatistics(samples);
    expect(stats.improved).toBe(1);
    expect(stats.worsened).toBe(1);
    expect(stats.unchanged).toBe(0);
  });

  test("handles missing post-processing data", () => {
    const samples: DeafSpeechSample[] = [
      {
        audio: "test.wav",
        reference: "test",
        prediction: "test",
        wer: 0.5,
      },
    ];

    const stats = computeErrorStatistics(samples);
    expect(stats.improved).toBe(0);
    expect(stats.successRate).toBe(0);
  });
});

/**
 * Test Suite: computeWERDistribution
 */
describe("computeWERDistribution", () => {
  test("creates correct bins", () => {
    const samples: DeafSpeechSample[] = [
      { audio: "1.wav", reference: "r", prediction: "p", wer: 0.05 }, // 0-10%
      { audio: "2.wav", reference: "r", prediction: "p", wer: 0.15 }, // 10-20%
      { audio: "3.wav", reference: "r", prediction: "p", wer: 0.5 }, // 50-60%
    ];

    const dist = computeWERDistribution(samples, 0.1);
    expect(dist.ranges.length).toBe(10);
    expect(dist.ranges[0].count).toBe(1); // 0-10%
    expect(dist.ranges[1].count).toBe(1); // 10-20%
    expect(dist.ranges[5].count).toBe(1); // 50-60%
  });

  test("handles empty array", () => {
    const dist = computeWERDistribution([]);
    expect(dist.ranges.length).toBe(0);
  });
});

/**
 * Test Suite: findExampleSamples
 */
describe("findExampleSamples", () => {
  test("finds best and worst samples", () => {
    const samples: DeafSpeechSample[] = [
      { audio: "1.wav", reference: "r", prediction: "p", wer: 0.1 },
      { audio: "2.wav", reference: "r", prediction: "p", wer: 0.9 },
      { audio: "3.wav", reference: "r", prediction: "p", wer: 0.5 },
    ];

    const { best, worst } = findExampleSamples(samples, 1);
    expect(best[0].wer).toBe(0.1);
    expect(worst[0].wer).toBe(0.9);
  });

  test("handles empty array", () => {
    const { best, worst } = findExampleSamples([]);
    expect(best).toEqual([]);
    expect(worst).toEqual([]);
  });
});

/**
 * Test Suite: safeAccess
 */
describe("safeAccess", () => {
  test("accesses nested properties", () => {
    const obj = { a: { b: { c: "value" } } };
    expect(safeAccess(obj, "a.b.c")).toBe("value");
  });

  test("returns fallback for missing path", () => {
    const obj = { a: { b: {} } };
    expect(safeAccess(obj, "a.b.c", "fallback")).toBe("fallback");
  });

  test("handles null/undefined gracefully", () => {
    expect(safeAccess(null, "any.path", "fallback")).toBe("fallback");
    expect(safeAccess(undefined, "any.path", "fallback")).toBe("fallback");
  });
});

/**
 * Test Suite: isValidExperiment
 */
describe("isValidExperiment", () => {
  test("validates good experiment", () => {
    const exp: DeafSpeechExperiment = {
      id: "test",
      name: "Test",
      date: "2026-01-01",
      config: "test",
      samples: 100,
      wer: 0.5,
      bestEpoch: 50,
      valWer: 0.45,
      checkpoint: "path",
      notes: "test",
      per_sample: [{ audio: "a.wav", reference: "r", prediction: "p", wer: 0.5 }],
    };
    expect(isValidExperiment(exp)).toBe(true);
  });

  test("rejects invalid experiment", () => {
    expect(isValidExperiment(null)).toBe(false);
    expect(isValidExperiment({})).toBe(false);
    expect(isValidExperiment({ id: "test", wer: 0.5, per_sample: [] })).toBe(false);
  });
});

/**
 * Test Suite: hasPostProcessed
 */
describe("hasPostProcessed", () => {
  test("detects post-processed sample", () => {
    const sample: DeafSpeechSample = {
      audio: "test.wav",
      reference: "test",
      prediction: "test",
      wer: 0.5,
      postprocessed: { prediction: "corrected", mode: "FILL", wer: 0.3 },
    };
    expect(hasPostProcessed(sample)).toBe(true);
  });

  test("detects non-post-processed sample", () => {
    const sample: DeafSpeechSample = {
      audio: "test.wav",
      reference: "test",
      prediction: "test",
      wer: 0.5,
    };
    expect(hasPostProcessed(sample)).toBe(false);
  });
});

/**
 * Test Suite: getComparisonMetrics
 */
describe("getComparisonMetrics", () => {
  test("generates comparison for post-processed sample", () => {
    const sample: DeafSpeechSample = {
      audio: "test.wav",
      reference: "test",
      prediction: "test",
      wer: 0.5,
      postprocessed: { prediction: "corrected", mode: "FILL", wer: 0.3 },
    };

    const comparison = getComparisonMetrics(sample);
    expect(comparison).not.toBeNull();
    expect(comparison?.improvement).toBe("+20.0pp");
  });

  test("returns null for non-post-processed sample", () => {
    const sample: DeafSpeechSample = {
      audio: "test.wav",
      reference: "test",
      prediction: "test",
      wer: 0.5,
    };

    const comparison = getComparisonMetrics(sample);
    expect(comparison).toBeNull();
  });
});
