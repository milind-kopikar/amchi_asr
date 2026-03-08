/**
 * Amchi Konkani ASR Experiment Results
 * 
 * Data source:
 * - 50-epoch baseline: results/experiments/20260302_031806/final_test_results.json
 * - Run C: results/experiments/run_c_story_split/final_test_results.json
 * - Run S: results/experiments/run_c_stratified_split/final_test_results.json
 */

export interface Speaker {
  label: string;
  id: string;
  n: number;
  mean: number;
  median: number;
  std: number;
  min: number;
  max: number;
  color: string;
  colorLight: string;
  colorBorder: string;
}

export interface HistogramBin {
  label: string;
  c: number[];  // counts per speaker
}

export interface ErrorType {
  label: string;
  frac: number;
  color: string;
  text: string;
}

export interface Experiment {
  id: string;
  name: string;
  date: string;
  config: string;
  samples: number;
  wer: number;
  werStd: number;
  bestEpoch: number;
  valWer: number;
  speakers: Speaker[];
  hist: HistogramBin[];
  errorTypes: ErrorType[];
}

export const EXPERIMENTS: Record<string, Experiment> = {
  baseline: {
    id: "baseline",
    name: "50-epoch Baseline",
    date: "2026-03-02",
    config: "Full fine-tune (115M params)",
    samples: 104,
    wer: 0.5467,
    werStd: 0.2022,
    bestEpoch: 47,
    valWer: 0.532,
    speakers: [
      {
        label: "Speaker 1",
        id: "ashaheble",
        n: 35,
        mean: 0.5169,
        median: 0.500,
        std: 0.2072,
        min: 0.000,
        max: 1.000,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Speaker 2",
        id: "dipti",
        n: 35,
        mean: 0.6030,
        median: 0.625,
        std: 0.1955,
        min: 0.143,
        max: 1.000,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Speaker 3",
        id: "lalimomadi",
        n: 34,
        mean: 0.5194,
        median: 0.500,
        std: 0.2005,
        min: 0.125,
        max: 1.000,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
    ],
    hist: [
      { label: "0–10%", c: [1, 0, 0] },
      { label: "10–20%", c: [1, 1, 2] },
      { label: "20–30%", c: [4, 1, 3] },
      { label: "30–40%", c: [2, 2, 4] },
      { label: "40–50%", c: [6, 4, 5] },
      { label: "50–60%", c: [8, 8, 9] },
      { label: "60–70%", c: [6, 10, 5] },
      { label: "70–80%", c: [4, 3, 2] },
      { label: "80–90%", c: [2, 4, 3] },
      { label: "90–100%", c: [1, 2, 1] },
    ],
    errorTypes: [
      { label: "Correct (40.9%)", frac: 0.409, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (27.8%)", frac: 0.278, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (9.4%)", frac: 0.094, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (3.5%)", frac: 0.035, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
  runC: {
    id: "runC",
    name: "Run C: Frozen Encoder",
    date: "2026-03-06",
    config: "Frozen encoder (132K params) + 100 epochs",
    samples: 104,
    wer: 0.4908,
    werStd: 0.2486,
    bestEpoch: 66,
    valWer: 0.504,
    speakers: [
      {
        label: "Speaker 1",
        id: "ashaheble",
        n: 35,
        mean: 0.4745,
        median: 0.500,
        std: 0.2483,
        min: 0.000,
        max: 1.000,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Speaker 2",
        id: "dipti",
        n: 35,
        mean: 0.5497,
        median: 0.500,
        std: 0.2449,
        min: 0.000,
        max: 1.000,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Speaker 3",
        id: "lalimomadi",
        n: 34,
        mean: 0.4469,
        median: 0.4365,
        std: 0.2485,
        min: 0.000,
        max: 1.200,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
    ],
    hist: [
      { label: "0–10%", c: [2, 0, 1] },
      { label: "10–20%", c: [2, 1, 3] },
      { label: "20–30%", c: [3, 1, 4] },
      { label: "30–40%", c: [4, 3, 4] },
      { label: "40–50%", c: [6, 4, 5] },
      { label: "50–60%", c: [7, 6, 6] },
      { label: "60–70%", c: [5, 8, 4] },
      { label: "70–80%", c: [3, 5, 3] },
      { label: "80–90%", c: [2, 4, 2] },
      { label: "90–100%", c: [1, 4, 2] },
    ],
    errorTypes: [
      { label: "Correct (37.0%)", frac: 0.370, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (26.6%)", frac: 0.266, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (4.3%)", frac: 0.043, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (8.0%)", frac: 0.080, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
  runS: {
    id: "runS",
    name: "Run S: Stratified Split",
    date: "2026-03-06",
    config: "Frozen encoder + stratified split (99 samples)",
    samples: 99,
    wer: 0.3414,
    werStd: 0.2579,
    bestEpoch: 88,
    valWer: 0.334,
    speakers: [
      {
        label: "Speaker 1",
        id: "ashaheble",
        n: 31,
        mean: 0.3716,
        median: 0.333,
        std: 0.2901,
        min: 0.000,
        max: 1.200,
        color: "#6366f1",
        colorLight: "#e0e7ff",
        colorBorder: "#a5b4fc",
      },
      {
        label: "Speaker 3",
        id: "lalimomadi",
        n: 29,
        mean: 0.2710,
        median: 0.267,
        std: 0.1930,
        min: 0.000,
        max: 0.833,
        color: "#10b981",
        colorLight: "#d1fae5",
        colorBorder: "#6ee7b7",
      },
      {
        label: "Speaker 4",
        id: "avkulkarni",
        n: 18,
        mean: 0.3290,
        median: 0.348,
        std: 0.2263,
        min: 0.000,
        max: 0.750,
        color: "#8b5cf6",
        colorLight: "#ede9fe",
        colorBorder: "#c4b5fd",
      },
      {
        label: "Speaker 5",
        id: "arursushama",
        n: 7,
        mean: 0.2711,
        median: 0.250,
        std: 0.2844,
        min: 0.000,
        max: 0.833,
        color: "#06b6d4",
        colorLight: "#cffafe",
        colorBorder: "#67e8f9",
      },
      {
        label: "Speaker 6",
        id: "sheela",
        n: 6,
        mean: 0.5683,
        median: 0.464,
        std: 0.3844,
        min: 0.273,
        max: 1.333,
        color: "#ec4899",
        colorLight: "#fce7f3",
        colorBorder: "#f9a8d4",
      },
      {
        label: "Speaker 2",
        id: "dipti",
        n: 5,
        mean: 0.4056,
        median: 0.333,
        std: 0.2775,
        min: 0.111,
        max: 0.833,
        color: "#f97316",
        colorLight: "#ffedd5",
        colorBorder: "#fdba74",
      },
      {
        label: "Speaker 7",
        id: "milindkopi",
        n: 3,
        mean: 0.3873,
        median: 0.400,
        std: 0.0489,
        min: 0.333,
        max: 0.429,
        color: "#84cc16",
        colorLight: "#ecfccb",
        colorBorder: "#bef264",
      },
    ],
    hist: [
      { label: "0–10%", c: [4, 4, 2, 1, 0, 0, 0] },
      { label: "10–20%", c: [4, 5, 2, 1, 0, 0, 0] },
      { label: "20–30%", c: [5, 6, 3, 1, 0, 0, 0] },
      { label: "30–40%", c: [6, 5, 4, 1, 0, 1, 1] },
      { label: "40–50%", c: [4, 3, 3, 1, 1, 0, 0] },
      { label: "50–60%", c: [3, 2, 2, 1, 1, 1, 0] },
      { label: "60–70%", c: [2, 1, 1, 0, 1, 0, 0] },
      { label: "70–80%", c: [1, 1, 1, 1, 1, 1, 0] },
      { label: "80–90%", c: [1, 1, 0, 0, 1, 0, 0] },
      { label: "90–100%", c: [1, 1, 1, 1, 1, 1, 1] },
    ],
    errorTypes: [
      { label: "Correct (39.1%)", frac: 0.391, color: "#d1fae5", text: "#065f46" },
      { label: "Substitutions (24.2%)", frac: 0.242, color: "#fecaca", text: "#991b1b" },
      { label: "Insertions (4.3%)", frac: 0.043, color: "#fed7aa", text: "#92400e" },
      { label: "Deletions (4.6%)", frac: 0.046, color: "#e9d5ff", text: "#6b21a8" },
    ],
  },
};

/**
 * Get a specific experiment by ID with validation
 * 
 * @param id - The experiment ID (baseline, runC, or runS)
 * @returns The experiment object, or undefined if not found
 */
export function getExperiment(id: string): Experiment | undefined {
  return EXPERIMENTS[id as keyof typeof EXPERIMENTS];
}

/**
 * Get all experiment IDs in display order
 */
export function getExperimentIds(): string[] {
  return Object.keys(EXPERIMENTS);
}
