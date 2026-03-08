/**
 * Layout and styling constants for Amchi Konkani ASR results visualization
 * 
 * All measurements are in SVG coordinate units unless otherwise noted.
 * Layout diagram:
 * 
 *   MT (margin top)
 *   ├─────────────────────────┐
 *   │  Chart Area (CW × CH)   │
 *   │  ┌───────────────────┐  │
 *   │  │ Stacked Bars      │  │
 *   │  └───────────────────┘  │
 *   └─────────────────────────┘
 *   MB (margin bottom)
 */

// ─── Stacked Histogram Layout ─────────────────────────────────────────────
export const HISTOGRAM_LAYOUT = {
  marginLeft: 34,        // ML: margin left (for y-axis labels)
  marginTop: 10,         // MT: margin top
  marginBottom: 50,      // MB: margin bottom (for x-axis labels)
  chartWidth: 476,       // CW: chart area width
  chartHeight: 168,      // CH: chart area height
  binsCount: 10,         // Number of WER percentage bins (0-10%, 10-20%, etc.)
  get slotWidth() {
    return this.chartWidth / this.binsCount;  // px per bin ≈ 47.6
  },
  get barWidth() {
    return this.slotWidth - 4;  // bar width with gap
  },
  yAxisMax: 28,  // y-axis maximum count (actual max usually ~25)
  get yScale() {
    return this.chartHeight / this.yAxisMax;  // px per count unit
  },
} as const;

// ─── Speaker Comparison Layout ───────────────────────────────────────────
export const SPEAKER_BARS_LAYOUT = {
  labelWidth: 128,
  barAreaWidth: 330,
  rowHeight: 44,
  get totalWidth() {
    return this.labelWidth + this.barAreaWidth + 55;
  },
  pilotBaseline: 0.351,  // Pilot baseline WER (35.1%)
} as const;

// ─── Error Type Breakdown Layout ─────────────────────────────────────────
export const ERROR_TYPE_LAYOUT = {
  barWidth: 460,
  barHeight: 28,
  minWidthForLabel: 38,  // minimum segment width to show percentage label
} as const;

// ─── Color Theme ────────────────────────────────────────────────────────
export const COLORS = {
  success: '#d1fae5',      // Light green
  successText: '#065f46',  // Dark green
  error: '#fecaca',        // Light red
  errorText: '#991b1b',    // Dark red
  warning: '#fed7aa',      // Light orange
  warningText: '#92400e',  // Dark orange
  neutral: '#e9d5ff',      // Light purple
  neutralText: '#6b21a8',  // Dark purple
  gridLine: '#f3f4f6',     // Very light gray
  axisLine: '#d1d5db',     // Light gray
  meanLine: '#dc2626',     // Red for mean line
  medianLine: 'white',     // White for median tick
} as const;

// ─── Typography ─────────────────────────────────────────────────────────
export const FONT_SIZES = {
  xAxisLabel: 9.5,
  yAxisLabel: 10,
  sampleCount: 9,
  binTotal: 9,
  meanLabel: 9,
  speakerLabel: 11,
  axisTitle: 10,
} as const;

// ─── Opacity ────────────────────────────────────────────────────────────
export const OPACITY = {
  barFill: 0.80,
  stdShading: 0.15,
  meanBar: 0.75,
} as const;
