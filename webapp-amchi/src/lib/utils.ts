/**
 * Utility functions for Amchi Konkani ASR results visualization
 */

/**
 * Format a decimal fraction as a percentage string
 * 
 * @param n - The fraction (0-1)
 * @param d - Number of decimal places (default 1)
 * @returns Formatted percentage string (e.g., "54.7%")
 */
export function formatPercentage(n: number, d = 1): string {
  return (n * 100).toFixed(d) + "%";
}

/**
 * Calculate SVG path data for error type bar segments
 * 
 * @param segments - Array of segment objects with `frac` property
 * @param totalWidth - Total width of the bar
 * @returns Array of segment objects with calculated x position and width
 */
export function calculateSegmentPositions(
  segments: Array<{ frac: number; [key: string]: any }>,
  totalWidth: number
) {
  let xPos = 0;
  return segments.map((seg) => {
    const w = seg.frac * totalWidth;
    const result = { ...seg, x: xPos, width: w };
    xPos += w;
    return result;
  });
}

/**
 * Type guard: Check if value is a valid experiment ID
 * 
 * @param id - The ID to check
 * @returns True if ID is valid
 */
export function isValidExperimentId(id: unknown): id is string {
  return typeof id === 'string' && id.length > 0;
}
