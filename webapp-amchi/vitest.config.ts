/**
 * Vitest configuration.
 *
 * Tests live next to their modules as ``*.test.ts``. Run with:
 *   npm test           — single run
 *   npm run test:watch — watch mode
 */
import { defineConfig } from "vitest/config";
import path from "node:path";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
  resolve: {
    alias: {
      // Mirror the Next.js path alias so ``@/lib/...`` imports resolve in tests.
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
