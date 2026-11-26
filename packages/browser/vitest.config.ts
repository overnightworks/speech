import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    sequence: { shuffle: true },
    include: ["src/**/*.test.ts"],
    coverage: {
      provider: "v8",
      // The sources the unit tests cover; scripts/page-coverage-report.js measures every other source from the browser flows.
      include: ["src/resampler.ts", "src/hop.ts", "src/wav-stream.ts", "src/timeline.ts"],
      // Vitest empties its reports directory, so it is never the repository root.
      reportsDirectory: "coverage",
      reporter: ["text", ["lcovonly", { file: "coverage-browser-unit.lcov", projectRoot: "../.." }]],
      thresholds: { branches: 100, lines: 100, functions: 100, statements: 100 },
    },
  },
});
