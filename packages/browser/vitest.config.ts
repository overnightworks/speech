import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    environment: "node",
    include: ["src/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/resampler.ts", "src/hop.ts"],
      // Vitest empties its reports directory, so it is never the repository root.
      reportsDirectory: "coverage",
      reporter: ["text", ["lcovonly", { file: "coverage-browser-unit.lcov", projectRoot: "../.." }]],
      thresholds: { branches: 100, lines: 100, functions: 100, statements: 100 },
    },
  },
});
