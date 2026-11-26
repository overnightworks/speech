// Turns the raw V8 coverage the browser flows collected from Chromium into an lcov of the page's own sources.
// Usage: node scripts/page-coverage-report.js <raw coverage directory>
// It writes coverage/flow/coverage-browser-flow.lcov.
import { existsSync } from "node:fs";
import { readFile, readdir, rm } from "node:fs/promises";
import path from "node:path";
import { CoverageReport } from "monocart-coverage-reports";
import vitestConfig from "../vitest.config.ts";

const [rawDirectory] = process.argv.slice(2);
// The report empties its directory before it writes, so the directory is its own.
const REPORT_DIRECTORY = "coverage/flow";
const LCOV_FILE = "coverage-browser-flow.lcov";
// The unit lcov measures the sources vitest's coverage names; this one measures every other source of the package, so
// that a new file lands in one of the two reports and, if no flow loads it, fails the check below.
const UNIT_MEASURED = new Set(vitestConfig.test.coverage.include);
const sources = await readdir("src", { recursive: true });
// A source is any file in an extension the package's tsconfig compiles, unless vitest runs it as a test.
const SOURCE_EXTENSION = /\.[cm]?tsx?$/;
const MEASURED = sources
  .map((name) => path.join("src", name))
  .filter((name) => SOURCE_EXTENSION.test(name) && !vitestConfig.test.include.some((pattern) => path.matchesGlob(name, pattern)))
  .filter((name) => !UNIT_MEASURED.has(name));
const BUILD_PATH = "/dist/";
const repositoryRoot = path.resolve("../..");
const lcovPath = path.join(REPORT_DIRECTORY, LCOV_FILE);
// Without coverage data the report writes nothing, so an earlier run's lcov must not stand in for it.
await rm(lcovPath, { force: true });

const report = new CoverageReport({
  name: "Browser flows",
  outputDir: REPORT_DIRECTORY,
  // The maps name sources relative to the package; the lcov names them from the repository root.
  reports: [["lcovonly", { file: LCOV_FILE, projectRoot: repositoryRoot }]],
  outputFilter: (file) => MEASURED.includes(file.sourcePath),
});

// Chromium reports a script's coverage without its text, and the build's maps carry no sources, so both are read here.
async function withBuildAndMap(entry) {
  const built = path.join("dist", new URL(entry.url).pathname.slice(BUILD_PATH.length));
  const sourceMap = JSON.parse(await readFile(`${built}.map`, "utf8"));
  const sources = sourceMap.sources.map((source) => readFile(path.join(path.dirname(built), source), "utf8"));
  return {
    ...entry,
    source: await readFile(built, "utf8"),
    sourceMap: { ...sourceMap, sourcesContent: await Promise.all(sources) },
  };
}

for (const file of await readdir(rawDirectory)) {
  const { result } = JSON.parse(await readFile(path.join(rawDirectory, file), "utf8"));
  const built = result.filter((entry) => entry.url.startsWith("http") && new URL(entry.url).pathname.startsWith(BUILD_PATH));
  if (built.length > 0) {
    await report.add(await Promise.all(built.map(withBuildAndMap)));
  }
}
await report.generate();

const lcov = existsSync(lcovPath) ? await readFile(lcovPath, "utf8") : "";
const reported = new Set(
  lcov
    .split("\n")
    .filter((line) => line.startsWith("SF:"))
    .map((line) => line.slice("SF:".length)),
);
const missing = MEASURED.map((name) => path.relative(repositoryRoot, path.resolve(name))).filter(
  (name) => !reported.has(name),
);
if (missing.length > 0) {
  console.error(
    `Sources with no browser coverage record (a flow must load each, or vitest's coverage.include must name it):\n  ${missing.join("\n  ")}`,
  );
  process.exitCode = 1;
}
