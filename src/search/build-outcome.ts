/**
 * The exit code a finished build run must produce.
 *
 * ⚠️ EXTRACTED SO IT CAN BE FALSIFIED. `scripts/build-index.ts` runs `main()`
 * on import, so nothing in the suite can execute it -- which meant the decision
 * "does a partial build count as success?" had no test at all, and a deliberate
 * mutation that silenced it stayed green. A one-line pure function is the
 * smallest seam that makes the answer checkable.
 *
 * The distinction is not cosmetic. The run this replaces died at 1,727 of 2,466
 * documents; the index was usable but missing 739, and the wrapper reported
 * success. A caller that cannot tell "complete" from "partial" will keep
 * serving an index that silently returns nothing for a third of the corpus.
 *
 * ⚠️ THIS FILE IS IN `src/` ON PURPOSE. It previously lived in
 * `scripts/build-index.ts`, and the test importing it dragged one file from
 * above `src/` into the compilation, which made TypeScript infer `rootDir` as
 * the repository root and emit `build/src/...` instead of `build/...`. The
 * package's `main` and `bin` both point at `build/index.js`, so the build
 * silently produced a package whose entrypoint was the PREVIOUS build -- and
 * `prepack` deletes `build` first, so a publish would have shipped nothing at
 * all. 473 tests and a clean `tsc` exit both missed it, because the suite runs
 * under ts-node where the output layout is irrelevant. Keeping this under
 * `src/` is what holds the root still.
 */
export function exitCodeForReport(report: { failed: number }): number {
  return report.failed > 0 ? 3 : 0;
}
