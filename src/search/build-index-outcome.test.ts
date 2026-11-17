import assert from "node:assert/strict";
import { describe, test } from "node:test";
import { exitCodeForReport } from "./build-outcome";

/**
 * A partial build is not a success.
 *
 * ⚠️ THE DEFECT THIS PINS, IN ONE SENTENCE: the run that died at 1,727 of 2,466
 * documents left an index missing 739 of them, and its wrapper reported exit 0.
 *
 * The decision was untestable before this because `scripts/build-index.ts`
 * launched `main()` on import, so no test could reach the branch without
 * starting a build. Extracting the one-line decision is the whole seam -- and
 * a mutation that silences it now fails here instead of surviving as green.
 */
describe("the build's exit code distinguishes complete from partial", () => {
  test("a clean run exits 0", () => {
    assert.equal(exitCodeForReport({ failed: 0 }), 0);
  });

  test("one failed document is a partial build, not a success", () => {
    // The corpus case: 739 documents never attempted. Returning 0 here is what
    // made the failure invisible -- systemd had nothing to report and the log
    // was the only instrument, which nobody opens after a clean status.
    assert.equal(exitCodeForReport({ failed: 1 }), 3);
  });

  test("many failed documents still exit on the same code", () => {
    // Distinct from 1 on purpose: 1 means the run could not proceed at all,
    // 3 means it proceeded and left the index incomplete. A caller that treats
    // them the same would either retry a corrupt index or ignore a usable one.
    assert.equal(exitCodeForReport({ failed: 739 }), 3);
  });

  test("the code is not the failure code, so a partial run is diagnosable", () => {
    for (const failed of [1, 7, 739, 2466]) {
      assert.notEqual(
        exitCodeForReport({ failed }),
        1,
        "a partial build must not be indistinguishable from an aborted one"
      );
    }
  });
});
