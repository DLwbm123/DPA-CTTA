# R51 second authorized retry: native-state qualification failed

**NOT_EVALUABLE_NATIVE_STATE_MISMATCH. Formal pairs: 0/96.** The newly authorized single retry was executed after CPU preflight passed. It stopped at the first anchor's exact native-parameter equivalence check. No further retry, replay, tuning, relaxed threshold or sample expansion occurred.

## Path repair and actual CPU evidence

Eight fixed private inputs were copied as real files into the new run root. Both earlier run directories and failure evidence remain unchanged. No scientific algorithm, threshold, content selection or anchor changed. CPU preflight and the GPU worker share the same snapshot loader.

The complete deployed entry and worker/scorer/upstream imports passed with CUDA disabled (1.952001s). Actual paths resolve inside the new root. Both anchors passed the existing journal integrity checks and CPU deserialization, and source weights were readable. The unchanged access guard rejected scorer data, the old anchor, an escaping symlink and an unregistered image; an explicitly registered image byte read succeeded (boundary check 0.374713s). There was no model forward, label read or CUDA initialization in preflight. Five generated CPU checks passed in 6.401113s including process startup; these cover cumulative accounting that retains both previous failures.

Executed source: `886491b6673caa920eb71080c75c0e6779c1dc65`. The path problem is resolved: real-model qualification was reached.

## Real-model qualification and failure boundary

For anchor order 0, execution passed exact snapshot restoration, historical binary-prediction equality, native-step/persisted-state repeatability, and instrumented/native binary-prediction equality. The diagnostic also returned from its full-gradient decomposition and representable-write matching checks without an error. These preliminary statements follow from the fail-closed execution order and the final traceback; a completed qualification row was not saved before the later failure, and no scalar matching measurements are claimed here.

The next check, `assert_native_equivalence(reference, host.snapshot())`, raised **`ValueError: diagnostic changed native state: parameters`**. This compares native state after the instrumented diagnostic with the original native step. Parameter equality is the first compared component, so later components (buffers, gradients, Adam, RNG, counters and modes) were not certified. Anchor order 1 was not attempted. The two-anchor qualification gate therefore failed and the formal matrix was never admitted.

The numerical size and exact cause of the parameter discrepancy were not measured or established. Matching binary masks does not imply equivalent parameters. This is an engineering/qualification failure, not evidence against the scientific joint-direction hypothesis. The partial first-anchor gradient checks are not a passed full qualification gate. The first qualification's saved private vectors remain preserved; no formal predictions, per-content scores or labels were produced/read.

## Costs and retirement

New worker charge: **24.761302948 GPU-worker wall seconds**. Previous carried debit: **5.773336821 seconds**, preserving the more precise prior accounting rather than rounding down to 5.773. Cumulative charge: **30.534639769 / 1800 seconds** (**0.008481844 / 0.5 GPU-worker hours**). All three worker attempts remain in `RESOURCE_LEDGER.json`; imports, initialization, qualification and failures count. Remaining time does not authorize another retry.

The controller and worker exited. A final owned-PID and GPU-process check found no remaining processes. Other preparation CPU time was not measured. No automatic monitoring or future execution was scheduled.

## Conclusion and preserved evidence

Formal coverage remains **0/96**. No Dice effect, confidence interval or continuation criterion can be evaluated. No scientific conclusion follows for either fixed late state, and none follows for earlier states or long-term streams. Further runtime work requires new authorization; this report does not alter the frozen gates.

Preserved earlier records: [initial import failure](../r51-same-state-diagnostic/REPORT.md), [first retry path failure](../r51-same-state-diagnostic-retry1/REPORT.md). Published scope: own code, frozen protocol/config without private paths, CPU checks, partial qualification evidence, cumulative costs and failure audit. Raw logs, private paths, data, labels, weights, content identities and qualification vectors remain private.
