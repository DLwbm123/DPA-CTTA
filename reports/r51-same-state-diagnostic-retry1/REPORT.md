# R51 authorized retry: stopped before anchor restoration

**NOT_EVALUABLE_ANCHOR_PATH_DENIED. Formal pairs: 0/96.** The one user-authorized retry has been used. No second retry, replay, tuning, tolerance relaxation, sample expansion or scientific outcome is reported.

## Deployment repair and checks

The original artifact omitted non-Python resources. The corrected deployment includes every tracked file under `src/`, including `r8_ba/protocol.json`, plus the R9 and R10 planning JSONs read by transitive imports. The complete entry check exercises worker and scorer imports and pinned upstream model/augmentation interfaces in the deployed environment with CUDA disabled. It passed in 2.297180 seconds and explicitly verified CUDA was not initialized. Two preceding CPU-only import checks exposed the additional package/R10 omissions; those failures are retained in the resource ledger and private logs.

Five generated CPU checks passed (8.366954 seconds including interpreter/import overhead). The added regression check verifies preservation of the first failed attempt, subtraction of its conservative 2.555-second debit, and cumulative accounting. The executed source is `fab9798397c129c41109f00f851284af8719f04a`.

## Retry failure

The isolated retry directory reused the existing anchor files through symlinks to preserve the original failed directory without duplicating weights. The reused filesystem guard resolves paths before admission: the anchor resolved into the previous run directory, outside the current run's allowed root. It correctly raised `PermissionError` at the first anchor read. This was an integration/deployment error in the retry setup, not a checkpoint-corruption finding or a scientific negative result.

The import gate passed; the runtime access boundary is a separate check. GPU admission and source-weight loading were reached. Anchor restoration, real-model construction/forward/backward qualification, gradient matching, formal predictions and label scoring were not reached. No guard or scientific threshold was weakened after the failure.

## Resources and preserved history

The retry consumed **3.218337 GPU-worker wall seconds**. Adding the conservatively rounded original debit of **2.555 seconds** gives **5.773337 seconds / 1800 seconds** (0.001603705 / 0.5 GPU-worker hours). Wall-time charging includes imports and failures, and does not represent GPU kernel execution time. The raw first measurement (2.554733992 seconds) remains in the cumulative ledger. Remaining budget does not authorize another attempt.

The controller and worker exited; a terminal process/GPU-PID check found no remaining owned processes. The [first failed attempt](../r51-same-state-diagnostic/REPORT.md), its receipts and remote directory remain intact. CPU preparation failure times other than the successful checks were not measured. No diagnostic prediction/vector artifacts or label scores were produced.

## Conclusion and public scope

Recovery consistency, real full-gradient matching and all direction/magnitude effects remain **NOT EVALUABLE**. There is no Dice estimate, interval or evidence for/against joint sensitivity from this retry. Any eventual valid interpretation would still be limited to the two frozen late states and the same 48 historically exposed contents.

Published: own source changes, frozen protocol/config without private paths, aggregate import/CPU checks, both attempt costs and failure audit. Excluded: raw logs/private paths, images, labels, weights, content identities and original private journals. No further execution is authorized or scheduled.
