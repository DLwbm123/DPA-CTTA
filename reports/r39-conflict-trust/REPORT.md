# R39: gradient-conflict selective BN trust cap

Status: **INCOMPLETE**. Frozen primary: CW_LSO_GC.

R38 cap activated on97.7712% of SEARCH steps and lost1.288808pp versus retained CW_LSO. R39 keeps radius0.0015 but caps only BN gradients opposing their preceding raw-gradient EMA (decay0.9); aligned and first steps remain unchanged. This selective-activation benefit is an unverified development hypothesis.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | NA | NA |
| W | NA | NA |
| CW_LSO | NA | NA |
| CW_LSO_GC | NA | NA |

Primary CW_LSO_GC must gain>=0.3pp against both C/W, each with both orders positive,>=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. Additionally its SEARCH mean minus CW_LSO must be>=-0.05pp; that attribution comparison does not require every order or trajectory to be positive. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 2.100/48h; CPU-worker 0.000h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

Stopped reason: registered task failed; no automatic scientific retry

## Resource failure; scientific result not evaluable

All online workers have retired. Eight trajectories completed (six C, two W),
one W trajectory failed before any target arrival, and fifteen were not run.
No primary trajectory ran, no scores were produced and REVIEW was not released.
The development gate is NOT_EVALUABLE, rather than evidence against the method.
All missing scores/cells remain NA; no incomplete value is treated as zero.

The failed worker consumed4.481051 GPU-worker seconds, included in total
7,561.221031 seconds (2.1003391754h); CPU scoring0. Preparation CPU33.206134s
including all three preparation failures remains separately recorded. Original
T0 and all attempt records are retained. R39 is not restarted or retuned.

The inherited worker admission used a default5GiB peak and demanded at least
6GiB free, rather than reading the completed profile measurement. All R39
profile peak reservations were below1GiB. A subsequent resource check found
5,884MiB free on the affected GPU, below the inherited threshold; the exact
failure-time free memory was not logged. It is therefore a resource admission
failure, with an identified overly conservative fallback after profiling, not
an observed CUDA out-of-memory event or a scientific negative result.

A separately registered technical recovery may use each condition's measured
profile peak plus the existing20%/512MiB safety margin for formal worker
admission. Algorithms, radius, EMA, seeds, orders, evaluation and gates stay
unchanged. Fresh full matrix and its own budget will be used, charging R39 in
cumulative cost; no scores or favorable conditions are selectively carried over.
