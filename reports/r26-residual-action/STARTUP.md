# R26 startup: bounded residual-output actions

Status: **RUNNING — startup only, no scientific results yet.**

R25 completed and its full positive/negative results are preserved in
[the R25 report](../r25-parameter-policy/REPORT.md). Its fixed primary failed the
development gate. R26 tests whether moving four-dimensional actions beyond the
near-saturated context gate creates useful candidate separation. This is a
hypothesis-driven development follow-up, not an independent confirmation.

Code: `1b01ca6fe21a7e373480faedeb8d5e16c2854fce`.
[Protocol](../../docs/protocols/R26_RESIDUAL_ACTION.md): six arms x three seeds x
two full orders =36 trajectories,70,236 arrivals. PG_RETAIN remains the fixed
primary. C, ANCHOR and REPEAT5 retain exact existing behavior; RANDOM4, GREEDY and
PG_RETAIN use bounded residual-output actions. No performance-based pruning,
additional seeds, or labels during online execution.

Eight grouped mechanical checks passed, including exact unchanged-control parity,
zero-action parity, finite/bounded actions, rollback/replay equality, a single
committed branch, frozen model checks, null-cap operation and legacy finite-cap
enforcement. All six real GPU engineering profiles passed. These establish
engineering readiness, not efficacy. Four first-wave formal workers were observed
with matching process identities and live heartbeats; their executable/entry names
were neutral. Other GPU tasks were left untouched.

T0: **2026-10-07 22:15:56 Asia/Shanghai**. Profile-based conservative online runtime
estimate approximately **15.5 hours**, subject to load and full-stream speed.
There is no cumulative GPU-worker cap. All costs and failures remain recorded;
per-task hang guards,64GiB storage limit and a seven-day supervisory safety timeout
remain. Hourly monitoring continues. Reports are generated only after all online
workers retire and independent CPU scoring completes.

SEARCH and the legacy SEALED_REVIEW field both denote previously exposed
**development data** in this round. Passing the development gate would not establish
novelty, independent generalization, patient independence or clinical validity.
No images, labels, prediction masks, identities, adapted weights, credentials or
private paths are included here. Final results and final delivery remain pending.
