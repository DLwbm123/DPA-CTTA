# R25 experiment startup

**RUNNING — effectiveness and final results are not yet available.**

Code: `bacbbf959a0b679f99f45de47b3e0143d80537e6`.
[Registered protocol](../../docs/protocols/R25_PARAMETER_POLICY.md).

The campaign began on 2026-10-06 at 20:31:10 Asia/Shanghai. It targets about
24 hours; conservative profiling estimates 26.96 hours. Online work stops no
later than October 8 at 18:31:10, and the absolute runtime deadline is October 8
at 20:31:10. The detached watchdog enforces the deadline. Hourly monitoring is active.

Ten conditions are frozen across three seeds (20260907,17011,29009) and both
stream orders: **60 full trajectories, 1,951 arrivals each, 117,060 formal arrivals**.
There is no performance screen, label-based pruning, or retuning. The primary is
PG_RETAIN; C is the main baseline.

| Priority | Conditions | Purpose |
|---|---|---|
| Core | C, ANCHOR, REPEAT5 | Existing baseline/adapter and additional-update control |
| Core | RANDOM4, GREEDY | Separate random exploration from reward-based selection |
| Core | PG_CONS, PG_STRUCT, PG_RETAIN | Learned parameter proposals, structural evidence, historical retention |
| Extension | PG_COV, RN_DPO | Correlated parameter distribution and disagreement-region preference learning |

Twelve grouped CPU mechanical checks passed, including exact C/ANCHOR parity,
proposal rollback, one committed branch, policy gradient direction, covariance,
region-normalized loss and all-arm exact continuation. All ten real 12-arrival
GPU profiles passed. The first four formal workers are producing predictions;
the startup receipt recorded 74–94 arrivals each. Process identity and neutral
command lines were verified. No target labels or source training images were opened.

The Gaussian policy is a contextual bandit. Threshold-component OT is a
TopoOT-inspired adaptation, not persistence-diagram reproduction; correlated
exploration does not reproduce EVON. Region preference learning uses an unlabeled
structural judge, not the original RN-DPO supervised quality model. These are
explicit mechanism-transfer experiments, not full-paper reproduction claims.

All labels remain closed until every formal online worker exits. Primary scoring
uses 512px hard Dice; private 64px proposal masks support retrospective reward
calibration only. The content is historically exposed, with patient linkage and
ROI provenance unknown. No independent generalization, clinical validity or novelty
claim follows from startup. Final paired results, adverse cells, costs and report
will be published after actual completion; private data and predictions stay private.
