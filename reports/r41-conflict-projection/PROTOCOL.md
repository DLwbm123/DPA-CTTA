# R41: one conflict-component projection on retained selective trust

User authorized autonomous evidence-driven non-RL increments on2026-10-09.
R40 completed24 trajectories/46,824 CPU scores and verified full delivery before
R41. Freeze this round before GPU profiles/formal execution/scoring. Old ledgers
and original T0 remain unchanged; no unsuccessful cell or seed is removed.

## SEARCH evidence and single hypothesis

R40 GC-minus-C/W+0.338359/+0.940271pp, both orders positive,6/6 positive trajectories;
imageweighted+0.697216/+0.547765pp. Mean versus CW_LSO-0.015707pp passes-0.05 margin.
Full gate fails because worst W cell is-3.638530pp: order0 REFUGE_Valid OC, OD
-3.046319pp and order1 REFUGE OC-2.318174pp. It improves on original CW_LSO's
-5.155135pp worst cell but does not pass. No REVIEW-based choice.

Conflict36.0701%, cap activation35.7096%, mean scale0.838850, raw displacement
0.00331399/actual0.00275426. REFUGE_Valid conflicts40.6652%/40.2116%, ORIGA
30.1136%/31.6288%. Association is not causality. Hypothesis: removing only the
negative gradient component against recent history preserves orthogonal
adaptation while reducing residual drift beyond selective norm clipping.
Gradient opposition can represent a necessary shift; projection may hurt, and
Adam preconditioning/moments do not imply a guaranteed nonconflicting step.
Own explicit gradient projection heuristic, not a reproduction or efficacy claim.

## Frozen method and matrix

Exactly C/W/CW_LSO_GC/CW_LSO_GP, seeds[20260907,17011,29009], both original orders,
1,951 arrivals each:24 fresh complete trajectories/46,824 formal rows. C/W/GC
are unchanged controls from R40. Primary GP adds only gradient-component projection
to GC. Radius0.0015/EMA0.9/negative-cosine trigger/LSO0.1 unchanged; no scan.

Before ordinary BN Adam, let g be the raw BN-affine gradient and m the preceding
raw-gradient EMA. Retain original GC cosine trigger and raw EMA update first.
If m exists and ||m||^2>1e-12, set g'=g-min(0,<g,m>/||m||^2)*m; otherwise g'=g.
Write g' to the same parameter gradients before Adam. Aligned/first/tiny-reference
steps remain unchanged. EMA still stores raw g, not projected g. No coefficient
or domain-specific rule; full removal of the opposing component is fixed.

Adam moments now update with g', rather than raw g; step counters/LR and algorithm
are unchanged. Existing post-Adam GC cap still uses the raw conflict flag and
displacement norm. Native final forward follows that cap. No extra model forward,
backward or optimizer step; no source images/retraining, online labels, teacher,
fusion, new evaluation set or RL. No extra persistent memory beyond retained EMA;
snapshot/restore retains complete raw EMA/BN/Adam/RNG states.

## Gate, diagnostics, validation and cost

Same historical SEARCH1017/REVIEW678; REVIEW descriptive only. All online workers
retire before separate CPU masks. Same hard Dice/both-empty1/equal domains and
OD/OC,3 seeds/2 orders. GP must exceed each C/W>=0.3pp, each both orders positive,
>=5/6 positive trajectories,imageweighted>=0,worst seed-mean domain/channel/order
cell>=-2pp. Also GP-minus-GC SEARCH mean>=-0.05pp; mean noninferiority only, all
negative order/trajectory/cells kept. Frozen primary; C/W gates never relaxed.
No independent generalization from repeated development exposure.

Publish all positive/negative cells and content bootstrap2000; patient linkage
UNKNOWN. SEARCH projection activity/removed norm/projected reference dot/norm,
conflict/cosine/EMA-ready and cap activity/scale/raw/actual displacement distributions.
No ASSD/Brier/soft Dice. Validate mathematical orthogonal retention/aligned and
zero-reference parity, exact disabled/full first-step output/state parity, forced
actual projection/finite/frozen parameters and snapshot continuation. Reuse GC,
publication/gates and R40 memory-guard tests. All4 GPU12-arrival profiles and whole
matrix admit with20% timing reserve. Use R40 same-round measured worker memory
guard including existing20%/512MiB margin; invalid profiles refuse.

At most3 workers, sufficient checked memory, neutral command lines, prescribed
mounted storage. Original T0 before profiles;22h online/24h wall/48GPU-workerh/
32GiB. All profile/formal/failed GPU attempts charged, preparation CPU separate.
Prior5 closed rounds38.112564256257485 GPU-worker h, including R39 resource failure.
No pruning, scientific retries, healthy worker restart, threshold tuning or budget
reset. Persistent state resets once per registered fresh trajectory only.

## Delivery and continuation

Publish own source/protocol and all anonymous results/negatives/costs/audits on
experiment/r41-conflict-projection-v1 in DPA-CTTA. Every GitHub operation uses
explicit local proxy; verify remote SHA and anonymous report before DELIVERY.
Exclude private images/labels/per-content predictions/scores, patient/content
identifiers or hashes, model states, private paths/logs and third-party PDFs.
Continue authorized hourly necessary checks; notify meaningful changes only.
Pause only after frozen development gate and verified delivery. Future changes
need a new concrete evidence-driven separately registered bounded hypothesis.

## User resource amendment, 2026-10-10

The user withdrew physical GPU 2 during the round and restricted this and future
work to physical GPUs 0 and 1. The GPU-2 worker was terminated with SIGTERM; its
partial output, original attempt receipt and full cost remain private and retained.
The two healthy workers on 0/1 retire normally. Completed trajectories are retained.
After controller/worker retirement, a dedicated resource continuation runs only
pending frozen jobs on 0/1. The sole user-interrupted job restarts once from its
original initialization as attempt 1; its old output is archived without deletion.
No other failed job may retry. No new profiles, method changes, score selection,
extra trajectories, scoring before retirement, T0 reset or budget extension.
Remaining full-matrix admission is checked against the original deadlines and
GPU cost including the interrupted attempt. This is an explicit user-requested
resource change, not a scientific retry selected by observed performance.
Original scientific config/provenance remains unchanged; the separate operational
driver revision and resource amendment are recorded. The final report must include
the user interruption, one repeat attempt and both costs.
