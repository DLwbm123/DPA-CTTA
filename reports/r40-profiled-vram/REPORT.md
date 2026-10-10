# R40: technical recovery with measured worker VRAM

Status: **COMPLETE**. Frozen primary: CW_LSO_GC.

R39 ended after a pre-arrival memory refusal;8 completed streams were unscored and no primary stream ran. R40 preserves the frozen R39 scientific method and full matrix, but formal worker admission reads same-round measured profile VRAM with the existing margin rather than the inherited5GiB fallback. This is technical recovery, not a new scientific module or outcome-based retuning.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_GC | 79.0299 | 78.4149 |

Primary CW_LSO_GC must gain>=0.3pp against both C/W, each with both orders positive,>=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. Additionally its SEARCH mean minus CW_LSO must be>=-0.05pp; that attribution comparison does not require every order or trajectory to be positive. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 6.807/48h; CPU-worker 0.159h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Full gate audit and residual failure

All24 online receipts complete1,951 arrivals each; all24 independent CPU receipts
complete1,951 rows each after ALL_WORKERS_RETIRED:46,824 scored rows, no failed
GPU attempt. No primary substitution. C/W and preservation gates are unchanged.

SEARCH primary-minus-C/W is+0.338359/+0.940271pp, both comparisons6/6 trajectories
positive and both orders positive. Imageweighted changes+0.697216/+0.547765pp.
Primary-minus-CW_LSO mean-0.015707pp passes its frozen-0.05pp noninferiority margin,
but is not an attribution gain. Its order changes-0.088631/+0.057217pp and3/6
positive trajectories are retained rather than interpreted as uniformly positive.

Worst seed-mean change versus W remains-3.638530pp (order0 REFUGE_Valid OC),
with OD-3.046319pp and order1 REFUGE OC-2.318174pp. Thus the full development gate
fails even though the mean/trajectory tests pass. Worst C cell-1.235359pp passes.
Compared with original CW_LSO worst W cell-5.155135pp, this is partial robustness
improvement, not complete success. All adverse cells/seeds/orders are delivered.

SEARCH gradient conflict frequency36.0701%, actual cap activation35.7096%, mean
scale0.838850, mean raw BN displacement0.00331399 and actual0.00275426. Mean cosine
0.055246, p10-0.162978, median0.057041, p900.274462; distribution is published in
CONFLICT_DISTRIBUTION.json. Conflict frequency in REFUGE_Valid is40.6652%/40.2116%
across orders, versus ORIGA30.1136%/31.6288%. This association does not establish
that conflict caused the residual accuracy losses. REVIEW remains descriptive.

A bounded next hypothesis is to add projection of the current BN gradient's
negative component against its preceding raw-gradient EMA, retaining orthogonal
components and the existing selective cap. No radius/EMA/threshold change or
domain-specific rule is implied. Projection could impede necessary adaptation,
and Adam preconditioning/moments do not guarantee a nonconflicting parameter
step. This needs a separately frozen full experiment before any success claim.

GPU-worker6.806752544972632h; CPU-worker573.7874252796173s; preparation CPU
10.074593305587769s retained separately. Original T0 and all4 profiles/24 formal
attempts retained. Cumulative5 closed rounds38.112564256257485 GPU-worker h,
including R39's pre-arrival resource failure. Independent generalization is absent.
