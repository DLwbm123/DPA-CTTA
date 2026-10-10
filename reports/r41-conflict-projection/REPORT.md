# R41: conflict-component projection on retained selective trust

Status: **COMPLETE**. Frozen primary: CW_LSO_GP.

R40 selective trust preserved positive SEARCH means but failed the worst-cell guard at-3.638530pp versus W. R41 adds only removal of the negative current-gradient component against preceding raw-gradient EMA, retaining orthogonal components and the selective0.0015 cap. Raw EMA and losses/views are unchanged; Adam moments consume projected gradients. Benefit remains an unverified development hypothesis.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO_GC | 79.0299 | 78.4149 |
| CW_LSO_GP | 79.0360 | 78.4223 |

Primary CW_LSO_GP must gain>=0.3pp against both C/W, each with both orders positive,>=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. Additionally its SEARCH mean minus CW_LSO_GC must be>=-0.05pp; that attribution comparison does not require every order or trajectory to be positive. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 6.975/48h; CPU-worker 0.152h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

User requested withdrawal of GPU 2 during execution. One interrupted trajectory was archived and restarted from original initialization on 0/1. Both attempt costs are retained; completed trajectories, frozen science and the original T0/budget remain unchanged. No other retry is permitted.

## Verified outcome and negatives

All 24 unique online trajectories retired; 24 independent CPU score jobs covered
1,951 arrivals each: 46,824 scored rows. The sole user-withdrawal interruption used
111.286 GPU-worker seconds, included in total 6.9745168532927835 GPU-worker h.
CPU worker time 547.5917465686798 seconds; execution wall 11,461.607 seconds.
Six closed rounds now total 45.08708110955027 GPU-worker h, including all profiles
and failures. No other attempt failed. The two-card continuation retained the
original configuration, T0, cost limits and all interrupted output evidence.

SEARCH GP-minus-C/W +0.344417/+0.946329pp, both orders positive, 6/6 positive;
imageweighted +0.590096/+0.440645pp. Worst seed-mean cell versus C -0.750358pp,
but versus W -4.126838pp (order0 REFUGE_Valid OC): the -2pp guard fails.
Other W deficits: order0 REFUGE_Valid OD -3.595496pp; order1 REFUGE_Valid
OD/OC -1.713852/-1.164867pp and REFUGE OC -2.643058pp. Worst individual cell
-4.910068pp; seed29009/order0 imageweighted minus W -0.243704pp.

GP-minus-GC mean +0.006058pp passes the -0.05pp margin but adds little: orders
+0.051266/-0.039150pp, 3/6 positive, imageweighted -0.107120pp, worst seed-mean
cell -0.549177pp. Its worst W cell worsens from GC -3.638530 to -4.126838pp.
All adverse seed/order/domain/channel cells remain in ALL_NEGATIVE_CELLS.

SEARCH projection activation 36.3324%; projected-gradient/reference dot minimum
-5.59e-9, with none below -1e-8 (floating-point roundoff). The gradient operation
worked mechanically, without establishing robustness. Raw Adam displacement
mean 0.00351955 versus GC 0.00331399; actual displacement 0.00285547 versus
GC 0.00275426. Nonopposing raw gradients do not guarantee nonopposing Adam steps.
Applied-step alignment was not logged: it is an untested mechanism, not a proven
cause. REVIEW remains descriptive and did not guide follow-up; no independent
validation has been performed.
