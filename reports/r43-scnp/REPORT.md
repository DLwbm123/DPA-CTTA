# R43: same-class neighbor loss on retained CW_LSO

Status: **COMPLETE**. Frozen primary: CW_LSO_SCNP.

Own unlabeled adaptation of CVPR 2026 SCNP. Retain CW_LSO and add fixed 0.1 weighted soft-target BCE on 3x3 worst same-pseudo-class logits. Original supervised SCNP uses ground truth; weak-view membership may propagate pseudo errors. No GC/GP, new views or additional optimizer step.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_SCNP | 79.0511 | 78.4181 |

Primary SEARCH gain against C/W each>=0.3pp and against CW_LSO>=0.1pp. All three comparisons require both orders positive,>=5/6 trajectories positive,imageweighted>=0,worst seed-mean cell>=-2pp. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 8.104/48h; CPU-worker 0.166h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Interpretation of the fixed SCNP increment

The CW_LSO foundation is reproduced in this fresh run (SEARCH 79.045619%);
SCNP reaches 79.051085%. Against CW_LSO, the incremental mean is only
+0.005466pp (2000-repeat coupled-content 95% CI [-0.001187,+0.013486]pp),
far below the frozen +0.1pp minimum. Order0/order1 deltas are
-0.000985/+0.011917pp; 4/6 trajectories are positive, below the required 5/6.
Imageweighted delta +0.050470pp is positive but cannot replace the primary
macro objective. Thus this fixed pseudo-SCNP variant has not demonstrated a
meaningful incremental benefit. No coefficient search follows this result.

Primary-minus-C/W means +0.359532/+0.961444pp and each has two positive orders
and 6/6 positive trajectories. However, the worst seed-mean W cell remains
-4.927264pp (order0 REFUGE_Valid OC), failing the -2pp guard. Its retained
CW_LSO counterpart is -5.155135pp: improvement +0.227870pp does not resolve
the adverse cell. Order0 REFUGE_Valid OD is also adverse at -4.586702pp.
The primary-minus-C comparison passes its frozen requirements; the W robustness
and CW_LSO incremental comparisons fail. Full stage success is false.

SCNP operates mechanically: SEARCH mean changed-logit fractions are
88.9554% OD / 87.3117% OC; mean absolute transformed-logit displacement is
0.871637/0.991559. Mean added BCE is 0.058568 (weighted contribution 0.005857);
base BCE is 0.054995 and LSO is 0.068913 (weighted contribution 0.006891).
These scalar losses are not gradient contribution measurements. Branch activity
rules out an entirely disabled module, but does not establish correct topology
or a causal explanation for the small Dice change. Original inference still uses
untransformed logits. No direct connectivity/hole or boundary-distance outcome
was evaluated, and this pseudo-label adaptation cannot inherit supervised SCNP's
reported topology benefits. Pseudo-membership propagation remains an untested
mechanistic possibility, not an established cause of the negative result.

SEARCH incremental improvements concentrate in order0 REFUGE_Valid OD/OC
(+0.322225/+0.227870pp), while order0 Drishti_GS OD/OC deteriorate
(-0.215068/-0.156005pp). These offsetting cells help explain the small aggregate
increment descriptively, without establishing a mechanism.

INCREMENTAL_DOMAIN_CHANNEL.csv reports every mean domain/channel/order delta
against CW_LSO, including negative cells and descriptive historical REVIEW.
REVIEW is not independent validation or a basis for tuning. All original aggregate
results, confidence intervals, mechanism means, negative cells, costs and attempts
are retained. No private per-content rows or identities are published.

## Verified execution and delivery boundary

All 24 unique online jobs and 24 independent CPU scoring jobs completed; each
scorer produced 1951 rows (46824 total). Four profiles plus 24 online and 24 CPU
score attempts all completed at attempt0; no retries or failures. Live active
workers are zero. GPU-worker cost including profiles is 8.104409671094682h;
CPU scoring is 598.7469563484192 worker seconds, with CPU preparation separately
33.13669395446777s and GPU0. Original-T0 wall time recorded by the final ledger
is 4.238793h. Eight closed rounds now cost 60.8491476114591 GPU-worker hours.
Code/protocol and startup precede execution; this report and complete aggregate
results form the scientific closeout. Remote commit and anonymous HTTP access
must be verified before DELIVERY is marked complete. Stop the R43 monitor after
verified delivery; no automatic R44 is authorized by this round.
