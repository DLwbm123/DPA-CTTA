# R38: fixed BN displacement trust cap

Status: **COMPLETE**. Frozen primary: CW_LSO_TR.

R37 CW_LSO improved C and W on average but failed the worst-cell guard. R38 retains the loss/views/Adam state updates and adds only a post-Adam BN displacement cap of0.0015. The radius is frozen from exposed development diagnostics, not REVIEW; the step-size mechanism remains a hypothesis.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0447 | 78.4123 |
| CW_LSO_TR | 77.7559 | 77.2184 |

Primary CW_LSO_TR must gain>=0.3pp against both C/W, each with both orders positive,>=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. Additionally its SEARCH mean minus CW_LSO must be>=-0.05pp; that attribution comparison does not require every order or trajectory to be positive. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 7.065/48h; CPU-worker 0.150h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Complete negative result and mechanism check

All24 online receipts completed1,951 arrivals each and all24 independent CPU
score receipts completed1,951 rows each:46,824 rows, after the retirement barrier.
No profile or formal attempt failed. All positive and negative cells are retained.

SEARCH primary-minus-C/W/CW_LSO is-0.935696/-0.333784/-1.288808pp, with0/6
positive trajectories for every comparison. The W worst seed-mean cell improves
from-5.155135pp for CW_LSO to-1.477074pp for the capped candidate, but the primary
still fails both mean gates and its preservation gate. Against C the worst cell
is-6.582720pp; against CW_LSO it is-7.325857pp. No gate is relaxed.

The cap activates on5,966/6,102 SEARCH steps (97.7712%). Its mean scale is0.496073,
mean raw displacement0.00369704 and actual displacement0.00149789. The cap is
working mechanically; broad suppression did not preserve useful adaptation.
In particular, SEARCH ORIGA OD/OC lose about6.00-7.33pp versus CW_LSO across
orders, and order0 Drishti OD/OC lose4.28/5.56pp. These are development observations,
not evidence that step size alone caused the original failures.

The retained CW_LSO control gains0.353112pp over C and0.955024pp over W. Its
mean differs from R37 by about0.000954pp, a small cross-run numerical
difference whose cause is not established. R38 uses its own paired controls
rather than borrowing R37 scores.
The same worst W cell remains-5.155135pp. No independent generalization is claimed.

A possible next bounded hypothesis is to apply the same radius only when the
current BN gradient conflicts with its preceding exponential average. R38's
near-universal cap motivates selective activation; no conflict benefit is yet
measured, and a selective cap may still prevent necessary adaptation after a
shift. This would require a separately frozen round, not a radius scan in R38.
