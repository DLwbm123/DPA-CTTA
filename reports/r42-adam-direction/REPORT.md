# R42: actual Adam descent projection

Status: **COMPLETE**. Frozen primary: CW_LSO_GDP.

R41 raw-gradient projection worked mechanically but failed the W worst-cell guard at -4.126838pp. R42 adds only post-Adam descent projection against preceding raw-gradient EMA, preserving GP losses, views, moments and selective cap. Applied-step interference is an untested mechanism, not a proven cause of R41 negatives.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO_GP | 79.0360 | 78.4223 |
| CW_LSO_GDP | 79.0360 | 78.4223 |

Primary CW_LSO_GDP must gain>=0.3pp against both C/W, each with both orders positive,>=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. Additionally SEARCH mean minus CW_LSO_GP must be>=-0.05pp; this extra comparison is mean noninferiority only. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 7.658/48h; CPU-worker 0.150h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Verified result, mechanism falsification and retained negatives

All 24 unique trajectories completed and retired, and 24 independent CPU jobs
scored all 1,951 arrivals each (46,824 rows). No profile/formal/scoring failures.
GPU-worker cost 7.657656830814149h; CPU-worker 540.6452205181122s; execution wall
14,237.097470521927s. Seven closed rounds cumulatively cost
52.744737940364416 GPU-worker h, including past profiles and failed/interrupted
attempts. The CPU preparation interpreter failure (estimated dispatch wall
2.350775s, GPU0) remains separate; successful CPU checks used 16.776557s.

SEARCH GDP-minus-C/W +0.344417/+0.946329pp, both orders positive, 6/6 positive,
imageweighted +0.590096/+0.440645pp. Worst seed-mean cell versus C -0.750358pp,
versus W -4.126838pp: unchanged full development gate FAIL. Other W deficits:
order0 REFUGE_Valid OD -3.595496pp; order1 REFUGE_Valid OD/OC -1.713852/-1.164867pp,
REFUGE OC -2.643058pp. Worst individual W cell -4.910068pp; seed29009/order0
imageweighted minus W -0.243704pp. All negative cells remain reported.

GDP-minus-GP SEARCH mean/imageweighted/order deltas are all zero, 6,102 paired
observations tied, 0/6 strictly positive. It satisfies only the extra mean
noninferiority margin; this does not relax C/W guards or establish added benefit.
Metric ties alone are not a byte-level prediction or state identity verification.

The resolved GDP flag was true. The new actual-step projection triggered on
0/11,706 formal arrivals (and 0/6,102 SEARCH arrivals); removed norm was zero.
With preceding EMA ready, 6,099 SEARCH raw-descent/reference dot products were
strictly positive: minimum 1.9898470782209188e-5, mean 0.0002857243974694427.
The GP control had the same distributions. Thus the registered interference
mechanism was not observed on this stream; the synthetic Adam sign-reversal
example established only mathematical possibility. It did not justify a practical
benefit. Pre-cap projected descent equals raw descent because the operation never
activated; its distribution was not independently logged. Raw/actual step and
raw-gradient/EMA/cap distributions are in ADAM_DIRECTION_DISTRIBUTION.json.

No further radius/EMA/threshold/strength scan or repeated projection run is
justified by these diagnostics. A follow-up must have a distinct, concrete,
evidence-supported hypothesis. None is ready for launch at this delivery:
continuation is BLOCKED_NO_NEW_EVIDENCE_SUPPORTED_HYPOTHESIS, not stage success.
No R43, RL run, expanded evaluation set or additional seed has been started.
Historical REVIEW is descriptive only and did not guide this decision;
independent generalization remains untested. Current GPU experiments are retired.
