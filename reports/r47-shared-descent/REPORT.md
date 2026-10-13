# R47 descent-constrained shared sensitivity: terminal failure report

**Scientific status: NOT_EVALUABLE. Execution status: INCOMPLETE / FAILED_NUMERICAL.**
No complete DESCENT trajectory or independent CPU score exists. Mean Dice,
paired confidence intervals, and domain/channel/seed/order improvements against
C, W, CW_LSO and R46 SHARED are all unavailable. This is an execution failure,
not evidence of inferior segmentation and not a failed fully evaluated gate.

## Coverage and retirement

| Condition | Complete trajectories | Failed | Not run | Independent scores |
|---|---:|---:|---:|---:|
| C | 6 | 0 | 0 | 0 |
| W | 6 | 0 | 0 | 0 |
| CW_LSO | 6 | 0 | 0 | 0 |
| CW_LSO_JS_SHARED | 6 | 0 | 0 | 0 |
| CW_LSO_JS_SHARED_DESCENT | 0 | 2 | 4 | 0 |

All 24 completed controls contain 1,951 retained visits each (46,824 visits).
The candidate retained 15 and 41 successful visits; its failing updates were
steps 16 and 42 of seed20260907/order0 and order1 respectively. Both original
attempt0 receipts, partial outputs, checkpoints and costs remain private.
Five same-round profiles passed; 29 attempts completed (5 profiles+24 controls),
two failed, and no CPU scoring started. All online workers, watch and supervisor
retired; fresh ps and nvidia-smi checks found no R47 process. No restart, retry,
parameter change, tolerance relaxation, extra scoring or next round occurred.
The expected frozen coverage remains 30 trajectories/58,530 scores, not achieved.

## Failure and evidence limits

Both exceptions were `real-model first-order loss descent violated`. The frozen
check is g^T delta_actual <= g^T delta_Adam + 1e-5*max(abs(g^T delta_Adam),1e-10)
when the native proposal derivative is negative. It demands preservation of the
native first-order descent amount, not merely a negative derivative. A failure
could still have a negative actual derivative; the exact failing values and sign
were not persisted because the wrapper raised before writing the visit record.
The failed step counts are receipt-backed; the precise offending residual and
its cause are unavailable. We did not replay checkpoints or run another GPU job.

Code inspection provides a specific numerical hypothesis: the active closed-form
correction lies on the constraint boundary, then the float64 calculation is cast
to the native update dtype and old+filtered is written back to parameters. The
committed displacement is measured again by subtracting the old parameters.
Those operations can change the directional derivative through rounding. This
is a plausible boundary-rounding cause, not confirmed diagnosis. There is no
finite-precision feasibility margin or repair in this frozen implementation.
The old exact-arithmetic KKT/disabled-parity tests and 12-image profiles did not
establish this property for every real-model committed update. We kept the guard
and frozen tolerance rather than accepting a violated update.

## Retained execution diagnostics (no efficacy interpretation)

FAILURE_DIAGNOSTICS.json aggregates all successfully logged pre-failure steps,
without reading labels, prediction values, content IDs or per-image scores.
These are execution aggregates, not SEARCH metrics or population trigger rates.
On the candidate's 56 successfully logged steps: 48 active constraint corrections,
7 inactive filtered updates, one exact native fallback for a non-descent proposal.
Thus 48/56=85.7143% triggered (48/55=87.2727% among non-fallback filter steps).
Shared rank was4 on55 steps and0 on the fallback. Mean relative metric update
change on55 filtered steps was0.361665, median0.310072, maximum1.541735.
Metric contraction is not required for the constrained update; exceeding the
proposal norm is not itself an error. Both acquisition/anatomy capture metrics
are construction diagnostics and do not demonstrate true anatomical protection.
The unconstrained derivative can be reconstructed as proposal_dot+gap; K
spectra describe the unconstrained filter, not the affine constrained operator.

All55 logged negative-proposal steps passed the committed derivative check.
Their largest relative committed-minus-proposal gap was7.88987997075e-6, near but
below the frozen1e-5 limit. This demonstrates rounding-sensitive observed gaps
among retained successes; it does not recover the two failing-step values.
Full aggregate distributions of derivatives, multiplier, rank, K eigenvalues,
capture, span energy and update norms are in FAILURE_DIAGNOSTICS.json.
REFUGE_Valid preservation and Drishti_GS/ORIGA costs cannot be assessed in R47.
No scientific positive or negative cell was suppressed: none was scored.

## Cost, frozen boundaries and delivery

Original T0:2026-10-13T03:08:57.368536+00:00. Terminal stop was
2026-10-13T06:44:36.283645+00:00 (Beijing14:44:36), wall12,939.013878s.
GPU-worker cost:25,652.225068s / **7.1256180744h**, including both failed attempts
(26.942078s+49.317671s) and all same-round profiles. CPU scoring cost0;
server preparation20.887793s and local preparation19.338s are recorded separately.
Post-stop aggregate-analysis CPU wall time is separate in FAILURE_DIAGNOSTICS.json;
its initial read-only pass took1.714605s and the final pass is separately recorded.
No old T0 or ledger was rewritten. Closed experimental rounds now total12;
closed-round GPU cost98.8103732082h, plus the two prior independent profiles
198.002250s gives cumulative98.8653738331 GPU-worker h. Physical GPU0/1 only.

Own source/protocol, anonymous attempt costs, coverage audit and this failure
analysis are delivered on experiment/r47-shared-descent-v1. Startup evidence is
historical and does not establish completion or efficacy. No private images,
labels, per-content predictions/scores, identifiers or their hashes, model states,
private paths/logs, or third-party PDFs are public. SEARCH/REVIEW were historically
exposed; no independent generalization claim, clinical claim or anatomy-protection
claim follows. Finite-step loss, ASSD/Brier and patient generalization are untested.
After remote SHA and anonymous report verification, monitoring is removed. No R48.
