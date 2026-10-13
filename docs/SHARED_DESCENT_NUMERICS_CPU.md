# Shared-descent committed-precision repair: CPU only

User authorized numerical diagnosis, repair, failure-derivative recording and CPU
qualification. No new GPU profile, deployment, experiment, monitor or retry.
Frozen R47 source fea5ca1, stopped attempts, protocol, receipts and costs remain
unchanged. This change lives on feature/shared-descent-numerics-cpu.

## Reproduced failure class and unchanged mathematical update

R47 checked the derivative of the actually committed displacement, not just the
float64 closed-form solution. The active KKT correction lies on the boundary;
casting to parameter dtype, adding it to old parameters, and subtracting those
parameters again can violate the original inequality after rounding.

A deterministic CPU float32 fixture (seed0,128 coordinates, old parameter1,
native displacement approximately1e-4, noncommuting random sensitivities) gives:

| Quantity | Directional derivative |
|---|---:|
| Native Adam proposal | -1.3849359667972294e-6 |
| Returned closed-form update | -1.3849359749531945e-6 |
| Old implementation's committed displacement | -1.3849081170747013e-6 |

The committed-minus-proposal gap2.784972253e-11 exceeds the original tolerance
1.3849359668e-11. This reproduces a concrete write-rounding defect. R47's two
offending-step derivatives were not saved, so this is not proof of their cause.

The existing Q/T/K/KKT formula, strength1/ridge0.01, rank cutoff, loss, W/LSO,
probes and gradient remain unchanged. No precision-mode change, line search,
gradient correction or coefficient search is introduced.

## Representable commit and exact native fallback

Only when the descent constraint is enabled:

1. Preserve a detached copy of the actual post-Adam BN parameters for a nonzero
   sensitivity rank. These are the already executed native proposal, not a new
   optimizer evaluation.
2. Commit the existing closed-form filtered displacement as before and measure
   g^T(theta_committed-theta_old) in float64 dot arithmetic.
3. Keep that exact commit if finite and within the unchanged check:

       actual - proposal <= 1e-5*max(abs(proposal),1e-10), when proposal<0.

4. If the committed candidate violates the check or has a nonfinite derivative,
   copy back the saved post-Adam parameters exactly. Recompute the final actual
   derivative; the outer guard still rejects any final violation/nonfinite value.

The native fallback is bitwise identical to the native proposal. It therefore
has zero derivative gap, without depending on a speculative precision margin
or relaxing the tolerance. It may remove the joint filter for some updates;
future trigger rates and segmentation tradeoffs are unmeasured. We do not call
the repair another closed-form constrained optimum. The candidate's original
closed-form derivative and committed derivative remain distinct diagnostics.

An inactive/feasible commit stays bitwise unchanged. A disabled constraint takes
the old SHARED path; a disabled joint module takes CW_LSO. No Adam moments, step,
gradient, persistent state, snapshot schema, RNG or frozen parameters change.
Both joint versions retain10F/5B/1Adam (four auxiliary VJPs and one base-loss
backward). CW_LSO stays8F/1B/1Adam. The additional transient native-parameter copy
is one BN vector (19,136 float32 parameters =76,544 bytes for the profiled model),
plus vector subtraction/dot arithmetic. Real GPU time and allocator peaks for
this repaired implementation have not been measured.

## Failure evidence and step-number correction

Successful repaired steps record proposal_dot, filtered_dot, candidate_actual_dot,
candidate_committed_gap, commit_repaired, fallback reason, final_actual_dot,
committed_gap and the unchanged tolerance. Candidate overflow values use explicit
string nan/inf representations so strict finite JSON remains writable.

The descent checker attaches the available derivative fields, rank and exact
visit to its ValueError before raising. Both the private physical failed-call
journal and worker attempt receipt persist these numerical diagnostics. NaN/inf
exception fields are likewise serialized as strings, not nonstandard JSON.
No image, patient/content identifier or prediction is added to the payload.

The online runner now captures next_visit before calling Host.step. The wrapper
can raise after Host has incremented its visit counter; computing h.visits+1
inside the old exception handler could report one step too high. Future journals
use the pre-call step. No historical R47 journal/receipt is rewritten. Errors
before directional diagnostics exist still retain their ordinary exception;
this does not fabricate unavailable derivatives.

## CPU qualification and boundaries

Nine new checks cover the rounding counterexample; both float32/float64 and two
gradient scales; feasible inactive/rank-zero commits; overflowing candidate
fallback with strict JSON; unchanged tolerance and exception fields; actual Host
fallback with native Adam state/calls and exact snapshot/RNG replay; durable
failed-call journal; pre-call step despite Host increment; and worker receipt
serialization/classification with zero GPU work.

Existing tests additionally verify disabled bitwise R46/CW_LSO snapshots, dense
noncommuting QR/KKT formula/complement, gradients, RNG, frozen parameters,
10F/5B/1Adam, C/W controls, memory admission and frozen evaluation gates.
The final test receipt is in reports/shared-descent-numerics-cpu/CPU_CHECKS.json.
This is implementation evidence on synthetic CPU fixtures, not real-model GPU
qualification, Dice improvement or anatomy protection. No new experiment began.
