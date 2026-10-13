# Shared sensitivity with a base-loss descent constraint: CPU only

Scope: implement and qualify the proposed R46 adjustment on synthetic CPU
fixtures. No GPU profile, real images, labels, source checkpoint, formal matrix,
scientific outcome, monitor or next-round launch is authorized by this document.
The completed R46 branch, protocol, results and immutable cost ledger are retained.

## Retained method and configuration

Use the same CW_LSO base loss (including W and LSO0.1), Sensor, six weak views,
two acquisition probes, four VJPs, Adam metric, shared rank<=4 basis, ridge0.01,
strength1 and Gram cutoff as R46. The new boolean `joint_descent_constraint`
defaults to false. The CPU-only candidate factory in `joint_sensitivity/descent.py`
retains C/W/CW_LSO/original JS/shared JS and adds
`CW_LSO_JS_SHARED_DESCENT` with this flag true. It installs no experiment runtime.
R46's factory and registration are unchanged.

`g` is the gradient from the existing CW_LSO loss backward, copied before Adam
steps. Auxiliary VJPs do not accumulate into it. It is not a label gradient, loss
reevaluation, anatomical correctness score or persistent state.

## Formula

Let delta_A=theta_after_Adam-theta_before and H=sqrt(v_hat)+Adam_epsilon.
Reuse Q and T=R S_Q R from R46; K=(I+strength*T)^-1. Then

    w_A=H^1/2 delta_A; v_A=Q^T w_A
    c=Q^T H^-1/2 g; v_0=K v_A

When g^T delta_A<0, solve

    min_v 0.5 ||v-v_A||^2 + 0.5 strength v^T T v
    subject to c^T(v-v_A)<=0.

If the unconstrained solution satisfies the inequality, keep the exact original
R46 result and its arithmetic. Otherwise, with violation=c^T(v_0-v_A)>0,

    mu=violation/(c^T K c)
    v_*=v_0-mu K c
    delta_*=H^-1/2 [w_A+Q(v_*-v_A)]
    theta_new=theta_before+delta_*.

K is positive definite, so the denominator is positive for nonzero c. A
nonpositive numerical denominator falls back to the actual Adam proposal. A
nonnegative original g^T delta_A (including zero gradient) also keeps the actual
proposal exactly. Zero strength, missing sensitivity or rank0 retain the proposal.
Nonfinite gradients/directional derivatives, wrong shapes or an enabled
constraint with the acquisition-only basis are rejected.

The constraint ensures that the base loss's first-order directional change is
no worse than the original Adam proposal in exact arithmetic. It is not a
guarantee of finite-step loss reduction, Dice improvement, or anatomical safety.
The H-orthogonal complement of Q is unchanged. Global H-norm contraction is no
longer a required property of the constrained solution. No line search or extra
loss evaluation is performed.

## Integration, diagnostics and cost

Only the opt-in path copies the flattened base-loss BN gradient. Adam moments,
step count, LR, frozen parameters and augmentation RNG are unchanged. The
constraint is transient and requires no additional snapshot field. Existing
snapshot/restore replays enabled execution exactly on the CPU fixture.

Calls remain10F / four auxiliary VJPs + one loss backward / one Adam, identical
to shared JS; CW_LSO remains8F / one loss backward / one Adam. Additional work is
one transient BN-gradient vector, directional dot products and a rank<=4 KKT
correction. Real-model latency and memory are unmeasured.

Diagnostics include constraint activation, unconstrained gap, multiplier, original
proposal dot, returned-filter dot and dot of the actually committed parameter
change. Zero-evidence/non-descent fallbacks remain identifiable through rank0
and, where applicable, the fallback reason. Existing eigenvalues describe the
unconstrained K, not the affine constraint-corrected update operator.

The shared profile checker now uses the directional-loss diagnostic for this
opt-in candidate, with rounding tolerance1e-5*max(abs(proposal_dot),1e-10).
Original candidates retain the original contraction check. This is a numerical
validation tolerance, not a scientific coefficient, and does not authorize a GPU
profile. Parameter dtype and parameter-addition rounding are audited via the
actual committed dot; tiny violations of an exact-real-arithmetic inequality may
remain at finite precision.

## CPU qualification and evidence boundaries

Seven new tests cover active closed form and inactive exact parity; independent
QR/KKT solutions for noncommuting sensitivities; unchanged complement and gradient
scale invariance; zero/non-descent/missing/invalid inputs; two-arrival bitwise
output/diagnostic/full-snapshot parity against frozen R46 source66097c9; full
CW_LSO parity when the joint module is disabled; native calls, cached gradient,
physical parameter formula, unchanged Adam/gradients/RNG, frozen parameters and
exact replay; and the opt-in profile validation rule.

The complete related suite passed28 tests in11.928s, including the original JS,
shared JS, profile gates, R46 controls/gates, C/W/LSO and measured-memory guards.
An earlier six-test pass took5.779s, before completing profile-check integration;
both test-body elapsed times are retained (17.707s total), GPU0. An existing tensor-to-
scalar warning in the base host was observed; it did not fail validation and was
not changed as part of this adjustment.

These checks establish implementation behavior on CPU synthetic fixtures. They
do not establish that the proposal preserves REFUGE_Valid gains or improves
Drishti_GS/ORIGA, and do not reuse R46 outcomes as new efficacy evidence. Any later
GPU profile or experiment requires separate authorization and a new frozen plan.
