# Joint sensitivity: shared-span adjustment, CPU only

User scope: clarify one adjustment and qualify it on synthetic CPU inputs. No GPU
profile, formal efficacy experiment, seed matrix, new data access or monitor.
R45 results/source execution and its immutable ledger remain historical evidence.

## Evidence and hypothesis

R45 JS-minus-CW_LSO=-0.006898pp, both orders negative,0/6 positive. All SEARCH
steps have acquisition span rank2. Mean metric-norm ratio0.998812 shows small
whole-update attenuation, but does not measure update rotation or establish why
accuracy failed. Prior traces did not record the proposed update's span energy or
the anatomical Jacobian coverage; do not infer those missing values from norms.

The original construction Q=orth(X) keeps only anatomical sensitivity projected
into the acquisition span. Any Y component orthogonal to that span cannot affect
the update. This is an explicit R45 modeling choice, not an implementation fault.
A new hypothesis is that imaging and anatomical responses should share an
estimated space: retain anatomy-sensitive directions even when the acquisition
Jacobian difference does not span them. This may also suppress beneficial
anatomical correction; no accuracy or safety benefit is guaranteed.

For H=I,X=e1,Y=e2,delta_A=e2, the old filter preserves delta_A at any strength,
because Q^T Y=0. The new filter attenuates e2 by1/101 with the unchanged0.01 ridge
and strength1. This synthetic counterexample demonstrates a representational
difference; it is not evidence that this case caused the actual R45 failure.

## Minimum fixed adjustment

Reuse the same Sensor, detached six-view structure evidence, two acquisition
probes, four VJPs, native CW_LSO loss and actual Adam proposal. The only algorithm
change is the span construction; strength1, ridge0.01, fields and thresholds remain
unchanged. C/W/CW_LSO controls are unchanged. Old JS configuration has the original
acquisition basis by default; the opt-in CPU configuration is CW_LSO_JS_SHARED
with joint_basis='shared'. Neither factory is a formal experiment registration.

    X = H^-1/2 [a1,a2]; Y = H^-1/2 [bOD,bOC]
    Z = [X/||X||_F, Y/||Y||_F]
    Q = orth(Z), rank <= 4
    A_Q = (Q^T X)(Q^T X)^T / ||X||_F^2
    S_Q = (Q^T Y)(Q^T Y)^T / ||Y||_F^2
    R = (A_Q+0.01 I)^-1/2
    T = R S_Q R; K = (I+T)^-1
    w_A = H^1/2 delta_A
    delta = H^-1/2 [w_A+Q(K-I)Q^T w_A]
    theta_new = theta_before+delta.

Normalization balances the two blocks during rank selection; A_Q/S_Q still use
the same separate Frobenius normalization as R45. Keep Gram eigenvalues strictly
above max(1e-12,1e-6*lambda_max), now applied to the normalized concatenation. Very
small dependent modes may be discarded, so coverage is approximate at that
cutoff. This is not a full anatomical/image Jacobian or a globally complete span.

Strength0, zero acquisition block, zero anatomy block or rank0 retains the exact
Adam proposal. No evidence-strength parameter, stronger coefficient, EMA, radius
cap, additional views, optimizer step or persistent state is added. All vectors
and metrics must remain finite with positive H. Small matrix arithmetic float64.
Both blocks are required for the new shared-span path.

In whitened coordinates this is the unique minimizer of

    0.5 ||w-w_A||^2 + 0.5 w^T Q T Q^T w.

The metric-orthogonal complement of Q is unchanged. K eigenvalues in(0,1] imply
metric-norm contraction, which is a numerical property, not anatomical safety.
Adding anatomy directions is the tested modeling change; attenuation alone does
not make the update an anatomy-improving gradient.

## Parameters, diagnostics and cost

No scientific coefficient changes: joint_strength1,ridge0.01, Gram relative cutoff
1e-6/absolute floor1e-12, LSO0.1,W temperature0.1/boundary boost2, LR1e-4, original
Adam moments/betas/epsilon, contrast1.10 and channel shift(0.05,0,-0.05).

Calls remain10F/5 reverse traversals (four VJPs+base backward)/1Adam, versus
CW_LSO8F/1B/1Adam. Four traced forward graphs are still processed in the original
sequence. Linear algebra changes from at most2x2 to4x4; no new network pass.
Actual GPU latency/VRAM for this adjustment is unmeasured; the old profile cannot
be relabeled as a shared-span measurement.

New diagnostic fields: acquisition_capture_fraction, anatomy_capture_fraction,
proposal_span_energy_fraction, relative_metric_update_change, plus the existing
rank/eigenvalues/proposal/filtered norms. Capture and energy fractions use the
whitened metric. Relative update change measures the full difference vector,
unlike subtracting two norm scalars. These fields are diagnostics only, not action
thresholds, label feedback or a reason to tune after evaluation.

## CPU qualification

New tests: missing-anatomy-span closed form; independent QR/dense rank4 solution
with noncommuting sensitivities; unchanged complement and contraction; block-scale
invariance and zero/disabled/invalid inputs; two-image complete disabled CW_LSO
parity; exact old-JS behavior against the frozen R45 Git source; enabled native
counts, actual parameter formula, unchanged Adam/gradients/RNG and exact restore.
The original joint module and C/W/LSO/profile-memory CPU regressions also run.

No statement in this document authorizes a GPU profile or new experiment. Any
later efficacy test needs separate user authorization, frozen conditions,
same-round resource admission and controls including CW_LSO and original JS.
