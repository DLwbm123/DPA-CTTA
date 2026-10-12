# Joint sensitivity shared-span CPU qualification

Date:2026-10-12. Final check **21/21 PASS**,10.718s test-body time. Earlier20-test
pass9.839s;21st test adds the enabled zero-evidence exact fallback. Known combined
test-body time20.557s; overall editing/setup CPU wall was not metered. GPU cost0.
User scope: clarify the adjustment and CPU checks only, no GPU experiment.

## Adjustment

Original R45 uses Q=orth(X), so only anatomy projected into the acquisition span
can constrain the proposal. The opt-in shared configuration uses
Q=orth([X/||X||,Y/||Y||]), rank<=4. Remaining evidence, probes, loss, VJPs, native
Adam, strength1,ridge0.01 and rank cutoffs retain their fixed values. Separate
block normalization prevents relative block scale from erasing one block during
rank selection. Missing either sensitivity block keeps the actual proposal exactly.

The unchanged old-JS default and the new disabled path are qualified bitwise.
This is a new modeling hypothesis, not a retroactive R45 bug fix or scientific
success. Actual anatomical coverage/proposal span-energy were not in R45 traces,
so its small norm attenuation cannot identify the cause of its negative result.

## Checks passed

- Shared disabled: two generated arrivals yield bitwise-equal output, details and
  full snapshots versus CW_LSO.
- Original JS: the current default path equals the immutable R45 source from Git,
  with two arrivals and bitwise output, details and full snapshot equality.
- Orthogonal anatomy counterexample: H=I,X=e1,Y=e2,delta=e2. Old JS keeps delta;
  shared JS gives e2/101 with ridge0.01/strength1. This illustrates a possible
  missing-direction mechanism, not a measured medical failure cause.
- Independent QR/dense rank4 reference agrees to1e-11 on a noncommuting case;
  unchanged metric-orthogonal complement, contraction and filter eigenvalues(0,1].
- Separate X/Y rescaling leaves the filter invariant; disabled/zero sensitivity
  retains the proposal exactly; invalid strengths/metrics/basis rejected.
- Enabled synthetic Host: physical10 forwards/5 reverse traversals/1Adam; actual
  parameters equal theta_before+filtered_proposal, with optimizer moments unchanged
  by filtering. First-arrival base loss gradients, Adam state and native RNG match
  CW_LSO exactly; snapshot/restore gives bitwise next-arrival replay.
- Enabled zero structure evidence: exact CW_LSO output/parameters/gradients/Adam/RNG,
  rank0 fallback, retaining declared10F/5B/1Adam physical schedule.
- Existing JS, C/W/LSO and profile-memory qualification checks also pass.

The six new tests and15 existing regressions use generated inputs and Tiny model,
CPU CPython3.10.20/PyTorch2.14.1 and two torch threads. The baseline emits its
existing tensor-to-scalar diagnostic warning; no unrelated baseline code changed.
Raw test output stays private because it contains local environment paths.

## Calls, diagnostics and limits

Still10F/5B/1Adam versus CW_LSO8F/1B/1Adam. The original2x2 eigensystem becomes at
most4x4 without additional network passes; new vector-column memory and GPU
allocator effects remain unmeasured. Do not reuse old GPU latency/VRAM as a new
profile result.

The shared path records acquisition/anatomy capture fractions, proposal span
energy fraction and relative full metric-update change. These distinguish missing
coverage, small proposal overlap and actual update change without label feedback.
They are measurements, not new decision gates or tuning objectives.

No full ResUNet check, real-data metric, GPU profiling, seed matrix, scientific
runner, queue or monitor was created. Existing R45 artifacts/T0/cost are unchanged.
Capturing more anatomy-sensitive directions can suppress useful corrective
adaptation, so formula validity is not evidence of accuracy or anatomical benefit.
A future effect test requires separately frozen controls and explicit authorization.

[Specification](../../docs/JOINT_SENSITIVITY_SHARED_SPAN_CPU.md).
