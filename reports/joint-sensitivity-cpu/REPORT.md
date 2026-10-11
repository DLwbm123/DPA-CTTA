# Joint sensitivity CPU qualification

Date: 2026-10-11. Result: **PASS, 15/15 tests**, 7.976 seconds test-body time.
Scope: implementation and synthetic CPU checks, not a scientific experiment.
No GPU allocation, server SSH connection, experiment launch, or R44 modification.

Implementation: `src/dpa_ctta/joint_sensitivity/method.py`.
Specification: `docs/JOINT_SENSITIVITY_CPU.md`.

## Verified

- Disabled module: two consecutive generated images give bitwise-equal output,
  returned details and complete snapshots against CW_LSO, including Adam,
  gradients, counters, modes, existing module state and RNG.
- Enabled module: actual model hook trace is
  [grad,no-grad,no-grad,no-grad,no-grad,no-grad,grad,grad,grad,no-grad].
  Physical counters are exactly 10 forwards / 5 backwards / 1 Adam per image.
- Four captured VJPs match both original anatomical columns and both acquisition
  difference columns, including the shared (b_OD+b_OC)/sqrt(2) subtraction.
- Actual Adam proposal is captured before parameter filtering. The final affine
  parameters exactly match theta_before+filtered_proposal. The optimizer state
  before and after filtering is unchanged. On the first arrival, baseline loss
  gradients, Adam state and native RNG exactly match CW_LSO.
- Rank-one filter matches its closed form. A noncommuting 2D sensitivity example
  matches an independent QR/dense reference to 1e-12 and preserves the whitened
  orthogonal complement. Metric norm contracts and filter eigenvalues are in (0,1].
- Rank-zero and disabled vector filters retain the proposal exactly; zero
  anatomical sensitivity is finite. Negative/nonfinite strength and nonpositive
  metrics are rejected.
- Sensor reference response is exactly zero; channel derivatives are separate
  and nontrivial. Probes preserve endpoints and do not disturb global RNG.
- Zero-evidence Host keeps the CW_LSO output, parameters, gradients, moments and
  native RNG bitwise while still executing the declared enabled schedule.
- Snapshot/restore reproduces the next enabled arrival and full snapshot exactly.
  Frozen non-BN parameters remain unchanged.
- Original C/W, local structure, snapshot and R44 spectral-entropy CPU checks pass.

New module: 6 tests. Existing regression modules: 5+2+2 tests.

## Preparation and repaired finding

The default local Python lacked torch; a separate ignored local environment was
created for these CPU checks. Tested: CPython 3.10.20, PyTorch 2.14.1, NumPy 2.2.6,
SciPy 1.15.3, batchgenerators 0.25.2. Device was explicitly CPU, two torch threads.

The first five-test run had 4 passes and one failure (43.634 seconds test-body
time): the reference sensor response was not bitwise zero when its reference
orientation used no-grad convolution arithmetic but the response used traced
convolution arithmetic. Computing the reference orientation with tracing enabled
before detachment repaired this. The next five-test run passed (3.834 seconds).
Additional VJP/zero-evidence checks and the regressions then passed 15/15.

An existing baseline diagnostic emits a tensor-to-scalar warning in this newer
PyTorch version. It did not affect equality checks; unrelated baseline code was
not edited.

## Boundaries

R44's original checkout remained clean on experiment/r44-spectral-entropy-v1 at
372de5d. New implementation is isolated on feature/cw-lso-joint-sensitivity.
No experiment entry point, scientific seed matrix, GPU profile or runtime
campaign was added. Existing remote workers and monitoring were not accessed.

There are no measured Dice gains, GPU latency/VRAM measurements, full ResUNet
qualification or clinical anatomy-preservation claims. Analytical compute is
10F/5B versus 8F/1B, approximately 1.84-2.14x under the stated cost assumptions.
This receipt establishes CPU implementation validity only.
