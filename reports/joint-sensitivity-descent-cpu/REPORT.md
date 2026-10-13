# Shared sensitivity: descent-constraint CPU qualification

Implementation and CPU qualification complete. No GPU work, remote scientific
execution, profile, data access, formal registration or monitoring was performed.

- Seven new tests plus related regressions:28/28 PASS,11.928s.
- Earlier six-test CPU pass:5.779s; total known test-body17.707s, GPU0.
- Constraint off: bitwise output, diagnostics and complete snapshot parity with
  frozen R46 source66097c9 over two synthetic arrivals.
- Entire joint module off: bitwise CW_LSO output/diagnostic/snapshot parity.
- Active closed form matches an independent QR/KKT solution. Inactive constraint
  preserves the original shared filter exactly. Rank0/zero evidence/non-descent
  proposal fallbacks, invalid inputs and gradient-scale invariance are checked.
- Enabled calls remain10F/4 auxiliary VJPs/1 loss backward/1 Adam. Native Adam
  moments/count, gradients, frozen parameters and augmentation RNG are preserved;
  snapshot replay is exact. Actual parameter updates and directional dots are
  checked with finite-precision tolerance.
- Shared-profile validation is updated to the new first-order loss criterion;
  original candidates keep their contraction criterion. This checker was tested
  with CPU scalar fixtures only.

The new flag defaults false; no frozen R46 candidate is modified. The CPU-only
factory is not a launch instruction. See
`docs/JOINT_SENSITIVITY_DESCENT_CPU.md` and
`src/dpa_ctta/joint_sensitivity/descent_tests.py` for formula and reproducible checks.

Scientific limits: true Dice/anatomical benefit, finite-step loss reduction,
patient generalization and real-model GPU cost are untested. Global metric-norm
contraction is not guaranteed after the affine constraint correction. Existing
unconstrained K eigenvalues must not be interpreted as eigenvalues of that
corrected update. The retained base-host tensor-to-scalar warning remains.
