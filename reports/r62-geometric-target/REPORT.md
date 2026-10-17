# R62 geometric pseudo-target diagnostic: complete, both gates failed

All 96 paired state-image evaluations completed, producing 480 scored masks. Both real-model qualification checks passed, including exact native historical masks, repeated full states, equal weak views/RNG/counts and zero-substitution native-state parity. GPU and CPU workers retired with exit code 0. Labels were read only after GPU retirement; online label reads were zero.

| Contrast | Mean Dice pp | Conditional 95% interval | Anchor 0 | Anchor 1 |
|---|---:|---|---:|---:|
| original_target_minus_native | +0.001350 | [-0.001580, +0.004329] | +0.002553 | +0.000146 |
| native_minus_no_update | +0.003519 | [-0.001516, +0.010212] | +0.001336 | +0.005702 |
| original_target_minus_no_update | +0.004869 | [-0.000183, +0.010977] | +0.003890 | +0.005848 |
| original_view_minus_native_target | -0.124533 | [-0.501996, +0.269508] | -0.142169 | -0.106896 |

The primary update gain is below the registered +0.1 pp threshold and its interval crosses zero. Both anchors are positive, but REFUGE_Valid OC is -0.001992 pp rather than the required +1 pp. The worst update cell is -0.004543 pp, passing the separate -2 pp floor. 6/16 primary cells are adverse. The additional target-quality gate also fails: original-view pseudo masks are lower on average and in both anchors. All 64 contrast cells and every negative cell are retained.

The original-view minus six-view target difference for REFUGE_Valid OC is -0.517325/-0.620694 pp across the two anchors, while ORIGA OC is +0.179575/+0.222384 pp. A domain-dependent pattern does not authorize domain-specific target selection. Hard-mask target quality and update quality are different outcomes; neither is promoted post hoc.

Mean update norm: NATIVE 0.00223349, ORIGINAL_TARGET 0.00214847. Mean gradient norm: 0.166855 versus 0.119342. The original-view target changes gradients, but this is not a matched-norm direction experiment. The retained native W and LSO still use all six views, so this is specifically a BCE target diagnostic.

R12 established sensitivity of a frozen inference path to horizontal flipping. R62 does not reproduce that setting: it uses late adapted states and compares original versus a six-view mean. Its result does not support extrapolating the R12 finding into an original-only BCE update policy. It also does not contradict R12 or prove that every geometric transform is useful. R61 and R62 jointly provide no passed allocation signal for removing strong-input augmentation or replacing the BCE target.

Source revision `350cb41`; [registered protocol](../../docs/protocols/R62_GEOMETRIC_TARGET_DIAGNOSTIC.md). Recorded GPU-worker time, including qualification: 175.305 seconds; CPU scoring: 3.555 seconds. Both are below the 1800/900 second limits. One preparation shell command used an unavailable python alias, corrected to the existing python3 before GPU launch; zero GPU work was lost. Preparation CPU wall time is not included in the worker ledger.

Decision: close this fixed original-target route. Do not rerun, expand seeds/states, scan view mixtures, remove LSO to rescue the result, or start its full trajectory. Continuous research authorization remains active, but this result supplies no justified successor experiment. The next evidence work must identify information beyond these tested self-prediction alterations and check it against the closed-mechanism ledger before registration; a renamed target mixture is insufficient.

Limits: 48 development-exposed contents, one seed and two late states; conditional content-cluster intervals are not independent patient inference. One-step effects do not bound a closed-loop trajectory. No independent-generalization or novel-method claim. Public artifacts contain own source/protocol, anonymous aggregates and completion/cost receipts; images, labels, identities, masks, weights and private paths remain excluded.
