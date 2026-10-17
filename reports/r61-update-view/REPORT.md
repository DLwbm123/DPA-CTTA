# R61 update-input diagnostic: complete, allocation gate failed

All 96 fixed state-image pairs completed (48 SEARCH contents, two late CW_LSO anchors), producing 288 scored predictions. Both real-model qualification checks passed. GPU and CPU workers retired with exit code 0; labels were read only after GPU retirement. Online label reads: zero.

| Contrast | Mean Dice pp | Conditional 95% bootstrap interval | Anchor 0 | Anchor 1 |
|---|---:|---|---:|---:|
| clean_minus_native | -0.005391 | [-0.010575, -0.001137] | -0.004769 | -0.006013 |
| native_minus_no_update | +0.003519 | [-0.001516, +0.010212] | +0.001336 | +0.005702 |
| clean_minus_no_update | -0.001872 | [-0.005644, +0.001749] | -0.003433 | -0.000311 |

CLEAN minus NATIVE fails the registered +0.1 pp mean, positive interval lower bound, both-positive-anchor and +1 pp REFUGE_Valid OC requirements. REFUGE_Valid OC mean is -0.000282 pp. The worst cell is -0.025547 pp (order 0, REFUGE_Valid OD), which passes the separate -2 pp floor. Nine of the 16 primary cells are negative. All 48 contrast cells, including every adverse cell, are published.

Mean parameter displacement is 0.00223349 for NATIVE and 0.00222277 for CLEAN; mean gradient norm is 0.166855 versus 0.143746. Removing strong appearance augmentation does not provide the proposed immediate improvement here. These aggregate norms cannot identify a causal direction effect, and the tiny one-step differences do not bound accumulated trajectory effects. NO_UPDATE comparisons are secondary and do not rescue the failed primary gate.

Qualification covered historical native hard-mask parity, repeated branch state/output equality, equal weak views and terminal native RNG, exact restoration, and frozen-parameter checks. Source revision: `b3acdad`. See the [preregistered protocol](../../docs/protocols/R61_UPDATE_VIEW_DIAGNOSTIC.md), qualification and completion receipts.

Recorded GPU-worker time, including qualification: 166.480 seconds; CPU scoring: 3.196 seconds. Both are below the 1800/900 second limits. Two preparation-only CPU failures (missing inherited metadata/configuration; incorrect native dependency path) were repaired before launch and consumed zero GPU seconds. Their private logs remain preserved. The ledger covers formal worker and scoring time, not all preparation CPU time or energy.

No retry, augmentation-strength grid, extra states/seeds or CLEAN full trajectory is justified by this allocation rule. R23 already tested disagreement masking/interval targets; R28/R29 already failed candidate/strength headroom gates. R61 adds a negative update-input result, not a new reliability signal.

Next bounded preparation: audit whether geometric-view pseudo-target bias is actually a distinct untested mechanism. Read the R11 current-view evidence and R20 teacher/target changes together with native view construction before registering anything. Separate output ensembling from target construction; do not assume original-view targets are better. If existing evidence duplicates the intervention or no fixed diagnostic has a defensible interpretation, close it without GPU work. No new experiment is registered by this report.

Limits: two late states, one seed and development-exposed contents; confidence intervals are conditional content-cluster summaries, not patient or independent-generalization uncertainty. Gradient magnitudes are unmatched. This does not establish that strong augmentation is universally necessary. Public files exclude images, labels, per-content identities, masks, checkpoints, private paths and third-party source.
