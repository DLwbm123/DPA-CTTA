# R70: f-CE component test — complete, allocation gate failed

All six registered source-initialized short trajectories completed 192 arrivals each; all 1,152 predictions were scored after successful GPU retirement. Both GPU and CPU workers exited zero. CPU formula checks and real-model qualification passed. There was no execution failure or retry.

| Arm | Domain/channel/order macro Dice (%) | Image-weighted Dice (%) |
|---|---:|---:|
| Retained CW_LSO / NATIVE | 75.951890 | 76.063471 |
| HARD | 75.836079 | 76.001891 |
| FCE (registered primary) | 75.739011 | 75.967789 |

| Contrast | Macro change (pp) | Conditional 95% interval | Order 0 | Order 1 |
|---|---:|---|---:|---:|
| FCE-NATIVE | -0.212879 | [-0.281390, -0.143887] | -0.283379 | -0.142379 |
| FCE-HARD | -0.097069 | [-0.149850, -0.041069] | -0.127404 | -0.066733 |
| HARD-NATIVE | -0.115810 | [-0.146195, -0.082521] | -0.155975 | -0.075646 |

FCE fails both required superiority comparisons: it is below NATIVE and HARD in both orders, with both conditional intervals wholly below zero. The required +0.2pp/+0.1pp thresholds are not approached. Its worst cell (-1.876451pp) passes the separate -2pp harm floor, but that does not rescue the failed conjunction. Image-weighted FCE-NATIVE is also negative (-0.095682pp). HARD itself is below NATIVE and is not promoted to the primary.

## All primary domain/channel/order cells

| Domain | Order | Channel | Contents | FCE-NATIVE (pp) |
|---|---|---|---:|---:|
| Drishti_GS | 0 | OD | 22 | -1.876451 |
| Drishti_GS | 0 | OC | 22 | -1.176756 |
| Drishti_GS | 1 | OD | 22 | -0.384107 |
| Drishti_GS | 1 | OC | 22 | -0.078898 |
| ORIGA | 0 | OD | 57 | -0.619545 |
| ORIGA | 0 | OC | 57 | -0.193473 |
| ORIGA | 1 | OD | 57 | -0.368906 |
| ORIGA | 1 | OC | 57 | +0.345546 |
| REFUGE | 0 | OD | 57 | -0.208687 |
| REFUGE | 0 | OC | 57 | +0.002109 |
| REFUGE | 1 | OD | 57 | -1.337482 |
| REFUGE | 1 | OC | 57 | +0.085316 |
| REFUGE_Valid | 0 | OD | 56 | +0.700251 |
| REFUGE_Valid | 0 | OC | 56 | +1.105521 |
| REFUGE_Valid | 1 | OD | 56 | +0.319393 |
| REFUGE_Valid | 1 | OC | 56 | +0.280101 |

All 48 cells across the three contrasts, including every negative cell, are in [ALL_CELLS.csv](ALL_CELLS.csv). REFUGE_Valid improves in all four primary cells while Drishti declines in all four. This heterogeneity does not authorize a domain-specific switch, post-hoc primary endpoint, or sign reversal.

## Mechanism and limits

FCE and HARD reverse the sign of the soft-BCE pixel derivative on approximately 59.5–59.7% of pixels in their own trajectories. This is an unweighted pixel count; tiny gradients count equally, and the states differ between arms. It is neither an error-correction rate nor evidence for a matched-magnitude directional mechanism. Mean BN update norms remain around 0.0040–0.0042. The modified arms take approximately 1.17–1.20 seconds per image versus 1.05 for NATIVE. See [MECHANISM.csv](MECHANISM.csv).

The experiment establishes a negative short-stream result for replacing the weighted strong-view BCE component while retaining W/LSO/native augmentation. It is **not** a full TeSLA reproduction: the original method's f-CE is applied on unaugmented predictions and is combined with marginal entropy, refined EMA pseudo-labels and adversarial-view distillation. Our registered substitution intentionally isolates a different integration location. Thus no claim about the efficacy of the complete TeSLA method follows. Wrong-label reinforcement is a plausible explanation, not a causal conclusion from the recorded diagnostics.

This uses the already exposed 192-image SEARCH cohort, one seed and two correlated orders, with domain counts 22/57/57/56. Macro and image-weighted endpoints differ because domain weights differ. Conditional content-bootstrap intervals do not establish independent-patient generalization. This is a persistent 192-step stream, not a full 1,951-arrival evaluation.

## Cost and decision

Retirement receipts record 1,413.501902 GPU-worker seconds (23.56 minutes), including qualification/startup, and 35.606741 CPU seconds: 20.159683 preparation plus 15.447058 scoring. The smaller internal worker-body timer (1,409.927039 seconds) excludes process startup/exit overhead; the process ledger is the cumulative cost authority. All workers retired and scoring coverage is complete.

Close this fixed FCE/HARD replacement route. Do not retry for scores, tune epsilon/LR, add favorable domains, or expand seeds/horizons. Retain CW_LSO. The next distinct TeSLA component to specify is entropy-seeking augmentation with feature-mean severity regularization. Its learned policy differs from R61's fixed random-versus-clean intervention, but requires a precise geometry-preserving transform set, state/RNG isolation and matched random augmentation control before any new registration. It is not yet implemented or launched. Do not use this negative result to claim the augmentation component failed, and do not combine it with the failed f-CE substitution merely to rescue the score.

Public delivery includes own source, protocol, all aggregate outcomes, qualification and completion/cost receipts. Private images, labels, identities, predictions, weights and third-party implementation files are excluded.
