# R72: fixed crop-target transfer fails the unchanged gate

All six 192-arrival trajectories and all 1,152 scored outputs completed. Real-model qualification passed, including repeated full-state/output equality and NATIVE parity with the retained host in both orders. GPU worker and CPU scorer retired with exit zero; no retry or protocol change occurred.

| Arm | Macro Dice (%) | Image-weighted Dice (%) |
|---|---:|---:|
| NATIVE | 75.951890 | 76.063471 |
| RESAMPLE | 75.952824 | 76.064540 |
| CROP | 75.401505 | 75.612737 |

| Contrast | Macro delta (pp) | Conditional 95% CI | Order 0 | Order 1 |
|---|---:|---|---:|---:|
| CROP-NATIVE | -0.550385 | [-0.628960, -0.471993] | -0.630929 | -0.469841 |
| CROP-RESAMPLE | -0.551319 | [-0.630203, -0.473272] | -0.631860 | -0.470778 |
| RESAMPLE-NATIVE | +0.000934 | [+0.000384, +0.001523] | +0.000932 | +0.000937 |

The registered conjunction fails: both CROP superiority contrasts are negative in both orders, and the worst CROP-NATIVE cell (Drishti/order0/OD, -2.898193 pp) also violates the original -2 pp floor. Image-weighted CROP-NATIVE is -0.450734 pp. The floor is not retrospectively changed to the later proposed -0.5 pp. RESAMPLE's tiny positive conditional interval does not imply a useful practical improvement or equivalence to NATIVE.

All four REFUGE_Valid cells improve (+0.430053/+1.574735 pp in order0 OD/OC, +0.249219/+0.621698 in order1), while all Drishti and ORIGA cells decline. REFUGE/order1/OD also declines -2.075670 pp. These trade-offs do not authorize domain routing, new cell selection, or uniform claims of improved error correction. [ALL_CELLS.csv](ALL_CELLS.csv) retains all 48 cells.

## Mechanistic limits

CROP changes the soft target by mean absolute .00642–.00646 and flips its thresholded decision on .444–.458% of channel-pixels; RESAMPLE changes it by .000161–.000163 with .0145–.0149% flips. These counts do not say whether changed pixels become correct. The arms have different evolving histories, so final Dice cannot separate bad pseudo-targets from failure to use a better target through BN-affine updates.

Mean BN update norms are .00438–.00445 for CROP and .00422 for NATIVE. CROP takes 2.21–2.29 seconds/image, NATIVE 1.50–1.61 under shared-device conditions. Modified arms each execute two additional read-only model forwards per arrival. No gradient-magnitude matching or controlled speed benchmark is claimed. Native zero target-change/support fields are diagnostic defaults.

The implemented intervention preserves original six-view W, native strong augmentation, and LSO's original target/coefficient. Only the BCE target receives support-corrected two-crop predictions. This is not complete TeSLA and does not establish that all crop configurations fail.

## Receipts, cost and decision

[QUALIFICATION.json](QUALIFICATION.json), [COMPLETION_AUDIT.json](COMPLETION_AUDIT.json), [RESULTS.json](RESULTS.json), [ARM_SCORES.csv](ARM_SCORES.csv), [MECHANISM.csv](MECHANISM.csv), and [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json) provide the anonymous evidence. Full GPU-process cost: 2,425.733739 seconds (40.43 minutes); inner body 2,422.179225 seconds is not substituted for process cost. CPU total 42.621676 seconds = preparation/check 19.181436 + scoring 23.440240. No online label reads, new source fitting, or added cohort.

Preserve protocol/code `9acee68` and startup `c02e6db`. Close this fixed configuration for performance promotion. The user subsequently authorized a separately registered same-state mechanism diagnostic using the unchanged crop intervention, to distinguish target-quality changes from one-step update effects. That is explanatory work, not an R72 retry, a new primary endpoint, or permission to search crop parameters. R72 remains a failed exposed-development short-stream experiment regardless of that diagnostic's outcome.
