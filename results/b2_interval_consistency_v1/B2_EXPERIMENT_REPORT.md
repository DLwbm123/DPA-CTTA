# B2 interval consistency experiment

B2_INTERVAL_CONSISTENCY_COMPLETE

Execution commit: 35d9d26c5554a23d81713f610d6ae53d93f54774

U/S/I use partition-mean, partition-shuffled, and spatial six-view population standard-deviation radii. C/G and P2 controls are reused, identity-matched historical trajectories.

Prespecified descriptive scales: {"two_order_mean_I_minus_C_pp": -0.16681948540005465, "overall_gain_scale_met": false, "robustness_tradeoff_scale_met": false, "I_above_U_and_S_both_orders": false, "spatial_necessity_established": false, "reason": "Descriptive development comparison only; positive matched comparisons can support a candidate, not establish necessity.", "REFUGE_Valid_OC_common_valid_ASSD_delta_px": [-4.613074126213545, -2.0941337292294753], "ORIGA_I_minus_C_pp": [-2.685300853964745, -1.18114776158677]}


## Interpretation and candidate decision

The run completed; the spatial interval candidate I did not establish a net improvement over C. On remaining_dev, I-C is -0.840015 / +0.506376 pp, averaging -0.166819 pp. The prespecified +0.5 pp overall scale and the full robustness-tradeoff scale were both unmet.

| Method | order0 Dice % | order1 Dice % | Two-order descriptive mean % |
|---|---:|---:|---:|
| C | 79.104256 | 78.006348 | 78.555302 |
| U | 79.180504 | 77.416035 | 78.298270 |
| S | 79.183325 | 77.838037 | 78.510681 |
| I | 78.264241 | 78.512724 | 78.388483 |

U is partition-mean radius; S is radius shuffled within pseudo-foreground/background partitions; I keeps the spatial radius. I-U is -0.916263 / +1.096689 pp and I-S is -0.919084 / +0.674688 pp. Rankings reverse with domain order, so this does not support a consistently superior spatial-disagreement method. S is closest to C in the two-order mean, but does not improve it overall.

### Domain tradeoffs on remaining_dev

| Domain | I-C order0 pp | I-C order1 pp |
|---|---:|---:|
| REFUGE | -0.735249 | -0.952988 |
| ORIGA | -2.685301 | -1.181148 |
| REFUGE_Valid | +7.879886 | +4.483398 |
| Drishti_GS | -7.819397 | -0.323757 |

REFUGE_Valid improves substantially: macro Dice +7.879886 / +4.483398 pp; OC Dice +6.951003 / +3.206111 pp; OC ASSD -4.613074 / -2.094134 px on all 736 jointly defined pairs in each order. Macro gains occur in 716/736 and 694/736 groups. The worst-decile macro paired changes are +1.078539 / -0.504717 pp; worst individual changes remain -10.094914 / -8.073931 pp. This is partial mitigation, not uniform per-image safety or full recovery to A.

That benefit is offset by ORIGA (-2.685301 / -1.181148 pp), REFUGE (-0.735249 / -0.952988 pp), and Drishti_GS (-7.819397 / -0.323757 pp). Drishti_GS loses on 36/37 groups in order0 and 35/37 in order1. Although its remaining_dev cohort is small, the prespecified primary metric weights each domain equally; do not replace it with pooled image weighting after seeing the result.

The robustness scale fails because order0 I-C is below -0.25 pp and ORIGA declines by more than 2 pp in order0, despite meeting the REFUGE_Valid improvement and OC ASSD conditions. I remains above historical A/G/O2 on the main aggregate in both orders, but C is the stronger comparator for this specific change.

Retain C as the primary reference. Record B2 as a mixed, order-sensitive domain tradeoff; do not promote I as the default replacement or claim spatial radius placement is necessary. The REFUGE_Valid signal is worth documenting but is not evidence for a GT-driven domain selector. No follow-up GPU experiment or parameter search was launched.

### Execution evidence

All six trajectories completed on GPU 7: 11,706 records/Adam/backwards and 93,648 formal forwards. Including smoke: 11,726 Adam/backwards and 93,808 forwards. Nine remote CPU tests and the single 20-update GPU smoke passed. Active-stage time including smoke was approximately 2 h 22 min; detached run and CPU reconstruction exited zero. No zero-gradient formal steps were observed. Only registered BN affine parameters adapted.

The independent CPU closeout checks coverage, order, state counters, scalar diagnostics and Dice reconstructed from pixel counts. It cannot recompute ASSD from discarded pixel maps. All subsets are development data; the two orders are descriptive repetitions on overlapping content.

## remaining_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G | U | S | I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 1695 | 68.200582 | 74.259279 | 73.426605 | 72.891721 | 72.686550 | 79.104256 | 77.372100 | 79.180504 | 79.183325 | 78.264241 |
| order1 | 1695 | 68.200582 | 74.542945 | 73.813586 | 75.423504 | 74.305555 | 78.006348 | 77.087285 | 77.416035 | 77.838037 | 78.512724 |

| Order | I-C | I-U | I-S | U-C | S-C | I-A | I-G | I-O2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | -0.840015 | -0.916263 | -0.919084 | +0.076248 | +0.079068 | +4.004962 | +0.892141 | +5.372520 |
| order1 | +0.506376 | +1.096689 | +0.674688 | -0.590313 | -0.168311 | +3.969780 | +1.425439 | +3.089220 |

## legacy_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G | U | S | I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 128 | 69.092892 | 74.538998 | 73.801632 | 73.653989 | 73.009535 | 79.306413 | 77.579453 | 79.399268 | 79.394272 | 78.296821 |
| order1 | 128 | 69.092892 | 75.140702 | 74.439300 | 75.885483 | 74.336631 | 77.919338 | 77.059848 | 77.322833 | 77.740362 | 78.469991 |

| Order | I-C | I-U | I-S | U-C | S-C | I-A | I-G | I-O2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.009592 | -1.102447 | -1.097451 | +0.092855 | +0.087858 | +3.757823 | +0.717368 | +4.642832 |
| order1 | +0.550653 | +1.147158 | +0.729629 | -0.596505 | -0.178976 | +3.329289 | +1.410143 | +2.584507 |

## p1_extension_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G | U | S | I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 128 | 66.452389 | 73.796285 | 72.745511 | 72.436336 | 71.836462 | 79.497100 | 77.709793 | 79.556190 | 79.580612 | 78.709606 |
| order1 | 128 | 66.452389 | 74.506257 | 73.700675 | 76.426326 | 74.433735 | 78.551758 | 77.384891 | 78.068718 | 78.426201 | 78.835984 |

| Order | I-C | I-U | I-S | U-C | S-C | I-A | I-G | I-O2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | -0.787495 | -0.846584 | -0.871006 | +0.059089 | +0.083511 | +4.913321 | +0.999813 | +6.273270 |
| order1 | +0.284226 | +0.767266 | +0.409783 | -0.483040 | -0.125556 | +4.329727 | +1.451093 | +2.409658 |

## all_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G | U | S | I |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 1951 | 67.964224 | 74.328553 | 73.338857 | 72.983086 | 72.604889 | 79.230033 | 77.622550 | 79.277908 | 79.303458 | 78.477113 |
| order1 | 1951 | 67.964224 | 74.691960 | 73.806501 | 75.548687 | 74.372899 | 78.241148 | 77.283082 | 77.658402 | 78.075828 | 78.733627 |

| Order | I-C | I-U | I-S | U-C | S-C | I-A | I-G | I-O2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | -0.752919 | -0.800794 | -0.826345 | +0.047875 | +0.073425 | +4.148561 | +0.854563 | +5.494028 |
| order1 | +0.492479 | +1.075225 | +0.657799 | -0.582746 | -0.165320 | +4.041667 | +1.450545 | +3.184940 |

All domain/channel paired distributions, signs, worst decile/single and ASSD common-valid cohorts are in public_aggregate.json. No macro ASSD.
All subsets are development data, including the historically named remaining_dev. Orders and subsets are not independent replicates. No significance, conformal coverage, clinical safety, novelty or SOTA claim.
U/S/I retain 41 BN layers and 19,136 affine scalars; eight forwards, one backward and one native Adam step per image. Zero new gradient may still move parameters due to Adam history.
CPU reconstruction checks every scalar record and Dice pixel counts; discarded ASSD maps cannot be independently recomputed. Optional q-error dispersion bins were not collected; no extra inference is authorized for them.
A/N/G/O2/D4 are historical context, not teachers or output fusion. No new source/DD/selector training, no radius/LR search, and no automatic next experiment.
Engineering completion does not establish the interval hypothesis. Compare I-C, I-U and I-S before any candidate decision.
