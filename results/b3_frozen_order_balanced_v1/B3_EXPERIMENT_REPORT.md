# B3 frozen balanced order comparison

B3_FROZEN_ORDER_COMPARISON_COMPLETE

Execution commit: 0fac9b2b1e762ed57b624953edf95058cb1f0380

No consistent matched support for replacing C. Retain C and end automatic expansion of this fixed radius configuration. Spatial radius placement has no consistent advantage over U/S under the prespecified comparison.

New evidence: orders 2/3 only; old 0/1 A from P2, C from B1, U/S/I from B2. N is rearranged by identity, without inference.

## Closeout interpretation

All ten new trajectories and independent CPU reconstruction completed. Active stages including smoke took 15,230 seconds (about 4 h 14 min); private output at reconstruction was 56,809,541 bytes. There were 19,510 new records/updates/backwards and 132,668 formal segmentation forwards; including smoke, 19,530 updates/backwards and 132,804 forwards.

On the two new orders, I-C is +0.703900 / -2.304035 pp (new-order mean -0.800068 pp). Across all four orders it is -0.483444 pp, with two positive and two negative orders. I-U and I-S also reverse signs, and their four-order means are -0.383849 / -0.554209 pp. The fixed spatial interval configuration does not earn promotion.

| Method | New-order mean Dice % | Four-order mean Dice % | Worst observed order % | Order range pp |
|---|---:|---:|---:|---:|
| A | 74.240135 | 74.320623 | 74.002581 | 0.540364 |
| C | 78.667019 | 78.611161 | 78.006348 | 1.319795 |
| U | 78.724862 | 78.511566 | 77.416035 | 2.024646 |
| S | 78.853172 | 78.681926 | 77.838037 | 1.819664 |
| I | 77.866952 | 78.127717 | 77.022107 | 1.689689 |

S has the highest four-order descriptive mean, but its advantage over C is only +0.070765 pp; this is not a demonstrated meaningful or significant gain. I has both a lower mean and a lower worst observed order than C, so the four-order evidence does not support calling it a mean-versus-worst-order improvement. I remains above A, but that does not replace the matched C/U/S comparisons.

REFUGE_Valid is the consistent local benefit: I-C macro improves in all four orders (+7.879886, +4.483398, +6.885207, +3.254111 pp). Its OC ASSD also improves in each order. However, I OC Dice still falls below A by 7.634197, 6.005356, 6.715221 and 4.531664 pp respectively. Improving C does not mean recovering to or exceeding A on the critical OC channel.

The other three domains have negative I-C macro differences in every order. In new order3 the costs are REFUGE -3.383795, Drishti_GS -7.241468 and ORIGA -1.844990 pp; REFUGE_Valid +3.254111 pp cannot offset them under the prespecified domain-equal metric. All corresponding paired signs, tails and ASSD common-valid cohorts remain in the aggregate.

Retain C as the principal matched baseline and stop automatic expansion of this fixed radius configuration. Do not change B2 conclusions, search a new radius, add routing/ensembling or select methods by GT/domain scores. Four orders cover four of 24 permutations on the same development contents and seed; these are descriptive comparisons, not independent patient validation.

## remaining_dev

| Order / descriptive mean | A | C | U | S | I |
|---|---:|---:|---:|---:|---:|
| 0 (reused) | 74.259279 | 79.104256 | 79.180504 | 79.183325 | 78.264241 |
| 1 (reused) | 74.542945 | 78.006348 | 77.416035 | 77.838037 | 78.512724 |
| 2 (new) | 74.477689 | 78.007896 | 78.009043 | 78.048643 | 78.711796 |
| 3 (new) | 74.002581 | 79.326143 | 79.440681 | 79.657701 | 77.022107 |
| new_orders_mean | 74.240135 | 78.667019 | 78.724862 | 78.853172 | 77.866952 |
| four_orders_mean | 74.320623 | 78.611161 | 78.511566 | 78.681926 | 78.127717 |

| Pair | order0 | order1 | order2 | order3 | New mean | Four mean |
|---|---:|---:|---:|---:|---:|---:|
| I-C | -0.840015 | +0.506376 | +0.703900 | -2.304035 | -0.800068 | -0.483444 |
| I-U | -0.916263 | +1.096689 | +0.702753 | -2.418574 | -0.857910 | -0.383849 |
| I-S | -0.919084 | +0.674688 | +0.663153 | -2.635594 | -0.986220 | -0.554209 |
| I-A | +4.004962 | +3.969780 | +4.234107 | +3.019526 | +3.626817 | +3.807094 |
| C-A | +4.844977 | +3.463403 | +3.530208 | +5.323562 | +4.426885 | +4.290537 |
| U-A | +4.921225 | +2.873090 | +3.531354 | +5.438100 | +4.484727 | +4.190943 |
| S-A | +4.924046 | +3.295092 | +3.570954 | +5.655120 | +4.613037 | +4.361303 |

## legacy_dev

| Order / descriptive mean | A | C | U | S | I |
|---|---:|---:|---:|---:|---:|
| 0 (reused) | 74.538998 | 79.306413 | 79.399268 | 79.394272 | 78.296821 |
| 1 (reused) | 75.140702 | 77.919338 | 77.322833 | 77.740362 | 78.469991 |
| 2 (new) | 75.664870 | 78.062062 | 78.068672 | 78.095748 | 78.677365 |
| 3 (new) | 74.554914 | 79.410404 | 79.434481 | 79.695298 | 77.073786 |
| new_orders_mean | 75.109892 | 78.736233 | 78.751576 | 78.895523 | 77.875576 |
| four_orders_mean | 74.974871 | 78.674554 | 78.556313 | 78.731420 | 78.129491 |

| Pair | order0 | order1 | order2 | order3 | New mean | Four mean |
|---|---:|---:|---:|---:|---:|---:|
| I-C | -1.009592 | +0.550653 | +0.615303 | -2.336619 | -0.860658 | -0.545064 |
| I-U | -1.102447 | +1.147158 | +0.608693 | -2.360695 | -0.876001 | -0.426823 |
| I-S | -1.097451 | +0.729629 | +0.581618 | -2.621512 | -1.019947 | -0.601929 |
| I-A | +3.757823 | +3.329289 | +3.012496 | +2.518872 | +2.765684 | +3.154620 |
| C-A | +4.767415 | +2.778636 | +2.397193 | +4.855491 | +3.626342 | +3.699684 |
| U-A | +4.860270 | +2.182131 | +2.403803 | +4.879567 | +3.641685 | +3.581443 |
| S-A | +4.855274 | +2.599660 | +2.430878 | +5.140384 | +3.785631 | +3.756549 |

## p1_extension_dev

| Order / descriptive mean | A | C | U | S | I |
|---|---:|---:|---:|---:|---:|
| 0 (reused) | 73.796285 | 79.497100 | 79.556190 | 79.580612 | 78.709606 |
| 1 (reused) | 74.506257 | 78.551758 | 78.068718 | 78.426201 | 78.835984 |
| 2 (new) | 73.975280 | 78.497782 | 78.514695 | 78.533136 | 79.021981 |
| 3 (new) | 73.787197 | 79.931469 | 80.010644 | 80.217729 | 77.526415 |
| new_orders_mean | 73.881238 | 79.214625 | 79.262670 | 79.375433 | 78.274198 |
| four_orders_mean | 74.016254 | 79.119527 | 79.037562 | 79.189420 | 78.523497 |

| Pair | order0 | order1 | order2 | order3 | New mean | Four mean |
|---|---:|---:|---:|---:|---:|---:|
| I-C | -0.787495 | +0.284226 | +0.524199 | -2.405054 | -0.940427 | -0.596031 |
| I-U | -0.846584 | +0.767266 | +0.507286 | -2.484229 | -0.988472 | -0.514065 |
| I-S | -0.871006 | +0.409783 | +0.488845 | -2.691314 | -1.101235 | -0.665923 |
| I-A | +4.913321 | +4.329727 | +5.046701 | +3.739219 | +4.392960 | +4.507242 |
| C-A | +5.700816 | +4.045501 | +4.522502 | +6.144272 | +5.333387 | +5.103273 |
| U-A | +5.759905 | +3.562461 | +4.539415 | +6.223448 | +5.381432 | +5.021307 |
| S-A | +5.784327 | +3.919945 | +4.557857 | +6.430533 | +5.494195 | +5.173165 |

## all_dev

| Order / descriptive mean | A | C | U | S | I |
|---|---:|---:|---:|---:|---:|
| 0 (reused) | 74.328553 | 79.230033 | 79.277908 | 79.303458 | 78.477113 |
| 1 (reused) | 74.691960 | 78.241148 | 77.658402 | 78.075828 | 78.733627 |
| 2 (new) | 74.659026 | 78.258312 | 78.249812 | 78.294068 | 78.972847 |
| 3 (new) | 74.090025 | 79.615384 | 79.668464 | 79.899530 | 77.302079 |
| new_orders_mean | 74.374525 | 78.936848 | 78.959138 | 79.096799 | 78.137463 |
| four_orders_mean | 74.442391 | 78.836219 | 78.713646 | 78.893221 | 78.371417 |

| Pair | order0 | order1 | order2 | order3 | New mean | Four mean |
|---|---:|---:|---:|---:|---:|---:|
| I-C | -0.752919 | +0.492479 | +0.714535 | -2.313305 | -0.799385 | -0.464803 |
| I-U | -0.800794 | +1.075225 | +0.723035 | -2.366385 | -0.821675 | -0.342230 |
| I-S | -0.826345 | +0.657799 | +0.678779 | -2.597451 | -0.959336 | -0.521805 |
| I-A | +4.148561 | +4.041667 | +4.313821 | +3.212054 | +3.762938 | +3.929026 |
| C-A | +4.901480 | +3.549188 | +3.599286 | +5.525360 | +4.562323 | +4.393828 |
| U-A | +4.949355 | +2.966442 | +3.590785 | +5.578440 | +4.584613 | +4.271255 |
| S-A | +4.974905 | +3.383868 | +3.635042 | +5.809506 | +4.722274 | +4.450830 |

Four-order method min/max/range and every domain at every block position (OD/OC/macro Dice, conditional ASSD, paired signs and lower tails) are in public_aggregate.json. REFUGE_Valid I-C and I-A both remain visible; all other domain costs remain in the same tables.
All means weight domains equally. Drishti_GS remaining_dev has 37 groups but retains one-quarter domain weight. Lower-tail groups are post-hoc paired-error descriptions, not a frozen hard-case cohort. No significance or clinical thresholds.
The 20-update paired smoke checks entry equivalence on two procedural images per path/arm; it is not a new full native-memory-lifetime test. Historical lifecycle validation is reused. Formal calls are not equal FLOPs between A and BN methods.
No target images, masks, maps, optimizer snapshots or model snapshots were persisted. Scalar Dice is reconstructed; ASSD pixels were discarded. Optional B2 q-error bins are not backfilled. No additional model, seed, radius, source training, Polyp, routing or ensemble run is authorized by completion.
