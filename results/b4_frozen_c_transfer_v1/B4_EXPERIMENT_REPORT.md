# B4 frozen C transfer and zero-update comparison

B4_FROZEN_C_TRANSFER_COMPLETE

Execution commit: e3c8f5ff80e626ee7e77bcbf75ebd99c71ffd1cb

No consistent positive transfer versus the prescribed strong controls. Limit prior C evidence to the tested Fundus setting; retain all negative Polyp results without retuning.

## polyp / remaining_dev

| Order | N | A | EA | O2 | D4 | C0 | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | 76.510373 | 79.428436 | 79.689126 | 78.734012 | 79.686450 | 74.307598 | 74.946326 |
| order1 | 76.510373 | 79.105353 | 79.256312 | 78.098019 | 77.616282 | 74.307598 | 73.133208 |

| Order | C-A | C-EA | C-C0 | C0-N | C0-A | C-O2 | C-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -4.482111 | -4.742800 | +0.638728 | -2.202776 | -5.120839 | -3.787686 | -4.740125 |
| order1 | -5.972146 | -6.123104 | -1.174390 | -2.202776 | -4.797756 | -4.964811 | -4.483074 |

## polyp / legacy_dev

| Order | N | A | EA | O2 | D4 | C0 | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | 78.461378 | 79.908197 | 81.034178 | 80.346747 | 80.412266 | 77.159592 | 77.173263 |
| order1 | 78.461378 | 79.738532 | 80.913022 | 78.294175 | 81.321255 | 77.159592 | 77.386514 |

| Order | C-A | C-EA | C-C0 | C0-N | C0-A | C-O2 | C-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -2.734935 | -3.860915 | +0.013670 | -1.301786 | -2.748605 | -3.173484 | -3.239004 |
| order1 | -2.352019 | -3.526508 | +0.226922 | -1.301786 | -2.578940 | -0.907662 | -3.934741 |

## polyp / p1_extension_dev

| Order | N | A | EA | O2 | D4 | C0 | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | 78.172439 | 81.054795 | 81.595596 | 79.369137 | 79.680475 | 78.802491 | 79.158248 |
| order1 | 78.172439 | 80.304710 | 81.282401 | 75.633052 | 80.582167 | 78.802491 | 78.049946 |

| Order | C-A | C-EA | C-C0 | C0-N | C0-A | C-O2 | C-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.896547 | -2.437348 | +0.355757 | +0.630052 | -2.252304 | -0.210889 | -0.522227 |
| order1 | -2.254764 | -3.232455 | -0.752546 | +0.630052 | -1.502219 | +2.416894 | -2.532221 |

## polyp / all_dev

| Order | N | A | EA | O2 | D4 | C0 | C |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | 76.951272 | 79.858266 | 80.068498 | 79.058924 | 79.903984 | 75.361448 | 75.868459 |
| order1 | 76.951272 | 79.463426 | 79.705695 | 78.085222 | 78.337377 | 75.361448 | 74.211589 |

| Order | C-A | C-EA | C-C0 | C0-N | C0-A | C-O2 | C-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -3.989808 | -4.200039 | +0.507011 | -1.589824 | -4.496818 | -3.190465 | -4.035526 |
| order1 | -5.251837 | -5.494107 | -1.149859 | -1.589824 | -4.101978 | -3.873634 | -4.125789 |

## fundus / remaining_dev

| Order | N | A | C | C0 |
|---|---:|---:|---:|---:|
| order0 | 68.200582 | 74.259279 | 79.104256 | 75.079420 |
| order1 | 68.200582 | 74.542945 | 78.006348 | 75.079420 |
| order2 | 68.200582 | 74.477689 | 78.007896 | 75.079420 |
| order3 | 68.200582 | 74.002581 | 79.326143 | 75.079420 |

| Order | C-C0 | A-C0 | C0-N |
|---|---:|---:|---:|
| order0 | +4.024836 | -0.820141 | +6.878839 |
| order1 | +2.926928 | -0.536476 | +6.878839 |
| order2 | +2.928476 | -0.601732 | +6.878839 |
| order3 | +4.246722 | -1.076839 | +6.878839 |

## fundus / legacy_dev

| Order | N | A | C | C0 |
|---|---:|---:|---:|---:|
| order0 | 69.092892 | 74.538998 | 79.306413 | 75.369676 |
| order1 | 69.092892 | 75.140702 | 77.919338 | 75.369676 |
| order2 | 69.092892 | 75.664870 | 78.062062 | 75.369676 |
| order3 | 69.092892 | 74.554914 | 79.410404 | 75.369676 |

| Order | C-C0 | A-C0 | C0-N |
|---|---:|---:|---:|
| order0 | +3.936737 | -0.830678 | +6.276784 |
| order1 | +2.549662 | -0.228975 | +6.276784 |
| order2 | +2.692386 | +0.295193 | +6.276784 |
| order3 | +4.040728 | -0.814763 | +6.276784 |

## fundus / p1_extension_dev

| Order | N | A | C | C0 |
|---|---:|---:|---:|---:|
| order0 | 66.452389 | 73.796285 | 79.497100 | 75.153030 |
| order1 | 66.452389 | 74.506257 | 78.551758 | 75.153030 |
| order2 | 66.452389 | 73.975280 | 78.497782 | 75.153030 |
| order3 | 66.452389 | 73.787197 | 79.931469 | 75.153030 |

| Order | C-C0 | A-C0 | C0-N |
|---|---:|---:|---:|
| order0 | +4.344070 | -1.356745 | +8.700641 |
| order1 | +3.398727 | -0.646773 | +8.700641 |
| order2 | +3.344752 | -1.177751 | +8.700641 |
| order3 | +4.778439 | -1.365834 | +8.700641 |

## fundus / all_dev

| Order | N | A | C | C0 |
|---|---:|---:|---:|---:|
| order0 | 67.964224 | 74.328553 | 79.230033 | 75.261256 |
| order1 | 67.964224 | 74.691960 | 78.241148 | 75.261256 |
| order2 | 67.964224 | 74.659026 | 78.258312 | 75.261256 |
| order3 | 67.964224 | 74.090025 | 79.615384 | 75.261256 |

| Order | C-C0 | A-C0 | C0-N |
|---|---:|---:|---:|
| order0 | +3.968777 | -0.932703 | +7.297032 |
| order1 | +2.979891 | -0.569296 | +7.297032 |
| order2 | +2.997056 | -0.602230 | +7.297032 |
| order3 | +4.354128 | -1.171232 | +7.297032 |

C-PraNet preprocessing: original and geometric views are CPU RGB with one original ImageNet channel normalization. Strong GraTa style works on an independent RGB numpy copy, then its RGB min/max normalization on CPU and exactly one ImageNet normalization. No probability min/max or output fusion. Fundus C is unchanged and reused.
C0 makes one current-statistics original-image forward per unique group. Its cross-order references are not extra predictions or independent repeats. Standard source-eval adapter parity and C0 versus first pre-update original view were checked in the fixed smoke.
All per-domain/channel paired signs, worst ceil(10%) differences, worst individual differences, conditional and common-valid ASSD are in the aggregate. No macro ASSD; CPU does not reconstruct discarded distance maps.
Network calls are not FLOPs. Old timings are historical and do not establish controlled speedups. No new source-clean scoring or source training.
All target data remain development data. No safety, significance, new algorithm or unseen generalization claim. No U/S/I/G expansion, DD training, radius/LR search or new domain order was executed. The B2/B3 interval series remains closed.
