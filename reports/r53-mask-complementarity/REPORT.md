# R53 fixed spatial-complementarity diagnostic

| Output | Primary macro Dice (%) |
|---|---:|
| CW_LSO | 78.786061 |
| VPTTA | 74.402788 |
| UNION | 78.728491 |
| INTERSECTION | 74.308258 |

Frozen UNION development gate: False.
This replays fixed per-image output operators on stored predictions from two independently adapted models; it performs no new training. Labels were first read after all union/intersection masks were written. Union and intersection do not use domain labels, image labels, thresholds or fitted selectors.
UNION minus CW_LSO is the sole primary inferential contrast. Other contrasts are descriptive; no best-arm promotion or independent-validation claim. Both source model execution costs would be needed in deployment. Offline CPU runtime is not end-to-end deployment latency.
All content is development-exposed. Frozen predictions, individual records and labels remain private.

## Interpretation and continuation

UNION minus CW_LSO is -0.057570pp (95% content CI [-0.197186, 0.085557]); all six paired trajectories are negative and the image-weighted difference is -0.900073pp. The worst seed-mean cell versus CW_LSO is -2.686813pp, and versus VPTTA -15.561195pp. Thus the registered development gate fails; no success claim or secondary-arm promotion is warranted. REFUGE_Valid OC order0 changes by -0.234417pp versus CW_LSO, so union does not repair the motivating deficit. INTERSECTION loses -4.477803pp versus CW_LSO.

The next separately registered diagnostic will reuse existing TP/FP/FN scalar records to distinguish excess and missing foreground; no additional model training or target-mask reading is needed. Existing R52/R53 results remain frozen.
