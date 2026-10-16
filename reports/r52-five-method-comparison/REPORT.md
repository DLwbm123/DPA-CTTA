# R52 five-method unified development comparison

Status: **COMPLETE**.

Primary: remaining_dev 1695 images; equal domains and OD/OC, then equal three seeds and two orders. Every adaptation stream has 1951 arrivals.
C and CW_LSO reuse R50; GraTa reuses R20. VPTTA and MGIPT are new. Historical cost is not a controlled speed comparison.
MGIPT is the previously registered paper_equation implementation on the shared checkpoint, not a new claim of exact released-code reproduction.
All subsets have historical development exposure. No independent generalization claim. No tuning, retry, additional seeds or next experiment.

| Method | Primary macro Dice (%) |
|---|---:|
| C | 78.437246 |
| CW_LSO | 78.786061 |
| G | 77.229480 |
| VPTTA | 74.402788 |
| MGIPT | 74.645267 |

Decision: {"status": "COMPLETE", "all_four_pass": false, "independent_validation": false, "automatic_followup": false}.
Four primary contrasts report ordinary 95% and Bonferroni-adjusted 98.75% content-bootstrap intervals; all seed/order observations of each content move together. Patient dependence remains unknown.
All four require >=0.1pp, positive both orders, >=5/6 positive trajectories, imageweighted >=0, worst seed-mean domain/channel/order cell >=-2pp, and adjusted lower bound >0.
Only anonymous aggregates and own code/protocol are public; private images, masks, model weights, individual content records, predictions and third-party code remain private.

## Interpretation and continuation

CW_LSO has the highest primary average, but only its comparison with C passes every frozen practical-gain condition. All four average deltas and multiplicity-adjusted lower bounds are positive; the worst domain/channel/order requirement fails against G, VPTTA and MGIPT. This is a heterogeneous trade-off, not uniform superiority.

| Baseline | CW_LSO delta (pp) | Adjusted interval (pp) | Worst seed-mean cell (pp) | Full gate |
|---|---:|---:|---:|---|
| C | +0.348815 | [0.301889, 0.394045] | -0.734948 | True |
| G | +1.556580 | [1.272050, 1.844118] | -6.395730 | False |
| VPTTA | +4.383273 | [3.767936, 5.024851] | -15.326778 | False |
| MGIPT | +4.140793 | [3.601576, 4.690280] | -13.060916 | False |

The dominant adverse cell is REFUGE_Valid OC in order 0: CW_LSO 58.985 versus VPTTA 74.312, MGIPT 72.046 and G 65.381. Conversely ORIGA OC in order 0 is 72.922 for CW_LSO versus 47.845 for VPTTA. These aggregate Dice differences alone do not determine whether errors are false positives or false negatives.

All 12 new streams and CPU scorers succeeded, with 23412 new scored visits; combined coverage is 30 trajectories and 58530 rows. Retirement timing is separately recorded. New execution consumed 12.0964 GPU-worker hours and 258.20 CPU-worker seconds, including profiles, with no failed GPU/scoring attempt. Historical cost is separately labelled. The initial CPU preparation import issue is documented in STARTUP.md.

After startup the user authorized continued analysis and subsequent experiments. This authorization supersedes the original operational no-follow-up instruction but does not change this round's frozen scientific decision or results. A separately registered mask-union/intersection diagnostic will test spatial complementarity using retained predictions, without selecting thresholds or feeding labels into either adaptive trajectory. It cannot establish a single-model improvement or independent validation.
