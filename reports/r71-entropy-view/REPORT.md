# R71 result: selection improves on random gamma, not on retained CW_LSO

The registered allocation gate failed. All six source-initialized 192-arrival trajectories completed and all 1,152 outputs were scored after GPU-worker retirement. Qualification passed exact retained-host parity and repeat-state checks in both orders. Both worker and scorer exited zero; no retry occurred.

| Contrast | Macro Dice delta (pp) | Conditional 95% interval | Order 0 | Order 1 |
|---|---:|---:|---:|---:|
| SELECT − NATIVE | -0.023882 | [-0.069472, 0.024628] | +0.095153 | -0.142918 |
| SELECT − RANDOM | +0.231611 | [0.118083, 0.367471] | +0.119763 | +0.343458 |
| RANDOM − NATIVE | -0.255493 | [-0.400635, -0.140189] | -0.024609 | -0.486376 |

Macro Dice: NATIVE 75.951890%, RANDOM 75.696397%, SELECT 75.928007%. SELECT-NATIVE image-weighted delta is -0.023312 pp; worst cell is ORIGA/order1/OC at -0.737532 pp. The primary contrast fails the required +0.2 pp, positive interval lower bound, both-orders-positive and nonnegative image-weighted conditions. Passing the SELECT-RANDOM subgate alone cannot promote the method. No equivalence test was registered: a confidence interval crossing zero is not proof of equivalence.

## Interpretation and all cells

Selection reduces the harm caused by replacing native augmentation with uniformly sampled gamma. It does not demonstrate an improvement over the actual retained method. ORIGA is positive in order0 (+0.546 OD, +0.939 OC) but negative in order1 (-0.450 OD, -0.738 OC), while REFUGE_Valid reverses the other way. These are order-dependent development results, not evidence for domain routing. [ALL_CELLS.csv](ALL_CELLS.csv) includes every one of the 48 contrast/domain/order/channel cells, including all negatives.

Selection is not collapsing to the identity. In order0 it chooses gamma .7 / 1 / 1.428571 on 144 / 1 / 47 arrivals; in order1, 140 / 3 / 49. RANDOM uses 60 / 77 / 55 in each order by its registered independent RNG. Thus SELECT uses nonidentity views on 380/384 arrivals. Its higher objective is expected from argmax and does not establish pseudo-label correctness. Average input change is about .092, versus .056 in RANDOM; candidate distributions differ by design, so the measured benefit is not a magnitude-matched causal effect of entropy alone.

The mean BN update norm is .00439–.00454 for SELECT, versus .00422 for NATIVE; no gradient-norm matching was imposed. SELECT takes about 2.13–2.17 seconds/image, NATIVE 1.37–1.58, under shared-device conditions. Modified arms each perform four additional read-only forwards per arrival (1,536 per arm over both orders), with their host state restored exactly. Timing differences are observational, not a controlled throughput benchmark. NATIVE's gamma=1 and change=0 in MECHANISM.csv are diagnostic defaults, not descriptions of its actual native strong augmentation.

## Evidence and costs

[QUALIFICATION.json](QUALIFICATION.json), [COMPLETION_AUDIT.json](COMPLETION_AUDIT.json), [RESULTS.json](RESULTS.json), [ARM_SCORES.csv](ARM_SCORES.csv), [MECHANISM.csv](MECHANISM.csv), and [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json) provide the anonymous receipts. Full GPU process cost is 2,360.125521 seconds (39.34 minutes); the inner worker-body timer 2,356.675966 seconds omits process overhead and is not substituted for it. CPU total is 33.555900 seconds: preparation/check 19.602170 plus scoring 13.953730. Online label reads: zero. No source training, new model, or extra cohort.

Code/protocol was registered at `998705b`, before qualification and scoring. This is an exposed development cohort with two orders of the same 192 contents; bootstrap intervals are conditional on that design, not independent validation. Private content rows, predictions, labels, images and weights are excluded from publication.

## Decision

Retain native CW_LSO. Close this fixed gamma-selector configuration without gamma/coefficient/seed/horizon scans or favorable-cell promotion. R71 tests a finite current-student selector inspired by TeSLA's augmentation objective. It does not test learned policy probabilities/magnitudes, the original EMA teacher, running augmentation average, or full TeSLA. Neither R70 nor R71 justifies a claim that the full method fails.

The next research decision must identify an independent, previously untested mechanism and check its fit to nested OD/OC outputs. Do not combine the failed fCE replacement with this selector or tune either in response to these scores. No successor experiment is registered by this report.
