# R71: bounded entropy-seeking update-view selection

Registered before target qualification/scoring. This is an independent augmentation component study following the negative R70 fCE result. CW_LSO's native weighted soft BCE and LSO coefficient .1 remain unchanged. No fCE, source training, EMA teacher, new model, geometry transform, or target-label selection.

## Hypothesis and precise scope

Selecting a difficult but normalization-compatible current-image view may improve adaptation over random view selection. Inspired by TeSLA's optimized augmentation objective (official code revision `69870cc7d6f1caee48a6264ad54135224cb54c69`, `networks/augmentation/segoptaug.py`), we test a finite discrete selector, NOT its learned policy, not its full method. The original implementation also optimizes probabilities/magnitudes, uses its teacher and a running augmented average. Here current frozen student predictions and a symmetric candidate average replace those components. A negative result excludes only this transfer.

Three independent source-initialized arms: NATIVE (unchanged CW_LSO), RANDOM, SELECT. The latter two replace the native strong update view with normalized current RGB raised to gamma in the fixed set {0.7, 1, 1/0.7}. Global gamma is monotone, preserves spatial coordinates, and survives per-image min-max normalization. Identity is included; selecting identity is an observed mechanism outcome, never a reason to retune. Native augmentation still runs and is discarded to preserve its random stream. Original six-view soft pseudo-target and final original-image output are unchanged.

Both modified arms make four extra read-only forwards: clean reference plus three candidates. For each BN layer collect its input spatial mean; penalty is squared difference from clean, averaged across batch/channels and then layers. Let p_j be two independent sigmoid channels, and a the candidate-mean p. Score_j = H_Bernoulli(p_j) - mean_penalty_j + KL_Bernoulli(a || p_j), all coefficients 1; probabilities clamped to [1e-6,1-1e-6]. This reproduces the *signs* of the actual code's entropy, norm-mean and KL terms, not its sequential policy training. OD and OC are nested and never a categorical softmax. SELECT chooses first argmax; RANDOM chooses uniformly with a separate deterministic RNG seeded by 20260907+710003+zero-based visit. Both evaluate all candidates for matched computation. Score improvement is definitional, not proof of segmentation accuracy.

Preview uses existing read-only forward and full host snapshot/restore; every selection verifies restoration including BN parameters, buffers, gradients, optimizer, counters, RNG, and modes. Preview counters are reported separately (4 per modified arrival), not presented as absent physical work. No future images, labels, or stored predictions enter selection.

## Fixed design and gates

Exactly existing R23 SEARCH cohort: 192 contents, domains D22/O57/F57/RV56; original two orders and seed 20260907. Six independent 192-arrival closed loops, total 1152 outputs. New baseline recomputed; no substitution of R23's older host or previously scored trajectories. SEARCH/REVIEW already exposed: development evidence only.

Before formal arrivals, each order's first image checks NATIVE exact full-state/output parity with retained host and each arm's repeat-from-identical-state output/full-state equality. Frozen weights checked. Maximum measured step duration x1152 x1.5 plus consumed qualification time must fit budget. No formal allocation if qualification fails.

After successful retirement of every online worker, CPU-only scoring requires all 1152 predictions and all six completion receipts. Primary metric: domain-macro mean OD/OC Dice, averaged over orders. Content-paired domain-stratified bootstrap 2000 draws with seed 20261016. Report all 48 contrast/domain/order/channel cells and all arm scores.

Conjunctive allocation gate: SELECT-NATIVE >=0.2 pp, 95% CI lower >0, both order deltas >0, image-weighted delta >=0, worst cell >=-2 pp; AND SELECT-RANDOM >=0.1 pp, CI lower >0, both order deltas >0. RANDOM-NATIVE is descriptive. No tuning of gamma, coefficients, seeds, horizon, or favorable-domain gates after results.

One GPU process capped at 5400 seconds including qualification; CPU scoring cap 900 seconds; output cap 2 GiB; no automatic scientific retry. Store setup, qualification, full GPU process, scoring and failed costs. A zero/failed scientific gate closes this configuration. Preserve previous evidence. Source, protocol, anonymous results and report delivered to GitHub through proxy; private inputs/predictions remain remote.
