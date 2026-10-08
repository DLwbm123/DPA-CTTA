# R30: delayed effects and cross-arrival state history

Status: **COMPLETE**.

H=64 local future oracle gain: +0.009007 pp; ZERO -0.005768, HALF -0.000568. Delayed allocation signal: False.

Any simple state intervention passed the development gate: False. Stronger fixed SEARCH reference: ANCHOR.

| Arm | SEARCH macro Dice (%) |
|---|---:|
| ANCHOR | 78.815873 |
| C_BOTH32 | 75.782003 |
| C_CONT | 78.691553 |
| C_EPISODIC | 75.466015 |
| C_OPT32 | 78.090739 |
| C_PARAM32 | 75.900853 |
| S_BATCH | 75.279662 |
| S_SOURCE | 68.678895 |

All paired deltas against both C_CONT and ANCHOR, six trajectory effects, descriptive coupled-content intervals, and negative domain/channel cells are in PAIRED_CONTRASTS.csv and ALL_NEGATIVE_CELLS.csv.

- C_BOTH32: -3.033869 pp against ANCHOR; worst seed-averaged domain/channel/order cell -13.966281 pp; gate False.
- C_EPISODIC: -3.349857 pp against ANCHOR; worst seed-averaged domain/channel/order cell -14.605028 pp; gate False.
- C_OPT32: -0.725134 pp against ANCHOR; worst seed-averaged domain/channel/order cell -5.862320 pp; gate False.
- C_PARAM32: -2.915019 pp against ANCHOR; worst seed-averaged domain/channel/order cell -13.221281 pp; gate False.

Next decision: **STOP_REGISTERED_ACTION_HOST_PERIOD_SWEEPS**. No R31 or RL training was launched.

Recorded GPU-worker time: 15.5610 h; wall time: 6.1774 h. Includes profile and failed attempts. R28 snapshot-generation cost is excluded from this incremental ledger.

SEARCH and old REVIEW are development-exposed; REVIEW is descriptive only. Patients and ROI provenance are UNKNOWN. Windows overlap and reuse content across seeds: they are not independent patients. Oracle includes FULL and is not executable. OPT32 resets moments and bias-correction time together; parameter-only restoration preserves potentially mismatched optimizer history. Reset period 32 is fixed, not selected from results. Interactions and bootstrap intervals are descriptive, without patient independence or selection-adjusted inference. No soft metric is inferred from hard masks.

Published artifacts contain code, protocol, aggregated metrics, and audit/cost summaries. Images, labels, masks, states, weights, private paths, and identities are excluded. The final public commit and anonymous access are verified separately during delivery.

## Mechanistic interpretation

The registered state-reset hypotheses are not supported. SEARCH gives the following fixed contrasts:

| Contrast | Difference (pp) |
|---|---:|
| Batch-stat frozen configuration minus native source eval | +6.600766 |
| One episodic C update minus batch-stat frozen configuration | +0.186354 |
| Continuous C minus episodic C | +3.225538 |
| Adam-only reset every 32 arrivals minus continuous C | -0.600815 |
| Affine-only reset every 32 arrivals minus continuous C | -2.790700 |
| Reset both every 32 arrivals minus continuous C | -2.909550 |
| ANCHOR minus continuous C | +0.124320 |

The model inventory contains no Dropout modules. These contrasts support a useful role for current-image normalization and retained learning history in this specific implementation. They do not prove all possible historical states are helpful or identify a unique biological or optimizer mechanism. The episodic and continuous conditions visit different model states by design. All reset candidates lose in both orders and all six groups relative to ANCHOR. Legacy REVIEW agrees in direction: continuous minus episodic is +3.098258 pp, while OPT32/PARAM32/BOTH32 minus continuous are -0.568943/-2.602535/-2.742800 pp.

ANCHOR's small SEARCH mean advantage is order-sensitive (+0.316288/-0.067649 pp), so it still does not provide a robust >=0.3 pp improvement. Do not replace the SEARCH endpoint with old REVIEW values. Older runs used different hardware; within-round references are the correct paired controls.

Delayed-window oracle gains remain +0.008586, +0.008381 and +0.009007 pp at H=8/32/64. Longer rollout does not reveal allocation-scale headroom for these one-step strength actions on the sampled ANCHOR states. Positive oracle values in all six groups do not establish deployable control because FULL is in the pool.

All 54 online jobs retired successfully before CPU label release. The audit verified 110,703 total algorithm/branch arrivals, 5,877 exact reference-mask checks, zero online label reads and zero failures. CPU scoring used 773.823 seconds. Total wall time was 6.1774 hours and GPU-worker time 15.5610 hours. The later user-authorized GPU-budget amendment preserved the original T0 and every cost receipt.

The next research decision is to stop the registered strength/reset sweeps. Continuing work requires a distinct hypothesis, such as testing the adaptation representation while retaining useful state; it must not be presented as an established benefit, a novel method, or a reason to train RL immediately. No independent confirmation is available.
