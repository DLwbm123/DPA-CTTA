# R24: C strong baseline and anchored conditional adaptation

Status: **COMPLETE**. Strong development signal: **False**.

The SEARCH advancement gate failed: ANCHOR-C was +0.1311 pp (required +0.2), with order changes +0.3452/-0.0830 pp; ANCHOR-STATIC was +0.0070 pp (required +0.1); ANCHOR-VIEW was -0.0094 pp (required +0.1). The worst SEARCH cell against C_LR15 was -2.0127 pp, below the -2 pp limit. Confirmation seeds were therefore not launched. This is a completed one-seed negative screening result, not a failed or missing run.

On descriptive REVIEW, ANCHOR-C changed +0.4046/-0.1132 pp across the two orders. The shared condition did not improve upon the matched VIEW control in either order. The small average improvement over C is also present with the static adapter; these data do not establish an added benefit from anchoring the condition. No causal claim about why the mechanism failed is made.

The frozen primary is ANCHOR; C is the main baseline. STATIC and VIEW are mechanism controls, and C_LR15 checks a simple learning-rate explanation. No W weighting, reliability gate, source retraining, source images, additional pretrained encoder or online labels are used.

Full-stream arrivals per trajectory: 1,951; primary SEARCH/REVIEW identities: 1,017/678. Frozen seeds: [20260907]. All content was historically exposed; this is campaign-held review, not independent generalization. Patient linkage and ROI crop-center provenance remain UNKNOWN.

| Condition | REVIEW macro Dice (%) | Imageweighted (%) | Seed-order trajectories |
|---|---:|---:|---:|
| C | 78.2053 | 77.6156 | 2 |
| C_LR15 | 77.7958 | 76.3992 | 2 |
| STATIC | 78.3408 | 77.6992 | 2 |
| VIEW | 78.3594 | 77.7284 | 2 |
| ANCHOR | 78.3510 | 77.7032 | 2 |

| ANCHOR vs control | REVIEW delta (pp) | Positive trajectories | Worst seed-mean cell (pp) |
|---|---:|---:|---:|
| C | +0.1457 | 1/2 | -1.0561 |
| C_LR15 | +0.5552 | 2/2 | -2.2878 |
| STATIC | +0.0103 | 1/2 | -0.2000 |
| VIEW | -0.0084 | 0/2 | -0.1679 |

Adverse REVIEW cells against C (all six of sixteen domain/channel/order cells):

| Order | Domain | Channel | Delta (pp) |
|---|---|---|---:|
| 0 | ORIGA | OC | -0.2259 |
| 0 | Drishti_GS | OD | -0.4750 |
| 1 | REFUGE_Valid | OD | -1.0561 |
| 1 | REFUGE_Valid | OC | -0.6486 |
| 1 | REFUGE | OD | -0.0874 |
| 1 | REFUGE | OC | -0.2332 |

A negative gate is retained. A winning static adapter, higher learning rate or per-view condition does not retrospectively become the registered primary. Conditional adaptation is established prior art; this experiment does not establish novelty. The descriptor may capture content as well as appearance. Zero-U initialization does not make later pseudo-labels correct.

Dice is 2TP/(predicted+GT), or 1 when both masks are empty. Domain and OD/OC receive equal weight; orders and seeds are then averaged. ASSD and soft-probability metrics are NOT_COMPUTED in this bounded round. FULL_RESULTS.csv, DOMAIN_CHANNEL.csv and ALL_NEGATIVE_CELLS.csv retain denominators and adverse results. Bootstrap resamples content within domains and couples orders/seeds; it does not resolve patient dependence or selection bias.

Elapsed runtime: 0.959 h; GPU workers: 3.047/12 h; CPU scorers: 0.077 worker h. Code: `bcdd706ad6fc5034ce08d307c6eac7eed0ce5708`. Timing includes initialization, I/O and actual concurrency; no exact equal-cost superiority claim.

All attempts, including profile and failed/recovered work, are charged in COST.csv. Code/protocol/aggregate metrics are public; images, labels, patient/content identifiers, predictions, checkpoints, private paths and credentials remain private. GitHub delivery is recorded separately after proxy push and anonymous verification.

Completion audit: all ten formal trajectories completed 1,951 arrivals (19,510 total), and all 34 physical attempts completed without a retry or failure. All registered processes exited. SEARCH scorers started after the last online worker retired; REVIEW was released after SEARCH scoring and before final CPU scoring. Recorded online label reads and source-image reads are zero. See COMPLETION_AUDIT.json and RESOURCE_LEDGER.json. Runtime ended at 2026-10-06 18:27:47 Beijing; output usage was 1.395 GiB.
