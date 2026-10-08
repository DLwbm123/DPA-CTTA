# R29 update-strength diagnostic: completed, headroom gate failed

All six jobs completed on A100 GPUs 0/1/2: 96 prespecified SEARCH states from the R28 reference trajectories, three seeds and two orders. Each state was evaluated from the same saved pre-arrival state. No new full trajectory or executable selector was trained.

| Comparison with FULL | Macro joint OD/OC hard-Dice change (percentage points) |
|---|---:|
| ZERO: skip the update | -0.010500 |
| HALF: halve both adaptation learning rates | -0.005696 |
| Label-informed oracle over FULL, ZERO, HALF | +0.010225 |

The oracle gains were +0.010475 and +0.009975 pp in the two orders, with positive gains in all six seed/order groups. Its mean nevertheless fell far below the preregistered 0.3 pp allocation threshold. The headroom gate is **false**. Because FULL belongs to this oracle pool, nonnegative oracle headroom is guaranteed; it is not evidence for a usable selector. R28 excluded FULL from its four-candidate oracle, so the two raw oracle values do not constitute an equal-definition comparison.

## Verification and cost

All 96 FULL hard masks exactly reproduced the saved R28 reference masks. Mechanical checks covered true update skipping, half-sized parameter steps with unchanged Adam moments, and state restoration. Three real GPU smoke checks passed before the six formal jobs. All online and scoring receipts exited successfully, with zero online label reads; CPU label scoring followed successful retirement of every online job. There were no GPU execution failures.

The recorded run ended at 2026-10-08 22:20:19 +08:00. Wall time was 86.842 seconds; aggregate GPU worker time, including smoke checks, was 198.706 seconds (0.0552 hours), and CPU scoring took 3.926 seconds. These are worker-time receipts, not measured GPU energy or exclusive device occupancy. Prior R28 snapshot-generation cost is excluded from this incremental R29 ledger and remains reported with R28. A preparation-time filesystem metadata-copy error was resolved before launch by archive extraction; it consumed no GPU work.

## Decision and limits

Do not expand the strength grid, seeds, or states in response to this negative result. These three actions offer insufficient immediate hard-Dice headroom on the sampled reference states to justify a learned selector or a full efficacy matrix. No R30 was launched. A further experiment needs a distinct, evidence-supported mechanism and a separately frozen protocol.

This is a local one-step diagnostic on development-exposed data, not a closed-loop upper bound, independent confirmation, or proof that all adaptation mechanisms fail. Multi-step effects and unsampled states remain untested; patient/ROI provenance remains unresolved. Every state aggregate and domain/channel delta, including negative cells, is retained in the accompanying CSV files.

## Reproducibility and publication scope

Runtime code: `5435b84caffe89743e3d606b8fabd92dfb78b91c`. See [frozen protocol](../../docs/protocols/R29_UPDATE_STRENGTH_DIAGNOSTIC.md), [registration](REGISTRATION.json), [decision](DIAGNOSTIC_DECISION.json), [completion audit](COMPLETION_AUDIT.json), [state aggregates](COMMON_STATE_CELLS.csv), [domain/channel deltas](DOMAIN_CHANNEL.csv), and [cost ledger](COST.csv).

Published: source, protocol, aggregate results, verification summary, and costs. Private images, labels, masks, snapshots, weights, identities, and storage paths are excluded.
