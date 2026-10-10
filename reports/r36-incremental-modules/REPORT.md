# R36: incremental modules on the retained W baseline

Status: **COMPLETE**. Fixed primary: W_TP. No controller training.

W retains R20 six-view pseudo supervision, variance/boundary weighting, GraTa and LR multiplier 1.5. C is the strong plain-consistency control. TP is a simplified cross-threshold connected-component overlap weight, not a PH/OT TopoOT reproduction. LSO/LSM are independently implemented sigmoid structure-tensor losses using current detached soft pseudo-targets. BAL borrows frequency weighting for BCE; it is not original entropy-based DSBR.

All seven conditions, three seeds and two orders were frozen before labels. Each full stream has 1,951 arrivals. No source retraining, external models, online labels, output ensemble or scientific retries. SEARCH and legacy REVIEW have historical exposure; REVIEW is descriptive, not independent confirmation. Patient linkage is UNKNOWN.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| W_TP | 78.1354 | 77.5683 |
| W_LSO | 78.2096 | 77.6483 |
| W_LSM | 77.9191 | 77.3780 |
| W_LS | 78.0930 | 77.5395 |
| W_BAL | 78.1317 | 77.5666 |

PAIRED_SUMMARY includes both W and C comparisons. ALL_NEGATIVE_CELLS retains all adverse domain/channel/order/seed cells. A development signal requires SEARCH gain >=0.3pp against both W and C, both orders positive, >=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell >=-2pp. These are investment thresholds, not clinical or significance thresholds. Secondary candidates cannot retrospectively replace the primary.

GPU-worker 14.370/48h; CPU-worker 0.261h. Wall limit 24h. Failed preparation/profile/formal attempts remain recorded; incomplete cells are not success or zero.

Dice uses 2TP/(prediction+GT), or 1 when both empty; equal domains and OD/OC, then equal seeds/orders. ASSD, soft Dice and Brier were not computed. Bootstrap couples repeated content across seeds/orders and does not resolve patient dependence. Novelty and medical benefit are unestablished.

Only anonymous aggregate results, own implementation, settings, cost and audits are public. Data, masks, predictions, model states, private logs and third-party PDFs remain private. Execution completion and verified GitHub delivery are distinct.

## Conclusion and next hypothesis

None of the five frozen modules passed the registered development gate against both W and C. The primary W_TP gained +0.0458pp over W but lost -0.5561pp to C and had negative imageweighted gain over W. Secondary W_LSO gained +0.1200pp over W (6/6 positive trajectories, both orders positive, content-cluster CI [0.0965,0.1437]pp) but lost -0.4819pp to C. W_LSM lost -0.1706pp to W in 6/6 trajectories; combined LS gained only +0.0034pp and 2/6 trajectories. BAL gained +0.0421pp but imageweighted gain over W was negative. These are complete negative investment decisions, not missing experiments.

The SEARCH W-to-C gap is uneven: ORIGA OD/OC and Drishti OD/OC lose while REFUGE_Valid gains. LSO mildly improves W without reversing this tradeoff. SEARCH mean BN gradient norms are C0.3046, W0.4072 and W_LSO0.5040 (see diagnostics for exact values). Associations do not prove a causal optimizer or weight mechanism. The authorized follow-up will test the same W pixel weighting plus fixed LSO on the plain-consistency C host, with exact matched C/W controls and a C+W ablation. This is a new pre-registered experiment, not a retrospective change of the R36 primary. No REVIEW-guided choice is made.

Public receipt provenance hashes for streams, predictions, scalar scores and traces were removed at publication; original private receipts remain intact. Source/configuration commit identifiers, anonymous counts, costs and all negative aggregates remain available.
