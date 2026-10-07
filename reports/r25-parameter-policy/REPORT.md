# R25: conditional adapter parameter exploration

Status: **COMPLETE**. Qualified complete matrix: **True**. Development signal: **False**.

All configurations and seeds are frozen before formal execution. No performance-based pruning or tuning. PG_RETAIN is the primary; a better extension does not replace it retrospectively. The policy is a contextual bandit, not long-horizon RL. Methods transfer selected mechanisms, not complete paper reproductions.

| Arm | REVIEW macro Dice (%) | Trajectories |
|---|---:|---:|
| C | 78.0791 | 6 |
| ANCHOR | 78.2244 | 6 |
| REPEAT5 | 72.6526 | 6 |
| RANDOM4 | 78.2030 | 6 |
| GREEDY | 78.2277 | 6 |
| PG_CONS | 78.2256 | 6 |
| PG_STRUCT | 78.2237 | 6 |
| PG_RETAIN | 78.2248 | 6 |
| PG_COV | 78.2265 | 6 |
| RN_DPO | 77.7241 | 6 |

Each trajectory contains 1,951 arrivals, with 1,017 SEARCH and 678 campaign-held REVIEW identities plus 256 context identities. Both orders and all seeds share historically exposed content. Patient linkage and ROI crop provenance remain UNKNOWN. The 512px hard-Dice metric weights domain/channel equally; empty/empty Dice is 1. ASSD and soft metrics are not computed. The 64px proposal calibration is only a post-hoc mechanism diagnostic and is not primary Dice.

PAIRED_SUMMARY.csv and ALL_NEGATIVE_CELLS.csv include all registered contrasts and adverse cells. Content bootstrap couples orders/seeds and cannot establish patient independence. Costs include initialization, failed work, all temporary proposals, policy operations and profiling. REPEAT5 is an additional-update control, not exact FLOP matching. Public aggregate files exclude images, masks, checkpoints and identities.

Runtime wall: 25.505 h; GPU-worker: 88.824 h; CPU-worker: 0.361 h. Hard cap: 48h from registered T0; online cap 46h. Code: `bacbbf959a0b679f99f45de47b3e0143d80537e6`.

GitHub final delivery is separate from execution completion.

## Completion audit and interpretation

The 60 formal online process receipts all exited with code 0 and each sealed 1,951 arrivals (117,060 total). All 60 independent CPU scoring processes also exited with code 0. The last online exit preceded REVIEW release, and release preceded the first scoring start. Online execution ended at 2026-10-07 21:50:27 Asia/Shanghai; scoring ended at 22:01:29. No attempt failure was recorded. These are engineering and protocol checks, not evidence of method superiority.

PG_RETAIN improves over C by **0.1457 percentage points**, below the registered 0.3-point threshold. Its two order effects are +0.3783 and −0.0869 points; only 3/6 seed-order trajectories improve. Its change relative to ANCHOR is **+0.000385 points** (content-bootstrap interval −0.000966 to +0.001733), and relative to GREEDY is **−0.002942 points**. Thus the added learned policy, structural and retention mechanisms do not establish an advantage over the existing conditional adapter. The positive content-bootstrap interval against C does not override the failed predeclared gates or establish patient-independent significance.

RN_DPO is **−0.5036 points** relative to GREEDY, with zero positive seed-order trajectories. REPEAT5 reaches 72.6526% versus ANCHOR's 78.2244%, showing that additional repeated updates can be harmful in this configuration. Both negative results remain part of the delivery.

The diagnostic evidence motivates examining the action space before expanding policy learning: PG_RETAIN's mean four-proposal Dice range is only **0.00724 points** on campaign-held REVIEW at 64px, and only 294/4,068 observations have estimable within-case ranking. On SEARCH, mean full-resolution candidate-mask variance is 0.00000314, reward standard deviation is 0.0000227, and normalized advantage standard deviation is 0.00285. These are diagnostics on exposed content, not independent performance evidence. They suggest weak separation between candidates; they do not by themselves prove which architectural change will help.

Any follow-up will use a new experiment identifier, retain this frozen negative result, and first test a specific action-space hypothesis with fixed controls. SEARCH and REVIEW are now development-exposed; reusing them cannot provide an independent confirmation of a tuned method. A publication claim still requires appropriate independent validation and a separate novelty assessment.
