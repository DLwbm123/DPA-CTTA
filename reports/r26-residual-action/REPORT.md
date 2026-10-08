# R26: bounded residual-space action exploration

Status: **COMPLETE**. Qualified complete matrix: **True**. Development signal: **False**.

All configurations and seeds are frozen before formal execution. No performance-based pruning or tuning. PG_RETAIN is the primary; a better extension does not replace it retrospectively. The policy is a contextual bandit, not long-horizon RL. Methods transfer selected mechanisms, not complete paper reproductions.

| Arm | REVIEW macro Dice (%) | Trajectories |
|---|---:|---:|
| C | 78.0791 | 6 |
| ANCHOR | 78.2244 | 6 |
| REPEAT5 | 72.6526 | 6 |
| RANDOM4 | 78.2272 | 6 |
| GREEDY | 78.2244 | 6 |
| PG_RETAIN | 78.2250 | 6 |

Each trajectory contains 1,951 arrivals, with 1,017 SEARCH and 678 campaign-held REVIEW identities plus 256 context identities. Both orders and all seeds share historically exposed content. Patient linkage and ROI crop provenance remain UNKNOWN. The 512px hard-Dice metric weights domain/channel equally; empty/empty Dice is 1. ASSD and soft metrics are not computed. The 64px proposal calibration is only a post-hoc mechanism diagnostic and is not primary Dice.

PAIRED_SUMMARY.csv and ALL_NEGATIVE_CELLS.csv include all registered contrasts and adverse cells. Content bootstrap couples orders/seeds and cannot establish patient independence. Costs include initialization, failed work, all temporary proposals, policy operations and profiling. REPEAT5 is an additional-update control, not exact FLOP matching. Public aggregate files exclude images, masks, checkpoints and identities.

Runtime wall: 12.307 h; GPU-worker: 44.377 h; CPU-worker: 0.209 h. No cumulative GPU-worker cap. Per-task deadlines, a seven-day supervisory safety timeout and 64 GiB storage guard remain. Code: `1b01ca6fe21a7e373480faedeb8d5e16c2854fce`.

GitHub final delivery is separate from execution completion.

R26 is a development follow-up selected after inspecting R25. Both SEARCH and the legacy SEALED_REVIEW field are development-exposed, not independent validation. The unchanged arm labels refer to residual-output actions in this round, not R25 gate-logit actions. The fixed six-arm, three-seed, two-order matrix has 36 trajectories; no extra seeds or performance pruning. PG_RETAIN remains primary even if another arm performs better.

## Completion audit and interpretation

All 36 online process receipts and all 36 CPU scorer receipts exited with code0.
Each online trajectory sealed1,951 arrivals (70,236 total). The last online exit
preceded label release, which preceded the first scoring start. Online execution
finished2026-10-08 10:27:56 Asia/Shanghai; scoring finished10:34:20. No failure was
recorded. Engineering completion is not evidence of method efficacy.

PG_RETAIN minus C is +0.145940pp, below the fixed0.3pp threshold. Order effects are
+0.383742 and -0.091861pp, with3/6 positive trajectories. Against ANCHOR the effect
is +0.000614pp (content-bootstrap interval[-0.002102,+0.003747]); against GREEDY it
is +0.000634pp (interval[-0.001061,+0.002199]). These changes do not support an added
benefit from learned residual-action selection. GREEDY minus RANDOM4 is -0.002805pp
(interval[-0.007224,+0.001796]), with2/6 positive trajectories. All adverse cells
and the unchanged poor REPEAT5 result remain included.

The intervention did increase action separation: PG_RETAIN's mean64px proposal
Dice range on the legacy review data rose from R25's0.007238pp to0.156347pp (~21.6x),
and rank-estimable observations increased from294/4068 to3430/4068. However, mean
within-case reward-versus-Dice Spearman remains0.01991. SEARCH candidate-mask
variance rose from0.000003139 to0.000090725 (~28.9x), and reward standard deviation
from0.00002266 to0.00047348. Greater action diversity alone therefore did not produce
useful selection. These are descriptive64px diagnostics on exposed data, not a
512px causal mediation proof or independent generalization result.

The next justified step is an offline audit of existing reward components on the
existing SEARCH candidate traces, before committing further GPU compute to another
policy variant. This will preserve all components and negative findings, rather
than search arbitrary reward weights until one appears favorable. The diagnostic
must not be treated as a new held-out test.
