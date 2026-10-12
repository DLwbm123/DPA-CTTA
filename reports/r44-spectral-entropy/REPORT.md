# R44: causal spectral entropy on retained CW_LSO

Status: **COMPLETE**. Frozen primary: CW_LSO_SE.

Own 2D per-binary-channel causal adaptation of ICLR 2026 spectral entropy regularization. Fixed coefficient0.01,5 radial bands excluding DC and outer corners,50 previous arrivals. Keep CW_LSO; no SCNP/GC/GP/RL. Not supervised reproduction; no independent calibration or anatomy claim.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_SE | 79.0463 | 78.4147 |

Primary SEARCH gain against C/W each>=0.3pp and against CW_LSO>=0.1pp. All three comparisons require both orders positive,>=5/6 trajectories positive,imageweighted>=0,worst seed-mean cell>=-2pp. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 10.671/48h; CPU-worker 0.151h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Incremental effect and negative results

The additional spectral branch yields only **+0.000715pp versus CW_LSO** on SEARCH (content-cluster bootstrap interval [-0.000837,+0.002422]pp). Order gains are -0.002429/+0.003859pp and only 3/6 trajectories improve. The required +0.1pp increment, both-positive orders and >=5/6 positive trajectories fail. The interval includes zero; this frozen setting provides no convincing additional Dice benefit.

Versus C/W, gains are +0.354781/+0.956693pp, both orders and all 6 trajectories positive. Imageweighted gains are +0.469676/+0.320225pp. These mostly retain the CW_LSO baseline's improvement and must not be attributed to the spectral branch. Worst seed-mean SEARCH cells versus W remain -5.099515pp (order 0, REFUGE_Valid, OC), -4.849082pp (same domain/order, OD) and -3.501607pp (order 1, REFUGE, OC); the -2pp protection gate fails. ALL_NEGATIVE_CELLS includes every adverse cell, including losses relative to CW_LSO.

## Mechanism observations and completion

On SEARCH, the entropy-deficit branch is active for 59.77% of OD and 54.02% of OC observations. Mean raw spectral loss is 0.00155793, contributing 0.0000155793 after the fixed 0.01 coefficient. Mean entropy is 0.174789/0.438303, reference entropy 0.175132/0.436570, and positive gap 0.013339/0.030999 for OD/OC. Mean reference count is 49.5059. SPECTRAL_BAND_POWER reports all five band-power means separately by channel and role; MECHANISM_DIAGNOSTICS also retains base/LSO losses, BN and regional logit gradients. The branch was active; small observed changes do not establish why it failed, and do not authorize increasing its coefficient. No gradient contribution decomposition or clinical calibration assessment was performed.

All 24 unique trajectories and 24 independent CPU scores completed with 46,824 rows. All receipt PIDs and the original watch/supervisor have retired; 52 attempts (4 profiles,24 online,24 scoring) completed without failure. GPU-worker cost includes all profiles: 10.671273883h; scoring CPU-worker cost 543.072s and preparation CPU cost 7.033796s. Execution wall time was 5.464350h from the unchanged T0. The cumulative nine-round GPU-worker cost is 71.520421494h. Shared GPU activity varied over the round, so condition wall times are not a controlled runtime comparison. No follow-up scientific experiment is authorized by this result.
