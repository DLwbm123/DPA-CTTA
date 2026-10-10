# R37: fixed C+W orientation transfer

Status: **COMPLETE**. Frozen primary: CW_LSO.

R36 found a consistent but small W_LSO improvement over W; all modules still lost to C. R20 previously showed a one-seed C-host W transfer signal. This follow-up tests that transfer across three seeds/two orders, with CW isolating the added LSO contribution. The host transfer also removes the native G entropy/perturbation and dynamic learning rate and uses fixed C learning rate; it is a bundled host change, not causal proof.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW | 78.8545 | 78.2293 |
| CW_LSO | 79.0456 | 78.4130 |

Stage success requires CW_LSO SEARCH gains >=0.3pp against both C and W and >=0.1pp against CW; each comparison must pass both orders, >=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell >=-2pp. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 7.770/48h; CPU-worker 0.150h. Original per-round T0 and attempts retained.

Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, extra views, output fusion or scientific retries. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Complete outcome and remaining failure

CW_LSO SEARCH gained +0.354066pp against C (6/6 positive; orders +0.189111/+0.519021pp; imageweighted +0.458205pp), +0.955978pp against W (6/6 positive; orders +1.269112/+0.642844pp; imageweighted +0.308754pp), and +0.191136pp against CW (6/6 positive). Thus the average gains and the LSO increment are supported in this development matrix. The full registered gate nevertheless failed: worst seed-mean cell against W is -5.155135pp (order0 REFUGE_Valid OC), with order0 REFUGE_Valid OD -4.908927pp and order1 REFUGE OC -3.528396pp. None of these adverse cells is omitted or excused by the mean gain. Against C and CW the worst cells are -0.788639/-1.066813pp.

SEARCH mean final learning rates are C/CW/CW_LSO1e-4 and W3.695513e-5; mean BN displacement norms are CW_LSO0.003228 and W0.001402. This is consistent with a step-size-related tradeoff, but the host change also removes the G entropy/perturbation pathway and the statistics cannot establish causality. The next bounded hypothesis is an added post-Adam BN displacement trust cap, retaining CW_LSO loss, views and pseudo-target; it tests whether limiting larger updates preserves the positive mean gain while protecting the adverse cells. The single proposed radius0.0015 is chosen from observed SEARCH-linked development diagnostics, not REVIEW, and must be frozen before a new round. A failed cap will be a failed hypothesis, not evidence to tune its radius automatically.

R36+R37 cumulative GPU-worker cost is22.140055h, including all profile/formal attempts. Completed-round wall time is2.782731h for R37; preparation CPU cost remains separately recorded. No independent medical confirmation has been performed.
