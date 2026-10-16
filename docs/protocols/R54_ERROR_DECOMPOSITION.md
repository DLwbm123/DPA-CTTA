# R54 fixed confusion-count decomposition

R52 showed a large REFUGE_Valid OC deficit of CW_LSO versus VPTTA. R53 union failed to repair it. Hypothesis: excess foreground, rather than missing foreground, dominates this relative deficit. This is a descriptive diagnostic, not a new algorithm or an efficacy trial.

Reuse only existing R52 private scoring records for CW_LSO and VPTTA: seeds 20260907,17011,29009; both orders; remaining_dev1695. Exact paired content/visit/domain/subset checks and confusion-count identities are required. No images, labels, predictions or model state will be regenerated/read. Aggregate TP/FP/FN by method/seed/order/domain/channel. Report FP/GT, FN/GT and predicted/GT, using ratios of pooled counts within each cell, not average image rates. Publish all cells and both error directions; retain raw per-image records privately.

Fixed primary diagnostic: for REFUGE_Valid OC, d=(FP_CW-FP_VPTTA)/GT - (FN_CW-FN_VPTTA)/GT. Support the excess-foreground hypothesis only if mean d>=0.05 and d>0 in at least5/6 pairs. Failure is inconclusive or opposing evidence and does not authorize promoting a different cell. No significance/causal claim. All data are development-exposed and no independent generalization can be claimed. The result can guide a separately registered follow-up, never retrospectively change R52/R53.

One CPU execution;10-minute hard wall cap,100MiB output,zeroGPU. Existing aggregation writer and JSON receipts; tiny assert self-check of normalization and zero-denominator behavior. Complete only with10170 paired primary observations and all expected cells, report and public GitHub delivery. No retries for unfavorable science.
