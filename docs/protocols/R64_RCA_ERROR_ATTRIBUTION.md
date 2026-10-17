# R64 RCA error attribution

Registered after R63 aggregate results, before reading its private choice traces and paired scores for this analysis. R63 failed its primary gate (-1.0394pp versus CW); the whole-mask oracle retains +2.5751pp headroom. This motivates a descriptive error diagnosis, not a retry or a new selector.

Use only R63's completed 96 choice records, 48x2x3x2 stored Dice array, and registered domain names. No image, label, source reference, model, or forest is read or fitted. Hypothesis: harmful switches to VPTTA and missed beneficial switches account for the observed loss and regret; RCA margin may fail to track actual joint Dice advantage.

Report fixed aggregate counts and mean/sum Dice advantage for correct switches, harmful switches, correct holds and missed switches; per-order/domain decomposition; Pearson correlation between signed RCA advantage (VPTTA minus CW) and true joint Dice advantage; mean absolute RCA margin for correct and incorrect decisions. Zero target advantage is classified separately. No threshold sweep, bin search, calibration fit or policy change. These are exposed-development diagnostics without an efficacy pass gate. A usable result requires all 96 pairs, finite values, selected-mask consistency, and exact decomposition identities within floating tolerance.

One CPU process, <=60 seconds, <=10MiB new outputs, zero GPU, no retry for scientific negatives. Publish only aggregated metrics, code and report. Any later intervention needs its own frozen hypothesis and registration; do not reuse these targets as independent validation.
