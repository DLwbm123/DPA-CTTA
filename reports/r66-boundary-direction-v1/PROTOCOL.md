# R66: local signed boundary evidence, frozen output diagnostic

Registered before real feature inference or new target scoring on 2026-10-17. This is an A-route information probe, not endorsement of A as the TMI main method. The latest literature assessment recommends a task-trained adaptation objective as a possible new main route; additional source training is not authorized by this protocol.

## Fixed inputs and scope

Reuse 48 previously exposed SEARCH contents (12 per target domain), the two seed17011 CW_LSO historical outputs already bound in R63/R65: 96 pairs. Retain original stream positions and archive receipts privately. No temporal checkpoint reconstruction, model-state replay, source images/masks, source fitting, candidate selection, or persistent target adaptation. A CPU input qualification found all96 CW masks nested; no repair was applied. These are development data, not independent validation; two orders are not two independent content samples.

Use the existing bound source ResUNet checkpoint and native Fundus min-max input normalization. Model in eval, gradients disabled, BN current-image statistics with running buffers disabled, as in the existing source segmenter. Extract native up3 float32 shape256x128x128 once per image (one first-image repeat for determinism:49 forwards total). Verify frozen parameters/buffers unchanged after the entire worker. No other feature layer, reference-image bank, or test-time optimization.

## Fixed geometry and controls

Use the prepared geometry without hyperparameter search: inward offsets-4,-3,-2; outward2,3,4; tangent-1,0,1; at least4 eligible side samples, cup exterior restricted to disc rim, unique inside-positive/outside-negative crossing within[-1,1]. All offsets are native128-grid pixels. Original512 output receives at most4px signed displacement, nearest-boundary ownership including abstainers, zero-displacement identity, nesting-conflict veto. Invalid input fails the run.

Seven branches: unchanged CW; LOCAL; SHUFFLE; SIGN0..SIGN3. SHUFFLE permutes all16384 spatial feature vectors jointly across channels before the same geometry. Its admissible support and edit area can differ; report both and do not interpret this comparison alone as a magnitude-matched sign test. Four SIGN controls retain LOCAL locations and exact absolute displacement fields, assigning independent random signs. Pixel edit areas can differ after rasterization and nesting veto and must be reported. One NumPy RNG seed20261017+pair index supplies each pair's permutation then four sign draws; no seed selection. All controls use identical reconstruction.

## Scoring and prospective gates

Only after the sole feature/output worker retires successfully, an independent CPU scorer reads target masks. Report all7x2x4x2 domain/order/channel absolute means, all contrast cells and every negative cell, support/abstention and inward/outward counts, edit pixels and vetoes. Main metric is equally weighted OD/OC hard Dice in percentage points, equal content/domain/order weighting (balanced panel). Three contrasts: LOCAL-CW, LOCAL-SHUFFLE, LOCAL-mean(SIGN0..3). Use existing stratified content bootstrap2000 draws, seed20261016, keep orders/channels paired. Confidence intervals are descriptive on repeatedly exposed development data, not independent confirmatory inference.

Direction gate is conjunction: LOCAL-CW mean>=0.10pp; all three contrast95% lower bounds>0; LOCAL-CW positive in each order; every LOCAL-CW domain/order/channel cell>=-1.0pp. No gate retuning, order/channel exclusion, parameter sweep or automatic efficacy retry. Failure is evidence against this fixed local signal, not a proof against all learned image-conditioned methods. Output gains alone cannot establish update usefulness or continual learning; those require separate registrations.

## Finite resources and qualification

One GPU feature/output worker capped1800 wall seconds; one CPU scorer capped900 seconds; new prediction/audit/log artifacts<=2GiB; two CPU threads. Reuse stored masks and existing code/runtime. Before GPU start, check device UUID/free memory>=6GiB and NAS mount/free capacity/write-read probe. First image repeat must be exact; projected remaining work uses1.5x first-image cost plus60s and must fit the cap. Preserve any failed attempt and costs; no score-based retry. Existing preparation cost1.018630028 CPU wall seconds plus measured current check/setup cost is carried in the ledger. GPU worker wall time includes its CPU geometry computation and is conservatively charged in full. No source-training authorization is inferred.

Public delivery: source, this protocol, anonymous aggregate outcomes and resource/retirement receipts. Never publish private image records, predictions, weights, labels or patient identifiers.

## Engineering recovery, before any score

Attempt1 generated96 pairs but failed in the final input-integrity audit: `TargetReader.after_check` reopened past registered images after current-image permission was cleared. No scorer started and no target labels were read; worker retired exit1 after101.499674 GPU-worker wall seconds. Keep its outputs and receipts unchanged. The fix grants one already-registered image at a time during the existing integrity checks and resets permission in finally; no broader data access or scientific configuration change. A two-file generated regression probe verifies both successful checks and restored denial, including the scorer directory. The first regression process passed assertions but failed temporary-directory cleanup under its guard; explicit unlink/rmdir fixed fixture cleanup, and the second process exited0. Both costs are retained.

A new retry directory/identity reruns the same matrix because attempt1 did not persist its complete in-memory support audit or successful completion receipt; old predictions cannot substitute for qualification. Cumulative GPU budget remains1800s: subtract the retired101.499674s from the retry worker cap. The scorer retains900s. All failed costs remain in the cumulative ledger; no score has been used to trigger this retry.
