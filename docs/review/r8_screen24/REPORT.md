# R8 Screen24 — completed reduced screening report

## 结论

**运行完整完成；本轮不支持 A/B 的持续历史收益。** A FULL 的主 Dice 为 **74.009%**，B FULL 为 **73.249%**；分别比各自 STATIC 低 **0.206 / 0.261 个百分点**，两个源种子方向一致。二者也均低于 CURRENT-MLP、C0、VPTTA 和 C/G。固定范围内 C_CTTA_FIXED_LR 的主 Dice 最高，为 **78.534%**。所有负结果保留，未进行目标标签驱动的选模或追加调参。

这是用户授权的 **10-source / 40-target Screen24 缩减筛查**，不是原 65-source / 724-target 完整 R8，也不是独立临床确认。4000 步只是原 16000 步学习率计划的前缀，不能据此宣称收敛或否定全部 A/B 设计。

## 完成与覆盖

- Run: `screen24-20260925T120130Z`; queue `EXECUTION_COMPLETE`, 96/96 worker tasks COMPLETE; no outstanding global stop.
- Source: 10/10 × 4000 fit steps; prescribed calibration/validation complete; all source-selected checkpoints are step 4000.
- Target: 40/40 online trajectories and 40/40 independent scoring tasks complete; 78,040 arrivals = 40 × 1,951; **67,800 principal records = 40 × 1,695**.
- Each trajectory includes all arrivals; only `remaining_dev` enters the main table. Each principal record has OD and OC metrics (135,600 channel observations, not independent patients).
- Last completion receipt: **2026-09-26T12:30:32.827850+00:00**, i.e. 2026-09-26 20:30:32 Asia/Shanghai; checked at 12:43 UTC with no remaining experiment processes.
- Original start: 2026-09-25 12:02:08 UTC; resumed 2026-09-26 02:49:59 UTC. Elapsed wall time was about **24 h 28 min**, including the documented stop/wait. The original 24-hour target was not met; user explicitly made timing estimates soft before continuation.
- Only physical GPUs 5/6/7 were used. Completed work was not rerun after recovery.

## Metric and protocol

Primary Dice = mean within domain/channel, then equal weight across four domains, OD/OC, orders 0/1 and available seeds. Domain principal counts per trajectory: Drishti_GS 37, REFUGE_Valid 736, ORIGA 586, REFUGE 336. Pooling all images instead would change the weighting and can reverse a comparison; pooled paired distributions below are descriptive only.

A32 / B64, amplitude 0.3, FULL/STATIC and CURRENT-MLP; two source seeds 20260924/20260925; orders 0/1. Native C/G/VPTTA use frozen baseline seeds 20260907/20260908. Deterministic controls have one instance per order. Capacity latent Adam LR 0.03 and source-val soft Dice selection remain frozen. Source-val is a selection set, not independent testing. Target labels were read only by scoring after the online seal and all source selection/artifact locking.

## Primary table

Values are percent Dice. Seed SD is descriptive across two seeds, not a confidence interval.

| Method | Jobs | Dice % | OD % | OC % | Seed SD (pp) |
|---|---:|---:|---:|---:|---:|
| C_CTTA_FIXED_LR | 4 | 78.534 | 86.869 | 70.199 | 0.030 |
| G_CTTA_RELEASE_TRANSFER | 4 | 77.261 | 85.735 | 68.787 | 0.044 |
| R7_C_STATIC | 2 | 75.080 | 83.271 | 66.889 | — |
| C0_CURRENT_STATS | 2 | 75.079 | 83.270 | 66.889 | — |
| R7_C_FULL | 2 | 75.079 | 83.273 | 66.886 | — |
| VPTTA_NATIVE | 4 | 74.401 | 82.804 | 65.999 | 0.000 |
| CURRENT_MLP | 4 | 74.366 | 83.628 | 65.104 | 0.138 |
| A_STATIC | 4 | 74.214 | 82.964 | 65.465 | 0.213 |
| A_FULL | 4 | 74.009 | 83.024 | 64.993 | 0.136 |
| B_STATIC | 4 | 73.510 | 83.020 | 64.000 | 0.279 |
| B_FULL | 4 | 73.249 | 82.937 | 63.561 | 0.315 |
| N_SOURCE_EVAL | 2 | 68.201 | 78.726 | 57.675 | — |

## Paired comparisons

Positive Dice delta favors FULL. Source seed pairs are exact for STATIC/MLP; baseline seeds pair by frozen ordinal (20260924→20260907, 20260925→20260908), which is a descriptive pairing, not shared random seeds. Deterministic C0 is reused for both source seeds.

| FULL − control | Macro Δ (pp) | Seed 1 Δ (pp) | Seed 2 Δ (pp) | Worst domain/order Δ (pp) |
|---|---:|---:|---:|---:|
| A_FULL − A_STATIC | -0.206 | -0.151 | -0.260 | -0.677 |
| A_FULL − CURRENT_MLP | -0.357 | -0.355 | -0.359 | -0.626 |
| A_FULL − C0_CURRENT_STATS | -1.071 | -1.167 | -0.975 | -4.141 |
| A_FULL − VPTTA_NATIVE | -0.393 | -0.488 | -0.297 | -5.980 |
| A_FULL − C_CTTA_FIXED_LR | -4.526 | -4.643 | -4.408 | -15.547 |
| A_FULL − G_CTTA_RELEASE_TRANSFER | -3.252 | -3.317 | -3.187 | -9.088 |
| B_FULL − B_STATIC | -0.261 | -0.286 | -0.235 | -0.553 |
| B_FULL − CURRENT_MLP | -1.117 | -1.242 | -0.992 | -2.637 |
| B_FULL − C0_CURRENT_STATS | -1.830 | -2.053 | -1.608 | -6.322 |
| B_FULL − VPTTA_NATIVE | -1.152 | -1.375 | -0.929 | -8.146 |
| B_FULL − C_CTTA_FIXED_LR | -5.285 | -5.529 | -5.041 | -17.559 |
| B_FULL − G_CTTA_RELEASE_TRANSFER | -4.011 | -4.203 | -3.820 | -11.100 |

### Pooled paired distribution and ASSD

Distribution unit = matched content × channel × order × seed (13,560 observations per comparison; repeated contents are not new patients). q05 uses linear interpolation, worst-decile mean uses the lowest ceil(10% × n) values. ASSD uses common finite support only, in pixels of the 512 × 512 evaluation mask; positive delta is worse.

| Pair | Mean / median Δ (pp) | Negative / zero / positive | q05 / worst 10% Δ (pp) | ASSD Δ (px) | ASSD common / undefined |
|---|---:|---:|---:|---:|---:|
| A_FULL − A_STATIC | -0.068 / 0.023 | 6249 / 75 / 7236 | -0.926 / -1.077 | 0.004 | 13560 / 0 |
| A_FULL − CURRENT_MLP | -0.412 / -0.362 | 9520 / 76 / 3964 | -1.753 / -1.916 | 1.113 | 13560 / 0 |
| A_FULL − C0_CURRENT_STATS | 0.168 / -0.027 | 6906 / 56 / 6598 | -6.418 / -6.817 | -0.765 | 13560 / 0 |
| A_FULL − VPTTA_NATIVE | 0.858 / 0.075 | 6649 / 39 / 6872 | -12.186 / -13.882 | 0.448 | 13510 / 50 |
| A_FULL − C_CTTA_FIXED_LR | -2.714 / -0.532 | 7093 / 0 / 6467 | -27.192 / -32.900 | 3.751 | 13560 / 0 |
| A_FULL − G_CTTA_RELEASE_TRANSFER | -2.311 / -1.231 | 8252 / 6 / 5302 | -15.690 / -18.637 | 2.987 | 13560 / 0 |
| B_FULL − B_STATIC | -0.127 / -0.071 | 9032 / 72 / 4456 | -1.259 / -1.449 | 0.065 | 13560 / 0 |
| B_FULL − CURRENT_MLP | -0.361 / -0.576 | 9097 / 64 / 4399 | -6.066 / -6.399 | 0.466 | 13560 / 0 |
| B_FULL − C0_CURRENT_STATS | 0.219 / -0.363 | 7434 / 56 / 6070 | -11.958 / -12.636 | -1.412 | 13560 / 0 |
| B_FULL − VPTTA_NATIVE | 0.909 / 0.348 | 6290 / 39 / 7231 | -13.269 / -14.934 | -0.205 | 13510 / 50 |
| B_FULL − C_CTTA_FIXED_LR | -2.663 / -1.263 | 7384 / 0 / 6176 | -29.559 / -35.364 | 3.104 | 13560 / 0 |
| B_FULL − G_CTTA_RELEASE_TRANSFER | -2.260 / -1.707 | 8390 / 6 / 5164 | -18.892 / -21.816 | 2.341 | 13560 / 0 |

## Source capacity diagnostic

64 frozen source-val anchors per space, 128 total; mean query soft Dice %, distinct from target hard Dice.

| Space | Zero | Ambient oracle | Oracle projection | Direct latent |
|---|---:|---:|---:|---:|
| A32 | 84.772 | 84.856 | 84.836 | 84.874 |
| B64 | 84.772 | 84.856 | 84.979 | 85.029 |

Direct latent improves mean source query soft Dice over zero by 0.102 pp (A32) and 0.258 pp (B64), but this small source-only headroom does not establish target utility or calibration coverage.

## Cost, recovery and provenance

- Charged ledger, including prior profile/startup costs and full failed reservations: **46.277 GPU-hours**, **7.051 GiB**, 560,411 backward calls, 552,577 optimizer steps, 2,272,763 model forwards and 1,376 VJPs. All below retained aggregate caps (60 GPU-hours / 16 GiB / 4.5M backward calls).
- Completed attempts in the effective run sum to 33.824 ledger GPU-hours. Ledger time includes reserved/elapsed worker accounting, including scoring; it is not a GPU utilization or energy measurement. The charged 46.277 hours must not be presented as pure measured GPU compute.
- NAS lock failure: fixed with host-local locks; all three oracle jobs used their one equivalent-snapshot recovery. Original failed-start reservation of 10.36 GPU-hours remains charged. Scaler hit its old estimated deadline at anchor 493/512; its full 4314-second failed reservation and uncommitted tail were retained. User-authorized resource resume completed it; this was not relabeled infrastructure failure.
- Scientific artifact identity: `2329fb625ee023f461c23d1256162eb5514906c6`. Actual resumed runtime: `544640fe51507509a97864fe1a58c8eb0275d3d0`; these have separate inventory/config/physical-receipt bindings. Final reporting commits only aggregate results and do not alter executed scientific code.
- Estimated worker duration and 24-hour target became soft by explicit user directive. Aggregate physical limits and identity/data isolation remained enforced. Historical stop records and pre-override text remain history, not current waiting-for-approval instructions.

## Verification, public artifacts and limits

- Verified 96 final physical attempt receipts, GPU assignment, 10 source fit/worker receipts, 40 online/score/worker receipt chains and sealed scalar digests; checked exact visit/content uniqueness, per-job principal coverage, finite Dice, source selection completion and aggregate caps. No private identifiers are emitted by the summarizer.
- Reproducible aggregate generator: `scripts/r8/report_screen24.py ROOT RUN_ID SPEC_JSON` (requires authorized private outputs). `scripts/r8/test_report_screen24.py` tests unequal-domain weighting and paired summaries; all real-data coverage assertions pass.
- [RESULTS.json](RESULTS.json): full precision aggregates, paired distributions, source selection and costs. [MAIN.csv](MAIN.csv), [JOBS.csv](JOBS.csv), [DOMAINS.csv](DOMAINS.csv): method, seed/order and domain/channel tables.
- Public release contains source, frozen configs, tests, protocol/recovery summaries and aggregate tables. Images, masks, predictions, per-content scalars/IDs, private manifests, checkpoints, raw runtime logs and authentication material remain private. Full execution requires separately authorized data and the pinned base checkpoint; public aggregates alone cannot reproduce model inference.
- No RESET comparison, extra seeds/orders, full configuration grid, long-horizon stress, convergence or independent-patient claims. VPTTA gives identical seed aggregates here; two seed labels do not create independent evidence. No significance or superiority claim follows from this screen.
- Mechanism traces remain in private execution records; this closeout reports fixed utility, capacity and coverage, not an exhaustive causal mechanism analysis. Negative target utility does not authorize a new research round.
