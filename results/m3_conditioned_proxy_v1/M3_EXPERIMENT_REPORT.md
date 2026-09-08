# M3 conditioned proxy experiment report

M3_CONDITIONED_PROXY_COMPLETE

Execution commit: 00d1da5b3a333e53b20d869ac9fcbbdcb8b67a8f

Exploratory single-seed/order comparison. Completion is not evidence of superiority. FDA-style partial mixing is an existing method family, not a new claim of style-transfer novelty.

## Final interpretation / 最终结论

本轮结果为 **Fundus 混合、Polyp 退化，未见清楚的跨任务方法增量**。四套训练及独立 CPU 重算均已完成；完成状态不等同于方法有效性。

R3 是条件化真实代理；D3 是条件化梯度匹配（GM）代理；O3 是条件化适配目标蒸馏代理；O2T 复用 M2 的 O2，仅在测试时加入同一条件化变换（仍执行在线适配）。N/A/R/D2/O2 是复用的既有基线。下述差值均为 Dice 百分点，任务内各域等权，Fundus 使用 OD/OC macro；两个任务不合并求均值。

- **Fundus：均值领先很小且不稳定。** O3 为 76.866594%，相对 D3/R3/O2T 分别为 +0.257399/+0.206976/+0.323932 pp，但相对上一轮最优 D2 仅 +0.023492 pp。O3−D3 在 REFUGE、ORIGA、REFUGE_Valid、Drishti_GS 分别为 −0.046994、+0.347991、+2.514233、−1.785636 pp；两个域改善、两个域退化。配对中位数仅 +0.005365 pp，128 条配对中 64 正、64 负，最差 13 条差值均值 −3.582243 pp。因此不能将正均值称为稳定或显著增益。
- **Polyp：本配置的 O3 无增量。** O3 为 78.151074%，低于 D3/R3/O2T，差值分别为 −0.259395/−0.296057/−0.150874 pp；相对 O2 为 −0.343290 pp。O3−D3 在 CVC-ClinicDB、ETIS-LaribPolypDB、Kvasir-SEG 三域均负（−0.207684、−0.515620、−0.054880 pp）。96 条配对为 48 正、2 零、46 负，中位数 +0.002007 pp，但最差 10 条均值 −4.078302 pp、最差单条 −12.099852 pp，说明负向尾部值得关注。共同有效 ASSD cohort 为 96、双方均无缺失，O3−D3 的平均 ASSD 增加 0.914943 像素（更差），不把未定义值置零。
- **单加测试时变换没有解释为稳定收益的依据。** O2T−O2 在 Fundus/Polyp 分别为 −0.182343/−0.192416 pp；R3−R 仅 +0.097955/+0.008449 pp。Fundus 的 O3−O2T 为正提示条件化训练可能补偿仅部署变换的损失，但 Polyp 不支持这一收益，且本轮没有 D2T，不能单独归因 GM 的离线训练作用。
- **源域保持问题仍存在。** O3 的 source Dice 为 Fundus 84.203395%、Polyp 96.706962%，相对 N 分别 −8.239338/−0.514397 pp；相对 D3 也无改善（−0.006292/−0.005513 pp）。Fundus 相对 N 的 target 增益 +7.773702 pp 不能替代对既有适配基线的增量比较。

按预先约定，本轮在完成固定预算后停止当前设置，未启动 M4。证据仅覆盖本次固定 seed/order、样本与配置；不能推广为所有条件化或所有数据蒸馏无效，不能声称独立患者统计显著性或 SOTA。

## Completion and public scope

Execution commit: `00d1da5b3a333e53b20d869ac9fcbbdcb8b67a8f`. This report and its adjacent public JSON files are the result release; their Git commit is distinct from the frozen execution commit. The prior launch snapshot remains in commit `db5b6971e00a5a83929730e350c954cbe350858a`.

CPU validation (31 selected tests), GPU smoke, formal run, independent CPU recompute and detached launcher all exited 0. Fundus D3/O3 and Polyp D3/O3 each completed exactly 600 training episodes. New records: 208 source + 896 target = 1,104; reused M1/M2: 1,932; combined: 3,036. Formal updates: 1,104 online Adam / 2,400 outer / 1,200 differentiable inner; smoke adds 16/4/2, yielding 1,120/2,404/1,202. History rebuild steps: 0. No automatic retry or extra experimental arm was run.

GPU-stage time including smoke: 3,756.985 seconds (62.616 minutes) on GPU 7. Peak PyTorch allocated memory was 5,743,882,752 bytes (5.350 GiB, smoke); maximum formal training allocation was 4,726,506,496 bytes (4.402 GiB). These are process allocations, not total device occupancy. Private output at recompute was 374,120,436 bytes (356.79 MiB), below the 2 GiB limit; this measurement precedes writing the final aggregate/report. Formal environment comparison against M2 had no differences. CPU validation provenance is retained in the final audit; full historical suites are reused evidence, not rerun.

Public delivery includes source, fixed configuration, runner/tests, this interpreted report, deidentified aggregates and execution counters. Raw per-sample records, identity maps, source images/masks, synthetic proxies, checkpoints and history tensors remain private and are excluded from GitHub. No private-patient linkage or permission claim is inferred from anonymized aggregates.

## Absolute Dice (%)

| Split / task / domain / channel | N | A | R | D2 | O2 | R3 | O2T | D3 | O3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source / fundus / domain-equal mean | 92.442733 | 84.118845 | 84.154460 | 84.211643 | 84.233354 | 84.126283 | 84.242390 | 84.209687 | 84.203395 |
| source / fundus / RIM_ONE_r3 / OD | 97.746073 | 95.876437 | 95.872333 | 95.905989 | 95.931647 | 95.913035 | 95.932049 | 95.908604 | 95.910963 |
| source / fundus / RIM_ONE_r3 / OC | 87.139392 | 72.361253 | 72.436587 | 72.517298 | 72.535061 | 72.339530 | 72.552732 | 72.510771 | 72.495828 |
| source / fundus / RIM_ONE_r3 / macro | 92.442733 | 84.118845 | 84.154460 | 84.211643 | 84.233354 | 84.126283 | 84.242390 | 84.209687 | 84.203395 |
| source / polyp / domain-equal mean | 97.221359 | 96.709932 | 96.705011 | 96.715436 | 96.705380 | 96.709977 | 96.704362 | 96.712476 | 96.706962 |
| source / polyp / BKAI / polyp | 97.221359 | 96.709932 | 96.705011 | 96.715436 | 96.705380 | 96.709977 | 96.704362 | 96.712476 | 96.706962 |
| target / fundus / domain-equal mean | 69.092892 | 76.606283 | 76.561663 | 76.843102 | 76.725005 | 76.659619 | 76.542662 | 76.609196 | 76.866594 |
| target / fundus / REFUGE / OD | 86.475070 | 87.251064 | 87.253066 | 87.332046 | 87.361752 | 87.311370 | 87.362495 | 87.342929 | 87.180294 |
| target / fundus / REFUGE / OC | 78.818326 | 82.266697 | 82.352793 | 82.499298 | 82.303978 | 82.215941 | 82.343166 | 82.483517 | 82.552165 |
| target / fundus / REFUGE / macro | 82.646698 | 84.758880 | 84.802930 | 84.915672 | 84.832865 | 84.763655 | 84.852831 | 84.913223 | 84.866229 |
| target / fundus / ORIGA / OD | 70.726519 | 79.909956 | 79.771308 | 80.250674 | 80.027514 | 80.130018 | 79.787451 | 80.074923 | 80.491366 |
| target / fundus / ORIGA / OC | 42.825480 | 64.290580 | 64.233738 | 64.935677 | 64.952958 | 64.424616 | 64.887015 | 65.097963 | 65.377502 |
| target / fundus / ORIGA / macro | 56.776000 | 72.100268 | 72.002523 | 72.593176 | 72.490236 | 72.277317 | 72.337233 | 72.586443 | 72.934434 |
| target / fundus / REFUGE_Valid / OD | 73.355174 | 83.408515 | 82.851044 | 83.301494 | 82.222487 | 83.434493 | 82.062345 | 82.053389 | 83.350553 |
| target / fundus / REFUGE_Valid / OC | 51.319756 | 66.041214 | 64.777607 | 63.005789 | 62.765128 | 65.674061 | 60.706524 | 60.981126 | 64.712428 |
| target / fundus / REFUGE_Valid / macro | 62.337465 | 74.724864 | 73.814326 | 73.153641 | 72.493807 | 74.554277 | 71.384435 | 71.517257 | 74.031490 |
| target / fundus / Drishti_GS / OD | 83.478070 | 85.266597 | 85.588999 | 86.005141 | 85.814036 | 85.473946 | 85.676785 | 85.931535 | 85.358849 |
| target / fundus / Drishti_GS / OC | 65.744745 | 64.415638 | 65.664751 | 67.414701 | 68.352190 | 64.612506 | 69.515518 | 68.908185 | 65.909599 |
| target / fundus / Drishti_GS / macro | 74.611408 | 74.841118 | 75.626875 | 76.709921 | 77.083113 | 75.043226 | 77.596151 | 77.419860 | 75.634224 |
| target / polyp / domain-equal mean | 78.461378 | 78.437953 | 78.438682 | 78.321466 | 78.494365 | 78.447131 | 78.301948 | 78.410469 | 78.151074 |
| target / polyp / CVC-ClinicDB / polyp | 78.481590 | 70.492085 | 70.281891 | 70.500488 | 70.505629 | 70.330274 | 70.512800 | 70.414615 | 70.206930 |
| target / polyp / ETIS-LaribPolypDB / polyp | 68.829560 | 81.532518 | 81.745763 | 81.174384 | 81.619834 | 81.722013 | 81.300635 | 81.643050 | 81.127430 |
| target / polyp / Kvasir-SEG / polyp | 88.072985 | 83.289255 | 83.288394 | 83.289527 | 83.357631 | 83.289107 | 83.092411 | 83.173743 | 83.118863 |

## Target paired differences (percentage points)

| Task / domain | O3-D3 | O3-R3 | O3-O2T | O2T-O2 | R3-R | O3-A | O3-N | D3-D2 | O3-O2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / domain-equal mean | +0.257399 | +0.206976 | +0.323932 | -0.182343 | +0.097955 | +0.260312 | +7.773702 | -0.233907 | +0.141589 |
| fundus / REFUGE | -0.046994 | +0.102574 | +0.013399 | +0.019965 | -0.039274 | +0.107349 | +2.219531 | -0.002449 | +0.033364 |
| fundus / ORIGA | +0.347991 | +0.657117 | +0.597201 | -0.153003 | +0.274794 | +0.834166 | +16.158434 | -0.006733 | +0.444198 |
| fundus / REFUGE_Valid | +2.514233 | -0.522787 | +2.647056 | -1.109373 | +0.739951 | -0.693374 | +11.694026 | -1.636384 | +1.537683 |
| fundus / Drishti_GS | -1.785636 | +0.590998 | -1.961927 | +0.513038 | -0.583649 | +0.793106 | +1.022816 | +0.709939 | -1.448889 |
| polyp / domain-equal mean | -0.259395 | -0.296057 | -0.150874 | -0.192416 | +0.008449 | -0.286878 | -0.310304 | +0.089003 | -0.343290 |
| polyp / CVC-ClinicDB | -0.207684 | -0.123344 | -0.305870 | +0.007171 | +0.048384 | -0.285155 | -8.274659 | -0.085874 | -0.298699 |
| polyp / ETIS-LaribPolypDB | -0.515620 | -0.594583 | -0.173205 | -0.319200 | -0.023749 | -0.405088 | +12.297870 | +0.468666 | -0.492404 |
| polyp / Kvasir-SEG | -0.054880 | -0.170244 | +0.026452 | -0.265220 | +0.000712 | -0.170392 | -4.954122 | -0.115784 | -0.238768 |

## Historical appendix (Dice %)

| Split / task / domain | D1 | O1 |
| --- | ---: | ---: |
| source / fundus / RIM_ONE_r3 | 84.203461 | 84.229438 |
| source / polyp / BKAI | 96.713062 | 96.717260 |
| target / fundus / REFUGE | 84.970985 | 84.956498 |
| target / fundus / ORIGA | 72.725189 | 72.162253 |
| target / fundus / REFUGE_Valid | 72.126796 | 71.918077 |
| target / fundus / Drishti_GS | 77.418744 | 77.603389 |
| target / polyp / CVC-ClinicDB | 70.321645 | 70.373507 |
| target / polyp / ETIS-LaribPolypDB | 81.576956 | 81.645775 |
| target / polyp / Kvasir-SEG | 83.052141 | 83.469554 |

## Training and online cost

| Task / method | Episodes | Training seconds | Peak bytes |
| --- | ---: | ---: | ---: |
| fundus / D3 | 600 | 1105.072 | 4723982848 |
| fundus / O3 | 600 | 1169.221 | 4726506496 |
| polyp / D3 | 600 | 389.200 | 2631858176 |
| polyp / O3 | 600 | 443.339 | 2755015168 |

| New scoring stream | Records | Host step seconds including T | Pipeline seconds | Peak bytes |
| --- | ---: | ---: | ---: | ---: |
| source_fundus_R3 | 20 | 7.965 | 12.800 | 4194536960 |
| source_fundus_D3 | 20 | 9.065 | 13.419 | 4194536960 |
| source_fundus_O3 | 20 | 9.037 | 12.750 | 4194536960 |
| source_fundus_O2T | 20 | 9.104 | 13.212 | 4194536960 |
| target_fundus_R3 | 128 | 56.407 | 81.861 | 4200828416 |
| target_fundus_D3 | 128 | 56.757 | 80.469 | 4200828416 |
| target_fundus_O3 | 128 | 51.585 | 77.513 | 4200828416 |
| target_fundus_O2T | 128 | 53.829 | 80.743 | 4200828416 |
| source_polyp_R3 | 32 | 10.427 | 12.768 | 2259566080 |
| source_polyp_D3 | 32 | 10.329 | 12.081 | 2259566080 |
| source_polyp_O3 | 32 | 10.325 | 12.091 | 2259566080 |
| source_polyp_O2T | 32 | 9.668 | 11.362 | 2259566080 |
| target_polyp_R3 | 96 | 29.043 | 35.995 | 2261477888 |
| target_polyp_D3 | 96 | 29.682 | 35.760 | 2261477888 |
| target_polyp_O3 | 96 | 29.904 | 35.996 | 2261477888 |
| target_polyp_O2T | 96 | 28.678 | 34.633 | 2261477888 |

## Evidence and limits

Counts including smoke: {"online": 1120, "outer": 2404, "inner": 1202, "records": 1104}.
Reused old records: 1932; combined display: 3036. GPU-stage seconds: 3756.985. Private bytes at recompute: 374120436.
The [public aggregate](public_aggregate.json) preserves all paired medians/signs/worst-decile Dice differences, OD/OC, shared ASSD cohorts/missing counts/adverse tails, D1/O1 historical arms, 100-episode losses, zero-gradient fractions, image changes, clipping and conditioning L2. [Execution audit](execution_audit.json) contains forward/backward/transform/retrieval/push and three distinct update ledgers.
D/O losses are different objectives. Fixed 600 steps do not prove convergence. Historical timing is shared-GPU observational evidence, not a controlled hardware benchmark. Source/target labels never enter online conditioning; target labels are accessed after prediction.
O2T controls test-only conditioning; no D2T was run. D3-D2 and O3-O2 mix training and deployment changes. Phase retention does not prove semantic retention or privacy. UNKNOWN patient/video linkage, previous target exposure, fixed off-policy Base history and one-step truncation remain limitations.
No automatic NET_GAIN based solely on a positive mean. Assess domain cancellations and O3-D3/R3/O2T jointly; no automatic next round.

References: [Yang and Soatto, FDA (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html); [Kang et al., ICML 2023](https://proceedings.mlr.press/v202/kang23a.html).
