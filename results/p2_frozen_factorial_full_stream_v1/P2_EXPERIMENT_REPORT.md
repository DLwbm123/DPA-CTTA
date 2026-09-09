# P2 frozen factorial full-stream comparison

P2_FROZEN_FULL_STREAM_COMPLETE

Execution commit: d3ee6901379be293f47abd1687808caaf5b04266

## 完成后的结论（主要子集 remaining_dev）

完整覆盖已通过，方法收益有限。固定平均在 Polyp 两序保留小幅域等权收益（EA−A 为 +0.260689 / +0.150959 pp），Fundus 两序仍退化（−0.832674 / −0.729359 pp）。这支持有限的任务特异经验，不能称为统一 CTTA 改进或稳定安全收益。

在匹配的融合输出下，Polyp 的 EO2、ED4 两序都低于 EA；Fundus 两者相对 EA 的优势随域序翻转。Fundus/order1 的 raw O2 为 75.423504%，高于 EO2 的 74.940462%，并非融合进一步超过单条原始方法。本轮没有证明封存 DD 在额外 source 前向相同的条件下具有稳定必要性，也没有满足“稳定超过 EA 且超过 raw 方法”的互补性结论。按本轮范围停止，不继续新 DD 训练或自动开启 P3。

P1 的 Polyp extension_dev 平均收益约 1.3–1.8 pp，本轮 remaining_dev 只有 0.15–0.26 pp；这是不同内容与适配历史下的描述性对照，不能归因于某个模块。N 在本轮主要子集的四个任务/顺序组合中均不是均值最优。

负向尾部仍明显：ORIGA 的 EA−A 两序均值约 −4 pp，最差 10% 的配对差约 −24 pp；ETIS 两序均值为 −1.31 / −1.97 pp，最差 10% 约 −23 / −24 pp。因此不能把小幅总体改善称作逐域或逐图安全性提高。这里“最差 10%”是配对变化最负的一组，不是预先固定的难例集合，也不是患者独立显著性检验。


### remaining_dev：EA−A 的逐域差与负向尾部

| 任务/顺序 | 域 | 均值 pp | 最差 10% 均值 pp | 最差单图 pp |
|---|---|---:|---:|---:|
| fundus/order0 | REFUGE | -0.1320 | -5.6657 | -13.4451 |
| fundus/order0 | ORIGA | -4.0503 | -23.8711 | -40.9544 |
| fundus/order0 | REFUGE_Valid | -0.1424 | -10.6244 | -29.0711 |
| fundus/order0 | Drishti_GS | +0.9939 | -6.0258 | -6.9959 |
| fundus/order1 | Drishti_GS | +0.4858 | -7.1926 | -7.4247 |
| fundus/order1 | REFUGE_Valid | +0.5120 | -9.6099 | -32.7037 |
| fundus/order1 | ORIGA | -3.8712 | -23.9923 | -37.1711 |
| fundus/order1 | REFUGE | -0.0441 | -5.7321 | -25.7836 |
| polyp/order0 | CVC-ClinicDB | +0.8788 | -11.8089 | -43.6803 |
| polyp/order0 | ETIS-LaribPolypDB | -1.3104 | -22.9651 | -79.0036 |
| polyp/order0 | Kvasir-SEG | +1.2136 | -5.1317 | -78.6145 |
| polyp/order1 | Kvasir-SEG | +2.5744 | -4.5876 | -51.8464 |
| polyp/order1 | ETIS-LaribPolypDB | -1.9681 | -24.0839 | -74.1379 |
| polyp/order1 | CVC-ClinicDB | -0.1535 | -11.0999 | -54.4554 |

### 匹配交互差（域等权，pp）

| 任务/顺序 | (EO2−O2)−(EA−A) | (ED4−D4)−(EA−A) |
|---|---:|---:|
| fundus/order0 | +0.753464 | +1.153938 |
| fundus/order1 | +0.246317 | +0.462481 |
| polyp/order0 | +0.354434 | -0.346963 |
| polyp/order1 | +0.425337 | +0.911544 |

交互差为正只说明融合给该父方法带来的变化大于给 A 的变化，不代表最终 EO2/ED4 高于 EA。

### 同次 16 格计数的解释例子

以下仅为完整域/通道聚合的描述性例子；各域、通道及三融合输出的全部计数均保存在 public_aggregate.json。

| remaining_dev 条件 / EA 相对 A | 新增 FP | 修正 FP | 新增 FN | 修正 FN |
|---|---:|---:|---:|---:|
| fundus/order0/REFUGE/OD | 19113 | 60635 | 157325 | 94256 |
| polyp/order0/CVC-ClinicDB/polyp | 76542 | 34554 | 127925 | 450589 |
| polyp/order1/Kvasir-SEG/polyp | 263546 | 29185 | 32759 | 1583700 |

例如 REFUGE/order0/OD 的平均修正了更多 FP，同时新增更多 FN；Polyp 例子则修正更多 FN，同时新增更多 FP。平均在两条预测间交换不同类型错误，不具备识别真值或保证安全的能力。像素规模不同，不能将这些计数作为域等权 Dice 的替代。

### 完成与预算证据

GPU 活跃阶段记录 7434.393 秒（约 2.065 小时）；该计时含 smoke，并包含运行阶段的评分/等待，不是 CUDA kernel 纯计算时长。launcher 总时长 7435.393 秒。完成时间 2026-09-09T06:58:33.699464+00:00。

登记 G=3753，正式 records=52542、online=22518、teacher_forwards=7506，加 smoke 12 次后 online=22530。CPU、登记、smoke、run、recompute 退出码全部为 0。重算前私有文件合计 81674494 bytes，低于 1 GiB；正式共享调度的最大 allocated 峰值 6779464192 bytes，不能拆作独立方法峰值。

执行 checkout 保持 d3ee6901379be293f47abd1687808caaf5b04266。本报告及聚合证据另行发布，旧实验及私有逐图资产保持不变。完整的 ASSD 共同有效 cohort、未定义数量及分布在聚合 JSON 中；CPU 只复核标量与 Dice 计数，不宣称重新计算像素距离。

## remaining_dev

| Task/order | Groups | N | A | EA | O2 | EO2 | D4 | ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 1695 | 68.200582 | 74.259279 | 73.426605 | 72.891721 | 72.812511 | 72.686550 | 73.007813 |
| fundus/order1 | 1695 | 68.200582 | 74.542945 | 73.813586 | 75.423504 | 74.940462 | 74.305555 | 74.038677 |
| polyp/order0 | 1610 | 76.510373 | 79.428436 | 79.689126 | 78.734012 | 79.349135 | 79.686450 | 79.600176 |
| polyp/order1 | 1610 | 76.510373 | 79.105353 | 79.256312 | 78.098019 | 78.674314 | 77.616282 | 78.678785 |

| Task/order | EA-A | EA-N | EO2-O2 | ED4-D4 | O2-A | D4-A | EO2-EA | ED4-EA | EO2-ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | -0.832674 | +5.226023 | -0.079210 | +0.321264 | -1.367558 | -1.572729 | -0.614094 | -0.418791 | -0.195303 |
| fundus/order1 | -0.729359 | +5.613004 | -0.483042 | -0.266878 | +0.880559 | -0.237390 | +1.126876 | +0.225092 | +0.901785 |
| polyp/order0 | +0.260689 | +3.178752 | +0.615123 | -0.086274 | -0.694424 | +0.258014 | -0.339991 | -0.088949 | -0.251041 |
| polyp/order1 | +0.150959 | +2.745939 | +0.576295 | +1.062502 | -1.007334 | -1.489071 | -0.581998 | -0.577527 | -0.004470 |
## legacy_dev

| Task/order | Groups | N | A | EA | O2 | EO2 | D4 | ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 128 | 69.092892 | 74.538998 | 73.801632 | 73.653989 | 72.938530 | 73.009535 | 73.791331 |
| fundus/order1 | 128 | 69.092892 | 75.140702 | 74.439300 | 75.885483 | 75.138639 | 74.336631 | 74.269723 |
| polyp/order0 | 96 | 78.461378 | 79.908197 | 81.034178 | 80.346747 | 81.326665 | 80.412266 | 80.782358 |
| polyp/order1 | 96 | 78.461378 | 79.738532 | 80.913022 | 78.294175 | 80.106449 | 81.321255 | 81.237874 |

| Task/order | EA-A | EA-N | EO2-O2 | ED4-D4 | O2-A | D4-A | EO2-EA | ED4-EA | EO2-ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | -0.737367 | +4.708739 | -0.715459 | +0.781796 | -0.885009 | -1.529464 | -0.863102 | -0.010301 | -0.852801 |
| fundus/order1 | -0.701402 | +5.346407 | -0.746845 | -0.066908 | +0.744782 | -0.804071 | +0.699339 | -0.169577 | +0.868916 |
| polyp/order0 | +1.125980 | +2.572799 | +0.979919 | +0.370092 | +0.438549 | +0.504069 | +0.292488 | -0.251820 | +0.544307 |
| polyp/order1 | +1.174489 | +2.451643 | +1.812273 | -0.083381 | -1.444357 | +1.582723 | -0.806573 | +0.324852 | -1.131425 |
## p1_extension_dev

| Task/order | Groups | N | A | EA | O2 | EO2 | D4 | ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 128 | 66.452389 | 73.796285 | 72.745511 | 72.436336 | 71.821030 | 71.836462 | 72.425654 |
| fundus/order1 | 128 | 66.452389 | 74.506257 | 73.700675 | 76.426326 | 75.215730 | 74.433735 | 74.181078 |
| polyp/order0 | 96 | 78.172439 | 81.054795 | 81.595596 | 79.369137 | 81.085137 | 79.680475 | 81.000099 |
| polyp/order1 | 96 | 78.172439 | 80.304710 | 81.282401 | 75.633052 | 78.651337 | 80.582167 | 80.683100 |

| Task/order | EA-A | EA-N | EO2-O2 | ED4-D4 | O2-A | D4-A | EO2-EA | ED4-EA | EO2-ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | -1.050774 | +6.293121 | -0.615306 | +0.589192 | -1.359949 | -1.959823 | -0.924480 | -0.319857 | -0.604624 |
| fundus/order1 | -0.805582 | +7.248286 | -1.210596 | -0.252657 | +1.920069 | -0.072522 | +1.515055 | +0.480402 | +1.034652 |
| polyp/order0 | +0.540801 | +3.423157 | +1.716001 | +1.319624 | -1.685659 | -1.374320 | -0.510459 | -0.595497 | +0.085038 |
| polyp/order1 | +0.977691 | +3.109962 | +3.018286 | +0.100934 | -4.671658 | +0.277457 | -2.631064 | -0.599301 | -2.031763 |
## all_dev

| Task/order | Groups | N | A | EA | O2 | EO2 | D4 | ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 1951 | 67.964224 | 74.328553 | 73.338857 | 72.983086 | 72.714658 | 72.604889 | 72.946240 |
| fundus/order1 | 1951 | 67.964224 | 74.691960 | 73.806501 | 75.548687 | 74.907783 | 74.372899 | 74.002336 |
| polyp/order0 | 1802 | 76.951272 | 79.858266 | 80.068498 | 79.058924 | 79.760641 | 79.903984 | 79.855730 |
| polyp/order1 | 1802 | 76.951272 | 79.463426 | 79.705695 | 78.085222 | 78.932581 | 78.337377 | 79.197371 |

| Task/order | EA-A | EA-N | EO2-O2 | ED4-D4 | O2-A | D4-A | EO2-EA | ED4-EA | EO2-ED4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | -0.989696 | +5.374633 | -0.268428 | +0.341351 | -1.345467 | -1.723663 | -0.624199 | -0.392617 | -0.231582 |
| fundus/order1 | -0.885459 | +5.842277 | -0.640904 | -0.370562 | +0.856727 | -0.319061 | +1.101283 | +0.195835 | +0.905447 |
| polyp/order0 | +0.210232 | +3.117226 | +0.701717 | -0.048254 | -0.799342 | +0.045718 | -0.307857 | -0.212768 | -0.095089 |
| polyp/order1 | +0.242270 | +2.754423 | +0.847359 | +0.859994 | -1.378203 | -1.126048 | -0.773114 | -0.508324 | -0.264790 |

## Audit and interpretation boundary

{"records": 52542, "online": 22518, "teacher_forwards": 7506, "outer": 0, "inner": 0}; additional smoke online=12.
All paired distributions, domain/channel results, two matched interaction contrasts and foreground/background 16-cell counts are in public_aggregate.json. Cells reconstruct N/parent/view Dice and explain retained/repaired/new FP/FN; background counts are not presented as a replacement for foreground segmentation.
Frozen A/O2/D4 are independent full streams. EA/EO2/ED4 use their own parent plus the same source q0, probabilities averaged at 0.5 then thresholded >=0.5. No GT gating, SA, new proxy training or new source evaluation.
Independent CPU reconstruction verifies coverage, identities, parent/view binding, counters, every Dice and each 16-cell marginal; it does not reconstruct discarded pixel probabilities or independently recalculate ASSD. Undefined ASSD stays undefined; no macro ASSD.
Shared schedule executes one teacher per visit; standalone ensemble costs include a source forward and extra model. Shared allocated peak is not an isolated per-method peak. See execution_audit.json for measured costs and forwards.
remaining_dev is primary, not a globally untouched test. Known role exclusions preserved; UNKNOWN linkage disclosed. Single seed and two orders on the same contents do not establish patient-independent significance. P1 comparisons change both contents and histories.
Completion is separate from scientific value. Apply predeclared interpretation after examining matched controls and tails; no automatic P3 or parameter search. Source data, masks, model/proxy files, private paths, identities and per-asset digests are not published.
