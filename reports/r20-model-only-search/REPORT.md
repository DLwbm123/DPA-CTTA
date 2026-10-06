# R20 model-only search

## 结果摘要

实验计算和评分于北京时间 **2026-10-05 19:44** 完成；106/106 条正式轨迹、90,848 次到达均完成，32 条完整流均完成独立评分，236 个物理运行/评分收据全部 COMPLETE。原生 G 的两个完整顺序与历史逐图硬指标完全一致。预检中的 CPU 工程错误保留在 MECHANICAL_TESTS.json；没有正式运行失败或恢复重试。

**主候选 W06_LR15 稳定超过原生 GraTa，但没有超过普通一致性 C，因此预登记最终胜出标准未达到。** W06_LR15 在 GraTa 上采用连续像素加权（方差温度 0.1、边界增益 2）并将最终动态学习率乘 1.5；C 是固定基础学习率的普通一致性 Adam，无辅助熵、临时扰动或余弦步长。

下表为同一 678 个封存复核内容身份、3 个种子 × 2 个顺序，按 domain×OD/OC 等权汇总后对 6 条轨迹取平均：

| 方法 | 封存复核 macro Dice | 相对 G |
|---|---:|---:|
| 原生 G | 76.9179% | — |
| 冻结主候选 W06_LR15 | 77.5338% | +0.6159 pp |
| 普通一致性 C | **78.0791%** | +1.1612 pp |

主候选对 G 的六条轨迹全部为正；配对内容 bootstrap 区间为 [+0.5229,+0.7119] pp。但其相对 C 为 -0.5453 pp，区间 [-0.8022,-0.2986] pp，仅 1/6 条轨迹为正。图像加权均值上主候选相对 C 为 +0.3480 pp，这不改变预登记域/通道等权主指标的结论。

最差 seed 平均退化在 REFUGE_Valid、OD、顺序 0：主候选相对 G 为 -1.2886 pp；相对 C 的最差单元为 -5.6037 pp。完整负向单元和小域区间均保留在 ALL_NEGATIVE_CELLS.csv 与 CONTENT_BOOTSTRAP_CI.csv。封存 Drishti 仅 15 个内容身份，患者依赖 UNKNOWN，区间不能消除开发选择与历史曝光偏差。

消融仅使用原始种子和两个顺序：去除方差权重后，主候选高 +0.1959 pp，支持该固定配置下的小幅增量。把同一权重机制移到 C 宿主后，Dice 为 78.3373%，比主候选高 +0.7826 pp；与同种子 C（78.2053%）相比仅 +0.1320 pp，不能和三种子主结果直接混为同级证据，也不能事后替换冻结主候选。该迁移同时取消动态步长并使用固定基础学习率，不是孤立的辅助梯度因果消融。

本轮完整复筛的实测单位图 worker 时间约为 G 0.652 s、C 0.571 s、W06_LR15 0.922 s；主候选约为 G 的 1.41 倍、C 的 1.61 倍。该时间包含实际 I/O/并发环境，不能声称严格等计算比较。主候选没有输出融合。G_LR15、G_K2、G_H025 仅在压缩筛选中比较，缺少完整 G_LR15 配对对照，因此不能将主候选全部增益归于新权重机制。

**研究判断：当前更强且更便宜的已确认基线是 C。W 的方差加权保留小幅机制信号，但证据不足以称为超过最佳简单对照的新方法。若继续研究，应先核实 C 宿主上该小幅增量的多种子稳健性；这只是结论，不授权新增实验或 R21。**

执行耗时约 7 小时 43 分；GPU-worker 19.174/48 小时，评分 CPU-worker 2.906 小时，私有存储实测 51.741 GiB。按 worker 未单独测量的存储字段为 NA，不能读成零；CPU 合成测试耗时另列，不与 GPU-worker 小时混算。实验完成时间与次日结果发布是两个独立时间点，原始 T0 未重置。

Status: **COMPLETE**. T0: 2026-10-05T04:01:05.993950+00:00. Final results are included in this release; remote publication is verified separately. Scientific results are complete only for sealed, fully scored trajectories.

GPU-worker time: 19.174 h / 48 h; CPU-worker time: 2.906 h; campaign elapsed: 7.716 h / 24 h. Measured disk: 51.741 GiB.

Outer selection uses SEARCH development labels in a separate CPU process. Online losses and memory are unlabeled. SEALED_REVIEW was historically exposed; this is a campaign-held review, not independent clinical or unseen test evidence. Orders and seeds reuse content; bootstrap keeps them coupled. Patient linkage and original ROI crop-center provenance remain UNKNOWN.

Frozen primary: **W06_LR15**; backup: A03_LR15; strongest full-SEARCH simple control: **C**. Seeds: [20260907, 17011, 29009]. The backup cannot replace the main conclusion after review.

Best full SEARCH configuration under the preregistered risk-first ranking (not the highest raw macro score): **W06_LR15**. The selected new-module primary is reported separately from a potentially stronger simple baseline.

SEARCH: W06_LR15 minus G: **+0.6396 pp**, paired content bootstrap 95% CI [+0.5691, +0.7144] pp; 6/6 positive order-seed trajectories; worst seed-averaged cell -1.1676 pp; worst single-seed cell -1.4805 pp.

SEALED_REVIEW: W06_LR15 minus G: **+0.6159 pp**, paired content bootstrap 95% CI [+0.5229, +0.7119] pp; 6/6 positive order-seed trajectories; worst seed-averaged cell -1.2886 pp; worst single-seed cell -1.5540 pp.

SEARCH: W06_LR15 minus C: **-0.6010 pp**, paired content bootstrap 95% CI [-0.7985, -0.4088] pp; 1/6 positive order-seed trajectories; worst seed-averaged cell -5.1379 pp; worst single-seed cell -5.1912 pp.

SEALED_REVIEW: W06_LR15 minus C: **-0.5453 pp**, paired content bootstrap 95% CI [-0.8022, -0.2986] pp; 1/6 positive order-seed trajectories; worst seed-averaged cell -5.6037 pp; worst single-seed cell -6.2538 pp.

Frozen base-seed mechanism/cost check: primary minus ABL_C_W06_LR15 = -0.7826 pp on review. A positive removal contrast supports an incremental effect only under this fixed configuration; it is not independent replication.

Frozen base-seed mechanism/cost check: primary minus ABL_KEY_W06_LR15 = +0.1959 pp on review. A positive removal contrast supports an incremental effect only under this fixed configuration; it is not independent replication.

Predeclared strong development signal: **False** (null means unqualified/incomplete, not a negative result).

Mechanism attribution: ABLATIONS.csv contains the frozen mechanism on C and key-component removal, or both component removals for a two-module primary. These are fixed-parameter transfer/ablation checks, not a fully tuned non-G route. COST.csv records every physical attempt, including profiling, failures and recovery. G_K2 and G_H025 remain registered screen controls; exact cost matching is not asserted.

Soft Dice/Brier use genuine saved float32 probabilities only; missing probabilities are NA. Every registered trajectory retains bit-packed masks and per-arrival scalars privately. Deployment requires the original checkpoint plus frozen algorithm configuration, fresh moments/memory and new-module initialization; adapted terminal weights are not initialization for another trajectory.

History: full-stream R19 G 77.229693%, G_HALF 76.403522%, G_VAL 76.641735%, G_MEM 75.725186%. These are historical references, not comparators for compressed SEARCH streams.

No automatic R21 or post-review retuning is authorized. Negative results and incomplete conditions remain in the ledger.
