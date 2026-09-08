# M2 实验最终报告

**实验已完成，科学结论为混合结果：未建立新 episode 覆盖带来的清楚增益，也未建立 O2 跨两任务一致优于 D2 的证据。**

执行提交：`fef00f5bb1ea9c557054215ebe00ca5ee932ab81`；本文件属于后续结果发布提交。分支：`experiment/m2-episode-coverage-v1`。

## 主要结果

目标域等权 Dice（%）；差值为百分点。Fundus 使用 OD/OC macro，两个任务不混合。

| 任务 | N | A | R | D1 | O1 | D2 | O2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus | 69.092892 | 76.606283 | 76.561663 | 76.810429 | 76.660054 | 76.843102 | 76.725005 |
| polyp | 78.461378 | 78.437953 | 78.438682 | 78.316914 | 78.496279 | 78.321466 | 78.494365 |

Q1，相同曝光次数和 600 步下，新组织是否改善 D/O？Fundus 的 D2−D1 为 +0.032674，O2−O1 为 +0.064951；Polyp 分别为 +0.004552、−0.001914。变化很小且域间方向不一致，未建立清楚改善。

Q2，O2 是否提供清楚增量？Fundus O2−D2 为 −0.118097；Polyp 为 +0.172898。相对旧方法差距的 interaction 分别为 +0.032277、−0.006466，仍没有跨任务一致优势。Polyp O2 相对 N 的均值仅 +0.032987，包含 CVC-ClinicDB −7.975960、ETIS +12.790274、Kvasir −4.715354 的显著方向抵消，不能把平均微增包装为稳定收益。这里“方向抵消”描述观测值，不表示统计显著性。

Q3，是否两个任务都支持？属于混合结果。O2−D2 在 Fundus 四域中 1 正 3 负，在 Polyp 三域均正；但相对 O1，Fundus 2 正 2 负、Polyp 1 正 2 负。源域上 N 仍高于所有适配臂：Fundus O2−N −8.209379、Polyp −0.515979。

配对分布也不支持仅看平均值：Fundus O2−D2 中位数 +0.012683，66 正/0 零/62 负，最差 13 对均值 −2.616852；Polyp 中位数 0，45 正/4 零/47 负，最差 10 对均值 −0.753372。Polyp 共同有效 ASSD cohort 为 96，均值差 −0.350089 px，中位数差 +0.001239 px。完整 OD/OC、所有域、尾部和 ASSD 未定义数量见机器可读结果。

## 完整性、预算和覆盖

- 四套训练均为 600/600；新评分 104 source + 448 target = 552，复用 M1 1,380 条，合并展示 1,932 条。
- 部署 CPU 23 项测试、修复 smoke、正式执行、独立 CPU 重算和报告生成退出码均为 0。历史完整 184 项回归仅复用，不声称本轮重跑。
- 修复后的正常运行总计 560 次在线更新、2,404 次外层更新、1,202 次可微内层；加入上次失败及用户授权修复诊断后，M2 历次合计 570/2,408/1,203。没有新评分重跑。
- 两个 32 状态历史库直接复用，重建更新数 0。Fundus 唯一三元组 160→600，Polyp 64→600；每 query 的 transform 数 1→4，搭配 state 数分别 4→12–15、1→8–10；每 query/state/transform 的边缘曝光次数与 M1 完全保持。
- 修复后 GPU 阶段合计 2,327.992 秒（38.80 分钟），含 20.962 秒 smoke。正式训练峰值 4,597,449,728 字节（4.28 GiB），包括 smoke 的峰值 7,854,393,856 字节（7.31 GiB）。独立重算时私有文件 373,134,411 字节（355.85 MiB），不含复用 M1 资产和执行代码目录。
- M1 正式计算环境差异为空。仅 smoke 配对比较启用严格确定性，随后恢复；无改 seed、容差、训练长度或方法目标。上次失败证据保留在 `c1af390`，修复过程见 [修复报告](../../docs/M2_NUMERICAL_REPAIR_REPORT.md)。

所有训练、评分和报告步骤已结束，本轮不自动扩展实验。源图像、mask、checkpoint、代理张量、历史状态、身份映射及原始私有 JSONL 均留在 NAS；公开内容仅为代码、配置、必要脚本、去标识统计和报告。

以下为独立 CPU 重算直接生成的完整表格。单 seed/顺序、目标域既有曝光、UNKNOWN 患者/视频关联、固定 off-policy history 和一步适配等限制保持，不能据此声称患者层面显著性、方法普遍优越性或 SOTA。

---

# M2 episode coverage experiment report

M2_EPISODE_COVERAGE_COMPARISON_COMPLETE

Execution commit: fef00f5bb1ea9c557054215ebe00ca5ee932ab81

All values below are Dice percent; paired differences are percentage points. Task means weight domains equally.

| Split / task / domain / channel | N | A | R | D1 | O1 | D2 | O2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source / fundus / RIM_ONE_r3 / OD | 97.746073 | 95.876437 | 95.872333 | 95.907991 | 95.908699 | 95.905989 | 95.931647 |
| source / fundus / RIM_ONE_r3 / OC | 87.139392 | 72.361253 | 72.436587 | 72.498931 | 72.550177 | 72.517298 | 72.535061 |
| source / fundus / RIM_ONE_r3 / macro | 92.442733 | 84.118845 | 84.154460 | 84.203461 | 84.229438 | 84.211643 | 84.233354 |
| source / polyp / BKAI / polyp | 97.221359 | 96.709932 | 96.705011 | 96.713062 | 96.717260 | 96.715436 | 96.705380 |
| target / fundus / REFUGE / OD | 86.475070 | 87.251064 | 87.253066 | 87.494084 | 87.517496 | 87.332046 | 87.361752 |
| target / fundus / REFUGE / OC | 78.818326 | 82.266697 | 82.352793 | 82.447887 | 82.395500 | 82.499298 | 82.303978 |
| target / fundus / REFUGE / macro | 82.646698 | 84.758880 | 84.802930 | 84.970985 | 84.956498 | 84.915672 | 84.832865 |
| target / fundus / ORIGA / OD | 70.726519 | 79.909956 | 79.771308 | 80.241515 | 79.815804 | 80.250674 | 80.027514 |
| target / fundus / ORIGA / OC | 42.825480 | 64.290580 | 64.233738 | 65.208862 | 64.508701 | 64.935677 | 64.952958 |
| target / fundus / ORIGA / macro | 56.776000 | 72.100268 | 72.002523 | 72.725189 | 72.162253 | 72.593176 | 72.490236 |
| target / fundus / REFUGE_Valid / OD | 73.355174 | 83.408515 | 82.851044 | 82.492141 | 82.440789 | 83.301494 | 82.222487 |
| target / fundus / REFUGE_Valid / OC | 51.319756 | 66.041214 | 64.777607 | 61.761451 | 61.395365 | 63.005789 | 62.765128 |
| target / fundus / REFUGE_Valid / macro | 62.337465 | 74.724864 | 73.814326 | 72.126796 | 71.918077 | 73.153641 | 72.493807 |
| target / fundus / Drishti_GS / OD | 83.478070 | 85.266597 | 85.588999 | 86.168763 | 86.178319 | 86.005141 | 85.814036 |
| target / fundus / Drishti_GS / OC | 65.744745 | 64.415638 | 65.664751 | 68.668726 | 69.028458 | 67.414701 | 68.352190 |
| target / fundus / Drishti_GS / macro | 74.611408 | 74.841118 | 75.626875 | 77.418744 | 77.603389 | 76.709921 | 77.083113 |
| target / polyp / CVC-ClinicDB / polyp | 78.481590 | 70.492085 | 70.281891 | 70.321645 | 70.373507 | 70.500488 | 70.505629 |
| target / polyp / ETIS-LaribPolypDB / polyp | 68.829560 | 81.532518 | 81.745763 | 81.576956 | 81.645775 | 81.174384 | 81.619834 |
| target / polyp / Kvasir-SEG / polyp | 88.072985 | 83.289255 | 83.288394 | 83.052141 | 83.469554 | 83.289527 | 83.357631 |

## Target paired comparisons

| Task / domain | O2-D2 | O2-O1 | D2-D1 | Interaction |
| --- | ---: | ---: | ---: | ---: |
| fundus / domain-equal mean | -0.118097 | +0.064951 | +0.032674 | +0.032277 |
| fundus / REFUGE | -0.082807 | -0.123633 | -0.055314 | -0.068319 |
| fundus / ORIGA | -0.102940 | +0.327983 | -0.132013 | +0.459996 |
| fundus / REFUGE_Valid | -0.659834 | +0.575730 | +1.026845 | -0.451115 |
| fundus / Drishti_GS | +0.373192 | -0.520275 | -0.708823 | +0.188548 |
| polyp / domain-equal mean | +0.172898 | -0.001914 | +0.004552 | -0.006466 |
| polyp / CVC-ClinicDB | +0.005141 | +0.132122 | +0.178843 | -0.046721 |
| polyp / ETIS-LaribPolypDB | +0.445450 | -0.025940 | -0.402572 | +0.376632 |
| polyp / Kvasir-SEG | +0.068104 | -0.111924 | +0.237386 | -0.349309 |

## Execution and interpretation

New-run counts (including corrected smoke): {"online": 560, "outer": 2404, "inner": 1202, "records": 552}.
Reused M1 records: 1380; combined display: 1932. History reused: True.
GPU-stage wall seconds: 2327.992; private file bytes at recompute: 373134411.
Each of Fundus D2/O2 and Polyp D2/O2 completed 600 episodes. Source and target scoring and independent CPU reconstruction completed.
The original failed smoke and authorized repair diagnostics are separate historical work, not included in the new-run counts above.
Smoke paired checks used strict deterministic CUDA execution, restored afterward. Formal training/scoring retained M1 settings. Original tolerance and fixed seed were unchanged.
Full paired medians, signs, adverse tails, OD/OC channels, common ASSD cohorts, source/target tables and training curves are in [public_aggregate.json](public_aggregate.json). Resource counts and backend scope are in [execution_audit.json](execution_audit.json).
Interpret task and domain directions jointly. Single seed/order, previously exposed target domains, UNKNOWN patient/video associations, fixed off-policy history and one-step adaptation limit inference. Completion does not establish method superiority or SOTA.
