# R3 五框架冻结筛选：实验完成报告

**R3_EXPERIMENT_COMPLETE**。85/85 条完整轨迹、165,835 条正式评分记录；两个机械 smoke 和 85 个正式进程全部退出码 0。独立 CPU 标量汇总已完成并置 valid=true；公开导出再次核对逐轨迹记录与物理计数，全部匹配。

结论：本冻结配置下，没有主框架提供清晰、可归因于新增机制的增益。按预先给定的选择规则保留 C，归档五个主框架及全部匹配控制，不晋级候选、不追加组合、种子或调参实验。这是开发证据的描述性决定，不是外部代码 review 通过结论。

## 完成时间、身份及执行历史

- 北京时间 2026-09-14 13:52:46 开始，2026-09-15 07:56:02 结束；含前后处理约 18 小时 3 分 16 秒。监督器墙钟 18.033 小时，总 active-worker 34.887 小时。
- 物理 GPU 6、7，最多 2 个 worker，RTX 3090。没有因中间效果跳过任何臂。
- 实际运行 implementation SHA：`185fd440b16920f367c1f5e3096ab495bd85c0ec`。
- science 文件 SHA256：`73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`。
- registration digest：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`。
- secondary stream digest：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。
- 生命周期：阶段 I 实现 → 环境修复 d601496a 获用户提供的外部 review → 第一次 GPU 尝试在首个 forward 前进程审计失败、停止 → 授权工程修复 185fd440 → 用户明确免除新增 review 并授权直接执行 → 双 smoke → 85 条正式轨迹 → COMPUTE_COMPLETE → 独立 CPU 标量验证 → R3_EXPERIMENT_COMPLETE → 本次公开交付。
- 原外部 review 对应 d601496a；用户的执行授权不等于修复 SHA 获得了新的外部 review。历史失败保留在 [首次尝试报告](../../review/r3/stage2_attempt1/REPORT.md)，未混入效果表。该失败尝试额外 2 次模型加载、0 forward/Adam/target 记录，active-worker 约 6.615 秒。

## 终点与完整 17 臂结果

主终点 remaining_dev：1,695 内容/轨迹；每图 OD/OC 平均，域内平均，四域等权，再四主序等权。Dice 单位 %，差值为百分点 pp。回访流单列，不做五流合并。Drishti_GS 仅 37 内容却占每序 25% 域权重；四序共享内容、固定同一 seed=20260907，不能作独立患者重复或显著性检验。

| 臂 | OD | OC | 主四序 macro | ΔC pp | 独立回访 macro | 回访 ΔC pp |
|---|---|---|---|---|---|---|
| C | 87.348 | 69.874 | 78.611 | +0.000 | 77.921 | +0.000 |
| RP | 87.294 | 70.266 | 78.780 | +0.169 | 77.869 | -0.052 |
| T_LR | 85.419 | 69.000 | 77.210 | -1.401 | 76.064 | -1.857 |
| T_ISO | 86.678 | 69.376 | 78.027 | -0.585 | 77.247 | -0.674 |
| T_DIAG | 85.539 | 68.158 | 76.849 | -1.763 | 75.678 | -2.243 |
| U_PCA | 87.295 | 69.658 | 78.477 | -0.135 | 77.881 | -0.040 |
| U_RAND | 87.317 | 69.737 | 78.527 | -0.084 | 77.910 | -0.011 |
| U_SCALE | 87.349 | 69.870 | 78.609 | -0.002 | 77.918 | -0.003 |
| S_JOINT | 85.816 | 69.265 | 77.540 | -1.071 | 77.404 | -0.518 |
| S_SHARED | 87.297 | 70.260 | 78.778 | +0.167 | 77.870 | -0.051 |
| S_NOPCA | 85.850 | 69.140 | 77.495 | -1.116 | 77.434 | -0.487 |
| M_TRANSPORT | 87.308 | 70.246 | 78.777 | +0.166 | 77.861 | -0.060 |
| M_IDPOST | 87.294 | 70.267 | 78.781 | +0.169 | 77.867 | -0.054 |
| M_SHUFFLE | 86.949 | 69.572 | 78.260 | -0.351 | 77.273 | -0.649 |
| G_PCA | 87.306 | 69.687 | 78.497 | -0.114 | 77.822 | -0.099 |
| G_ISO | 87.331 | 69.765 | 78.548 | -0.063 | 77.851 | -0.070 |
| G_ORDER | 87.349 | 69.879 | 78.614 | +0.003 | 77.922 | +0.001 |

## 五组候选与匹配控制

| 候选 | 对照 | 主 Δpp | 正向主序 /4 | 最差同序 Δpp | 回访 Δpp |
|---|---|---|---|---|---|
| T_LR | C | -1.401405 | 1 | -2.936 | -1.857 |
| T_LR | RP | -1.570165 | 0 | -2.801 | -1.805 |
| T_LR | T_ISO | -0.816876 | 1 | -2.053 | -1.183 |
| T_LR | T_DIAG | +0.361201 | 4 | +0.062 | +0.385 |
| U_PCA | C | -0.134523 | 1 | -0.288 | -0.040 |
| U_PCA | RP | -0.303284 | 1 | -0.951 | +0.013 |
| U_PCA | U_RAND | -0.050081 | 0 | -0.132 | -0.028 |
| U_PCA | U_SCALE | -0.132475 | 1 | -0.286 | -0.036 |
| S_JOINT | C | -1.070770 | 0 | -1.782 | -0.518 |
| S_JOINT | RP | -1.239530 | 0 | -1.968 | -0.465 |
| S_JOINT | S_SHARED | -1.237985 | 0 | -1.995 | -0.466 |
| S_JOINT | S_NOPCA | +0.045393 | 2 | -0.038 | -0.030 |
| M_TRANSPORT | C | +0.165832 | 2 | -0.128 | -0.060 |
| M_TRANSPORT | RP | -0.002929 | 2 | -0.026 | -0.007 |
| M_TRANSPORT | M_IDPOST | -0.003557 | 2 | -0.028 | -0.006 |
| M_TRANSPORT | M_SHUFFLE | +0.516513 | 4 | +0.249 | +0.589 |
| G_PCA | C | -0.114493 | 0 | -0.126 | -0.099 |
| G_PCA | RP | -0.283253 | 1 | -0.759 | -0.047 |
| G_PCA | G_ISO | -0.051351 | 0 | -0.064 | -0.029 |
| G_PCA | G_ORDER | -0.117339 | 0 | -0.128 | -0.100 |

预设 +0.5pp 且至少 3/4 主序正向是筛选参考，不是统计显著性或临床门槛；没有候选满足该增益参考线。最差同序 Δ 是 min(candidate_order − control_order)，不等于两个臂各自最差序之差。

- **T_LR：unsupported_configuration。** 比 C 低 1.401pp，比 RP 低 1.570pp；低于 T_ISO。虽高于 T_DIAG，不能支持低秩密度优于简单控制的统一收益。
- **U_PCA：prefer_simple_control（组内），不晋级。** 低于 C、RP、U_RAND 和 U_SCALE；单纯匹配步长控制 U_SCALE 接近 C，增加 VJP 没有换来收益。
- **S_JOINT：unsupported_configuration。** 比 C 低 1.071pp，明显低于 S_SHARED；比 S_NOPCA 仅高 0.045pp，回访流反而低于 S_NOPCA。实际有状态创建和切换，仍未带来性能提升。
- **M_TRANSPORT：prefer_simple_control（组内），不晋级。** 比 C 高 0.166pp，但比 RP 低 0.002929pp，比 M_IDPOST 低 0.003557pp。优于 M_SHUFFLE 说明打乱配对有害，不能证明正确迁移相对恒等 POST 有价值。M_IDPOST 的全表最高分比 RP 仅高 0.000628pp，不作实质优势主张。
- **G_PCA：prefer_simple_control（组内），不晋级。** 低于 C、RP、G_ISO、G_ORDER；图教师的机械变化不等于有效性。

## 各域与尾部风险

下表为候选相对 C 的四主序平均域内差值，REFUGE_Valid OC 单列在 OC 列。各候选相对 RP/两直接控制的逐序、逐域 OD/OC/宏指标及分母见 CSV；不据域标签拼接不同方法。

| 候选 | 域 | OD Δpp | OC Δpp | macro Δpp |
|---|---|---|---|---|
| T_LR | REFUGE | -0.167 | -4.069 | -2.118 |
| T_LR | ORIGA | +0.405 | +1.455 | +0.930 |
| T_LR | REFUGE_Valid | -8.539 | -8.967 | -8.753 |
| T_LR | Drishti_GS | +0.585 | +8.086 | +4.335 |
| U_PCA | REFUGE | -0.019 | +0.299 | +0.140 |
| U_PCA | ORIGA | -0.276 | -0.835 | -0.556 |
| U_PCA | REFUGE_Valid | +0.335 | +0.946 | +0.640 |
| U_PCA | Drishti_GS | -0.252 | -1.274 | -0.763 |
| S_JOINT | REFUGE | -2.000 | +0.660 | -0.670 |
| S_JOINT | ORIGA | -2.462 | -2.197 | -2.330 |
| S_JOINT | REFUGE_Valid | +1.608 | +1.028 | +1.318 |
| S_JOINT | Drishti_GS | -3.274 | -1.930 | -2.602 |
| M_TRANSPORT | REFUGE | +0.019 | -0.394 | -0.187 |
| M_TRANSPORT | ORIGA | -0.007 | +0.791 | +0.392 |
| M_TRANSPORT | REFUGE_Valid | -0.247 | -0.639 | -0.443 |
| M_TRANSPORT | Drishti_GS | +0.072 | +1.730 | +0.901 |
| G_PCA | REFUGE | +0.111 | -0.068 | +0.022 |
| G_PCA | ORIGA | +0.076 | -0.216 | -0.070 |
| G_PCA | REFUGE_Valid | -0.739 | -0.259 | -0.499 |
| G_PCA | Drishti_GS | +0.383 | -0.205 | +0.089 |

主四序合计 6,780 个内容-顺序配对（不是独立患者）：正/零/负比例按内容计数汇总；中位数、最差 10% 均值是各序对应统计量的平均，最差单内容取所有序最小值。这些是样本权重的风险诊断，不能替代上面的四域等权主均值。

| 候选−C | 正 % | 零 % | 负 % | 配对中位数均值 pp | 最差10%均值 pp | 最差内容 pp |
|---|---|---|---|---|---|---|
| T_LR | 30.63 | 0.75 | 68.61 | -2.828 | -12.984 | -22.062 |
| U_PCA | 58.05 | 0.77 | 41.18 | +0.169 | -1.463 | -9.801 |
| S_JOINT | 45.59 | 0.90 | 53.51 | -0.043 | -8.144 | -49.204 |
| M_TRANSPORT | 39.62 | 0.88 | 59.50 | -0.112 | -1.022 | -2.329 |
| G_PCA | 30.21 | 1.03 | 68.76 | -0.120 | -0.824 | -1.596 |

## ASSD（共同有效配对）

以下按四序共同有效内容配对汇总，距离单位为评估栅格像素，负值较好；undefined 不填 0。valid/缺失数计内容-顺序，不计独立内容。完整逐域、逐序的中位数、上侧最差 decile、条件 ASSD 和全部 20 组对照见 CSV/完整 JSON。通用 distribution 的 `worst_decile_mean` 总是下侧 decile；用于绝对 ASSD 时代表较小距离，不能误称最差距离。

| 候选−C | 通道 | 共同有效 | 非共同有效 | 候选未定义 | C 未定义 | 均值 Δpx |
|---|---|---|---|---|---|---|
| T_LR | OD | 6780 | 0 | 0 | 0 | +3.7222 |
| T_LR | OC | 6780 | 0 | 0 | 0 | +3.6240 |
| U_PCA | OD | 6780 | 0 | 0 | 0 | -0.0142 |
| U_PCA | OC | 6780 | 0 | 0 | 0 | -0.1599 |
| S_JOINT | OD | 6780 | 0 | 0 | 0 | +0.9388 |
| S_JOINT | OC | 6780 | 0 | 0 | 0 | +0.0040 |
| M_TRANSPORT | OD | 6780 | 0 | 0 | 0 | +0.0656 |
| M_TRANSPORT | OC | 6780 | 0 | 0 | 0 | +0.1083 |
| G_PCA | OD | 6780 | 0 | 0 | 0 | +0.2459 |
| G_PCA | OC | 6780 | 0 | 0 | 0 | +0.1282 |

## 实际资源开销

正式：1,385,210 次 network forward、165,835 次 loss backward/Adam、231,784 次 Jacobian VJP、32,851 次参数替换；VJP 低于 234,120 上限。两个 smoke 共 632 forward、76 backward/Adam、36 VJP、10 次参数替换。正式加 smoke 共 1,385,842 forward、165,911 backward/Adam、231,820 VJP。

下表包含五条流的平均单轨迹墙钟与主机端耗时；耗时受共享设备及不同时段负载影响，不能当作受控速度基准。peak allocated 为 PyTorch tensor 分配峰值，不是 nvidia-smi 总显存或 reserved；bank 仅记录 active bank 张量，context 为已记录上下文张量，不能当作完整模型/优化器/进程 RSS。C/RP bank=0 表示没有该 R3 计数字段，不表示没有状态。

| 臂 | 轨迹 min | host ms/图 | forward/图 | VJP 总数 | allocated MiB | active bank KiB | context KiB |
|---|---|---|---|---|---|---|---|
| C | 21.38 | 578.4 | 8 | 0 | 588.5 | 0.00 | 0.00 |
| RP | 21.87 | 593.4 | 8 | 0 | 637.0 | 0.00 | 0.00 |
| T_LR | 23.45 | 642.8 | 9 | 0 | 644.1 | 75.25 | 0.00 |
| T_ISO | 23.30 | 638.0 | 9 | 0 | 644.1 | 75.25 | 0.00 |
| T_DIAG | 23.34 | 638.2 | 9 | 0 | 644.1 | 75.25 | 0.00 |
| U_PCA | 25.50 | 705.8 | 8 | 77258 | 793.7 | 75.25 | 0.00 |
| U_RAND | 26.43 | 734.7 | 8 | 77262 | 793.7 | 75.25 | 0.00 |
| U_SCALE | 25.70 | 712.3 | 8 | 77264 | 793.7 | 75.25 | 0.00 |
| S_JOINT | 26.93 | 750.1 | 9 | 0 | 690.4 | 75.25 | 975.71 |
| S_SHARED | 23.15 | 633.4 | 9 | 0 | 690.4 | 75.25 | 302.00 |
| S_NOPCA | 26.75 | 744.3 | 9 | 0 | 642.1 | 75.25 | 975.71 |
| M_TRANSPORT | 23.13 | 633.1 | 8 | 0 | 603.7 | 75.25 | 0.00 |
| M_IDPOST | 24.60 | 677.8 | 8 | 0 | 603.7 | 75.25 | 0.00 |
| M_SHUFFLE | 24.75 | 682.2 | 8 | 0 | 603.7 | 75.25 | 0.00 |
| G_PCA | 26.52 | 737.5 | 8 | 0 | 557.3 | 75.25 | 0.00 |
| G_ISO | 26.00 | 722.0 | 8 | 0 | 557.3 | 75.25 | 0.00 |
| G_ORDER | 24.62 | 679.4 | 8 | 0 | 557.3 | 75.25 | 0.00 |

完整 pipeline 时间、逐轨迹 host/wall/物理调用/状态载入开销见 costs CSV。机制细分计时（密度、Jacobian、SVD/transport、graph）保留在 aggregate 的 mechanism。没有独立的全进程 host RSS 峰值记录，不补造。有效性未胜 C 的候选无法凭额外开销获得晋级；M 的微小 C 增益由简单控制覆盖，不能建立新的性能-成本优势。

## 回访、机制与限制

回访流仍为 1,951 个不同内容，回访的是环境，不重复旧图。全 17 臂都已运行该流，五个主候选均低于 C。S_JOINT 在 stream4 实际建立 3 个 slot、切换 289 次、overflow 3 次，参数装载/替换 290 次（含初始装载）；累计记录载入约 36.459 秒，上下文张量峰值 999,128 bytes。S_SHARED 同样发生路由切换却不替换参数，效果优于 S_JOINT；不能把“存在复用”写成“复用有效”。完整逐域 slot 访问次数与连续域块统计在 contexts.json；slot 标签只用于事后汇总。

全部已保存机制分布在 aggregate.json.gz，主/次流加权标量均值在 mechanism_means.csv。举例，主序 0：T_LR 教师 ready 约 99.18%，平均绝对 logit 修正约 1.310；U_PCA 实际改变更新、平均 cosine 约 0.99593；M_TRANSPORT 的平均 ||Q−I|| 约 0.000959，拟合残差从 2.797e−6 降至 2.134e−6；G_PCA ready 时 1,984 条有效边，图能量均值 3.246→3.136，教师包含违反归零。这些证明路径有被执行，不能代替效果或建立因果归因。

未保存可支持真实教师 q/q* 对 GT 纠错/误纠正、完整 feature covariance、Jacobian、图几何、ASSD 几何重建的材料；本次 CPU 汇总不声称重建它们，也没有增加 GPU 诊断轮。未访问源数据、源代理或源原型，未重训源模型。没有独立新患者验证，四序共享内容不能支撑患者层面的统计显著性。结论限于该 checkpoint、冻结配置和开发流，不外推为五类思想普遍无效。

实际正式 backend：Python 3.10.6、Torch 2.2.1+cu121、CUDA 12.1、cuDNN 8902；TF32 off，cuDNN benchmark off / deterministic on；正式 deterministic_algorithms=false、warn_only=false、CUBLAS_WORKSPACE_CONFIG=null。smoke 的同设备比较实际进入严格确定性上下文，完整 before/after 记录见 execution.json；固定 seed 不是正式轨迹位级确定性的证明。

## 复现与证据索引

- [实际实现](https://github.com/DLwbm123/DPA-CTTA/tree/185fd440b16920f367c1f5e3096ab495bd85c0ec)、[冻结科学配置](../../../configs/r3_science_v1.json)、[实验方案](../../review/r3/03_EXPERIMENT_PLAN.md)、[方法规范](../../review/r3/02_METHOD_SPEC.md)、[方法来源](../../review/r3/METHOD_PROVENANCE.md)、[状态生命周期](../../review/r3/STATE_LIFECYCLE.md)、[预设选择标准](08_RESULT_SELECTION.md)。
- [执行/退出/85 完成凭据及两个 smoke 真实 traces](execution.json)、[CPU 导出真实日志](export_cpu.log)、[导出脚本](export_scalars.py)。原独立 CPU 分析器位于上述实现的 src/dpa_ctta/r3/analyze.py；完成标志由其核验后原子发布。
- [完整 17 臂主/次表](primary_and_secondary.csv)、[四个 subset、五序域等权结果](all_subset_domain_equal.csv)、[候选对照及顺序风险](candidate_comparisons.csv)。
- [remaining_dev 全臂各域分布/ASSD](remaining_domain_arms.csv)、[各域配对](remaining_domain_pairs.csv)、[逐序内容配对](remaining_pooled_pairs.csv)。
- [逐臂资源](costs_by_arm.csv)、[85 轨迹资源](costs_by_trajectory.csv)、[回访与 slot](contexts.json)、[机制均值](mechanism_means.csv)。
- [完整去身份聚合结果 gzip](aggregate.json.gz)：全部四个 subset、五流、逐域/逐序/配对的完整分布及标量机制；标准 gzip JSON，可直接用 Python gzip.open/json.load 读取。
- [本报告与 CSV 重建脚本](build_tables.py)。在此目录执行 `python3 build_tables.py`，只使用公开导出，无第三方依赖；内置 17 臂/5 流、配对加权和 85 轨迹覆盖断言。实际运行检查见 [本地构表日志](tables_cpu.log)。

发布范围为代码、冻结配置、去身份聚合指标、过程证据和脚本；不发布原始 RGB/mask、逐内容记录/文件名、checkpoint、设备 UUID、服务器路径和私有授权材料。既有科学配置、main、历史结果均保留。本批已结束，后续执行授权为 false；不自行追加实验或产生新的外部 review 通过结论。
