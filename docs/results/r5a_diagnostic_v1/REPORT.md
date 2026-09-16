# R5-A 完成报告

**R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE**。3/3 完整轨迹、5853/5853 无标签和评价记录；CPU 标量核验有效，launcher exit=0，6 个 smoke/formal 子进程全部 exit=0，无失败前缀。实验完成与晋级失败分开：本次为有效的 NO_ADVANCE，不是机械失败。B 未启动、next_execution_authorized=false。

北京时间 2026-09-16 13:37:22 完成，启动至 CPU 分析结束 1022.32 秒（约 17 分钟）。监督器墙钟 1014.97 秒，累计 worker 活动 2794.69 秒。共享设备和流水线耗时仅作运行记录，不证明方法加速。

Implementation SHA：`b4b71601a5bdf87bb3a7e5d3db352610adcdff74`。Science SHA256：`89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5`。C fingerprint：`71c8be1e77e1fe45eb9d9c84ba2c46edbc3936720117399eb342de4a89ccb829`。Run ID：`2516876ea26641fe885e18aba7cb0220`。文档发布提交与执行提交分开；未改变规则或阈值。

## 结果

主指标为 remaining_dev=1695 内容，先 OD/OC 同图平均、域内平均、四域等权；order0/order1 等权作为 A 主结果，recurrence 单列。下面 Dice 为百分制、G 为百分点。每条全部 1951 内容都参与更新和无标签历史，前 32 次强制接受，eligible=1919。

| 流 | pre Dice | q Dice | trial=emit Dice | eligible 影子接受 | G（pp） | 随机影子 G（pp） | G−随机（pp） |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| order0 | 79.091874 | 79.269042 | 79.107259 | 1403/1919 (73.111%) | -0.009868 | -0.004045 | -0.005822 |
| order1 | 77.986155 | 78.099511 | 77.999061 | 1471/1919 (76.655%) | -0.005511 | -0.002658 | -0.002852 |
| recurrence | 77.904961 | 78.050658 | 77.922143 | 1424/1919 (74.205%) | -0.010612 | -0.004344 | -0.006268 |

两个主序平均 G=-0.007689 pp，未达到 +0.20 pp。三个流 G 均负，且两个主序均低于同接受率的解析随机影子参考。因此该冻结影子规则在 C 的既有状态路径上没有显示要求的即时收益；这里没有真实执行 VERIFY 回滚，不能把 G 当成 B 的长期效果或上界。

## 冻结 gate 逐项

| 条件 | 结果 |
| --- | --- |
| 三流机械完整、身份/计数/可实现性/decision replay 有效，原 C GPU smoke 保持性通过 | 通过 |
| 每流 eligible>0、至少一次影子拒绝、eligible 接受率≥0.50 | 通过（拒绝 516/448/495） |
| order0 和 order1 的 G 均≥0 | 未通过 |
| 两主序 G 均值≥0.20 pp | 未通过 |
| recurrence G≥0 | 未通过 |
| 两主序各自 G−G_random>0 | 未通过 |

不调整阈值、不追加 seed/轨迹、不自动进入 B。可观测性通过不等于方法有效。

## 成本与状态

每条实际 C 都提交 1951 次，真实拒绝/参数恢复均为 0；影子拒绝不改变参数、Adam、输出或增强 RNG。三条正式成本为 46824 forward、5853 backward/Adam、0 VJP。三设备 smoke 均按原配方完成，每设备 64/8/8/0，共 192/24/24/0；总计 47016 forward、5877 backward/Adam、0 VJP。没有失败或重试前缀、新 C0 或额外诊断模型运行。

| 轨迹 | 记录 | 主体秒数 | 实际提交 / 拒绝 |
| --- | ---: | ---: | --- |
| o0a0 | 1951 | 1001.48 | 1951 / 0 |
| o1a0 | 1951 | 873.80 | 1951 / 0 |
| o4a0 | 1951 | 879.71 | 1951 / 0 |

## 无标签标定与边界

固定校准 numerator=4298、denominator=5757，p_accept=0.7465693937814835，仅从三条完整无标签 trace 派生。摘要和原执行 binding 见 LABEL_FREE_CALIBRATION.json。它是开发流离线标定，不是零预扫描对照；A 未晋级，因此该值不构成运行 B 的权限。

PUBLIC_AGGREGATE.json 保留 pre/q/trial/emit 的 OD/OC、逐域/逐序/固定窗口、L 的正零负/尾部与接受拒绝分母、e/r/可靠区域计数，以及原 evaluator 的共同有效 ASSD 和 undefined 数。描述性全内容均值与域等权主指标分开，不混减。ASSD 单位为评估栅格像素，undefined 未填 0。

CPU 核验是在 launcher 内原实现完成的独立标量重算：验证三流覆盖、身份、计数、metric 可实现性、规则/窗口、gate 和 calibration；它不重建概率、梯度或 ASSD 几何。closeout 读取已完成的权威结果与退出记录，没有重复模型推理。历史 R4 逐内容 parity 和匹配 C0 未另行验证，H_t=null。全部为已暴露开发内容，顺序不代表独立患者，不作显著性、泛化或临床安全声明。

## 交付索引

- [完整去身份聚合](PUBLIC_AGGREGATE.json)
- [运行/CPU 完成证据、每设备 smoke 与逐轨迹成本](EXECUTION_AUDIT.json)
- [冻结无标签标定](LABEL_FREE_CALIBRATION.json)
- [本次授权与 caps](AUTHORIZATION.json)
- [交付清单](DELIVERY.json)

公开件不含逐内容行/身份、RGB、mask、权重、optimizer、私有路径、PID 或设备 UUID。完整无标签/评价账本保留在私有运行目录。保留历史实现与审阅材料，不改 main、旧科学配置或旧结果。
