# 分层可靠性 CTTA V2：完整开发实验结果

2026-10-19；冻结源码 `6e5947155b2757db3df917ca470337a7ed732cbd`。结论：**工程执行完整，科学推进门槛未通过**。主比较只有 +0.2449 pp，95%描述性区间 [−0.0817, +0.5865]，低于预设 +0.5 pp 且区间跨零。不能宣称可靠性分层优于精确配额随机，更不能据此宣称 TMI 创新。

## 路线与解释

本轮承接已下载全文的 ICLR 2027 投稿 Reliability-Gated TTA（[论文](https://openreview.net/forum?id=9Sqw7jJ1wn)），是源 checkpoint 和当前无标签图像驱动的真实参数适应；跨图像保留仿射参数和 Adam 状态，具有 CTTA 状态语义。论文启发的 RG 与我们的分层 STRAT 分开标记，本轮不声称原文复现或录用。分层平衡、门控、源 KL 都是已有思想，其组合本身不足以构成 TMI 创新。

相比此前 learned loss 和源监督输出校正，这条路线无需新增源训练，能直接检验首次到达收益、长期状态伤害和源返回保护。实际结果显示“避免累积崩溃”有价值，但“持续积累带来增益”尚不成立。

## 覆盖与主要结果

40 轨迹、1600 arrival、3600 PRE/POST/静态读出、128/128 旧参考再现，GPU 与 CPU 均 exit 0 并退役。10 learner × 两种状态策略 × 两个顺序，强控制完整保留；RDC、ENGRAM 关闭。SOURCE_EVAL 使用源 BN 统计，CW/旧 D/R 使用 current-image BN 语义，不能把这一区别带来的优势归因给新模块。

| 方法 | 状态 | 时点 | Dice % | ASSD px | HD95 px | vCDR MAE |
|---|---|---|---:|---:|---:|---:|
| ENT | RESET | POST | 90.9718 | 3.626 | 7.905 | 0.0627 |
| ENT | RETAIN | POST | 78.6497 | 9.427 | 16.842 | 0.0975 |
| RG | RESET | POST | 92.0392 | 3.350 | 7.703 | 0.0536 |
| RG | RETAIN | POST | 92.1049 | 3.311 | 7.648 | 0.0572 |
| STRAT | RESET | POST | 92.6026 | 3.231 | 7.619 | 0.0672 |
| STRAT | RETAIN | POST | 91.9442 | 3.527 | 8.312 | 0.0675 |
| RANDOM | RESET | POST | 92.4796 | 3.300 | 7.890 | 0.0718 |
| RANDOM | RETAIN | POST | 91.6992 | 3.668 | 8.630 | 0.0766 |
| ALL | RESET | POST | 92.4456 | 3.326 | 7.991 | 0.0730 |
| ALL | RETAIN | POST | 73.3605 | 19.127 | 35.157 | 0.1543 |
| GLOBAL | RESET | POST | 91.9807 | 3.319 | 7.680 | 0.0578 |
| GLOBAL | RETAIN | POST | 91.7671 | 3.476 | 8.106 | 0.0575 |
| NOANCHOR | RESET | POST | 92.4911 | 3.310 | 7.897 | 0.0672 |
| NOANCHOR | RETAIN | POST | 73.1276 | 25.080 | 49.848 | 0.1387 |
| ENT_NORM | RESET | POST | 91.2958 | 3.485 | 7.695 | 0.0598 |
| ENT_NORM | RETAIN | POST | 58.2169 | 13.891 | 21.752 | 0.1620 |
| CW1 | RESET | POST | 85.1714 | 8.604 | 19.290 | 0.1550 |
| CW1 | RETAIN | POST | 84.8890 | 8.716 | 19.572 | 0.1547 |
| CW2 | RESET | POST | 84.9093 | 9.018 | 19.637 | 0.1575 |
| CW2 | RETAIN | POST | 84.0939 | 9.285 | 21.591 | 0.1519 |
| SOURCE_EVAL | REFERENCE | STATIC | 92.1572 | 3.290 | 7.583 | 0.0569 |
| CURRENT_BN | REFERENCE | STATIC | 85.3987 | 8.028 | 18.310 | 0.1557 |
| D_OLD | REFERENCE | STATIC | 85.5856 | 7.803 | 17.661 | 0.1530 |
| R_OLD | REFERENCE | STATIC | 85.8067 | 7.476 | 17.066 | 0.1443 |
| R_SHUFFLE_OLD | REFERENCE | STATIC | 86.3431 | 7.200 | 17.011 | 0.1412 |

- STRAT−RANDOM / RETAIN_POST：+0.2449 pp，区间 [−0.0817,+0.5865]；160 个通道条件单元中 74 个下降，最差 −11.9740 pp。
- STRAT−RG / RETAIN_POST：−0.1608 pp，区间 [−0.7281,+0.4156]；STRAT−GLOBAL +0.1771 pp，区间 [−0.4164,+0.8784]。没有可靠证据优于更简单门控。
- SOURCE_EVAL 达 92.1572%，STRAT RETAIN_POST 为 91.9442%，相差 −0.2130 pp，区间 [−0.8535,+0.4339]。相对旧 R 的 +6.1374 pp 不能替代相对同 BN 语义源模型的比较。
- RESET_POST 92.6026%，比源模型高 0.4454 pp，但 RETAIN_POST 比 RESET_POST 低 0.6584 pp。全流 RETAIN_PRE−RESET_PRE −0.9080 pp，区间 [−1.8701,+0.0198]；首块首次暴露 −0.2600 pp，区间 [−0.9695,+0.3957]。因此没有首次到达正迁移证据。
- 返回 C0 的 STRAT POST 相对首次 C0 POST −0.8488 pp，区间 [−1.7551,+0.0822]；RANDOM 为 +0.1345 pp。源锚相对 NOANCHOR 的返回 PRE 保护 +28.4913 pp，区间 [23.2476,33.4385]，但这是防崩溃，不能解读为优于源模型或产生持续收益。
- ENT_NORM RETAIN_POST 58.2169%、NOANCHOR 73.1276%、ALL 73.3605%，显示强制每步相同参数范数并不能保证相同功能影响或安全性。源锚是重要稳定控制。

## 机制与结构诊断

全部 1920 个匹配更新满足实际 L2=.01±5e−7，没有零提案，源模型与 running statistics 不变。STRAT RETAIN 的平均 raw Adam 步幅 .0058733，实际 .010000008；RANDOM raw .0054444，实际 .010000004，故主结果不能归因于“X 几乎没更新”。参数范数匹配不等于函数变化匹配。

STRAT RESET_POST 对源预测的 logit RMS 平均 .1889，RETAIN_POST .3563、RETAIN_PRE .4065；相应平均源仿射距离从 .01873 增至 .18320。状态确实在积累，当前图像更新还部分纠正了到达前损害，但现有结果不能分清损害来自参数累积还是 Adam 旧矩。

RG 在真实 3px 边界选择比例 OD 0.1848%、OC 0.0027%，全图排除错误富集约 2；边界内富集却仅 1.0015/1.0000。STRAT 将边界选择比例提升至 33.44%/31.04%，但边界排除错误富集仅 1.0727/1.0001，尤其 OC 缺乏有用排序。这解释了为何分层修正覆盖后仍未稳胜随机。所有门控诊断对应第二次更新前的实际预测，未错配 PRE/POST。

STRAT RETAIN_POST 相对源模型 ASSD 3.5273 vs 3.2899、HD95 8.3121 vs 7.5831、vCDR MAE .06750 vs .05688，结构未优于源模型。相对 CURRENT_BN 的前景 corrected/spoiled：OC 5.585%/7.412%；OD 边界 15.418%/15.555%，不能用全图纠错掩盖前景伤害。STRAT 无新增空预测或嵌套违规，surface 160/160、vCDR 80/80。冻结 worst-no-harm 门槛失败；结构方向门槛只相对 CURRENT_BN，不能误说所有结构指标超过源模型。

## 科学资格与后续依据

失败门槛：主效应、主区间、首次内容暴露 PRE、最差单元保护；完整布尔结果见 public/RESULTS.json，未改冻结标准。当前不推进为 TMI 方法，也不扩大样本、seed 或流长。

下一步有依据的最小因果拆解是：保留模型仿射状态但每个 arrival 清空 Adam 矩，与同时保留参数和矩、完全 RESET 对照。只改变跨图像优化器状态，所有损失、数据、步幅和视图固定，并加入同机制 RANDOM 控制。若连这一拆解也不能得到首次到达/源模型收益，应放弃把当前门控包装为持续学习贡献。这是待验证假设，当前结果没有证明旧矩是原因。

## 证据范围

只有 8 个已多次开发暴露的图像内容、一个 checkpoint、两条短合成流。患者关联和骨干预训练成员未知；用户授权的是内容隔离开发实验。只有首块是本流首次暴露，后续块重现同内容。bootstrap 按 8 个内容组绑定两顺序和重复条件，仅为描述性区间，不是独立患者/跨医院/临床泛化证据。评分标签在 worker 退出后由独立 CPU scorer 读取，不进入适应；后续设计利用本轮结果属于开发反馈，不能称盲测。

详见 [执行审计](EXECUTION_AUDIT.md)、[协议](PROTOCOL_ZH.md)、[完整聚合表](public/absolute.csv)、[条件与通道](public/condition_channel.csv)、[全部差值](public/contrasts.csv)、[前景/边界转移](public/transitions.csv)、[门控](public/gate_audit.csv)、[实际步幅](public/update_audit.csv)。
