# CTTA 自主实验窗口关闭：2026-10-02

本窗口已完成授权的4轮后继R12–R15。所有实际计算已结束，无活跃本实验父子worker；R15仅源端机制验证，没有新增目标访问。四轮上限用尽，停止新启动并暂停每小时自动化，未因剩余GPU或墙钟预算追加搜索。原+.5pp且两order均正的目标优先尺度没有达到；RL仍退出主线。

| 阶段与评价范围 | 结果 | GPU-worker 秒 |
|---|---|---:|
| R11，原短流1024/888，各order | half-C0 -0.679668pp；最差REFUGE_Valid/OD -4.154192pp，风险保留 | 853.928360 |
| R12，短流1024/888，各order | quarter-C0 +0.135522pp；flip-only-C0 -4.665606pp，最差域通道 -13.020236pp | 1640.771897 |
| R13，完整1951/1695，各order | quarter75.239183%，C075.079420%，GraTa77.229693%；quarter-C0 +0.159763pp，最差Drishti_GS/OC -0.084311pp | 773.143510 |
| R14，完整1951/1695，各order | U1075.190854%，U10-C0 +0.111434pp，U10-quarter -0.048329pp，最差Drishti_GS/OC -0.029691pp | 958.338602 |
| R15，源16episode×32，目标NOT_RUN_SOURCE_ONLY | DS源hard86.834606%精确保持quarter；soft85.429024%较quarter恢复+8.495772pp、较C0 +0.014460pp | 116.730575 |

源C0soft85.414565%，quartersoft76.933253%，U10soft85.447182%；R15在native/quarter硬决策相同的像素保留native概率，只对平均0.173182%的源像素保留quarter概率。它通过冻结源soft恢复机制尺度，但按构造与quarter硬mask相同，不能再提高目标hardDice，因此没有重跑目标或伪造目标soft/online seal。softDice恢复不证明概率校准或未见目标泛化。

R13quarter完整结果由旧短流和新不相交补集标量按冻结manifest合成，未声称新完整online seal；完整C0/GraTa历史轨迹按原seals复用。R14/G/C0/quarter全部域、OD/OC、soft缺失角色、资格、成本与负结果见各轮报告。R11四视图因目标访问前预算准入失败暂缓，不是性能失败。历史bitmask无C0/G soft概率，明确NA。短/完整流范围不同，不按跨阶段绝对Dice变化声称改善。

所有开发数据早已暴露于R7/R8等研究，确定性共享内容的两order不构成独立复制，患者依赖未知；没有独立确认、假多seed、盲测解封、新CTTA方法或临床效果声明。原比较门槛不放宽、坏域不删除，保留多重开发比较限制。本轮正结果仅支持“保护决策无关的概率扰动可以避免源soft损失”这个机制，目标hard改善仍很小，距GraTa约1.99pp。未见可部署反馈/动作/长时序收益证据，不能恢复RL。

保守计入全部R11GPU后，本campaign总GPU-worker 4342.912945秒（约1.206365小时）。R15全部attempt1184F/0BP/optimizer/VJP，原T0墙钟697.582701秒，原计费血缘累计38370.442624秒。阶段唯一键终态替换暂估，未重置旧账本或重复相加。原窗口2026-10-01 22:49:47至10-02 22:49:47北京时间、GPU24小时、最多4后继规则不变；停止原因是轮次上限，而非耗尽24小时GPU。

已封存公开报告：

- [R11_CURRENT_VIEW_V1](https://github.com/DLwbm123/DPA-CTTA/blob/19b6b08729073f87582051867072618eedcef01a/docs/results/r11_current_view/REPORT.md)，提交`19b6b08729073f87582051867072618eedcef01a`。
- [R12_HORIZONTAL_WEIGHT_V1](https://github.com/DLwbm123/DPA-CTTA/blob/ccb30a28e967f1728c4a39471e8e7466f2c59dea/docs/results/r12_horizontal_weight/REPORT.md)，提交`ccb30a28e967f1728c4a39471e8e7466f2c59dea`。
- [R13_FULL_COVERAGE_V1](https://github.com/DLwbm123/DPA-CTTA/blob/45df799edc2506c216d7d59c4459bfaecfdc64c8/docs/results/r13_full_coverage/REPORT.md)，提交`45df799edc2506c216d7d59c4459bfaecfdc64c8`。
- [R14_UNCERTAINTY_FLIP_V1](https://github.com/DLwbm123/DPA-CTTA/blob/df8cd0ddd4414eed1c8b33fc5ad032284149d7fe/docs/results/r14_uncertainty_flip/REPORT.md)，提交`df8cd0ddd4414eed1c8b33fc5ad032284149d7fe`。
- [R15 完整源端结果与资格](../results/r15_decision_support/REPORT.md)，与本关闭文件同批提交。

关闭不代表原目标性能达标。继续研究需要新的明确授权和有信息增益的冻结方案，不能自动增加第五轮或换名重跑既有负结果。
