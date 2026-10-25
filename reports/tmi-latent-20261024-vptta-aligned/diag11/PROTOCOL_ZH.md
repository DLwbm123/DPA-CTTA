# diag11 随机源光度风险诊断（运行前冻结）

run10源分割均值+.1944但p10仍负；diag09 oracle并非逐例改善。检验gamma参数精度与任务风险的一致性，不读目标数据/结果、不训练或选参、不改变既有gate。

固定run01/run06/run10权重，同源dev422图各4个随机独立已知RGB/gamma扰动，seed92511每源重置，logRGB~U[-.25,.25]、loggamma~U[-.5,.5]，沿用run06 clipping和输入。共1688 source条件。评估ZERO/I(run06)/G(run10)/O(只以真实gamma替换I、RGB保持预测)，记录OD/OC Dice、I/G gamma误差、高端裁剪率。无clean额外前向，无目标，无优化。

聚合源等权G−I及O−I；pooled p10、退步率，gamma误差改善的子集中分割退步比例，source条件mean gamma MSE ratio。预设解释：若gamma误差改善子集中至少25%分割下降，优先考虑带任务风险约束的gamma学习，后继另立协议与对照；若未达到，则先检查源分布覆盖/可辨识性，不直接增大残差或再调学习率。此分组关联不是因果效应/安全保证，源dev重复开发/base暴露局限保留。

1小时硬截止，300秒TERM/60秒KILL，200MBsoft500MBhard预计20MB；总新增<50GB45GB预停。唯一30min健康静默，完整结果公开匿名聚合源码，按源证据自主推进。
