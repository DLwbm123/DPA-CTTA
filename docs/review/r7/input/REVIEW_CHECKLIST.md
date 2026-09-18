# 外部审阅重点

1. 借鉴与实际实现边界：不把自拟过滤叫原RP-GSSM，不把basis模块叫已证明的前门调整。
2. SOURCE训练新增信息：fit/cal/val及proxy oracle隔离、source预算、同预算独立STATIC。
3. 三个核心算子：A full F/P与基数学；B温度和5步ISTA；C方差校准与3步IRLS。
4. 患者原特征保留与source-query不同内容：不能把source支持图拟合当作跨患者有效。
5. 目标因果性：不读GT/未来，state只存声明变量，target不更新训练权重。
6. 同期C/C0与prepare-static控制，公平性与实际成本，不按目标分数重训挑权重。
7. Full random network与source训练梯度验收；数学22项不替代此步骤。
8. 每组独立可审阅，CPU调试不强制再次外部审阅；真实source/target入口必须等待独立授权。

当前此清单不是已完成的审阅。
