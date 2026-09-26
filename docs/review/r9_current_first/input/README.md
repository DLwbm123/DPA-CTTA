# R9 Current-First 计划包

入口：`R9_ANALYSIS_AND_PLAN.md`。
实施交接：`CODEX_R9_IMPLEMENTATION_PROMPT.md`。
机器配置与完整展开矩阵：`R9_SPEC_AND_MATRIX.json`。
Screen24 公开数字核算：`SCREEN24_VERIFIED_SUMMARY.json`。
计划计数校验：`python verify_plan.py`。

本包状态为设计/实施提议，不是已实现的训练系统、运行授权或有效性证明。
没有启动模型、访问医学图像或checkpoint、修改GitHub仓库、创建监测。

矩阵：43源任务，正常新增648,000主fit更新；
610核心目标评估槽位，固定16k敏感性最多161槽位，总最多771。
源和目标任务复用/新执行/恢复费用分别核算；槽位数不是新增物理任务或独立患者数。
512 GPU-worker小时为拟议资源上限，不是实测耗时或完成承诺。

原实验是Screen24的4k缩减筛查；不能宣称完整16k R8已完成或方法已收敛。
本轮优先测试“当前图观察/预测”训练对齐与代理辅助目标，保留原配方长训练对照，
并测试有限目标梯度、同权重RESET和匹配的BN-reset控制。
所有新参数选择只用源数据，在本轮目标分数解封前锁定。
