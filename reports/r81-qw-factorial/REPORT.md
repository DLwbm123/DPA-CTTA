# R81：q与W反馈的2×2连续机制对照

状态：**RUNNING**。本轮独立CPU检查、48次GPU资格更新和资源准入均通过，2026-10-19 17:22:07（北京时间）正式启动；截止20:22:07，之后不得重置。本报告是启动交付，不是科学结果或完成报告。

科学执行源码 `3f5a37816f17ca3c6909d1d7fea88394e871183c`；完整冻结设计见 [PROTOCOL.md](PROTOCOL.md)。本轮索引为(q来源,W来源)，不同于R80的(Adam,jointQ)。仅新增q1W0和q0W1，36条新连续轨迹、6912次正式更新；q0W0复用R80cell10，q1W1复用R80cell11，36条旧端点轨迹只读复用。新PRE/K预测13824个、复用PRE/K13824个。保留192既有内容、2既定顺序、3既定种子、BN19136 affine、非BN/head冻结、原W/0.1 orientation LSO公式和Adam连续状态，不扩展模块。

q_i来自所选供体的六弱概率均值，W_j的方差、边界(依据W供体自身均值)和归一化全部来自W供体。BCE和LSO统一使用q_i、W_j，LSO内部由q导出的结构仍随选定q。当前预测来自本格自身连续状态；初态参照通过同一模型临时恢复BN只读生成并完整恢复状态/RNG，不新增教师模块。混合格每步额外6次初态前向，NATIVE共14F、VERTEX/BARY共15F，加PRE分别15/16F；一次backward、一次Adam。物理forward/update独立计数，不随快照恢复。

## 本轮资格与资源准入

- CPU检查PASS，6.613534451秒，无CUDA初始化，强增强/RNG、q/W损失及分解恒等式检查通过。
- 三方法共48/48资格物理更新PASS：新00/11对原R80对应格两次到达的完整状态与输出exact；PRE/初态六弱观察无副作用；初态q/W对独立oracle exact；两混合格q/W来源exact、输出/梯度/BN/Adam/年龄/buffer/RNG核心状态对独立原公式实现exact；初态混合首步与对角格一致；Adam连续、BN19136冻结、一次backward/Adam及真实F计数通过。
- 资格物理前向656次，全部计入资格GPU进程75.221260548秒，未重复叠加body时间。本轮独立GPU上限18000秒，资格后剩余17924.778739452秒。
- 冻结准入投影8394.770489502 GPU进程秒，使用六方法×混合格的第二完整步实测时间计算，通过准入后才正式启动，未重测或改公式。
- 启动核验两个正式worker实际RUNNING，各16次提交/240次物理前向，共32/6912次正式更新；完整可见命令行中性，固定GPU0/1 UUID绑定已核。账本是事件快照，all_workers_retired旧值不能替代实际进程状态。
- 每worker allocator8GiB，NAS输出上限2GiB含64MiB预留；CPU评分上限900秒；全部资格、诊断及失败GPU成本计账。

见 [STARTUP.json](STARTUP.json)、[QUALIFICATION.json](QUALIFICATION.json)、[FORMAL_ADMISSION.json](FORMAL_ADMISSION.json)、[FROZEN.json](FROZEN.json) 和 [RESOURCE_LEDGER.json](RESOURCE_LEDGER.json)。进度为启动快照，非后续实时值。

## 终态分析与交付

全部三GPU工作进程真正退休exit0/GPU_COMPLETE后才读标签评分。沿用H=PRE−P0、C=K−PRE、T=H+C，分别计算BASE_FIXED=00、Q_AT_W0=10−00、W_AT_Q0=01−00、交互11−10−01+00；四项和11，后三项和已有联合q/W反馈效应11−00。另报Q_AT_W1=11−01、W_AT_Q1=11−10及各格。每项均绝对方法与V−N/B−N/B−V，重点D1OC/OD、D0/D2/D3OD，保留全部正负单元、最差seed/order、内容中位数/覆盖、FP/FN/前景及实际早晚位置。

这是连续学习轨迹中的q/W路径作用，Adam保留但会因梯度历史内生分化，不是同入站状态直接梯度的纯效应；不进一步拆W方差/边界或BCE/LSO，不把初态参照视为真值。192内容是已暴露开发集，非患者独立/FULL验证，旧端点不是本轮重复验证。

半小时监测已恢复，仅通知重要变化。失败、资源耗尽或边界/状态/计数检查不通过即保留停止，不自动修复重试。结束后交付完整匿名结果、终态/资格/成本审计及中文报告到同公开分支，核远端SHA与匿名公开访问并保存新交付回执，然后暂停r52。当前尚无本轮科学结论。
