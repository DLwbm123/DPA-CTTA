# 方法来源与限制

直接科学定义来自随附 STAGE_I_PROMPT.md、EXPERIMENT_PLAN.md；预期原始 science JSON 尚缺失，见 BLOCKERS.md。实现不把已提供文本冒充该 JSON，不宣称外部参考八项检查已由本次执行验证。

基线 e271098e2a12baa166fc7b77848af4a2300d9cc3；R5-A 历史执行 b4b71601a5bdf87bb3a7e5d3db352610adcdff74。已读基线 REPORT/PUBLIC_AGGREGATE 与审计修复材料：R5-A 两主序 G=-0.007689130379057169pp，NO_ADVANCE 保留；旧 C 路径即时 GT 选优量不作为新 loss 长期增益上界。B2 interval 的半径/投影改变 target 或残差，与 R6 整体 BCE 乘正权重不同，没有复用旧 B2 为本轮运行证据。

原 C：b1_host → 固定 GraTa.cal_consis_loss；模型与预处理沿既有 SourceOnlyHost / CTTA dbff0d985c6c95345d9fb78f5b1daef57b392564。科学变化限定为 q≥0.5 分区的正权重；整项 BCE 同乘，不能替换为 pos_weight。rho 限幅 1/8..8；未限幅两区各占一半质量，限幅后不作 50/50 声称；单区回退为 1。

R_SCALE 匹配当前自身逐通道 logit 梯度范数，不是 global-LR 匹配；R_SHUFFLE 全通道置乱基础权重，专用 CPU RNG。基础直方图不变，b 乘后绝对权重总量未必为 1。像素面积不是梯度贡献，记录真实 residual SSE 与梯度 hook 支持检查；同调用数不等于同 FLOPs/时间。

工程复用：R1 supervisor/evidence/assets 保留有限排程、拥有进程组清理、ENOENT 限定容忍；R3 worker_environment/deterministic smoke；R5-local 可实现性 validator 只复用 TP/FP/FN/TN 检查，不调用 R5 recompute 或 gate。R6 独立 marker/run_id/scope 保护先于 invalidate。B 新增 8 条且复用本轮 A 全部 12 条及原执行绑定。

该设计是普通损失重加权的受控开发实验，不宣称发明类别平衡、显著性、临床安全或已证实有效。附带计划的外部文献说明未被扩大为本仓库实验结论。本轮只读公开 CPU 聚合和登记元数据，未读取真实像素、mask 或 checkpoint。
