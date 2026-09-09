# B1 GraTa matched baseline: interface failure and CPU-validated repair

**状态：INCOMPLETE。正式实验尚未启动，没有 C/G 目标评分结果。**

失败执行提交：`91fc1178ee1ade6335e36dae39e2f4a398e182e5`。
修复代码提交：`029a79340057b58b8074ccf86c7b5671b5fdbd1c`。本报告另行发布。
分支：`experiment/b1-grata-matched-baselines-v1`。

## 已完成与停止位置

- 复用 P2 的 1951 个 Fundus 内容组：remaining_dev=1695、legacy_dev=128、p1_extension_dev=128。
- 两个域序及旧 P2 七输出记录的身份、顺序、覆盖、父视图、计数与 Dice 重算通过；本轮报告计划使用 N/A/EA/O2/D4 五个旧对照。
- 原 6 项 CPU 测试通过。GPU 7 有 16668 MiB 可用显存，超过 12288 MiB 准入预留；与其他任务共存，没有终止其他进程。
- 首轮 GPU smoke 在 C 参考路径的第一个弱视图前向后解包报错：`too many values to unpack (expected 2)`。
- 持久化 optimizer-call 前缀为 **0 条**：已完成 GPU base Adam=0，正式记录=0。失败发生在读取正式目标前；没有方法效果结论。
- 控制流与 traceback 对应 2 次 source 对齐前向和 1 次失败接口的弱视图前向；这是控制流重建值，不冒充前向 hook 的实测计数。

## 原因与修复

错误来自本轮遗漏包装接口。P2 的规范模型实际通过既有 reference builder 构造，返回 `(logits, skips, head_input)`；GraTa 的 `cal_consis_loss` 要求恰好 `(logits, feature)`。原程序化小模型只返回两个值，因此未覆盖实际接口。这不是 GraTa 分割性能失败，也不影响既有 P2 结果。

修复在生产和参考调用点共同加入 OutputPair：保留规范模型与参数名称，仅将现有输出收窄成 `(logits, skips)`。ent/consis 不使用 feature；不增加模型前向、参数或随机辅助 head，不改变 logits、BN 配置、权重映射、增强、loss、扰动、学习率或 Adam 设置。所有官方更新函数仍来自固定源码，禁用标签及无关辅助头入口。

新增实际完整分割模型的 CPU 回归，同时把小模型改为三返回值。回归对 C/G 的官方参考和新入口执行程序化访问，检查 logits、梯度、Adam、RNG 和冻结参数。修复后的 **7 项测试本地及服务器全部通过**；服务器用时 45.757 秒，0 failures / 0 errors / 0 skips。CPU 回归使用程序化输入及合成状态，不是新的 GPU smoke，也不是目标评价。

## 冻结预算与后续状态

计划预算仍为正式 7804 条新评分、7804 次 base Adam；含计划 smoke 为 7820 次。
正式模型前向预期 C=31216、G=35118，总66334；反向11706，G扰动/恢复各3902。
这些是尚未执行的计划数，不能写成完成数。修复后的 GPU 一致性与正式运行均未执行。

原失败执行 checkout、receipt、smoke 日志及零更新前缀保持原样。修复放在独立 checkout，未复用旧 receipt 冒充新执行。
计划第 12 节要求“工程故障记为 INCOMPLETE 并保留前缀”，并禁止“自动续跑形成一个伪完整实验”；因此本轮在 CPU 修复复核后停止。重新执行 GPU 验收与正式流须由用户明确允许继续，不能静默自动重试。

## 公开边界与来源

公开源码、冻结配置、MIT 许可证、去标识登记统计、失败计数和修复审计；私有图像、mask、身份、资产路径、checkpoint、概率图及原始 traceback 未发布。
模型依赖仍为 CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564，B1 基线为 P2 发布 2fcbc0a69645e34da42b97830935f0e290e488aa。
增强库 batchgenerators 0.25.2 独立安装，没有更换服务器 PyTorch/CUDA。

官方函数来源：[GraTa 固定源码](https://github.com/Chen-Ziyang/GraTa/tree/33ae20d664f305af34739ec54a5bec7da53ffa0b)。
本轮定义为匹配权重、连续流的移植；不是完整原论文协议复现、原创算法或已证实的性能提升。

