# R6 prompt 定义实现与阶段 I 阻塞

实现提交 **d875f20c11cc7e617c9f39dba04ed46381aba450**，基线 e271098e2a12baa166fc7b77848af4a2300d9cc3，分支 experiment/r6-bounded-regional-consistency-v1。仅新增 R6 文件，未改 main、历史配置/结果、旧C或依赖；完整代码patch见 IMPLEMENTATION.patch，路径清单见 PRESERVATION.json。

**阶段 I 未达到 READY**：缺少 prompt 指定的原始 R6_SCIENCE_PROPOSAL.json、R6_MATH_REFERENCE.py、math_history。没有重建并冒充这些原件；science预期摘要与实际验证分开。详见 BLOCKERS.md。代码、CPU证据和真实登记元数据可审阅，不能自发签发外部review PASS或执行许可。

已实现 C、R_BAL、R_SCALE、R_SHUFFLE；只改变实际 q 上的普通 BCE 正权重，原C标量loss与八次调用链保持。分区double构造，actual float32权重回传double审计；残差先用loss dtype计算，scale detached，专用CPU generator全通道置乱。只读logit梯度hook与代数预测核验，记录BN梯度和Adam位移。没有新增网络、teacher、source组件、历史图库、拒绝或回滚。

host分离RGB与提交后GT评价，稠密payload当前图结束即释放，失败不可续跑。独立R6分析器复用既有可实现性wrapper，验证真实登记身份、覆盖、行序、计数、权重/能量/scale/seed、GT一致性；只有R6 marker/run_id/scope匹配才能invalidate。报告所有子集、逐域逐序、OD/OC、等域主指标、内容权重辅助指标、配对正零负/最差ceil10%/最差单内容，以及ASSD共同有效/undefined；无macro ASSD。

A为12条，B_NEW新增8条、复用A完整12条及原SHA/fingerprint，总计20。AB只能metadata。A无论gate是否通过均停；B独立授权。固定gate无降门槛、无自动扩臂/重跑。未来smoke为旧C及四臂各4步，每设备每阶段160F/20B/20Adam/0VJP；只要求新旧C parity。执行复用R1有限监督、NFS和R3环境边界，中性入口先于runner，首失败及live下界保留。

真实登记metadata dry-run已生成12/8/20矩阵与去身份recurrence摘要；1951唯一内容/流、remaining_dev1695。A正式187296F/23412B/Adam，B_NEW124864F/15608B/Adam，AB312160F/39020B/Adam，VJP均0。它们是预算，非已运行成本；devices=null，caps只有proposal，没有批准。真实计算为0。

最终候选完整CPU结果以本地/服务器JSON为准，覆盖说明见 TEST_COVERAGE.md。首次数学参考精度测试失败、修正及最终日志全部保留。缺失的外部八项参考与其历史不能用这些新日志替代。研究依据与继承来源见 METHOD_PROVENANCE.md；不声称有效性或显著性。实际科学实现仍需与原始JSON核对后才能冻结。

## 服务器临时目录偏差

两次服务器CPU验证使用继承runner的fresh TemporaryDirectory，启动时未显式设置TMPDIR；随后只读确认服务器默认临时目录为/tmp，而非全局服务器约定的NAS临时路径。这里仅有程序化IO/标量ledger fixture，没有真实目标或source数据；代码、bundle和原始日志一直位于规定NAS项目目录。该存储位置偏差如实保留，不能通过改写日志声称未发生。正常结束的完整suite由其TemporaryDirectory上下文清理；被主动终止的旧候选不声称已保证清理全部临时fixture，也不冒险删除来源不确定的/tmp目录。后续服务器复现必须显式设置TMPDIR到该任务的新NAS临时目录。此偏差不改变CPU数值测试结论，但不是完全符合存储约定的执行记录。

## 最终实际 CPU 结果

固定提交 d875f20c11cc7e617c9f39dba04ed46381aba450：本地 Python 3.12.9 / Torch 2.6.0，166/166通过，400.9239秒；服务器 Python 3.10.6 / Torch 2.2.1+cu121，166/166通过，1151.5609秒。两端failure/error/skip均0、进程exit=0、CUDA未初始化、真实RGB/mask/checkpoint/source读取为0。服务器测试后checkout干净。B1覆盖范围实测各4514F/561梯度hook/551Adam-post；继承全套VJP总量未统一计数，不将其预置字段当实测值。完整证据见日志索引。最终READY仍被原始配套材料缺失阻塞。
