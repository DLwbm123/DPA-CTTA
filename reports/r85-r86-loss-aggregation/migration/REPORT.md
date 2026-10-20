# R85/R86：维修后的同协议迁移

**fresh qualification PASS，已启动静态基线与后续自动实验；尚无新性能结果。**

用户明确要求替代服务器仅GPU0/1，所有数据、模型、环境、日志、中间产物放在remote-home。原服务器维护导致旧结果暂不可取，不记为性能失败，不覆盖旧输出。迁移的共同S0、连续Adam、192内容/两order/三seed、q/W、强增强及损失定义全部保持，科学门槛不变。迁移细则见 [PROTOCOL_AMENDMENT.md](PROTOCOL_AMENDMENT.md)。

原192队列已依据原R23准备脚本，从历史R20 pool384重建；其SEARCH/REVIEW assignment与保留的历史映射完全一致。域数量22/57/57/56、两order同192集合；只核元数据与路径可读，未据分数重新选样。数据、checkpoint和验证环境直接复用，没有新大副本。

由于旧逐预测certificate和端点文件不可取、GPU硬件环境改变，本attempt重新运行VV/BB/NATIVE/VB/BV共同硬件对照。明确不声称跨硬件与旧R79逐预测bit-exact；fresh VV/BB对原代码及VB/BV/NATIVE独立公式的完整核心state/output exact仍强制。旧原服务器至少已观察到2752更新及静态192F，尾部覆盖和费用UNKNOWN，恢复后另取证。迁移重跑不是独立验证，R87仍NOT_AVAILABLE。

CPU准备PASS14.563460秒。此前一次系统解释器缺少torch._C，在metadata模块导入处失败、无模型/数据/标签/GPU调用，已改用现存实验环境；失败证据保留，短暂失败CPU时长未计量，不伪造零值。

新GPU资格20/20更新、196F、16次诊断反向遍历全部PASS：四格对原实现/独立公式output/gradient/BN/Adam/RNG核心state exact，PRE完整恢复、q/wholeW来源、19136参数/冻结/连续Adam通过；增加NATIVE原公式core exact、P0 current-stat与native只读输出exact。资格worker body77.077721秒，是GPU进程费用子集，不重复相加。

计划本attempt5924新更新、58324F、400额外诊断反向；30连续轨迹，PRE/K11520、同状态探针192预测、SOURCE_ONLY/P0各192预测。自动阶段：fresh资格→静态SOURCE_ONLY/P0→VV+48探针→新BB/NATIVE→VB/BV→所有GPU真实退休→CPU评分。资格后的墙钟规划5571.270098秒，只是估计，不当实测完成时间。

原T0与24小时硬截止保留：北京时间2026-10-21 13:04:01.100432。独立watchdog已启动；新root预留4.4GB，cap4GiB及128MiB恢复余量，既有data/model/env只读复用。保守含旧原服务器预留、local和headroom合17.4GB<20GB。allocator固定8GiB，只GPU0/1且UUID绑定，主/子可见argv中性。唯一r52已改为替代服务器半小时跟进，没有增加心跳。

scorer启动前修正迁移报告预测计数字段为11520 PRE/K（而非继承的6912），并单列P0静态192；此改动仅修正交付元数据，不改损失、更新、数据、评分数值或门槛。科学方法资格对应6eed982；该报告计数修正单独提交保留。

仅公开源码、协议和匿名资格/准备证据；图像、标签、身份、私有预测、权重、原始日志和机器路径均不发布。此为启动交付，正式结果及TERMINAL_AUDIT/全部成本/真实空间/正负结果报告待本attempt真实终态后补齐。

启动现场进一步确认：资格与SOURCE_ONLY/P0静态worker均exit0、注册PID消失；静态共192到达/384F。两个VV参考worker已在GPU0/1实际RUNNING，startticks、中性ps argv及nvidia-smi UUID绑定均通过。已收尾GPU进程上界151.563334秒（资格90.875878、静态60.687456），CPU准备14.563460秒；参考活跃elapsed尚未包含，不能当最终成本。ledger中的all_workers_retired=true仍是上一组收尾的陈旧事件快照，现场活跃核验优先。
