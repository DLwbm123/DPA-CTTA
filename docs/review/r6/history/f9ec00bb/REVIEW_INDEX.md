# R6 阶段 I 交付索引

**R6_STAGE_I_BLOCKED_MISSING_ORIGINAL_MATERIALS**。四臂代码和真实登记元数据已实现；整体未达到 IMPLEMENTATION_READY_FOR_REVIEW。缺失原始science JSON、数学参考及math_history，详见 [BLOCKERS.md](BLOCKERS.md)。不会自行签发外部review PASS，也不启动 GPU/A/B。

Implementation SHA：`d875f20c11cc7e617c9f39dba04ed46381aba450`。

Science 预期原始字节 SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`；**原件未收到，本次未验证此摘要**。

- [实施报告](IMPLEMENTATION_REPORT.md)、[机器交付状态](DELIVERY.json)、[历史保持性](PRESERVATION.json)、[完整新增代码patch](IMPLEMENTATION.patch)
- [原prompt](STAGE_I_PROMPT.md)、[原计划](EXPERIMENT_PLAN.md)、[方法来源](METHOD_PROVENANCE.md)、[生命周期](STATE_LIFECYCLE.md)、[parity合同](PARITY_CONTRACT.md)
- [测试覆盖](TEST_COVERAGE.md)、[复现](REPRODUCE.md)、[日志索引与成本](logs/INDEX.json)、[最终本地](logs/final-local.log)、[最终服务器](logs/final-server.log)、[当前host→标量集成](logs/host-scalar-probe.log)
- [首轮失败](logs/core01.log)、[修正后R6检查](logs/r6-01.log)、[补充独立目标/零梯度Adam检查](logs/host02.log)
- [真实登记dry-run与去身份流摘要](metadata/DRY_RUN.json)、[A 12条](metadata/A_MATRIX.json)、[B新增8条](metadata/B_NEW_MATRIX.json)、[AB 20条](metadata/AB_MATRIX.json)

仅新增本轮文件；main、旧C/P2/R1–R5配置与结果、固定依赖不变。R5仍NO_ADVANCE，B关闭。本轮real execution_started=false，external_review=NOT_RUN，R6A=NOT_RUN，R6B=NOT_RUN，devices=null，caps批准/review/waiver/receipt未生成。

公开日志保留实际顺序、异常、失败和退出码，仅将checkout与解释器安装路径前缀替换为占位符以去身份；未截断或删除测试失败。未编辑原始日志保留在本地私有证据目录。数学参考的既往8项日志没有提供，不拿本轮日志冒充。Synthetic ledger与mock设备仅为CPU测试fixture，未作为真实实验/授权receipt发布。

此目录为实现提交之后的证据发布，发布SHA在最终交付消息中单独给出，不把文档发布SHA当成已测试实现SHA。
