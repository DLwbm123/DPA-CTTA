# R5 方法与代码来源

- 用户附件 R5_STAGE_I_CODEX_PROMPT.md 与 EXPERIMENT_PLAN.md 是本轮冻结的新规则/阈值/分阶段预算来源，不代表先前结果已验证该方法。
- 原 C：src/dpa_ctta/r4_three/host.py 的 C 分支 → src/dpa_ctta/r1/host.py 的 C → src/dpa_ctta/b1_host.py；固定 GraTa cal_consis_loss 内部创建六弱视图，先逆变换 logits、CPU stack/sigmoid/mean，再把 q 移到模型设备进入 BCE。R5 criterion 只 clone 实际 q 后调用原 BCEWithLogitsLoss，不替代 q 的归约。捕获的六视图只用于可靠区判断。
- 固定 CTTA dbff0d985c6c95345d9fb78f5b1daef57b392564 与 GraTa 33ae20d664f305af34739ec54a5bec7da53ffa0b；调用原 ResUNet34，无依赖升级或源码补丁。
- Adam 与 BN 政策直接复用 B1。R5 不调用 load_state_dict 重建 Parameter；逐参数 deepcopy Adam 状态并原位恢复 group/parameter ownership，以保留延迟初始化语义和额外字段。
- 指标/ASSD 来自 p1_analysis.evaluate；单像素计数核验来自 p2_analysis.validate_metric；分布、配对/ASSD 分母沿用 host_diagnostic_analysis 与 m1_analysis。
- 流与 registration 摘要沿用 R3/R1；没有重建 recurrence。监督、NFS、原子证据和命令保护沿用 R1/b3_runtime，未改共享代码。
- 只实现新提出的更新接受诊断；不主张首次创新、独立确认、统计显著性、安全或泛化。旧 R4 与 B4 报告只是设计背景。无源数据、代理、原型或新增预训练组件。
