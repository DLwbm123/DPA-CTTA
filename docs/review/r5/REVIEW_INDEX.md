# R5 Stage I Review Index

**R5_IMPLEMENTATION_READY_FOR_REVIEW**

execution_started=false · external_review=NOT_RUN · R5A=NOT_RUN · R5B=NOT_RUN

Implementation SHA: `0c08cece6a91bdf5e3d06c9862dcd192df7787d8`

Science SHA256: `89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5`

本包只请求外部审阅；不签发 review PASS 或 GPU 授权。实际执行另需绑定 checkout SHA 和新阶段 receipt。

1. [原始阶段 I prompt](R5_STAGE_I_CODEX_PROMPT.md)、[实验计划](EXPERIMENT_PLAN.md)：范围、阈值、停止条件。
2. [实现报告](IMPLEMENTATION_REPORT.md)、[方法来源](METHOD_PROVENANCE.md)：完成内容、验证、限制。
3. [状态生命周期](STATE_LIFECYCLE.md)、[保持性约定](PARITY_CONTRACT.md)：真实 C 链、完整回滚、标签顺序、CPU/GPU 分界。
4. [窄代码 patch](IMPLEMENTATION.patch)、[源码](../../../src/dpa_ctta/r5_update_acceptance)、[CPU 测试入口](../../../scripts/check_r5_cpu.py)、[冻结 science](../../../configs/r5_science_v1.json)、[disabled 授权默认值](../../../configs/r5_execution.defaults.json)。Patch 从规定基线到 implementation SHA，只有新增源代码/配置/测试/脚本。
5. [CPU 复现](REPRODUCE.md)、[全部检查索引](logs/INDEX.json)、[最终本地日志](logs/final02.log)、[最终本地 JSON](logs/final02.json)、[继承回归日志](logs/final01.log)、[服务器日志](logs/cpu-server-01.log)、[服务器 JSON](logs/cpu-server-01.json)。早期和最终日志均保留，失败/跳过按原样记录。
6. [metadata dry-run](DRY_RUN.json)、[A 3-job](A_MATRIX.json)、[B_NEW 17-job](B_NEW_MATRIX.json)、[A+B 20-job](AB_MATRIX.json)、[原登记流摘要](STREAM_SUMMARY.json)。这些是计划，不是正式结果。
7. [去身份交付清单](DELIVERY.json)：SHA、状态、预算与 CPU 验收出处。

四臂 C / C_HALF / C_RANDOM / C_VERIFY 已实现。当前未使用真实目标或给定 checkpoint，未初始化 CUDA。可选 C0 历史分解未有匹配逐内容证据；H_t=null。A/B 资格分析中的 fixture gate 仅为程序测试，不代表真实科学 gate。
