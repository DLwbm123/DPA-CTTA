# 本轮新增测试

全部位于 tests/test_r5_audit_fix.py，默认由 scripts/check_r5_cpu.py 加载。共 9 个 test methods，含参数化 subtests。

| 测试 | 覆盖 |
| --- | --- |
| test_metric_feasibility_boundaries_and_original_dice_guard | 满图不可能交集、下界恰等/低一、空/非空/满图、整数、原 Dice guard |
| test_33_valid_shadow_rows_impossible_counts_all_predictions | 33 条合法影子 trace，pre/q/trial/emit 自洽但不可实现计数 |
| test_complete_A_impossible_metrics_invalidate_without_scientific_result | 完整 A fixture 污染，recompute 失效与有效指针撤回 |
| test_smoke_partial_forward_all_hosts | OLD + 四臂各在第二访问完成 3 个 forward 后抛错 |
| test_smoke_after_backward_before_Adam_all_hosts | OLD + 四臂 backward 后、Adam 前抛错 |
| test_smoke_after_Adam_before_return_all_hosts | OLD + 四臂 Adam 后、step 返回前抛错 |
| test_smoke_success_A_B_exact_budgets | 实际 CPU smoke 编排 A=64/8/8/0，B=160/20/20/0 |
| test_smoke_preserves_first_failure_and_original_exception | 首失败记录不覆盖，原异常继续传播 |
| test_rollback_next_step_reference_never_attempted_candidate | 独立原 C 从未执行拒绝图/从未调用被测恢复函数；空/已有 Adam 下一步精确一致 |

三个异常 smoke methods 都用独立真实计算 hook 交叉核对 live counters，验证停止、不继续后续臂、不发布 completion。它们不包含真实 checkpoint 或像素。失败的 hook 计数被标注为可证实的观测下界，不能视为无法观测的 in-flight 工作完整成本。

原 22 项继续覆盖完整随机 ResUNet、固定增强、规则/窗口/随机流、状态回滚、标签隔离、A/B gate、完整 3+17 fixture、授权、身份污染和历史保护。112 项继承覆盖 IO/NFS、进程监督和既有 R1–R4 路径；共享实现未修改。完整套件 143，而非 143+31+9 个不同测试。
