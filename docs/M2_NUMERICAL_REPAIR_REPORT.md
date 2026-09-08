# M2 数值一致性修复报告

状态：**修复及正式实验均已完成。四套 600 步、552 条评分、独立重算和报告生成均成功。** 最新结论见 [最终实验报告](../results/m2_episode_coverage_v1/M2_EXPERIMENT_REPORT.md)。下文启动阶段快照保留作为过程记录。上次 M2_PARTIAL 证据保留在提交 c1af390；本轮是用户明确要求后的修复执行。

## 定位结果

两个独立 host 的模型、prompt、Adam、memory 与 counters 初始状态精确相同。程序化 Polyp 诊断中，适配前 live logits 和 proxy logits 精确相同，但 prompt 梯度最大差为 6.05359673500061e-09；该次微小差异没有跨过最终 logits 容差。原始失败运行的逐算子中间结果没有保存，因此不声称重现了完全相同的 0.0003593 差值或确定唯一首个分歧算子。

问题在于 smoke 将相同 seed/cuDNN deterministic 当作全 CUDA 路径确定性保证。固定环境中 `deterministic_algorithms=False`，Polyp 使用多处 bilinear 插值，原生 `upsample_bilinear2d_backward_out_cuda` 的严格确定性探针明确报错。部署的 Torch 2.2.1 `nn.functional.interpolate` 在严格确定性启用时改用已有确定性 decomposition。无需重写插值或更换 Torch。

第一次严格模式预检还指出 cuBLAS 需要在 CUDA 初始化前设置工作区；此预检没有发生在线 Adam 更新。设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后，同一临时代理的两条路径在前向输出、prompt 梯度、更新、状态和 RNG 上全部精确一致。诊断共 4 次在线更新、1 次图像外层更新、0 次可微内层；另有一个无模型/optimizer 的程序化原生插值探针。它们单独计费，不混入正式实验步数。

## 最小修复和验证范围

仅在 smoke 的两个在线 host 配对比较期间使用 `torch.use_deterministic_algorithms(True, warn_only=False)`，上下文退出时恢复原 deterministic 与 warn-only 标志，异常也恢复。smoke 子进程在 CUDA 初始化前配置 cuBLAS 工作区。正式训练和评分使用独立进程、原 M1 环境，不启用该上下文；图像外层目标、seed、episode 列表、source/target 划分和 `rtol=1e-4, atol=1e-5` 均保持。

改进失败证据保存：先记录已通过的外层梯度/像素检查，再运行在线比较；失败时保存计时和峰值显存。独立重算导出中保留 smoke 的确定性作用范围，不能把这种受控入口等价检查误写为原 CUDA 路径逐比特可复现。

本地 8 项 M2 测试通过，含新增上下文成功/异常恢复检查。部署环境 23 项相关测试通过，退出码 0（先前测试加载器模块名写错已纠正，原加载错误日志保留；不属于方法或代码测试断言失败）。

完整四组 smoke 通过，退出码 0：Fundus D2、O2 和 Polyp D2、O2 的 logits 最大差都为 0，图像梯度有限且非零，prompt、Adam、counters、memory、RNG、source 及对应梯度检查全部通过。实际 8 次在线更新、4 次外层更新、2 次可微内层，耗时 20.962 秒；Fundus 峰值 7,854,393,856 字节，Polyp 峰值 5,231,086,592 字节。作用域退出后 deterministic_algorithms=False，正式阶段通过独立进程恢复 M1 环境且不继承 cuBLAS 工作区覆盖。

修复执行提交：`fef00f5bb1ea9c557054215ebe00ca5ee932ab81`。GPU 7 正式任务通过一次有限后台 launcher 执行，不依赖当前 SSH/会话；训练→评分→独立 CPU 重算→Markdown 报告按序执行，任一步失败即停止，不重启。四套训练各 600 步，全部冻结后生成 552 条新评分。报告渲染器只接受成功的独立重算，已通过程序化正例和拒绝未完成结果的检查。

一次启动检查确认 Fundus D2 完成 47/600 步，loss 0.41203469、梯度范数 0.86340833，日志可读且无失败记录；正式环境与 M1 差异为空。这是启动时快照，不是最终结果。

此时暂无新方法效果，O2−D2、O2−O1、D2−D1 和七域方向均待完成结果，不能用 smoke 通过替代方法有效性。上次失败报告及旧 1,380 条评分未覆盖。当前报告是修复与启动报告；完整实验报告由成功的 CPU 重算结果生成。

[诊断证据](../results/m2_episode_coverage_v1/numerical_repair_audit.json) · [诊断脚本](../scripts/diagnose_m2_smoke.py) · [上次失败报告](M2_EPISODE_COVERAGE_REPORT.md)
