# M2 数值一致性修复报告

状态：修复已实现，修复后的完整 smoke 和正式实验待执行。上次 M2_PARTIAL 证据保留在提交 c1af390；本轮是用户明确要求后的修复执行。

## 定位结果

两个独立 host 的模型、prompt、Adam、memory 与 counters 初始状态精确相同。程序化 Polyp 诊断中，适配前 live logits 和 proxy logits 精确相同，但 prompt 梯度最大差为 6.05359673500061e-09；该次微小差异没有跨过最终 logits 容差。原始失败运行的逐算子中间结果没有保存，因此不声称重现了完全相同的 0.0003593 差值或确定唯一首个分歧算子。

问题在于 smoke 将相同 seed/cuDNN deterministic 当作全 CUDA 路径确定性保证。固定环境中 `deterministic_algorithms=False`，Polyp 使用多处 bilinear 插值，原生 `upsample_bilinear2d_backward_out_cuda` 的严格确定性探针明确报错。部署的 Torch 2.2.1 `nn.functional.interpolate` 在严格确定性启用时改用已有确定性 decomposition。无需重写插值或更换 Torch。

第一次严格模式预检还指出 cuBLAS 需要在 CUDA 初始化前设置工作区；此预检没有发生在线 Adam 更新。设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 后，同一临时代理的两条路径在前向输出、prompt 梯度、更新、状态和 RNG 上全部精确一致。诊断共 4 次在线更新、1 次图像外层更新、0 次可微内层；另有一个无模型/optimizer 的程序化原生插值探针。它们单独计费，不混入正式实验步数。

## 最小修复和验证范围

仅在 smoke 的两个在线 host 配对比较期间使用 `torch.use_deterministic_algorithms(True, warn_only=False)`，上下文退出时恢复原 deterministic 与 warn-only 标志，异常也恢复。smoke 子进程在 CUDA 初始化前配置 cuBLAS 工作区。正式训练和评分使用独立进程、原 M1 环境，不启用该上下文；图像外层目标、seed、episode 列表、source/target 划分和 `rtol=1e-4, atol=1e-5` 均保持。

改进失败证据保存：先记录已通过的外层梯度/像素检查，再运行在线比较；失败时保存计时和峰值显存。独立重算导出中保留 smoke 的确定性作用范围，不能把这种受控入口等价检查误写为原 CUDA 路径逐比特可复现。

本地 8 项 M2 测试通过，含新增上下文成功/异常恢复检查。下一步在原部署环境运行相关测试和完整四组 smoke，通过后执行原定四套 600 步训练及 552 条新评分。此时暂无新方法效果，不能判断 O2−D2 或七个域的方向。

[诊断证据](../results/m2_episode_coverage_v1/numerical_repair_audit.json) · [诊断脚本](../scripts/diagnose_m2_smoke.py) · [上次失败报告](M2_EPISODE_COVERAGE_REPORT.md)
