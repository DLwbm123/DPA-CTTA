# Parity 与局部梯度匹配

固定 GraTa cal_consis_loss 来自提交 33ae20d664f305af34739ec54a5bec7da53ffa0b，调用 B1 C 原链；不是 R5 host 或另一 teacher 分支。R6-C 仅捕获实际 q、首弱 pre、最后 post 并增加只读诊断。完整随机 Fundus ResUNet34 的 OLD R1-C 与 R6-C 四步比较 logits、BN affine、完整 Adam、augmentation RNG，CPU bitwise。41 BN 层、82 affine tensors、19136 scalars；frozen state finish 检查不变。

独立统一状态测试验证 R_BAL/R_SCALE/R_SHUFFLE 的逐 OD/OC strong-logit 梯度范数：float64 rtol=1e-10/atol=1e-14；float32 rtol=1e-5/atol=1e-12，首次运行前已写入代码，失败后未放宽。实际 residual 在 loss dtype 相减后提升 CPU double；CPU 审计使用 float32 实际权重取值。a/b detach，不额外 backward/VJP。该关系不延伸到参数梯度、Adam 位移或状态已经分叉的轨迹。

新增臂不要求 post 与 C 一致；小模型另用独立目标公式对比参数、Adam、RNG 和输出。全部权重为 1 时数学退化为 BCE。完整随机网络每臂至少四访问。未来 GPU OLD/R6-C parity 继承 deterministic_smoke_pair、rtol=1e-4/atol=1e-5 和 R3 cuBLAS 边界；未进行 GPU 资格验收。
