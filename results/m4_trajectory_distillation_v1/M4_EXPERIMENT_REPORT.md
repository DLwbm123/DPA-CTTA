# M4 repair report — numerical acceptance passed, formal experiment running

Status: **M4_RUNNING**, verified at 2026-09-08T23:30:14+08:00. The formal experiment is not complete; no method-effectiveness conclusion is available.

Execution commit: `f88e99d212e914b74a7c521bac8c418fc73e5d6c`. The report release is a separate commit. Baseline M3 release: `538a768363a79a5dca27de186e2b776c858c5e2b`; pinned external implementation: `dbff0d985c6c95345d9fb78f5b1daef57b392564`.

## 修复结论

最初失败来自功能化轨迹与原生数值计算的差异；修复后暴露了 Polyp T4 的标准差二阶导数问题。本次确认并修复三处：

1. Prompt 梯度采用原生 host + 0.1 × proxy 联合 loss 一次求导，消除分开求导后相加的浮点累加差异。
2. M4 功能化 Adam 对齐实际 CUDA foreach 分派及运算顺序。第三步原先出现一个 float32 ULP 的 prompt 差异，会经后续模型前向放大。原生在线 Adam 与 M1–M3 实现保持不变。
3. 离线 AdaBN 的空间标准差使用原生前向及原生一阶导数运算顺序，并对二阶导数采用等价的解析 Hessian-vector product。PyTorch 2.2.1 的除法二阶反传在 std=0 或极小正值（诊断最小约 3.743392e-23）处产生非有限中间值。新计算避免 std 平方/立方分母；没有 epsilon、正值截断或精度更换。std 恰为 0 时明确采用零 Hessian 延拓；该点没有经典二阶导数，此约定是方法实现的公开边界。[原生 std_backward](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/csrc/autograd/FunctionsManual.cpp#L1724)

Memory 邻居选择、权重和组合未因本次故障改变；诊断不支持把它认定为根因。完整 host Hessian、窗口内 Adam/memory 梯度和既有 BN stop-gradient 语义均保留。没有修改数据、种子、K=4、分辨率、窗口长度、容差或效果门槛。

## 验收证据

本地与服务器均通过 26 项相关 CPU 检查，无失败、错误或跳过。覆盖原生 Adam 运算顺序、四步冷/热轨迹、标签隔离、跨步梯度、状态截断，以及标准差的一阶精确性、零通道二阶约定、正常输入 gradgradcheck、极小 float32 输入的有限 Hessian 与 double 参考比较。服务器沿用 Python 3.10.6 / Torch 2.2.1+cu121，环境比较 M2 无差异。

完整模型 GPU smoke 已完成 Fundus、Polyp × D4/L4/T4 × 四步：原生参考预测最大绝对差均为 **0.0**；初始 prompt、梯度、更新后 prompt 也均精确一致，Adam moments、memory、counter 通过原有固定容差/离散比较。L4/T4 前向轨迹一致；全部六个 meta-gradient 有限且非零，外层更新有效。实际 smoke 为 **8 online / 6 outer / 24 inner**，退出码 0。机械验收损失与梯度不作为源域/目标域效果指标。

## 已启动的正式实验

使用 GPU 3，允许与已有进程共存。正式启动前空闲 23,238 MiB；成功 smoke 峰值 14,301,518,336 bytes，对应加 25% 与 512 MiB 余量的准入要求 17,561 MiB。GPU 4–7 当次空闲不足要求，未启动其他副本。遵守原计划单卡执行六套训练。

后台 launcher 已启动且首次检查存活，日志可读；Fundus D4 已完成第 3 个窗口（12 次 source 访问），梯度有限，没有即时失败。随后按固定顺序执行六套各 600 次 source 访问 / 150 次 outer 更新的训练，再生成 2,172 条新评分，并执行独立 CPU 重算。当前 formal run/recompute 退出码为 null，训练结果及 source/target 指标保持 pending。

运行不依赖 SSH 或当前会话持续开启。阶段失败会保存错误并停止，不自动重跑；未创建持续监测或额外实验。

## 修复前缀与预算

用户明确授权修复和重新验收；保留原始失败及 R1–R3 的独立提交、目录与 receipt，没有覆盖旧失败。原始失败报告保存在 [历史提交](https://github.com/DLwbm123/DPA-CTTA/blob/51c5b9cc80eb0ff437f63e0d7d2bf0e88f50a0f7/results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md)。

| 阶段 | Online | Outer | Inner |
| --- | ---: | ---: | ---: |
| 原始失败、R1–R3 失败及图像诊断合计 | 17 | 16 | 84 |
| 本次通过的 smoke | 8 | 6 | 24 |
| 正式实验计划（未完成） | 2172 | 900 | 3600 |

另外有两次 75 元素 CUDA Adam 算术检查，无图像或模型调用；未单独计时，单列保守 60 秒预算扣除。成功 smoke 结束时累计预算扣除 267.739 秒，其中已测量阶段时间 207.739 秒。历史前缀文件累计 51,406,714 bytes；这些时间和空间继续计入原六小时 / 2 GiB 上限。实际失败前缀不伪装成计划内成功步骤。

## 公开范围与后续结果

本提交公开 M4 修复源码、配置、测试、数值证据及启动审核；[execution audit](execution_audit.json) 与 [repair evidence](repair_parity_evidence.json) 给出详细计数。公开 aggregate 明确标记运行中，不填造效果指标。原始图像/标注、模型、合成代理、历史张量、身份与路径清单、私有 receipt 均留在 NAS。

完成后仍需校验全部覆盖、独立重算和公开结果交付，才可写 `M4_TRAJECTORY_COMPARISON_COMPLETE`。单 seed、已暴露 target-dev、未知 patient/video 分组等原有限制不变。
