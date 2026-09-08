# M4 execution report — smoke failed, formal experiment not started

Status: **M4_SMOKE_FAILED_STOPPED**. This is an engineering failure report, not `M4_TRAJECTORY_COMPARISON_COMPLETE` and not evidence that trajectory distillation is ineffective.

Execution commit: `956662e94cc4c2b66109acf4075652fd9f29aa55`.
Starting M3 release: `538a768363a79a5dca27de186e2b776c858c5e2b`.
Fixed external reference: `dbff0d985c6c95345d9fb78f5b1daef57b392564`.
The Git commit containing this report is a separate report release; it was not used to execute the failed smoke.

## 已完成与停止点

独立 M4 分支、无条件化四步 D4/L4/T4 训练实现、确定性 source 序列、两种域序评分与独立 CPU 重算入口已实现。M1–M3 科学代码、配置、历史记录和固定外部依赖未修改。部署前已冻结每任务 600 条 source 访问序列，并绑定新 receipt；未重建 Base 历史。

本地相关 CPU 测试 22/22 通过（1.445 秒），服务器相关 CPU 测试 22/22 通过（3.120 秒），无失败、错误或跳过。测试覆盖单步 M2 目标/梯度、冷启动和已有 memory 的四步数值、L4/T4 值一致性、可控跨步梯度、标签隔离、窗口 detach、120 次访问延续，以及原生 memory 的选择/覆盖/容量。CPU 程序化通过不能替代完整模型 GPU 前向验收。

GPU smoke 使用原始 Real 代理和程序化四图窗口，以原有 history index 16 为起点。Fundus D4/L4/T4 各完成四次功能化内层与一次外层更新。L4 与 T4 的四步前向/损失比较通过，两者窗口 loss 均为 5.218537330627441；image meta-gradient norm 分别为 0.1038665771484375 和 0.11489642411470413。这仅是机械 smoke 证据，不能作为源域或目标域效果。

随后原生 host 实际参考走到第二张图（history position 18）时，**D4 功能化预测与原生预测超出固定容差**：

- rtol = 1e-4，atol = 1e-5，未修改。
- 524,288 个元素中 1,673 个不满足比较（约 0.319%）。
- 不满足比较元素中的最大绝对差 0.0003333091735839844。
- 异常类型 AssertionError；smoke 退出码 1。

第一张图的三个臂与原生预测/prompt/Adam/memory/counter 比较已走完；第二张图在 D4 预测比较处停止，不能推断该位置 L4/T4 或后续位置也通过。Polyp GPU smoke 尚未执行。

## 实际计数和资源

| 项目 | 本次实际值 | 完整计划值（含 smoke） |
| --- | ---: | ---: |
| 真实 online Adam | 2 | 2,180 |
| outer 图像 Adam | 3 | 906 |
| 功能化 inner Adam | 12 | 3,624 |
| 正式 source 训练访问 | 0 | 3,600 |
| 新 source/target 评分记录 | 0 | 2,172 |

六套正式训练均为 NOT_STARTED；没有正式代理、源域/目标域 Dice、配对增量或 ASSD 结果。没有启动正式后台 launcher，未生成 formal run/recompute 退出码；这些值为 null，不能写为成功 0。CPU validation/registration 退出码为 0，GPU smoke 为 1。

使用 GPU 7，启动时可用显存 24,124 MiB，未修改其他进程。已记录 GPU smoke 阶段时间 23.107 秒。失败处理器未保存峰值 allocated memory，因此峰值为 unavailable；事后空闲显存不能替代峰值测量。NAS 挂载、容量和写入/读取探针在执行前通过。停止后运行目录为 13 个文件、322,182 字节（含执行 bundle 和私有登记，不含独立代码检出及只读旧资产）；确认 smoke 进程已退出且没有正式评分日志。运行使用既有 Python 3.10.6 / Torch 2.2.1+cu121，环境对比 M2 无差异；配对阶段按约定启用严格确定性，无容差调整。

## 差异定位的当前证据与边界

失败位于新增功能化轨迹对原生在线轨迹的数值验收，而非训练效果门槛、数据缺失、OOM 或模型性能负结果。现有日志没有保存失败位置全部中间张量，尚不能唯一确认根因。

源码中两个应优先核对的数值路径是：

1. `offline/trajectory_dd.py` 分别求 host/proxy prompt gradient 后相加；原生 image-step 对联合 loss 一次 backward。二者数学等价不保证 CUDA 浮点累加完全一致，微小更新差可通过后续 Adam/memory 放大。
2. 功能化 memory 使用原生 NumPy 邻居/权重，但 tensor prompt 值在设备上组合；原生值在 NumPy 中组合。CPU fixture 通过并不能单独证明真实 GPU 多步路径在固定 logit 容差内。

这两点是源码级候选，**未被本次证据证明为根因**。不能用“不确定性噪声”“只有零点几毫”的解释跳过失败，也不能删掉 memory 梯度、降成一步、改精度/分辨率或扩大容差来改写验收。

按执行计划第 10/15 节的实现/数值错误停止规则，本轮已停止，保留失败前缀；没有自动重跑 smoke、没有启动六套训练或额外诊断轨迹，也没有 M5。修复与重新 GPU 验收应作为明确的新续行范围，使用新干净执行提交和新的记录，不覆盖本次失败 receipt。

## 公开范围

公开提交包含 M4 源码、固定配置、测试、执行/重算入口、本失败报告、无效果结果的显式状态 JSON，以及去标识实际计数和错误摘要。原始数据/标注、路径和身份清单、封存代理、历史状态及私有 receipt 留在 NAS；未发布患者信息或模型资产。

[Execution audit](execution_audit.json) 保留可核对的 smoke 更新计数、CPU 结果和环境比较。[Public aggregate status](public_aggregate.json) 明确 source/target/training 结果为空，不伪造实验指标。
