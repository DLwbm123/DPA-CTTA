# 服务器原运行环境 CPU 集成检查

**38/38 通过，0 失败、0 错误、0 跳过，226.17 秒。**

- 完整 SHA：`36ae77f7629150b244a57c1a888c7577b38e7f97`；独立 detached checkout，检查前后均干净，与公开实现相同。
- 实际 Python **3.10.6**、Torch **2.2.1+cu121**。CUDA 未初始化；真实目标 RGB、mask、登记 checkpoint、源数据读取数均为 0。
- 包含 12 项 kernel、15 项 graph、7 项 host、4 项 execution 检查。它们与本地完整套件中的 38 项重叠，不作为额外独立实验/seed。
- 随机初始化完整 ResUNet 的实际 CPU 物理计数：**260 network forward、32 loss backward、32 Adam、0 VJP**。同设备比较上下文 deterministic=true / warn_only=false，CPU cuBLAS workspace 未设置，结束恢复原策略。
- 此次是前台程序化 CPU 集成检查。没有 GPU smoke、真实 checkpoint 加载、目标像素读取或正式后台队列。启动后 ps 验证了中性命令行。

[完整日志](logs/cpu-server-01.log)、[状态 JSON](logs/cpu-server-01.json)、[完整模型 traces](SERVER_FULL_MODEL_CPU_TRACES.json)。服务器绝对路径已在公开日志中替换，原始日志私有保留。
