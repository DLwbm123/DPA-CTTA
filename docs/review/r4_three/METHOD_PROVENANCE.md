# 方法来源与实现变更

本批实现用户 R4T 提案，不是论文完整复现或已验证有效的方法。来源全文和配方见 [主计划](input/R4T_COMBINED_EXPERIMENT_PLAN.md) §11；原始包、继承计划、数学参考及其测试记录保留在 input，不能把提供者的参考测试当作本实现测试。

- A：Mean Teacher/CoTTA 是教师时间尺度的概念来源；本方案仅 BN affine EMA，m=.99，无 warm-up。FT 是初始 affine 加 C 统计政策，非源 BN 推理。RP 原样复用项目 R1 region_memory 的坐标、采样、绝对重建与累计统计。
- B：受 PAID 参数几何思想启发的项目 KDG 和匹配控制，不使用该论文源统计或源数据，不宣称完整 PAID。选定 kernel 的 DC/detail 与角度性质是局部代数性质，不是整网边界或遗忘保证。
- C：直接改造原 R3 第五条 G 的图教师接口；SPEGC 与 seeded graph/random walker 仅为图监督/图传播来源。没有其完整提示池、图聚类或 OT 稀疏化；soft 自伪标签锚点不等于人工正确 seed。

## 实际实现差异

1. 原十臂科学字典、teacher/student/RP/kernel/resources 块与继承 R4D 完全相同；指定基础提交没有 R4/R4D 代码；既定作业目录未发现其真实运行。A/B 首次在此接入，新 G 与同一 70-job 入口一起交付。
2. boundary_graph.py 与包内数学参考逐字节相同。host 只传 raw RGB、q、六概率图、臂和 visit，不传 GT/域/样本 ID。
3. KernelGeometry 的计算与参考相同，DC reduction 将常数基提出求和，A0=sum(W0)/sqrt(K)，修复浮点加权抵消产生极小非零源 DC 的问题。没有 eps 源阈值、clipping、伪造方向或零态导数断路。增加实际 DC/effective/detail 位移及 flattened cosine Gram 审计。
4. 有效卷积 forward 挂在运行时模型实例的两个固定模块上，原模型依赖源码、源 .weight 及其他原参数不变。R4 新坐标单独列入同一 Adam，不能把“源参数未改”写成“函数未改”。
5. ObservedMemory 复用原 prepare，并用同一局部 seed 重建精确写入位置供延后评价；与旧 C/RP 的完整两步状态和输出逐值一致。无额外 shadow bank。
6. 复用 R1 IO、监督器、原子 evidence、R3 流及阶段 cuBLAS 修复；新 namespace 拥有自身 14 臂配置/授权/分析器，未修改旧 R3 或 R4D 常量。

科学条件固定在 configs/r4t_science_v1.json，保持提案原字节。只读给定 checkpoint 的未来执行权限不等于读取源数据、训练源模型或获取新模型的权限。
