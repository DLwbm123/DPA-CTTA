# R7 GPU-first TARGET_SCREEN 交接文档

更新时间：2026-09-23（Asia/Shanghai）

## 当前结论

GPU-first TARGET_SCREEN 的完整 24-job 矩阵已经完成，当前只做了只读科学整理，没有启动新模型、追加实验、重新评分原始图像、加载 checkpoint 或解码新的 target image/mask。

- 24/24 jobs 完成；每 job 1,951 arrivals、1,695 scored contents。
- 冻结物理计数：122,913 forwards、5,853 backwards、5,853 Adam。
- completion、terminal、counts、worker cost、scalars、report 均已读取并整理。
- 自动提名关闭；没有科学 winner。
- execution-layer review 保持 `USER_WAIVED`；不得写成 external PASS。
- GPU qualification 仅保留内部资格证据；不得写成外部 GPU PASS。
- 旧 CPU 中断和旧资格测试均保留，不能改写成 PASS，也不能混入本轮 GPU wall time。

## 固定身份

- 执行代码 SHA：`c1763d00f4f29de1c168ccd75219199667478825`
- 固定完成发布 SHA：`09eb77691c54d9942edb6691b4c26b46316b656a`
- 本次只读聚合材料发布 SHA：`f4adb85434e5ef36faecc1109f907d7a2f29d17f`
- GitHub branch：`review/r7-target-gpu-cost-fix-v1`

## 公开结果入口

仓库内：

- `docs/review/r7_target_screen/results_gpu_first/RESULTS_PUBLIC.md`
- `docs/review/r7_target_screen/results_gpu_first/RESULTS_PUBLIC.json`
- `docs/review/r7_target_screen/REVIEW_INDEX.md`
- `docs/review/r7_target_screen/DELIVERY.json`
- `docs/review/r7_target_screen/GPU_FIRST_COMPLETION_PUBLIC.md`

`RESULTS_PUBLIC.json` 包含：

- 24 个 arm/order 组合的绝对结果；
- 96 条 domain/order 的 OD、OC、macro Dice；
- FULL − 自身 STATIC、FULL − `C_BASE_GPU_FP32_V1`、FULL − C0；
- order 4 recurrence 的独立记录；
- 配对 tail quantiles、negative/positive/zero、ASSD common-valid/undefined/mean delta；
- 最差 domain/order；
- 24 个 job 的在线耗时、终态、物理调用；
- 三条 worker lane 的 job 数、order 分配、物理调用及已记录 online wall sum；
- 无现成计时处统一写 `NOT_MEASURED`。

公开材料不包含患者/内容标识、私有路径、receipt ID、权重、checkpoint 或完整私有账本。

## SOURCE_PREP 身份摘要

六份 SOURCE_PREP artifact 的公开身份摘要已写入 `RESULTS_PUBLIC.json`：

- inventory SHA256：`4ffba842009617625e0ad44de6158d07d4ce4773d08982b66071f61081fd2c3e`
- manifest SHA256：`39d429baceb015dbf99948bb04c9a2bfca4d087c9f452a07be9528f6ecb046d1`
- split SHA256：`8fdb8c869319c086048a36047d184c9844a986bd52ad098ab265158acd50deb9`
- artifact count：6
- file identity：`BOUND`
- trusted loader：`true`
- source status：`USER_ACCEPTED_VERIFIED_ARTIFACTS`
- external review：未记录为 external PASS

## 成本边界

本轮 GPU-first 的启动时间有记录；精确 execution end、端到端 wall timer、评分耗时、IO/审计耗时、lane overlap 未形成可公开的独立计时，因此结果中保留为 `NOT_MEASURED`。不得用文件 mtime 冒充精确 execution timer，也不得据此声称完整 CPU 矩阵加速倍数。

历史 CPU 中断单独保留：4,335.491 worker seconds、4,136 forwards、2,067 backward、2,067 Adam；它不是成功 SOURCE_PREP，也不属于本轮 GPU-first wall time。

## 运行和实验边界

继续处理本任务时：

1. 不自动重跑 TARGET_SCREEN、SOURCE_PREP、MECHANISM 或 EXTENSION。
2. 不恢复旧暂停/失败运行，不 retry/resume，不减少预算或改 seed/order/算法。
3. 不读取新的真实 target pixels、RGB/mask，不加载 checkpoint，不重新评分原始图像。
4. 不把 `USER_WAIVED`、内部 qualification 或历史测试写成 external PASS。
5. 不按结果删除不利方法、域、order 或失败记录。
6. order 4 是 recurrence，不能当成独立患者重复。
7. 当前结果只支持描述性科学审阅，不支持自动选 winner 或机制因果结论。
8. 用户曾要求移除长期监测；`r7` automation 已确认不存在，不应重新创建 heartbeat。

## 如用户开启新训练/实验

必须把它视为新 scope，先重新确认：

- 新实验目的和比较对象；
- 是否允许真实 target/model/checkpoint 访问；
- exact code SHA、artifact/context identity、device/resource、output root；
- 预算、seed、orders、停止条件和失败策略；
- 是否需要新的 external execution review；
- 新结果是否另建目录、receipt 和发布 SHA。

在这些条件明确前，不能直接开始训练。
