# P2 冻结方法全量连续流：启动报告

状态：**P2_RUNNING**。本文件是启动时快照，不是实验完成或方法有效性的声明。
快照时间：2026-09-09T04:55:02.577Z。

执行提交：`d3ee6901379be293f47abd1687808caaf5b04266`。基线发布：`18b8f32186678e8930a6c540f96941365fd511b9`。
P1 执行提交：`f65e119016f2d6a32cd0cffbee2bb2e94f570c2d`。
固定外部依赖：`DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`。
独立分支：`experiment/p2-frozen-factorial-full-stream-v1`。

## 冻结范围与实际登记

七输出 N / A / EA / O2 / EO2 / D4 / ED4；仅 A、O2、D4 各自在线更新。
O2-600 与 D4-150 沿用原封存代理。固定概率平均权重 0.5、阈值 >=0.5；K=4、proxy weight=0.1、boundary=0。
逐当前图像共享一次 source 前向，三父轨迹独立保存/恢复 RNG，七预测固定后才读取标签评分。
每个任务的两个域序各自从 source 和空 history 开始，跨域连续适配，顺序 1 仅反转域块。

| 任务 | 域 | legacy_dev | p1_extension_dev | remaining_dev | all_dev |
|---|---|---:|---:|---:|---:|
| fundus | REFUGE | 32 | 32 | 336 | 400 |
| fundus | ORIGA | 32 | 32 | 586 | 650 |
| fundus | REFUGE_Valid | 32 | 32 | 736 | 800 |
| fundus | Drishti_GS | 32 | 32 | 37 | 101 |
| polyp | CVC-ClinicDB | 32 | 32 | 548 | 612 |
| polyp | ETIS-LaribPolypDB | 32 | 32 | 126 | 190 |
| polyp | Kvasir-SEG | 32 | 32 | 936 | 1000 |
| 合计 | 七域 | 224 | 224 | 3305 | 3753 |

旧 P1 的 448 组全部保留，missing_P1_groups=0。按既有规则排除 3 个 mask 冲突组。
复用 896 个 P1 文件的既有校验记录及大小/mtime 检查，新选中内容验证 6610 个文件。
remaining_dev 是主要内容子集；所有内容仍为探索性开发池，不能称为全球未见或患者独立盲评。
子集只用于完整流结束后的分层统计，不分子集重置或复用 P1 分数。

## 预算与验收

正式计划：3753 组 × 两序 × 七输出 = **52542 条评分**；三父轨迹共 **22518 次 online Adam**；
共享 source 前向 **7506 次**。另有已完成 smoke **12 次**，总 online 预算 22530。
新 source/proxy/selector 训练、outer Adam、可微训练 inner 均为 0。
资源配额为 GPU 活跃阶段最多 6 小时、新私有输出最多 1 GiB；这不是预计耗时。

六项相关 CPU 测试本地及服务器通过；服务器 0 failures / 0 errors / 0 skips。
一次 GPU smoke 已通过：两任务 × A/O2/D4 × 原入口/P2 入口，共 12 次更新。
六个比较的最大 logits 差均为 0，状态/RNG 比较通过，teacher 保持冻结。
smoke 用时 19.939 秒；Fundus/Polyp 峰值 allocated 为 3631961600 / 2314573824 bytes。
该峰值是 smoke 的分阶段轨迹测量，不代表正式三个父 host 同时驻留的峰值。
正式运行保留 P1 backend 设置，不将 smoke 的严格确定性设置传播到正式实验。

## 已确认的启动状态

物理 GPU 7；启动前可用显存 24124 MiB，准入要求 12288 MiB。NAS 挂载、容量和小型写读探针通过。
可靠后台 launcher PID 439239，不依赖 SSH/Codex 会话持续连接。
快照时 Fundus/order0 的七个输出各有 14 条可读记录，三个父方法 Adam step 均为 14；
平均视图记录父 step，但自身不额外更新。N 的 step 为 0。
运行环境与 P1 一致，未见立即失败。此结论仅覆盖该启动快照。

后台按 run → 独立 CPU recompute 顺序执行；任一阶段非零退出即停止，保留前缀，不自动重试。
完整结果与 16 格计数复算、覆盖/身份/顺序/计数校验尚未完成，最终 paired differences、交互差、ASSD 和分布指标均待定。
不得据此写 P2_FROZEN_FULL_STREAM_COMPLETE 或宣称存在收益。

## 成本口径与公开边界

共享研究调度的前向计数按实际物理执行计；独立部署的融合方法仍承担额外 source 前向及驻留模型成本。
pipeline_elapsed_seconds 是本次访问开始至相应输出评分后的时间，不能跨输出相加当作 wall time。
独立部署 step 时间是父方法时间加其必要 source 时间，不包含 evaluator。
不构造 OD/OC macro ASSD，不把未定义 ASSD 填零，不把两个域序当成独立患者。
ASSD 汇总沿用逐图标量，16 格计数用于重建 Dice 与前景/背景互补关系。

本次公开源代码、冻结配置、协议和去标识的启动审计；没有发布私有 registration、逐图记录、资产路径、图像、mask、目标概率图或模型权重。
完整实验结果将于后续确认完成后单独汇总交付；本轮停止后不扩展 P3。

