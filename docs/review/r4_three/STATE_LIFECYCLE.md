# R4T 状态生命周期

## 共同顺序与信息边界

每条轨迹从同一给定 checkpoint 独立初始化，固定 seed 20260907。每图接收单张已 resize 的 raw RGB；host 签名不接受域、ID、GT、未来图或评价结果。原数据读取/逐内容核验和 mask 映射不变。

1. 读取当前 RGB；读取过去 RP snapshot（若存在）。
2. 六弱视图概率均值 q；当前图强增强的 RNG 顺序沿用 C。
3. RP 使用本教师的 q/六视图选位置，但准备写入学生 pre 原图特征；MT_RP/FT_RP 额外一次学生前向。
4. G 只在本图的 CPU float64 128 网格上求解；q* detach 后用于原 full-image BCE。
5. 一次 loss backward、一次 Adam；K 的 BN 与坐标在同一个 Adam 中。
6. 学生更新后的原图预测固定。
7. 合并 RP 的 pre 特征；然后 MT 更新 teacher affine，FT 不更新。host 清空内部当前特征，phase=IDLE。
8. take_evaluation 单向移出当前 q/q*/位置/支持 mask；evaluator 此时才读取当前 GT。完成主/辅助指标后清空临时 payload，落盘只有标量。

若上一张的评价 payload 尚未移出，下一次 step 拒绝；失败的 host 不能原地续跑。任何已有状态都不依赖当前评价返回值。

## 各状态的所有权

| 状态 | 所有者 | 初始化及更新 | 跨图保留 |
|---|---|---|---|
| 底层源卷积/非 BN 参数 | 学生 | 固定给定 checkpoint；版本及结束时逐张量检查 | 冻结 |
| BN affine/Adam | 学生 | 原 C 配置；每图一次更新 | 是，无域 reset |
| MT affine | 无梯度教师 | ψ0；当前最终输出与 memory commit 后 .99/.01 EMA | 是 |
| FT affine | 无梯度教师 | ψ0，C 当前输入统计政策；无 optimizer | 冻结 |
| RP bank | 三个 RP 臂 | 原 R1 REGION、累计 Welford、rank≤8，旧快照先于当前 merge | 是，四 bank |
| K 坐标 u/a/E | 对应两层 adapter | 全零、float32 参数；小矩阵 float64；同一 Adam | 是 |
| K 有效 weight | 对应卷积前向 | 从只读源 weight 与当前坐标生成，不改 source .weight | 不缓存历史 |
| 原始 q/六视图、G 边/自由域/q* | 当前图 | 完全 detach；G 无额外可学习参数 | 否 |
| 评价 GT/完整概率 | 独立 evaluator | 仅在全部在线状态 commit 后 | 否，只存标量 |
| RNG | 轨迹 | 继承 C 增强；克隆教师不改 RNG，图 shuffle 使用局部 generator | 是 |

当前教师 C/RP 直接复用旧 host；A 的其他臂没有 source-BN N 的替换。K 零初始化只保证更新前函数相同，生产代码保留新增坐标梯度。G 的固定支持只约束教师，不能声称最终学生变化仅在边界。

## 调度与失败生命周期

默认配置 disabled。实现/科学字节 SHA、原 registration/recurrence、70-job 计划和设备列表都绑定到私有授权与 run receipt。external pass 与明确用户 waiver 是不同字段值，不能互相冒充；公开代码提供该显式分支，不生成实际批准。

每个实际设备一次 smoke，成功后原有限监督器执行固定轮换队列，每设备最多一条轨迹，最多三 worker。cuBLAS smoke/formal 环境策略直接复用 R3 修复函数；GPU 进程审计仍在现有模型分配后、首个 forward 前。进程命令用中性入口，真实路径只在环境/私有配置中。

共享监督器保留自有进程组管理、SIGTERM/SIGKILL 回收、失败清理先于日志、NFS 枚举后 ENOENT 容忍及其他 IO 错误传播。无效果 gate、自动 retry、续跑或后台等待批准。

全部 70 completion、全部进程退出和逐记录 CPU 标量核验通过，才能原子发布 R4T_EXPERIMENT_COMPLETE；任何缺失/腐败都会使当前结果 valid=false。四主序与回访流始终分开，下一轮执行授权始终 false。
