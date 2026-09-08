# M3 实验启动报告

**M3_RUNNING：实现和完整 smoke 已通过，正式训练已后台启动。尚无方法效果结论。**

执行提交：`00d1da5b3a333e53b20d869ac9fcbbdcb8b67a8f`。从 M2 发布提交 `bc9d622` 新建 `experiment/m3-conditioned-proxy-v1`，旧代码、配置和结果保持。

- 本地和部署环境各 31 项相关测试通过，退出码 0。历史完整回归仅复用。
- 单次 GPU smoke 退出码 0，严格执行 16 在线 / 4 外层 / 2 functional inner。
- Fundus、Polyp 的零强度路径及完整状态比较通过。O3 functional/实际 Adam 的最大 prompt 差分别为 1.1920929e-7 和 5.9604645e-8，原容差不变；counter、step、RNG 检查通过。
- 四个新臂的 conditioning/Adam/push 生命周期、固定代理和 mask 不变、标签隔离检查通过。O3 对照使用 outer 更新前的同一 Real RGB，复用那一次 functional 结果。
- Smoke 耗时 31.328 秒；Fundus/Polyp 峰值分配分别为 5,743,882,752 / 5,233,503,232 字节。
- 两份 M2 episode 清单、两个 32-state 历史库直接复用；O2T 的两份 O2-600 文件与 M2 封存摘要对应，无历史重建、无额外目标选择。
- 正式阶段：Fundus D3→O3→Polyp D3→O3，各 600 步；全部冻结后统一评分 R3/D3/O3/O2T。预定新评分 1,104 条，复用 1,932 条，联合展示 3,036 条。
- 条件化严格使用 rho=.5、beta=.01、eps=1e-8；原 seed、学习率、K、mask、Adam、AdaBN 和指标保持。严格 deterministic 只用于 smoke 配对检查，正式阶段使用独立进程恢复 M2 环境。

后台执行为一次有限任务，训练/评分后独立 CPU 重算并生成最终报告，任一步失败即停止，无重启循环或 M4。此前通用 9,000 MiB 显存阈值未通过时没有执行正式更新；按完整 smoke 实测峰值加 25% 与 512 MiB 余量计算需求为 7,360 MiB，实际空闲 8,536 MiB，符合共卡运行条件。

一次启动检查确认 Fundus D3 已完成 41/600 步，loss=0.87157452，梯度范数=0.29154471，裁剪比例=0.00266075，T 前后 L2=93.60167694；无失败记录，正式环境差异为空。此为启动时快照，不是最终结果。

O3−D3、O3−R3、O3−O2T、O2T−O2、R3−R、O3−A/N 和 D3−D2/O3−O2 全部等待正式结果。不得将非零梯度或 smoke 通过当作方法有效性。固定 T 仅是 FDA 式部分混合变体；代理风格匹配已有先例，详见 [实现说明及引用](../../docs/M3_IMPLEMENTATION_REPORT.md)。

[执行证据](execution_audit.json)记录具体计数和数值检查。私有 source 数据、mask、代理张量、history、身份映射和原始日志不公开。
