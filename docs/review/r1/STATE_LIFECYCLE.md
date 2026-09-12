# R1 状态生命周期

Host 只接当前 RGB tensor；身份、域名、数据路径、mask 均由外部 runner 持有。各臂/序独立模型、Adam、随机流、控制器与 bank；唯一共享是只读 checkpoint 和登记元数据。

```mermaid
sequenceDiagram
    participant R as Runner
    participant H as Host
    participant M as Past PCA snapshot
    participant E as Evaluator
    R->>H: current RGB only
    H->>H: advance global visit; isolated probes if SENS
    H->>H: optional restore before teacher
    M-->>H: detached center/U from earlier visit
    H->>H: six separate weak views; q + pre-update f0
    H->>H: strong input; BCE + optional subloss
    H->>H: one backward + Adam
    H->>H: original final prediction, detach
    H->>M: merge cached sampled f0 after prediction
    H-->>R: detached logits + scalars; clear temporaries
    R->>E: only now read current target mask
```

| 状态 | C | PER256 / SENS | 三个 PCA 臂 |
|---|---|---|---|
| 全局visit/累计Adam | 每图+1 | 恢复不重置 | 每图+1 |
| BN affine/Adam | 连续更新 | 恢复初始affine，原对象不替换；清空moments/step/grad，固定lr/betas；当前图重新做完整C | 连续更新 |
| BN统计 | 当前输入，track=False，mean/var=None | 恢复后仍是C当前统计 | 同C |
| 周期age | 等于当前Adam age | optimizer_steps_since_reset与global_visit分开；257、513…当前teacher之前恢复 | 连续 |
| 强增强RNG | 私有snapshot连续 | 不恢复seed；探测私有CPU generator不消耗它 | 采样/shuffle私有seed不消耗它 |
| SENS趋势 | 无 | 有效s递推；触发记录保留触发前age/ema/best，控制器内部清空，下一图从age1重建；零触发合法 | 无 |
| PCA | 无 | 无 | n/mean/M2/m固定维度；每16个贡献图刷新；center/U/spectrum/version同时快照 |

每 bank 只保留 d=32 的 CPU float64均值/M2和rank≤8快照/谱及标量计数；四 bank 数值状态低于1MiB。无可靠或全零输入不贡献图像数。SHUFFLED在排除零向量后按自身REGION的相同输入数量重排；损失分配对全部伪分区token独立重排，使用另一盐值。

当前时刻先复制旧快照，再建立强损失，最终预测后合并；逐日更新mean不会修改mu_basis。快照version必须小于global_visit；eigh异常直接抛出并停止该轨迹，不静默跳过。基向量符号任意，测试比较projector。

临时保留只限当前步：六视图预测（继承原C）、1024×32原图采样向量、强采样特征的计算图、≤128个选中token、快照副本。finally清空pending。跨步不保留RGB/mask/dense feature/logits/样本向量列表，也没有replay。

恢复计数reset_count是R1恢复；物理counts中的perturb/restore是旧GraTa扰动操作，均为0。恢复自身不算Adam/网络调用。SENS三个探测不与C视图复用，所以每图11前向；其他臂8前向。

执行入口默认dry-run。只有另行提供真实用户授权文件、完整code SHA、science摘要、registration摘要和显式GPU列表，才允许进入未来Stage II。最多3独立worker，每卡1个本作业worker；不等待审批、不重试、不自动修复续跑。每卡smoke后的正式部分使用新进程；失败前缀保留，矩阵不标完成。资源上限不是效果gate。
