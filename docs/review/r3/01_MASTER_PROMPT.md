# 交给 Codex：R3 五框架阶段 I 实现任务

## 0. 本次授权与结束点

你要在 `DLwbm123/DPA-CTTA` 中实现 `R3_FIVE_REGION_FRAMEWORKS`。当前只授权**代码实现、程序化 CPU 测试、元数据 dry-run、去身份审阅材料及 GitHub 新分支发布**。

当前不得启动 GPU、读取真实目标 RGB/mask、运行正式实验、启动任何待批准后自行运行的后台任务。允许从现有私有配置解析路径、读登记元数据；可只读加载已给定 checkpoint 到 CPU 配合程序化输入，不访问源图像、源 mask、source query、源代理或源原型。

完成全部代码后，以 `R3_IMPLEMENTATION_READY_FOR_REVIEW` 停止。不得自行填写外部 review 通过、不得把本文当作 GPU 执行授权。阶段 II 模板在 `05_STAGE_II_AFTER_REVIEW.md`，必须等外部审阅固定提交后使用。

## 1. 研究目的与不变约束

目标是在 C-CTTA 上，一次比较区域 PCA 的五种不同职责：

1. `T_LR`：历史区域密度修正当前软目标；
2. `U_PCA`：历史区域方向约束真实 Adam 位移；
3. `S_JOINT`：联合保存／检索 BN affine、Adam 和区域 memory；
4. `M_TRANSPORT`：将历史统计迁移到当前表示坐标；
5. `G_PCA`：区域子空间决定局部图传播几何。

这对应上一轮候选编号的 **1、3、4、2、5**，顺序仅表示实现优先级，不表示后面的框架必须等待前面的分数通过。五条路线分别实现，不拼成一个大 loss，不进行全部模块组合。

只给定现成源 checkpoint；不改变源训练，不获取源数据。源权重副本可以作为只读测量器，但不得使用其外部源统计、样本或预先未交付的原型。目标标签只由 evaluator 在预测与在线状态固定后读取。

## 2. 工作分支与已有代码

从固定提交 `f52f132e4be576ea871467432a6f4b12337209a5` 建立独立分支，例如 `experiment/r3-five-region-frameworks-v1`。不要使用旧 main，不 force-push，不改历史报告。

先读现有：

- `src/dpa_ctta/b1_host.py`：C 参数政策、GraTa 固定依赖与增强；
- `src/dpa_ctta/r1/host.py`：弱视图、f0/fs 缓存和因果 memory 时序；
- `src/dpa_ctta/r1/region_memory.py`、`streaming_pca.py`：grid、区域、采样和累计统计；
- `src/dpa_ctta/r2/`：薄 runner、分析器和矩阵；
- `src/dpa_ctta/r1/supervise.py`、`evidence.py` 及实际读写辅助模块；
- `docs/review/r2_io_continuation/LIVE_DIRECTORY_REPAIR.md`；
- 现有 Fundus registration、R1/R2 science 和结果绑定。

继承已解决的资产字节核验、mask 后读、失败输出失效、信号／进程收尾，以及 live-directory ENOENT 枚举竞态处理。不要重写整个审计系统。R2 有实际 IO 续跑：新实现必须继承最后修补，不把“必要证据文件缺失”误当成可以跳过的容量扫描竞态。

旧代码尽量不动。新增 `src/dpa_ctta/r3/` 与薄命令行脚本。不要 monkey-patch R1/R2 的全局 arm 列表或 SHA 常量；通用函数需参数化时做小型、向后兼容的改动并附相关回归。

## 3. 实现顺序

先完成共同 C adapter、snapshot、统计和 trace 接口，再按 `T → U → S → M → G` 实现。每条路线同时实现必要控制，不允许只实现主候选。

全部正式臂严格为：

```
C, RP,
T_LR, T_ISO, T_DIAG,
U_PCA, U_RAND, U_SCALE,
S_JOINT, S_SHARED, S_NOPCA,
M_TRANSPORT, M_IDPOST, M_SHUFFLE,
G_PCA, G_ISO, G_ORDER
```

`RP` 是当前批次重新执行的原 R1 `C_PCA_REGION`，不是 R2_E。

算法事实以 `02_METHOD_SPEC.md` 和 `r3_science_proposal.json` 为准；`reference/kernels.py` 是可执行参考，不是完整宿主。如果发现正文、JSON、核函数有不一致，先完成不受影响部分，在交付中列出差异和建议，不能静默选择更有利的配置，也不能整轮搁置等待无关材料。

## 4. CPU 工作及必要检查

先运行包内参考测试，再写真实 host 集成测试；保留现有仍适用的 R1/R2 数学、源参数、标签隔离和 IO 回归。测试名称和数量由实际 coverage 决定，不为凑数量复制测试。

至少覆盖：

- 关闭所有新分支时逐步对齐原 C；RP 对齐旧 REGION；相同 seed 的增强与局部 RNG 隔离。
- T 的三种协方差、低秩密度与显式稠密参考、bank 未 ready 回退、source 测量器确实不变。
- U 的 Jacobian 与有限差分、两个参数空间维数正确、真实 Adam 一次 proposal、投影位移／范数匹配／moments 语义。
- S 的多状态独立性、只有一个 active slot、容量上限、slot 计数与全局计数分离、切换不重播随机增强、模型副本不随 slot 增长。
- M 的同位置前后配对、正交解、均值／M2／缓存快照一起迁移、POST 身份控制和 shuffle 控制。
- G 的边数、KL 方向、数值稳定、单调包含投影、零边退化为 ORDER、无图反向、空间索引对齐。
- 当前图不进入自己的统计；真实图像和 mask 不进入阶段 I；没有未来图像、域名或 GT 进入算法。
- 五条路线及控制的 cold/ready CPU 分支实际调用，不仅 import 成功；trace 计数可以核对。
- 85 个 job 无重复无遗漏；新增块交错流每个内容只出现一次，域内顺序不变；原四序不改。

不能通过更换 seed、阈值、半衰期、rank 或缩短序列修补性能。数学错误和真正工程错误可以修，必须记录。

## 5. 日志与物理计数

明确区分 `network_forwards`、`loss_backward_calls`、`jacobian_vjp_calls`、`adam_calls`、`actual_parameter_replacements`。U 三臂的额外 VJP 不得被藏在“一次更新”的表述里。S 的状态装载不是 Adam；M 的 SVD 与 G 的图迭代不是模型 forward，但都要计时。

只保存有界状态与标量；不保存原始目标图像、mask、概率图、dense feature、Jacobian 数组或患者身份到公共目录。在线当前图 tensor 在本步内暂存是允许的；不能跨步保存 replay queue。

## 6. 发布材料

按 `04_REVIEW_RETURN_CHECKLIST.md` 生成可审阅的代码提交和发布材料。必须有完整 patch、实际 CPU 日志（包括首次失败记录及原因）、science 字节 SHA、registration 不变说明、新 stream digest、17 臂及 85 job dry-run、状态生命周期、方法来源和明确未运行项。

把源代码与必要去身份材料推送到新公开分支；可用单独的后续文档提交发布报告。固定 implementation SHA 与结果文档 SHA 分开。确认远程 ref 和匿名可读后给出链接。不上传论文 PDF、checkpoint、实际私有路径、GPU UUID 或原始逐样本记录。

最终只报告：实现范围、实现 SHA、科学配置 SHA、真实测试结果、残留问题、审阅索引、零 GPU／零正式数据读取状态。然后停止。

**禁止在本阶段输出 `R3_REVIEW_PASS`，禁止自动进入 GPU smoke 或正式执行。**
