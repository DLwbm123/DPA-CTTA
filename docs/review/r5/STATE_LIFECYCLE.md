# R5 状态生命周期

当前为阶段 I。C 的真实候选链：R4 C → R1 Host('C') → B1 Host.normalized_step → 固定 GraTa.cal_consis_loss → 原 Adam → 原图最终前向。R5 构造相同 B1 C 并调用同一个 normalized_step；R1 的 C 层仅增加统计/断言，没有另一套增强或优化器。原入口及旧断言没有修改。

1. **IDLE**：仅接收当前 RGB，无 ID、domain、subset、GT 参数。上一张 payload 必须已移出。
2. **CANDIDATE**：VERIFY/RANDOM 拍摄 BN affine、全部模型 buffers、Adam 完整 state 和 param-group 配置的无别名快照。六弱视图照旧；捕获原图 logits、逆变换后弱视图 logits，以及实际送入 BCE 的 q。原强增强、backward、Adam、trial 原图前向均不变。
3. **DECIDE**：detach 后在 CPU float64 全 512 栅格归约 SSE/count。门限只用过去最多 128 个有效 r。先 warmup，再空区，再历史不足；否则判定 eligible。RANDOM 每次调用自己的 random.Random(20260908).random() 一次，不消费增强 RNG。
4. **COMMIT / ROLLBACK**：提交候选，或原位恢复 affine、buffers、optimizer state、group 元数据及延迟初始化的存在性。Parameter 对象与 group ownership 保持；拒绝时清空梯度。旧 B1.steps 在候选前表示已提交数，候选内部检查 committed+1；拒绝后恢复 committed，旧断言本身不变。
5. 追加本图有效 r（包括被拒绝候选）。不恢复增强/RANDOM RNG、访问/候选计数、真实物理调用或风险历史。分别记录 committed/rejected/forced/eligible/restoration 次数；一次 restoration 表示一次完整事务恢复，不是每个张量算一次。
6. **EVALUATION_RELEASE**：输出对应状态的原生 logits；接受 trial，拒绝缓存 pre，绝不重新前向。内部临时 views 清空；只保留当前评价 payload。
7. take_evaluation 单向移出 payload 后回到 **IDLE**；evaluator 才读取当前 mask，沿原 mapping/resize/Dice/ASSD 打分，finally 清空 payload。标签/分数无返回 host 的通路。可以合法丢弃移出的 payload，不影响算法后续状态；正式流水线必须完整记录评价，否则 analyzer 拒绝完成。

失败进入 **FAILED**；该 host 不允许原地继续。非有限值不会被写成普通拒绝。候选调用失败仍由底层实际计数保存在 failure 证据，不把拒绝更新从 8/1/1/0 配额扣除。

跨图算法状态只保留学生、Adam、至多 128 个风险标量、计数和隔离 RNG；固定源参数只读。没有 RP、PCA、teacher、kernel、图、输出融合、回源或域路由。瞬时快照不包含历史图像。

## 阶段与监督

A/B_NEW 分别授权，AB 只用于计划和汇总。默认均 disabled。A 完整运行三流后由 CPU gate 给出 NO_ADVANCE 或 ELIGIBLE_FOR_REVIEW，两种结果都返回并停止；没有调用 B 的分支。B 只新增 17 条，复核 A 的完整 trace、来源 SHA、C 路径指纹、science/data/stream 及无标签 calibration，引用三条 A C 记录而不重跑。

未来运行复用原 R1 有限监督器、原 NFS ENOENT 处理、其他 IO 错误传播、自有进程组清理、原子失效/发布，进程审计在 GPU 模型分配后首个 forward 前。新入口先装 neutral_subprocesses。失败不自动 retry、不等待新授权、不修改已有结果。
