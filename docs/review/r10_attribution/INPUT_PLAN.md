# R10 下一轮：收益归因与监督训练对照

状态：PROPOSED；仅为实验计划，未修改工程、未启动任务。
建议实验身份：R10_ATTRIBUTION_SUP_8H_V1。

## 1. 证据与结论边界

本轮完成情况来自用户转交的 Codex 汇总：执行 SHA 116f86b3421df4bfd582d9fb034b5443146ef6bf，报告本地提交 2126c7f；1000 WARM 更新、1024 post 采样轮、8 条目标轨迹、8192 次到达、7104 次主评分，约 1 小时 59 分钟含预检，无失败/恢复。

当前未取得 2126c7f 的完整 REPORT.md、resolved config、训练日志和逐图记录。GitHub 读取该 ref 返回 No commit found。不得将本计划中的根因假设写成对真实执行版本已经查实的故障。

根据用户提供的百分数取两条 order 等权均值：

| 方法 | 平均 Dice % | R10 减该方法，pp |
|---|---:|---:|
| Source-only | 68.17120 | +3.82555 |
| GraTa | 76.21165 | -4.21490 |
| R10 | 71.99675 | 0 |
| R10 CONST_HALF | 71.99690 | -0.00015 |

由此只能说：该预算、种子和两条开发流中，R10 未超过 GraTa，学习 writer 未体现平均效用。不能将相对 Source-only 的改善直接归因于 RL；不能根据两个均值接近证明逐图输出、状态或全部历史机制等价。

已核对的其他证据：
- 实际下发的 CODEX_EXECUTE_R10_12H_CORE.md：固定 B_FULL_20260924、原 N/G 归一化路径、同权重 CONST_HALF、1000/1024 档、LR 终点压缩到选定长度、固定终点而非 checkpoint 搜索。
- 公开旧快照 84837e997a60d7c98e4751b7ee737b1f074693b3 的 R10 learning.py、controller.py、原 R10_EXPERIMENT_PLAN.md：用于确认原设计，不代表未 push 执行版完全相同。
- 同快照 docs/review/r8_screen24/REPORT.md：R8 B_FULL 73.249%、B_STATIC 73.510%、C0_CURRENT_STATS 75.079%；不是本轮相同目标子集，不能直接与本轮百分数相减。该报告的 B64 source direct-latent 有限优化诊断较 zero 改善 0.258 pp，不是可证明的能力上界。

## 2. 唯一研究问题

区分四种来源：归一化变化、现有 B 载体、当前 use 控制学习、持久历史和 writer 学习。

下一轮不新增 RL 配方，不增种子，不重跑 WARM，不延长到 4000，不重建 basis，不改为新的分割主干。只补低成本必要控制，最多新增两条已有监督训练路径。

## 3. 资产和旧结果复用

先读真实本地 REPORT.md、RESULTS.csv、RESOLVED_CONFIG.json、RESOURCE_LEDGER.json、封存/评分回执、WARM 和 post checkpoint，以及执行版相对其基座的 learning.py/target.py 修改。

保留旧结果、旧工作树和账本。新计划使用独立身份及输出目录；运行期间固定实际代码 SHA。不将历史旧快照视为要求回退本地实现。不自动 push。

N/G/R10/R10_CONST_HALF 的旧两条短流结果只有在完整模型、运行配置、到达清单、顺序、计分资格、评分定义及实际执行身份一致时才复用。状态型方法不能截取旧 R8 1951 图轨迹中的相同图像来冒充本轮 1024 图轨迹；只复用其源端模型资产，在新短流中重新运行。

旧证据不足时不杜撰分域值或重建假轨迹；将相应比较标为证据不足。确需重跑的成本先纳入剩余预算，不默认重跑整批旧基线。

## 4. 阶段 A：定向检查与无训练控制

### A1. 定向检查，最多 30 分钟

只查影响解释的事项，不重复全仓历史审计：
1. 真实加载的 actor 是规定 post 1024 终点而非 WARM 或随机初始化；两个原部署臂共享同一个 actor 摘要；仅 CONST_HALF 覆盖 writer。
2. writer 是否进入 optimizer、requires_grad 是否正确，WARM→post 的 use/write 参数变化分别有多大。全局 grad_norm 不能替代 writer 独立梯度；初始化最后层为零时，不要求首步所有隐藏层梯度都非零。
3. 分别检查源训练采样动作和目标确定性动作，记录 writer 的分位数、标准差、max|w-0.5|，不能仅看训练/探针混合均值。
4. 读取 reward_std、zero_advantage_fraction、实际 advantage 范数、EMA、两次 optimizer epoch 各自 clip_fraction、KL。PPO 第一次更新 ratio=1、clip=0 和组均值 advantage=0 都可正常，不能单凭 loss≈0 判定没学习。
5. 同前缀同 use 的单步 w=0/0.5/1 必须给相同当前输出；验证完整 m/q/h 的提交及随后状态传播。目标标签不得进入适应器。

日志缺失的诊断只在 checkpoint 副本及固定小份源数据上补测；不修改已封存模型、不把补测伪装成旧训练记录。工程错误仅修复受影响范围；科学定义改变须另记版本，不能混算旧结果。

### A2. 五个无新增训练控制

所有目标控制都使用本轮原来的两个短流 manifest，每流 1024 次到达，保留完整到达及原计分资格。以下各跑两条，共 10 条新轨迹、10240 次目标访问。

| 名称 | 权重与部署定义 | 主要用途 |
|---|---|---|
| C0_CURRENT_STATS | 同源 backbone、同预处理、与 R10 匹配的 current-statistics BN，零 FiLM 调制，无持久图像状态 | 检查改善是否主要来自归一化；保持 C0 实际调用成本，不人为增加前向 |
| B_PARENT_FULL | 本轮实际使用的 B_FULL_20260924 父载体，按其原生 FULL 部署，无 R10 actor | 检查 actor 是否改善/损坏原载体 |
| WARM_CONST_HALF | 本轮实际 WARM1000 actor，R10 状态与求解路径，每步 w=0.5 | 与 post CONST_HALF 比较，归因 post 对 use 的新增效果 |
| R10_RESET_ALL | 原 post1024 同一 actor，每图读之前清空 m/q/h | 检查同权重模型对历史的依赖 |
| R10_FORCE_WRITE | 原 post1024 同一 actor，每步强制 w=1，m/q/h 一并写入 | 检查改变写入策略是否改变后续行为和分数 |

原 R10 与同权重 CONST_HALF 保留为历史精确配对基准，不再次训练。RESET 是同权重部署干预，不等价于独立训练的 STATIC。FORCE_WRITE 会改变后续 use，不能宣称整条轨迹只有 writer 改变而所有当前输出不变。

新目标评分统一封存到本轮终态后解封，不根据中途目标分数选择训练方法或调整参数。

### A3. 源端反事实写入诊断

固定现有 post1024 actor；不训练、不构造新方法。使用原 source-val 的四种模式各 8 个上下文，共 32 个，按原 schedule 的实际模式映射选取并锁定 ID，不假定 episode 编号取模就是 val 模式。保持 group 划分；不得换用 target 样本。

每个上下文取 32-visit episode：prefix 为 visits 0..11，当前 t=12，未来 visits 13..28。prefix 使用该固定 actor 和 w=0.5。当前观察、候选 use 和当前分割只计算一次；复制同一提交前状态，分别强制当前 w=0/0.5/1。随后各分支都以该固定 actor 的确定性 use、w=0.5 继续同一未来 16 图。

从同一未来序列提取前 4 和前 16 图的结果，不为 H=4 另跑一套。共 32×3×16=1536 次未来分割预测，另计当前预测、prefix/观察和实际 I/O。所有计算纳入预算。

记录：当前预测等价误差；提交后 m/q/h 差；未来概率差、硬掩码翻转比例、OD/OC hard Dice 与 soft Dice 的成对差；不同模式的符号及分布。未来使用码可因历史不同而不同，这是干预的正常后果。

可报告每上下文三个候选的最好值相对 0.5 的优势，但只能叫有限候选源端反事实诊断，不能称严格上界、目标收益或可部署性能。小样本最优选择有乐观性。H=16 仅是诊断时间范围，不改本轮训练 H=4。

## 5. 阶段 B：最多两条监督训练，不再盲目加 RL

两任务都从同一个实际 WARM1000 终点独立分叉，不从已完成的 RL 终点续训。复用其原 parent/basis/observer/scaler、固定 policy seed 20260924。若合法 WARM checkpoint 缺失，仅阻塞依赖它的训练；阶段 A 中不依赖它的控制继续，不重新训练 WARM 伪造同一初始化。

| 新训练 | 固定目标 | writer 与状态 | target 部署 |
|---|---|---|---|
| SUP_STATIC_BUDGET1024 | 原 R10 SUP_STATIC：soft Dice + 原 use 维度参考 KL | 每图 reset，writer 冻结，不参与梯度 | SUP_STATIC |
| SUP_RET_BUDGET1024 | 原 R10 SUP_RET：4 步 soft Dice + 原回访保留项 + 原参考 KL | 4 步内保持可微状态传递；prefix detach；允许 use/write 梯度 | SUP_RET；同 checkpoint SUP_RET_CONST_HALF |

固定 post 1024 个采样轮、每轮 2 次 optimizer 更新；G=4，H=4，sigma=0.35。共最多 2048 个新采样轮、4096 次新 optimizer 更新。源样本、外生模拟、group 约束和四模式×八窗口调度与旧预算版对齐；严格沿用各自原方法定义，不加额外 BCE/边界项。

AdamW 的 peak LR=3e-5、终点=3e-6、前 100 轮 warmup、其余 cosine、wd/betas/eps/clipnorm 均按原 R10 及本轮 1024 档锁定，复核实际执行旧档，不以猜测参数替换 resolved config。每个 post 任务重新初始化 optimizer。参考策略均为同一个 WARM1000 副本，整个 post 冻结。

只更新允许的 actor 参数。分割主干、B载体与 basis 权重冻结不等于切断输入梯度：SUP 必须保留 loss→use→状态→actor 的可微路径；不能把主干整个包在 no_grad 中。允许对冻结前向观察缓存做身份绑定复用；动作相关预测不得复用为另一个动作的结果。

固定 1024 终点为主结果，不用目标挑 checkpoint。各任务结束后在原短版固定的 16 个 source-val×32 visits 上做一次确定性验证，四模式各 4 个，restore RNG，不更改训练。已有中间 checkpoint 可作明确标记的源端诊断，但不变为目标选模。

SUP 与旧 GR 的目标分别含 soft/hard Dice，且 SUP 第二 epoch 需重算可微主干路径；因此比较是相同初始化/采样轮数的训练配方比较，不是完全同目标的估计器因果实验，也不是相同 FLOPs 或 GPU 时间。

## 6. 目标矩阵、评分与解封

三种新训练部署（SUP_STATIC、SUP_RET、SUP_RET_CONST_HALF）各跑原两条流，共 6 条新轨迹、6144 次到达。

完整新增目标上限：阶段 A 10 条 + 阶段 B 6 条 = 16 条，16384 次到达。若核验每条仍为 888 次主评分，则为 14208 条新增主评分记录；以真实 manifest 计数为准，不能因为 7104/8=888 就跳过逐条资格核验。

原八条仅在完整身份吻合时复用。新旧结果分列，生成共同口径的比较表。不增加 LONG10/MIXED、其他种子、其他源域或超参 sweep。

主要差值：
- R10 − C0：超出归一化的净效果；
- R10 − B_PARENT_FULL：相对父载体的净效果；
- R10_CONST_HALF − WARM_CONST_HALF：共同固定 writer 下 post 对 use 的效果；
- R10 − R10_RESET_ALL：同权重历史效用；
- R10 − R10_CONST_HALF：原已知 learned writer 效用；
- SUP_RET − SUP_RET_CONST_HALF：监督训练后的 learned writer 效用；
- SUP_STATIC / SUP_RET − 原 R10 与 GraTa：不同训练配方与强基线的差距。

SUP_RET−SUP_STATIC 同时涉及训练目标和状态使用，不单独归因于 writer 或保留项。报告 OD/OC、每域/每 order、域等权主指标、逐图配对差分布和最差域/后段；只有 manifest 真实存在回访才报告回访。不要将 pooled 结果替代域等权主表。

两 order 是同一 checkpoint、同一内容集合的相关测量，不是两个训练 seed。可用可信患者/来源 group 做配对重采样的描述性区间，不能用图像/像素重复访问伪造训练随机性置信区间；患者依赖未知则注明。

## 7. 时间与资源：新增最多 8 小时，不重置旧账

设新阶段 T0 在首次预检时持久化：墙钟新增硬上限 8h，同时不得突破原 12h 预算包剩余的同口径资源额度。不要将保守收费、预留 GPU-worker 时间和实际墙钟混成一项实测值。用户汇总约 1h59 及此前 1.68h 不足以替代真实账本去重/余额核验；如两者确为互不重复的同口径计费，则再加 8h 约 11.66h，仅为示意。

| 部分 | 分配上限 |
|---|---:|
| 读记录、局部正确性与缺失计时核验 | 0.5h |
| 五个无训练控制 + 源反事实诊断 | 1.5h |
| 两条 SUP 训练及源验证 | 合计 4h，按完整依赖链准入 |
| 三种新部署的两流评估及评分 | 1h |
| 有限故障恢复储备 | 0.5h |
| 最终汇总、归档与退出 | 0.5h |
| 总计 | 8h |

这些是预算分配，不是实测工期保证。通常计算安排在 T0+7h 前完成；恢复最多延至 T0+7.5h；最后半小时只汇总/归档，不开始新的训练或目标适应；T0+8h 所有本轮进程退出。若原预算余额更小，提前相应截止，不能新建 ID 重新获得 8h。

使用旧本轮按阶段实际计时与可迁移 profile。SUP 的新增路径仅补必要完整 32-round schedule block，计入预检；保留 1.3 安全因子，包含读图、最长 prefix 分布、source validation、反传、checkpoint、目标预测写入和 CPU scorer。不按 2h 旧总耗时线性估算 SUP；不重跑所有 51 个历史 profile 单元。

正式训练前按成本一次性确定可装入的分支。优先保留 SUP_RET 及其 CONST_HALF 的完整成对目标评估；SUP_STATIC 只有连同源验证和两流目标能装入时才启动。无法装入不暗中削减轮数后作等预算比较。若任一完整训练分支都不可行，完成可行的诊断并将训练标 NOT_RUN_BUDGET，不声称训练对照完整。

一张已授权且空闲的 GPU、最多一个 GPU worker。CPU 评分按现有资源规范；不终止其他任务、不借新 GPU 或付费资源。沿用原存储上限与概率退休合同，保存本轮拥有的充分审计证据；最多一个 job 一次等价快照恢复，总失败/恢复不超过 0.5h 且不延长绝对截止。

## 8. 如何用结果决定后续方向

本节是本轮结束后的科研决策，不是中途目标分数驱动的早停/调参：

| 观测组合 | 合理的下一步 |
|---|---|
| C0 ≥ R10，WARM_CONST_HALF≈post_CONST_HALF | 不能将 R10−Source-only 解释为 RL 收益；暂停盲目扩大该 RL 配方 |
| RESET 不劣于有历史，源反事实写入也几乎不影响未来 | 当前工作点未显示历史效用；优先当前图像适配，不增加 writer 复杂度 |
| 强制写入明显改变结果，学习 writer 仍≈0.5，SUP_RET writer 能获得收益 | 优先排查 RL 的延迟信用分配/动作探索；下一轮才研究 writer 专用反事实目标，不同时改网络/奖励/时域 |
| 源端 writer 收益清楚，目标端无收益 | 优先检验源序列模拟/观察分布迁移，不按目标标签改 gate |
| SUP_STATIC 最好或 SUP_RET 的半写不差 | 优先简单静态/固定写入方案，不能宣称持续记忆贡献 |
| SUP 与 RL 均不如同口径 C0 | 停止在该冻结 B 上叠加策略复杂度，转向当前适配/载体有效性；不是全体 CTTA/RL 的理论否定 |

预先采用两个研发参考尺度：相对原 R10 +0.5pp 视作可继续研究的效果信号；learned writer 相对同权重 CONST_HALF +0.2pp 且两 order 不反向，才称有值得复验的 writer 信号。阈值是本次设定的研发尺度，不是统计显著性/临床标准，不授权以此早停或改配方。仍需同时报告相对 C0、GraTa 和最差域/通道的差值。

本轮不主张跨种子稳定性，不称新盲测，不宣称 SUP 是数学上界，也不承诺新增训练必然提高 Dice。

## 9. 最小交付

保留 PROTOCOL、RESOLVED_CONFIG、模型/数据/流身份、真实代码 SHA、运行/评分/封存回执、RESOURCE_LEDGER、原始允许保存的逐图标量。

额外交付四张小表：
1. ATTRIBUTION：上述关键差值及分域/通道/order 分解；
2. WRITER_DIAGNOSTICS：参数变化、确定性 gate 分布、状态/概率/掩码影响、训练信号；
3. SOURCE_COUNTERFACTUAL：32 个源上下文的 4/16 步成对写入后果；
4. COST_AND_STATUS：各任务的实际计算/IO/评分成本、复用、新执行、失败和缺失。

最后给出“改进主要来自哪里、writer 是否真正改变未来预测、当前应继续哪条研究线”三个明确回答。未执行或证据不足不能填 0，不能由退出码/文件名推断成功。
