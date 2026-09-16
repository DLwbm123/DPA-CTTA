# R5：更新收益诊断与验证后提交

**状态：实验设计提案；未实现、未执行、未获得新的外部 review 或 GPU 执行授权。**

固定依据为 DLwbm123/DPA-CTTA@b2bfce6cb29cea2df026194120f45b4f7252d53f。本文不改写 R4 结果，不将本轮计划视为 R4 授权的延续。可直接交给 Codex 的自包含实现规范见同目录 R5_STAGE_I_CODEX_PROMPT.md。

## 1. 决策与假设

R4 的 KDG 相对 C 为 +0.037384pp，MT 为 -0.536836pp，G_BOUND 为 -0.070703pp，均不晋级；KDG 对 K_FREE 的差仅 +0.008929pp，G_BOUND 也没有超过 G_GLOBAL。继续组合这些路线缺少已建立的独立增量依据。[S1]

本轮只检验一个新假设：C 的历史适配可能值得保留，但部分候选更新的即时损伤能否被当前图的无标签预测响应识别？先检查信号，再测试完整回滚轨迹。假设不由 R4 自动成立；规则失败只限制该冻结规则/配置，不排除全部选择性适配。

候选名称 C_VERIFY 只是工程名称，不是首次创新或有效性声明。与六视图软目标更一致、可靠区变化更小，也不保证真实分割更正确。

## 2. 三个阶段与停止点

| 阶段 | 做什么 | 允许的真实模型计算 | 完成后 |
|---|---|---|---|
| I | 实现、程序化 CPU 测试、metadata dry-run、审阅材料 | 0；给定 checkpoint 与真实像素均不加载 | R5_IMPLEMENTATION_READY_FOR_REVIEW |
| R5-A | C 在 order0/order1/recurrence 上增加只读 pre/q/trial 与影子决策记录 | 3 条完整轨迹 | 根据冻结 gate 报告资格；无论结果如何均停止 |
| R5-B | C/C_HALF/C_RANDOM/C_VERIFY × 五流 | 全矩阵 20 条，复用 A 的 3 条，仅新增 17 条 | 完整比较后结束；不调参重跑 |

当前提供的 Codex prompt 只推进阶段 I。阶段 A 与 B 分别要求新的、绑定实际实现/science/数据/流的授权。A 的 gate 通过仅说明值得审阅完整比较，不是自动执行 B，也不是方法成功。

A 主诊断只有两个主序平均，不能称四序主结果。B 才采用四主序平均。回访始终单列；它是同一开发内容的另一种到达流，不是独立患者重复或固定旧域遗忘探针。[S1,S2]

## 3. 固定资产、信息与优化协议

保留 1,951 内容/轨迹，其中 remaining_dev=1,695；所有内容参与适配，只有评分汇总使用 subset。主评分沿用每图 OD/OC 平均、域内平均、四域等权，再主序等权。四域 remaining_dev 数分别为 336/586/736/37，不改权重追求正结果。[S1,S7]

保持原 Fundus ResUNet34、512×512、独立 sigmoid 与原阈值/GT 解释、当前输入统计、41 层 BN 的 19,136 个 affine、Adam 1e-4/0.9/0.999/1e-8/wd=0、float32、AMP 关闭、seed=20260907。C_HALF 仅 lr=5e-5，从第一张开始生效。[S2,S3]

不引入源数据/代理/原型、源端训练、额外组件、RP、教师时间平均、kernel 或图模块。算法只看当前 RGB、到达计数和允许的过去状态，不看标签、ID、域或真实边界。GT 在全部状态提交后才进入 evaluator。[S2]

## 4. 诊断与分数分解

用 p_pre 表示原六视图中的原图预测，q 为实际传入 C loss 的六视图目标，p_trial 为原图候选更新后预测。捕获不新增模型前向。每图仍六弱+一强+一候选原图，共 8 forward、1 backward、1 候选 Adam。[S3]

模型输出继续使用 logits，概率仅用于诊断与门控，避免把概率传入旧入口后再次 sigmoid。额外的 Dice/ASSD 评价有 CPU 成本，须单列，不能称完全零成本。

令 D 是百分制 Dice：

L_t = D(p_trial,y) - D(p_pre,y)。

它是同一历史下当前候选一步的即时分数差。仅在历史 C0 逐内容分数完整匹配且可只读获取时，补充：

H_t = D(p_pre,y) - D(p_C0,y)，
D(p_trial,y) - D(p_C0,y) = H_t + L_t。

缺失 C0 时 H_t=null；不根据聚合表回填、不新增 C0 运行。H_t 不是固定旧域遗忘测量。旧 B4 确有 C0，但公开表格不等于当前已获得私有逐内容记录。[S6]

## 5. 唯一候选规则

在当前图的完整概率栅格，以 CPU float64 对 detach 概率归约：

e_pre = mean((p_pre-q)^2)，e_trial = mean((p_trial-q)^2)。

对 OD/OC 各自的可靠前景/背景形成四区：前景要求 q>=0.9 且六视图均>=0.5；背景要求 q<=0.1 且六视图均<0.5。非空区域内计算 mean((p_trial-p_pre)^2)，r 为四区有效均值的最大值。空区域忽略；全部为空则 r=null。

过去缓冲区最多 128 个有效 r。门限为过去值的 90% 线性插值分位数，当前值不得提前入窗。先决策和提交/回滚，再把本图有效 r 入窗，包括被拒绝候选；最后 evaluator 读 GT。

前 32 张、当前全部可靠区为空、或过去有效 r 不满 32 个时强制接受。其余 eligible 访问接受的条件为：

e_trial <= e_pre + 1e-12 且 r <= Q90_past + 1e-12。

数字全部事前冻结，不扫描。任何非有限概率、loss、梯度、参数或 optimizer 状态都是硬错误，不是假装一次普通拒绝。规则可能保护错误的一致预测，因此必须有 A 的配对证据和 B 的简单控制。

## 6. A 的影子资格检查

A 实际始终执行 C；shadow_accept 只用于事后分析：

G_s = MacroAvg_s((1-shadow_accept)*(-L_t))。

a_s 由该流全部内容的 eligible 接受数/eligible 总数得到。同 eligible 范围、同平均接受率的均匀随机影子参照用解析期望：

G_random_s = MacroAvg_s(eligible*(1-a_s)*(-L_t))。

这里 MacroAvg 仍只在 remaining_dev 按四域等权。随机参照不需要额外模型或多 seed。它是事后同路径参照，不是把未来接受率提供给在线方法。

A 资格要求全部满足：
- 三轨迹和全部机械/CPU/数据/计数核验有效。
- 三流 eligible 均非空、均至少拒绝一次，eligible 接受率各 >=0.50。
- G_order0、G_order1 各 >=0，二者平均 >=0.20pp。
- G_recurrence >=0。
- 两主序各自 G_s-G_random_s >0。

这是资源筛选，不是显著性或安全判断。通过：R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW；未通过：R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE。机械失败另记，不作为科学负结果。

影子结果没有改变 C 的后续状态，不能当作真实回滚性能。使用 GT 的 pre/post 事后选优量也只能叫同路径即时机会量，不能叫长期可达上界。

## 7. B 的四臂与完整事务

| 臂 | 行为 | 排除的简单解释 |
|---|---|---|
| C | 原更新、全部提交 | 原有有效适配 |
| C_HALF | lr=5e-5，全部提交 | 是否只是步长小一些 |
| C_RANDOM | 同样计算完整候选；eligible 时按常数概率接受 | 是否只是少提交一些更新 |
| C_VERIFY | 固定的响应验证规则 | 是否按当前响应选择有额外价值 |

C_RANDOM 的 p_accept 是 A 三流全部内容的 sum(eligible*shadow_accept)/sum(eligible)，只读无标签 trace 派生；分母为零则禁止继续。A 前该值为 null，A 后记录为 B 的派生冻结资产，不按 Dice 搜索。独立 random.Random(20260908)，每次访问消耗一次 random()。RANDOM 使用与 VERIFY 相同的强制接受政策，但在自己的轨迹上判定 eligibility，因此实际接受率只近似匹配。 这项常数使用了 A 完整开发流的离线无标签标定，应披露它不是零目标预扫描对照；标定成本已计入 A，VERIFY 本身不使用该常数。B 及未来独立确认流内不得重新标定。

拒绝时恢复本图前的 BN affine 与全部 Adam 状态，包括 step、矩、实际存在的额外状态、param-group 元数据及首次初始化的存在性；不更换 Parameter 对象。输出缓存的 pre logits。只恢复参数而留下 Adam 记忆不是本计划的回滚。[S4,S8]

不能回滚增强/RANDOM RNG、到达次数、物理调用计数与风险历史。尝试更新、已提交更新和拒绝次数必须分开；Adam 已提交 step 对应 n_committed，而不是总访问数。旧 C 的访问数断言保留，新 host 单独实现。[S4]

B 晋级的开发门槛为：VERIFY-C >=0.50pp 且 3/4 主序正向；VERIFY-HALF 和 VERIFY-RANDOM 各 >=0.20pp 且各 3/4 正向；最差同序 VERIFY-C >=-0.50pp；每域四序均值差 >=-2.00pp；回访差 >=-0.10pp。尾部与 ASSD 必须报告，这些门槛不等同临床安全。简单控制同样有效时采用简单解释，不归功于验证。

## 8. 预算与停止条件

| 范围 | 正式轨迹 | 访问 | forward | backward/候选 Adam |
|---|---:|---:|---:|---:|
| A | 3 | 5,853 | 46,824 | 5,853 |
| B 新增 | 17 | 33,167 | 265,336 | 33,167 |
| A+B 总计 | 20 | 39,020 | 312,160 | 39,020 |

0 VJP；C0 新运行为 0。上述为算术预算，不是实测结果；CPU 测试与未来有界 smoke 另算。候选计算后才拒绝，所以不节省模型 forward/backward，不宣称加速。最多三 worker，每卡一个；设备与资源 scope 在实际执行前另行冻结，不沿用 R4 的 GPU 编号。

A 未达 gate 即停止该规则，不搜分位数/窗口/置信阈值。B 不达晋级门槛即归档，不换 seed/子集/指标求正结果。运行失败保留历史，不自动重试或隐去失败成本。新执行 scope 不能复用旧 waiver。

## 9. CPU 实现交付与证据边界

阶段 I 必须验证多步旧 C 一致性、强制全接受退化 C、首步和已有 Adam 状态回滚、下一张状态一致性、超过 32/128 的窗口边界、空区/非有限错误、标签隔离、独立 RNG、8/1/1/0 计数、3/17/20-job 矩阵、解析随机影子参照、gate 边界与被篡改日志失效。完整模型使用随机权重，不读取给定 checkpoint 或真实图像；源/真实资产读取为 0，沿用既有 CPU 检查的边界。[S5]

保存私有标量以复核 SSE/count、窗口、决策、配对结果与成本；不保存当前概率/图像/激活或适配后的模型。公开只给聚合与去身份日志。CPU 标量关系重算不等于重建概率、梯度或 ASSD 几何。[S1]

所有当前内容仍是已暴露开发材料。A 参与选择的主序与 B 新增主序分别标注；新顺序或新 seed 不构成独立内容确认。方法通过后才设计冻结的新内容确认，不提前承诺其结果。

## 依据索引

以下仓库路径均锁定至 b2bfce6cb29cea2df026194120f45b4f7252d53f；为避免把旧 README 的历史状态当作本轮状态，优先使用具体完成报告和实现文件。

- [S1] docs/results/r4_three_track_v1/REPORT.md。
- [S2] docs/review/r4_three/STATE_LIFECYCLE.md；docs/review/r4_three/input/R4T_COMBINED_EXPERIMENT_PLAN.md。
- [S3] src/dpa_ctta/r4_three/host.py，及其 C 分支调用的 src/dpa_ctta/r1/host.py、src/dpa_ctta/b1_host.py。
- [S4] src/dpa_ctta/b1_host.py：normalized_step 中 Adam step 与 steps 的原有断言。
- [S5] docs/review/r4_three/REPRODUCE.md；IMPLEMENTATION_REPORT.md。
- [S6] results/b4_frozen_c_transfer_v1/B4_EXPERIMENT_REPORT.md。
- [S7] docs/results/r1_recovery_target_subspace_20260912/EXPERIMENT_REPORT.md。
- [S8] PyTorch 官方 torch.optim / Adam 文档：参数与 optimizer state 分别保存；Adam 使用一阶矩、二阶矩和 step 状态。这里只核实状态语义，不据此升级仓库固定环境或引入新版 API。

本轮假设、规则、门槛和分阶段预算均为新提案，不应被解读为上述来源已经验证过它们。
