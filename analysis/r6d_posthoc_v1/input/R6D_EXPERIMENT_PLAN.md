# R6-D：区域加权失败的只读、事后机制诊断

## 0. 决策与范围

下一项任务不是 R6-B，也不是修改 gate 后重判 R6-A，而是 **R6-D**：分析既有四臂、三流账本，区分平均抵消、历史状态差异、当前更新差异与监督贡献分配。本轮新增 GPU 轨迹、模型 forward、backward、Adam、VJP 均为零。

本分析是在已知 R6-A 汇总结果后提出的事后探索（post hoc exploratory analysis）。在读取更细的账本前固定本方案有助于限制分析分支，但不能将其称为结果未知时的预注册或独立确认。

当前交付包含计划、执行 prompt、机器可读分析规格与完整性清单；不声称已经访问私有账本或完成 R6-D。

### 固定历史对象

- 仓库：`DLwbm123/DPA-CTTA`。
- R6-A 结果发布 SHA：`aa732b42b03d378149028a3770879b72ebc55db3`。
- 原执行 SHA：`c94fff7c05cec38d541c441b62a9381ee46ba076`。
- 原 science SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`。
- 原 run_id：`a70dc79f23db47289532792141b61f57`。
- registration digest：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`。
- stream digest：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。
- production fingerprint：`a6ef161066d3560dc2dd540cb507bff2e4a7cadf94ab711545877c525dee2f2e`。
- 原结果：`R6A_COMPLETE_NO_ADVANCE`，原 R6-B 保持 `NOT_RUN`。

## 1. 已知结果与待回答问题

原主指标为 remaining_dev 的 1,695 内容，OD/OC 同图平均、域内平均、四域等权，再对 order0/order1 等权。每条 1,951 内容均参与原适配；order4 回访单列。R_BAL−C 两主序均值约 −0.018004pp，order0 −0.277569pp、order1 +0.241560pp、order4 −0.834770pp。R_BAL 相对 SCALE、SHUFFLE 的两主序平均差为 +0.438002、+0.506100pp。[S1]

本轮不再用平均 +0.5pp 作为诊断是否完成的条件。要回答：

1. 最终差距在本图更新前是否已经存在？当前一步是否加重或缓解差距？
2. 哪些域和通道造成主序反转与回访下降？是否集中于已有域切换位置？
3. 伪前景是否真的控制了主要监督梯度，还是只占据面积优势/劣势？加权是否同时提高了错误伪标签的权重？
4. 现有结果足以提出一个可证伪的新干预，还是仍不能区分解释？

**回访不是重复图像测试。** 已有 stream 4 使用全部 1,951 个不同内容，维持域内内容顺序，按注册的 64 内容 chunk 调度并使域再次出现；重复的是域，不是同一图像。本轮不能把不同内容的早晚分数直接称为遗忘。[S3]

## 2. D0：只读输入、身份与复算

### 输入

有权访问时，只读取原运行目录内的标量账本、结果和执行证据，以及注册 JSON 元数据。至少包括 12 个 job 的 `unlabeled.jsonl`、`evaluation.jsonl`、`completion.json`，以及原 receipt、scope、结果指针和支持完整性核对所需的退出/smoke/流程记录。解引用注册元数据中的 RGB/mask/checkpoint/source 路径属于禁止行为。

在新私有工作目录创建独立快照。不能用指向原件的符号链接或硬链接替代快照。对允许读取的原文件与快照分别记录 SHA256；分析后复核原文件清单、字节摘要和原结果指针未变化。不要对原目录 chmod、移动、补写日志或生成 pycache。快照和新输出也不能位于原运行目录中。

### 审计

按 frozen source 的纯只读 `completed/replay/join/validate_metric` 等检查逻辑验证原记录；采用其函数前检查实际调用链没有写操作。禁止调用 `analyze.recompute`、`invalidate`、`publish`，即使只是为了获得 joined rows。它们会修改结果指针。[S4]

要求：12 个 arm×stream 组合；每组合 1,951 无标签行与 1,951 评价行；各轨迹 1,951 个唯一 group_id/sample_id；两类账本各 23,412 行。不能将三流视为 5,853 个独立患者。原 receipt 的 worker/job binding 也需核对，不只比较全局 run_id。

重建 OD/OC 和 macro 的 pre/q/post 评分、原主指标、三个控制配对、原 gate 与回访分数。核对像素计数可实现性、GT 跨预测/跨臂/跨流一致性、q 前景数等于 trace.n_fg、原每访次 8/1/1/0 及累计计数。不得修改旧容差；若仅对展示至六位小数的 REPORT 比较，明确使用 0.000002pp 的显示舍入核对容差，不将其用于底层科学校验。

若只有公开聚合，允许完成能从均值、分母、配对摘要恢复的部分；逐图联结、联合分布、分块和关联不可恢复时必须为 null/NOT_AVAILABLE，最终状态为 `R6D_PARTIAL_AGGREGATE_ONLY`。不能从边际分布或若干分位数合成私有行。

## 3. D1：精确分解最终差距

对同一 stream、同一 group_id、同一 channel，以百分制 Dice 定义：

- `L_m = post_m - pre_m`：方法 m 当前一步的即时变化。
- `Delta_post = post_BAL - post_control`。
- `Delta_pre = pre_BAL - pre_control`。
- `Delta_step = L_BAL - L_control`。

逐图逐通道精确恒等式：

`Delta_post = Delta_pre + Delta_step`。

主要 control 为 C；SCALE/SHUFFLE 保留为机制对照。对三个流、四域、OD/OC/macro 全部计算相同分解。

`Delta_pre` 是两个实际状态路径在进入当前更新时的表现差距，包含此前所有适配影响；它不是匹配 C0 的 H_t，不是已测量的遗忘量，也不是隔离了所有因素的历史因果效应。`Delta_step` 比较的是各臂自己的当前状态/目标下的一步收益，不是从同一个 theta 出发的两种 loss 干预。

同一组权重/行集下均值可相加，**中位数、分位数和 worst-decile 均值不可这样相加**；后者必须从逐内容分差重新计算。

### 精确贡献及抵消

每条流的主评分行权重为 `omega_i = 1/(4*N_domain_remaining)`；跨两个主序再乘 1/2。单通道贡献写入 macro 时再乘 1/2。

报告 signed net、positive mass `sum(omega*max(Delta,0))`、negative mass `sum(omega*min(Delta,0))` 及每个域/通道/流段的贡献。所有分组贡献应能加回同一主指标；不以某个 chunk 的局部均值直接代替其全流贡献。

当净差接近零时，不输出“历史解释了几千百分比”一类除以小净差的比例；报告 signed pp 和绝对量即可。

## 4. D2：顺序反转与域切换

### 同内容跨流差异

用 `(group_id, channel)` 在三个流之间联结。对 BAL−C 的 Delta_post/pre/step 计算：

- order1 minus order0；
- recurrence minus order0；
- recurrence minus order1。

对域内相同内容与同一评分子集计算，不按 visit 直接匹配。报告每域/通道的均值、符号、分布和主指标贡献。它控制了内容组成，但仍包含历史、访问位置及访问序号相关增强分配等差别，不能称为纯顺序因果效应。

### 注册 chunk 与真实域切换

只从注册 `chunk_schedule` 和 `contiguous_domain_segments` 还原顺序。相邻同域 chunk 不算新域切换。全部 1,951 到达内容均用于位置、先前域暴露数和间隔计算，再对 1,695 主评分内容汇总。

对每个连续域段记录域、段起点、domain 内出现次数、先前该域暴露数、距上次该域内容的访问间隔；初始段不是切换。固定比较真实切换后第 1–8 个访问与第 9 个及以后访问。不扫描 4/8/16、最佳 cut-point 或最坏窗口。小段不足 8 条时照实记录，不填补。

对所有 chunk/连续域段保留 n 与 signed contribution。可将 BAL/C 的 pre 分差与当前一步差异随段变化并列，但不把不同内容的早晚差值叫作“遗忘曲线”。“post hoc timing association”须出现在报告中。

## 5. D3：已有监督贡献与伪目标误差

### 5.1 有记录即可精确恢复的无标签量

每图每通道读取：n_fg/n_bg、rho/rho_clipped/fallback、applied_w_fg/bg、S0/Sw/Sperm、a/b、residual_fg/bg_sse、weighted_fg/bg_sse、BCE 分区和、residual_weighted_dot/residual_shuffled_dot、actual_logit_gradient_l2、BN gradient norm 与 Adam displacement。[S2]

定义：

- 前景面积 `pi = n_fg/N`。
- 原 BCE 前景占比 `bce_fg_sum/bce_sum`。
- 原 logit 梯度平方范数的前景占比 `f0 = residual_fg_sse/S0`。
- 基础区域加权后的占比 `fw = weighted_fg_sse/Sw`。
- 前景贡献转移 `fw-f0`，幅度倍率 `a = sqrt(Sw/S0)`。
- 区域加权与普通梯度的通道内余弦 `residual_weighted_dot/sqrt(S0*Sw)`。
- 基础置乱权重与普通梯度的通道内余弦 `residual_shuffled_dot/sqrt(S0*Sperm)`。

这些是输出 logit 空间的代数量，不是 BN 参数梯度方向；只能分别使用每个臂自己的局部状态，不能称跨臂梯度匹配。在 C/SCALE/SHUFFLE 的日志中，基础区域加权量属于在该臂局部状态上计算的 BAL 参考，不一定是该臂实际使用的像素权重。

任一比例分母为零时使用 null 与原因，不加 epsilon 或填 0。fw 随权重变化属于公式决定的效果，不可作为“方法有效”的独立发现。

### 5.2 从 q 的计数推导伪目标误差

由评价中 q 的 P=pred_pixels、G=gt_pixels、I=intersection 得：TP=I、FP=P−I、FN=G−I、TN=N−P−G+I。

报告伪前景 precision、recall、FP/N、FN/N、(P−G)/N 与 q Dice。分母为零时 null，不将两个不同分母的平均值混用。

额外计算基础区域权重下的 hard-error mass：

`E_plain = (FP+FN)/N`

`E_regional = (applied_w_fg*FP + applied_w_bg*FN)/N`。

FP 属于 q 伪前景，FN 属于 q 伪背景，因此该量不需要像素图即可由既有标量恢复。C/SCALE/SHUFFLE 下将其标注为 `hypothetical_base_regional_hard_error_at_own_state`；不冒充实际 loss 或实际错误梯度。

**不能恢复：** TP/FP/FN/TN 各自的 BCE、残差或参数梯度。现有记录没有 GT 与 soft residual 的像素级联合信息。不能将整个伪前景的残差都当作错误监督，也不能声称识别出每个错误 token 的梯度。

GT 只用于本轮离线解释，不能被带入将来的在线权重函数。

### 5.3 分层与关联，避免样本构成混淆

主要固定分层使用 C 的同图 q 区域属性，使不同臂共享同一批比较内容。五类：EMPTY_FG，0<n_fg<N/9 的 FG_MINOR_UPPER_CAP，N/9<=n_fg<=8N/9 的 UNCLIPPED，8N/9<n_fg<N 的 BG_MINOR_LOWER_CAP，EMPTY_BG；用整数不等式实现边界。

在各 stream×domain×channel 内，额外按 C 的 a、fw−f0、Adam displacement 各自单独分四分位组；每个变量独立分析，不做笛卡尔组合，不按 Delta 或 GT 决定分组。四分位使用当前有限值的线性分位数；边界重复允许空组，等于边界放低组，禁止用 ID 打破 ties。记录分母、边界和空组。

BAL 自己的权重/梯度/目标变化作为次要描述，须承认它们是处理后的变量。q precision 等 GT 量只作离线解释，不用于构造主要无标签分层。

报告所有预定分层的 Delta_pre/step/post、伪目标误差和监督贡献。不仅挑选最好或最坏一个。少于 30 个唯一内容的单元保留结果并标注 LOW_SUPPORT，但不单独据此推荐新 GPU 实验；这不禁止小域贡献进入原主指标。没有跨所有分层的全胜 gate，也不做 p<.05 筛选、门控分类器训练或重加权超参数搜索。

按域内相同内容累计位置的前后半再检查一次主要描述是否反向：mid=floor(N_domain_all/2)，所有到达内容决定 split，评分内容仍为 remaining_dev。此检查用于展示时间混杂，不充当独立验证或要求每半均赢。

## 6. D4：结果解读与新的探索资格

必须交付一张竞争解释表：每项列支持、反证、不能识别的部分。

- **history/path gap**：主要损伤是否在 pre 已可见？
- **current-step allocation**：Delta_step 是否持续负向，或者反而缓解了 pre 差距？
- **pseudo-target error reweighting**：FP/FN 方向与 E_regional 是否显示潜在错误监督被赋予更大质量？
- **intensity interaction**：干预较强的 C 锚定分层中是否更容易出现损伤，且不是只由单个低支持单元贡献？
- **unresolved/domain tradeoff**：混合结果能否区分上面几项，还是只有现象没有可操作解释？

不设置“报告必须发现机制”的 gate。合法结果包括：新假设可提出；只有弱关联不足以提出；字段不足。`HYPOTHESIS_PROPOSED` 不表示因果成立、方法成功、review PASS 或 GPU 授权。

### 唯一预留的小规模干预提案（本轮不实施）

若诊断支持“区域干预强度在部分状态下过大”而不仅是区域分配根本不合适，下一次可单独提出 R6-E：

`w_half = 0.5 + 0.5*w_BAL`，q 仍为软目标，权重/全部系数 detach；公式是每像素权重向 1 收缩，不是将学习率减半，不是事后混合两个预测，也不是令 ratio_cap=4。

固定比较 **C、原强度 R_BAL、R_HALF** × order0/order1/recurrence，最多 **9 条新的完整轨迹**。同期 C/BAL 默认新跑，不把旧路径当同期对照；需要新科学规格、实现检查、外部审阅及独立用户授权。本轮不编写模型 runner、不发起该实现，也不把此提案自动转成任务。

该三臂只检验强度折中；不声称重复证明了空间位置机制，也不替代之后必要的 matched controls。先报告新三臂全部的净效应、域/通道与回访风险。探索完成不以 +0.5pp 为必要条件，但主序略有改善而回访继续受损也不是可替换 C 的证据。真正的推广/非劣效/稳健性门槛必须在新数据运行前另行确定并说明实际价值依据。

如诊断主要显示通道冲突、伪目标质量或仍不能区分机制，则不要硬跑 R_HALF，也不要立即追加 OC-only、多个 cap 或多个 eta 网格。最多写出一个最有信息价值的后续问题。

## 7. 计算与实施成本

| 范围 | 新轨迹 | 新 forward | 新 backward/Adam/VJP |
|---|---:|---:|---:|
| 本轮 R6-D | 0 | 0 | 0/0/0 |
| 仅作为未来提案的 R6-E | 最多9 | 140472 | 17559/17559/0 |

R6-E 数字仅是 9×1,951×8/1/1 的拟议正式预算，不含未来 smoke，不是授权或耗时预估。

R6-D 可用单进程、最多2 CPU线程；单次分析 wall cap=1,800秒，新输出 cap=2GiB，快照空间另列并在开始前核对实际源文件字节。未结束须记录 INCOMPLETE，不自动后台重试。CPU测试与运行均记录实际耗时/内存/字节；0模型调用不等于0成本。

新工具置于 `analysis/r6d_posthoc_v1/` 等独立目录，不能修改原 `src/dpa_ctta/r6_regional_consistency/`、原 science、gate 或正式 runner。完成单独的程序化 CPU 测试后，在有权访问现有账本的环境直接运行只读分析；无需为本轮另走一遍完整模型166项验收。若触及生产代码，属于越界而非自行扩大测试解决。

必需测试：精确分解、domain weighting与chunk贡献求和、同内容跨流联结、真实域切换识别（相邻同域chunk不算切换）、空分母、q计数可实现性、ties分箱、重复/缺失/错绑定拒绝、无写保护/原指针不变、禁止GPU/模型入口、公开导出不含标识。测试通过数量按实际报告，不事先指定虚假的条数。

## 8. 交付与完成状态

公开：REPORT.md，MAIN_DECOMPOSITION.csv，DOMAIN_CHANNEL.csv，ORDER_CONTRASTS.csv，RECURRENCE_SEGMENTS.csv，SUPERVISION_SUMMARY.csv，PSEUDO_TARGET_ERRORS.csv，STRATIFIED_ASSOCIATIONS.csv，HYPOTHESIS_DECISION.md，ANALYSIS_BINDING.json，READONLY_AUDIT.json，字段清单与缺失项，测试摘要与去身份日志，DELIVERY.json。

私有：原件/快照清单、原始逐内容联结行、完整过程日志。公开件不含 group_id/sample_id、逐内容行、绝对私有路径、PID、设备UUID或原图/标签/模型状态。

完整成功并提出假设：`R6D_COMPLETE_HYPOTHESIS_PROPOSED`。
完整成功但未形成新假设：`R6D_COMPLETE_NO_NEW_HYPOTHESIS`。
只有聚合：`R6D_PARTIAL_AGGREGATE_ONLY`。
完整性/机械/保护检查失败：`R6D_INCOMPLETE`。

共同状态：`new_model_execution_started=false`，`new_model_forwards=0`，`R6A=R6A_COMPLETE_NO_ADVANCE`，`R6B=NOT_RUN`，`R6E=PROPOSAL_ONLY_OR_NOT_PROPOSED`，`next_gpu_execution_authorized=false`。区分 analysis_executed 与历史模型 execution_started，不能把已经完成的 R6-A 写成 NOT_RUN。

## 9. 来源

[S1] 固定结果发布提交下 `docs/results/r6a_bounded_regional_v1/REPORT.md` 与 `EXECUTION_AUDIT.json`。
[S2] 原执行提交下 `src/dpa_ctta/r6_regional_consistency/loss.py`：已记录的权重、残差、点积、分区BCE等标量。
[S3] 原执行提交下 `src/dpa_ctta/r3/plan.py`：回访流唯一内容、64-content chunks、连续域段与域内顺序。
[S4] 原执行提交下 `src/dpa_ctta/r6_regional_consistency/analyze.py`：指标校验、只读联结、主指标与结果重写边界。
[S5] Center for Open Science, Preregistration：区分结果前规划与事后探索，保留并标明未计划分析。https://www.cos.io/initiatives/prereg

本方案的公式为上述标量定义的代数推导；后续R6-E为条件性研究提案，不是既有事实或实测结果。
