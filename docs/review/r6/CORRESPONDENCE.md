# R6 原件逐项对应与科学差异审计

核对对象：原件 `originals/r6_plan/`，生产代码 `d875f20c11cc7e617c9f39dba04ed46381aba450`，补交候选 `c94fff7c05cec38d541c441b62a9381ee46ba076`。两候选之间生产 src、现有 scripts/tests 和 execution defaults 无变化；新增科学配置为此前缺失的原提案原样副本。新增比较脚本不在生产导入链中。

原始 MANIFEST 的全部 11 项均逐个验证长度及 SHA256；MANIFEST 自身与补交提取记录的摘要一致。12 个原件及额外 ORIGINALS_VERIFICATION.json 均与本次 ZIP 对应成员逐字节一致。完整结果见 [机器核对记录](ORIGINALS_VERIFICATION_CURRENT.json)。原始设计 ZIP 未在本轮独立取得，补交记录中该设计 ZIP 的摘要只是原记录，不冒充本轮对它的重新核验。

原 prompt、plan 还与 f9ec00bb 已归档的两文件逐字节一致；本次没有更换科学版本。原始 JSON 的 DESIGN_PROPOSAL_NOT_IMPLEMENTED 是历史状态，保持原值。原始数学最终日志、first_attempt 日志和 first_reference.py 均保持原样，历史首次失败仍可审阅。补交 ORIGINALS_VERIFICATION 只证明提取，不是新的数学或实现通过凭证。

以下“完全一致”指科学定义和协议语义；不默认意味着每条等价浮点表达式 bitwise 一致。配置文件本身是逐字节一致。

| 原提案字段 / 规则 | 生产实现与证据位置 | 分类与结论 |
| --- | --- | --- |
| schema、status | configs/r6_science_v1.json；plan.science | 完全一致：保留原始 proposal schema 和历史 DESIGN 状态；当前实施状态独立写 DELIVERY |
| execution_enabled、background_authorized | 原 JSON false；execution defaults enabled=false/background_allowed=null；plan.authorize 先拒绝 | 仅实现/绑定字段差异：null 表示未授权，未把 false 改为授权；正式运行仍为零 |
| base_publication_sha、historical_r5a_execution_sha | plan.BASE；历史报告及当前 lineage | 完全一致：e271098e…基线，b4b71601…历史 R5-A；不把历史 SHA 当 R6 候选 |
| registration_digest、stream_digest、dependencies | plan.REGISTRATION/RECURRENCE；r1.plan REF/GRATA；b1_host.official | 完全一致：原登记及依赖提交均保持；登记 dry-run 只读元数据 |
| arms | loss.ARMS；plan.matrix | 完全一致：C / R_BAL / R_SCALE / R_SHUFFLE，无第5臂 |
| stages A / B_NEW / AB | plan.matrix、analyze.reuse_A | 完全一致：orders 0,1,4→12；2,3→8；0..4→20，AB 中12条复用 A；不是新增20条 |
| groups_per_trajectory、remaining_dev、adapt_all_contents | plan.stream / dry_run；execution.trajectory | 完全一致：每条1951全部更新，1695只作主评分筛选，不缩小适配流 |
| seed | execution.smoke / trajectory seed_all(20260907) | 完全一致：独立轨迹同seed，跨域无reset |
| optimizer 全部字段 | b1_host.configure / Host；r6 Host；完整网络 CPU 测试 | 完全一致：Adam 1e-4/(.9,.999)/1e-8/wd0；41 BN、19136 affine、float32、无AMP、每访问1次更新 |
| rule.partition、grid | loss.weights；Host._criterion | 完全一致：实际软q的q>=.5仅构造分区，生产1×2×512×512 |
| rule.ratio_cap、ratio | loss.partition | 完全一致：clip(n_bg/n_fg,1/8,8)，无搜索；runtime名rho对应原ratio |
| rule.w_bg、w_fg | loss.partition / weights | 完全一致：N/(rho*n_fg+n_bg)、rho*w_bg，显式double赋值 |
| rule.empty_partition | loss.partition | 完全一致：任意一分区空，该通道全部1；ONE_PARTITION_EMPTY |
| rule.normalization | loss.weights；tests/test_r6_loss | 完全一致：理想通道mean=1；应用float32舍入值独立审计；限幅时不声称严格50/50 |
| rule.placement、pos_weight | loss.objective | 完全一致：整项 BCE 同乘，不使用pos_weight |
| rule.q_modification、output_modification | Host._criterion；evaluation.current；固定 GraTa cal_consis_loss | 完全一致：保留完整软q和post学生logits；不改target/输出；GT提交后才读取 |
| C 原标量路径 | loss.objective 的 torch.nn.BCEWithLogitsLoss()；旧/新C完整模型parity | 完全一致：未把C替换为新归约；共同输入C loss/gradient均bitwise一致 |
| controls.norm_space、matching | loss.objective；Host._criterion readonly gradient hook | 完全一致：仅各自当前z/q逐通道strong-output logit范数；不匹配参数/Adam位移/跨轨迹 |
| controls.delta | objective d=(sigmoid(z.detach())-q.detach()).cpu().double() | 完全一致：减法先在loss dtype，再CPU float64 |
| controls.S0、Sw、Sperm | objective CPUdouble归约；原reference.coefficients | 完全一致：用实际已舍入权重；共同输入三能量逐值一致 |
| controls.R_SCALE | 生产 mean(channel BCE)*a 再通道平均；reference.loss 先逐像素乘a再全局mean | 仅实现字段/浮点求值顺序差异：JSON及计划明确通道mean后乘；实数公式相同，舍入顺序不同，240共同输入比较使用原冻结容差，不要求SCALE bitwise；无科学公式变更 |
| controls.R_SHUFFLE | objective (bb*pd*per).mean；reference.loss | 完全一致：b*w_perm 均detach，使用同一实际舍入顺序 |
| 权重 / a / b detach | objective无梯度CPU权重、math.sqrt及新tensor倍率；固定GraTa no_grad q | 仅实现/绑定字段差异（合法生产输入语义一致）：辅助函数输入检查不同，reference.checked_prob 拒绝 requires_grad q；低层production.weights会detach用于权重但不单独拒绝grad-q；生产实际q由固定GraTa no_grad构造。共同输入均为真实合同中的detached q，不声称辅助API对非法输入完全等价 |
| controls.zero_residual、positive_residual | loss.scales | 完全一致：S0==0要求Sw=Sperm=0，倍率1；非正/非有限硬失败，无epsilon、裁倍率或跳Adam |
| controls.extra_backward_or_vjp | objective代码；R6 CPU no-VJP/physical tests | 完全一致：生产额外backward/VJP=0；新数学对照用autograd作验收，不是生产成本 |
| shuffle.domain、implementation | objective专用CPU Generator+randperm(all H*W) | 完全一致：全通道置乱，基础直方图保持；b后不承诺绝对直方图等值 |
| shuffle.seed_encoding | permutation_seed / original.local_seed | 完全一致：ASCII R6_WEIGHT_PERM_V1\|20260907\|visit\|channel，SHA前8字节big endian低63位；visit从1/channel从0 |
| shuffle.identity_input、uses_global_rng | objective输入只有visit/channel；专用CPU RNG | 完全一致：无ID/domain，全局RNG不改变；共同输入检查了RNG/seed/置乱字节摘要 |
| primary、recurrence_separate | analyze.gate_inputs / summarize | 完全一致：remaining_dev、每图OD/OC平均、四域等权、主序等权；order4单列，不混入主序均值 |
| A_gate 全部字段 | analyze.gate；完整A账本验证先于gate | 完全一致：均值.5/.2/.2pp；每主序每配对>=0；recurrence>=-.1；每域两序>=-2；全12条完整 |
| B_gate 全部字段 | analyze.gate / reuse_A | 完全一致：均值.5/.2/.2pp；每配对>=3/4严格正；最差C>=-.5；每域四序>=-2；recurrence>=-.1 |
| budgets A / B_NEW / AB | plan.dry_run，逐字典对原JSON比较 | 完全一致：23412/15608/39020 visits，187296/124864/312160 forwards；backward=Adam=visits，VJP0 |
| smoke_proposal.recipe | 原OLD_C4_AND_FOUR_R6_ARMS4_V1；生产R6_ALL_ARMS4_OLD_C4_V1 | 仅实现/绑定字段差异：字符串别名。执行顺序实际OLD,C,BAL,SCALE,SHUFFLE，各4步；未改配额、算法或授权 |
| smoke其余字段 | plan.SMOKE；execution.smoke；smoke CPU故障/成功测试 | 完全一致：indices0..3/seed20260907；160F/20B/20Adam/0VJP每设备；只要求新旧C CPUexact、GPU拟议rtol1e-4/atol1e-5；不要求新臂等于C |
| resources.devices / max_workers / threads_per_worker / one_worker_per_device | defaults；plan.authorize；execution.context/worker | 完全一致：设备未授权null，最多3worker、每卡1、每worker2线程；未来授权不能突破worker数 |
| resources.proposed_hard_caps | plan.CAPS；defaults.proposed_caps | 仅实现/绑定字段差异：private_output_bytes→bytes；数值21600/86400/172800/8589934592全部一致；approved caps仍null |
| statistical_claim | 方法来源及报告边界 | 完全一致：仅development screening，不宣称显著性、临床安全或独立患者确认 |
| no_new_C0_or_source_run | execution只含4臂；本轮CPU资产隔离 | 完全一致：无新C0、源数据/代理/原型/重训；阶段I不读真实checkpoint |
| reuse_old_R4_R5_C_as_new_controls | matrix/reuse_A及标记/绑定验证 | 完全一致：同批新C；旧R4/R5只作背景，不计入R6 |
| automatic_B、automatic_retry | analyze next_execution_authorized=false；execution有限监督/失败保留 | 完全一致：A完成仍停，B需新绑定；本次不生成授权/receipt，不重跑真实轨迹 |
| code/science/registration/stream/fingerprint/scope bindings | plan.authorize、execution.context；check_originals_cpu | 仅实现/绑定字段差异：实施SHA/指纹独立于原提案历史字段；原science摘要保持固定，config逐字节验证；错误摘要拒绝 |

**真正科学差异：未发现。** SCALE求值顺序、smoke recipe别名、caps字段名及辅助输入校验的边界均明列，没有通过改写原JSON或放宽容差消除差别。生产额外的rho_clipped、applied_w_*、权重checksum、实际hook范数等是审计字段，不进入新的目标或更新规则。该结论限于阶段I规范/程序化检查，不是有效性结论或外部review PASS。

共同输入：两dtype×10类（0/1/17/71/126/142/143前景、随机软q、精确零残差、微小残差）×3个visit（1/17/1951）×4臂=240组。检查loss、逐像素梯度、理想/应用权重、S0/Sw/Sperm、a/b、seed、置乱字节摘要和全局RNG。另有原数学8项的新执行及3个新检查方法（含402组proposal gate对照）；它们与原历史8项、166项实现回归分别保存，不能相互替代。

所有本次新检查在候选提交上运行，历史日志未修改；两端实际数值误差、退出码、时间见 [本次验收汇总](ORIGINALS_ACCEPTANCE.json) 和 logs/originals-redelivery/。未读取源数据、真实目标RGB/mask或已登记checkpoint，未调用GPU smoke。
