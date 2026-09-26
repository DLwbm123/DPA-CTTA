# R9 当前图适配优先：Screen24 结果分析与下一轮完整实验计划

**版本：** R9_CURRENT_FIRST_V1  
**状态：** DESIGN_ONLY_NOT_IMPLEMENTED_NOT_AUTHORIZED  
**事实来源：** DPA-CTTA `a9d3a43959da73ef552401342cc087d5ae8e6ba2`，Screen24 完成报告及代码。  
**本包没有启动模型、加载训练权重、读取医学图像、修改仓库或创建监测。**

本轮不盲目执行原 65-source / 724-target 容量网格，也不因 Screen24 负结果关闭整个 A/B 方法家族。保留原配方 16,000 步训练对照，把新增资源集中在一个已经核实的设计差异上：**源训练从 support 图像推断状态，却预测另一个 query；部署则观察和预测同一张图。** 在可解释的单因素比较之外，加入预先规定的低维目标梯度、同权重历史重置和参数化对照。

所有新的阈值、预算、实验臂与选择规则都是本计划的提议，不是已完成实验的事实。正文引用 [S1]–[S8] 对应文末固定来源；机器配置与展开任务位于 `R9_SPEC_AND_MATRIX.json`。

---

## 1. 已完成实验：精确结论

### 1.1 实际完成范围

Screen24 是 **10 个源训练任务，每个 4,000 步；40 条目标轨迹及其独立评分**，不是原长程 R8 全部任务。两种目标顺序为 orders 0/1；每条处理 1,951 次到达，主要评分子集 1,695 个内容。累计 78,040 次到达、67,800 条主评分，OD/OC 共 135,600 条通道观察，不是独立患者数。[S1]

源训练种子为 20260924、20260925；原生 VPTTA/C/G 种子为 20260907、20260908。目标模型是 A32 / B64、FiLM amplitude=0.3；A/B 的实际观察器、辅助项等应从执行配置读取，不能由缩写推断。[S1]

主指标沿用：先在域内平均，再对四域、OD/OC、orders 0/1、可用种子等权。按图像池化的平均仅为次要描述指标。[S1]

| 方法 | 主 Dice % | OD % | OC % | 相对 C0，pp | 相对 VPTTA，pp |
|---|---:|---:|---:|---:|---:|
| C-CTTA | 78.5341 | 86.8693 | 70.1988 | +3.4547 | +4.1330 |
| G-CTTA | 77.2606 | 85.7345 | 68.7867 | +2.1812 | +2.8595 |
| C0 | 75.0794 | 83.2699 | 66.8890 | 0 | +0.6783 |
| VPTTA | 74.4011 | 82.8037 | 65.9985 | −0.6783 | 0 |
| CURRENT-MLP | 74.3660 | 83.6284 | 65.1036 | −0.7134 | −0.0351 |
| A_STATIC | 74.2142 | 82.9639 | 65.4646 | −0.8652 | −0.1869 |
| A_FULL | 74.0085 | 83.0242 | 64.9929 | −1.0709 | −0.3926 |
| B_STATIC | 73.5099 | 83.0195 | 64.0004 | −1.5695 | −0.8912 |
| B_FULL | 73.2491 | 82.9368 | 63.5614 | −1.8303 | −1.1520 |
| N_SOURCE_EVAL | 68.2006 | 78.7258 | 57.6754 | −6.8788 | −6.2005 |

来源 [S2]；差值是对公开汇总的算术计算。R7_C_FULL/STATIC 约 75.0794/75.0803，仍是接近 C0 的冻结参照。[S1]

**当前判断：A 比 B 高 0.7594 pp，但不能据此提名为有效新方法。** 两条 FULL、两条独立 STATIC 和 CURRENT-MLP 都没有超过 C0；A/B FULL 也都没有超过同期 VPTTA。此前 R7 相对 VPTTA 的小幅正差不能移植为 Screen24 的结论。

### 1.2 历史不是唯一、也不是最大的已观察问题

A_FULL−A_STATIC = −0.2057 pp；B_FULL−B_STATIC = −0.2608 pp；两个源种子的方向一致。[S1–S2]

可以作如下代数拆分：

- A_FULL−C0 = (A_STATIC−C0) + (A_FULL−A_STATIC) = −0.8652 −0.2057 = −1.0709 pp。
- B_FULL−C0 = −1.5695 −0.2608 = −1.8303 pp。

这**不是因果贡献率**，因为 FULL 与 STATIC 有独立训练权重。它只表明：没有跨图历史的完整逐图适配版本本身已明显低于 C0。因此下一步不能只在“忘记历史/保留历史”之间调参数。

### 1.3 主要损失集中在 OC

相对 C0：

- A：OD −0.2457 pp，OC −1.8961 pp。
- B：OD −0.3330 pp，OC −3.3275 pp。
- CURRENT-MLP：OD +0.3585 pp，OC −1.7854 pp。[S2]

因此应把 OC 的概率分布、前景面积、FP/FN、边界和软/硬 Dice 一起报告。现有聚合不能证明是“杯扩大”“杯缩小”或某种病理结构误差，不能凭 Dice 编造原因。也不建议直接依据目标 OC 分数更改 OC 权重或分割阈值。

### 1.4 分域收益与损失被放大

下表为两个源种子和 orders 0/1 等权平均，单位为百分点，差值 FULL−C0：

| 域 | 内容数/轨迹 | A Macro Δ | B Macro Δ | A OC Δ | B OC Δ |
|---|---:|---:|---:|---:|---:|
| Drishti_GS | 37 | −4.0676 | −6.3144 | −7.3582 | −11.6455 |
| ORIGA | 586 | −2.0369 | −4.0481 | −3.6432 | −7.2738 |
| REFUGE | 336 | −0.7209 | −1.8465 | −0.9503 | −2.7109 |
| REFUGE_Valid | 736 | +2.5419 | +4.8879 | +4.3674 | +8.3201 |

来源 [S3]。这是分域平均，不是报告中的“单个种子最差域/order”。

A、B 都在一个大域改善、三个域下降。REFUGE_Valid 占 736/1695≈43.4%，Drishti_GS 仅占约2.2%。所以报告里 pooled A/B−C0 分别为 +0.168/+0.219 pp，却不能替代主指标的 −1.071/−1.830 pp。[S1]

B 的 pooled A/B 比较中位数、负向尾部也不能冒充域等权提升。B−C0 的 pooled 最差10%约 −12.636 pp；A 约 −6.817 pp。该分布以内容×通道×顺序×种子为观察单位，不是独立患者样本。与 VPTTA 的 ASSD 配对共同有效数为13,510，另有50未定义，不能填零后比较。[S1]

### 1.5 4,000 步不是收敛结论，源代理也不是目标上限

实际代码 warmup=200，之后沿 16,000 步 cosine 日程：
`lr = 3e-5 + 0.5*(3e-4−3e-5)*(1+cos(pi*(step−200)/15800))`。
第4,000步约 `2.6326e-4`，终点才是 `3e-5`。[S4]

这说明短筛查没有覆盖完整降学习率阶段；**不说明训练更久必然改善目标性能**。需要继续到16k的原配方对照回答这个问题，不能凭直觉选择结论。

64个固定 source-val anchors 的平均 query soft Dice：
A32 direct latent 相对 zero +0.102 pp；B64 +0.258 pp；ambient oracle约 +0.084 pp。[S1]
这些结果提示目前源模拟任务中可转移的监督干预收益很小，但不是目标域硬 Dice 上限。投影/低维优化能超过有限优化的 ambient oracle，更说明“oracle”不是全局最优证明。

### 1.6 优先检验的源/部署条件差异

[S4] `SourceTrainer.fit_step`：
1. 读取并扰动 support；
2. 用 support 的 raw/tokens 更新 state；
3. 把这个 state 用在不同分组的 query；
4. query 标签仅进入离线 loss。

[S5] `OnlineHost.step`：
1. 观察当前图；
2. 更新 state；
3. 用这个 state 预测同一当前图。

跨患者 source-query 训练是原来刻意设计的迁移约束，不是自动成立的实现 bug。但它对“根据当前患者图像作即时调整”可能构成不匹配。CURRENT-MLP 也接受这个训练方式，所以其失败不能当成“正确训练的逐图条件化必然无效”的证据。

---

## 2. R9 的研究假设和范围

### 2.1 三个可区分假设

H1：较长训练和完整学习率日程可以改善原配方；用 LEGACY-16k 检验。  
H2：观察/预测同一图的部署对齐训练可以改善逐图适配；用 SELF−LEGACY 检验。  
H3：对小收益源代理的辅助拟合干扰任务学习；用 SELF_TASK−SELF 检验。  

辅助问题：部署校准是否改变有效调制幅度？当前图无标签梯度能否修正源端预测误差？BN 参数化与固定 FiLM 子空间在哪些条件下表现不同？

这些都是待检验假设，不保证得到正结果。

### 2.2 固定不改的东西

保持原源 checkpoint、ResUNet34、512×512输入、OD/OC标签和评分、current-statistics BN、源分组、source simulator、A32/B64容量和0.3幅度。复用已验证的 oracle、基和scaler；不重新开一次昂贵的全源准备网格。

不新增几何增强、分割头、OC阈值搜索、target-label gate、目标风格供体或模型集成。C/RBE保持冻结，不重新扩容。不能把所有因素同时改掉后将收益归给“历史更好”。

### 2.3 数据用途

既有源 fit/cal/val 的组隔离保持；source-val明确用于选择，不是独立测试。当前目标池已被反复用于研究开发，不能重新命名为盲测。未来独立患者确认不假定已经存在，本轮所有结论限于开发数据和预定预算。

目标在线API只接收当前图像；目标标签只在完成封存后进入隔离评分。**本轮所有选配方、选checkpoint、选LR均在本轮目标分数解封前冻结。** 不能说研究方向完全没受到过去目标结果启发，但不能让新一轮目标分数进入自动搜索。

---

## 3. D0：固定 Screen24 权重的机制诊断——40个目标评估槽位

使用 Screen24 已冻结 A/B FULL 的两个种子，orders0/1。普通表现不好不取消后续训练；分数隔离直到源选择完成。

### 3.1 六个部署消融，24槽位

| 消融 | 改动 | 不改变 |
|---|---|---|
| A_RESET | 每张图重置m/P | FULL权重、基、cal参数 |
| A_PRECAL | tau=1 | 其余FULL任务模块及历史 |
| B_RESET | 每张图重置z/d历史 | FULL权重和五步ISTA |
| B_PRECAL | kappa=1 | 其余FULL，按kappa同步计算ISTA步长 |
| B_PRED_ONLY | delta=0 | FULL预测器和历史 |
| B_ISTA20 | 固定20步近端迭代 | 能量定义、FULL权重、历史 |

PRECAL不是重新训练，也不能改tau/kappa后忘记相应数值尺度。RESET不是独立训练的STATIC。

### 3.2 输出强度探针，16新增槽位

A/B分别固定 alpha=0.25、0.5；两个源种子×orders0/1。
定义：
`h_alpha = h + alpha * [expm1(0.3*tanh(v_gamma))*h + 0.3*tanh(v_beta)]`。

**只缩放输出调制，不缩放持久z、不改变观察器或递推。**
alpha=0对应C0路径，alpha=1对应原FULL；只有完整输入/BN/预处理/数值路径等价时才能复用原结果。

不从目标分数挑alpha作为发布配置。若alpha=0最好，只说明更接近C0更好，不是发明了有效新方法。

### 3.3 记录

逐visit仅存私有标量和必要低维状态：z/v范数、增量、tanh饱和比例(|v|>3)、FiLM残差RMS、prior项；A的traceP/Q/R；B的kappa、能量、稀疏率与校正强度。只用评分端标签记录OD/OC FP/FN、precision/recall、硬/软Dice。不能把logits前景统计当作准确率或用作按域选方法的真值代理。

---

## 4. 源端训练：14种配方、28个首批任务

A/B各3种训练配方，每种FULL/STATIC；另有两种CURRENT-MLP。各用原两个源种子20260924/25。

| 配方 | 状态的观察图 | 分割损失对应图 | 辅助拟合项 | 初始化 |
|---|---|---|---|---|
| LEGACY | styled support | 不同group styled query | 原定义 | 验证后从原fit step4000分叉继续 |
| SELF | 当前styled query | 同一styled query | 与LEGACY相同 | 原相同种子初始化，从0训练 |
| SELF_TASK | 当前styled query | 同一styled query | fit阶段去掉所有原辅助项 | 原相同种子初始化，从0训练 |

**SELF的标签只参与离线loss，不进入状态更新函数。**
A的clean/styled consistency参考也改为同一当前query的clean版本；不能继续把support的clean内容拿来约束query。B/MLP使用同一当前图的观察。源分组和光度调度保持，不引入新的目标信息。

SELF_TASK的校准流程仍与其他组匹配，单独通过PRECAL测试校准影响。因此“TASK”仅表示主fit目标为原OD/OC同权 `seg_loss`，不是取消所有源校准监督。

两种MLP：
- MLP_LEGACY：保留原跨图支持训练，继续到16k。
- MLP_SELF：同样的MLP/基/容量/优化预算，观察并预测当前query，无跨图适配状态。

这不是完整2×2因子设计，不能声称估计训练对齐与辅助项的全部交互作用。

### 4.1 原配方继续训练的严格条件

10个LEGACY源任务使用**主fit阶段 step4000 的完整快照**，而不是仅有cal后的部署权重。恢复模型、AdamW、step/LR、RNG、TBPTT位置、历史状态、数据调度和artifact绑定。新任务另建R9目录，保留Screen24原任务与费用。

如果完整快照不可用，不能用部署权重加空Adam冒充等价继续。预登记替代是同种子从0独立训练16k，单独记录“fresh替代”；最坏新增fit更新从648,000变为688,000。采用替代前须确认资源上限容纳该分支。

### 4.2 所有任务完成16,000步

- AdamW；peak LR3e-4、final3e-5、warmup200，原16k cosine日程。
- wd1e-4，betas(.9,.999)，eps1e-8，global grad clip1。
- 每条源序列32visit，8visit TBPTT；每8visit一次loss.mean/backward/update。
- FULL跨TBPTT边界只detach，不清状态；每32visit episode重新初始化。
- STATIC每visit初始化，保留当前图的适配求解。
- batch1；主干/MLP float32，必要小矩阵float64，无AMP。
- 保存4k、8k、12k、16k。所有任务跑至16k，不因中间目标/源分数差而科学早停。
- 观察特征来自冻结零干预主干；query分割计算必须对新模块/状态可微。禁止错误no_grad覆盖整个任务图。
- SELF_TASK不必悄悄保留为辅助目的而训练的loss；额外只读诊断前向必须计费。

### 4.3 源端选择，不能选目标最优

每个checkpoint独立执行256步原方法校准，避免后一个checkpoint复用前一个的cal状态。验证为64个固定source-val 32visit episode，每类序列16个；使用**部署对齐的同图观察/预测**。旧跨support/query验证另报为诊断，不混用。

每个sequence-mode先对内容、OD/OC平均，然后：
`S = 0.5*mean(mode_mean_hard_Dice) + 0.5*min(mode_mean_hard_Dice)`。

本轮明确将这个部署对齐的hard-Dice源选择规则作为新协议；它不同于Screen24的soft-Dice选择，不能隐瞒。soft Dice仍完整记录。

每个模型从4个checkpoint中选S最高；差≤1e-8选更早。每条路线用首两种子的FULL与STATIC所选S平均挑一个训练配方，平局顺序LEGACY→SELF→SELF_TASK。FULL/STATIC可以各自选checkpoint，但必须属于同一个入选训练配方。

这只是供下一阶段固定执行的配方选择，不签发科学winner，也不要求某组达到正增益才继续。对A、B各入选一组，并固定MLP_SELF，进入种子确认。

### 4.4 三个新增种子

20260926、20260927、20260928，各训练：
A所选FULL/STATIC、B所选FULL/STATIC、MLP_SELF，共15个16k任务。从0开始，不以旧seed权重热启动。不得在看到这些种子的目标分数后改配方。

**源任务总计43个：10个继续任务、18个首批新训练、15个确认训练。**
正常新增fit更新648,000，query访问5,184,000；不含校准、验证、诊断、在线优化等，它们另计。

---

## 5. 无标签目标梯度：六个固定臂，150槽位

这不是另起一个任意TTA算法网格。优先复用[S6]中已实现但未被Screen24运行的低维诊断结构，并增加匹配RESET/冷启动/BN控制。

| 臂 | 每图初始化 | 优化后状态跨图保留 | 步数 | F/B/Adam |
|---|---|---|---:|---:|
| B_G1 | B提出的状态 | 是 | 1 | 8/1/1 |
| B_G3 | B提出的状态 | 是 | 3 | 10/3/3 |
| B_RESET_G3 | FULL权重但每图清历史后提出状态 | 否 | 3 | 10/3/3 |
| COLD_G1 | z=0 | 否 | 1 | 8/1/1 |
| COLD_G3 | z=0 | 否 | 3 | 10/3/3 |
| BN_RESET_G1 | 每图恢复源BN affine及空Adam | 否 | 1 | 8/1/1 |

每个臂5个随机种子实现×5个域序。COLD使用同一B控制空间；BN控制只允许更新已注册标准BN affine，卷积与head不变。共享基导致某些确定组件在种子间相同，应如实报告。

### 5.1 固定目标与优化器

沿用[S6]的六视图**零调制C0网络**软目标：原图加5个几何视图、逆变换、sigmoid后平均且detach。不是继承已适配教师的输出，不额外引入未来图/目标供体。零干预原图前向同时作为观察器前向，不能重复计为免费。

同一张图使用一个固定强外观增强，K步重复使用它；归一化顺序复用已固定R8/GraTa路径。

低维优化：
`u=z/s`，s来自source-fit oracle的既有尺度约定、floor1e-3，digest冻结。
`L = mean BCEWithLogits(strong_logits,q_six_C0) + 0.01*mean((u-u0)^2)`。
每图新建Adam，betas(.9,.999)、eps1e-8、wd0。

latent LR候选为{0.001,0.01,0.1}。每个臂、每个候选LR只在64个固定source-cal四visit episode上用标签评分和同一平衡规则选择，平局更小LR。BN_RESET_G1候选为{1e-5,1e-4,1e-3}，使用同一BCE目标；一阶prox在初值处梯度为0，不影响唯一第一步。未更新参数前六视图统一来自C0。

### 5.2 这些比较分别回答什么

- B_G1/G3 vs B_FULL：增加目标端无标签反馈的实际作用，不能归因于历史。
- B_G3 vs B_RESET_G3：同一权重、同一K，历史部署的变化。
- B_RESET_G3 vs COLD_G3：源端预测初始化相对逐图从零优化的价值。
- COLD_G1 vs COLD_G3：有限在线优化预算变化。
- BN_RESET_G1 vs COLD_G1：同一弱目标、同一1步/调用预算下的参数化比较；参数数目、几何和LR不同，不能称纯rank因果检验。
- BN_RESET_G1 vs原C：不能自动当作唯一历史因子；需同时检查原C教师和参数历史差异。

B_G1/G3不再是gradient-free；改善不代表零backward主方法成功。冻结backbone也并不要求禁止所有latent梯度，原生VPTTA同样包含逐图优化。[S8]

---

## 6. 目标矩阵与重复单位

### 6.1 主比较

| 阶段 | 公式 | 槽位 |
|---|---|---:|
| 首批源配方 | 14配方×2种子×orders0/1 | 56 |
| 入选首两种子扩展 | 5模型×2种子×orders2/3/4 | 30 |
| 三个新增种子 | 5模型×3种子×5orders | 75 |
| 最终同权重机制 | 6消融×5种子×orders0/1 | 60 |
| RESET扩展 | A/B两RESET×5种子×orders2/3/4 | 30 |
| 六个梯度臂 | 6×5×5 | 150 |
| 原生VPTTA/C/G | 3×5种子×5orders | 75 |
| N/C0 | 2确定性控制×5orders | 10 |
| 两类压力流 | 见6.3 | 84 |
| Screen24固定权重诊断 | D0 | 40 |
| **核心合计** | | **610** |

基线种子20260907–20260911。源种子20260924–20260928。ordinal配对只作描述，不假装两个算法使用相同RNG路径。VPTTA相同聚合不当作多份独立患者证据。

### 6.2 固定16k终点敏感性

主比较中161个源模型评估槽位同时预留固定16k终点。若源选择已选16k且完整artifact一致，复用结果；否则新增运行固定16k版本。

**最多161个新增槽位，总上限771。** 不通过看目标最佳checkpoint来选择主结果；主结果始终是预定源选择版本。报告终点与源选择版本的差异，避免“总是能挑到一个好checkpoint”的不透明自由度。

### 6.3 混合与长流

五个入选源模型（A/B各FULL/STATIC＋MLP_SELF）和原生VPTTA/C/G，共8类×5种子×2流=80；N/C0×2流=4，合计84。

- MIXED：按`SHA256(R9_MIXED_V1|registration_digest|content_identity)`固定排序1,951次到达，平局原manifest序号；同一流给所有方法。方法不接收身份或域信号。
- LONG10：order0连续十轮，19,510次到达，轮间不重置历史；STATIC/冷启动按其定义仍逐图重置。记录每轮结果，不挑最好轮。每visit用独立访问序号，内容ID可回访但不得误去重整轮。
- 本轮不把梯度臂都扩到LONG10，防止在线计算矩阵无限膨胀。当前stress方法在计划中固定，不按targetwinner挑选。

在没有复用或失败的情况下：
核心1,927,588次到达、1,674,660条主要评分访问；
加入全部固定16k槽位，最多2,241,699次到达、1,947,555条主要评分访问。
这些数包括重复域序、种子、回访与十轮流，不是新图数或患者数。

### 6.4 复用政策

旧结果只在源checkpoint、源码路径、算法、预处理、目标manifest完整顺序、seed/RNG、artifact、评分版本全部一致时复用。Screen24同order基线可能满足，新recipe/新校准不自动满足。禁止“只把old分数搬进新表”却不标历史来源。

展开JSON中的数字是**评估槽位上限**，不是承诺771条新的物理运行。复用项记录原receipt和零新增模型调用，实际物理账本另报。无论复用还是新跑都不得重复计为独立证据。

---

## 7. 数学/程序检查与可归因性

实施前只跑有针对性的合成/程序验证，不要求复跑所有历史测试：

1. SELF观察器读取的图像必须与该visit分割图像一致；同图ID/字节测试不依赖标签。改变label只改变source loss/评分，不改变同一步在线输入和预测。
2. LEGACY保持原support/query约束；SELF改变条件方式后必须单列新方法身份，不冒充旧trainer bugfix。
3. A clean/styled参考来自同一图；source loss梯度到新模块、FiLM状态，不到冻结原网络参数。
4. FULL跨TBPTT边界不清state；STATIC和同权重RESET逐图清历史，且RESET不能加载STATIC权重。
5. PRECAL的尺度/步长正确；ISTA20精确20步而非能量早停；不伪称精确argmin。
6. output-alpha只改变最终FiLM残差；alpha0零干预路径数值等价须实际验证。
7. calibrated artifact与pre-cal fit snapshot分别保存；禁止把calibrated温度混入等价续训optimizer历史。
8. 8F/1B、10F/3B等逐图真实计数；六视图sigmoid/inverse/预处理与独立参考一致，BN_RESET恢复完整BN和空optimizer。
9. Source selection before target seal release；检查所有sourceval、LR和跨seed配方选择不读取目标分数。
10. 同模型save/restore下一图一致；断点后未提交尾部不能重复评分或将已评分目标加入训练。
11. 分域等权聚合必须通过不等域样本数的反例测试；pooled与primary字段不得相同命名。
12. ASSD undefined保留，按OD/OC共同有效cohort报告；失败臂不把工程缺失填成0Dice。

这些是实施验收，不是对模型有效性或临床安全的认证。

---

## 8. 资源和长期执行

### 8.1 提议上限，不是耗时保证

| 项目 | 提议上限 |
|---|---:|
| 同时GPU worker | 3，每GPU单个任务 |
| GPU-worker账本时长 | 512小时 |
| 输出空间 | 64 GiB |
| 模型forward | 32,000,000 |
| backward调用 | 3,000,000 |
| optimizer步骤 | 3,000,000 |
| VJP/JVP调用 | 4,096 |

模型调用预算包括新增fit、cal/val、梯度cal、online、smoke/profile和允许恢复；复用资产的历史成本另列。source fit的648,000不等于总backward数。

启动前基于最重A/B fit、sourceval、gradient、评分/IO的实测profile，展开计算证明确切配置装得进预算。若不够，必须在发射前修订并冻结明确的任务包；不能运行中偷偷缩短步骤、删种子而声称完整完成。固定16k敏感性和无完整snapshot时的fresh分支均须纳入容量核验。

512是拟议资源限额，不是预测将运行512小时；不要求耗尽，也不继承旧60GPU小时或GPU5/6/7权限。实际device/新SHA/私有输出root/数据绑定需在启动前填写，当前均未授权。

### 8.2 不重复本轮资源故障模式

Screen24记录46.277 GPU-worker账本小时，完成attempt之和33.824小时，另含失败启动/保留预约等成本；不能把这些全叫纯GPU计算。[S1]

- 基于已有verified资产复用oracle、basis、scaler，不再重复整个准备网格。
- host-local锁，带lease/token的任务所有权，原子写commit；网络存储仅作受控持久化，避免再次依赖不受支持的NAS锁行为。
- 估计单任务耗时是soft进度参考，不能因为scaler达到估计时间就自动终止。用户停止、隔离违规、数值失败和总硬预算才是不同类型的停机条件。
- 物理GPU计时、worker预约/elapsed、CPU评分、IO、峰值显存、失败预约分别报告，不做虚假的纯算力加速比较。
- GPU任务完成后交给CPU评分队列，及时释放GPU；计费方式若仍按worker预约必须明确。
- 每个任务最多一次预授权等价快照基础设施恢复；保留失败receipt和所有成本。
- NaN/Cholesky非SPD、绑定错误、数据越界不能通过改LR/ridge/seed/AMP自动修复。共性代码或数据隔离故障应阻断受影响的整条依赖路径；独立任务的数值失败按预登记规则保留且不覆盖，其他确实独立任务可继续。
- 不把普通负性能当故障，不因结果差重抽seed、追加算法、删除域。
- 不创建小时监测或heartbeat，用户已删除旧监测。

---

## 9. 报告与研究决策

### 9.1 必须交付

主指标与所有seed/OD/OC/domain/order绝对值；FULL−STATIC、FULL−RESET、方法−C0/VPTTA/C/G/MLP；配对尾部和负向比例；各域OC FP/FN/前景统计；ASSD有效分母与undefined；PRECAL、alpha、ISTA诊断；4k到16k源学习曲线；源选checkpoint与固定16k；LONG10逐轮；实际物理成本、故障与复用映射。

首两seed用于配方选择，后三seed用于固定配方重复；五seed均可描述，不冒充五个新患者群体。用患者/眼group bootstrap仅在可验证分组和估计量定义允许时使用；不能以像素、通道、回访和域序当独立样本制造显著性。静态保存预测上的bootstrap不等于重跑适配历史的流不确定性。

### 9.2 预定解释

| 观察 | 下一步含义 |
|---|---|
| LEGACY16k明显优于4k | 训练长度有价值，但仍需超过C0和强基线，不能自动宣称历史有效 |
| SELF优于LEGACY | 支持部署条件对齐的具体实现；不是原代码错误的证明 |
| SELF_TASK优于SELF | 支持主fit辅助项需要调整；不能归因于某一个被同时移除的辅助子项 |
| MLP_SELF/STATIC达到或超过FULL | 保留有价值的逐图适配，取消没有证据的历史贡献主张 |
| FULL同时优于独立STATIC与同权重RESET | 获得更强历史效用证据，继续检查域间OC/尾部与长流 |
| 只有梯度臂提高 | 接受少量latent优化方向，不称为零backward收益 |
| BN_RESET_G1优于COLD_G1 | 参数化或优化几何值得重新考虑，不能只盯历史；并非证明所有FiLM空间无效 |
| alpha0最好或所有新方法≈C0 | 不能把退回零干预叫方法创新 |
| 长训练、部署对齐和有限梯度都未获净收益 | 本轮预算内关闭当前源摊销FiLM作为主方法的继续扩容；保留失败，不把有限预算当理论不可能性 |

研究进展参考：相对**同期C0和VPTTA均≥+0.5pp**，且5个seed中至少4个平均正差；这只是人为预先规定的进展目标，不是统计/临床保证，也不用于提前取消已登记任务。超过C/G则是更强性能目标，需同时披露额外源成本与在线计算。所有方法即使达不到参考也按协议报告。

目标数据已经用于开发；任何“超过VPTTA”的结论限于本协议，不自动推广到原论文全部任务、独立患者或临床部署。模型选择与TTA比较本身需要严格说明选择集及预算。[S7]

---

## 10. 实施边界与交付

建议新分支名：`experiment/r9-current-first-v1`，只作为建议，不已创建。
先实现/复核本包定义，发布新的exact SHA，绑定数据/快照/GPU资源和输出目录后才能执行。旧Screen24报告commit和runtimecommit不互相冒充。原32/64维基与source来源需明确hash。

自动化是一个预先审阅过的有限任务依赖图，不是无限搜索。新目标分数不回流选择器；没有外部审阅不得写external PASS。原R7/R8停止/恢复记录不重写。

本包提供规划配置、展开矩阵和规划一致性验证脚本，不提供已实现的训练runner。`verify_plan.py`仅验证计划的计数/字段/算术，不能当作医学模型测试或GPU执行证据。

## 来源


[S1] Screen24 完成报告：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/docs/review/r8_screen24/REPORT.md

[S2] Screen24 主表 MAIN.csv：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/docs/review/r8_screen24/MAIN.csv

[S3] Screen24 分域/通道 DOMAINS.csv：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/docs/review/r8_screen24/DOMAINS.csv

[S4] 实际源训练器 trainer.py：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/src/dpa_ctta/r8_ba/trainer.py

[S5] 实际部署接口 host.py：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/src/dpa_ctta/r8_ba/host.py

[S6] 现有有限目标梯度 gradient.py：https://github.com/DLwbm123/DPA-CTTA/blob/a9d3a43959da73ef552401342cc087d5ae8e6ba2/src/dpa_ctta/r8_ba/gradient.py

[S7] On Pitfalls of Test-Time Adaptation（原始研究）：https://arxiv.org/abs/2306.03536

[S8] VPTTA（CVPR 2024，官方会议论文页）：https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Each_Test_Image_Deserves_A_Specific_Prompt_Continual_Test-Time_Adaptation_CVPR_2024_paper.html
