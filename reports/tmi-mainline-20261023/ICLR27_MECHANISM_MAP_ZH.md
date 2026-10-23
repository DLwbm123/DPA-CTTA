# ICLR 2027 投稿机制：原文、实际实现、失败证据与未采用项

与 [Pro主报告](PRO_HANDOFF_12H_ZH.md)配套；北京时间2026-10-23 10:00实验快照。以下引用的是本地投稿材料与已保存关键方法核查，不代表录用，本次没有在线重新核实审稿状态。拥有PDF不等于逐页精读，更不等于复现作者代码。原论文的性能数字不拿来替代本项目实验。

## A. 实质进入本窗口方法或实验设计的来源

### 1. Intervention-Dependent Selection Credit in Stateful Test-Time Adaptation

[OpenReview](https://openreview.net/forum?id=nKr4NcT4jj)。本次补核本地完整PDF的摘要、引言及干预定义；旧下载清单曾标MISSING，但后来已有PDF，不能继续沿用旧缺失状态。

**原机制：** 记录原生每阶段选择数量，以这些数量重跑各流；各臂根据自己的模型状态选原生排名或同数量随机项，交叉形成阶段因子实验，分解选择身份贡献与交互。进一步匹配平均损失权重，说明阶段“功劳”会随干预定义改变。独立轨迹、数量、权重和恢复规则都重要；负交互也不自动等于组合有害。

**实际采用：** r01–r04每头64区域选32、选中权重平均归一，设置同数量RANDOM和独立轨迹EPISODIC等读出；r05独立训练均匀融合，r06以后独立训练固定DIRECT+同残差CONTROL，避免把容量/监督/权重变化都归给选择。

**没有实现/不能冒称：** 没有完整复现原文两阶段2×2因子、Shapley归因及原始DeYO/EATA/SAR系统。r06的“源标签下单区域替换整头Dice信用”是本项目设计的监督目标，不是该论文现成算法；它也不等于原文stateful trajectory credit。r05以后已经无持久状态，不能借用原文stateful结论为本模型背书。

**结果与启示：** 匹配CONTROL改变了故事：r11对SOURCE平均+2.254pp，但对CONTROL仅+0.058pp。该论文最有用的作用是约束归因，而不是证明我们的选择器有效。

### 2. LSTA — Learning Where to Adapt: Learned Subspace Test-Time Adaptation for Medical Time Series

[OpenReview](https://openreview.net/forum?id=zqPPaYF6kt)。本地全文关键方法§3已核查。

**原机制：** 冻结源模型，在同受试者不相交support/query窗口上，以无标签support熵步和有标签query分类损失学习LayerNorm更新的低秩子空间；新受试者重置，只适应子空间坐标。

**实际采用：** r01–r04纠错器有16维状态及16×4可学习基，QR正交化后投影内层梯度；源标签通过外层任务损失塑造允许的更新方向。r03再用16个空间logit基增加状态可作用性。

**关键区别：** 我们更新的是附加纠错器状态，不是原LayerNorm参数；support/query是源图episode，不是经核实的同受试者窗口；无标签能量是预测一致性而非原熵。骨干冻结，没有实现完整LSTA医学时序系统，也未做原计划的学习低秩vs固定随机低秩全面匹配实验。

**结果：** r01–r04 FULL−NO_UPDATE分别为+.000294/−.023497、−.001956/−.025357、−.110909/−.060471、−.050290/+.032732pp。没有稳定更新收益，r05起撤除该部署更新路径。失败不能外推为LSTA原方法无效。

### 3. AR-JEPA — Adaptation-Ready JEPA: Source Training for Test-Time Adaptive World Models

[OpenReview](https://openreview.net/forum?id=7owSB84uah)。本地全文§3.1–3.4关键机制已核查。

**原机制：** 保持AdaJEPA部署更新算子，在源训练中展开同一更新，使用源侧物理状态及未来动力学监督更新后表示/功能。核心是监督可用性不同条件下，源训练塑造部署适应响应。

**实际采用：** r01–r04训练中真实展开无标签状态更新，以源标签监督更新后的分割；三图episode也训练对下一不同源图的任务收益估计，试图让状态传递与后继任务挂钩。

**关键区别及欠缺：** 我们冻结旧骨干，只训练附加纠错器/子空间，未实现此前设计中“改变源骨干初始化”的更强路线。静态2D图及同外加扰动不是世界模型的真实时序/物理后继；未实现Post-Phys/Post-Dyn本体。不能仅因存在内外循环就称完整复现AR-JEPA。

**结果：** 外层训练完成且有真实inner_grad，仍未形成稳定post−no-update收益。说明本实现没把源监督成功转成有效部署更新，不证明“训练适应响应”这个广义方向无效。

### 4. CP-JEPA

[OpenReview](https://openreview.net/forum?id=WhRpXMEjTr)。本地全文§3.2–3.4、任务适配及状态接口已核查。

**原机制：** 每体积拟合跨平面预测与容量匹配轴向控制，以真实观测方向差异构造证据；源监督任务适配器将其映射为有界logit残差，训练包含Exact/Axial/Shuffled/Zero等证据和identity/ranking约束。病例预测器丢弃，持续版本仅保留任务证据并先读后写。

**实际采用：** 源监督把候选预测/图像信息转成任务纠错的思想，及病例临时状态与跨病例状态分开的设计。r01–r04有临时更新与收益写入；r05以后只保留病例级前馈纠错。

**未采用：** 没有真实正交切面、跨平面预测器，也没有完整原证据生成/身份排名损失。canonical/翻转仍来自同一2D图、同一骨干，不能冒充独立正交观测。当前也没有完整Exact/Shuffled/Zero证据实验，因此源监督残差有用不证明图像条件证据独有增益。

**归因：** 有界纠错与状态分离不是我们的独有创新；当前最可靠的正结果恰恰集中在源监督残差这一已有近邻机制。

### 5. SIFT — Boundary-Free Online Adaptation via Semantic Isolation and Functional Testing

[OpenReview](https://openreview.net/forum?id=elH9mnInOk)。本地关键方法§4.1–4.4已核查。

**原机制：** 隔离暂存与成熟模块，语义邻近不足以合并，还须sentinel功能验证；原在线文本(x,y)提供答案token NLL用于训练/验证。

**实际采用层级：设计启发，而非算法移植。** r01–r04把当前状态与持久写入分开，以源端后继分割收益训练critic，部署预测收益>0才写入。

**不可偷换的条件：** 我们部署没有真实答案，critic预测不等于SIFT有标签功能验证。没有SIFT模块库或sentinel原系统。早期FULL−EPISODIC近零/不稳定，写入未获独立支持；r05起无持久写入。

### 6. When Is Test-Time Adaptation Identifiable from Unlabeled Evidence?

[OpenReview](https://openreview.net/forum?id=FHWxXiGOV7)。本地正文与既有可识别性核查提供概念边界；没有复现理论实例或证明。

**作用：** 无标签部署观测未必足以确定有益更新，不能把置信/一致性/能量下降等价为真实任务改善。源标签只在训练时提供额外知识，仍需明确目标部署证据通道及适用分布。

**落地：** 将低熵代理转向源任务收益监督，并保留强CONTROL与目标封存；它是问题定义/结论约束，不是加入了一个名为Identifiability的模块。我们没有得到本医学配置可识别的理论保证。

## B. 阅读过，但本窗口没有作为算法模块实现的2027来源

|来源与链接|核查到的机制/条件|本窗口实际状态|
|---|---|---|
|[DTS-Meta](https://openreview.net/forum?id=vzr0vrK407)|临时MetaNet内更新，同批增强query对齐detach教师，在线无标签外循环|用于快慢状态和在线元学习排重；我们的源标签外层不是它的原协议；未部署其MetaNet|
|[IRP](https://openreview.net/forum?id=Nw4Nr7QocE)|域返回恢复方向，对Adam实际位移作保护投影|作为历史状态风险提醒；无已证明有用历史，不加入域签名检索或保护库|
|[ENGRAM](https://openreview.net/forum?id=bkwlR0ND2g)|描述符索引低秩参数记忆|旧项目曾单独试验，本窗口不运行，不与16维状态混称；记忆活动不等于任务收益|
|[LQN](https://openreview.net/forum?id=Ay7ifrLrHJ)|冻结教师中间token、逐图拟合坐标条件学生及相关目标|只借鉴病例模块可丢弃这一资源分工；未实现其学生/PAD/LCR系统|
|[SEER](https://openreview.net/forum?id=hEBqQJhg1b)|SAM2候选与VLM闭集区域核验|作为不同证据通道备选；没有新增SAM2/Qwen、没有VLM医学核验结果|
|[SphereTrust](https://openreview.net/forum?id=dTDhhW11MJ)|选择/受限更新等近邻思想，需简单和随机控制|只有既有阅读和对照提醒；本窗口无该算法实现，不引用其结果证明本方法|
|[HRAC](https://openreview.net/forum?id=fb1drQMQmW)|空间对应/增强一致性相关机制|旧路线的历史背景；r07普通仿射一致性和r09翻转不等于HRAC复现，亦不构成新颖性|
|[RDC](https://openreview.net/forum?id=SzNyAJVTuM)、[Soft-PSCL](https://openreview.net/forum?id=ZNLqVK9tLm)|关系排序一致性、软目标等近邻|本窗口未运行其原损失/系统，不能列为本方法已验证组件|
|[UnfoldCRF](https://openreview.net/forum?id=5rrS3uswYR)|共享能量、局部pairwise、潜在区域/null、展开阻尼mean-field|已有正文核查，作为独立结构纠错候选；本窗口未实现，不把其null/深监督说成本项目创新|
|[REVIVE](https://openreview.net/forum?id=HFfx4EcMJa)|诊断当前状态、同状态备选纠错动作、源监督动作排序|已取得PDF，早期路线稿曾只有摘要；当前未实现agent或交互动作系统。r06局部信用不能直接称REVIVE复现|
|[AlphaClick](https://openreview.net/forum?id=jsXr3lFB0R)、[CoRD](https://openreview.net/forum?id=upq4DrE0xu)|测试搜索/临时与持久状态等摘要线索|未纳入已验证机制，本窗口未实现；不依据摘要补写其精确监督和性能|

“未实现”只针对本窗口r01–r12，不抹去更早项目中独立实验；旧实验与此轮样本、BN、分辨率和监督不同，不能合并排名。

## C. 机制谱系和可声称范围

|当前组成|主要来源/性质|可声称内容|
|---|---|---|
|源监督残差纠错|CP-JEPA等已有近邻，加IT3/既有S3纠错研究背景|本数据上追加源训练有开发收益，不能声称首次|
|源训练展开部署更新|AR-JEPA、LSTA、通用meta-TTA、ViT3|已实现本项目状态变体，但独立更新收益未成立|
|低秩状态更新|LSTA启发|不是原LN子空间，r05后已移除|
|临时/持久隔离与收益写入|CP-JEPA/SIFT等设计启发|功能实现与任务收益分开；当前无持续收益证据|
|同数量/权重/独立对照|Selection Credit|用于防止误归因；不是新增临床算法贡献|
|单区域整头Dice信用|本项目针对固定候选干预设计的源监督代理|有实现和精确区域替换单元检查；与最终复合输出仍错配，尚不能认定原创或充分有效|
|双视图一致性、4p(1−p)支持、翻转/颜色候选|常见训练/预测构造与本项目组合|不是仅因来自2027阅读而变成新机制；需实证增量|

本项目没有完成跨全部近邻的系统新颖性排重。尤其“源标签训练残差＋低维更新＋写入门”并非独有组合证据。Pro应先评估有益信息和任务机制，再讨论命名与论文叙事。

## D. 阅读证据的精度

主要依据为已保存全文关键方法核查：LSTA§3、AR-JEPA§3.1–3.4、CP-JEPA§3.2–3.4、SIFT§4.1–4.4、IRP§4.3–4.4、LQN§3.2及Selection Credit干预定义。其余条目遵守上表范围；不声称全篇/附录/代码都审完。

旧READING_AND_ROUTE草稿曾把Selection Credit列摘要线索、下载清单列缺失；实际已获取PDF并在本次重新核对。REVIVE的下载状态也已从早期摘要阶段推进到本地PDF，但本报告不据“有全文”冒称完成全部方法核查。所有正文留在本地，不推送第三方材料。
