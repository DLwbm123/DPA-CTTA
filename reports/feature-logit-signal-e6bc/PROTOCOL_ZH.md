# E6-B/C：固定FEATURE vs logits的块撤回信息资格（设计冻结）

准入依据E6-A：OD_LOCK保留组合OD，OC-only块仍有+.495794724pp标签空间，去单像素后+.289148720pp；不是无标签方法证据。用户要求空间门通过后冻结，最新持续自主授权允许工程检查通过后直接执行，不逐轮审核。本文件与config只冻结科学设计；执行代码、CPU/NAS/显存/硬超时准入另记录，不能将本文件称为已运行。

## 固定范围及输出

现有8开发内容×4条件/512×512，既有source checkpoint、预处理、seed不变，不新增源训练/模型/数据/权限，RDC/ENGRAM关。源全eval/无grad/无BN统计或参数更新，无历史/teacher/增强/图扩散。只产生无标签分数和独立CPU标签资格统计；selector_trained=false、selector_applied=false、network_updated=false、history_used=false、continual_benefit_tested=false。即使资格PASS也不把分数安装成新分割成绩，性能轮必须新协议。

动作沿E6-A：SOURCE/B两头分歧并集8连接联合块，所有OD-active块固定B，评价OC-only块从B撤回S的效用。不得读E5/E6-A逐块GT效用、oracle选择或私有矩阵定义分数；组件和元数据由限定端点bits重新生成。

## 源特征和坐标

唯一主FEATURE取现有`head_input=ReLU(bnout(...))`，32通道、原生输出网格。代码已有1×1线性head接口，不新增投影。运行必须核f=(1,32,512,512)、z=(1,2,512,512)，源head(f)精确重构z且阈值位图逐输入等封存SOURCE；若分辨率不符停止，不临场插值/下采样/换层。源参数和BNbuffers保持不变。每输入独立前向和完整重演一次，共64source forward；逐输入流式算分，不保存密集features/probs/logits。

所有表示与分数在CPU FP64计算。cos(v,w)=点积/(欧氏范数乘积)，不中心化/白化/拟合、无epsilon/floor。任一表示非finite整轮硬停；精确零范数导致对应块该表示不可评分，报告缺失，不改成高置信保留。恒定但非零的特征可产生合法常分数，按诊断失败处理，不造噪声或方差。

## 单一局部锚点与共同评分集合

每OC-only块的局部范围是距块像素集合欧氏EDT距离≤16个输出像素；排除所有联合分歧区域的欧氏距离≤3带（含该区域）。锚点必须S/B OC同意：前景且source sigmoid(z_OC)≥.9，或背景且≤.1。每类按距块最近、再行、再列选择8点；等距不按特征/GT择优。任一类不足8，该块不可评分，不放宽条件。比较所有分数用同一组锚点和真实OC翻转像素。

共同有效集要求锚点齐全、原FEATURE/FEATURE_SHIFT及LOGIT_MATCH的全部查询/锚点范数均严格>0且数值finite。分别报告各表示固有有效集与共同交集，主比较只在共同集；不能让FEATURE只挑容易块、对照被迫处理全部块。共同集的全部分数必须可重演。

## 六个固定分数（越大越支持从B撤回S）

记a(p)=S_OC(p)−B_OC(p)，Δ_j是真正OC翻转点。对表示v，`e_v(p)=mean_{fg8}cos(v_p,v_q)−mean_{bg8}cos(v_p,v_q)`，`s_v(j)=mean_{p∈Δ_j} a(p)e_v(p)`。

1. FEATURE：32维源f，唯一主候选。
2. LOGIT_MATCH：源双头原始logits组成2维表示，同一余弦/锚点/聚合；不把B的±1位图载体当置信。
3. MARGIN_DIRECT：`mean_{p∈Δ_j}|z_OC(p)|`，原始OC置信强对照，不经余弦丢弃幅值。其自然符号不代表有效阈值，本轮只评价排名。
4. GEOMETRY_ONLY：`e_geo(p)=(mean distance(p,bg8)−mean distance(p,fg8))/16`，沿同a(p)块均值，不用GT位置。
5. FEATURE_SHIFT：整张f沿宽roll W//2+1，原锚点选择/标签/坐标不变，再同式计算，破位置对应；不声称保局部语义或改动剂量。
6. STRATIFIED_ROLL：共同集内FEATURE分数按每输入、SOURCE→B方向（仅扩张/仅收缩/两者）、原联合块面积（1/2–4/≥5）、预测边界平均距离（≤1、(1,3]、>3）分层，扫描块ID序列循环移动1位。单例层不动，报告映射/实际分数变化比例；不是新seed或随机样本。

预测边界距离为SOURCE/B OC表面并集的EDT，在OC改动点均值，沿E6-A定义；不使用GT边界在线分层。所有方向与坐标规则先固定。没有次要特征层或正交残差赢家臂，主失败不换臂救分。

## GT隔离和撤回效用

分数worker只允许读取8登记RGB、既有checkpoint、代码/协议及新root内限定64端点bits/索引；拒scorer/全部旧E5/E6-A root/旧评分及标签，不能读本轮效用实现模块。score、anchors、uid和ACCESS封存且worker exit0退役后，独立CPU scorer才读8GT。

每块效用以完整B为参照。B OC面积P、GT面积G、TP=T，撤回该块的δP/δT，`u=(2δT−D_B δP)/(P+G+δP)`，D_B=2T/(P+G)，0–1 Dice单位；CPU用Fraction直接比对安装单块后Dice变化。空分母按既有两空Dice1定义单列，不加epsilon。u>0有益、u<0有害，u=0不塞入二分类；全部比例另报。不用E5 oracle选择作训练/评分标签，多块效用不相加。

逐输入计算FEATURE等六分数对u正负的AUROC（固定高分=有益撤回），缺任一类不可定义，不填0/.5。并列分数按.5；utility-weighted AUROC对正负配对分别用|u_pos|×|u_neg|权重；pixel-weighted同理用OC编辑像素数权重。块数权重、像素与效用权重均报，不用几千块充独立患者。

主聚合：每内容对可定义条件按固定[2,1,1,1]权重归一，再8内容等权；差值同一可定义输入集。必须至少24/32输入有正负两类，全部8内容各≥1可定义条件，否则资格失败；所有定义覆盖分母明确。2000次seed20261018只重采绑定内容，描述区间不作独立泛化。

score生成后仅允许共同有效块动作的OD_LOCK oracle另算覆盖上界（其余固定B），不是分数安装。报告可评分块数/像素、大小/方向/条件覆盖和可覆盖空间；逐内容/逐块值私有。

## 合取资格门（研发门，非临床/方法收益）

- E6-A主空间门已过；本轮共同可评分集合的oracle−B OC≥.2pp且内容描述CI下界>0。
- 定义覆盖≥24/32输入且8/8内容均可定义；FEATURE内容均值AUROC≥.60。
- FEATURE−LOGIT_MATCH AUROC均值≥.05且配对描述CI下界>0。
- FEATURE分别对MARGIN_DIRECT、GEOMETRY_ONLY、FEATURE_SHIFT、STRATIFIED_ROLL差值均值>0且配对描述CI下界>0；效用加权相对LOGIT_MATCH不反向（均值≥0）。
- 分层置乱至少50%共同块实际换了分数位置，否则对应控制不足，不宣称排除分层混杂；常分数/位置退化与覆盖不足如实失败。
- 源状态/端点一致、隔离、全部输入/分数重演、worker/scorer原身份退休与成本空间完整。

这些固定门不是从本轮标签结果选取。任一失败停止本固定配置的推进，不扫半径/置信/锚点数/层/距离/阈值；科学负结果保留，工程问题独立source/修复记录。通过也只称开发面板信息资格，不能自动宣称Dice提升或持续收益。

## 执行与资源完成定义

未来工程准入先做合成单像素/坐标/零范数/常特征/锚点不足/分数符号/并列AUROC/精确效用/隔离/完整重演检查；GPU前NAS实际挂载容量小写读、空间注册、GPU6 UUID现场核/显存余量≥8192MiB/中性父子ps+nvidia。已有只读checkpoint不复制；输出/代码/日志/缓存预约320MiB，root写入cap256MiB，本任务累计20GB/18GB水位/辅助1GB继续守卫，且每独立轮单台新增严格<50GB。

单次从首次CPU实际运行至全部退出总历时≤24h，预留保存/终止余量。启动CPU准备前记录唯一原始CLOCK_BOOTTIME/UTC起点，所有修复/子阶段共享；独立远程守护硬超时（不能依赖心跳或SSH），到86340秒先保存状态/停止，≤86400秒全部退役。工程修复不重置时钟，不杀其它聊天任务。GPU阶段至多一本聊天，健康每半小时一次轻量检查。

完成要求32源输入/64模型forward含全重演、SOURCE32bit parity、32头重构与源全状态不变、score/anchors/uid/coverage全重演、ACCESS0GT旧分数读取；worker/scorer exit0inactive、CPU8GT资格/共同覆盖oracle/定义统计完整、controller最终ledger退休。公开仅源码/冻结协议/匿名聚合无路径回执，经代理核账号branchremote/push/SHA/匿名200；绝不发RGB/GT/权重/features/probs/logits/bits/逐块坐标分数/逐内容值身份或私有路径。
