# R8：B 主攻、A 辅助的长程性能与机制边界实验

日期：2026-09-24  
状态：**实验设计；尚未实现、未启动模型或训练。**  
建议新分支：`experiment/r8-ba-performance-envelope-v1`  
历史参照：`DLwbm123/DPA-CTTA@f4adb85434e5ef36faecc1109f907d7a2f29d17f`。R7 执行代码为 `c1763d00f4f29de1c168ccd75219199667478825`。[R1]

## 0. 决策摘要

主攻 B/RCA，辅以 A/PSF 的较小容量搜索；C/RBE 只保留原有 FULL、STATIC 两个冻结控制。B 的域间收益与损失较明显，适合检验观察信息、表达能力、任务训练与历史的作用；A 提供不同的概率过滤结构，作为第二机制家族。这个优先级是研究资源安排，不是已证明 B 或 A 更优。[R2-R4]

本轮不是“1,000 步试一下，没涨分就停”，也不是无限追加参数。冻结以下整包任务：

- **12 个配置，全部配独立训练 STATIC；两个探索种子，共48个源端训练任务。**
- 每个任务训练 **16,000 个优化步骤**，每步8个 query，源序列长32，截断反传跨度8；任何早期分数不触发性能早停。
- 由源端验证选择每个配置/模式的检查点，并为 B、A 各选一个配置；各补三个种子，FULL/STATIC 共增加12个训练任务。
- 再训练5个同预算的 CURRENT-MLP 简单逐图控制。**合计65个离线训练任务。**
- 执行同期 VPTTA/C-CTTA/G-CTTA/N/C0、全部配置、五种顺序、同权重消融、B 的1/3步目标梯度诊断、混合流和10轮长流。
- 核心目标端 **724条物理轨迹任务、2,325,592次到达**；这是预设最大完整矩阵，不是新增患者数量。资格合格的三种额外基线另有最多75条预留，默认关闭。

**目标：在本轮规定的源端预算、控制空间与目标推理预算内，测出可以达到的性能、成本和失败边界。不能把这个有限搜索的最大值称为理论极限。**

历史比较提示：R7 的三条 FULL 相对项目内历史 VPTTA 的主指标约高0.53—0.68个百分点，但没有超过更简单的C0；相对原C约低3.48—3.62个百分点。因此主目标不能只有“赢VPTTA”，还必须检查超过C0的增量、超过逐图适配的增量、与强基线的差距。[R2,R5]

## 1. 本轮需要回答的四个问题

**H1：观察信息是否不足？** B 原来主要根据32维appearance描述推断状态。补充当前图的结构tokens统计，是否提高跨分组query分割，而不依赖保存患者tokens？

**H2：表达能力或调制强度是否限制收益？** 固定1024维ambient FiLM接口，增大低维控制空间与调制幅度后，源监督代理和真实目标表现怎样变化？rank与幅度的独立影响先通过源端容量探针检查；B主网格中二者组成一个联合容量因素，不将其解释成单独rank因果效应。

**H3：源端是否学得充分、辅助目标是否压过分割任务？** 比较完整训练曲线，并在B中比较原辅助损失权重与其十分之一。训练loss下降不能替代query Dice提高。

**H4：部署时是否确实需要目标图像的梯度信号？** 在同一B候选上额外加1/3步无标签latent梯度，与完全无反传以及zero-init梯度控制比较。不更新网络权重，但必须承认新增了模型反传，另命名为梯度版本。

“不带历史的版本最好”也可以成为有效结果：转向逐图适配，不强迫最终方法保留持续状态。

## 2. 不改变的基础条件

使用已有登记的RIM_ONE_r3源checkpoint、ResUNet34、512×512输入、OD/OC两个独立通道。冻结主干与原分割头；A/B新方法和C0继续使用匹配的current-batch/spatial BN统计、冻结BN affine。[R3]

VPTTA保留其原生prompt、memory和warm-up，C/G保留各自发布移植规则；不为了统一外观而更改其算法内的归一化。共同的是源权重、输入/标注编码、合法内容、因果顺序及评分，不是强迫所有算法使用一样的适配机制。[R5-R7]

所有主干前向保持物理batch_size=1；可以累积loss，但不能把多个support/query/目标图拼成一个batch而改变current-statistics BN语义。

主流仍处理全部1,951次到达，随后按原标签筛出remaining_dev的1,695个评分内容。不能只运行1,695张再假装历史一致。主指标为OD/OC同权、四域同权、order0/1同权；order2/3单独扩展，order4回访单列。[R2,R3]

所有旧运行、旧失败和R7文件只读保留；新实验不覆盖旧目录，不恢复R7的中断任务，不把`USER_WAIVED`改写为外部PASS。[R1]

## 3. 数据与选择隔离

### 3.1 源端

沿用已验证的新模块fit/cal/val分组，不为这轮换一个更好的划分。fit只训练任务模块和基；cal只校准温度/可靠性与B梯度分支的部署超参数；val选择配置、检查点并提供源端外推诊断。

这次source-val被用于模型选择，必须称为**验证集**，不再称其为独立最终测试。源checkpoint的预训练暴露另行披露。缺乏患者ID时只能报告已有content/eye分组，不把分组等同独立患者。

### 3.2 目标端

目标标签永远不进online函数、损失、历史、学习率调整、回滚或候选配置选择。只有预测固定后，独立evaluator读mask。所有候选源端训练、配置选择、梯度分支LR选择和最终种子训练锁定之后，才进入目标评分。

既有目标内容已参与项目开发，不能重新划一个holdout就称它为从未见过的新患者。没有核实的新患者集合时，这一整包明确是**充分的开发研究，而不是独立临床确认**。不自动访问任何sealed-final集合。

输出两种表：

1. **Source-selected主表**：配置和检查点都不看目标标签，最终5个种子。
2. **Observed development envelope**：固定网格及梯度分支中事后数值最高的点，单独标为探索性上包络；不把它改名为正式方法，不据此生成下一批候选。

TTA的模型选择策略会改变结论，在线历史也会放大超参数依赖，不能省略这层区分。[P2,P3]

## 4. 一次性完成的源端准备与容量诊断

### 4.1 光度组合

沿用R7九参数光度模拟器的操作和每个参数范围，不引入目标风格供体，不做改变mask几何的变换。固定局部seed20260924，fit512/cal128/val128个style anchors，各fold的0号为identity。

非identity的fit/cal按1、2、3、4个active因素循环；val的1—63号按1、2个active因素交替，64—127号按5、6个active因素交替。每个fold单独生成，参数值与噪声realization分开，support/query使用独立噪声。更多合成扰动不是新增真实患者，复合扰动不能叫临床域全覆盖。

### 4.2 更充分但仍只是代理的source oracle

每个anchor固定两个不同分组的support，另固定两个不在support内的query。因此保留原最小cal/val四组要求；跨anchor允许复用并披露。

对幅度a=0.1和a=0.3分别从v=0优化1024维ambient干预，Adam256步，lr0.03，betas(.9,.999)，eps1e-8，wd0。目标为support平均Lseg+1e-3 mean(v²)。记录16/64/256步support loss、query Dice、state与调制范数；不按query分数选择oracle权重，固定256步供后续基与代理监督使用。query标签不参与该oracle求解。

共2×768个oracle，名义786,432次model forward、393,216次合并backward与393,216次Adam，额外query诊断另计。

B按各自幅度的fit-oracle矩阵构造嵌套U32/U64；A按原预测协方差构造嵌套16/32维基。不得拿cal/val拟合基，不补随机列掩盖rank不足。A两种幅度的Jacobian计算按原规则共2,048次basis VJP，单列成本。[R3,R4]

### 4.3 区分投影损失和真正的低维可达性

固定64个source-val anchors（固定选前32个mild和前32个compound），针对A16/A32/B32/B64和两种幅度，直接在相应latent空间优化128步，保持同一support/query身份。比较：零干预、ambient oracle、oracle投影、直接latent优化。

新增名义131,072次support forward、65,536次backward/Adam；query诊断另计。该结果仅是源模拟下有限优化的可达性探针，既不是保证达到的数学上界，也不是目标域oracle。

如果ambient干预能改善query而低维不能，支持检查控制空间；如果低维能改善而观察器不能，支持检查推断/训练；如果support改善而不同group query不改善，提示代理迁移性问题。以上都只是证据线索，不将单个代理结果当作自动否决后续预设配置的门槛。

## 5. B路线：8个固定配置

B保留“预测—五步稀疏修正”的核心。所有配置均另训练STATIC，不用仅测试时reset冒充。

三个二水平因素：

| 因素 | 水平1 | 水平2 |
|---|---|---|
| 控制容量包 | r=32，FiLM幅度a=0.1 | r=64，a=0.3 |
| 当前观察 | 原appearance d32 | d32 + 当前tokens均值64 + 当前tokens总体std64，共160维 |
| 辅助目标权重 | 原辅助项×1 | 原辅助项×0.1 |

2×2×2=8个配置。容量因素同时改变rank与幅度，是联合因素；源端第4.3节提供四格诊断，主网格本身不能把收益只归于rank。

新观察定义：u=d，或u=concat(d,mean_tokens(E),std_tokens(E))。b为u→64 GELU→r，o为u→64 GELU→64；状态预测仍为A z_prev+b(u)+G(d-d_prev)。只将d32与z保存到下一图，不保存u中的token统计、E或概率图。不过z仍可能受患者结构影响，不声称已实现因果解耦。

H保持64×r、每列归一化，固定5步ISTA，科学下界与原kappa校准规则不改。损失明确为：

`L = Lseg(query) + eta * [0.1*MSE(z,z_star) + 0.1*MSE(o,H*z_star) + 0.001*coherence(H)]`

eta仅取1或0.1。所有时间/维度均按原定义平均。不添加teacher ensemble、target entropy gate、域ID路由或额外边界loss搜索。

B_GLOBAL_COMPACT_AUX1是新长程配方下最接近原B的结构控制；它并不是原R7完整配方的精确复跑，因为本轮源oracles、样式组合、序列与训练预算已经变更。

## 6. A路线：4个固定配置

完整协方差过滤、双代码本观察器与原辅助损失保留。只做rank∈{16,32}和幅度a∈{0.1,0.3}的四格。方差头输出2r维的均值/方差参数；F、Q、P同步扩展维度，保留完整P、Cholesky求解及固定正下界。

FiLM统一写作：`h_out=h+expm1(a*tanh(v_gamma))*h+a*tanh(v_beta)`。a是明确的实验因素；a=0.3不代表已经验证安全。零v仍需精确恢复匹配C0且导数可达。

A的预算小于B，是预先声明的研究优先级。不能据这个不等搜索预算宣称A/B方法家族能力的全面排名。

## 7. 长程训练：每个配置真正训练充分

### 7.1 固定优化量与长历史

第一批48个任务=12配置×FULL/STATIC×两个源训练种子20260924/20260925。全部16,000 steps，保存1,000/4,000/8,000/12,000/16,000检查点。

每个source macroepisode有32次support/query访问，按8次截断反传，4个optimizer steps。FULL在chunk边界detach而不清零state，只在macroepisode边界重置；STATIC每个visit都重置。每个optimizer step严格平均8个query loss，不能以不同chunk长度悄悄改变样本预算。

四类episode等比例轮换：突变16+16、32时刻nearest-anchor渐变、8+8+8+8回访、逐图独立style组合。渐变不得复用已选anchor，距离并列按固定ID。source-fit每个32时刻episode的64个support/query角色尽量无放回复用；无法满足时报告复用，始终保证当前support/query不同group，query不属于该anchor oracle support pair。

每visit沿用clean support观察、styled support观察、同style异group query预测；query不提交episode状态。每模型128,000个query访问，是R7每模型4,000个query访问的32倍，不是32倍真实样本。[R3]

### 7.2 优化器与精度

AdamW，基准lr3e-4，wd1e-4，betas(.9,.999)，eps1e-8，global grad norm clip1。t=1..200：lr=3e-4*t/200；其后cosine衰减，t=16000时到3e-5。该调度固定，不看target曲线调整。

主干/MLP FP32、不用AMP；小矩阵及原basis路径float64。允许通过数值/梯度等价检查的activation checkpointing，重算forward真实计数。冻结主干参数不等于不需要通过主干向adapter反传。

### 7.3 校准和检查点选择

每个A/B训练任务的五个保存检查点均独立执行1,024步source-cal。每个快照从固定校准初始化和空optimizer开始，冻结任务模块，A只拟合一个R温度、B只拟合一个kappa并同步修正ISTA步长。每step四个support形成原定义cal序列。

source-val固定64个32时刻episode，四类模式各16个，其中4个identity、6个mild、6个compound episode；报告所有检查点校准前/后的query分割、代理误差与机制指标。

对每个配置c和模式FULL/STATIC，计算各检查点的：

`S(c,mode,t)=0.5*四种val模式平均Dice + 0.5*最差val模式平均Dice`

每类模式先对其16个episode和两个训练种子平均，再计算S。各配置/模式独立选择同一个全局检查点t*，并不按每个seed自由选。A/B分别用FULL的S选一个配置；STATIC跟随同配置，但使用其自己源端选择的检查点。差值在1e-6以内按较低记录成本、再按配置ID决定。

**所有模型仍训练满16,000步**；源端选择较早快照不等于提前停掉训练。最终三个新增种子也训练满，然后使用前两个种子已冻结的checkpoint index，不能为新seed再看目标结果挑点。

源端评分为负也照常按规则选出本家族的最佳可用点，继续预设确认；不设置必须涨0.5pp才准继续的科学gate。只有数值/身份不合格才判为不可用。一个家族不存在两个种子均有效的配置时，该家族记录不可用，不从网格外补新点。

### 7.4 补足五个种子与简单控制

每家族最终配置增加种子20260926/27/28，各FULL/STATIC，增加12个16,000-step任务。源模型、源划分、oracle与basis在不同训练种子之间共享；这不是5个独立基础模型或5组新患者。

新增CURRENT-MLP：用B选中配置的同一u、rank、幅度、固定basis，MLP u→64 GELU→r，仅query Lseg，逐图推断，不使用H、ISTA、历史或proxy辅助项。独立训练5个种子、同样16,000步和数据调度；由前两个种子的source-val选择统一快照，不做温度校准。

这是检验“是否一个普通逐图回归器就解释了收益”的必要控制。总训练数65，fit更新总数1,040,000，query访问8,320,000；无缓存、无激活重算时共享三forward写法的名义fit forward上限24,960,000。60个路线模型×5快照×1024校准，共307,200次小模块校准更新和1,228,800次逻辑校准forward。实际缓存与重算成本另列。

## 8. B的在线梯度诊断：提前纳入同一包，不事后追加

这不是替原B偷偷加梯度后仍叫“无反向传播”。三个单独命名臂：

- B_G1：RCA提出状态后，执行1步当前图无标签latent Adam。
- B_G3：同样执行3步。
- COLD_G3：同basis/幅度，从零latent开始每图独立做3步；不使用学习预测器或历史。

对每次到达，使用零FiLM/current-statistics冻结网络对当前图的原图+五个几何弱视图预测，逆几何对齐，停止梯度的sigmoid平均为q。原图零调制观察forward可复用为六视图中的一个。强外观视图沿用固定C实现，每visit抽一次，在该visit的全部梯度步中固定。

使用source-fit代理的每坐标RMS尺度s=max(sqrt(mean(z_star²)),1e-3)，不做均值中心化。优化u=z/s，令u0为本visit初始状态：

`L_online=BCEWithLogits(f(strong_x;U*(s*u)),q)+0.01*mean((u-u0)^2)`

仅u进入Adam，所有网络和新模块权重冻结。Adam moments每visit重置，betas(.9,.999)、eps1e-8、wd0。lr从{1e-3,1e-2,1e-1}在固定source-cal的64个四时刻episode中按query分割选；query标签只评分、不进入梯度，前三个臂分别选一个全局LR，平均前两个源seed后冻结。该选择在任何target评分前完成。

B_G1/G3将校正后z作为下一图状态，并保留原d_prev；COLD_G3无跨图状态。新线上输入仍只接收当前图像。

名义模型调用：B_G1=8F/1B/1Adam；B_G3和COLD_G3=10F/3B/3Adam。这里已经计算六个teacher视图、强视图与最终原图输出，不能宣称仍然2F。若实现出现额外forward，按实际记录，而不是修改计数器迁就预算。

这组诊断可能仍失败：固定源软目标不保证真实标签正确。它检验的是受约束的目标梯度是否有帮助，不把优化能量下降视为Dice提高。

## 9. 目标比较矩阵与强基线

### 9.1 必须运行的同期基线

N、C0、原生VPTTA、固定LR C-CTTA、G-CTTA发布移植版、CURRENT-MLP、R7的C_FULL/C_STATIC冻结控制。A/B所有配置各自STATIC全部保留。

VPTTA/C/G用相同5个目标随机种子20260907..20260911，每条轨迹独立从登记源状态开始。新方法的5个source seeds与5个target seeds逐一绑定，不做25个组合穷举。N/C0和原R7_C两控制没有算法随机性，按每order一次物理运行；它们在不同seed比较表中的引用不算新增独立重复。

目标模型构建、参数清单、归一化、增强、输出时机、metric/evaluator、source checkpoint和content/order身份必须检查。VPTTA的memory/warm-up不能被reset错误或统一BN配置破坏。历史分数作为context，不替代同期主要结果。

MGIPT、PEOA-CTA、SPEGC各预留5seeds×5orders，共最多75条；只有在主计划启动前已有合格、固定版本的同checkpoint接入时才激活。涉及不同主干/checkpoint者另列，不硬拼主表；缺少实现或结果不填论文分数。不得运行中启动新的复现工程拖住核心724条任务。

### 9.2 核心物理任务计数

| 部分 | 计算 | 任务数 |
|---|---|---:|
| 全网格 | 12配置×2模式×2seed×5orders | 240 |
| 最终配置新增seed | 2家族×2模式×3seed×5orders | 60 |
| CURRENT-MLP | 5seed×5orders | 25 |
| VPTTA/C/G | 3方法×5seed×5orders | 75 |
| N/C0 | 2方法×5orders | 10 |
| R7_C_FULL/STATIC | 2控制×5orders | 10 |
| B_G1/B_G3/COLD_G3 | 3臂×5seed×5orders | 75 |
| 同权重机制消融 | 5臂×5seed×5orders | 125 |
| 混合与10轮长流 | 10个seed绑定方法×5seed×2流+2个确定性控制×2流 | 104 |
| **合计** | **不含可选额外基线** | **724** |

所有网格目标结果使用事先在source选定的快照，不在target上对每个训练快照取最大值。

## 10. 五个同权重机制消融

A：RESET_HISTORY、ISOTROPIC_R。B：RESET_HISTORY、PRED_ONLY、ISTA_20。全部复用各家族选中的FULL权重，独立从初始状态跑完整5orders×5seed。

RESET只清规定历史，不重抽增强RNG或打乱visit计数。A清m/P为0/I；B清z_prev并使d_prev=d。A_ISOTROPIC_R将R替换为trace(R)/r I，保留完整P和其余机制。B_PRED_ONLY不做修正；B_ISTA20固定20步，kappa对应能量与步长不变。

FULL−RESET回答同权重下使用历史的部署差异；FULL−独立STATIC回答同预算持续训练方案与逐图训练方案的差异。两者不等价。PRED_ONLY/ISO_R/ISTA20是部署消融，不是另行最优重训替代算法。

## 11. 长流压力：验证“跑得久”，不是制造新患者

两种新流在预测前冻结ID和SHA：

- MIXED：对现有1,951个到达ID做一次固定局部seed的全排列，不给online域ID。这是受控混合压力，不冒充真实临床到达分布。
- LONG10：原order0整体重复10轮，19,510次到达，轮间不reset；每轮1,695个主要评分内容。逐轮报告，不只挑最好的第一轮或最后一轮。

参与10个seed绑定方法：VPTTA、C、G、CURRENT-MLP、B_FULL/STATIC、A_FULL/STATIC、B_G1/G3；另加N/C0。五种source/target seed绑定不变。

报告第1至10轮的域同权Dice、最差域、OC/OD、相对第一轮的配对变化、与STATIC差值、state/FiLM范数、饱和比例、累计成本。重复图像不是新样本，不能用10轮增加患者样本量或声称新患者泛化。

## 12. 不只交一个平均分

### 12.1 最终性能与成本

主表：5seed的source-selected模型，四域×OD/OC×orders0/1同权；另列orders2/3、order4和全部压力曲线。12配置两seed网格单独报告，不把两seed与五seed混成一个排名。

各比较至少包含：FULL−VPTTA、FULL−C0、FULL−同配置STATIC、FULL−同权重RESET、FULL−CURRENT-MLP、FULL−C/G。给出paired mean、median、negative/zero/positive数量、q05、最差10%平均差、最差domain/order，以及ASSD共同有效分母和undefined。

报告每seed差值和离散度，不把order或像素当独立患者。可做固定预测轨迹上的描述性group-cluster bootstrap：已知patient优先，否则content群组；所有方法/order/seed同步抽样。它没有重跑历史，不能代表对新到达流的完整反事实不确定性，也不能替代独立患者测试。

### 12.2 机制量

通用：state norm、最终v范数、各FiLM通道改变量、tanh饱和比例、logit差/硬mask改变率、source query Dice vs proxy error、各loss分量和已有backward的梯度范数。

A：prior/posterior协方差特征值、R和校准尺度分布、创新量、prior与observation的相对作用、source-val分箱proxy误差。不要将source代理方差校准称为target覆盖保证。

B：预测state与修正state的任务差、delta/z比例、稀疏率、五步与20步能量及输出差、不同域的改善/损害。能量下降不代表任务改善。

所有新增诊断forward/autograd都进入账本，不算免费。可复用已算梯度，不为了打印诊断而隐式调用target标签梯度。

### 12.3 预先定义的结果判读，不是运行中止gate

- **只超过VPTTA、未超过C0**：不算新适配机制成功。
- **STATIC/CURRENT-MLP最好**：逐图学习有价值，但不能宣称持续历史增益。
- **FULL同时超过STATIC和RESET，跨seed/域有一致趋势**：获得支持历史利用的证据，仍报告失败域。
- **仅G1/G3改善**：方向转为低维目标梯度适配，删除“零backward”贡献主张。
- **源低维oracle有query headroom、target无增益**：优先考虑source-to-target推断/偏移覆盖问题，而不是已达到表示极限。
- **最大容量、长训练仍不改善**：结论仅限本轮搜索空间与预算，不推断所有B/A或CTTA无效。

可将“相对VPTTA和C0均≥+0.5pp、5seed中至少4个平均正差、无单一域均值低于C0超过1pp”作为一个预先登记的**研究进展参考**；不是显著性、非劣、临床安全保证，也不用于提前停后续配置。进一步竞争目标为追平/超过同期C/G，同时解释成本与域风险。

## 13. 资源、缓存和稳健连续执行

### 13.1 有限而充分的预算

建议最多4个worker，一GPU一任务，不预设目前可用GPU编号或具体型号。提议总预算硬上限**1,024 GPU·小时**、100,000,000次物理model forward、4,000,000次backward、4,000,000次optimizer step、50,000次VJP、200 GiB输出/缓存。GPU时按所有实际分配设备的占用时长相加，包括占GPU时的评分/IO等待，不等于日历用时。

这些是新计划的提议上限，不是实测耗时、完成承诺或默认采购额度。启动前用最大rank/长chunk、oracle、三类baseline与G3及评分路径的受控profile，按队列估算并乘1.3安全系数，出一次容量报告。若完整矩阵超过上限，就在正式启动前确定资源；不能运行中把16,000缩成4,000、删seed/域或减少校准次数后声称跑满。

### 13.2 缓存与显存

优先缓存source的零FiLM原始d/E或clean特征，只在checkpoint、预处理、content、style、noise、BN语义和fold身份完全一致时复用。source缓存可以服务多个新模块；没有免费改变源数据预算。不可缓存依赖当前state的query调制输出并断开梯度，不可预读完整target流建立特征bank。

采用经过等价测试的8步TBPTT和activation checkpointing控制显存。重算forward进入成本；主干权重只保存一次，模型快照保存新模块/optimizer/state及源checkpoint引用，不将65份完整主干复制到公开输出。

### 13.3 运行中不需要科学决策，但必须处理故障

SOURCE_PREP→48任务→源选择→12任务与5个MLP→梯度LR校准/锁定→目标全矩阵→压力→只读汇总，在新master manifest覆盖范围内连续执行。任务依赖和动态选中ID的推导规则预先固定，运行中无自动新增候选。

本轮允许一个**新的、有界基础设施恢复策略**，与R7的不resume记录区分：源训练每250步保存完整状态；target每50个已提交visit保存状态和输出偏移；每job最多一次从已验证快照的等价恢复。必须恢复RNG、Adam、source episode位置、m/P或z/d、VPTTA memory/warm-up/未注册可变属性及其计数、已提交输出offset等全部必要状态。恢复前后下一步预测/更新的roundtrip验证必须通过。

不支持完整状态恢复的native方法遇到中断就标PARTIAL，不从新state接着跑，也不把缺失段补成完整轨迹。恢复中重算的任何前缀另计成本，已提交评分不能重复入统计。

数值NaN/Cholesky失败停止该job并保留，禁止自动增ridge、改LR、降幅度、换seed“救结果”；与其无关的预设job可按新授权继续。标签隔离、内容/split/checkpoint身份、共享代码完整性出现问题，或触及全局资源cap，则停止整包并出真实部分报告。科学负结果不触发停止，跑得差仍然完成预设预算。

本计划不创建持续监测automation，不恢复旧heartbeat，不承诺聊天系统异步执行。

## 14. 交付与授权边界

交付至少包括BASELINE_PARITY、SOURCE_CAPACITY_REPORT、ALL_CONFIGS、SOURCE_SELECTED_CONFIGS、PRIMARY_RESULTS、PAIRWISE_TAILS、CALIBRATION_DIAGNOSTICS、B_GRADIENT_DIAGNOSTICS、STRESS_CURVES、COST_LEDGER、FAILURES、FINAL_REPORT；保留每个无效/失败配置和实际成本。

`R8_EXPERIMENT_SPEC.json`和`CONFIG_GRID.csv`给出配置与预算；它们是**设计规格，不是现成训练器**。当前`execution_enabled=false`。新runner和测试仍需实现。

正式启动时一次性绑定新code SHA、环境锁、源checkpoint/inventory/split、target registration和orders、GPU UUID、output root以及覆盖全部阶段的scope授权；源端生成物由预先授权pipeline的producer/参数/输入身份绑定，并在进入target前锁定。是否需要外部execution review沿用新的明确决定，不自行补写PASS。

用户当前请求是写计划，因此此包不授予真实模型、图像或计算资源访问；没有运行训练、checkpoint或target图像评测，也没有修改GitHub仓库。

## 15. 如何理解“把这一轮做满”

做到的是：源端有更充分的oracle与跨group检查、每配置固定长训练、真实同预算STATIC、更多当前观察信息、两档B容量/四格A容量、5seed确认、完整外部强基线、同权重记忆与求解消融、无梯度/少量梯度比较，以及重复10轮的长期行为。

不做的是：根据每次目标分数临时增加loss或数据、把已暴露样本伪装新患者、为了论文优势删除C0/不利域、无上限训练到碰巧出现最好seed。

**完成后应知道哪些预算仍换来收益、哪个组件确有贡献、哪个域在受损、哪些增益需要target梯度，以及还能否保持最初的无反传/持续状态主张，而不是只拿到一个“最好Dice”。**

## 参考材料

[R1] R7 HANDOFF：
https://github.com/DLwbm123/DPA-CTTA/blob/review/r7-target-gpu-cost-fix-v1/docs/review/r7_target_screen/HANDOFF.md

[R2] R7 固定公开结果JSON：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/review/r7_target_screen/results_gpu_first/RESULTS_PUBLIC.json

[R3] R7 COMMON_PROTOCOL与规格：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/review/r7/input/COMMON_PROTOCOL.md
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/review/r7/input/specs/COMMON_PROTOCOL.json

[R4] R7 B/A方法：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/review/r7/input/TRACK_B_RCA_METHOD.md
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/review/r7/input/TRACK_A_PSF_METHOD.md

[R5] 项目B1基线报告：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/results/b1_grata_matched_baselines_v1/B1_EXPERIMENT_REPORT.md

[R6] B1冻结移植合同：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/B1_GRATA_TRANSFER_CONTRACT.md

[R7] P2原生VPTTA身份：
https://github.com/DLwbm123/DPA-CTTA/blob/f4adb85434e5ef36faecc1109f907d7a2f29d17f/docs/P2_FROZEN_FULL_STREAM_CONTRACT.md

[P1] Chen et al., Each Test Image Deserves A Specific Prompt, CVPR 2024：
https://openaccess.thecvf.com/content/CVPR2024/html/Chen_Each_Test_Image_Deserves_A_Specific_Prompt_Continual_Test-Time_Adaptation_CVPR_2024_paper.html

[P2] Zhao et al., On Pitfalls of Test-Time Adaptation：
https://arxiv.org/abs/2306.03536

[P3] Cygert et al., Realistic Evaluation of Test-Time Adaptation Algorithms: Unsupervised Hyperparameter Selection：
https://arxiv.org/abs/2407.14231
