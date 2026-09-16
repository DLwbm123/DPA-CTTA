# R6 阶段 I Codex Prompt：有界区域重加权一致性

请在 DLwbm123/DPA-CTTA 中实现下述R6。当前仅授权阶段I的代码实现、程序化CPU验收、元数据dry-run和审阅材料交付。不得启动GPU smoke、真实目标图像/标签/给定checkpoint模型运行、R6-A或R6-B。不得沿用R4/R5的执行授权或waiver。

配套文件为R6_EXPERIMENT_PLAN.md、R6_SCIENCE_PROPOSAL.json和R6_MATH_REFERENCE.py。本prompt自包含；科学定义以本prompt与原始JSON一致部分为准，发现冲突应报告而不是选择有利版本。数学参考只是可审阅的公式参考，不是生产host或已经验收的R6实现。

## 1. 固定来源、工作区与范围

基线发布SHA：e271098e2a12baa166fc7b77848af4a2300d9cc3
历史R5-A执行SHA：b4b71601a5bdf87bb3a7e5d3db352610adcdff74
registration digest：8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
recurrence digest：cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
CTTA依赖：dbff0d985c6c95345d9fb78f5b1daef57b392564
GraTa依赖：33ae20d664f305af34739ec54a5bec7da53ffa0b
建议分支：experiment/r6-bounded-regional-consistency-v1
提供JSON提案字节SHA256：2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff

先检查工作区、已有R6工作与真实历史，不覆盖用户改动，不重置或重复已完成任务。使用上述基线，不擅自拉取main的新科学变化。禁止修改main、旧C、旧P2/R1–R5科学配置/结果、固定依赖。优先仅新增src/dpa_ctta/r6_regional_consistency、独立scripts/tests/configs/docs。少量不可避免的共享工程改动须单列且证明历史行为不变。

先阅读：
- docs/results/r5a_diagnostic_v1/REPORT.md、PUBLIC_AGGREGATE.json（只读CPU聚合）
- docs/review/r5_audit_fix/REVIEW_INDEX.md、FIX_REPORT.md、IMPLEMENTATION.patch
- src/dpa_ctta/b1_host.py、r1/host.py及它们调用的固定GraTa cal_consis_loss
- src/dpa_ctta/r5_update_acceptance/analyze.py、execution.py、evaluation.py、plan.py、run.py
- src/dpa_ctta/r1/supervise.py、r1/evidence.py、r1/assets.py、b3_runtime.py
- results/b2_interval_consistency_v1/B2_EXPERIMENT_REPORT.md及对应目标函数
- 原scripts/run_r5.py的neutral_subprocesses保护
沿import核对C实际路径，不能拿另一个教师/归一化分支冒充C。若发现已执行的等价R6完整配置，报告确切证据，禁止重复运行；不要把旧interval或PCA区内正则误判为本轮普通BCE重加权。

R5继续NO_ADVANCE，B关闭。R6不含接受/拒绝、回滚、门控、RP、EMA、固定教师、kernel、图、额外loss、输出融合、源端训练或按域路由。R6改变的是BCE像素权重，不能称已证明有效或首次发明类别平衡。

## 2. 原协议与数据边界

只使用现有给定Fundus ResUNet34；无源RGB/mask/query/代理/原型、新预训练组件或重训。阶段I连给定checkpoint也不读取，只允许程序化张量和随机初始化完整固定网络。保持原环境和依赖，不pip升级。

1,951内容/轨迹全部参与适配，remaining_dev=1,695仅是评分子集。保持原四主序和recurrence登记；不重新构建流或按结果选择内容。512×512、OD/OC独立sigmoid、原mask映射/resize/0.5评分判定/ASSD约定不变。

每轨迹独立初始化seed=20260907；跨域不reset。host只接收当前RGB，不接收GT/domain/sample/group/subset/未来图像或评价值。模型与evaluator间返回原生logits，不二次sigmoid。最终输出是一次Adam后学生原图预测；q仅作软监督和辅助评分。

可更新参数仍是C的41个BN层、19,136 affine标量，当前输入统计。Adam lr=1e-4，betas=(.9,.999)，eps=1e-8，wd=0，float32，AMP关闭。每图六弱前向产生实际q、一strong可微前向、一次loss.backward/Adam、一次post原图前向，共8/1/1/0。零梯度仍进行原Adam调用；不因loss为零跳过。

新增诊断不得改变增强RNG、q计算顺序或原C loss。用旧C无改动候选链加窄criterion接口；新建R6 host，不继承R5门控状态、不触发R5 analyzer/授权入口。

## 3. 唯一权重规则

对实际进入原C loss的q.detach()，每通道c、每图N=262144，h_cu=(q_cu>=.5)。h只生成权重，loss target仍然是完整软q。

若n_fg>0且n_bg>0：
  rho = min(8, max(1/8, n_bg/n_fg))
  w_bg = N/(rho*n_fg+n_bg)
  w_fg = rho*w_bg
  w_u = w_fg when h_u else w_bg
否则该通道w=1，fallback=ONE_PARTITION_EMPTY。

每通道数学mean(w)=1，权重正且在[1/8,8]内；未限幅时两分区各占一半权重质量，限幅后不是严格平衡。上限8、阈值.5固定，不搜索，不用GT或历史域均值决定。

权重构造可用CPU float64，但避免Python标量通过默认float32中间张量造成伪float64；显式float64初始化/赋值。正式loss用原float32，在CPU归约审计时使用实际施加的float32权重转float64，而非未舍入理想权重。

L_BAL = mean_(c,u)(detach(w_cu) * BCEWithLogits(z_cu,q_cu,reduction='none'))

权重乘整个BCE，不能用pos_weight，不能分别逆频率放大正负项来暗中改变软目标。不能对q锐化、hard化、包含投影、阈值裁剪或加入Dice/Tversky/熵loss。最终学生原图输出不做面积约束/后处理。

## 4. 四臂及必要匹配控制

臂名固定：C、R_BAL、R_SCALE、R_SHUFFLE。

C：生产loss必须保留原C标量BCE路径；诊断只读，不把C改成新reduce实现后强称bitwise。
R_BAL：第3节语义区域权重。

以下量只由当前自己的strong z和当前自己的q计算。在loss dtype先求d=sigmoid(z.detach())-q.detach()，之后转CPU float64：
  S0_c = sum(d_c^2)
  Sw_c = sum((w_c*d_c)^2)

R_SCALE：a_c=sqrt(Sw_c/S0_c)，该通道所有像素用相同detach(a_c)乘未加权BCE，再OD/OC等权平均。它是通道内均匀缩放，不叫global-LR匹配，也不改变Adam lr。

R_SHUFFLE：每通道全部H×W像素上的w独立随机置乱得到w_perm；不是在各分区内部打乱。保留基础权重直方图，不能打乱q/z/图像/特征/GT。
  Sperm_c = sum((w_perm_c*d_c)^2)
  b_c = sqrt(Sw_c/Sperm_c)
  loss = mean(detach(b_c)*detach(w_perm_c)*BCE)

S0_c==0时要求Sw_c和Sperm_c也为0，a_c=b_c=1。S0_c>0但加权能量<=0、倍率非有限等均硬失败，不能为了继续运行加epsilon/裁倍率/跳更新。正有限权重使上述比值在合法输入上有界。所有倍率和权重detach，不能让归一化倍率的导数进入backward。

在同一z/q局部状态下，R_BAL/R_SCALE/R_SHUFFLE的每通道strong-output logit梯度范数一致，允许冻结浮点误差；并不是BN梯度、Adam位移或跨轨迹范数一致。各臂独立，不能读取另一个臂的状态、GT或未来数据完成匹配。生产不得为控制调用autograd.grad、额外backward、VJP或额外网络前向；公式直接计算即可。

独立置乱：每次访问t(从1)、通道c(0/1)，字符串精确为
R6_WEIGHT_PERM_V1|20260907|{t}|{c}
ASCII编码SHA256的前8字节按大端整数，再 & ((1<<63)-1)。用此seed新建专用CPU torch.Generator，torch.randperm(N,generator=g)。不使用Python全局hash、不影响任何增强RNG，不引入ID/域。可以所有臂都计算只读候选置乱诊断以统一代码，但实际C/R_BAL/R_SCALE不使用置乱权重。

## 5. 生命周期与只读诊断

建议明确：IDLE -> WEAK -> STRONG/LOSS -> UPDATE -> POST -> COMMIT -> EVALUATION_RELEASE。
pre来自原六弱中的原图，q来自实际criterion，post来自原本最后一次前向；不新增获取pre的模型调用。新host未移出上一张payload前拒绝下一张；任何硬失败不得原地续跑。

每图所有参数/Adam/RNG状态提交后，evaluator才读取mask。稠密q/pre/post/权重只在当前图生命周期存在，评价后清空，不存历史图、完整预测、激活或适配后模型。

无标签日志至少每通道保存：
- n_fg/n_bg、rho是否限幅、w_fg/w_bg、实际float32权重值、空分区原因；
- 标准BCE总和、fg/bg和分母、语义加权BCE、置乱加权BCE；
- S0、Sw、Sperm、a/b，fg/bg未加权与加权残差平方和；
- 实际loss及strong logit梯度hook的L2范数（hook只读返回原梯度），实际BN梯度范数、Adam affine位移；
- 全部实际forward/backward/Adam/VJP累计及本步delta，source不变证据、host时间。

预测评分包含pre/q/post；只把post当正式方法输出。每图OD/OC像素计数可实现性、Dice/ASSD、gt/pred空满标记都保留。可在evaluator额外统计q分区与GT错配，但不能把这种统计回传给host或用于动态选择阈值。

注意：像素面积小不等于背景梯度主导；诊断不得把n_bg/n_fg直接当成梯度贡献。只记录实际残差能量能支持的结论。输出梯度匹配不等于参数空间匹配，应在报告中明确。

## 6. 阶段矩阵、资源和授权隔离

A：四臂×原order0/order1/order4=12条完整轨迹。
B_NEW：四臂×order2/order3=8条，A的12条复用，总计20。
AB仅metadata展示，不是可执行scope；真实授权仅A或B_NEW，禁止自动串联。

顺序固定按order外层、臂(C,R_BAL,R_SCALE,R_SHUFFLE)内层生成；按job_index%workers分配，1到3worker。不能将某臂永久绑定某设备；每条独立从源checkpoint初始化。

A正式23412访问、187296F、23412B/Adam；B_NEW15608访问、124864F、15608B/Adam；AB39020访问、312160F、39020B/Adam，VJP全部0。

本轮C新跑同批，不复用R4/R5的旧C充当同期控制。旧C结果只是背景，不能认为它们与新C bitwise相同。A完成后无论结论如何都停止；B要求A完整合格、原账本可重新核验且单独授权。

当前devices=null、execution_enabled=false；max_workers上限3、threads/worker=2。拟议硬caps为21600秒/trajectory、86400秒/stage wall、172800秒/stage累计worker、8589934592字节新私有输出；不是预估耗时。实际执行需用户明确资源与新receipt，阶段I不查询GPU卡或猜卡号。

未来每设备每阶段一次smoke，固定程序化pixels('fundus',0..3)、seed=20260907：旧C4步+四个R6臂各4步=160F/20B/20Adam/0VJP。只有旧C与R6-C要求parity，CPU exact，GPU继承rtol=1e-4/atol=1e-5及原deterministic_smoke_pair/R3 cuBLAS边界。R_BAL/R_SCALE/R_SHUFFLE不能被错误断言为post等于C。smoke只验机械正确，不能按toy Dice上涨验收。B也须新阶段smoke，不沿用A的进程状态。

保持中性入口、受控子进程组、NFS ENOENT限定容忍、失败先清理后落盘、有限排程、禁止自动retry。异常时记录live hook已观测计算并标注下界，不能用成功次数×8填补。构造/未触发hook工作或不可捕获信号明确为未知，保留首失败和原异常，不假装精确0。

## 7. 主终点和唯一晋级gate

每条全部1951内容适配，评分主子集remaining_dev1695。先同图OD/OC平均，再域内，再四域等权，再主序等权。recurrence单列不做五流平均。

A只有两个主序：对control=C/R_SCALE/R_SHUFFLE，两主序平均R_BAL-control阈值依次为+.50/+.20/+.20pp，且每个配对在每个主序均>=0。recurrence对三个控制的差分别>=-.10pp。四域各自两主序平均R_BAL-C>=-2pp。全部12轨迹、身份、调用、可实现性和CPU验证有效后，才能计算科学gate。

通过状态R6A_COMPLETE_ELIGIBLE_FOR_REVIEW；不通过R6A_COMPLETE_NO_ADVANCE。两种状态均next_execution_authorized=false。机械缺失/坏记录是INCOMPLETE，不是科学NO_ADVANCE。

B补齐四主序后：平均阈值保持+.50/+.20/+.20pp，三个配对各至少3/4主序严格>0；最差同序R_BAL-C>=-.50；每域四主序平均R_BAL-C>=-2；recurrence三个配对仍>=-.10。状态R6_EXPERIMENT_COMPLETE并单列selection结果，无自动追加。

这些是预定资源筛选，不是显著性/临床安全。不要使用R5的影子G或即时oracle作为R6晋级gate。新目标产生新的长期状态，其完整轨迹增益不受旧C路径pre/trial选优量限制；但这也不保证它会有正效果。

## 8. 分析器与审计

建立独立R6标量分析器，不调用会修改R5历史结果的recompute。先确认R6输出marker/run_id/scope所有权，再invalidate/publish。完整核验registration/stream/代码/science/生产指纹/顺序/唯一ID/子集/通道/数据覆盖。

R5-local可实现性修复必须复用或等价保持：所有预测的TP/FP/FN/TN均为非负整数，Dice从计数重算；同时检查pre/q/post的GT计数一致、跨轨迹同内容一致、ASSD有效性与非有限数。

按记录的n_fg/n_bg重算权重和空分区；检查S0/Sw/Sperm与各分区平方和关系、倍率、均值权重和有限性。S本身来自真实当前张量的标量归约，不能宣称CPU由这些标量重建了全部权重位置、梯度或概率。置乱seed可以重算；实际weight-hist/checksum与hook匹配要在runtime验收，不靠summary自证。

报告全部子集、逐序逐域OD/OC/macro、固定配对(R_BAL-C/R_SCALE/R_SHUFFLE)、正零负、最差ceil10%和最差单内容、ASSD共同有效/undefined、内容加权辅助表及等域主表。只在已定义ASSD配对上汇总，不填0；不计算macro ASSD。保存方法自身所有前景错误与尾部，不能只选OC或Drishti_GS好结果。

实际网络调用、worker时间、host/评价时间、peak allocated显存按实际记录，区分smoke/正式/失败前缀/CPU程序化测试。相同8F/1B不等于相同FLOPs或耗时。

## 9. 必需CPU验收（在最终candidate SHA，不合并旧日志冒充）

使用现有环境；禁torch.cuda初始化、源代理、真实pixel/mask/checkpoint loader。旧IO回归只允许现场创建且受路径约束的临时fixture。完整网络用固定结构随机初始化权重，不能以真实checkpoint替代。

至少覆盖以下独立性质，不以固定“必须N项”替代实际测试覆盖：

A. 数学和梯度
1. n_fg/n_bg不同配置，包括0/N、1/N、N/2、极少前景、几乎满图；mean(w)=1、正性、cap、精确fallback。
2. 未限幅的两区权重质量相等；限幅时报告非50/50。float64 reference生成不得偷偷先float32再提升。
3. 置乱基础直方图保存，跨两分区确实改变权重位置，不改q；独立seed复现和增强RNG不变。
4. 固定0<q<1且p=q时R_BAL梯度为0；使用soft q，pos_weight替代应被对照测试识别。
5. 同一z/q的R_BAL、SCALE、SHUFFLE逐通道logit梯度范数符合独立解析参考；系数必须detach。CPU float64建议rtol=1e-10/atol=1e-14，float32建议rtol=1e-5/atol=1e-12，事前记录容差来源，不看失败后放宽。
6. S0=0精确分支、极小合法残差、不有限z/q/weight/state硬失败。不能添加数值自救导致科学变化。
7. 所有权重=1时新增loss与标准C退化；真实非均匀权重时不要求与C输出相同。

B. 完整host与隔离
8. 原R1-C与新R6-C随机完整ResUNet多步bitwise比较logits、affine、Adam、RNG和physical；用实际q而非另算q。
9. 随机完整网络四臂至少4步真实执行；与小模型独立目标参考对比，实际8/1/1/0；尺度对照的主要代数验收在共状态梯度测试，不误称独立轨迹相等。
10. 改变/延迟/省略GT评价不影响后续预测/状态/RNG；host不接受域/ID参数；上一payload未移出不能再step。
11. 生产只有一次backward/Adam，不能在norm控制中调用autograd.grad/VJP；非BN参数与buffer policy保持。

C. 分析与运行边界
12. 12/8/20矩阵无重叠/遗漏；B只复用本轮A的12条，保留原execution绑定；旧R5结果不能冒充R6-A。
13. CPU合成完整A/B ledger测试有效完成、所有gate恰等边界与任意一项不满足；A完成不会生成/启动B。
14. 指标不可能计数、Dice直接污染、重复ID、缺失/换序、权重/cap/scale/seed污染使valid=false；历史外部目录绝不能invalidate。
15. 真实旧C/R6类+程序化Toy的smoke成功配额，以及各host中途forward、backward后、Adam后异常计数；原异常/首失败保存，清理hook和对象。保留所有注入失败日志，区分预期测试事件与实际测试失败。
16. 默认disabled在任何GPU/真实资产查询前拒绝；错scope/SHA/science/data/stream/设备授权拒绝。neutral_subprocesses入口先于runner。
17. 在最终固定SHA运行完整新套件+必要既有回归（至少R5审计修复与当前R3_REGRESSION所含项目）；两端分别记录实际解释器/Torch/退出码/skip原因和CPU实测成本。缺服务器环境如实列阻塞，不拿旧版本日志代替，不启动GPU去补证据。

配套8项数学参考检查不是阶段I验收成绩；可运行并核对，但仍需上述完整host及继承回归。其首失败/修正历史附在math_history，不删除。

## 10. 交付和停止

建议新增路径：
- src/dpa_ctta/r6_regional_consistency/{loss,host,evaluation,analyze,plan,execution,run}.py
- scripts/run_r6.py、scripts/check_r6_cpu.py
- tests/test_r6_loss.py、test_r6_host.py、test_r6_execution.py、test_r6_analysis.py
- configs/r6_science_v1.json、r6_execution.defaults.json（始终disabled）
- docs/review/r6/REVIEW_INDEX.md、IMPLEMENTATION_REPORT.md、EXPERIMENT_PLAN.md、METHOD_PROVENANCE.md、STATE_LIFECYCLE.md、PARITY_CONTRACT.md、REPRODUCE.md、TEST_COVERAGE.md、PRESERVATION.json、DELIVERY.json
- 真实metadata A_MATRIX(12)/B_NEW_MATRIX(8)/AB_MATRIX(20)、去身份流摘要和预算。
- 窄代码patch、science原始字节/摘要、实现SHA、全部首失败/修复/最终日志。metadata不存在时报告准确阻塞，不用算术fixture冒充实际登记。

提供的提案JSON可以原样作为science内容保留字节；若需加入实现字段，请把原JSON副本保留并另加实现绑定，不能更改公式、arm、cap、gate、矩阵或限制。若确有科学改变需求，先交待修改和原因，不自行冻结另一个方案。

所有真实设备、caps批准、review/waiver和receipt仍null/NOT_RUN。不得伪造review PASS。Implementation SHA与后续文档发布SHA分开记录；任何run必须绑定实际checkout，不把证据提交冒充执行代码。

全部必要CPU/metadata验收完成后仅输出：
R6_IMPLEMENTATION_READY_FOR_REVIEW
execution_started=false
external_review=NOT_RUN
R6A=NOT_RUN
R6B=NOT_RUN

若未通过，报告实际阻塞及已完成证据。到此结束，不启动真实计算，不后台等待，不自动补跑或进入A/B。
