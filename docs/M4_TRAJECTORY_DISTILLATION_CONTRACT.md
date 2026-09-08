# M4：短窗口连续轨迹蒸馏——下一轮实验计划

状态：**拟执行方案，尚无 M4 结果**。本文件可作为下一轮 Codex 任务的实验说明；真正的执行授权来自用户对本轮任务的直接授权，不复用 M1/M2/M3 已结束的 receipt。

研究优先级：有效性第一、创新性第二、SOTA 第三。

## 1. 本轮决策

结束 M3 的当前 FDA 条件化设置；不继续搜索幅度混合强度、频带、K、norm 或学习率。不恢复 H2，不重建已解决的开发/患者登记，也不再优化 M2 的采样器。

下一轮直接检验一个核心假设：

> 代理是否必须在自身连续使用所产生的历史中训练，并让较早更新对后续图像损失的影响进入蒸馏梯度，才能比局部一步目标更有用？

这不是已确定的 M3 失败原因。它是 M1–M3 尚未直接比较的训练目标问题。完成一次完整方法比较，不设置 Real/source/某个中间指标必须先提升的效果门槛。

M4 使用**无条件化、原生 VPTTA**在线路径。回到 M2 的 identity proxy 设置是为了排除 M3 已显示无稳定收益的额外模块，不是选择不同任务的“最佳风格强度”。两任务使用同一规则。

## 2. 已知证据及解释边界

M3 发布提交：`538a768363a79a5dca27de186e2b776c858c5e2b`。
M3 执行提交：`00d1da5b3a333e53b20d869ac9fcbbdcb8b67a8f`。
固定依赖：`DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`。

M3 O3 的 Fundus 目标等域 macro 为 76.866594%，仅比历史 D2 的 76.843102% 高 0.023492 pp；Polyp O3 为 78.151074%，低于 O2 的 78.494365%。不把相对退化的 D3 的领先写成优于历史最强基线。

现有训练从 Base 的固定 32 个历史状态取样、做一步、然后丢弃新状态。M2 扩大了配对覆盖，M3 改了代理外观，但这个结构没有改变。M4 专门改变这一结构。

三个历史轮次不是三个独立随机种子的重复验证。现有 source-clean 表是在独立新 host 上计算的保持性检查，不是目标流之后的 source 回访，不能据此量化灾难性遗忘。

## 3. 基线和文件边界

从 M3 发布提交建立新分支：

`experiment/m4-trajectory-distillation-v1`

新增独立的 M4 配置、离线训练模块、必要测试和评分编排。不要修改固定外部依赖、旧实验配置及历史记录。原生在线 host、预处理、medical region loss、Adam 前向语义和 memory 规则保持不变。

读取但不重新设计：

- `results/m3_conditioned_proxy_v1/M3_EXPERIMENT_REPORT.md`
- `configs/m2_episode_coverage_v1.json`
- `configs/m3_conditioned_proxy_v1.json`
- `src/dpa_ctta/offline/adaptation_dd.py`
- `src/dpa_ctta/hosts/vptta.py`
- 原生 `VPTTA/{OPTIC,POLYP}/utils/memory.py`、`prompt.py`、`convert.py` 及 image-step。

建议配置文件：`configs/m4_trajectory_distillation_v1.json`。

## 4. 三个新训练臂：一次训练完，不逐臂过关

每个任务训练三个代理集。在线部署方式完全相同，区别只在离线训练梯度目标。

| 实验臂 | 离线目标 | 窗口内跨步梯度 | 训练时前向历史 |
|---|---|---|---|
| D4 | 每步 prompt-space GM loss 的四步均值 | 每步输入历史 detach；局部 GM | 由自身当前代理联合 host 更新产生 |
| L4 | 每步适配后 source query region loss 的四步均值 | 每步输入历史 detach；局部一步 meta-gradient | 同上 |
| T4（主候选） | 与 L4 相同的四步 region loss 数值目标 | 窗口内保留可导状态的跨步依赖 | 同上 |

D4 不是原论文 GM 的完整复现，仍是本项目的 prompt-space GM 控制。L4 是关键消融：它与 T4 的窗口、数据、外层步数和前向状态演化规则相同，只截断跨图像的梯度。

**T4 与 L4 在同一个 S、同一个窗口起点及同一 query 序列下，前向状态和 loss 数值必须一致；meta-gradient 可以不同。** 训练开始后代理各自变化，轨迹自然会分化；不得强行让三个训练臂共享相互污染的状态。

历史 N/A/R/D2/O2/O3 均保留原名称。不要把 D4 或 L4 的最好结果改名为 T4/Ours。

## 5. 所有新臂共用的固定设置

- Fundus ResUNet34，512×512；Polyp PraNet，352×352。
- 同一原 source checkpoint；源模型不训练、不更新参数和 running buffers。
- 同一原 K=4 图像初始化和固定 source masks/layout；不从 D2/O2/O3 权重续训。
- 无 FDA、无其他外观条件化，代理 transform=identity；query 仍使用原来四种源端变换。
- 代理监督 region-only，lambda=0.1，beta_boundary=0。
- 原生 prompt、native Adam lr=0.05/0.01，betas=(0.9,0.99)，eps=1e-8，weight decay=0；每图一次更新。
- 原生 warm_n=5、neighbor=16、memory_size 参数40；保留原先可能为41条的容量语义。
- 外层图像 Adam lr=0.01，betas=(0.9,0.999)，eps=1e-8，weight decay=0；每个窗口结束更新一次；像素投影到[0,1]。
- query batch=1，proxy batch=4；四步指四张连续图各适配一次，不是同一图更新四次，也不是在线 batch=4。
- seed=20260907，所有训练臂共用预登记 source 序列。
- 不增加 risk loss、CVaR、boundary、router、模型选择器或额外 adapter。

## 6. 数据复用与600次训练图像访问

继续使用 M1/M2 已登记的40个 Fundus、64个 Polyp source train-query；原20/32 source评分内容组及七域224个 target-dev组完全不变。患者/视频链接缺失继续标 UNKNOWN，combined split 和既有开发角色沿用，不重新制造 H2 登记阻塞。

每个新训练臂执行：

- 5条 source-only 流；
- 每条120张访问，每30张为一个变换阶段；
- 每4张构成一个 meta-gradient 窗口；
- 总计600次训练图像访问、150次外层图像 Adam、600次功能化内层更新。

五条流的变换阶段顺序固定为：

1. clean → gamma_0.7 → gamma_1.5 → blur_5_sigma_1
2. gamma_0.7 → blur_5_sigma_1 → clean → gamma_1.5
3. gamma_1.5 → clean → blur_5_sigma_1 → gamma_0.7
4. blur_5_sigma_1 → gamma_1.5 → gamma_0.7 → clean
5. clean → blur_5_sigma_1 → gamma_1.5 → gamma_0.7

每种变换总计150次。对当前 M2 的600条 query/transform 配对分四个桶，各150条，忽略旧 state_index，但保留 query身份和对应transform的曝光次数。使用仅依赖固定 seed/task/原episode编号的确定性排序，在各阶段依次消耗30条。桶内可用一次预定的无评分排列，使同一个四步窗口优先使用不同内容组；不得看模型结果、不得搜索 seed。

在生成器单测中验证600条完整消耗、query/transform身份次数与M2相同、没有丢行或重复使用同一条episode。这里只验证索引，不再启动一个采样效果实验。具体最终顺序在任何训练前冻结为私有清单和摘要，三个训练臂相同。

**与 M2 的比较不是只改变一个导数开关：训练历史、序列组织和外层聚合次数也变了。** 不将 T4−O2 全部归因于跨步梯度；该因素的主要证据是 T4−L4。

## 7. 训练历史：前向持续，块边界只截断梯度

每个任务、每个训练臂有自己的原生状态 h=(prompt, Adam m/v/step, memory, counters)。

每条120访问流开始时：恢复同一source状态，prompt原始状态、空Adam、空memory、counter=0。图像代理S和其外层Adam不重置，继续累计150个窗口的训练。

120访问流内部：

1. 每张图都执行真实 native 初始化规则；
2. 每张图一次联合 host+0.1*proxy 更新；
3. 用更新后的prompt预测当前图；
4. 一次native memory push；
5. 持续携带Adam和memory，不在30张的变换边界重置。

每4张完成一个窗口：计算外层目标、更新S一次；窗口结束的状态值继续用于下一窗口，但全部 detach，释放上一窗口计算图。**detach不等于清空、重新初始化或回到Base。**

代理S在窗口间发生外层更新，因此历史由先前版本的S产生。这是随当前学习过程收集、带窗口截断的近on-policy历史，不是为每个S重新计算整个前缀的严格on-policy固定点，更不是贯穿120步的完整BPTT。

M1的固定32状态库不用于正式训练。可在机械smoke中复用其index16的状态作为检索已发生的参考起点；这是测试夹具，不计为重新训练历史。

## 8. 数学目标与必须保留的梯度

记当前窗口四张source query为(x_j,y_j)，j=1..4。S在一个窗口内固定。

内层每一步：

    phi0_j = NativeInit(x_j, memory_{j-1})
    g_j = grad_phi [L_host(x_j; phi0_j) + 0.1*L_proxy(S; phi0_j)]
    phi_plus_j, adam_j = NativeAdam(phi0_j, g_j, adam_{j-1})
    pred_j = f_theta,phi_plus_j(x_j)
    memory_j = NativePush(memory_{j-1}, raw_key(x_j), phi_plus_j)

query标签y_j仅进入离线GM参照或外层region loss，不参与任何内层更新、初始化、memory键/权重或状态挑选。

L4/T4 数值目标：

    J = (1/4) * sum_j L_region(pred_j, y_j)

L4：每一步开始先detach输入状态，保留本步S→proxy gradient→Adam→query loss的路径。
T4：仅在窗口起点detach历史；窗口内保留Adam moments和新memory prompt值到未来预测的可导路径，求完整窗口导数。

D4：每步在当前前向历史上构造 g_real=stopgrad(grad_phi L_region(query)) 与 g_syn=grad_phi L_region(proxy)，计算原GM cosine loss。外层目标是四步GM的均值。为产生后续前向历史，D4仍以无query标签的native host+proxy内层更新推进状态，不能用g_real推进。各步历史对GM梯度detach。

重要：

- 现有一步代码把host gradient整体detach，在原来的固定输入状态下有其含义。T4第二至第四步的输入状态已可能依赖S，不能机械照搬该detach；否则丢掉应有的跨步路径。L4/D4则按局部目标定义截断。
- 原生AdaBN内部已定义的stop-gradient统计规则保留，不悄悄改成另一种BN。T4是该计算规则下的可导窗口目标，不声称等于所有stop-gradient操作都不存在时的数学全导数。
- 采用已核验的功能化Adam前向、bias correction、eps和零二阶矩导数约定；不将离线SGD与在线Adam混用。
- 不能用常规load_state_dict、copy_、nn.Parameter重新包装等方式在窗口内部悄悄切断phi/m/v/memory值到S的图。
- 初始历史视为常量；窗口外的长期导数不计算。

## 9. Memory必须保留原生前向，不能把四步写成四次独立episode

原生prompt每张图先从memory重新初始化；memory不足16时使用全1。不得改成直接沿用上一步phi而不检索。

离线实现可使用功能化的tensor-value memory：

- 原生低频key与查询/合成代理无关的离散选择继续按原代码得到；保持dtype、key字节、argpartition邻居集合、权重、插入顺序、重复key覆盖和eviction规则。
- 旧历史memory值在窗口起点detach；本窗口新写入的prompt值对T4保留图。
- 保留原NumPy的邻居选择/权重作为常量，用tensor值执行加权组合并通过与native前向的数值比较验收。
- 对离散key选择、字节索引和eviction不求导，不使用任意straight-through选择器。
- 推理部署仍调用现成native memory；功能化memory只用于离线可导展开。

若实现只保留Adam路径却detach所有memory值，必须标注为另一个近似；不得以本计划的T4名字提交。不能为方便而删除native初始化、将warm-up固定为某个数，或让计数每窗口归零。

## 10. 必要正确性测试：一轮完成，不作为科研效果gate

复用原有回归，新增并运行受影响模块测试一次；不重复原H1、H2诊断。

1. 单步模式在相同输入状态下复现原M2离线目标和更新；D4/L4的局部梯度定义正确。
2. 四步功能化前向与原生host四次step在每个位置比较预测、prompt、Adam、memory、counter及源状态；涵盖冷启动与已有16条memory的起点。
3. 相同S、同一起点下，L4与T4逐步前向和外层loss一致，差别仅在gradient truncation。
4. 一个程序化可控fixture上验证：较早proxy更新能影响较晚query loss，并进入T4的meta-gradient；L4对应跨步路径截断。真实样本不要求每条跨步梯度都非零。
5. 更换source query标签不改变内层前向轨迹，只改变离线损失/外层梯度；目标标签仅在评分时读取。
6. 窗口边界图释放但状态值不变；变换阶段边界不reset；每条流120访问、600总访问精确。

沿用M2/M3已修复的严格确定性配对smoke上下文：只在必要的配对比较期间启用，CUBLAS配置须在CUDA初始化前；正式训练使用与M2一致的已登记环境。不得因数值不同反复扩大容差或搜索seed。旧报告的严格smoke不代表所有正式CUDA算子逐比特确定。

机械smoke预算建议：每任务用原始Real初始化S和同一程序化四图窗口，D4/L4/T4各一次功能化四步和一次外层更新；共24次功能化内层、6次外层。每任务再以固定S用原native host实际走四步作为共同前向参考，共8次真实online Adam。

冷启动和额外边界案例放CPU程序化测试，不额外启动未记账真实GPU轨迹。Smoke状态、代理均丢弃；所有正式训练从原始Real S和空outer Adam开始。

若真实实现/数值非有限/数据身份/OOM错误出现，保留日志并停止，不把错误当科学负结果。不存在“smoke分数必须更好”的条件。

## 11. 正式训练预算：六套代理，不是600个外层step

每任务D4/L4/T4各：5×120=600次source图像访问，150次外层更新，600次功能化内层。

两任务共：

- 六套训练；
- 3,600次source训练访问；
- 900次外层图像Adam；
- 3,600次功能化内层更新。

加smoke：906次外层，3,624次功能化内层。

这是按相同的600次source图像访问控制新臂预算；外层更新次数从旧方法的600改为150。必须公开写清，不能报告为“每套600步外层训练”。三种新臂采用同样聚合频率，避免只给T4扩大query曝光。

保存外层step 0、50、100、150；正式评价只使用150，无target选checkpoint。三个目标的loss不可直接跨方法比较。

需要保存的合成图像、outer Adam和小型状态可私有保存，不能沿用“诊断不允许保存代理”来阻断DD。checkpoint不作为自动续跑许可。

## 12. 一次完整评价：原顺序与反向域序一起做

所有六套代理完成并冻结后，统一评分，不根据source结果取消目标域或取消某个新臂。

source：原20+32=52组，D4/L4/T4各一次，host从源状态重新开始，共156条。

target：原M1/M2已登记224组，原域内图像顺序不变，使用两套完整连续域序：

- Order-0：M1原域序；
- Order-1：只反转域块顺序，域内顺序不变。

Fundus：

- Order-0 = REFUGE → ORIGA → REFUGE_Valid → Drishti_GS
- Order-1 = Drishti_GS → REFUGE_Valid → ORIGA → REFUGE

Polyp：

- Order-0 = CVC-ClinicDB → ETIS-LaribPolypDB → Kvasir-SEG
- Order-1 = Kvasir-SEG → ETIS-LaribPolypDB → CVC-ClinicDB

D4/L4/T4每臂每序各224访问，共1,344条新target记录。

为使Order-1公平，额外运行原Base A、封存D2和封存O2在Order-1上的各224访问，共672条。不能将旧Order-0的自适应结果当作Order-1基线。N无状态，可按同一图像身份复用既有逐图预测指标，明确标注重排复用；R/O3等只有Order-0时仅放该序的历史补充表。

新评分总计：156+1,344+672=2,172条，每条均属于适配臂，因而2,172次真实online Adam。加smoke8次，默认总真实online Adam=2,180。N及原顺序旧记录的CPU复用不重复计作模型运行。

每任务/每臂/每个序列独立从相同source状态开始；跨域不reset，不输入域名信号；source、两条target序列互不继承状态。每张目标图每个序列只访问一次。第二种顺序是同一批内容的敏感性检查，不是新患者或新的独立数据集。

## 13. 主比较和结果表

主问题：T4是否同时优于L4与D4，以及是否超过已跑通的Base和历史DD？

必报：

- T4−L4：窗口跨步meta-gradient的主要匹配对照。
- T4−D4：在相同前向历史/窗口预算下，任务轨迹目标与GM控制的差异。
- L4−O2：新历史/组织/聚合训练相对旧固定Base历史的描述性差异，不能称纯on-policy因果效应。
- T4−A、T4−D2、T4−O2、T4−N：实际净增量。
- Order-0另列T4−O3，不能不提M3已看到的历史最佳。

每任务分别报告每个顺序的等域Dice、OD/OC、每域差值、共同有效ASSD、empty/full、配对中位数/胜负数、最差10%及最差单例；两个任务不混合，两个顺序不当独立样本估计患者显著性。

source保持性表与target分开。没有目标后source回访，不宣称测量了遗忘。训练平均loss不保证尾部安全，本轮没有新增尾部风险目标，尾部只能作为必要效果评价。

时间分别报告训练、在线每图和完整pipeline；将旧实验历史时间与本轮增量时间分开。T4多步meta-gradient训练可能更昂贵，不将同访问数写成同FLOPs。

## 14. 科研结论与终止范围

所有规定训练/评价不由中间分数取消。结束后根据完整对照形成结论，不自动再加一轮。

- T4跨两个任务/两个序列均优于匹配的L4/D4，并相对A及历史DD有有用增量：支持继续做独立验证，但仍不宣称SOTA。
- L4改善而T4不额外改善：更贴近自身历史的训练组织有价值，未建立跨步梯度必要性。
- D4最强：优先承认更简单GM目标有效，不改名字归功于T4。
- 只有一个顺序/一个域的均值有优势，或均值提高伴随明显负向尾部：报告混合，不称稳定有效。
- 三者仍只在零点几百分点徘徊、未超过强对照：**结束当前“固定K=4 + 该低频prompt空间 + source-only短轨迹DD”的主线扩展，不再自动开始M5幅度/采样/norm搜索。** 这不等于证明所有DD或所有CTTA不可行。

不得把历史最优逐域拼接成可部署baseline，也不得靠target结果选择训练checkpoint、流顺序或是否加入条件化。

完成状态是 `M4_TRAJECTORY_COMPARISON_COMPLETE`，科学结果另外写。实施前freeze配置不是独立评审PASS要求，也不增加逐模块性能gate。

## 15. 执行、资源和输出边界

单卡，允许物理GPU4–7，优先7，实际UUID由本轮授权和receipt绑定；允许共存，不停其他进程。无资源则记录并停止，不后台等待，不自动切换多卡。GPU活动阶段上限6小时、私有新增输出上限2GiB，二者是资源上限而不是保证完成时间。

功能化四步保留更多计算图可能增加显存；实现可使用经过正确性验证的图释放/安全重算，但不能为省显存把proxy K=4拆分后改变AdaBN统计，不能静默降到L=1、降低分辨率或改精度。OOM是工程未完成，不能填入算法负分。

使用现有登记、已有去重和已知排除记录；UNKNOWN关联和combined split维持探索性角色，不再次启动无限NAS查找。真实字节摘要冲突、标签编码错误、明确禁止的资产仍须停止。

新receipt绑定本轮干净commit/config/资产引用/生成序列/smoke/最终代理；继承旧基线必须核对相同数据、评估和环境口径。不能让旧guard阻挡合法新入口，也不能改旧receipt伪造过去授权。

流水线：实现 → 一次受影响测试及smoke → 六套训练 → 所有评分 → 独立CPU重算 → 去标识报告。不得仅发布“准备完成”就结束，也不承诺异步未来交付；在工具/资源允许的任务会话内完成有限工作。真实故障时保存前缀和具体状态，不掩盖失败、不自动重跑已完成评分。

私有目录0700、文件0600。数据、masks、代理、optimizer、history、路径和逐图日志留NAS；公共只发布代码、配置、去标识结果及实际计数。

建议公共文件：

- `configs/m4_trajectory_distillation_v1.json`
- `docs/M4_TRAJECTORY_DISTILLATION_CONTRACT.md`
- `results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md`
- `results/m4_trajectory_distillation_v1/public_aggregate.json`
- `results/m4_trajectory_distillation_v1/execution_audit.json`

明确区分执行commit与报告发布commit。保留M1–M3历史，不force push，不发布患者信息或模型私有资产。

## 16. 参考与创新性边界

本计划的推断来自本项目结果和已核对源码，不声称M3已经证明“历史不匹配”是失败原因。

- M3结果：https://github.com/DLwbm123/DPA-CTTA/blob/538a768363a79a5dca27de186e2b776c858c5e2b/results/m3_conditioned_proxy_v1/M3_EXPERIMENT_REPORT.md
- M3配置：https://github.com/DLwbm123/DPA-CTTA/blob/538a768363a79a5dca27de186e2b776c858c5e2b/configs/m3_conditioned_proxy_v1.json
- 现有一步目标：https://github.com/DLwbm123/DPA-CTTA/blob/538a768363a79a5dca27de186e2b776c858c5e2b/src/dpa_ctta/offline/adaptation_dd.py
- 原生memory：https://github.com/DLwbm123/CTTA/blob/dbff0d985c6c95345d9fb78f5b1daef57b392564/VPTTA/OPTIC/utils/memory.py
- Dataset Distillation by Matching Training Trajectories, CVPR 2022：https://openaccess.thecvf.com/content/CVPR2022/html/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.html

多步/轨迹蒸馏已有前作，不得宣称本轮首次通过多步更新优化合成数据。这里拟检验的是固定分割宿主及其原生prompt/Adam/memory生命周期下，短窗口跨图像梯度相对于匹配局部目标的增量；有无价值由本轮完整结果决定。

## Implementation checkpoint

The new entry point is `scripts/run_m4_trajectory_distillation.py` (smoke/run/recompute), with `scripts/launch_m4_once.py` for detached run and reconstruction. No inherited scientific source/configuration was modified. `TrajectoryEpisode` reuses upstream model/prompt forwards, existing medical loss and functional Adam. It changes the memory batch combination to tensor values while calling upstream neighbour selection and push unchanged. Discrete byte-key selection and float32 NumPy attention remain constants; within-window tensor values preserve gradients only for T4.

The score-blind sequence uses one SHA-256 ordering of the fixed seed/task/original episode number, then greedily prefers distinct query indices within each four-image window while consuming the required transform bucket. No seed search or model score enters this ordering. This hash orders indices; no additional dataset hashing is performed. The final private sequence is receipt-bound before training.

Local affected validation: 22 tests passed (exit 0, 1.445 seconds), including seven M4 tests and 15 relevant M1/M2 regressions. Cold/warm four-step comparisons use a CPU procedural model with native memory and actual torch Adam; deployed full-resolution/source-model parity is checked by the single budgeted GPU smoke. The local CPU tests are not a substitute for that GPU result. Full historical suites are reused evidence, not rerun. Runtime status is recorded separately; this implementation checkpoint does not claim formal completion or scientific benefit.

The user's current long-experiment convention applies: once smoke passes and a reliable detached formal pipeline is verified started, the session may end without keeping an SSH connection or automatically monitoring. Actual completion, independent reconstruction and public result delivery remain separate from successful startup.


## Runtime outcome

Full-model GPU smoke failed at the second Fundus native reference prediction under the unchanged tolerance. Formal training and scoring were not started. See [failure report](../results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md) for the actual 2 online / 3 outer / 12 functional inner updates and exact stopping boundary. No GPU retry was performed.

## Authorized repair R1

The user subsequently authorized fixing the parity failure and running on any GPU 3–7 with adequate free memory. This supersedes the original 4–7 admission range and permits the bounded repair diagnostics and a new smoke attempt. Single-device training, fixed data/sequence/seed/K/resolution/precision/tolerances and six-fit/two-order budgets are unchanged. Existing failed receipts and the historical Adam implementation remain immutable.

Two registered Fundus fixture images reproduced the failure: first-step initialization and input moments were identical; separate-backward gradient differed by 7.45e-9, prompt update by 1.19e-7. The next retrieval differed by 1.19e-7, then next gradient by 1.98e-5 and prediction by 3.333e-4. Keys and insertion order matched. Joint-loss autograd and native-order addcmul/addcdiv Adam operations removed all observed differences across both images, including prompt, moments, memory values and logits. Memory arithmetic was not changed. This verifies the combined numerical fix on the failing prefix, not a universal bitwise GPU claim or formal method benefit.

The two repair diagnostics used a total of 4 native online / 0 outer / 4 functional inner updates. The original failed smoke used 2/3/12; these are recorded separately from the fresh smoke (8/6/24) and formal run (2172/900/3600). Cumulative GPU time and private bytes are charged against the original resource caps. Local affected validation now passes 23 tests, including a native-operation-order Adam regression. Full-model four-step smoke remains the admission check before training.

`diagnose_m4_parity.py OUTPUT_DIRECTORY FAILED_REGISTRATION_DIRECTORY` reproduces the bounded mechanical comparison under the configured dependency root and visible single GPU; raw tensors remain private. The final smoke now captures initialization and gradients, writes per-position comparison errors, and preserves diagnostic tensors and peak memory if a failure occurs. This adds no optimizer steps or target scoring.

### Full-window R1 refinement: native foreach dispatch

The four-step R1 smoke made all initial prompts, gradients, updates and predictions identical for D4/L4/T4 at positions 17 and 18. At position 19 the D4 initialization and gradient were still identical, but the prompt update differed by one float32 ULP, resulting in 44 out-of-tolerance logit entries. It stopped with 3 native online / 3 outer / 12 functional inner updates; no formal training or scoring ran.

A saved-tensor CUDA arithmetic check (75-element prompt, two optimizer primitive checks; no image/model calls) isolated the remaining cause: pointwise addcmul/addcdiv matches native single-tensor Adam, whereas the deployed CUDA optimizer actually dispatches to foreach. The foreach scalar-list bias-correction division and final update match the reference exactly at the failing state. M4 now mirrors that dispatch: CUDA foreach and CPU single-tensor, with graph-preserving out-of-place operations and the existing zero-second-moment derivative rule. The online optimizer remains unchanged. A CPU primitive regression exercises first/second derivatives and zero moments in the foreach branch.

Previous image-trajectory updates before the next full smoke total 9 online / 6 outer / 28 functional inner. The two tiny CUDA optimizer primitive checks are separately disclosed, not counted as image visits. Their runtime was not individually timed; a conservative 60-second charge is applied to the six-hour cap. Failure prefixes and both earlier execution commits remain preserved. No tolerances, seeds, labels, scientific objectives, per-arm visits or formal evaluation budgets changed.

### R2 higher-order std diagnosis and zero-channel convention

R2's full Fundus smoke passed all four positions for D4/L4/T4 with exactly zero logit, initialization, gradient and prompt differences. Polyp D4/L4 had finite meta-gradients; T4 reached four finite forwards but its outer derivative was nonfinite. The failed prefix consumed 4 online / 5 outer / 24 functional inner updates. An authorized one-window Polyp anomaly diagnostic (0 online / 0 outer / 4 inner) identified `DivBackward0` induced by native AdaBN `StdBackward0`.

PyTorch 2.2.1's [std_backward](https://github.com/pytorch/pytorch/blob/v2.2.1/torch/csrc/autograd/FunctionsManual.cpp#L1724) divides by twice the std and then masks zero std. Its existing zero first-gradient branch has a singular intermediate during double backward. The isolated offline AdaBN forward now calls a custom spatial std with exactly native forward and first-derivative multiplication/reduction order. It replaces the zero denominator before division, then applies the native zero mask. At exactly zero spatial std, the selected higher-order convention is zero Hessian; away from zero the native derivatives remain. This is an explicit extension at a nondifferentiable point, not a claim of a classical Hessian there. No epsilon, new loss, new statistics, source-buffer update, host-gradient detach or online-host change is introduced. The pinned external files remain unmodified; only the two isolated offline model instances receive the checked one-call substitution.

A mixed constant/nonconstant-channel CPU regression verifies exact forward and first gradient, finite zero-channel Hessian, and gradcheck/gradgradcheck away from zero. The original native AdaBN statistics' stop-gradient semantics are preserved. Counts preceding the next full smoke are 13 online / 11 outer / 56 functional inner, plus the two separately disclosed tiny CUDA optimizer primitive checks. All actual failed and diagnostic prefixes remain separately reported; no target scoring has run.

### R3 diagnosis and stable nonzero Hessian

R3 repeated exact four-position Fundus parity, but the Polyp T4 derivative still failed (4 online / 5 outer / 24 inner). A second one-window anomaly diagnostic (0/0/4, 9.982713 seconds) isolated division double backward at positive std as small as 3.743392e-23. Masking zero alone cannot prevent a squared float32 denominator from underflowing.

The custom first-gradient forward keeps the exact native arithmetic. Its backward now supplies the complete analytic Hessian-vector product using z=(x-mean(x))/std and n=HW-1: g*(v-mean(v)-z*sum(z*v)/n)/(n*std). The derivative with respect to upstream g is sum(z*v)/n. This avoids intermediate std squared/cubed; no small-positive clamp, epsilon, higher precision, detached host Hessian or altered forward is used. The saved std's dependence is already included in this complete Hessian and is not counted a second time. The zero-channel convention remains explicit; third derivatives are not required or validated by M4.

All 26 affected CPU tests pass, including double-precision gradgradcheck and a float32 1e-20 activation regression with exact native first-gradient values and a finite Hessian consistent with a double reference. Full-model smoke is still required. Before the next attempt, actual earlier image updates total 17 online / 16 outer / 84 inner, plus two tiny optimizer arithmetic checks. Measured and conservative time charges remain separately accounted for; no formal training or target scoring has run.
