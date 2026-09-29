# R10：把即时适配与持久写入分开——源端组相对策略学习实验

版本：`R10_USE_WRITE_RL_V1`  
代码起点：`DLwbm123/DPA-CTTA@13bd6a8cdf9c30a0a5ed7fa464c67e72301ac8c6`  
建议分支／命名空间：`experiment/r10-use-write-rl-v1` ／ `r10_use_write_rl`  
交付状态：实验设计、展开矩阵与独立数学参考；不是已实现的完整 runner，也没有启动远端任务。

## 0. 这一轮要得到什么

核心假设：**改善当前图像的适配状态，不一定适合写入后续图像的历史。** 将当前使用码与持久状态分开，在有标签的源端短序列中学习写入决策；部署时冻结主干、B载体和策略，只依据当前图像及持久状态执行一个确定性动作。

本轮首先检验机制，然后比较性能，不承诺 RL 必然优于监督训练、C0、VPTTA 或原 C/G。A/C 不增加 RL 网格，R9 不改动、不停止、不重新解释。R10 不依赖 R9 结果或完成状态。

### 来源与本方案的区别

- **RaPO**，论文 §3.2.2、§3.2.3：保留奖励先加入总奖励、再计算组内 advantage；用跨步骤持续的奖励标准差 EMA 稳定归一化。原文使用相对上一任务 MLLM 策略的 token-log-ratio 漂移，研究有可验证奖励和明确任务边界的训练；本轮不照搬它的 token 漂移，也不声称复制其 rehearsal-free 协议。
- **QPrompt-R1**，论文 §3.3、§4 implementation：组相对辅助训练留在源端，部署移除奖励／参考模型等组件。原文 GRQA 是带监督分割损失的 query–prototype 对齐；本轮没有其 query 匹配瓶颈，不引入原型库或新的 query decoder。
- **本轮新设计**：读写分离的低维控制、源端后续收益与回访损失、独立监督／策略梯度对照、具体动作及预算。以下超参数均为预先规定的探索设置，不是两篇论文证明适用于 Fundus 的最优值。

## 1. 冻结载体，避免等待 R9 或另开基模型搜索

固定使用已封存的 **Screen24 实际部署 B_FULL_20260924**，从真实 source-job index、部署回执和校准 artifact 解析文件及摘要。它是按身份预先指定的起点，不按目标分数挑选；不要改成某个自行猜测的 `fit.4000.pt`。父载体对全部五个新策略种子相同。

保持：原 ResUNet34 checkpoint、512×512输入、两个独立 OD/OC sigmoid 通道、current-statistics BN、原预处理和评分；B64、FiLM amplitude=0.3、global observer、原五步 ISTA、冻结的 W/G/H/bias/head、basis、observer scaler 与校准尺度。复用经过验证的 source-fit B64 gradient scale `s`（floor=1e-3）。

此选择使本轮可以立即开始，不依赖未知的 R9 运行状态；代价是结论只针对这个冻结载体，不能宣称试尽了所有 R9 或 B 配方。五个新种子是**策略训练随机性**，不是五个独立 backbone 或载体训练。

源 fit/cal/val 分组沿用；不同 group 不自动等于不同患者。报告分组证据及患者依赖未知项。训练使用源标签；target标签、域名、患者ID、模拟风格参数、当前step所属域／模式、未来图像均不得进入策略输入。

## 2. 明确状态、动作与顺序

### 2.1 持久状态

持久的图像派生状态仅为 `m∈R64, q∈R32, h∈[0,1]`，初始全零。`q` 是未做质量归一化的外观加权矩，`h` 是对应累计权重；因此 `h*d-q` 在 h>0 时等于 `h*(d-q/h)`。无需除以很小的 h。

另有只用于日志／恢复的访问计数器；它不是策略特征。原 B 的 `z_prev/d_prev` 不得偷偷另行保留。源奖励EMA是训练器状态，不能部署到目标端。

### 2.2 当前候选

对当前图像先做一次零调制前向，得到冻结观察器的 `d_t` 与原 B 所需 transient tokens。按原 B 的 frozen 数学算子计算：

```
difference = h * d_t - q
prior = stable(W) @ m + bias(d_t) + stable(G, 1.0) @ difference
z_tilde = original_five_step_ISTA(prior, observation(d_t), H, kappa)
```

使用原函数的精确 `stable`、列归一化、能量、kappa、步长和 dtype，不另创同名近似。`h=1,m=旧z,q=旧d` 与原 B 的读历史一致；初始 h=m=q=0 时没有假外观差分。

### 2.3 10维动作与193维观察

策略只接收：

```
o_t = concat(d_t[32], m/s[64], z_tilde/s[64], difference[32], h[1])  # 193
```

前四块作为策略特征时逐元素 clip 至[-10,10]，h不clip；不因此截断真实 m/q 或改变 B 数学。策略观察全部来自当前及过去。

用两个**不共享参数**的网络：

- use network：193→128→64→9，SiLU；
- write network：193→64→32→1，SiLU。

输出原始动作均值 `mu=5*tanh(raw_head/5)`。训练动作 `a~Normal(mu, 0.35² I)`，标准差固定、不训练。存储采样的原始 a 与其 joint log-prob；不能把经 sigmoid/tanh 变换后的动作当作原始高斯变量算概率。

动作映射：

```
gain = sigmoid(a[0])                    # (0,1)，第一轮只探索缩放，不无限放大状态
residual[:8] = 0.5 * s[:8] * tanh(a[1:9])
residual[8:] = 0
u_t = gain * z_tilde + residual          # 当前使用码
w_t = sigmoid(a[9])                     # 写入强度
```

校正只使用已有 B64 坐标中的前8个；不是新学习的子空间，不对这8维作语义解释，不宣称它覆盖全部可行控制。原始 use head 最后层权重置零，偏置使 gain=.8、residual=0；write head 最后层权重／偏置置零使 w=.5。其他层使用固定种子的 PyTorch初始化。gain均值对应的有界 head 偏置需解 `5*tanh(b/5)=logit(.8)`，不要把两个参数域混用。

### 2.4 先输出，后提交

```
prediction = frozen_segmenter(x_t, basis @ u_t)   # 第二次主干前向
m_next = (1-w_t)*m + w_t*u_t
q_next = (1-w_t)*q + w_t*d_t
h_next = (1-w_t)*h + w_t
```

写入 w 不影响同一步 u 或预测，只影响之后的状态；预测和状态必须事务性提交。w=0时m/q/h全部精确不变；w=1时全部写入当前信息。不要只门控 m，却把 d 或其他图像相关字段照常写入。

部署：`a=mu(o)`，不用随机采样、不生成G个候选、不计算奖励、无 pi_ref、无目标端 backward／Adam。主干名义 **2F/0B/0Adam每图**，另计冻结B求解与小网络计算；以真实hook为准，不从名义计数推断速度。

## 3. D0：先观察“当前相同，写入后果不同”

两个初始策略种子各完成 warmup 后，各用64个固定 source-val 上下文（四种序列各16个）。以 `episode=index (0..63)` 调原 val schedule。当前位置 `t=4+4*(index%6)`，范围4..24；prefix为0..t-1，未来四图为t+1..t+4，均在32visit内。

用同一warmup策略、w=.5、确定性动作运行prefix（无需为prefix再生成有监督分割输出）。当前采样4个use动作；每个动作固定同一个u，分别强制w=0和w=1，再在完全相同未来4图上用同一确定性warmup策略、w=.5继续。当前图预测共享，未来状态分开。

报告当前差值=0的数值检查、未来macro/OD/OC差、正负比例、回访差、`|未来差|>.002`比例，以及只在源端计算的有限候选最好值。前述0.2pp是预定描述阈值，不是训练准入门槛。无收益也继续已登记训练；非有限值、状态不等价、隔离故障才是工程／数值失败。

D0含64×2个上下文、4个use候选、成对写入。不是新增128个独立患者；源端候选最好值不是目标上限或可部署方法。

## 4. 源端序列和严格分组

训练沿用 `r8_ba.schedule.anchors/episode_styles/episode_roles`，原 source simulator、query分组约束不改。只使用query，不将support标签带入策略。

post-training零基轮次 k：

```
mode = k % 4
episode = 4*(k//32) + mode
window_index = (k//4) % 8
t = 4*window_index
window = visits t..t+3
prefix = visits 0..t-1
```

每32个训练轮完整遍历4种模式×8个四visit窗口。风格和角色函数使用该模型policy seed；所有同seed方法使用相同外生图像序列。图像模拟键固定为 `R10_QUERY|fold|seed|episode|visit|group`，不含method或candidate。源val继续用其原本64episode分布，不能误套fit的mode索引。

每次收集前冻结 pi_old（当前策略）；从零状态用其确定性均值重新走prefix得到 s_pre，并detach。prefix不反传，不在不同训练轮之间复用旧策略生成的m；只可复用冻结零调制的图像观察。四个候选从同一个 s_pre 出发，见到相同后续4张图和模拟噪声；候选间只有动作噪声独立。

观察缓存以完整输入、冻结模型、预处理及scaler摘要绑定。当前组内一次零调制观察可共享；不要重复计为G次真实前向。可逐个候选运行以降低显存。不得缓存有调制且依赖动作的输出作为另一个动作的结果。

### 源端回访探针

每个窗口选2张额外源图，只在训练器／奖励器使用，同一fold、不同group、互相不同且不与当前4图重合；还排除该probe风格的原oracle support groups。用SHA256(`R10_PROBE|fold|seed|episode|t|probe_id|group`)确定候选顺序；没有足够group是数据准入阻塞，不偷偷重复当前患者。

两个probe风格分别取episode第0个风格和t-1处风格（t=0时都取第0风格）。风格可以相同，但图像group不同。probe不是目标图或未来目标信息。

probe读出：复制记忆→正常观察该probe→产生确定性use动作→分割；不提交写入，两probe各从同一待测记忆独立读出，不相互污染。它测的是这段历史对旧成像条件下其他图像的可迁移影响，不要求不同患者的预测一致。

## 5. 奖励：当前／序列任务表现与记忆后果分开

Dice范围[0,1]，阈值及空集定义沿用已登记评分。OD/OC是独立二值通道，不改为互斥softmax。

### 5.1 任务回报

序列方法：`T_i = mean_{h=0..3,c∈{OD,OC}} Dice_hard(p_i,h,c,y_h,c)`。

GR_CUR：逐个时间位置只用当前Dice；每个位置重新从**同一个共同状态**采样4个use候选，构成一个单步组。writer恒为.5且冻结。按预先固定的candidate index0提交状态以进入下一位置，不选择最高奖励候选。四个位置各自做组相对标准化；不把已经分叉、上下文不同的轨迹在同一位置冒称同一个单步输入组。

GR_SEQ/RET/EMA：4条完整候选轨迹，各4步；整条轨迹一个总回报R_i，赋给该轨迹4个决策的clipped surrogate。不是将不同患者／域难度直接组成任意奖励组。

### 5.2 保留项

同一采样策略pi_old，分别从 s_pre（保持该段历史未写入的参考）与每条轨迹 s_post_i 读出上述两个相同probe，得到形状[2 probes,2 channels]的Dice。

```
d_i = mean_{probe,channel}( relu(D_anchor - D_candidate_i) )
ret_i = exp(-20*d_i)
R_i = T_i + 0.05*ret_i
```

对每通道先hinge再平均，避免OD提升完全掩盖OC损失。reference probe值是共享常量，候选probe值随候选的状态后果改变。t=0没有既有历史时将保留项固定为1（组内常量），仍记录probe用于诊断。

λ=.05、α=20固定，不开启目标调参。保留项最多改变5pp的奖励数值，其在d=0附近的斜率绝对值为1；这不是5pp分割收益保证。原文λ=.5不适合不加解释地照抄到本轮奖励尺度。

**这是基于源真值的功能性回访损失，不是RaPO的token-KL漂移；没有无遗忘、临床安全或域解耦保证。** 所有RL奖励／advantage detach，不让奖励支路直接反传成另一种监督损失。

### 5.3 advantage 与EMA

非EMA组：`A_i=(R_i-mean(R))/[max(std_population(R),.005)+1e-8]`。

EMA组：初始 `sigma_ema=.01`；每个采样轮用4条轨迹总奖励的population std更新 `sigma_ema=.99*sigma_ema+.01*std(R)`，再用 `max(sigma_ema,.005)+1e-8`作分母。跨模式、episode、检查点和恢复不重置；PPO的第二个optimizer epoch不重复更新EMA。

全部RL组统一clip A到[-5,5]。同组奖励完全相同时A=0；不补噪声、不筛除失败候选、不重采直到有奖励差。记录zero-spread率；公共KL项仍按定义计算。

EMA是借鉴CTAN的持续奖励尺度归一化，本轮没有原论文那种明确任务边界；不声称完整复现CTAN的原实验。

## 6. 七个post-training对照：只开一条B载体研究线

| 名称 | 当前／未来任务目标 | writer | 保留项 | advantage |
|---|---|---|---|---|
| SUP_STATIC | 可微soft Dice，逐图reset | 不训练 | 无 | 不使用 |
| SUP_SEQ | 可微4步soft Dice，重参数化BPTT | 学习 | 无 | 不使用 |
| SUP_RET | 与SUP_SEQ相同 | 学习 | 同形式的soft Dice回访项 | 不使用 |
| GR_CUR | 当前硬Dice，4个单步组 | 恒.5、冻结 | 无 | 组内std |
| GR_SEQ | 4步硬Dice回报 | 学习 | 无 | 组内std |
| GR_RET | 4步硬Dice回报 | 学习 | 硬Dice回访项 | 组内std |
| GR_RET_EMA | 与GR_RET相同 | 学习 | 相同 | 持续EMA |

**增加SUP_STATIC和SUP_RET是为了防止把结构、额外监督或保留目标的作用全部归功于RL。** 这仍不是估计器的完全同目标因果实验：SUP用soft Dice、RL用hard Dice，报告时需说明。

所有SUP在post阶段使用 `L=1-mean(soft_Dice)`；SUP_RET再减`.05*ret_soft`；再加共同的策略参考KL项。soft Dice沿用(2sum(p*y)+1e-6)/(sum(p)+sum(y)+1e-6)。warmup才用原`seg_loss`，不得在post阶段偷偷添加另一套BCE、OC权重或边界loss。

SUP从相同的固定高斯ε重参数化采样use/write动作，4候选×4visit；B主干和载体权重冻结，但通过当前u、固定B运算和状态传递对策略进行BPTT。prefix detach，4visit内部不detach。SUP_RET的probe对s_post和当前use策略可微；anchor值detach。SUP_STATIC每次访问清m/q/h，writer不参与梯度。

## 7. PPO实现：不能把“奖励×任意loss”叫策略优化

两个reference严格区分：

- `pi_old`：生成当前候选组的精确采样策略；每采样轮更新一次，2个optimizer epoch内不变。
- `pi_ref`：该seed完成warmup后的冻结策略；整个post任务不变。

在RL收集过程中，冻结B与动作驱动的状态转移属于环境。缓存每一步实际observation、原始动作、old joint log-prob、回报；optimizer重放时observation和环境状态detach。不要重新用更新后策略生成环境状态，却继续使用旧log-prob。

```
ratio_i,t = exp(log_pi_theta(raw_action_i,t | stored_obs_i,t) - old_logprob_i,t)
L_PG = -mean(min(ratio*A, clip(ratio,.8,1.2)*A))
L = L_PG + .005*mean_active_dimensions(KL(N(mu_theta,.35²)||N(mu_ref,.35²)))
```

联合动作log-prob对实际有效动作维度求和；GR_CUR只包含9维use动作，writer冻结，不通过共享层间接改变writer。其余有记忆RL包含10维。raw正态到固定sigmoid/tanh映射的Jacobian在比率中抵消；对raw分布求精确解析KL。不能用EMA reference冒充pi_old；不能只将某个与候选无关的KL常量放入组奖励，随后声称它改变候选排序。

每轮2次full-group optimizer更新，缓存rollout不重新采样；第一次比率在旧参数处为1，第二次才可能触发clipping。fp64计算log-ratio与小矩阵，模型/策略fp32；所有非有限值按数值失败处理，不更换seed或静默修正log-ratio。

SUP每轮也2次optimizer更新，每个epoch重放同一外部窗口及固定ε，重新计算自身可微轨迹。第一轮候选预算相同，但SUP第二epoch需要额外主干前向／反传，而PPO重放仅计算小策略；**不可宣称相同GPU成本**。同时报告按采样轮、真实forward和GPU时间的曲线。

公共KL在各方法自己的当前/缓存observation上计算，reference预测detach；KL观察张量也detach，避免把KL对环境状态的梯度误算作保留学习。SUP_STATIC/GR_CUR只对有效use维度计算KL。

## 8. 训练规模、初始化与选择

### 8.1 每seed共享warmup

policy seeds=20260924..20260928。每seed先做一个2000更新的WARM任务，每更新8张源query，各自m=q=h=0，确定性均值动作，原`seg_loss`；只训练use网络，writer固定.5。AdamW lr3e-4，warmup100步，cosine到3e-5，wd1e-4，betas(.9,.999)，eps1e-8，clipnorm1。

使用原source fit query的8visit分块调度，固定模拟键`R10_WARM_QUERY|seed|episode|visit|group`，每32visit episode分4个更新。整个warmup完成2000步，固定终点作所有同seed方法的共同初始化与pi_ref，不用目标或sourceval选择warmup终点。

### 8.2 post-training

全部方法4000个采样轮、每轮2个optimizer epoch，G=4,H=4，不因早期负结果停。AdamW peak3e-5，前100轮线性warmup，cosine到3e-6；同一采样轮两个epoch使用相同LR。每个post任务重新初始化optimizer，不继承warmup Adam动量。保存500/1000/2000/4000轮。

每次checkpoint后，可用复制的只读策略作验证；restore所有训练RNG/计数，验证不能影响下一轮采样。恢复快照必须包括actor、optimizer、sampling-round/epoch位置、EMA、所有PRNG、环境调度与有效rollout／old policy身份。更简单的合法实现是在完整2epoch轮结束后原子提交，最多重放未提交轮并完整计费。

### 8.3 先14个探索任务，再6个固定配方重复

首两seed各训练7方法=14 post任务，全部完成后按源端规则分别选出一个SUP和一个GR方法。随后后三seed各训练已选SUP/GR=6 post任务。加5warmup，共 **25个训练任务**。

总计10,000 warmup optimizer更新、80,000 post采样轮、160,000 post optimizer更新。post首次rollout共1,280,000个候选预测访问；不包括第二个SUP epoch、prefix观察、回访probe、验证、基线或失败重放。

选择使用64个source-val、每个32visit、四模式各16个、确定性动作，`S=.5*mean(mode_macro_hardDice)+.5*min(mode_macro_hardDice)`。每模型选检查点，差≤1e-8选更早；每family按首两seed所选S等权平均选方法。SUP平局顺序STATIC→SEQ→RET；GR平局CUR→SEQ→RET→RET_EMA。不得强制选中RET_EMA，也不得按R10 target表改选择。

同权重随机/确定部署差距：仅在每个post任务source-selected checkpoint，用16个source-cal 32visit序列、四模式各4个，跑4个随机重复及确定性动作，作为诊断，不参与选参，不做目标best-of-G。

某方法数值失败不换种子，保留状态；只在两个发现seed都完整的family方法中按上述规则选择。family全部失败时，其确认分支不可执行，标记不完整；其他独立分支继续。缺失不填0、不将不完整轮伪称完成。

## 9. 目标矩阵：340核心，最多410槽位

所有R10源选择（包括确认seed检查点）锁定后才运行R10 target。主要订单0/1，2/3扩展，4回访。源端奖励和probe不部署。

| 部分 | 公式 | 槽位 |
|---|---|---:|
| 首两seed全7方法 | 7×2×2 orders | 28 |
| 两入选方法首两seed扩展 | 2×2×3 orders | 12 |
| 后三seed两方法 | 2×3×5 orders | 30 |
| 入选GR同权重消融 | 3×5×5 | 75 |
| 原生VPTTA/C/G | 3×5×5 | 75 |
| N/C0/载体FULL/载体RESET/原独立B_STATIC | 5×5 | 25 |
| WARM_STATIC | 5×5 | 25 |
| MIXED/LONG10 | 35个方法实例×2 | 70 |
| **核心** | | **340** |
| 70个主源模型槽位固定4000轮终点 | 完整身份相同则复用 | **最多70** |
| **总上限** | | **410** |

外部baseline seed按项目原生20260907..20260911，不假装与策略seed共用同一RNG路径。冻结父B及其RESET、原独立B_STATIC都来自20260924且无目标随机优化，按每order一份确定性控制。

主源模型70槽位的terminal敏感性永远是4000轮，不从target选择最好检查点；主结果永远是source-selected。主表统一四域、OD/OC、主orders、对应policy seeds等权；pooled值仅次要列。

### 三个同权重部署消融

- RESET_ALL：每图读之前清m/q/h；访问计数仍用于审计但不影响策略。永远不写初始记忆与此预测等价，作为alias而不是第四份证据。
- FORCE_WRITE：每图w=1，其余策略／u规则不变。
- CONST_HALF：每图w=.5，其余不变。

长期运行中这些消融的当前u会因过去记忆不同而不同；不能宣称整个流中“只有写入、当前输出完全不变”。只有D0的同一前缀单步配对才有当前输出严格一致。

### 压力流

每个流包含入选SUP/GR各5seed=10、WARM5、3外部基线各5seed=15、5确定控制=5，共35。

MIXED：用`SHA256(R10_MIXED_V1|registration_digest|content_id)`排序原1951次到达，平局原manifest索引。LONG10：原order0串接10轮、共19510次，持久方法不按轮清状态。逐轮报告，不把重复访问当新患者。

无失败、无复用时核心1,277,905次到达，最大1,414,475；主要评分访问分别1,110,225与1,228,875。slot与真实物理轨迹、历史复用、缓存计算分开记账。

R9结果仅可作已明确标注的附加背景；不依赖其是否完成，不强行解封。旧VPTTA/C/G只有完整输入、模型、算法、seed/顺序、预测与评分定义一致且有足够评分证据时可复用；缺少soft/OC/ASSD必要字段就按本轮定义重跑，不能补造值。

## 10. 数学与端到端测试：机器通过后直接继续

本包reference_math.py只给数学参考，不代替以下实际工程验证。

1. 同s_pre、同use动作，w=0/1当前预测相同；差异只在提交后。
2. w=0保持全部m/q/h；w=1、强制gain1/residual0时真实B full轨迹等价；reset等价原same-weight B reset。
3. raw动作log-prob、维度求和、pi_old比率=1、两epochclipping、解析KL、奖励detach均通过小型网络梯度测试。
4. GR_CUR组内同状态、固定branch0续流、writer严格无梯度；非共享网络不存在间接写入更新。
5. SUP BPTT梯度可经过4步状态流到早期use/write；主干和父B参数不更新。单步task对当步write梯度应为0，延迟／probe可非零。
6. 所有奖励相同A=0；EMA跨恢复精确一致、每collection仅更新一次；lambda0时ret奖励不改变排序。
7. 改变target mask不影响任何预测/写入；source label只进loss／verifier。遮蔽域ID、风格ID、future、score日志不改变策略接口。
8. 真实每图2F/0B/0Adam；GPU/profile smoke使用源数据；score进程仅在完整trajectory seal后读target masks。
9. EIO→真实worker evidence→queue restart→journal恢复→score→临时预测删除→磁盘reconcile及下一任务准入串联通过；全局三job恢复、每job一次。
10. 新协议不能复用R9硬编码43-source锁或771图去冒充R10；不用把R9常量全局monkey-patch成新值。通用数学和IO可复用，task graph/identity/budget/source_lock必须新定义。

普通性能不好、source-val没涨分或gate坍缩不触发科学早停；继续规定4000轮并如实报告。数值／身份／越界按影响范围停止，不自动更改奖励、LR、sigma或seed救结果。

## 11. 资源、存储、恢复与直接执行

**这是新增R10独立预算，不继承或挪用R9的512小时账本。**

| 资源 | 上限 |
|---|---:|
| GPU-worker账本时长 | 256小时 |
| 同时GPU worker | 1，串行 |
| 输出占用 | 64 GiB |
| 模型forward | 12,000,000 |
| backward调用 | 2,000,000 |
| optimizer步骤 | 1,000,000 |
| VJP/JVP资产构建 | 0（复用既有基） |
| 有限profile子预算 | 8 GPU小时、250k F、20k B／optimizer，计入总包 |
| 恢复 | 整轮最多3个job，每job一次额外attempt |

上述是预算上限，不是时间保证。真实profile覆盖最重SUP_RET、GR_RET_EMA采样、2epoch重放、source validation、probe、基线、IO、CPU score；保留完整首轮+逐资源前三项额外attempt的预算，profile成本也计入。采用逐个candidate及低维缓存，不为吞吐默认并发多个模型。若全矩阵不能装入预算，报一个真实OVER_CAP，不缩步数、不删除方法、不自动涨上限。

每次只保留一个完整float32概率轨迹；LONG10约38.11 GiB。封存→独立CPU评分→验证回执→删除R10临时概率→核对实际占用。训练actor/checkpoint、scalars、trace、prediction digest、失败证据保留；不重复拷贝父backbone。对旧R9文件绝不删除。长流临时文件与保留文件、原子写临时空间都要计入64GiB。

资源绑定从用户已准备的CTTA私有包和实际主机读取，不要求重新手填已有数据。历史资产协议仍使用已修复的Screen24身份，R10执行使用自己新协议；进程R8_SCOPE=FULL，但入口必须是R10。不启动旧R8/R9矩阵。

如果R9正在运行，不终止它、不篡改工作树、不超卖GPU；使用已授权且空闲的同资源集合，必要时排队。不要自行借用新的云服务、GPU账户或付费API。无小时heartbeat，只写本地进度／最终回执，只有硬阻塞才对用户中断通知。

原生baseline的名义每图调用必须计入：VPTTA=2F/1B/1Adam，C=8F/1B/1Adam，G=9F/2B/1Adam。不能只按新策略2F/0B计整轮资源；所有baseline、压力流和profile的反向与optimizer步骤都进入全包上限。

### 不再反复审阅的执行合同

用户转交本包prompt即授权：在上述科学／资源边界内实现R10、运行合成验收、绑定实际资产／GPU、有限真实profile、在真实admission PASS后自动执行完整矩阵并最后发布匿名汇总。允许由Codex产生新实现SHA及digest、填入私有执行授权；不要求每阶段再向用户询问。

本包不是声称基座13bd6a8已经实现了RL。Codex须先在独立worktree实现新增模块，自动测试通过并冻结实际新SHA后运行；旧SHA、原实验定义和结果不覆盖。第一次source训练前冻结全部数学与选参规则。预启动小型工程修复可在边界内完成并记录新SHA，不能借修bug换算法；正式运行后需要改变科学定义的缺陷必须停止受影响范围，不混合两个版本算一轮。

## 12. 统一交付与结论边界

目标分数整轮结束后统一解封；内部评分不得给操作者／另一个agent看中途排行榜。所有source-selection可以按规则自动完成，但不得消费target分数。

必须报告：所有7方法首两seed；固定入选SUP/GR的5seed；source-selected与4000终点；同权重writer消融；C0/VPTTA/C/G；OD/OC、四域、每order、共同有效ASSD、OC FP/FN和前景面积；gate分布、写入幅度、状态饱和、candidate reward spread/zero率、PPO clip率、KL、EMA、source随机/确定差距；LONG10逐轮；完整新调用、缓存／复用和费用。

研究进展参考：相对同口径C0及VPTTA均≥+0.5pp，且5个策略seed至少4个正差。若同时超过最强SUP控制和同权重CONST_HALF/RESET，才能进一步支持“奖励学习写入有额外价值”。阈值不是临床／显著性保证，也不用于早停。若只优于旧B、不优于SUP／C0，不能宣布RL成功；若写入基本关掉且与STATIC相同，不能宣布持续记忆贡献。

SUP与RL的软／硬目标以及实际计算成本不同，须披露；两个种子曾用于family选择，后三个用于固定方法重复；目标池已参与过去研发，不能改称新盲测。预算内负结果不等于RL或整个适配家族理论无效。

## 13. 固定证据定位

- RaPO：用户上传论文，§3.2.2、§3.2.3（PDF pp.4–5）；Appendix A（p.10）关于detached reward与非全局保证；§E（p.19）关于可验证奖励和任务边界。
- QPrompt-R1：用户上传论文，§3.3（pp.5–7）及p.2关于fully supervised/train-only GRQA；本方案没有复用它的类别prototype或claim其原指标。
- 当前B：`src/dpa_ctta/r8_ba/methods.py`，固定基座提交13bd6a8，`R8B.observe/update`及`correct`。
- 源序列：同提交`src/dpa_ctta/r8_ba/schedule.py`；协议分离：`docs/review/r9_current_first/ASSET_PROTOCOL_FIX.md`。
- R9只读工程参考：其admission/ledger/recovery/queue，复用时不得携带R9固定任务身份与执行许可。
