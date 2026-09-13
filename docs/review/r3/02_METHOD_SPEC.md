# R3 精确方法与实现规范

## A. 来源与设计边界

这五个框架是论文启发下的**新分割提案**，不是论文原方法的完整复现。BayesTTA 启发类别条件密度监督；KeepLoRA/SplitLoRA 启发更新方向约束；MoIE/ReservoirTTA 启发有界知识状态复用；LCA 启发历史对象与当前表征的兼容性；SPEGC 启发关系结构引导分割。正交统计迁移、下面的 BN 位移近端解、固定局部图求解器均为本项目明确指定的设计。

两份用户总结是摘要级材料，不自动支持这里所有公式、默认值或“首次”声明。完整来源范围见 `06_PAPER_PROVENANCE.md`。本批只筛选这五个固定实现，不声称已证明抗遗忘、风险校准、临床安全、跨任务统一或 TMI 可发表性。

## B. 共同 C 流程与允许修改点

### B1. 输入、输出与参数

- 仅 Fundus：原 ResUNet34、512×512 RGB、OD/OC 两个独立 sigmoid 通道。
- 原 checkpoint 与原预处理不变。C 的41层BN、82参数张量、19,136 affine标量在线可更新；其他模型参数冻结。
- C BN 使用当前单图前向统计，`track_running_stats=False`，running mean/var 为 None。不要新估计 source statistics。
- Adam：lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；每个当前访问恰好一次 Adam proposal。
- 六弱视图：原图＋固定依赖中的五个 Rotate_and_Flip 视图，分别 batch=1 前向、逆变换对齐。保留原 CPU 堆叠 logits→sigmoid→mean 的运算与随机顺序。
- `q` 是 detached 原 C 六视图软目标。强外观增强沿用原函数和原图保留方式，几何位置不变。
- 最终评分一律使用更新后原图的**原始网络概率**。不得把 density 教师或 graph 教师当成最终输出，不额外做预测集成或测试标签选臂。
- 每个完整流独立初始化；域之间不按真实域名 reset。模型只接收当前 tensor，不接收图像ID、域名、subset、未来内容。

### B2. 通用接口

建议组合式 `CStepEngine`，不要为了复用父类硬编码的计数而 monkey-patch 正式代码。接口表达为：

```
prepare_current(x, global_visit, old_state)    # 当前图已合法到达
weak_teacher() -> q_full, six_hard_grid, f0_grid
make_target_and_loss(q_full, strong_logits, fs_grid, snapshots)
propose_one_adam() -> delta, updated_moments
transform_actual_delta_if_needed(delta)       # 仅 U 三臂
predict_original() -> logits, optional_post_feature
commit_state_after_prediction()               # 当前图不得进入自己的记忆
return detached_logits, scalar_trace
```

`f0` 是更新前原图特征，`fs` 是强增强特征，`fplus` 是更新后最终原图特征。固定读取 `seg_head` 输入32通道。先按旧 `grid()` 的32×32中心位置采样，再沿特征维做 L2 normalize；norm eps=1e-6。

C 与 RP 必须保留旧运算时序。新方法不必继承旧私有方法的所有 hook，但必须在相同程序化输入、seed 下验证 C 与旧C，RP 与旧REGION 的步进等价。

### B3. 区域与统计

四个区域固定 `[OD_bg, OD_fg, OC_bg, OC_fg]`；同一个空间位置可在两个通道各出现一次。hard 定义是 q>=0.5。可靠条件：q<=0.1 或q>=0.9，且六视图 hard 均与q hard一致。

每区域最多选32个可靠token，使用独立、局部CPU RNG。尽量沿用 R1 的 `sample_<region>` seed 盐，不因为方法名不同就改变采样随机数。所有新随机控制有独立固定 namespace，不消耗强增强 RNG。

只存 CPU float64 的 n、mean、M2、贡献次数、basis/eigenvalues/版本。至少16张贡献图且128个真实token才 ready；每16次非空贡献刷新；rank<=8。当前访问消费旧快照；当前特征在最终输出固定后才进入统计。本批除明确 M 迁移外，均使用无限累计统计，不再把 R2_E 遗忘混入全部候选。

不能存图像、mask、dense feature、逐token队列或 Jacobian 到跨步 memory。固定32×32、32维，memory有明确上界。统计不具备差分隐私保证，不把无原图保存称为形式化隐私保护。

原 RP 重建项保持：

\[
L_{RP}=\operatorname{Avg}_{r,u}\frac{\|(I-U_rU_r^T)(v_s-\mu_r)\|^2}{\operatorname{sg}[\|v_s-\mu_r\|^2]+10^{-6}},\quad L=L_C+0.05L_{RP}.
\]

各区域内均值、有效区域等权，不多除一次特征维32。可靠筛选仅用于建 memory；RP loss仍对相应hard区域全部token计算。

### B4. 固定测量器

仅 T 三臂和 S 三臂需要额外一次 reference forward。测量器从**同一给定 checkpoint**构建，全参数冻结、标准 source BN eval、保留checkpoint已有buffers；不得用真实source图做重校准。与active C 不得共享会改变的BN对象或buffers。

它只测量特征（S还用特征全局统计），不输出一个要被平均进最终结果的source概率，不提供源训练监督。每张到达图运行一次，只在本步缓存；没有历史重编码或未来预缓存。构造模型、额外前向应隔离global RNG，不能改变 C 的强增强序列。三种T/S本批各自拥有独立测量器状态，不跨轨迹共享可变统计或预读缓存。

## C. 框架一：区域概率教师 T

主候选 T_LR；控制 T_ISO、T_DIAG。三臂均9次网络前向、1次loss backward、1次Adam。

### C1. 统计坐标与更新

从固定测量器提取单位特征 z_ref[1024,32]，用**本臂原C的q、六视图hard**选区域及可靠token。labels不使用source预测，也不使用修正q*或最终预测。预测固定后才合并 z_ref。

每个通道的前景/背景统计均ready，才允许该通道修正。不要求所有通道同时ready。query零向量不修正。

### C2. 协方差定义

由累计协方差 S=M2/(n-1) 构造共同收缩：

\[
\tau=\max(\mathrm{tr}(S)/32,10^{-4}),\quad S_\beta=0.9S+0.1\tau I.
\]

对数值舍入的小负特征值先投影到0；明显非PSD/非有限是错误，不能吞掉。收缩后特征值下限1e-4。

- **T_LR**：保留前8特征值，剩余24方向用其平均特征值代替。
- **T_DIAG**：使用 S_beta 的对角协方差。
- **T_ISO**：同一通道两类共用 `sigma²=(trace(S_beta_bg)+trace(S_beta_fg))/(2*32)` 的球形协方差，均值分开。这是简单原型教师控制。

不要把均衡采样的计数当成真实前景先验；第一版不额外估计／重平衡类别先验。

### C3. 教师修正

计算各类省略共同常数的对数密度：

\[
\ell_b(z)=-\frac12[(z-\mu_b)^T\Sigma_b^{-1}(z-\mu_b)+\log\det\Sigma_b].
\]

支持指标 g 为：前景/背景都ready、z非零，且两类最小 `Mahalanobis²/32 <=4`。这是固定的支持启发式，不是概率校准或置信界。

\[
\delta_c^{grid}=0.5g\,\operatorname{clip}((\ell_1-\ell_0)/32,-4,4).
\]

双线性上采样**log-odds增量**到512×512（align_corners=False），不把原q下采样后重新上采样以免无故损失分辨率：

\[
q_c^*=\sigma(\operatorname{logit}(q_c)+\operatorname{Up}(\delta_c^{grid})).
\]

求logit时q数值截断到[1e-5,1-1e-5]；delta=0的位置直接返回原q字节，cold全0时精确回到C。修正最大绝对logit增量2。loss=`BCEWithLogits(zstrong, detach(q*))`，**不再加RP重建项**。

Cholesky/线性求解在低维CPU float64中做；不对covariance/eigh反向。本轮不复制BayesTTA的CLIP文字先验、协方差假设检验或所有自步进机制。

## D. 框架二（原候选3）：区域几何约束实际更新 U

主候选 U_PCA；控制 U_RAND、U_SCALE。三臂均8次网络前向、1次loss backward、1次Adam，另有0–8次Jacobian VJP/图。

### D1. 构建特征到BN参数的映射

当前图的第一次原图前向仅在这三臂中启用gradient，保留f0图；用于q的logits仍detach。其余弱视图仍no_grad。

用旧快照、当前q/六视图的可靠集合，构建每个ready区域的两个探针（实际rank不足2时取实际rank）：

\[
b_{r,j}(\psi)=u_{r,j}^T\operatorname{mean}_{u\in selected(r)}v_0(u;\psi),\quad j=1,2.
\]

使用原memory的采样配额，不额外采更多token。空区域不生成probe；最多8个标量。对有序BN参数向量求一阶 `autograd.grad`，`create_graph=False`，得到A_raw[m,19136]。不对q、ids、basis反向。探针相对某参数未用到时该列填0，不把相应参数从optimizer移除。

A每行做L2归一化，norm<=1e-12的行省略。由逐标量反向得到的每一次都计入jacobian_vjp_calls，即使最后整行为0。最后一次VJP后释放原图graph，再运行强增强loss backward，防止跨步图残留。

U_RAND 用每区域固定seed的32×2正交随机特征方向代替前两PCA向量；其余probe采样、ready与归一化规则相同。它不是在19136维随意加噪声。

### D2. 必须处理真实Adam位移

先以**原C loss**计算g，调用base Adam一次更新moments并获得candidate参数，然后记录：

\[
\Delta=\psi_{Adam}-\psi_{before}.
\]

lambda=1、参数空间度量M=I，求：

\[
\Delta_* = \arg\min_d \tfrac12\|d-\Delta\|^2+\tfrac12\|Ad\|^2
=\Delta-A^T(I+AA^T)^{-1}A\Delta.
\]

只解m×m系统，m<=8；不得建立19136×19136矩阵。参数最终原地设为 `psi_before + delta_star`，随后原图预测。空A直接采用原Delta。

Adam moments按原g正常前进一步，不投影、不清空、不再调用第二次Adam；下一图从实际参数与这些moments开始。这是本方法明确定义的proposal-transform优化器，不冒称与“对投影后梯度做Adam”等价。

U_SCALE 在自身独立轨迹上也算同样A与Delta_star，但仅采用：

\[
\Delta_{scale}=\frac{\|\Delta_*\|}{\|\Delta\|}\Delta,
\]

Delta为0则仍0，比例仅处理数值舍入到[0,1]。因此比较的是方向保护是否超过同图同状态的范数缩小；不是用U_PCA轨迹的记录控制另一条轨迹。

### D3. Memory与断言

U三臂不加RP loss，仅维护shadow区域memory作为probe来源。最终预测后合入本图pre-update f0，累计方式与RP相同。记录真实proposal范数、应用范数、夹角、m、||A Delta||/||A Delta_star||与真实post probe差。小线性probe变化不是旧域功能保持保证。

32维PCA不能直接投影19136维参数梯度；必须有上述Jacobian。不能通过对原图no_grad后的特征调用requires_grad来伪造到BN参数的路径。

## E. 框架三（原候选4）：适配状态与区域记忆联合检索 S

主候选 S_JOINT；控制 S_SHARED、S_NOPCA。三臂均9次网络前向，恰好一个active slot及一次Adam/图。

### E1. 路由独立于各专家的当前参数

固定测量器当前head_input原始特征在空间上计算逐通道mean和population std，拼成64维向量再L2 normalize。descriptor不包含真实域名、图像ID、mask、subset或设备编号。

每条流最多3个条目，固定容量独立于真实域数4。每条目存descriptor向量和、assigned_count、历史到达时距离的Welford mean/M2。中心为归一化向量和，距离为单位descriptor间平方欧氏距离。

选择规则：

1. 空库则创建slot0。
2. 查询与已有center距离，选最小值（tie取最小slot id）。
3. 若最近slot此前至少接收16图，且本次距离大于 `max(1e-6, past_distance_mean+3*past_distance_sd)`，并且容量未满，则新建下一个slot。
4. 容量满时总是用最近slot；本批不淘汰、不融合、不扩容、不另外搜索阈值。
5. 新条目使用source affine、空Adam、空区域memory，不使用额外知识迁移初始化。这是与完整MoIE/KTI的明确差别。

新slot的源初始化只是算法依据无标签当前输入创建状态，不是按真实域边界reset。既有slot不被清空。路由中心与距离统计均在预测固定后提交；新条目首次距离作为0记录，不让其与旧条目的距离污染新半径。

该“均值+3SD”仅是启发式，非IID像素/图像下不作统计覆盖声明。可能只建立一个slot，可能过早用满3个；正常报告，不能为了让模块触发而改参数。

### E2. 三个控制的状态含义

| 臂 | affine/Adam | 区域memory | 在线目标 |
|---|---|---|---|
| S_JOINT | 每slot独立 | 每slot独立 | C+0.05*RP |
| S_SHARED | **全流共用一套** | 每slot独立 | C+0.05*选中slot的RP |
| S_NOPCA | 每slot独立 | 同容量shadow bank但不进loss | 原C |

S_NOPCA 与 S_JOINT 使用相同descriptor路由机制；没有把“移除PCA”的控制同时改成另一种router。S_SHARED检验只分记忆、却共享活跃模型是否已经足够。

所有slot共享一份冻结卷积/解码器/分割头，不复制3个完整模型。slot仅保存BN affine、按规范参数名关联的Adam状态、统计。不得混合Adam moments，未选slot完全不变。

全流只有一个强增强RNG流；切换slot不恢复过去随机数。每slot有自己的optimizer_step，所有slot步数和等于global_visit（S_SHARED则是唯一Adam计数）。冷创建、切换、装载不是额外optimizer step。

S_SHARED的region统计可能因共享模型变化而失配，这是控制有意保留的现象，不在这一臂偷偷加入M_TRANSPORT。

## F. 框架四（原候选2）：正交迁移区域统计 M

主候选 M_TRANSPORT；控制 M_IDPOST、M_SHUFFLE。三臂均8次网络前向、1次loss backward、1次Adam。

### F1. 三臂共同部分

当前更新仍使用旧RP绝对重建loss和旧快照。区别发生在预测以后：三臂都使用**同一预选位置**的post-update原图特征更新memory，而非旧RP的pre-update特征。

因此M_IDPOST是必要控制；仅与旧RP比较会混淆“是否迁移”与“存pre还是post”两个改动。

### F2. 迁移解

从原图更新前/后抓取同位置单位特征x_i,y_i，候选位置仍由pre-update q/六视图可靠性决定。每个非空区域权重相同，区域内按选中token数均分；OD/OC重复token权重照实计入。

若不同空间位置少于32个，Q=I并记录支持不足；不是重跑、不是停止适配。否则：

\[
Q=\arg\min_{Q^TQ=I}\sum_iw_i\|y_i-Qx_i\|^2+0.01\|Q-I\|_F^2.
\]

约定特征是列向量，代码样本是行向量；交叉矩阵 `H=sum w*y*x.T + 0.01 I`，SVD H=L diag(s) V.T，则 `Q=L@V.T`。属于O(32)，不人为加det=+1修正；不引入平移或尺度。

- M_TRANSPORT：同位置x/y配对。
- M_SHUFFLE：每个原区域内打乱y配对，x不变、token配额和w不变。单token区域无可打乱时照实记录。
- M_IDPOST：始终Q=I，不需要SVD，保留相同post存储。

### F3. 迁移所有历史对象，再合入当前post特征

在最终预测固定后，三臂分别执行自己的Q政策：

\[
\mu_r\leftarrow Q\mu_r,\quad M_{2,r}\leftarrow Q M_{2,r}Q^T,\quad U_r\leftarrow Q U_r.
\]

同时迁移**缓存快照中的mean和basis**。不能只转动live mean/M2，却让loss继续用旧frame快照。basis的特征值不因正交变换改变；重分解仍按每16次贡献刷新，不额外每图eigh。

区分 `frame_version`（每步维护的当前模型坐标）和 `basis_stat_version`（最近特征分解采用的数据截止位置）。当前t只消费此前t-1已经维护好的状态；不能用由当前post图拟合的Q去改变当前已完成的loss。

迁移后合入y_i。没有保存或重新读取旧图。此方法是局部正交近似，不等于精确重新编码所有旧特征；当前图拟合误差下降不是分割收益证据。

## G. 框架五：区域PCA驱动的局部图教师 G

主候选G_PCA；控制G_ISO、G_ORDER。三臂均8次网络前向、1次loss backward、1次Adam。固定32步小图求解不算网络forward，但单独计时。

### G1. 共同teacher与memory

使用当前原图pre-update单位f0、q-grid、可靠性，不使用额外冻结网络。memory仍按原C q/六视图构建，预测固定后并入pre-update f0；不能使用刚刚修正的Q*自证本图memory标签。

三臂维护同类shadow累计统计并采用相同启用规则：四个bank都ready才修改teacher，否则直接q* = q_full精确返回C。独立轨迹的实际ready次数不保证相同，只匹配规则。

### G2. 图边和度量

32×32网格，只有水平和垂直无向邻边，共1984条，无自环、无N²稠密全连接，不使用mask或真实边界。

G_PCA对每bank采用T中相同收缩及rank8协方差，求precision并trace归一化：

\[
M_r=\Sigma_r^{-1}/(\mathrm{tr}(\Sigma_r^{-1})/32).
\]

对节点u，四个软区域权重为 `[(1-qOD)/2, qOD/2, (1-qOC)/2, qOC/2]`。边度量为两端权重的均值混合：

\[
M_{uv}=\tfrac12\sum_r(a_{u,r}+a_{v,r})M_r,\quad
w_{uv}=\exp[-(z_u-z_v)^TM_{uv}(z_u-z_v)/0.1].
\]

G_ISO以I代替全部M_r；G_ORDER不传播边，只做相同类别包含约束。不能按hard区域完全切断所有跨区域边，使错误预测没有修正空间。

### G3. 图目标与求解

独立Bernoulli目标，不是OD/OC softmax。固定 `a=10` 对可靠通道token，其他 `a=1`：

\[
\min_{0<Q_{OC}\le Q_{OD}<1}
\sum_{u,c}a_{u,c}KL(Ber(Q_{u,c})\Vert Ber(q_{u,c}))
+0.25\sum_{(u,v),c}w_{uv}(Q_{u,c}-Q_{v,c})^2.
\]

注意KL方向。q概率数值clip到[1e-5,1-1e-5]。使用固定32次mirror/prox迭代，不随机收敛、不追加迭代看分数：

```
Q = weighted_nested_projection(q, a)
repeat 32:
    g[u] = 2*0.25 * sum_v w[u,v]*(Q[u]-Q[v])
    b = 1 + 0.25*a
    v = sigmoid((logit(Q) + 0.25*a*logit(q) - 0.25*g)/b)
    Q = weighted_nested_projection(v, b)
```

当vOC>vOD时，weighted forward-KL投影将两者设为：

\[
\sigma\left(\frac{b_{OD}\operatorname{logit}(v_{OD})+b_{OC}\operatorname{logit}(v_{OC})}{b_{OD}+b_{OC}}\right).
\]

否则不改变v（数值clip除外）。G_ORDER直接用a对q做一次相同projection。零边情况下图求解必须退化到ORDER控制。

32步是固定计算预算，不宣称精确全局求解。eigh、图权重、求解过程全部detach，不能再加另一套图网络训练。

### G4. 回到全分辨率与最终输出

将 `logit(Q_grid)-logit(q_grid)` 双线性上采样，与原q_full的logit相加；再以双线性上采样的a_grid为权重，在每个full-resolution位置做同一个OD/OC KL projection。

这样不把原q_full整体低分辨率重建，同时让**教师目标**满足包含关系。G_ORDER也走相同grid correction→full projection路径。只有teacher受这个约束，最终原图网络输出不再额外投影，不能宣称其hard mask保证嵌套。

loss=原 `BCEWithLogits(zstrong,detach(q*))`；无旧RP项。memory只使用未修正q。

## H. 数据流因果性与状态归纳

| 类别 | 修改发生处 | 跨图像保留 | 本图memory写入 |
|---|---|---|---|
| C | 原BCE更新 | affine/Adam | 无 |
| RP | 额外绝对重建 | affine/Adam＋累计bank | pre f0 |
| T | teacher生成 | affine/Adam＋固定坐标bank | reference f0、原q labels |
| U | 真实Adam位移 | affine/Adam＋shadow bank | pre f0 |
| S | 模型状态选择＋局部RP | 有界slot池、全局RNG | 只写选中slot |
| M | 预测后的统计维护 | affine/Adam＋可迁移bank | post fplus |
| G | teacher的空间求解 | affine/Adam＋累计bank | pre f0、原q labels |

可选的teacher纠错计数只能在host已固定输出与状态后，由evaluator读取mask计算，放在本条记录的offline diagnostics中。它们不产生模型前向，不反馈teacher、router、memory或参数。不得为这些诊断缓存当前mask供下一图使用。

## I. 建议代码文件与异常边界

```
src/dpa_ctta/r3/
  core.py             # composition C engine, declared phases, feature taps
  stats.py            # stable small covariance snapshots, shared regional selection
  density.py          # T family
  displacement.py     # U family and parameter flatten order
  contexts.py         # S family and independent small optimizer states
  transport.py        # M family incl cached-frame migration
  graph.py            # G family
  hosts.py            # explicit dispatch; no undocumented arm switches
  streams.py          # metadata-only extra recurrence
  plan.py / run.py / analyze.py
scripts/run_r3.py
configs/r3_science_v1.json
```

架构/路径名称是建议，不强制无意义拆文件；科学定义与计数不能变化。使用已安装PyTorch/NumPy，不下载新训练组件。不需要引入scikit-learn IPCA、外部QP求解器、图神经库或训练额外模型。

数学非有限、错误shape、真值进入算法、超容量、不可解释的参数变化属于工程错误。bank未ready、无前景、零probe、router只用一个slot属于正常算法状态，记录但不停止整条流。不要再引入先验效果gate。
