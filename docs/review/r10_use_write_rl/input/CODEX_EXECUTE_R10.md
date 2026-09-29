# 给 Codex：直接实现并运行 R10_USE_WRITE_RL_V1

本条是用户转交后的**新增R10实现、验收、有限真实profile、准入通过后自动完整执行**任务，不是再交一个“等待审阅”的实现稿。完整科学定义见本包 `R10_EXPERIMENT_PLAN.md`，机器矩阵见 `R10_SPEC_AND_MATRIX.json`；`reference_math.py` 是数学参考，不是完整实验runner。

## 一、执行授权与不变边界

我授权你在本任务规定的科学、数据和资源范围内完成以下工作，无须在每个阶段再次请求人工批准：

1. 在独立worktree实现R10、运行必要合成回归，生成并冻结真实新代码SHA及inventory；
2. 从我已经准备的CTTA私有材料解析资产、主机、GPU UUID及输出目录，完成只读元数据／二进制验证；
3. 在8 GPU-worker小时的有限子预算内进行真实源数据、生产核、评分与IO profile；
4. 真实完整矩阵admission PASS后，依据本指令生成绑定最终SHA、资产／GPU／root、profile摘要的私有授权，自动运行完整R10；
5. 全部任务终结后统一解封目标分数，输出并发布匿名汇总、配置和代码，不发布患者级数据或私有路径。

不要求我再次填写已有材料中的字段。动态生成的SHA／profile digest由实际结果填入，不能伪造。公开disabled模板仍保持禁用；profile阶段正式队列开关仍false，PASS后才开启私有运行配置。不得绕过检查或把旧R9许可当成新许可。

**新预算**：R10单独最多256 GPU-worker小时、64 GiB输出、12M模型forward、2M backward、1M optimizer步骤、0新增VJP/JVP；至多1个GPU worker。profile最多8小时/250k F/20k B及optimizer，全部计入上述总量。科学矩阵不变的有限工程修复可自动完成并冻结新SHA；不能改变方法定义、任务数、种子、奖励、hyperparameters来绕过失败。

R9是否正在运行请从真实环境判断；不停止、不抢占、不修改其worktree、输出、授权或预算。在现有已授权GPU集合内使用空闲资源或排队，不购买新资源、不接外部API、不创建小时monitor。常规完成一个阶段后直接进入下一阶段；只在真实硬阻塞或必须改变科学定义时一次性报告。

## 二、代码和资产起点

- repository：`DLwbm123/DPA-CTTA`
- base SHA：`13bd6a8cdf9c30a0a5ed7fa464c67e72301ac8c6`
- 新命名空间：`src/dpa_ctta/r10_use_write_rl`
- 新分支建议：`experiment/r10-use-write-rl-v1`
- 进程 `R8_SCOPE=FULL`；运行R10入口，不发射旧R8/R9矩阵。
- 固定父载体：Screen24真实用于目标部署的 `B_FULL_20260924`，B64/global/amplitude .3，所有权重冻结。从index/receipt找真实部署包；不依据目标分数换成R9赢家，也不等待R9完成。
- 五个R10 policy seeds：20260924..20260928；共享同一个父载体。源数据fit/cal/val、backbone、basis、scaler、gradient scale及校准kappa全部按真实摘要绑定。
- 新R10协议与历史Screen24资产协议分开。不要再次通过切换SCREEN24进程范围来通过旧资产校验。

允许复用原B的observe、stable、correct、网络和原生基线数学，以及通用journal/evidence实现；不能将R9的43-source source_lock、spec、caps、queue直接换几个数字后运行。R10必须有自己的完整任务图、授权、身份及预算账本。

## 三、先实现这一个读写分离控制器

持久状态只有 `m[64], q[32], h(scalar)`，初值0；审计visit不进策略。无额外旧d、z、probability、token缓存跨目标图保留。

```
difference = h*d - q
prior = stable(W) @ m + bias(d) + stable(G,1) @ difference
z_tilde = 原B五步ISTA(prior, o(d), H, frozen_kappa)
obs = concat(clip(d), clip(m/s), clip(z_tilde/s), clip(difference), h)  # 193
```

clip只对策略特征至[-10,10]，不改真实状态和载体数学。use network=193→128→64→9，write network=193→64→32→1，SiLU、无共享参数；mu=5*tanh(head/5)，raw action正态std固定.35。

```
gain = sigmoid(a0)
residual[:8] = .5*s[:8]*tanh(a1..a8); residual[8:] = 0
u = gain*z_tilde + residual
w = sigmoid(a9)
pred = frozen_seg(x, basis@u)
m_new = (1-w)*m+w*u
q_new = (1-w)*q+w*d
h_new = (1-w)*h+w
```

先预测后原子提交。部署仅a=mu，2F/0B/0Adam；不做G候选搜优，不部署reward/reference/EMA，不读取标签、模式或未来。

完成真实函数测试：同prefix/同u时w不改当前pred；w0保持全部字段；w1+identity-use与原B full轨迹等价；reset与同权重B reset等价。

## 四、训练流程固定，不让实现者再选算法

### 4.1 共同warmup

每seed一个2000更新任务，每步8张源query逐图reset；只训练use network，writer=.5。use初始gain=.8/residual0；用原seg_loss，AdamW lr3e-4→3e-5/warmup100，wd1e-4/betas(.9,.999)/eps1e-8/clip1。固定2000终点同时作为所有该seed方法的初始化与pi_ref。

### 4.2 七个post方法

- `SUP_STATIC`：逐图reset，soft Dice的可微训练。
- `SUP_SEQ`：4步序列，重参数化BPTT，soft Dice。
- `SUP_RET`：SUP_SEQ加同形式soft Dice回访项。
- `GR_CUR`：单步硬Dice组比较，writer恒.5、无梯度；每位置4候选共同state，固定index0续流。
- `GR_SEQ`：4条4步轨迹，序列硬Dice回报。
- `GR_RET`：GR_SEQ加功能性回访奖励。
- `GR_RET_EMA`：GR_RET再加持续奖励尺度EMA。

每方法4000collection rounds，G4/H4，每轮2个optimizer epoch；AdamW peak3e-5→3e-6/warmup100 rounds，其余同warmup。2epoch同LR，第二个PPO epoch复用rollout，不再生成动作。保存500/1000/2000/4000。SUP每epoch用固定ε重新计算可微轨迹，记录其额外模型前向；不宣称与RL相同GPU预算。

prefix在每collection按当前pi_old确定性重建，detach；4步内部SUP不detach。计划中的具体源episode/window和probe哈希规则必须原样实现，不把风格ID交给策略。

### 4.3 奖励和策略梯度

GR任务回报为OD/OC同权硬Dice。RET用两个不同源group的旧风格probe，从同一个pi_old分别读取pre-window memory与candidate post-memory；每probe克隆、不提交：

```
deficit = mean_{probe,channel}(relu(D_anchor-D_candidate))
ret = exp(-20*deficit)
R = task + .05*ret
```

t=0无既有history时ret=1常量。两probe不得与当前4图同group。全是source fit标签，不能用target reward。

非EMA用组内population std；EMA=.99*old+.01*std，初始.01，跨恢复不清、每collection仅一次。分母floor=.005、eps1e-8、A clip[-5,5]。同组奖励相同A=0，不重采。

pi_old是精确采样策略，pi_ref是冻结warmup，不混淆。使用raw高斯joint log-prob；PPOclip=.2；共同KL系数.005，解析等方差正态forward KL按有效维度平均。缓存observation/action/old logprob/reward，重放时环境state和reward detach；GR_CUR只对use9维算比率／KL，其writer网络完全冻结。

SUP post主损失是`1-mean(softDice)`，RET再减`.05*ret_soft`，加共同KL；只在warmup用原seg_loss。目标difference源soft/hard不同如实披露，不宣称纯RL估计器单因素比较。

### 4.4 D0和选择

首两warmup各64个source-val上下文，每当前use候选强制write0/1，共4use候选、未来4图。当前必须相同，报告后续差；诊断无增益不取消训练。

首两seed七方法全训=14post。按64个source-val 32visit、四模式各16、确定性动作选检查点，S=.5*模式均值+.5*最差模式均值，tie1e-8更早。两个seed的S平均在SUP三组、GR四组分别选一组；顺序tie按机器spec。确认seed三组各训练所选SUP/GR=6post。加5warmup，共25源训练任务。

奖励和选择规则第一项真实训练前冻结。不得读取R10 target分数挑方法、checkpoint或奖励尺度。不要强制让GR_RET_EMA成为最后赢家。

## 五、目标矩阵与baseline

执行JSON中340核心+最多70固定4000终点槽位；同artifact可alias但不能假造新物理运行。

- 主方法70槽位（首两seed全方法28，入选扩展12，后三seed确认30）。
- 入选GR `RESET_ALL/FORCE_WRITE/CONST_HALF` 同权重消融75。
- VPTTA/C/G各5seed×5order=75；原生算法不改。
- N/C0/固定父B_FULL/其RESET/原独立B_STATIC共25。
- 五个WARM_STATIC共25。
- 35方法实例的MIXED和LONG10共70。

重用旧结果须满足完整身份和全部本轮评分字段，缺少soft/ASSD/OC信息就重跑。N/C0/固定父B等确定控制不冒充五种子证据。baseline seed为20260907..11。

全部源选择锁定后才能运行R10 target。每trajectory预测封存→CPU独立score→验证回执→回收本R10临时概率→下一个任务。全程不向人或其他agent公开中途target leaderboard，不让target score反馈排程或训练。LONG10逐轮报告，不挑最好轮。

## 六、自动验收、profile、执行和失败边界

必要新测试见plan §10。运行本包数学/计数检查，但不能把它们当成实际ResUNet/CUDA验收。新增source RL runner、真实B读写、PPO旧概率、writer梯度、counterfactual probe、真实queue故障恢复必须各有连接测试。

通过后自动冻结新实现SHA和inventory、完成资产二进制绑定并profile。有限profile使用源数据和实际生产核，不发射整个16k或4000轮训练来测速度。预算包含PPO/SUP差异、全部baseline和LONG10；原生VPTTA/C/G计数为2F/1B、8F/1B、9F/2B，每图一次Adam，不能漏算其反向。

实际admission需证明完整首次任务+逐资源最贵三个额外attempt+profile都不超过硬上限。每job最多一次恢复，整轮最多三个job；target online/score共用job额度；沿用已修复的结构化evidence、持久attempt、磁盘当前占用核账。

超限、资产摘要冲突、数据隔离违规、非有限数值、真实实现缺陷才按范围停止。任务负性能、gate关闭、source reward plateau不早停；不能自动调LR/奖励/sigma/seed、删域、删除baseline或改标签阈值来救实验。

启动后输出真实launch receipt，不是“启动计划”；然后持续执行预定有限queue。不创建新小时heartbeat。若源job数值失败，只继续真正独立分支并保留缺失，不能自动把失败方法替换为未登记新方法。

## 七、最终交付

交付真实运行新SHA、配置及完整任务状态；所有首两seed方法、入选五seed、source-selected/4000终点、writer消融、C0/VPTTA/C/G、OD/OC/域序/ASSD/OC错误、source gate/reward/clip/KL/EMA/采样部署差距、LONG10每轮及实耗/恢复/缓存/复用。

最终问题是：

1. 相同当前分割下，写入决定是否显著改变后续表现（描述与统计边界分清）？
2. 序列奖励是否比当前奖励更有价值？
3. 保留奖励／EMA是否超出普通同结构监督训练？
4. 是否同时超过C0和VPTTA，并接近或超过C/G？
5. 结果改善来自RL信用分配，还是更多源训练、简单门控或回到无记忆？

不能只交最高Dice、宣称“用了GRPO因此有创新”或把方法族失败判为理论不可能。整轮运行状态完整／不完整和原因据实报告。

**请从实现开始连续推进到真实实验结果；不再停在“已写代码、等待下一次审阅”。**
