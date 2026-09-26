# Codex 实施任务：R9_CURRENT_FIRST_V1

请基于 DPA-CTTA 的 Screen24 固定发布提交
`a9d3a43959da73ef552401342cc087d5ae8e6ba2`
实现本包的 R9 计划。先完整阅读 `R9_ANALYSIS_AND_PLAN.md` 与
`R9_SPEC_AND_MATRIX.json`；后者含展开 source/target 槽位。

**本条是实现/审阅任务说明，不是已有真实训练授权。**
`execution_authorized=false`；真实数据、checkpoint、GPU身份、私有输出root及新的exact代码SHA必须绑定后，另按用户明确授权运行。不要自行发射旧R8全量矩阵，不新建小时监测，不覆盖旧报告。

## 一、先核实科学起点

Screen24 实际是10源任务×4000步、40目标轨迹、orders0/1、两个源种子20260924/25；不是65/724全R8。A32/B64 amplitude0.3的真实配置从已执行科学artifact读取。实际原训练日程warmup200、总16000步cosine，4000只是前缀。

主表约为：C0 75.0794，VPTTA 74.4011，MLP 74.3660，
A_STATIC74.2142、A_FULL74.0085，
B_STATIC73.5099、B_FULL73.2491，单位Dice%。
不得采用报告首页以外不匹配的按图平均替换四域同权主指标。

已核实旧源码：
- `src/dpa_ctta/r8_ba/trainer.py`：support观察→state→不同query分割。
- `src/dpa_ctta/r8_ba/host.py`：当前图观察→state→同当前图分割。
- `src/dpa_ctta/r8_ba/gradient.py`：已有B_G1/B_G3/COLD_G3数学和输入路径。
旧设计差异不是未经验证就应“修复”的bug；R9 SELF是独立实验因子。

## 二、实施内容

建议在独立 `r9_current_first` 命名空间实现，不改写R8科学定义。

### 1. 三种源训练规律

A/B各 `LEGACY`、`SELF`、`SELF_TASK` ×FULL/STATIC，
另 `MLP_LEGACY`、`MLP_SELF`。首批种子20260924/25，共28个任务。

LEGACY保存原support/query及aux定义，验证完整pre-cal fit快照后从4000分叉到16000。
SELF让观察图与分割query是同一styled图，A clean参考也来自同一query，其他aux不改。
SELF_TASK只在fit中去掉所有aux，保留原`seg_loss`及匹配的校准流程。
MLP_SELF是匹配MLP的同图训练，不能新增容量。
每个visit标签只进入离线loss，不能进入state update的输入。

A32/B64、0.3幅度、source分组、基、scaler、源checkpoint、BN、
预处理、原simulator与原图内OD/OCloss均不改。
保留query调度和RNG绑定；不把更好的target表当作source选择信息。

### 2. 16000步，不以分数提前终止

原AdamW/200 warmup/16k cosine，32visit episode、8visit TBPTT。
FULL chunk间detach不reset；STATIC每visit reset。
checkpoint=4000/8000/12000/16000，全部跑到16000后再按source规则发布。
校准每checkpoint独立256步。

LEGACY必须恢复主fit模型、optimizer、RNG、步数、episode状态、身份；
禁止从只有模型的部署artifact装空Adam并宣称“等价恢复”。
不可恢复时按预登记fresh16k替代、单列原因，额外40k更新计入688k最坏上限。

源验证64条32visit episode，四类各16条，部署对齐同图预测。
S=.5*四类平均hardDice+.5*最差类别平均hardDice；
OD/OC同权，softDice另外报告。
各模型选checkpoint，tie1e-8更早；各路线用首两seed FULL/STATIC的S平均选配方，
平局LEGACY/SELF/SELF_TASK。MLP_SELF固定。
先冻结配方再增加20260926/27/28三个种子，共15任务；不按新增seed目标结果重选。
总43个源任务，正常新增648k主fit更新，最坏fresh替代688k。

### 3. 固定Screen24部署诊断40槽位

原A/B Full两seed、orders01：
A_RESET/A_PRECAL/B_RESET/B_PRECAL/B_PRED_ONLY/B_ISTA20共24；
A/B输出FiLM residual-only alpha=.25/.5共16。
alpha不改变存储z及递推，alpha0/1经真实等价检查后复用C0/原FULL。
PRECAL为原tau/kappa=1，不是重训，B步长必须同步。
目标分数不回流新的source选择器。

### 4. 最终同权重机制90槽位

六消融×5seed×orders01=60，
A/B_RESET×5seed×orders234=30。
RESET用FULL权重，不能用独立STATIC代替。
所有方法输出封存后再评分，不能按结果更改策略。

### 5. 六种目标梯度诊断150槽位

B_G1/B_G3/B_RESET_G3/COLD_G1/COLD_G3/BN_RESET_G1，
5seed×5order。B_G1/G3将校正后z提交历史；RESET/COLD不保留适配历史。
六视图teacher全部来自未调制C0，固定detach；
原观察前向可复用，五个几何视图必须inverse；
一个强外观视图按旧归一化路径生成并在K步复用。
latent尺度取已验证source-fit约定，floor1e-3。
loss=BCE+.01*mean((u-u0)^2)，每图freshAdam。
latent LR{.001,.01,.1}、BN-reset LR{1e-5,1e-4,1e-3}；
每臂/LR在64个source-cal四visit episode中评分选择，不读目标标签。
BN只更新已注册affine，每图恢复源值、清Adam；
单步prox初始梯度0，不能把它宣传为独立新稳定机制。
逐图验证8F/1B/1Adam或10F/3B/3Adam，无未计量前向。
不要称梯度臂为gradient-free。

### 6. 基线与压力流

VPTTA/C/G按各自已冻结算法运行，5baseline seeds20260907–11、5orders；
N/C0每order一份确定性结果。不能为了“统一”取消VPTTA memory/warmup。
基线旧数据仅在所有身份/流/代码/评分一致后复用，记录零新模型调用。

压力方法=5个入选源模型＋VPTTA/C/G，各5seed，
加N/C0确定性控制，共84槽位：
MIXED固定一次SHA256顺序；
LONG10十次原order0串接、不清历史，逐轮输出。
不把10轮/5seed/5orders视为新患者，也不把重复内容误去重掉。

### 7. 任务数和资源

以机器矩阵为准，核心610目标评估槽位；
161个主源模型槽位另留固定16k敏感性，重复checkpoint去重后最多771。
无复用最大2,241,699次到达（含LONG10），主要评分访问最多1,947,555。
槽位数不等于新物理轨迹数，分别记录。

拟议资源上限：3个独立GPU worker，512 GPU-worker账本小时，
64GiB输出，32M forward，3M backward，3M optimizer，4096 VJP/JVP。
这是计划预算不是时间预测。先profile并推导整包是否装得进上限，
也计入校准、验证、旧快照不可恢复替代和允许的基础设施恢复。
不得在执行中缩短步数或删任务却宣布完成；不需要刻意耗尽预算。

## 三、必须通过的针对性验证

- SELF同图输入、A clean参照、source梯度到模块不误更新backbone；
- LEGACY保持性、split/alias/version校验；
- FULL/TBPTT/STATIC/RESET语义；
- PRECAL尺度/ISTA20准确次数、output-alpha不改历史；
- 目标标签隔离、source选择前不能读目标分数；
- 现有GradientHost六视图、强增强归一化与单独reference数值一致；
- BN-reset每张真正恢复参数与空优化器；
- pervisit模型F/B/Adam实际hook计数；
- source snapshot不是cal后部署包，完整恢复下一步等价；
- 原子commit/host-local锁/单次基础设施恢复；
- unequal-domain metric test、ASSD undefined、LOOP10重复访问唯一索引；
- 复用旧结果的精确身份与物理零增量。

`verify_plan.py`只是计划计数检查，绝不能作为上述模型验收的替代。

## 四、工程和科学边界

使用host-local锁、原子journal和逐任务receipt。单任务预估时长为soft；
总量cap、数据隔离、数值错误及用户停机分别处理。禁止因为接近预计时间
而像上轮scaler一样提前中止；禁止以“24小时必须完成”静默缩减科学矩阵。

每任务最多一次预授权等价基础设施恢复；保留失败记录/预约费用/实际资源。
不能自动调LR/ridge/seed/AMP修复NaN，也不能把资源越界改名为基础设施故障。
科学负结果照常完成，数值或身份错误按影响范围停止。
不得创建新的小时heartbeat或后台监测授权。

公开只含匿名aggregate、代码、配置和cost，
不公开患者标识、逐图私有轨迹、源/目标图像、mask、checkpoint、凭据和私有路径。

## 五、交付次序

1. 实施代码、disabled配置、解析后的任务清单、差异说明和针对性验证。
2. 新exact代码SHA、既有artifact/snapshot身份、source/target/资源绑定及profile账本。
3. 获得明确的新运行授权后再执行，不沿用Screen24已完成scope。
4. 完整报告所有配方、源码机制、各seed域序、软/硬Dice/OC误差、ASSD、
   源选择与固定16k、压力逐轮、实际新调用/复用/恢复成本和未完成原因。
5. 不自动签发scientific winner、external PASS、显著性/临床安全或全家族极限。

最终应回答：长训练是否有用；SELF是否比LEGACY好；
代理辅助项是否造成损失；同权重历史是否有益；
目标梯度是否必要；FiLM与BN参数化在哪些条件下有效。
不能只交一个最高Dice。
