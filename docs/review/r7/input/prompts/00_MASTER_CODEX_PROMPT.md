# 给Codex的总任务：R7三组独立实现，先审阅后实验

请以整个R7_THREE_TRACKS_IMPLEMENTATION_PACKAGE.zip为输入。本任务授权的范围仅Stage I：实现三组方法、程序化CPU验证、元数据dry-run与审阅交付。**不授权真实源端训练、目标图像运行、checkpoint读取或任何GPU查询/初始化。**

工作仓库：DLwbm123/DPA-CTTA。
基线发布SHA：00ccb401942e9f1d3fed48ccb7fb3777c87714b8。
旧C实现SHA：c94fff7c05cec38d541c441b62a9381ee46ba076。
固定CTTA：dbff0d985c6c95345d9fb78f5b1daef57b392564。
固定GraTa：33ae20d664f305af34739ec54a5bec7da53ffa0b。
建议协调分支experiment/r7-three-track-implementation-v1，各组独立worktree。

## 开始前
1. 核对MANIFEST全部字节/长度/SHA。只把input原件归档，不重排JSON、不更改status为实现状态。科学配置与运行绑定分别摘要。
2. 读取MASTER_PLAN、COMMON_PROTOCOL、三份METHOD、specs、PROVENANCE和当前代码。代码定义与spec发生冲突则提交差异，不自己发明缺失方法。
3. 保留用户工作区、main、R1–R6历史结果。不得修改旧C、旧science/gate/validator。新功能置于src/dpa_ctta/r7_*和tests/r7_*及新scripts目录。
4. 先由协调任务完成共享接口，再分组执行三个Prompt；共享代码只能单写者维护。若不能并行，按A/B/C顺序完成，不能声称使用了不存在的子agent。

## 必须实现的共享部分
- 原随机初始化完整ResUNet的两处up1/up3 FiLM（1024ambient维度），零state精确identity但source梯度不能被跳过。
- current统计BN且backbone/BN-affine冻结；C_BASE保留原路径，C0单forward。新方法observer/final两forward，所有online可训练参数冻结。
- 公共观察器134→64→32与64×64患者tokens；fit-only scaler；患者tokens不持久保存。
- 源fit/cal/val登记、光度模拟器、source proxy oracle、A基构建和共享B/C基构建、训练/校准/只读target接口。真实训练入口全部disabled。
- FULL与STATIC分别训练的能力、序列中support/query隔离和4步source unroll。禁止query进入state，禁止target图或label参与源训练、阈值、字典、scaler。
- 原metadata流、独立manifest/split hashes、实现/科学/训练资产/推理绑定；新输出owner保护，第一异常与计算成本保留。

阶段I可以缺真实source资产绑定，但必须记录SOURCE_BINDING_PENDING和所有null字段；不能拿target替代，也不能为了补真实绑定去读checkpoint。元数据足够时补source split dry-run，但不读像素。方法核心、维度、源训练器、数值算法未完成则不允许READY。

## 三组目标
A_PSF：16维Gaussian filtering、full16×16稳定F、完整P、双代码本、source predictive covariance basis。只做causal filtering，不做smoothing，不宣称完整RP-GSSM复现。
B_RCA：32维adapter递归状态，appearance descriptor差分，源学习字典和恰好5步ISTA，source calibration kappa必须同步进入能量与step-size。
C_RBE：64patch×8维条件证据，噪声课程variance meta-model，恰好3步Huber IRLS；不假装实现了前门因果识别或原GUIDE分类头。

## 实验计划入口
TARGET_SCREEN：8臂×三流=24条；C_BASE/C0/A_FULL/A_STATIC/B_FULL/B_STATIC/C_FULL/C_STATIC。
TARGET_MECHANISM：A_ISO_OBS/B_PRED_ONLY/C_CONST_R×三流，最多9条；未授权。
TARGET_EXTENSION：最多两组FULL/STATIC + C_BASE/C0×order2/3，最多12条；未授权。
SOURCE_PREP只生成源权重不自动跑TARGET_SCREEN。所有scope receipt必须分别批准，不能用代码review PASS冒充执行授权。

不要恢复旧+.5pp全AND gate。本轮实现科学评审报告生成器，不根据target分数自动提名或启动后续。报告FULL-C和FULL-own-STATIC两类效应，不把所有源端准备收益归因持续性。

## CPU验收
先运行包内独立math reference，再用独立实现比较生产模块。设置PYTHONDONTWRITEBYTECODE=1，新日志和R7_MATH_RESULT必须指向新work目录，不覆盖input历史日志。Math reference通过不等于完整方法通过。实际测量测试数量、耗时、失败、skip和调用，不预填通过数。

至少覆盖：
- FiLM零identity及非零导数、介入点形状、原C窄保持性。
- A过滤精度/协方差SPD/full-F相关性、source基的rank和预测协方差截断；无未来观测。
- B soft threshold/5步能量/零差分/kappa≠1；source梯度穿过近端迭代；state恢复。
- C Huber/3步IRLS/离群证据/constantR来源；variance校准不更新task模块。
- source support/query组不重合、fold隔离、val/cal oracle不能进入fit、prepared STATIC训练时也无history。
- online函数只收current image；修改/删除/延迟GT或未来图像不影响当前状态与输出。
- batch1、当前统计BN、无目标running scaler、缓存生命周期、跨臂RNG隔离、有限数失败。
- 全新输出与历史输出分离；合法发布链接只映射固定版本普通文件；路径越界/写源/错误binding拒绝。
- 所有真实运行入口disabled；没有授权不能读取资产、启动GPU、子进程或训练；AB/auto-run拒绝。
- 至少各组随机完整ResUNet两次online visit、一个有梯度的程序化source micro-episode，实测模块/梯度到达与冻结backbone不变。资源不足如实报告，不用toy冒充完整模型。
- 目标模型计数A/B/C各2F0backward0Adam，C0为1F；旧C8F1B1Adam；source/VJP/小矩阵调用分别计数。

不要求重新执行全部166项历史重模型回归；共享旧代码没改就只做必要窄回归。如两端环境可用，在最终固定SHA对各组跑同一套CPU验收，保留首失败和最终原始日志。缺服务器环境不伪造；只报告已完成环境。

## 交付结构
- 每组src/tests，源训练器/校准器/在线host，默认disabled配置。
- IMPLEMENTATION_SHA、科学原件及摘要、生产fingerprint、source binding模板及null原因。
- 三组METHOD_CONFORMANCE：逐公式/维度/损失/状态/计算对应表，精确复现与新迁移的界限。
- TEST_COVERAGE、完整日志、独立reference误差、首次失败、完整网络实测。
- metadata的24/9/至多12矩阵与source训练预算；不生成fake receipt。
- REVIEW_INDEX可逐组浏览；窄patch；去身份DELIVERY。真实路径/ID/RGB/mask/checkpoint/token不得公开。

最终各组只能报告R7_A_IMPLEMENTATION_READY_FOR_REVIEW等，或具体INCOMPLETE；三组及共享均通过才可报告：
R7_IMPLEMENTATION_READY_FOR_REVIEW
external_review=NOT_RUN
real_source_training_started=false
real_target_execution_started=false
GPU_queries_or_initializations=0
SOURCE_PREP=NOT_RUN
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
procedural_CPU_model_calls=<actual measured>

不得自行签发review PASS，不以论文录用证明新方法效果，不执行下一阶段。交付后停止，等待本对话外部审阅。
