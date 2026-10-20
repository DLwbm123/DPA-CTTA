# R85/R86：损失聚合位置实验终态报告

**实验与CPU评分完成；VB、BV均NOT_PASSED，无候选晋级，保留原VV。** 全部8个GPU worker和CPU scorer均exit0且PID退休。不是工程失败，不重试性能结果、不追改门槛。

本轮维修迁移使用同硬件fresh五方法对照，原192个已暴露开发内容、两order、三seed、BN affine/连续Adam/q/wholeW/strong和冻结损失全部不变。VV=B_V+.1S_V，VB=B_V+.1S_B，BV=B_B+.1S_V，BB=B_B+.1S_B；V逐视图损失均值，B对齐logits均值后计算损失。NATIVE为原单strong方案。原协议及迁移修订保持有效。

## 主要结果

所有Dice为百分数，差值为百分点；主指标为四域等权、OD/OC等权、六anchors等权宏平均。

|方案|K macro Dice|相对VV|
|---|---:|---:|
|VV|76.822123|+0.000000|
|VB|76.783067|-0.039056|
|BV|76.489540|-0.332582|
|BB|75.972087|-0.850036|
|NATIVE|75.843778|-0.978345|

NATIVE绝对分数由同硬件VV分数减配对VV−NATIVE差值还原；原始配对结果见CONTRASTS.csv。VV−NATIVE为+0.978345pp。SOURCE_ONLY（stored source BNstats）68.733077，P0（current-stat）74.749237，二者不可混称。上述均为开发集，不能称未暴露确认。

|候选|相对VV macro|Bonferroni双侧97.5%内容CI|D1 OC|正anchor|域伤害门槛|结论|
|---|---:|---|---:|---:|---|---|
|VB|-0.039056|[-0.077777, -0.003927]|-0.164447|0/6|通过|NOT_PASSED|
|BV|-0.332582|[-0.510963, -0.168442]|-1.018066|0/6|未通过|NOT_PASSED|

门槛为macro≥+0.1pp且97.5%CI下界>0、D1 OC≥−0.2pp、相对NATIVE域×通道≥−0.5pp及域×通道×order跨seed≥−2pp、两order均正/至少5个anchor正/每seed两order均值正。两候选的六个anchor全部为负；VB通过伤害限制但未过收益与稳健性门槛，BV另违反D1 OC及域伤害限制。2000次按域content bootstrap，六anchors耦合；不是患者独立或seed总体CI。完整gate及每anchor数值见PERFORMANCE_GATE.json。

## 历史状态、当前更新与机制边界

H=PRE−P0，C=K−PRE，T=H+C。VV的H=+2.026773pp、C=+0.046112pp；VB−VV的H=−0.038815pp、C=−0.000240pp；BV−VV的H=−0.328453pp、C=−0.004130pp。差距主要呈现在已分化的入站状态上，不能据此断言单一BN/Adam因果。全轨迹factorial T为−0.478398pp，是算法交互效果，不是同H纯梯度因果。

VB相对VV的D1域macro为−0.158572pp，对四域macro贡献−0.039643pp；其余三域平均+0.000783pp，不能将这个事后子集当主收益。BV的D1为−1.152785pp，去掉D1仍−0.059181pp。所有域及leave-one-domain-out见GAIN_CONCENTRATION.csv。

48个冻结同H探针的梯度四格最大残差4.9173832e−7，满足预设代数检查。相对VV的平均梯度余弦：VB 0.984329、BV 0.741387、BB 0.494517；实际Adam delta余弦分别0.997561、0.987825、0.977733，说明梯度差异不等于同幅度参数更新差异。D_B平均0.024294、D_S平均0.087162；D_S无BCE Jensen非负保证。同状态虚拟K−PRE的OD/OC等权探针均值分别VV +0.143540pp、VB +0.138623pp、BV +0.114163pp、BB +0.106268pp，仅描述预设探针，非全流主指标或晋级依据。原始浮点余弦未裁剪，VV自余弦均值1.0000015是数值舍入，不能解释为超出1的真实相似度。

完整正负域×通道×seed×order见ALL_CELLS.csv与NEGATIVE_CELLS.csv；内容median/正负覆盖见CONTENT_DISTRIBUTIONS.csv；整图分母FP/FN/foreground见DOMAIN_CHANNEL_ERRORS.csv与ERROR_COMPONENT_CELLS.csv；按两个顺序真实到达位置配对的earlier/later见POSITION_DECOMPOSITION.csv，不把order号当早晚。MECHANISM.csv含30轨迹摘要；SAME_STATE_PROBES.csv及SAME_STATE_SCORES.csv保留全部匿名探针。

## 终态与资源

物理更新5924/5924、前向58324、训练反向5924、额外诊断反向400；30连续轨迹、48探针、11520 PRE/K预测、192探针预测、SOURCE_ONLY/P0各192。8个GPU状态均GPU_COMPLETE，8个GPU进程与CPU评分均exit0、PID消失。CPU scorer在核全体GPU真实退休后才读标签；online标签/source-image读取均0。controller有COMPLETE状态收据且PID消失，独立数值退出码未记录；watchdog以all_owned_children_retired正常退出。详见COMPLETION.json及TERMINAL_AUDIT.json。

本attempt退休GPU进程墙钟上界12859.413038秒，CPU准备14.563460秒、评分进程180.026107秒，已计量CPU合194.589567秒；评分body170.130256秒是子集，不双计。进程时间不是能耗，退出观察可含约5秒轮询余量。本轮首次系统解释器导入失败CPU时长未计量，保留UNKNOWN。原服务器最后可证退休GPU1132.053968秒及CPU准备16.241517秒，后续尾账仍UNKNOWN；不将上述迁移费用称跨attempt完整总成本。

2026-10-20 16:27:23北京时间完成，自原T0起12202.569061秒，原24小时截止2026-10-21 13:04:01.100432未重置。终态实际分配空间du为1,999,883,264bytes（补交报告前的一次观测），低于4GiB cap及4.4GB reservation。报告同步增加少量文本，保留128MiB恢复余量。保守预留总envelope17.4GB<20GB，各服务器新增严格<50GB。仅批准GPU0/1，数据/模型/环境只读复用，未复制大型资源。

## 研究决定与限制

保留VV，关闭本轮VB/BV晋级；不继续聚合系数扫描，不重复已完成reset或旧负候选。该二因子试验没有支持改为B聚合的证据，也不证明任何数据集上的普适无效。接下来承接匿名结果与历史机制证据的分析，只有出现区别于已否决因素、可证伪且符合资源边界的新假设，才另冻结独立协议与预算。当前不为维持实验活动空跑GPU，持续自主控制与唯一半小时心跳保留。

注册1951内容全部历史暴露，patient/video linkage及source患者重叠UNKNOWN；R87在当前cohort NOT_AVAILABLE，R88条件未满足，换seed不能冒充确认。迁移不声称跨硬件对旧R79逐预测exact；fresh对原代码/独立公式核心state/output exact已严格PASS。原机维修后仅补取尾部收据，不复活旧任务。

只发布源码、协议、匿名汇总/探针统计和失败/资源证据；图像、标签、私有预测、身份映射、权重、私有路径和原始日志不公开。科学版本6eed982，scorer计数元数据修正a7586c8未改公式、评分值或门槛。
