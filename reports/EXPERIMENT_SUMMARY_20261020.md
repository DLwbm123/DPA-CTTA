# CTTA实验总结：R79–R86及全BN组诊断

截至2026-10-20。本报告汇总已完成、已公开的实验与离线诊断，不代表新增模型运行。

**结论：保留原VV（逐视图完整损失平均）。VB、BV均未通过冻结晋级门槛；全层诊断没有提供足以选择“VV有害层”并开展干预的因果依据。当前没有新GPU作业。**

## 1. 研究问题与冻结条件

研究目标是在strict model-only条件下，理解并改善连续分割适配：为什么逐视图训练比对齐logits平均后训练更有效，能否保留D1收益同时减少其他域损害。

本轮迁移使用192个既有暴露开发内容、2个固定顺序、3个固定seed（20260907/17011/29009），保持19136个BN affine参数、current-stat且无running-stat、连续Adam、原六弱视图q与完整W、原strong及同strong水平flip、原BCE与0.1方向LSO。非BN参数与head冻结。没有线上标签、源图像、额外模型或新数据访问。

V表示逐视图计算损失后平均，B表示先平均对齐logits再计算损失：

|方案|定义|作用|
|---|---|---|
|VV|B_V + 0.1 S_V|原VERTEX参考|
|VB|B_V + 0.1 S_B|预设主候选|
|BV|B_B + 0.1 S_V|反向对照|
|BB|B_B + 0.1 S_B|原BARY|
|NATIVE|原单strong CW/LSO|原方案对照|

维修迁移后五方案均在同一硬件fresh运行；不跨硬件拼接旧分数，不声称与旧R79逐预测bit-exact，也不将迁移当独立验证。[冻结协议](r85-r86-loss-aggregation/PROTOCOL.md)、[迁移修订](r85-r86-loss-aggregation/migration/PROTOCOL_AMENDMENT.md)。

## 2. R85/R86主要结果

K输出，四域等权、OD/OC等权、六个seed×order锚点等权；Dice单位为%，差异单位为百分点（pp）。

|方案|macro Dice|相对VV|
|---|---:|---:|
|VV|76.822123|0|
|VB|76.783067|−0.039056|
|BV|76.489540|−0.332582|
|BB|75.972087|−0.850036|
|NATIVE|75.843778|−0.978345|

NATIVE绝对分数由VV得分减同硬件配对VV−NATIVE差值还原；配对原始结果在CONTRASTS.csv。VV优于NATIVE约+0.978345pp，是当前开发参考，而不是独立确认结论。

|候选−VV|macro差|Bonferroni双侧97.5%内容CI|D1 OC差|正锚点|结果|
|---|---:|---|---:|---:|---|
|VB−VV|−0.039056|[−0.077777, −0.003927]|−0.164447|0/6|NOT_PASSED|
|BV−VV|−0.332582|[−0.510963, −0.168442]|−1.018066|0/6|NOT_PASSED|

晋级要求同时满足macro≥+0.1pp、校正CI下界>0、D1 OC≥−0.2pp、相对NATIVE的域/通道及order伤害限制，以及两order、各seed和至少5/6锚点的方向一致性。VB未过收益和稳健性；BV另外违反D1保护与域伤害门槛。无候选晋级，不通过重试或修改阈值挽救负结果。

VB退化主要集中D1：该域macro−0.158572pp，去掉D1后其余三域仅+0.000783pp，不能将这个事后子集当主收益。BV去掉D1仍为−0.059181pp。[完整结果、全部正负单元与统计表](r85-r86-loss-aggregation/migration/REPORT.md)、[正式门槛](r85-r86-loss-aggregation/migration/PERFORMANCE_GATE.json)。

## 3. 与R79–R84证据如何衔接

|实验|已经支持的结论|未支持的推论|
|---|---|---|
|[R79](r79-mechanism-control/REPORT.md)|主要方法差异在当前更新前的PRE中已经存在；当前单次strong替换效应较小|不能把H全部归因于某一BN层、Adam或历史增强无关|
|[R80](r80-history-factorial/REPORT.md)|Adam与q/W反馈有条件交互；两反馈关闭后仍有基础积累差异|不能把清空Adam等同于只消融动量，也不能称普适纯BN机制|
|[R81](r81-qw-factorial/REPORT.md)|q反馈放大D1收益，W存在条件交互，其他域并非同一模式|不能把q整体致害或W整体无关作为统一解释|
|[R82](r82-channel-feedback/REPORT.md)|硬拆分OD反馈使VV macro−0.335251pp、D1 OC−1.516356pp|未支持只改OD目标就能保护OC|
|[R83](r83-half-feedback/REPORT.md)|半反馈相对原VV仍macro−0.158690pp、D1 OC−0.717303pp|不继续扫描混合比例|
|[R84](r84-paired-style/REPORT.md)|换第二strong仅macro+0.010326pp，未达到+0.1pp门槛|不支持继续NEXT配对/强度扫描或声称已估计梯度方差|
|R85/R86|仅改变BCE或LSO聚合位置都未优于VV|不支持继续全局B聚合路线|

H=PRE−P0，C=K−PRE，T=H+C。VB−VV的H/C分别−0.038815/−0.000240pp，BV−VV分别−0.328453/−0.004130pp。差异主要体现在入站状态，但约99%的端点算术占比不是历史因果解释比例，也不表示当前梯度不重要。无BN runningstats，不能称runningstats污染。

## 4. 全BN组CPU诊断

仅分析已保存48个同H探针的向量，覆盖41个BN模块、10个架构阶段、gamma/beta及全BN，共156组。29952详细行、7488分层汇总行全部保留，未按结果仅挑top层。6份终态映射的82个参数tensor名称、形状和顺序一致，总19136维；分组覆盖与数值自检通过。

layer4占37.46%的参数，承载52.48%的VB−VV位移差能量；三个seed均值52.23%–52.80%，两order为51.81%–53.16%，两个机械位置为50.07%/54.90%。这种改动敏感结构较稳定，但D1与D3份额分别44.31%/42.39%，没有建立“只减少D3OC损伤、保留D1收益”的层分离。

全BN通道冲突稀少不等于每层无冲突：VV的41模块×48探针中，有116/1968个局部OD/OC梯度余弦为负；VB/BV/BB分别142/222/297。局部通道梯度与实际VV位移的负内积，仅表示代理损失的一阶下降方向，不能当Dice改善证据。

实际位移四格残差R_delta=delta_BB−delta_BV−delta_VB+delta_VV，全BN L2均值3.8734744e−5，相对四cell位移范数之和的均值比例0.001762777。它同时包含Adam当前分母非线性与float32 post−pre相减量化误差；转float64汇总不能恢复保存精度。缺少逐probe入站参数及Adam m/v/t，可靠量化下界与各通道Adam响应不可识别，不使用terminal状态冒充，不为补状态复跑GPU。

**位移差定位的是“聚合改动敏感组”，不是“VV有害层”。** 所有探针都在VV历史上，差异大的组可能承载VV的有益机制。48个探针不是48独立患者或重复，稀疏位移也不能相加冒充完整192步漂移。当前不指定层开展冻结、限幅或梯度投影。[完整诊断报告](post-r86-layer-diagnostic-v1/REPORT.md)、[协议与全组数据](post-r86-layer-diagnostic-v1/PROTOCOL.md)。

## 5. 完成、费用及缺失

|项目|迁移主实验|后继CPU层诊断|
|---|---|---|
|覆盖|5924更新、58324前向、5924训练反向、400诊断反向；30轨迹、48探针|48探针×156组，无新模型/GPU调用|
|终态|8个GPU worker及CPU scorer均exit0、PID退休|timeout wrapper exit0，CPU进程及timeout父进程退休|
|已测时间|GPU进程12859.413038秒；CPU准备+评分194.589567秒|脚本墙钟38.366102秒；独立父计费CPU进程时间未记录|
|实际分配空间，一次终态观测|1,999,883,264bytes|35,752,960bytes|
|预算|原始历时12202.569062秒，24h内；4GiB root cap内|600秒/64MiB预算内；writer guard按逻辑字节，非allocated硬quota验证|

进程墙钟不是能耗；body是进程时间子集，不重复相加。补交报告前空间观测不代表后续文本逐字节不变。主实验原时钟没有因迁移重置。

原服务器维修后的尾部状态、覆盖与费用仍UNKNOWN；首次系统解释器导入失败的短CPU时长未计量。迁移controller有COMPLETE状态且PID消失，但独立数值退出码未记录。不能将当前费用称跨attempt完整总账。[主实验终态审计](r85-r86-loss-aggregation/migration/TERMINAL_AUDIT.json)、[CPU诊断审计](post-r86-layer-diagnostic-v1/TERMINAL_AUDIT.json)。

## 6. 科学边界与后续决定

SOURCE_ONLY（stored source BNstats）为68.733077%，P0（current-stat）74.749237%，VV K为76.822123%。SOURCE_ONLY→P0约+6.01616pp不能代替持续适配论证；VV比P0+2.072885pp也不构成未暴露泛化证明，需与已有episodic/reset控制一起解释。

192内容是已暴露开发集，注册1951内容全部曾历史暴露；patient/video linkage与source患者重叠UNKNOWN。2000次按域content bootstrap耦合六anchors，不是患者独立或seed总体CI。R87当前没有合格未暴露cohort，R88前置条件未满足；换seed不能冒充独立确认。

保留VV；关闭VB/BV全局聚合晋级及无依据的按位移差抑制层路线。不继续q比例、增强/NEXT或层组合扫描，不重复已完成reset。只有出现新的可证伪证据，才另冻最小因果协议、匹配控制、明确的参数/优化器操作、D1 OD/OC保护、原晋级门槛及硬预算。持续自主控制与唯一r52保留，当前无新GPU作业；缺乏数据资格或方向依据时如实记录，不虚报运行。

本次公开范围为源码、冻结协议、匿名汇总/全组统计、负结果与审计；图像、标签、私有预测、身份映射、权重、原始私有日志和机器路径不公开。
