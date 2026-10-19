# G2：GOTTA启发的整图记忆多样性对照（运行前冻结）

用户授权：固定回放单位、距离和基础目标，CPU通过后直接小规模强基线实验；保现有数据模型预算，RDC/ENGRAM关，跑完分析push。此为机制迁移，不是原论文复现；参数和统计均在读取本轮标签前冻结，不扫描容量、ε、目标或步幅。

## 一个主要干预与科学依据

只改变过去真实输入的保留策略。原GOTTA的Norm/RoTTA同容量对照有正证据，但TENT等仍劣源；本轮验证回放分布是否改变纯熵的持续损害，不假定必有增益。它不使用H1类后验、不借翻转教师、不同时引入EMA/增强/源KL。源预测只构造选择描述，不充当监督。原多类按图伪类与原ε/CSTU不移植：盘杯图同时含多区域，不能造图级互斥类。

## 固定回放、距离和目标

- **整图**：保留归一化后的完整512×512 RGB输入，CPU内存容量2个到达条目，无标签、域ID、内容ID或未来数据。重复输入按到达仍可占槽，不声称独立患者/内容；仅条目到达序号用于确定性排序。
- **冻结源描述**：只用当前源模型预测，OD/OC各分前景内部、背景内部及3像素宽方形形态边界（现有7×7 max-pool strata），共6区。各区16个源概率逆经验CDF分位数j/15，取整数rank=max(ceil(jN/15)−1,0)，j=0…15。每区/每头等权，不因背景面积大而获更多权重。任何区空、源logit或描述非有限，硬停，无fallback/临时改权。
- **距离**：96对对应分位数的Bernoulli JSD取平均，除ln2后开根；FP64，范围[0,1]。这里的“对应”是区域分位数，不是跨个体像素对应；没有几何配准或真值边界进入描述。该描述仍会反映源置信度，不能保证识别正确性或病理差异。
- **DIVERSITY**：暖启动直接填2槽；以后对旧2条+当前候选的3个合法二元子集，保留距离最大的那一对。精确距离并列时保留按uid排序字典序最新的一对。固定容量2时这等于最大化最小配对距离，无ε、λ或容量搜索，是GOTTA启发的无阈值class-agnostic coreset变体，并非原CSTU/FPS算法。
- **随机**：标准容量2的reservoir sampling，CPU独立random.Random(20261018)；每轨迹重置，不搜随机seed。
- **FIFO**：最近两个过去条目。
- **时序**：先输出PRE，使用当前图和仅过去记忆更新、输出POST，再接纳当前图。没有“先把当前写入再当历史”的重复权重。每次3个顺序微批：当前+2历史槽。无历史则当前重复3次；仅1条历史则该条占2槽。CURRENT不存历史，始终重复当前3次；因此不是部署速度对比，计算匹配的无回放控制。
- **统一目标**：每微批原图双Bernoulli自然对数熵均值，3份各乘1/3反传累加；不加teacher、增强、一致性、边界损失、源KL或置信过滤。一次Adam更新，lr1e−4、(.9,.999)、eps1e−8。仅82个BN仿射张量/19136坐标，runningstats、conv/head冻结。实际FP32参数L2=.01±5e−7、方向cos≥.999999（沿用已验证数值实现），zero/nonfinite硬停不造方向。参数等幅不保证函数等幅，另报输出RMS/改像素。
- **FRESH状态**：保留仿射及记忆，每图清Adam；同图3微批共享Adam，避免把跨图Adam差当记忆效果。RESET每图恢复源仿射、空Adam、空记忆及memory RNG。没有跨轨迹混存。

## 冻结矩阵及覆盖

核心：DIVERSITY RESET/FRESH；RANDOM_MEMORY FRESH；FIFO FRESH；CURRENT RESET/FRESH，6 jobs×2顺序×40到达=480更新。每次3反传，共1440科学loss backward/480 optimizer events。真实CW1/CW2各RESET/RETAIN共320到达，原生更新语义保留；两种BN语义不同必须披露。五静态SOURCE_EVAL/CURRENT_BN/旧D/R/平移R保留。

总20学习轨迹、800科学arrival、2000 PRE/POST/静态读出、25绝对聚合行。8已暴露内容、4条件，五块[0,1,2,3,0]×两顺序；不扩数据、seed、流长、模型、源训练或权限。source归一化/扰动、图像与标签清单沿既有冻结队列。预加载RGB仅IO优化，Host只接当前图和过去条目，不可读其它未来缓存。

另加DIVERSITY FRESH正序40到达重演，80对完整logits、末model state、完整记忆输入/描述/uid/seen/RNG、末Adam step/moments逐元素完全相同。STRICT warn_only=False，不回退。重演纳入GPU成本，不增科学样本；只认证该流同进程，不认证跨进程/native全部流。两个RESET臂每个读出bit parity必须一致。CPU检查涵盖距离/区域/复制支持、确定性max-min/随机/FIFO、先预测后接纳、微批量/源不变/函数实际改变、RESET与FRESH完整state、配置与OC统计形状。

## 主比较、强对照与损害报告

唯一主：DIVERSITY_FRESH−RANDOM_MEMORY_FRESH **OC PRE**，均值≥+.5pp且描述性95%CI下界>0。2000 bootstrap按8内容重采样，条件与两顺序绑定；预先指定OC不事后换头/POST/顺序。还须OC PRE对FIFO/CURRENT下界>0。

推进合取：首次块、全流PRE对SOURCE全双头均值CI下界>0；逐单元最差≥−1pp；原条件返回PRE全双头均值≥−.2pp；OC PRE真值前景和3px边界净纠错率CI下界均>0；POST全双头均值超过五静态、DIVERSITY RESET、CURRENT FRESH及CW1/CW2 RESET/RETAIN。无新增空/嵌套，ASSD/HD95/vCDR不差源且定义覆盖不低；全部实际步幅匹配、无zero。任一不满足advance=false；不挑有利副臂/只赢弱随机称持续收益。

保留所有臂PRE/POST的OD/OC、五块/两顺序、首块/返回、参数/函数变化、实际记忆量/年龄/描述距离/回放和反向预算。净纠错以SOURCE_EVAL硬预测为参照；GT前景/边界只在GPU退役后的CPU评分使用。C/H及C+H是80逻辑读出的加权总数，不当32唯一内容或独立患者；另报逐图宏净值及CI，分母不可混用。scorer缓存相同metric只是省计算，不删除读出覆盖。

## 预算、准入与停止

GPU硬1200s、独立CPU评分900s、输出2GiB，一次派发无重试。先CPU合成检查（真实RGB/标签/权重/CUDA0）、NAS实际挂载/容量小读写、指定GPU UUID/free≥8192MiB、中性入口父子ps+nvidia准入。首DIVERSITY RESET6次完整到达计时：`已用+(480+40−6)*平均秒*1.35+320native+60 ≤1180`，否则停止保留现场，不削控制/改描述/减槽/延预算重跑。空区/非有限/zero/步幅/访问/重复性不合规均NOT_EVALUABLE，工程停止不是科学负结果。

标签仅GPU worker exit0且inactive/完整生成和重演后独立CPU开放。worker拒旧E2-A全root、scorer和旧评分快照，只允许登记8RGB、现有checkpoint与旧冻结reference heads。完整GPU_COMPLETE2000/800、FULL_REPEATABILITY40/80全state、MECHANISM/ACCESS、score exit0与8标签/25行/覆盖、最终ledger/controller原身份退役后才宣告完成。失败亦分析交付，不删除初失败/成本。

历史累计GPU4530.565845727862s、已计CPU2219.0993401709266s，加本轮实际CPU准备/worker/scorer；旧本地.631、公式body .0010669171/.00009529106318950653与历史CPU基数.0084569158差/未计时阅读编排保持。只公开自有代码、协议、匿名聚合/无路径回执；不发数据、标签、记忆图像/描述、权重、逐例评分、身份路径或第三方全文。结果经代理push并核remoteSHA/匿名HTTP200。8内容反复开发、患者/预训练未知，不作独立患者、临床或TMI创新结论。
