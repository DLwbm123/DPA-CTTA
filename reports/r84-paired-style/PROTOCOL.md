# R84：第二强视图来源的单候选对照

## 动机、授权与唯一改动

用户授权12小时自主性能优化至北京时间2026-10-20 11:33，无GPU/时间cap但全部准备、资格、失败和运行费用照记，新增空间≤20,000,000,000字节。R82/R83的OD q硬固定/半混合均未超过原VERTEX，不继续扫描该系数。R78输出平均无稳定macro收益且伤害D3 OD，不将其当默认修复。R79显示当前strong替换影响较小但非零（VERTEX D1OC平均绝对差0.050681pp，26/342当前一步符号翻转）；其不能排除不同增强安排通过连续状态积累影响结果，也没有证明新候选会提升性能。

只测试一个不同假设：原VERTEX第二视图为“同一strong的水平flip”，现改为“另一个既有strong样本的水平flip”，逆flip对齐后仍分别计算完整原W/LSO损失、各0.5、一次联合backward和一次Adam。第一strong完全不变；第二strong取R79已经用过的NEXT[seed]内容绑定样本，不增加seed或重抽。NEXT循环20260907→17011→29009→20260907；两次样本偶然相同也不重抽。实现标签PAIRED_STYLE不代表统计独立观测，三个既定种子因此耦合。

固定原strong sampler经代码核对只含亮度、对比度、gamma、噪声、模糊，无额外空间变换；第二view只有确定水平flip需逆对齐。不改增强分布/强度，不新增变换、数据或模型模块。候选可能通过覆盖两个style而改变梯度/历史；是否减少噪声或改善性能是未验证假设，不能从当前U−K小推断已成立。

## 保留与最小规模

原六弱当前q、完整当前W(.1,2)含供体边界/方差/归一化，原BCE+0.1 orientationLSO；19136 BN affine/current-stat/no running-stat、非BN/head冻结，连续Adam1e-4 betas .9/.999 eps1e-8。192既有内容、2既定顺序、3既定seed，R79 ordinal和SeedSequence([seed,ordinal,79])不变。无q混合/初态教师、EMA、输出平均、新模块或FULL。

只新增VERTEX候选6条连续轨迹、1152物理更新、2304 PRE/K预测；原VERTEX与NATIVE的12条端点评分（4608预测）从已交付R83继承的ORIGINAL indices中只读复用，不重跑NATIVE/BARY或失败的q候选。新的值数组(192,6,1,2,2,2)，cell0=VERTEX_ORIGINAL，1=VERTEX_PAIRED_STYLE；NATIVE_ORIGINAL另取参考。旧端点不是新重复验证。每arrival原六弱+两strong+post=9F，加PRE为10F；正式11520F。

## fresh资格与资源准入

CPU先检R79内容样式与RNG隔离、计数及评分代数，然后freshGPU8更新：两个arrival将新第二style设为第一style，new/原R79 VERTEX host各一步，共4更新；两个arrival使用真正NEXT样本，new/独立原公式oracle各一步，共4。要求collapsed候选对原核心完整state/output exact；候选对独立公式的output/gradient/model/BN/Adam/buffer/RNG核心state exact；PRE完整状态恢复，当前q/W来源exact，连续Adamage、BN冻结、一步一次backward/Adam、每step9F计数正确。两组只新host读取PRE，资格合76物理F，总1160次GPU物理更新/11596F。不能要求真实不同第二style首步等于旧第一style方案，那是科学干预而非资格条件。

先在同一NAS窗口预留新轮2.2GB；加R82/R83各2.2GB、本地1GB、全局余量1GB，共8.6GB<20GB。每轮output2GiB含64MiBreserve，data/model/env复用，不删除旧结果。启动GPU固定0/1 UUID且≥10GiB空闲、allocator8GiB；中性argv，不动他人进程。projection仅规划，GPU和时间预算null。工程失败可诊断修复另attempt，但保留失败及成本和空间；性能负结果不重跑调阈值。

## 终态与门槛

全部GPU实际exit0/GPU_COMPLETE后才能CPU读labels和旧score。H=PRE−P0、C=K−PRE、T=H+C；报告候选−VERTEXoriginal、候选−NATIVEoriginal、原VERTEX−NATIVE及绝对H/C/T。所有domain/channel、seed/order负单元、最差锚点、FP/FN整图分母/foreground、内容median覆盖、实际earlylate均保留。

成功门槛不放宽：候选macro对VERTEXoriginal≥+0.1pp且95%contentbootstrap下界>0；D1 OC对VERTEXoriginal≥−0.2pp；任domain×channel对NATIVEoriginal均值≥−0.5pp，任domain×channel×order跨seed均值≥−2pp。2000按域contentbootstrap六anchors耦合、无多重校正，暴露开发集非独立患者/FULL；已有结果驱动候选选择有开发适应偏差，不称外部验证。梯度历史、BN、Adam和自反馈均内生演化，不能称同H直接作用或证明梯度方差下降。

terminal核6/6、1152attempt=complete、资格8另计、2304新预测、12旧轨迹覆盖、实际PID/GPU退休和评分屏障、access0、全部成本（body不重复）和新增空间一次du。仅匿名public下载，完整负结果/中文报告/源码proxy push新branch，remoteSHA/匿名REPORT200与local+remote新DELIVERY。半小时健康安静监测，窗口后不启动新阶段，健康已运行任务完成后交付。无后续第二style或seed组合穷举。

## v2工程修正

第一attempt的fresh资格在collapsed比较首arrival失败，正式0更新。原因已定位：原strong sampler输出在native cal_consis_loss内还调用原normalize_image_to_0_1，候选第二视图遗漏该既有步骤；不是性能负结果。v2候选与独立公式均调用同一既有归一化函数，collapsed资格额外要求第二输入与第一实际strong exact。保留失败2次已完成物理更新/19F及所有CPU/GPU费用，独立新root/entry/space reservation，完整8更新资格重做。不是新增归一化模块或变换。门槛不变。旧2.2GB预留不删除，v2另2.2GB，含本地/余量共10.8GB<20GB。
