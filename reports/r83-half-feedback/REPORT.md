# R83：OD半反馈折中的性能与机制结果

状态：**COMPLETE，性能门槛NOT_PASSED**。3456次正式更新、36次fresh资格更新全部完成；18条新连续轨迹，无失败重试。科学执行源`7f7d5ff0d9a3f75efd2e00fd40bdc01f7e551af2`，冻结方案见[PROTOCOL.md](PROTOCOL.md)。

**半反馈相对R82硬拆分恢复了VERTEX macro +0.176561pp，但相对原方案仍下降−0.158690pp，95%内容bootstrap区间[−0.213473,−0.105177]。D1 OC仍下降−0.717303pp，超过预设−0.2pp容许线。** 各域/顺序对NATIVE原方案的伤害两项通过，不能替代macro和D1门槛。三种方法相对各自原方案的macro都下降。本轮不支持把OD反馈混合当作性能优化；不再扫描OD混合比例。

## 冻结条件与复用

只改变OD伪目标来源：qOD=(初态六弱OD+本格当前六弱OD)/2，qOC=本格当前六弱OC；W完整取本格当前六弱，其方差/供体边界/归一化保持原公式。数值实现先逐weak OD概率相加乘0.5再六weak平均；独立公式oracle同次序。BCE和0.1 orientation LSO使用相同half q及current W。没有域路由、在线真值或checkpoint选择。

192既有开发内容、2既定顺序、3种子20260907/17011/29009；19136 BN affine/current-stat、无running-stat、非BN/head冻结；原连续Adam1e-4 betas .9/.999 eps1e-8。NATIVE原完整CWLSO、VERTEX逐视图完整loss均值、BARY对齐logits均值完整loss，原strong+确定flip分开batch1，内容绑定增强与R79–R82一致。无新数据、模块、teacher、EMA、输出平均或FULL。

新增3456物理更新/6912 PRE与K预测；旧R82五格90轨迹/34560预测只读复用。原R81四格indices0..3，R82 HARD_SPLIT index4，新HALF index5。旧端点不是本轮独立重复验证。H=PRE−P0，C=K−PRE，T=H+C，P0为旧零更新评分参考而非真值教师。变化单位为Dice百分点pp。

## 最终K结果

|arm|cell|macro_Dice_percent|OD|OC|imageweighted_Dice_percent|
|---|---|---|---|---|---|
|NATIVE|ORIGINAL|+75.838678|+84.191921|+67.485434|+75.941705|
|NATIVE|HARD_SPLIT|+75.728186|+84.075952|+67.380420|+75.848029|
|NATIVE|HALF|+75.782522|+84.129025|+67.436019|+75.895037|
|VERTEX|ORIGINAL|+76.820033|+84.991714|+68.648351|+77.128472|
|VERTEX|HARD_SPLIT|+76.484782|+84.702057|+68.267507|+76.757364|
|VERTEX|HALF|+76.661343|+84.849068|+68.473618|+76.954963|
|BARY|ORIGINAL|+75.961377|+83.846440|+68.076314|+76.124668|
|BARY|HARD_SPLIT|+75.777074|+83.749495|+67.804654|+75.904245|
|BARY|HALF|+75.868658|+83.808859|+67.928457|+76.010458|

VERTEX关键总体配对对比：

|contrast|macro_delta_pp|OD_delta_pp|OC_delta_pp|
|---|---|---|---|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|-0.158690|-0.142646|-0.174733|
|VERTEX/TOTAL/HALF_MINUS_Q0_W1|+0.298220|+0.175292|+0.421148|
|VERTEX/TOTAL/HALF_MINUS_HARD_SPLIT|+0.176561|+0.147011|+0.206111|

HALF−硬拆分的macro区间[+0.115675,+0.243368]，HALF−全初态q/currentW为[+0.186999,+0.423520]；均为对较弱对照的恢复，不证明超过原方案。HALF对原方案的全域OC −0.174733，区间[−0.243800,−0.107311]，两顺序分别−0.126150/−0.223317，六锚点均负。

## D1与其他域：恢复不等于保住收益

|domain|channel|mean_delta_pp|median_delta_pp|positive_contents|negative_contents|
|---|---|---|---|---|---|
|D0|OD|-0.100553|-0.107025|0|22|
|D0|OC|-0.072399|-0.048120|3|19|
|D1|OD|-0.570593|-0.287146|1|56|
|D1|OC|-0.717303|-0.196696|5|52|
|D2|OD|-0.081806|-0.109759|11|46|
|D2|OC|-0.045264|-0.077043|22|35|
|D3|OD|+0.182369|+0.179801|46|10|
|D3|OC|+0.136032|+0.143394|54|2|

D1 OD有56/57内容下降，OC有52/57下降；D0 OD 22/22下降。D3 OD 46/56、OC 54/56内容改善。半反馈保留了部分D3改善，但没有同时保住D1。不能用D3改善抵消预设D1限制，也不能用全局均值抹去具体域的负结果。

其他方法及相对NATIVE原方案的实际伤害：

|contrast|domain|channel|mean_delta_pp|
|---|---|---|---|
|NATIVE/TOTAL/HALF_MINUS_ORIGINAL|D0|OD|-0.145465|
|NATIVE/TOTAL/HALF_MINUS_ORIGINAL|D1|OD|-0.184790|
|NATIVE/TOTAL/HALF_MINUS_ORIGINAL|D1|OC|-0.180556|
|NATIVE/TOTAL/HALF_MINUS_ORIGINAL|D2|OD|-0.093633|
|NATIVE/TOTAL/HALF_MINUS_ORIGINAL|D3|OD|+0.172303|
|VERTEX_HALF-NATIVE_ORIGINAL/TOTAL|D0|OD|-0.339690|
|VERTEX_HALF-NATIVE_ORIGINAL/TOTAL|D1|OD|+2.578659|
|VERTEX_HALF-NATIVE_ORIGINAL/TOTAL|D1|OC|+3.805404|
|VERTEX_HALF-NATIVE_ORIGINAL/TOTAL|D2|OD|-0.045353|
|VERTEX_HALF-NATIVE_ORIGINAL/TOTAL|D3|OD|+0.434973|
|BARY/TOTAL/HALF_MINUS_ORIGINAL|D0|OD|+0.126007|
|BARY/TOTAL/HALF_MINUS_ORIGINAL|D1|OD|-0.471739|
|BARY/TOTAL/HALF_MINUS_ORIGINAL|D1|OC|-0.600363|
|BARY/TOTAL/HALF_MINUS_ORIGINAL|D2|OD|+0.051280|
|BARY/TOTAL/HALF_MINUS_ORIGINAL|D3|OD|+0.144130|
|BARY_HALF-NATIVE_ORIGINAL/TOTAL|D0|OD|-0.692203|
|BARY_HALF-NATIVE_ORIGINAL/TOTAL|D1|OD|+0.101603|
|BARY_HALF-NATIVE_ORIGINAL/TOTAL|D1|OC|+1.394443|
|BARY_HALF-NATIVE_ORIGINAL/TOTAL|D2|OD|-0.732986|
|BARY_HALF-NATIVE_ORIGINAL/TOTAL|D3|OD|-0.208662|

## 贡献分解与解释强度

|contrast|domain|channel|mean_delta_pp|
|---|---|---|---|
|VERTEX/HISTORY/HALF_MINUS_ORIGINAL|D0|OD|-0.099538|
|VERTEX/HISTORY/HALF_MINUS_ORIGINAL|D1|OD|-0.539835|
|VERTEX/HISTORY/HALF_MINUS_ORIGINAL|D1|OC|-0.689502|
|VERTEX/HISTORY/HALF_MINUS_ORIGINAL|D2|OD|-0.080082|
|VERTEX/HISTORY/HALF_MINUS_ORIGINAL|D3|OD|+0.182626|
|VERTEX/CURRENT/HALF_MINUS_ORIGINAL|D0|OD|-0.001016|
|VERTEX/CURRENT/HALF_MINUS_ORIGINAL|D1|OD|-0.030758|
|VERTEX/CURRENT/HALF_MINUS_ORIGINAL|D1|OC|-0.027800|
|VERTEX/CURRENT/HALF_MINUS_ORIGINAL|D2|OD|-0.001724|
|VERTEX/CURRENT/HALF_MINUS_ORIGINAL|D3|OD|-0.000258|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|D0|OD|-0.100553|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|D1|OD|-0.570593|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|D1|OC|-0.717303|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|D2|OD|-0.081806|
|VERTEX/TOTAL/HALF_MINUS_ORIGINAL|D3|OD|+0.182369|

有证据的结论：对OD q来源做固定半混合，会改变整条连续学习轨迹，D1损失比硬固定弱，但最终仍未超过原方案；多数变化已存在于PRE历史项。规则上OC仍取当前六弱，实际OC供体依赖本格BN/Adam轨迹，所以OC预测与原方案并不相同。

与代码/结果相容的解释：OD梯度经共享BN改变后续OC供体及Adam历史，产生跨通道反馈耦合。未证实的部分：没有分离BN层、Adam一二阶矩、q在BCE/LSO中的作用，不能将损失归于某个单独共享层、动量或边界。半混合在概率层线性，不意味着Dice或轨迹响应线性；这次较接近中间值不能证明普遍插值定律。

## FP/FN、前景和位置

|domain|channel|metric|mean_delta_pixel_pp|
|---|---|---|---|
|D0|OD|FN|+0.025264|
|D0|OD|FOREGROUND|-0.025304|
|D0|OD|FP|-0.000040|
|D1|OC|FN|+0.036795|
|D1|OC|FOREGROUND|-0.016936|
|D1|OC|FP|+0.019859|
|D1|OD|FN|+0.097167|
|D1|OD|FOREGROUND|-0.059895|
|D1|OD|FP|+0.037271|
|D2|OD|FN|+0.022851|
|D2|OD|FOREGROUND|-0.031257|
|D2|OD|FP|-0.008406|
|D3|OD|FN|+0.001757|
|D3|OD|FOREGROUND|-0.034253|
|D3|OD|FP|-0.032495|

误差以整图像像素为分母，非FPR/FNR，不能代数重构均值Dice或据此命名具体空间形变。

|domain|channel|earlier_delta_pp|later_delta_pp|later_minus_earlier_pp|
|---|---|---|---|---|
|D0|OD|-0.006326|-0.192749|-0.186423|
|D1|OD|-0.408308|-0.671362|-0.263053|
|D1|OC|-0.560270|-0.818735|-0.258465|
|D2|OD|-0.004376|-0.155787|-0.151411|
|D3|OD|+0.057439|+0.307814|+0.250375|

较晚位置同时改变前序内容、BN和Adam历史，不是单独时间干预。最差seed/order及六锚点描述全部保留：

|domain|channel|seed|order|delta_pp|
|---|---|---|---|---|
|D0|OD|29009|0|-0.358924|
|D1|OC|29009|1|-0.900961|
|D1|OD|29009|1|-0.780117|
|D2|OD|29009|1|-0.231779|
|D3|OD|20260907|1|+0.024584|

公开81个对比、3888个域×通道×seed×order单元，其中1426个负值完整保留，无协议内缺失。[FOCAL_MECHANISM.csv](FOCAL_MECHANISM.csv)列焦点H/C/T，[ERROR_DOMAIN_SUMMARY.csv](ERROR_DOMAIN_SUMMARY.csv)列聚合误差，其余全量表不筛阳性。

2000次按域内容bootstrap六anchors耦合，未多重比较校正；已暴露开发集，非患者独立/FULL验证。R82后选择该候选进一步增加开发适应偏差；旧端点只读复用不增加独立性。连续Adam虽保留同公式和超参数，其数值状态随干预内生分化，不是同入站状态纯梯度效应。

## 资格、真实退休与全部资源

36/36 fresh资格PASS：对角原host完整state/output exact；HALF独立公式output/gradient/BN/Adam/buffer/RNG核心exact；OD_HALF_OC1/Wcurrent来源、首步对角、PRE/初态observer完整恢复、BN19136冻结、连续Adamage、一次backward/Adam、真实F计数均通过。资格438F计账；资格是工程证据而非性能证据。

两个正式worker各1728次attempt=complete/9条COMPLETE/27072F，正式合计54144F，加资格为3492新GPU物理更新/54582F。注册GPU与CPU子进程全部exit0且PID消失，GPU无本轮PID；CPU评分开始在全部GPU退休之后。在线label/source读取均0。控制器COMPLETE且PID消失，但未独立记录控制器退出码。详细见[TERMINAL_AUDIT.json](TERMINAL_AUDIT.json)。

累计GPU进程4865.690438秒=资格65.175126+formal0 2382.825230+formal1 2417.690081；CPU准备4.166599+评分25.031752=29.198350秒，主体是进程子集不双算。正式墙钟2443.144340秒，于2026-10-20T01:14:34.560781+08:00完成。用户已取消GPU/时间上限，没有失败或额外资格重跑。

本轮NAS一次解析路径统计562652KiB（576155648字节），在2.2GB预留内；报告/收据仅增加小文件，保留整个预留。窗口NAS两轮共4.4GB、本地1GB、全局余量1GB，共6.4GB，低于20,000,000,000字节。R82旧费用/空间未重置或隐藏。

继承收据FORMAL_ADMISSION.reused_endpoint_updates=6912仅为旧两个锚点计数，完整复用应按FROZEN/评分和本次终态核对的90条/17280旧更新/34560预测解释；新3456更新不受影响。该元数据局限在启动时已披露，不改写执行源。

## 后续取舍

本轮不再扫描OD混合权重。R78已有证据显示输出平均缺少稳定总体收益且可能伤害D3 OD，不将其当作无依据的新性能修复。继续在授权窗口内核对既有增强随机性、连续历史和梯度构成证据；只有存在可解释且不重复已失败路径的小步候选才另冻结协议运行。没有足够依据时做零模型调用的机制/代码分析，保持监测而不空烧GPU，也不声称仍有运行中的实验。所有负结果保留。
