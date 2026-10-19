# R84：第二强视图来源对照的完整结果

状态：**v2 COMPLETE，性能门槛NOT_PASSED**。6条新连续轨迹、1152次正式更新/2304 PRE、K预测全部完成；v2另有8次fresh资格更新。此前v1工程资格失败2次更新已保留，未进入正式实验。执行科学源码`e89a31cdd7d9fcea680b55dae76c282d4efb6a68`，冻结协议及修复边界见[PROTOCOL.md](PROTOCOL.md)。

**第二strong取另一个既有增强样本，VERTEX总体仅比原方案提高+0.010326pp，95%内容bootstrap区间[+0.007127,+0.013548]，低于预定+0.1pp效应门槛。** D1与域/顺序伤害要求通过，但不能据此宣布候选成功。总体OC只有2/6锚点正、macro 3/6锚点正，正的内容区间不能替代跨seed稳健性。未继续扫描增强配对、强度或q混合比例。

## 唯一改动与配对边界

原VERTEX第一strong不变；第二从同strong的水平flip改为NEXT[seed]既有内容绑定strong的水平flip，逆flip对齐后分别计算完整原损失，各0.5，一次backward与一次Adam。NEXT循环既定三个seed，不重抽，偶然相同也保留。strong sampler原有亮度/对比度/gamma/噪声/模糊，不加几何或新分布；两个strong均走原normalize_image_to_0_1。PAIRED_STYLE是标识，不代表独立统计样本。

原六弱当前q、完整当前W(.1,2)含供体边界/方差/归一化、原BCE+0.1 orientationLSO保持；19136 BN affine/current-stat无running-stat、非BN/head冻结、原连续Adam1e-4 betas .9/.999 eps1e-8。192既有内容、2既定顺序、3seed20260907/17011/29009，原R79 SeedSequence([seed,ordinal,79])不变。无新数据、模块、teacher、EMA、q混合、输出平均或FULL。

只新增VERTEX候选6轨迹，不重跑NATIVE/BARY。原VERTEX和NATIVE的12旧端点/4608 PREK预测从R83继承评分只读复用，非本轮独立重复。逐内容配对H=PRE−P0、C=K−PRE、T=H+C；旧零更新P0不是真值教师。所有差值单位为Dice百分点pp。

## 总体K与各域变化

|arm|cell|macro_Dice_percent|OD|OC|imageweighted_Dice_percent|
|---|---|---|---|---|---|
|VERTEX|ORIGINAL|+76.820033|+84.991714|+68.648351|+77.128472|
|VERTEX|PAIRED_STYLE|+76.830359|+85.002658|+68.658060|+77.139078|

|domain|channel|mean_delta_pp|median_delta_pp|positive_contents|negative_contents|
|---|---|---|---|---|---|
|D0|OD|+0.006142|+0.004822|14|8|
|D0|OC|+0.011476|+0.007032|13|9|
|D1|OD|+0.021103|+0.016065|47|10|
|D1|OC|+0.016354|+0.010925|35|22|
|D2|OD|+0.004778|+0.005791|39|18|
|D2|OC|+0.003157|+0.006147|33|24|
|D3|OD|+0.011752|+0.011728|35|21|
|D3|OC|+0.007849|+0.007050|38|18|

八个域×通道均值都略正，但内容覆盖并非全部正；D1 OD 47/57内容正，OC仅35/57。整体图像加权从77.128472增至77.139078，+0.010606pp。不是只改了某一域，但效应量很小，不满足原性能门槛。

## 六个seed×order锚点：正区间的解释范围

|seed|order|macro_delta_pp|OD_delta_pp|OC_delta_pp|
|---|---|---|---|---|
|20260907|0|+0.138692|+0.061571|+0.215812|
|20260907|1|+0.067425|+0.090810|+0.044041|
|17011|0|-0.013832|+0.054347|-0.082011|
|17011|1|+0.020978|+0.052493|-0.010537|
|29009|0|-0.095357|-0.092725|-0.097989|
|29009|1|-0.055948|-0.100833|-0.011062|

总体OC两顺序均值+0.011938/+0.007480pp，但20260907两个锚点正，另外四个负。macro为3/6正。按内容bootstrap的区间只表示固定这组耦合锚点的内容不确定性，不能解释为种子总体置信。由于NEXT循环共享三个既有样式流，各锚点更不独立。汇总的正均值不足以支撑稳定性能升级。

## 历史与当前一步贡献

|contrast|domain|channel|mean_delta_pp|
|---|---|---|---|
|VERTEX/HISTORY/PAIRED_STYLE_MINUS_ORIGINAL|D0|OD|+0.008566|
|VERTEX/HISTORY/PAIRED_STYLE_MINUS_ORIGINAL|D1|OD|+0.021262|
|VERTEX/HISTORY/PAIRED_STYLE_MINUS_ORIGINAL|D1|OC|+0.018357|
|VERTEX/HISTORY/PAIRED_STYLE_MINUS_ORIGINAL|D2|OD|+0.005553|
|VERTEX/HISTORY/PAIRED_STYLE_MINUS_ORIGINAL|D3|OD|+0.011844|
|VERTEX/CURRENT/PAIRED_STYLE_MINUS_ORIGINAL|D0|OD|-0.002424|
|VERTEX/CURRENT/PAIRED_STYLE_MINUS_ORIGINAL|D1|OD|-0.000159|
|VERTEX/CURRENT/PAIRED_STYLE_MINUS_ORIGINAL|D1|OC|-0.002003|
|VERTEX/CURRENT/PAIRED_STYLE_MINUS_ORIGINAL|D2|OD|-0.000775|
|VERTEX/CURRENT/PAIRED_STYLE_MINUS_ORIGINAL|D3|OD|-0.000092|
|VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL|D0|OD|+0.006142|
|VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL|D1|OD|+0.021103|
|VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL|D1|OC|+0.016354|
|VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL|D2|OD|+0.004778|
|VERTEX/TOTAL/PAIRED_STYLE_MINUS_ORIGINAL|D3|OD|+0.011752|

本轮识别的是第二style安排改变后的连续轨迹总效果。PRE历史项与当前C分别列出，不能把最终增益都说成当前图像纠错，也没有固定BN/Adam/q再单独测梯度variance。两个不同style可能改变梯度覆盖是与实现相容的假设；本轮没有证明方差降低。R79的同H当前替换较小也不能推出历史一定无影响。

相对NATIVE原方案的伤害：

|domain|channel|mean_delta_pp|
|---|---|---|
|D0|OD|-0.232994|
|D0|OC|+0.011859|
|D1|OD|+3.170355|
|D1|OC|+4.539060|
|D2|OD|+0.041230|
|D2|OC|+0.248430|
|D3|OD|+0.264356|
|D3|OC|-0.108845|

各域/顺序伤害门槛均通过，说明本候选未触发预设安全线；这与未满足+0.1pp性能门槛是两件事。没有放宽阈值、选checkpoint或删去负seed。

## FP/FN、前景与实际位置

|domain|channel|metric|mean_delta_pixel_pp|
|---|---|---|---|
|D0|OD|FN|-0.002139|
|D0|OD|FOREGROUND|+0.002089|
|D0|OD|FP|-0.000049|
|D1|OC|FN|-0.000885|
|D1|OC|FOREGROUND|+0.000154|
|D1|OC|FP|-0.000731|
|D1|OD|FN|-0.003736|
|D1|OD|FOREGROUND|+0.001908|
|D1|OD|FP|-0.001827|
|D2|OD|FN|-0.001331|
|D2|OD|FOREGROUND|+0.001480|
|D2|OD|FP|+0.000149|
|D3|OD|FN|-0.000275|
|D3|OD|FOREGROUND|-0.001508|
|D3|OD|FP|-0.001782|

误差分母为整图像像素，非FPR/FNR，不能代数重建平均Dice或命名某种空间形变。

|domain|channel|earlier_delta_pp|later_delta_pp|later_minus_earlier_pp|
|---|---|---|---|---|
|D0|OD|+0.005983|+0.011149|+0.005166|
|D1|OD|+0.016203|+0.026320|+0.010117|
|D1|OC|+0.012628|+0.024085|+0.011458|
|D2|OD|-0.002373|+0.013479|+0.015851|
|D3|OD|+0.012643|+0.011045|-0.001598|

较早/较晚位置同时改变前序内容和BN/Adam历史，不是独立时间因果。全部seed/order、最差单元、内容median和正负覆盖都保留。

|domain|channel|seed|order|delta_pp|
|---|---|---|---|---|
|D0|OD|20260907|0|-0.321058|
|D1|OC|29009|0|-0.715799|
|D1|OD|29009|0|-0.685042|
|D2|OD|29009|1|-0.193056|
|D3|OD|17011|0|-0.267577|

全量15个对比、720个域×通道×seed×order单元，其中241个负值全部公开，无协议内缺失。2000次按域contentbootstrap六anchors耦合、无多重比较校正；已暴露开发集非患者独立/FULL，依据旧结果选择新候选有开发适应偏差。

## 工程失败、修复与fresh资格

v1在collapsed首arrival与原host核心state/output精确比较失败，未启动正式阶段。根因：第二原strong样本遗漏了native sampler后既有normalize_image_to_0_1；不属于性能负结果。2次物理更新已完成、19F，GPU进程15.106729秒、CPU准备5.300599秒保留，原现场和日志私有保存，匿名[FAILURE_AUDIT.json](attempt1-failure/FAILURE_AUDIT.json)公开。

v2在独立新root/entry/空间预留下，候选与独立公式oracle均复用原normalize helper，不加新模块；collapsed额外要求第二input与第一实际strong exact。完整8/8资格重新PASS：collapsed对原R79 VERTEX核心state/output exact；真实NEXT候选对独立原公式output/gradient/BN/Adam/buffer/RNG核心exact；PRE完整恢复、q/W当前来源、BN冻结、连续Adam、一次backward/Adam和物理F计数通过。76F资格计账，不能用旧失败资格作PASS，也不能将资格通过当作性能证据。

## 退休、覆盖、全部成本与空间

两个正式worker各576次attempt=complete、3任务COMPLETE、5760F；正式1152更新/11520F，v2加资格1160更新/11596F；跨attempt加失败，共1162物理更新/11615F。2304新预测和12旧端点覆盖正确。三个GPU子进程及CPU scorer实际exit0，全部注册PID消失且GPU无这些PID；scorer开始在全部GPU退休之后，在线labels/source读取均0。控制器COMPLETE且PID消失，退出码未独立记录。详见[TERMINAL_AUDIT.json](TERMINAL_AUDIT.json)。

|计费项目|GPU进程秒|CPU进程秒|
|---|---:|---:|
|v1失败|15.106729|5.300599|
|v2资格|25.151868|0|
|v2 formal0|605.933900|0|
|v2 formal1|615.765771|0|
|v2准备/评分|0|19.321143|
|跨attempt总计|1261.958269|24.621742|

v2正式墙钟631.235625秒，于2026-10-20T01:47:46.580316+08:00完成。worker body是process子集不重复。CPU准备及评分包含在表内，零模型匿名聚合另列[ANALYSIS_RECEIPT.json](ANALYSIS_RECEIPT.json)。按用户授权没有时间/GPUcap，不以旧18000秒或3h停止。

R84v2 NAS终态一次解析统计256324KiB=262475776bytes；失败v1 103760KiB仍保留。加R82/R83，已观察NAS四目录合计1486296KiB=1521967104bytes；报告/收据只有小量追加，保留各2.2GB原预留。整个窗口NAS8.8GB+本地1GB+余量1GB=10.8GB<20GB。整个共享Git目录保守观察35932KiB，已观察本地新增源码/报告/archives/private helpers仅数MB，1GB本地预留仍有充足余量；不复制旧数据或模型。

## 当前性能取舍

R82硬固定OD q使VERTEX macro下降0.335251pp，R83半混合下降0.158690pp；两者都伤D1。R84仅+0.010326pp且种子不一致，不够预设实质改善门槛。保留原VERTEX作为当前主要参考；不把这次小正值包装成已验证升级。窗口内继续复核既有结果、代码和候选依据，不因一次交付机械关闭监测；没有合理的新干预依据时做零模型调用分析，不空烧GPU或虚报任务仍运行。
