# A09：同前状态拆分目标梯度与优化器历史

用途：主线机制定位。A08删除不确定强制前景后相对A06 OC−0.0123pp且CI跨零，面积/FP也未减少；不能用目标标签变化解释A07的一步扩张。下一检验分割损失、0.1频域特征损失，以及历史Adam/weight decay分别贡献什么。不将loss数值大当成梯度主导，不调损失系数。

原8内容/4外观/两序/GPU1/源checkpoint，完整重演A08 REPAIR唯一轨迹64输入。每序每内容固定k=i%4，16位置，不使用标签选位置。全64 PRE/POST/loss及SOURCE_BN输出逐位A08（9读出缓存）。不要求A02参数轨迹相同。

每固定位置从同一完整输入前模型、Adam、FFT和梯度状态分叉，使用同一输入、教师输出/特征、增强随机键、稀疏支持与原图BN观测：
- FULL：原seg+freq真实更新，并从前状态重演一次，必须逐位输出/loss且全部模型/Adam/梯度/FFT状态等于真实更新后状态。
- HOLD：前参数、不更新，读出与FULL相同的当前原图观测BN；避免把BN观测误归为优化。
- SEG：seg+0*freq；FFT：freq+0*seg，freq保留原系数0.1。零系数仅为让两分支与FULL保持相同的grad-None参与模式，明确断言一致，避免Adam跳过参数混杂。
- ZERO：0*(seg+freq)，当前梯度必须严格为0，但仍执行一次原AdamW及稀疏恢复，保留历史矩/步数与weight decay。

FULL/SEG/FFT/ZERO的前向证据及两个损失分量必须一致，仅改变用于本次backward的组合。每次恢复optimizer使用独立副本，避免CPU step计数别名修改共同前状态。探针后完整恢复真实后模型/Adam/梯度/FFT/模式/RNG，主轨迹只推进一次；分叉不得进入历史。记录每分支原稀疏支持上的梯度L2、全参数实际步长L2及seg/加权freq梯度余弦，不将标量loss比例等同梯度比例。

主要诊断：FULL/SEG/FFT/ZERO分别−HOLD；SEG−FULL、FFT−FULL；FULL/SEG/FFT−ZERO；FULL−SEG−FFT+ZERO仅为Adam及输出非线性的交互，不声称梯度线性分解。ZERO−HOLD包括历史矩与decay，不能分别归因。SEG/FFT保留FULL历史，局部效果不是长期独立训练效果。

所有读出比较SOURCE/B、缓存A08RESET/BN_ONLY，报告OD/OC Dice、ASSD/HD95、vCDR绝对/有符号误差、面积/FP/FN、空头/嵌套。内容和两序等权bootstrap，16位置不同于旧64输入[2,1,1,1]权重总体。诊断不晋级、不得事后撤A08等旧正式失败门。若SEG相对FULL有一致任务/结构改善，下一只检验去除freq的完整主线修复；若ZERO接近FULL而当前梯度作用小，下一定位Adam历史，不扫loss系数。混合/跨零保留不确定，不挑局部最佳当正式结果。

每worker64锚输入各6F，加16位置(四个5F更新分支+1F HOLD)，共720F128BW128更新；双worker1440F256BW256更新。保存5动态分叉及完整锚float/mask/loss、分量/范数/余弦，独立worker完整逐位重放。GT仅两worker退出后独立评分，重复开发内容非患者独立证据，CI未校正长期多重比较。

1h原钟3530stage3540独立guard，1GiB硬输出1.25GiB预约，预计<=768MiB。首次含全部分叉27F/5BW/5更新，预算预测=首输入时间*(1440/27)*1.5+已历时+300s评分余量，>3600即停。不减少分支救预算；状态/输出闭环失败为工程失败。只当前root，不触R106或查询退休身份。自有源码/协议/聚合公开，私有数据/模型/预测不公开。
