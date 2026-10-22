# A10：持续参数、逐输入清空AdamW状态

主线假设：A09零当前梯度仍有平均0.019455参数步长，完整步长0.024579；ZERO−HOLD杯扩张与FULL接近，FULL−ZERO OC仅−0.001463pp且CI跨零。既不能把loss数值大小当梯度主导，也不能将ZERO全归为一阶动量。本轮检验保留整个优化器历史是否阻碍当前主线，不做baseline优化。

唯一改动：A08每输入开始将训练臂AdamW.state.clear()，包括一阶/二阶矩与step。模型参数、FFT记忆持续，原图BN仍按A06逐输入从SOURCE统计观测；保留A08 hard教师标签、结构遮挡/增强、损失及freq系数.1、稀疏支持、lr1e-4、betas/.eps与weight_decay=.01。每次更新后所有optimizer state.step必须等于1。此干预联合改变矩历史和偏差校正年龄，不能单独解释成动量清除；weight decay保持相同配置。

原8登记内容、4外观、两序、原GPU1/源checkpoint。REPAIR及逐张量同数RANDOM均清optimizer；RESET_ALL/MEMORY_ONLY原来已经每输入重置，输出与loss必须逐位等于A08同臂，保证未削弱控制。BN_ONLY保持A05RESET_POST逐位闭环；SOURCE闭环、首主与两个reset一致。SOURCE/B、缓存A08 ORIGINAL完整历史、A02 PARAM_ONLY/OLD_DOMAIN及新参数SOURCE_BN_POST全部作为对照。全部9动态float/mask/loss独立重放，GT隔离。

主OC相对B>=.3pp且CI下界>0；胜SOURCE及全部同信息控制；OD对B下界>=−.1pp，最差头>=−1pp，无新增空头/嵌套，完整表面，OCASSD/HD95比值CI上界<=1.05，vCDR误差差CI上界<=.005，semantic-random输出变化比例>=.1，双顺序各自过OC门，首次内容PRE相对SOURCE/BN_PRE/RANDOM_PRE下界>0。所有旧门保留，不以改善ORIGINAL或减少步长/扩张代替成功。首次/后续外观及MEMORY−RESET、面积/FP/FN/signed-vCDR解释预列。完整参数继续积累，不将局部分叉拼接成新成绩。

双worker2432F512BW512updates不变。1h原钟/3530stage/3540guard，2GiB硬输出2.25GiB预约，预计1.25GiB，首输入预计超预算停止，不删对照。只一个配置，不扫描betas/lr/decay或选择性报告臂。若仍失败先解释失败机制，不自动开展Pro三态监督；若通过正式门再冻结验证持续价值。开发8内容非患者独立，CI描述性未校正长期多重比较。

所有缓存只读、生成禁GT，评分在workers退休后。e417及更早退休身份不查询，R106不碰。只公开自有源码/协议/聚合及执行记录，私有数据/权重/预测不公开。
