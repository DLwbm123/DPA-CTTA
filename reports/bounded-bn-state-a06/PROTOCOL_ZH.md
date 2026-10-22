# A06：逐输入重建BN、持续保留稀疏学习状态

A05持续原图BN比逐输入重置OC低3.628pp，单次写入相对SOURCE均值+0.421但CI跨零且未胜B；据此只移除跨输入BN统计累积，不把单次BN视为成功方法。用途是主线修复，不优化独立baseline。

唯一修改：A04每输入的原图train/no_grad统计观测先完整恢复SOURCE named_buffers，然后在更新前的当前参数下写一次；恢复观测前BN执行原增强、原损失和稀疏Adam，再提交这一次原图统计。参数、Adam、FFT保持跨输入持续。PRE仍读取上次提交状态，其旧冻结历史收益门保留。无伪标签、阈值、学习率、支持或BN层/动量扫描，不接细化。

原8内容、4外观、两顺序、原GPU1/源checkpoint。SOURCE/B、完整RESET_ALL、只留FFT的MEMORY_ONLY、BN_ONLY（源参数和每输入一次源BN写入）、逐张量同数RANDOM，以及缓存A02 PARAM_ONLY/OLD_DOMAIN、A04 OLD_REPAIR。随机控制也施同样BN重建，严禁削弱控制。

主切回源BN后的logits逐位对齐A02 N10，仅称输出一致，不宣称全参数/Adam状态一致。BN_ONLY POST逐位对齐A05 RESET_POST；每序首输入主与两reset精确一致。SOURCE闭环、完整float/mask/loss独立重放，GT仅worker退出后独立评分。每worker1216F256BW256更新，两worker2432F512BW512更新。

保留A04全部正式门：主对B OC均值至少0.3pp且CI下界>0，主胜SOURCE及所有同信息控制（新增旧修复对照也必须胜），OD对B下界>=−0.1pp，最差头>=−1pp，无新增空头/嵌套，完整表面指标，OC ASSD/HD95比值CI上界<=1.05，vCDR误差差CI上界<=0.005，语义随机差异比例>=0.1，两序各自通过OC门，以及首次内容PRE相对SOURCE/BN_PRE/RANDOM_PRE CI下界>0。该历史PRE门失败不等于所有CTTA无效，不事后撤门。

提前列出首次/后续外观切片和MEMORY−RESET解释，不作新增筛选门；追加面积/FPFN/signed-vCDR。全部结果披露，开发8内容bootstrap不是独立患者证据。若仅改善旧修复仍低SOURCE/B，不晋级；若BN安全而主低于BN，应定位稀疏学习目标，不能继续调整BN救结果。

1小时原钟/3530s stage/3540s独立guard；2GiB硬输出、2.25GiB预约，预计1.25GiB。首输入完整双worker及评分预测超过1小时即停，不删控制。闭环失败属工程失败。R106不触碰，已退休身份不查询。
