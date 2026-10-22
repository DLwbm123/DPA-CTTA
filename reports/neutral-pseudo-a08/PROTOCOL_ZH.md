# A08：去除不确定像素的强制前景标签

用途为主线目标机制修复。A07在固定BN下本步更新增加杯面积/GT约0.010和假阳性24–28像素，但OC收益CI跨零；更新后BN重观测OC仅+0.011pp且CI跨零，不继续BN细化。假设原OR规则将高不确定像素强制置为前景，驱动无任务收益的扩张。A07未直接证明该规则因果性，本实验用于证伪。

唯一训练改动：原pseudo=(teacher_p>=0.5) OR (official_uncertainty>=0.75)改为pseudo=(teacher_p>=0.5)。不改教师、0.5阈值、损失系数、稀疏支持、原增强/FFT、lr/Adam、源BN逐输入观测或顺序。不引入置信度筛选、soft目标或超参扫描。记录每头删除的强制前景数，并断言新标签是原标签子集。

原8登记开发内容、4外观、两序、原GPU1与原checkpoint。主线REPAIR、完整RESET_ALL、只留FFT的MEMORY_ONLY、同数RANDOM均采用新目标；BN_ONLY不更新参数，POST必须逐位A05 RESET_POST。主线每序首输入与两个reset逐位一致。SOURCE/B固定；缓存A02 PARAM_ONLY/OLD_DOMAIN和A06 ORIGINAL对照，另评估同一新参数下SOURCE_BN_POST作为同信息控制（该读出已有forward）。SOURCE闭环、所有9个动态float/mask/loss独立重放、生成禁GT。新目标改变轨迹，不能继续要求或声称新主线SOURCE-BN输出等于A02 N10。原A06输出/损失证据保留。

主OC相对B>=0.3pp且CI下界>0；胜SOURCE与全部控制（reset、memory、BN、random、旧parameter/domain、原A06以及新SOURCE_BN_POST）；OD对B下界>=−0.1pp、最差头>=−1pp、无新增空头/嵌套、全部表面有效、OC ASSD/HD95比值CI上界<=1.05、vCDR误差差CI上界<=0.005、semantic-random输出变化比例>=0.1、两序分别过OC门、首次内容PRE相对SOURCE/BN_PRE/RANDOM_PRE下界>0。旧门不撤。预列首次/后续外观、MEMORY−RESET、面积/FP/FN/signed-vCDR，避免以减少假阳性替代任务成功。

双worker2432F512BW512更新，与A06相同；新增SOURCE_BN_POST保存不新增forward。只一个配置，失败不扫阈值或损失权重。全部输出披露，不按标签后选位置。反复使用8开发内容不是患者独立验证，CI描述性且未校正长期多重比较。

1h原钟/3530stage/3540guard；2GiB硬输出2.25GiB预约，预计1.25GiB，首输入估计超预算停；完整独立重放与GT隔离。仅当前root，退休e415及更早不查询，R106不碰。完成后公开自有源码/协议/聚合报告，不公开私有数据、权重或预测。
