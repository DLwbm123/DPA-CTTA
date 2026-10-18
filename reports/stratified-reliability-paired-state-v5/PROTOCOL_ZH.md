# V5：严格确定性、同进程状态对照（冻结新轮）

2026-10-19。V2显示持续参数/Adam累积没有首次到达收益；V3跨轮不变控制parity失败，NOT_EVALUABLE保持不变；V4独立工程诊断在DEFAULT八组出现差异、STRICT八组logits和末模型逐元素完全一致，已公开提交a99ca12a9532d6a6712d82704f3a8a7e03aacabd。V4只认证短流同进程，不认证完整流或跨进程，也没有评价Dice。

本轮是新的科学设计，不恢复V3、不用旧跨轮预测/分数当对照。所有方法在同一个worker、严格deterministic algorithms(True,warn_only=False)、cudnn.deterministic=True、禁benchmark/TF32、CUBLAS :4096:8下重算。

唯一科学干预仍是跨图像Adam记忆：STRAT和精确配额RANDOM各有RESET（仿射/Adam均重置）、RETAIN（均保留）、FRESH（仿射保留、每arrival清空Adam矩/step）。同图两步共享Adam，.01实际步幅/5e−7容差/.999999方向不变。先在完全相同内核设置下重新获得所有对照，不能把V2 DEFAULT与本轮STRICT差值当Adam效应。

强控制包含RG、NOANCHOR、ENT_NORM的RESET/RETAIN，原生CW1/CW2的RESET/RETAIN，以及新算SOURCE_EVAL、CURRENT_BN、旧已训练D/R/平移R静态参考。原生CW仍用current-image BN，不伪称与源BN同语义。此轮聚焦状态因果，未重复V2的ALL/GLOBAL/原始ENT；保留其公开负证据，不用额外臂替代同配额随机与同BN源模型主控制。无新源训练、RDC/ENGRAM关闭。

矩阵：12个core方法/状态组合×两顺序×40arrival=960，加CW320=1280科学arrival、32轨迹；PRE/POST2560+5静态参考400=2960读出。37绝对聚合行，每行80图像条件×2通道=160单元。8内容、5块[C0,C1,C2,C3,C0返回]、正逆顺序不变；首块仅本流首次暴露。1600科学范数匹配更新和320原始RG更新。末模型buffers/source锚和冻结卷积必须不变；FRESH/RESET optimizer_age均2，RETAIN随arrival递增。

新增完整40arrival重复性准入：首个STRAT/RESET/正序科学轨迹保留80个CPU logits，仅内存；在同一worker用新host重演同一40arrival，80对logits和末model.state_dict必须逐元素一致。额外40arrival/80优化步计入本轮资源，不计科学矩阵，不保存逐例logits。失败即NOT_EVALUABLE，不放宽。该资格不扩写为其它长流/跨进程认证。头两arrival时间投影使用(1000−2)*实测均时*1.35+native320秒+60秒+已用时，须≤cap−20；这1000包含重复准入40。最终GPU硬上限仍控制，不根据结果改参数。

唯一主比较STRAT_FRESH−STRAT_RETAIN:PRE，至少+.5pp、八内容组描述性bootstrap95%下界>0。另冻RANDOM对应状态效应、STRAT−RANDOM状态效应差中差、首次/全流/返回PRE对SOURCE_EVAL、FRESH PRE对RANDOM FRESH、FRESH POST对所有强控制及完全RESET、返回遗忘、真实结构和前景损害；全部报告。推进还需首次/全流PRE对源下界>0、全流PRE最差≥−1pp、返回PRE对源均值≥−.2pp、同配额随机PRE下界>0、POST均值超过源/RG/CW1/CW2、全部规范匹配无零提案、无新增空/嵌套、ASSD/HD95/vCDR及覆盖不劣于源。

bootstrap2000次，按8内容组绑定两顺序和重复条件，仅描述开发数据。前景纠正/损害以SOURCE_EVAL为参照，不沿用V2的CURRENT_BN参照；门控审计仍对应第二次更新前预测，在全图与真实3px边界报告。新独立scorer在worker正常退役后才读8标签；worker不读旧预测/评分，旧静态头只读冻结权重/epoch元数据。新轮历史parity字段为0且明确不执行跨设置历史预测比较，不改V3原2320parity要求。

预算新GPU≤1200秒、CPU评分≤900秒、输出≤2GiB；V1至V4累计GPU1060.8567349911秒、已计CPU537.2086229324秒继续保留，加本轮CPU准备和进程，不清零。先CPU回归/比较器与分组检查、实际NAS读写/空间/GPU显存≥8192MiB及中性入口准入，再单次启动。任一预算/数值/访问/ownership/确定性资格失败都保留停止、不重试。

8个反复开发内容、患者与预训练成员未知，源模型强基线仍必须尊重；无临床/独立患者泛化或TMI创新结论。即使Adam状态诊断阳性，清空优化器也不是新机制；若首次到达/源模型门槛仍失败，不继续把本路线包装成CTTA收益，不盲调步幅/损失或扩数据seed长流。
