# A05 原图 BN 累积与逐输入重置诊断

A04主和BN_ONLY均低于SOURCE，触发仅此分支，服务主线持续域状态的安全性判断。非baseline优化，不改伪标签、损失、优化器、支持、BN层/动量或阈值。

原8开发内容、4固定外观、正反两序、原GPU1/checkpoint。所有参数冻结，无backward或更新。PERSIST完整重演A04 BN_ONLY：PRE eval、原图train/no_grad写所有BN buffers、POST eval。RESET每输入先完整恢复SOURCE state，然后执行同样三次forward。SOURCE和B固定只读。PERSIST PRE/POST每输入逐位对齐旧A04，RESET PRE逐位SOURCE，每序首输入两POST一致。独立worker逐位重放全部float/mask/loss；参数逐输入检查不变且无梯度。

每worker64×7=448 forward，两worker896，0 backward，0更新。生成无GT；退出后独立评分原完整Dice/ASSD/HD95/vCDR并附signed vCDR、OC面积相对GT、每头FP/FN，空头和嵌套。内容bootstrap与原[2,1,1,1]权重；首次和后续外观预先列出，8内容不是患者独立验证。

主要诊断PERSIST_POST−RESET_POST和各自−SOURCE，不从多臂挑新方法。若持续显著更差而RESET安全，支持累积损害；若两者均低，单次原图BN写入也不安全。仅诊断，不以任一均值正数替代既有正式全任务门。SOURCE+B一并披露。

1小时原钟，3530s阶段、3540s独立guard，1GiB输出硬限，1.25GiB预约，预计700MB。首输入预测完整双worker及评分若超过1小时则停止，不删控制。工程闭环失败只算工程失败。不得复活已退休身份或触碰R106。
