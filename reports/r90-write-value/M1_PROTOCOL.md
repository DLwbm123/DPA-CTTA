# R90 M1：单次历史写入的后续效应（评分前冻结）

前置M0_GATE=PASS。本阶段不是补旧探针日志，也不是性能门控方法；只检验新的动作价值与无标签辨识假设。继承PROTOCOL.md的192开发内容、2order、3seed、VV全部公式/原增强/currentstat BN/连续Adam和共享T0/deadline、5000updates/50000F、1.5GiB输出cap及GPU0/1。M0与资格费用不清零。M2仍条件未准入。

## 成对设计与状态语义

每条原VV流arrival8,32,56,80,104,128,152,176设置8个探针，共48；窗口为随后16张，无失败后换长度。新采样设计不复用旧48探针的结果。主VV6流共1152更新；USE-ONLY分支各16更新共768。WRITE后续输出直接复用本轮主流，不另跑WRITE窗口。资格拟7个真实GPU更新/70F，均计量；正式1920updates/19200F。任何工程失败另保留attempt并共享原始预算，不自动科学重试。

当前t先正常原VV更新得K_t；WRITE保留完整学习状态，USE-ONLY输出同一个已计算K_t然后撤销本次学习写入。恢复BN参数、梯度缓存、Adam一二阶矩/age、相关优化器学习状态，native.steps随Adam年龄回退。数据到达visits保持post值，独立物理计数不回退；外部RNG与native_rng取post-current值，避免撤销随机数消耗。未来增强仍按既有content ordinal绑定，后续q/W由各自分支状态生成。恢复主线post snapshot后继续主线，USE窗口不污染主线。不是每图reset、跳过当前输出或纯参数/动量效应。

每探针只保存真实入站BN参数/Adam state/steps与现有反向产生的g，不存图像/label/新概率。当前K只算一次，不把其复制视为独立输出。主线全部PRE/K对同硬件R85迁移VV packed masks逐次exact，不能放宽为分数接近。这个parity仅为工程参照，不新增独立验证。任何parity/状态/冻结/访问/计数失败NOT_EVALUABLE，非性能负结果。

## 唯一无标签信号

在真实入站m,v,t和原lr/betas/eps、weight_decay=0、无AMSGrad下，float64 CPU计算零新梯度Adam理论位移delta0：m'=beta1*m，v'=beta2*v，以t+1做bias correction，delta0=−lr*mhat'/(sqrt(vhat')+eps)。当前完整VV梯度g从本次已有backward读取，无新增autograd。未初始化状态delta0为0。

s=<g,delta0>/(||g||||delta0||)。精确零范数记NA/不触发；正内积超过预定float64符号保护bound=64*eps64*(n+1)*(sum(abs(g*delta0))+||g||||delta0||)才触发，否则近零区域不触发。此常数只作保守数值符号保护，不按Dice调节。理论delta0不包含float32参数相减量化；不能与保存的actual post−pre量化位移混称。资格用同一真实state的float64 CPU Adam零梯度clone验证公式，atol1e-12/rtol1e-9，不初始化或修改第二个GPU模型来补此检查。零梯度下非零历史位移与None-grad跳过不同，不把零梯度当撤销。

## 资格/profile先于正式

2个完整原Host，同共同初态；用已有固定前三个内容和同seed：双host第1步exact；第2步及真实入站state/output exact；一host恢复USE状态，另hostWRITE继续第3步；随后恢复主线post第2步并重做第3步，须与独立WRITE输出与完整核心state exact。检查恢复后的参数/m/v/age等于pre、visits/RNG等于post、下一步实际Adam年龄差1、PRE无副作用、冻结19136/非BN、一次训练BW/Adam、每交易PRE+K=10F、实际物理计数7/70以及CPU delta0公式。资格失败不放宽条件。

记录冷启动和全部profile费用；正式准入按资格除首个冷交易外最慢完整交易秒×1920×1.5作为串行GPU上界，保守并行时间取其一半+1800秒评分+600秒余量，必须小于共享剩余时间。预计累计更新/前向须在5000/50000内。worker各固定960updates/9600F，charge在不可逆外部计数且超限前拒绝，状态/失败计数持久化。GPU启动每卡≥10GiB free、固定UUID，allocator8GiB。正式两worker及qualifier真正exit0/GPU_COMPLETE/PID退休后才CPU读标签。GPU审计拒绝M0衍生评分、scorer/labels/source数据；只有当前image、checkpoint、注册本流原VV binary参照、代码/环境及新worker根允许访问。

独立watchdog在共享deadline−90秒TERM、5秒后KILL身份匹配自有组；worker−180秒合作停止。单阶段CPU scorer最多1800秒且受共享deadline，原T0不重置。实际空间按campaign root全部分配计，不删除M0/旧尝试输出。

## 终态指标与晋级（新标签评分前固定）

D为每图OD/OC Dice等权平均；R_t,16为后16张WRITE K减USE K的均值，正表示写入有益。另报PRE版、双头/域和原样/同M0后处理版本，沿用M0结构指标及空/NA定义；主判据固定原样K，不以post结果重选窗口/信号。窗口内按到达图像等权，不冒称四域macro。

每anchor8探针：G_rule=mean(−zR)，pi=mean(z)，G_random=pi*mean(−R)，G_identify=G_rule−G_random。各anchor独立计算同频随机期望后六anchor等权汇总，不池化混淆。两个G的整体、两order均值、每seed两order均值均>0且各≥5/6正anchors才通过。并要求总trigger≥12，每anchor1–7个trigger，触发涉及≥6个不同现有内容且≥3个匿名域；触发过少/饱和/集中均停止。报告全部48结果与未触发项，不将probe/窗口作为独立患者，不做虚假的独立窗口CI，不把窗口收益相加冒充完整策略收益。

只有上述动作价值、辨识增量和触发覆盖全部PASS才准许冻结M2唯一候选及非信息性少写控制。失败不扫sign阈值、频率、窗口或改信号；M1通过也不是全流有效证明。M2操作及随机控制频率来自无标签触发统计冻结，必要独立计数/同硬件parity与原+0.1pp/校正CI/D1OC/域伤害/稳健性门槛及新增D1OD保护、优于少写控制要求不变。该路线始终是暴露开发集的诊断，非R87确认。
