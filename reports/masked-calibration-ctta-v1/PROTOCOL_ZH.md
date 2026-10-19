# MARS 掩码率条件化损失：冻结小规模对照 V1

本协议在新一轮真实图像/标签评分前冻结。来源为 [MARS](https://openreview.net/forum?id=RhOcb9yeO4) 的 Lrank 模块、K=2 消融，不是其完整默认 K=8 分类系统或官方分割复现。WCW22页与MARS19页（含参考文献、理论、实现和全部结果表）已读完提取文本，长输出截断处另读补齐。研究理由见 ROUTE_REVIEW_ZH.md。

## 唯一机制与迁移约定

唯一方法变量是掩码率条件化伪目标；不添教师、原型、源数据、RDC、ENGRAM、边界损失、源KL或新训练。只更新原 checkpoint 的82个 BN 仿射张量/19136个坐标，BN running statistics、卷积和分割头冻结。

同现有8个已暴露开发内容，四条件，五块[0,1,2,3,0]、正反两顺序，每流40到达。患者/预训练成员未知，图像内容隔离仅指既有用户批准的开发范围，不声称患者或临床泛化。不得扩数据/seed/流长/模型/权限。

每到达优化一次：Adam lr1e-4、beta(.9,.999)、eps1e-8，无权重衰减。最终实际FP32仿射L2=.01±5e-7，与原Adam提议方向cos≥.999999；零提议/失败停止。一次完整梯度由原图熵与两个掩码视图的梯度累加后共同更新，三个backward不等于三个optimizer步骤。所有匹配损失臂使用相同参数、步幅与一步预算，独立的原生CW1/CW2保持一/二步定义及原BN语义。不同于V6两步，不作跨轮步数因果比较。

OD/OC各自Bernoulli，不对两头做softmax。未增强当前预测按logit≥0取各自伪标签。Ltta为两头所有像素Bernoulli熵（自然对数）。Lrank只在各头熵≤该头当前图平均熵的像素上平均；这将原文样本过滤迁移为像素过滤，是需验证的新假设，不是原文已证明的可靠性。

模型输入已经处于[0,1]；对4×4块均匀无放回掩码，直接将模型输入选中块置0，不再改变裁剪或归一化。两视图各rho~Uniform(.2,.4)，块数floor(rho*总块数)，伪目标使用抽样rho，记录实际比例。种子沿用既有SEED和case/condition固定派生，所有对照同内容/条件共享掩码；返回块重复同一掩码，非新内容/随机seed独立性。原图及掩码都只来自当前图，不读未来图、历史预测/评分或标签。

MARS：q=(1−rho)yhat+rho(1−yhat)。FIXED_RHO：完全相同掩码/过滤/计算，q=(1−.3)yhat+.3(1−yhat)，检验是否仅普通软标签有效。MASK_HARD：完全相同掩码/过滤，q=yhat，检验是否仅掩码一致性有效。总损失 Ltta+.2*mean_k BCE(logits_masked,q_k)。不实现完整原文 r(x) 失败检测分数，不声称其全部FD收益；核未增强输出的校准作为辅助诊断。

## 科学矩阵、重复性与信息隔离

7个core工作：(MARS RESET/FRESH)、MASK_HARD FRESH、FIXED_RHO FRESH、ENT_NORM RESET/RETAIN/FRESH。FRESH保留仿射但每图清Adam，RESET都清，RETAIN都留。另真实native CW1/CW2各RESET/RETAIN；SOURCE_EVAL、CURRENT_BN、旧D/R/平移R静态参考全保留。静态预测缓存只用于读出，不进入适应。共22条轨迹、880科学arrival（560core+320native）、2160科学读出、27绝对聚合行。规范匹配科学更新560次。

同STRICT进程额外重演MARS RESET正序完整40到达：80对PRE/POST logits与末model.state_dict逐元素完全一致；40额外更新计费，不计科学矩阵。unsupported算子或任何差异停止，不warn_only/回退，不恢复旧V3资格，不把局部资格写成所有状态/RNG的重复性证明。

worker禁scorer及既有评分目录/V6整个root；只8张登记RGB和既有checkpoint/旧冻结头白名单。GPU全部退役后独立CPU进程读8个标签并核完整coverage。全精度logit阈值生成bit预测；另存uint8概率用于近似Brier/ECE（最大概率量化误差1/510），不驱动更新/主Dice，也不上传逐例概率。校准按内容宏平均、分别OD/OC及全图/源前景膨胀3px ROI汇报，避免用海量背景像素伪造独立样本数。

## 冻结主比较和推进门槛

唯一主比较 MARS_FRESH−FIXED_RHO_FRESH 的全流PRE Dice：均值≥+.5pp，按8内容绑定条件/顺序/通道的95%描述性配对bootstrap下界>0。不是独立患者统计推断。

进一步推进还需：PRE相对MASK_HARD和ENT_FRESH的区间下界均>0；首块/全流PRE对SOURCE下界>0、最差单元≥−1pp、返回PRE对源均值≥−.2pp；MARS_FRESH POST宏Dice高于五个静态参考（包括旧D/R/平移R）、ENT_FRESH、MARS_RESET、真实CW1/CW2的RESET与RETAIN；无新增空预测/嵌套违规，ASSD/HD95/vCDR与覆盖不劣于源，所有实际步幅匹配且无zero。任一失败整体advance=false。Brier/ECE、每块/通道/顺序、foreground与boundary corrected/spoiled、函数变化、排序损失、实际范数及过滤覆盖全报，均不替代主Dice。无标签调阈值/系数/步幅，不删负单元。

## 预算和停止

新GPU worker cap1200秒，独立CPU评分900秒，输出2GiB，控制器无重试；每次唤醒至多一新轮/全聊天至多一活跃GPU。先CPU/配置回归、NAS实际挂载/容量/小读写、GPU free≥8192MiB、中性父子入口及nvidia-smi准入。

首MARS RESET正序前两到达后：投影=已用时+(320+40−2)*实测MARS均时*1.35+240（plain预留）+320（native预留）+60收尾，必须≤1180秒，否则停止不评分。不改系数/砍控制/延长预算后重试。2GiB规划：2160份q8约1.13GB、bit与gate约.22GB，其他文件保留充足余量；实际输出超限停止。源参数/BN缓冲改变、非法访问、非finite、范数/方向/覆盖/重复性失败亦停止。工程停止不称算法负结果。

历史本聊天累计GPU1800.2773697376秒、已计CPU760.6522974672498秒，旧未计时CPU准备错误和只读诊断披露保持；新准备和实际进程继续累计。R75/R76为另一次已交付历史（641.002056 GPU秒、39.669574 CPU秒另事后计时1.174357秒及未计导入），只作已知选择证据，不伪称计入本聊天既有ledger，也不删除/改写。

实际完成或停止均写中文结果与执行审计、保留失败/成本；仅匿名聚合与自有代码/配置/无路径receipt经代理推送公开GitHub，核远程SHA及匿名HTTP200。不得上传图像/标签/权重/概率、逐例预测评分、身份、私有路径或第三方全文。
