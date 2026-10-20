# E3：固定RGB亲和邻域logit扩散，独立冻结静态资格

## 证据与假设

已读SRDF的两侧图像对比提供局部外观信息；B1显式窄带移动虽有微弱次要阳性，却没有主增量。P1固定概率平滑有OC+.467pp，但ASSD劣源，P5后续填孔只再加+.033pp、仍未解决边界。P4表明无约束共享网络优化可把局部先验传播成巨大背景误报。源相对Dice证书虽代数有效，却对所有最佳平滑OC提案零接受，不能靠拒绝全部获得收益。

因此这轮检验**当前RGB空间对应是否可为固定局部平滑提供比均匀滤波更有益的权重**，针对跨真实边缘混合的可能代价；不进行BN替换、全网络优化、伪类中心、记忆、坏flip教师或证书阈值调参。不是原SRDF训练系统复现；双边滤波/边缘感知平滑是成熟技术，不包装ICLR/TMI创新。图像纹理/血管/反光可误导权重，原源图像也不是独立真值，不能预设更优。

用户已授权独立改造/融合有据直接小规模强对照；不新增源训练、不扩8内容/模型/seed/权限，RDC/ENGRAM关，BN尺度与Calibration Maps停止。未来时间GPU配额取消，新增20GB硬限/18GB水位/辅助1GB保留。

## 完全固定的输入与运算

沿用既有归一化RGB输入x、原尺寸512×512源logit z，单图batch1，源全网络eval冻结，无参数/buffer/optimizer/backward更新。每图源只前向一次共享全部臂，两头独立Bernoulli，不造softmax类。四个既定条件、既有seed不改。

反射补边，3×3空间核为[1,2,1]外积。计算当前x原生相邻水平/垂直像素的RGB通道均方差，取严格正值的**下中位数s**；若全部差为0，数学定义所有颜色亲和为1（等于均匀核），不是基于标签的fallback。每个邻域偏移权重w=空间核×exp(−RGB均方差/s)。中心权重4保证归一分母正，无epsilon/floor/拟合超参。s仅同一当前图，无跨图估计。

EDGE_LOGIT是原logit的w归一加权平均。只安装共同区域0:H−1,0:W−1，末行末列保留源，与P1概率平滑控制安装范围一致。它在每个已安装位置处于3×3源logit凸包内，但这不是Dice/结构保证，仍可能跨错误边缘混合。

ROLL_MATCH：先把完整9偏移权重场统一沿宽roll W//2+1，保每偏移权重分布及联合归一分母分布，破坏和当前源logit的空间对应；仍用当前位置邻域logit计算，再按每头全图源logit位移L2匹配到EDGE_LOGIT。UNIFORM_MATCH：均匀空间核logit平均的位移同样匹配到EDGE_LOGIT。匹配是正标量、无截断或剪裁，可能越出局部凸包，不能靠打败它们独自声称安全。UNIFORM_LOGIT保原始均匀logit平均。

匹配目标为当前EDGE实际位移norm，不由标签选。函数中常数logit的零目标合法恒等；科学真实输入每头EDGE方向必须非零，否则整轮资格停止，不能造方向。如果控制norm=0但目标>0则停止。FP64实际norm绝对误差≤1e−9×max(1,目标norm最大值)，不把参数L2=.01套给零更新方法。

## 八臂与强基线

SOURCE、SMOOTH（严格复现P1既定概率binomial平滑，logit转换沿用其1e−12饱和夹取）、FILL_SOURCE（源4背景填孔）、TOPO_SMOOTH（平滑后8前景最大组件再4背景填孔）、EDGE_LOGIT、ROLL_MATCH、UNIFORM_MATCH、UNIFORM_LOGIT。

四静态参考SOURCE/SMOOTH/FILL_SOURCE/TOPO_SMOOTH在scorer阶段与P2封存128bit逐位一致；GPUworker拒P2/P5/P4整个root/标签/旧分数，不用参考评分择优。原生CW已多轮完成且劣源，本轮零更新局部运算由源和当前最好静态及匹配剂量机制控制约束，不机械重跑CW优化器。无硬包含/面积修复、无额外模型或训练。

## 唯一主与完成门

唯一主EDGE_LOGIT−SOURCE OC Dice≥+.5pp且8内容绑定2000次seed20261018描述bootstrap下界>0。另OC对SMOOTH/FILL_SOURCE/TOPO_SMOOTH/ROLL_MATCH/UNIFORM_MATCH/UNIFORM_LOGIT各CI下界>0；源双头CI>0/worst≥−1pp/原条件再读≥−.2pp；OC GT前景与3px边界净纠错CI均正，ASSD/HD95/vCDR点值不差源/定义覆盖不降/空包含不增。任一失败advance=false，不选臂/条件/带宽/权重或按当前标签调参。即使只因更弱函数改动赢某控制，也不能替代最好静态和结构合取。

32输入×8=256bit预测、8绝对64区域64条件头35对比。五块[0,1,2,3,0]两逻辑重读共640，不是独立患者/CTTA；原条件再读非持续返回。C/H只32唯一输入，宏率分母按固定内容/块，区域重叠不相加。GT仅GPUworker退休exit0后独立CPUscore读取。

全部32输入重推encoder并重算8输出，256数组及诊断精确重演，同进程不认证跨进程；共64完整源前向。源全state_dict不变/无grad，GPU中性可见性现场检查，QUALIFICATION真实首输入两头方向非零/匹配PASS；MECHANISM/ACCESS/128静态bitparity/SCORE_COMPLETE/worker scorer exit0inactive/controller原身份退休/finalledger完整后交付。失败保部分/成本日志源，工程修复另记录不涂改原失败，不缩控制或扫描参数。

## CPU与资源

9项新CPU检查须PASS：常数logit/常RGB/共同RGB仿射尺度性质/局部凸包/控制实际norm与错位分布/同输入重演及头交换/旧P1概率控制精确等价/非法输入/冻结配置。合成无RGB文件/GT/权重/CUDA，不能当真实性能证据。

写前NAS实际挂载容量小写读、空间reservation2.5GiB计code/archive/cache/log。root2GiB守卫复用只忽略ENOENT的安全计数；用户已取消时间GPU预算，无旧1200/900timeout。启动free≥8192MiB并核GPU UUID/中性父子ps+nvidia，只用一本聊天GPU，不碰其它聊天。保历史失败/空间账，不删除资产腾空间。

完成或失败中文报告审计、自有source/匿名聚合/无路径receipts，经指定proxy push/SHA/匿名HTTP200；禁RGB标签权重特征概率bits逐内容值身份私有路径第三方全文。8重复开发内容、患者和预训练未知，不作独立患者临床/TMI成功。
