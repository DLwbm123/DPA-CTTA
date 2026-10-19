# P3 冻结协议：二维持久同调与完整性保护的每图 RESET 优化

## 假设与证据

不新增源训练，不更换数据、checkpoint、既有 seed20261018，RDC/ENGRAM关闭，BN尺度/Calibration Maps路线不启用。P2源投影OC+.407735pp、填孔前景/边界净纠错正，说明结构先验含有限有用任务信息，支持检验其通过网络优化是否超越成熟后处理。它避免仅熵自强化、坏翻转教师及伪类中心；但网络Jacobian可把关键像素更新传播到其它位置，先验/目标下降不保证Dice改善。

TopoSculpt https://openreview.net/forum?id=gbjyzPp7eb 正文Eq2–7/附录G.4–G.6为机制来源。本轮是二维独立Bernoulli迁移，非原3D实验复现或TMI新贡献。P2与此前结果已用于开发选择，8个内容反复暴露，患者/预训练成员未知，不声称独立泛化。

## 固定目标和优化

每个输入/臂从同一源state_dict重置，无跨图状态。所有既有网络参数（包括BN affine）可更新；模型eval，BN运行均值、方差和计数不变，不做统计替换。源模型另一个实例全冻结。

每头原尺寸概率p=sigmoid(float64(logit))，不造softmax类、不降采样、不归一到图前景面积。GUDHI3.11.0二维top-dimensional closed cubical superlevel（用1−p子水平），前景8/背景4，列优先coface映射回行优先；保留包括零长度在内的有限H0/H1对。唯一essential H0目标长1，其余H0与所有H1目标长0：Lcor=1−p[essential]+Σ(p[birth]−p[death])，两头均值，不除pair数。相等值按库固定实现选择，重复可检查但不主张Clarke保证，不加jitter/floor。

完整性取前一步概率及其support detach。二值mask p>.5，二维binary soft-skeleton：十字侵蚀、方形3×3膨胀组成开操作，累积 image−opening，侵蚀至空；外部0，不引入截断迭代超参。它是原soft-skeleton的明示二维二值实现约定，不是作者未公开设置的认证。support×当前概率构成SP，precision=Σ(SP_current*p_previous)/ΣSP_current，sensitivity=Σ(SP_previous*p_current)/ΣSP_previous。Lcom为两头平均 10000*MSE(p_current,p_previous)+1000*(1−harmonic(precision,sensitivity))。常数1相较原负clDice不改梯度。第一次previous等源，MSE梯度零但结构项并非恒零；空骨架/非正分母硬停，不能加epsilon或回退源。

90步step0..89，0..30每步重算PH，>30每3步重算，共50次/head/臂；其它步缓存coface索引、重评当前p，不使用旧标量。L=Lcor+Lcom（step≤30），后期L=Lcor+.1*Lcom。AdamW lr1e−4（原CoW/CTCA设置）、betas(.9,.999)、eps1e−8、weight_decay.01（明确采用库默认，原文未给decay）。单图batch1、FP32网络参数与反传，概率/损失FP64，无AMP/TF32、无梯度裁剪，非finite梯度/权重停止。不同目标实际步幅不强行匹配.01；完整优化器相同不等方向/函数变化相同，需报告梯度与logit变化。

四个优化臂：TIB完整；PERMUTED每次用自身当前p计算同PH对后所有critical索引按全图扁平N//2+1循环移位，完整性不动；COR仅Lcor；INTEGRITY仅按同curriculum权重Lcom。同一p下置换保留critical梯度norm，但各臂轨迹不同，不保证实际更新norm相同。四臂均90步每图RESET，不挑中间最佳步。另原生CW1/CW2 RESET按既有固定实现1/2在线步，语义含current-image BN，不把对它们的胜出归为损失独有优势。

## 矩阵与标签隔离

8内容×4固定条件=32唯一输入。12臂：SOURCE/SMOOTH/FILL_SOURCE/TOPO_SOURCE/TOPO_SMOOTH/CURRENT_BN/CW1_RESET/CW2_RESET/TIB/PERMUTED/COR/INTEGRITY，共384bit输出。五静态臂独立重算，评分时同封存P2对应160bit parity；worker禁止P2整个root/所有scorer/旧分数，不用旧结果或标签择优。

核心科学更新11520；额外第一输入四优化臂各90步重演360，总11880。全encoder、末model及Adam state精确重复，只认证同进程该输入四臂，不认证全部32输入/native/跨进程。原生64 RESET arrival，实际forward/backward另计。主生成完成后GPUworkerexit0且inactive，独立CPU才读8GT。只输出bits/私有诊断，不落权重或密集概率。

沿用[0,1,2,3,0]条件权重、两重复逻辑顺序，960逻辑静态读出不是独立样本/学习轨迹，原条件再次读取不是持续返回。不能声称CTTA收益，continual_benefit_tested=false。

## 唯一主比较和推进条件

主 TIB−PERMUTED OC Dice≥+.5pp，8内容绑定条件/头的2000次描述bootstrap下界>0。另TIB对INTEGRITY/COR/SMOOTH/FILL_SOURCE/TOPO_SOURCE/TOPO_SMOOTH/CURRENT_BN/CW1_RESET/CW2_RESET各OC下界>0；相对SOURCE OC均值≥.5且下界>0、双头下界>0、最差条件/头≥−1pp、原条件再次读取双头均值≥−.2。OC真值前景与3px真边界净纠错CI均>0；ASSD/HD95/vCDR点值不差源、定义覆盖不减、无新增空预测/包含违例。任一失败advance_to_new_protocol=false，不换副臂、不选条件或步数。净纠错宏率和32唯一输入C/H分开报告，区域重叠不相加。

12绝对、96条件头、96区域、54对比；评分覆盖384，C+H=D。未来时间/GPU配额已取消，有限固定完整矩阵不是无限参数搜索；每轮全root2GiB空间分配+总新增20GB硬限/18GB水位+辅助1GB。新root写前reservation2.5GiB，包含源码/archive/log，依赖复用独立工程target。NAS、CPU、显存≥8192MiB与中性父子ps/nvidia准入。一次派发，无科学自动重试；工程错误独立修复源/记录保留原失败，不涂改结果。

完成须GPU_COMPLETE384/32/11880，FIRST_INPUT四臂重演模型Adam输出exact，MECHANISM/ACCESS/160staticparity、worker/scorerexit0inactive、SCORE_COMPLETE8GT/完整行、finalledger/controller原身份退休。失败保留部分及成本，不冒称完整科学对照。中文报告审计/匿名结果公开GitHub，经指定有效代理及远程SHA/匿名200核交付；禁止图像标签权重features概率逐内容值/身份私有路径/第三方全文。
