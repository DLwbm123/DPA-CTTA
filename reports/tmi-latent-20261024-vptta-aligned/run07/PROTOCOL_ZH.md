# run07 源校准的持续外观滤波

依据run06源结果：外观误差仍约零估计的41%，但源扰动收益+.326pp且通过既定gate；检验跨病例共享采集状态能否减少逐图外观估计噪声。目标结果仅报告，不参与滤波参数选择。冻结run01 base与run06 appearance，不新增神经网络训练。

每源使用run06 fit索引均匀至多64张，每图两次同run06随机gamma/RGB变化，已知参数残差e=h(x_aug)−true。观测R=mean(eeᵀ)+1e−4I，未中心化以保留偏差误差。随机种子92507。只用source-fit，不用dev或target估R。

连续滤波F：初始m=0,P=4R，每病例Ppred=P+R/16；若innovationᵀ(Ppred+R)^−1innovation>13.3，则m=当前观测,P=R并直接返回观测；否则K=Ppred(Ppred+R)^−1，m←m+K(obs−m)，P←(I−K)Ppred。输出裁剪至RGBloggain±.25/loggamma±.5。当前图像先更新状态再逆变换预测；跨域不断状态，无目标边界标签。RESET分支每图恢复初始状态，控制单病例收缩；I为run06原始独立观测。F/RESET参数固定不搜索。

源dev沿用run06 split及图像顺序，clean→fixedappearance→color→blur四块，病例在块内按固定CSV索引顺序（run06 dev列表）。fixedappearance同run06；color gamma1、RGB[.85,1.2,.95]；blur9。三分支I/F/RESET同病例，另ZERO只作clean安全与报告。每源新流状态重置，四块间不断。源gate：两种已知外观块F−I五源均值≥.1pp、F−RESET≥.05pp、每源F−I≥0；clean F−ZERO≥−.1pp；已知外观块F−I差值p10≥−1pp。未过即停止不目标；过后F/RESET五源各一次真实目标，I复用run06，ZERO复用run01，不重算这些对照。source dev重复开发/base源暴露如实披露。

首次起点共享calibration/sourcegate/可选target，4h硬限、300秒预停60秒KILL，1GBsoft2GBhard预计<200MB。只读共享权重数据，不复制，不改既有实验/阈值，新增总<50GB。报告持续增益、复位对照、域转换及退化，不把平均改善当患者安全性。
