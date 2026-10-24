# run06 源监督低维外观反演

依据diag02/run03/run04/run05源证据：内容/边界更新不可靠，条件能量及方向监督未建立有效图像条件修正。停止该能量变体系列。本轮仅估计4维外观变量，用已知源合成采集变化监督其逆，不直接移动内容/边界。

冻结run01每源分割网络。外观估计器输入128RGB，conv5 stride2 16/ReLU、conv3 stride2 32/ReLU、conv3 stride2 64/ReLU、global mean、Linear64→4零初始化，输出范围RGB loggain±.25、loggamma±.5。不用归一化层抹掉颜色统计。逆变换x'=(x/exp(loggain))^exp(-loggamma)，clamp[.001,1]，不做空间变形。没有测试权重更新，无持续状态；这是低维潜变量的摊销估计，不称原迭代反演或持续成功。

每域源split seed92504（沿用run04/05），80fit/20dev，基模型见全部源；每域1500stepbatch4 AdamW lr.001 wd.0001。pair clean与已知gamma expU[-.5,.5]/RGBgain expU[-.25,.25]，本轮不加不可逆shade，clip保留不可逆局限。loss=4维参数MSE+0.5逆变换图像L1+冻结网络的分割loss，clean参数目标0，扰动参数目标真实已知。固定末模型、无参数搜索。

源dev clean/fixedappearance(gamma.7，RGB1.15,.9,1.05)/blur9比较ZERO与估计逆；fixedappearance另有已知参数逆ORACLE，仅源机制上限，不作为部署方法。源gate：扰动两条件五源均值增益≥.2pp，每源扰动均值≥0；clean均值下降≤.1pp，扰动差值p10≥−1pp；已知外观条件参数MSE较零估计减少≥20%，逆变换图像L1较未校正减少≥10%。失败不目标，不追改门/训练长度。全过才按VPTTA五源各一次目标评价，ZERO复用run01，记录时间与外观变量，不读取目标标签调参。

源dev重复开发且base源暴露，不称独立临床验证；光度可逆不保证语义分割保真，需报告真实评测。原空间分辨率仍512，128仅外观估计网络入口。整个实验4h硬限、300秒预停60秒KILL；2GBsoft3GBhard预计<500MB，原钟含全部步骤，复用权重数据不复制。窗口累计新增<50GB。
