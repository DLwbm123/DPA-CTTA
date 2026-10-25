# run08 源监督异方差外观滤波

run07固定噪声F较I仅+.04349pp且一源下降，未目标。新假设是测量可信度随病例变化；不将其断言为已确认失败原因。不调Q/R倍率、重置阈值，不改外观均值模型。

冻结run06均值预测器和run01 base，沿用run07源fit校准R、split与所有流/gate。新增置信头：冻结外观CNN pooled64特征→Linear32/ReLU→Linear4，logvariance clamp[-9,0]。初始最后层weight0、bias=log(diagR)。每源1200步batch8、AdamW lr.001 wd.0001、seed92508，clean+已知随机gamma/RGB源fit pair，标签为冻结均值估计残差，最小化0.5*mean(e²/var+logvar)，不再训练均值/base。

Rt=diag(sqrt(vt))*Corr(R)*diag(sqrt(vt))；Q=R/16，P0=4R，innovation阈值13.3、均值裁剪与run07相同。异方差F持续和RESET每图复位，I为原均值。源流与run07同四块、source dev、原Dice门保持：F−I≥.1pp、F−RESET≥.05pp、每源F−I≥0、cleanF−ZERO≥−.1pp、外观p10≥−1pp。另加入可信度门：clean/两种已知光度source dev的高斯对角NLL较常diagR平均至少低.05 nats/dimension。不以未建模blur作参数真值。失败不目标，不放宽门。

通过才五源F/RESET各一次目标，I/ZERO复用已有；无目标标签调参。source dev多轮开发/base源暴露披露。3小时硬限含训练/验证/可选目标，300秒预停/60秒KILL；1GBsoft2GBhard预计<200MB，原钟不重置，整个窗口新增<50GB。未过即保留阴性，避免同族倍率搜索。
