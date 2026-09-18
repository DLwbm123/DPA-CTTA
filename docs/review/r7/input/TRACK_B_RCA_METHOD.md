# B组：递归预测—稀疏修正 Atlas RCA

## 科学问题
源端学习的状态预测器遇到新的光度组合时，小规模逐样本迭代是否能修正摊销推断误差，并改善其他患者的结构预测？借鉴WARP与UAI2026 sparse-coding工作，但不声称复现原始WARP或继承compressed-sensing保证。

## B1. 控制空间与状态预测
共享fit oracle的未中心化SVD得到固定U32，v=Uz；z*=U^Tv*。它是经验source干预空间，不自动是解剖不变空间。
持久状态只有z_prev32、d_prev32与counter，不保存患者图像或tokens。
z^- = A z_prev + b(d) + G(d-d_prev)。
A=.95 W/max(1,||W||2)，初始化W=.9I；G=G_raw/max(1,||G_raw||2)。b是32→64 GELU→32 MLP。
第一张d_prev=d，因此差分0；不能用原始患者图像差分替代appearance d差分。
W的谱约束仅约束线性预测，不能推导整个含稀疏修正和非线性观察器的全局抗遗忘保证。

## B2. 观察势能和固定5步修正
o=MLP32→64 GELU→64(d)。H是64×32可训练字典，每列归一化到norm1（除max(norm,1e-6)，raw从局部seed正态初始化）；部署时冻结。
求delta的有限5步近似：
E(delta)=||o-H(z^-+delta)||²/(2 kappa²) +.01||delta||1 +.1||delta||²/2。
训练主阶段kappa=1。每visit delta0=0，eta=1/(||H||2²/kappa²+.1)。
g=H^T[H(z^-+delta)-o]/kappa²+.1delta。
delta_next=sign(delta-eta*g)*max(abs(delta-eta*g)-eta*.01,0)。
z=z^-+delta5。

只优化32维latent，目标阶段显式no_grad矩阵运算，不调用网络backward。源码必须标明5次近端梯度不是精确argmin；数学测试可检能量单调，但不能据此声称Dice上升。

## B3. 源端任务学习
loss=query Lseg +.1 mean((z-z*)²)+.1 mean((o-Hz*)²)+.001 mean(offdiag(H^TH)²)。所有4时刻平均。query与support是同style不同group，只用support观测更新state；query标签只产生离线训练梯度。
训练1000步unroll5次修正，必须能对H、observation、A/G/b反向；frozen backbone与U不更新。不加MAML内循环或额外source oracle per-step运行。
随后冻结所有任务参数，source cal学习单个kappa=.25+3.75sigmoid(a)，256步最小化mean((z-z*)²)，需要按kappa同步更新能量和eta。不得把kappa偷偷当learning-rate参数或改lambda。

B_STATIC单独同预算训练，source/target每visit z_prev=0,d_prev=d；修正5步保留。
B_PRED_ONLY是部署消融，复用FULL checkpoint令delta=0，只执行预测；不叫最优重训预测器。它通常比FULL少latent运算，但模型forward仍2。

## B4. 审阅点与否证
独立soft-threshold和ISTA参考；H=I的一维/对角闭式解校验；固定5步计数；凸能量不增测试；kappa≠1的步长正确；first visit差分0；same d不同raw图像不能被偷偷差分；未来图改变不影响当前输出。
source gradient穿过5步，零state的FiLM derivative可达；state-save/reload等价；全局RNG不被局部生成器消耗；只保留规定state。
若能量下降但分割不改善、预测器已解释全部收益、或STATIC胜出，则不称稀疏修正解决组合泛化。报告所有结果，不参数扫描到偶然涨分。
