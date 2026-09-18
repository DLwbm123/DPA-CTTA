# C组：可靠性加权基证据 RBE

## 科学问题
当前图像不同位置的适配证据可靠性不同时，能否以源端学习的观测方差和稳健估计融合这些证据，而不是把所有低置信度像素简单丢弃？
借鉴Bridge的query/basis中介特征和GUIDE的噪声课程可靠性学习。**不是Bridge的前门调整复现，不声称因果解耦；不是原GUIDE的Dirichlet分类头。** 两个独立二值分割通道仍由原网络输出。

## C1. 条件化patch证据
使用共同64个E_i∈R64和d∈R32。内容基Pc32×64、appearance基Pa16×64，行作cosine attention规范化。
kci=softmax(cos(E_i,Pc)/.2)，c_i=kci Pc。
ka_i=softmax(MLP96→64 GELU→16(concat(E_i-c_i,d)))。
o_i=ka_i O，O为16×8的source学习矩阵。
H_i=reshape(MLP64→64 GELU→256(c_i),8,32)，每行除max(norm,1e-6)。
content/style重建为c_i+ka_i Pa，仅作为源端约束和观测生成；患者分割特征原路径保留。

64 patches×8坐标构成M=512条latent观测：o≈Hz。state z32映射共同固定U32至vFiLM。不要把这些8维观测称为8类分割。

## C2. 噪声meta-model
h_R: concat(E_i-c_i,d)96→64 GELU→8，R_ij=1e-4+(10-1e-4)*sigmoid(h_R)。
主task-fit时R=I，h_R不训练；其余模块按下面task目标训练。
校准阶段冻结观察器、H、基及其他任务模块，在source-cal上学h_R：
L_R=.5 mean(log R +(o-H z*)²/R)。
噪声课程progress=(step+1)/256，nominal style后的source support加extra Gaussian RGB noise std=.04progress，以及观察token加独立Gaussian std=.02progress；query不参与校准。z*指nominal anchor的proxy，不根据extra noise再求oracle。该训练是在教元模型反映偏离nominal证据的误差，不是声称真实病理/设备不确定性已被分离。

所有输入只来自当前图；目标无GT且不再训练可靠性网络。C_CONST_R使用source-cal全体未额外加噪观测R的每坐标geometric mean（8维）冻结替代R_i；不从完整目标流计算全局均值。

## C3. 稳健状态更新
z^- =.9 z_prev。定义
E(z)=mean_j Huber_delta1((o_j-H_j z)/sqrt(R_j)) +.1/2 ||z-z^-||²。
固定IRLS3步，从z=z^-开始。e=(o-Hz)/sqrt R；nu=min(1,1/abs e)，e=0时nu=1。
w=nu/R。
A=H^T diag(w)H/512+.1I；b=H^T diag(w)o/512+.1z^-；z_next=CholeskySolve(A,b)。

模型定义保证A有正ridge，不允许失败后提高ridge或删patch。目标no_grad，三次小矩阵solve，2个backboneforward，无网络backward/Adam/VJP。
IRLS步数不按能量/分数早停。能量性质属于这个固定凸latent目标，不保证预测正确。

## C4. 源任务训练
每模型1000步，query Lseg +.1 state proxy MSE +.1 observation MSE +.05 feature reconstruction +.05 clean/styled content coefficient consistency +.001 basis row coherence。
feature reconstruction=mean(||c+kaPa-E||²)；coefficient consistency=mean((kc_styled-kc_clean.detach())²)。coherence对Pc和Pa各自归一化行Gram的offdiag平方均值再平均，不要求Pc/Pa跨基正交=因果独立。
source主训练使用R1、unroll3solve并反向，原source分割器参数/U冻结。后续cal才用learnedR，明确存在目标估计器从homoscedastic到heteroscedastic的更改，这是预定方法步骤，不能略去。

C_STATIC同预算独立训练/cal，每visit z_prev=0；它仍进行3步当前证据求解。
C_CONST_R同FULL checkpoint只移除patch变异方差，保留Huber和历史。这个消融不是“去掉所有稳健性”，不额外称为等参数最佳重训基线。

## C5. 审阅与否证
IRLS与独立小型M-estimator参考；无outlier时与ridge weighted least squares相同；outlier权重降且不为负；R尺度、mean除512与ridge相对比例不变；3次solve计数固定。
source variance NLL梯度只到h_R；GUIDE-style校准不能反向修改backbone/H/observer。STATIC无history，constantR只来自source-cal。噪声级别/源标签不进入online API。
若R与heldout proxy error关系差、CONST_R已解释全部收益、basis重建改善却不改善任务，则不声称可靠性驱动更新成立。
