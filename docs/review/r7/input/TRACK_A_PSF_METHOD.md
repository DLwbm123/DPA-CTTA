# A组：概率子空间过滤 PSF

## 科学问题
不同患者图像的成像观测不确定时，持续推断一个小型域适配状态分布，能否比同预算的逐图推断更可靠？医学结构tokens每图独立，不以历史患者形态作平滑目标。

借鉴RP-GSSM的识别到潜变量证据、MC-TTDG的共享/域相关代码本与Faller/Martin的预测协方差相关子空间。**本实现为任务改造，既非RP-GSSM的完整归一化最大似然模型，也非精确原代码本算法；不继承原文理论保证。**

## A1. 任务相关的16维适配基
在fit中选互不重叠的16个cov和16个probe group，固定原current-statistics网络、v=0。取两个通道4×4平均池化logits作32维readout。

J_cov和J_probe分别512×1024。每图只一次前向，32个readout逐行对v求VJP；两集32F、1024VJP，单独计数；不是在线免费量。
Lambda=0.01 I + J_cov^T diag(sigmoid(r)*(1-sigmoid(r))) J_cov /512。
Sigma=Lambda^{-1}，全部float64，Cholesky solve；对称化仅消除舍入。
S=Sigma^{1/2} J_probe^T J_probe Sigma^{1/2}，取top16特征向量V，B=Sigma^{1/2}V。实际控制v=Bz。

这对应池化logit局部线性化代理的预测协方差主子空间，不是整个分割网络后验。在这个代理下，J_probe B B^T J_probe^T是该预测协方差的秩16截断；实际tanh-FiLM非线性后不保留全局最优性。任意调用“最优贝叶斯医学子空间”都不成立。

投影source proxy oracle：z*=argmin||Bz-v*||²，使用full-rank least squares，记录投影误差和condition number。B不随目标流改变。不偷换成随机基/PCA或仅largest Hessian eigenvectors；计算失败则source prep未完成。

## A2. 双代码本观测
d是共同32维appearance descriptor。8×32风格代码本归一化cosine attention，temperature.2，得到d_hat。64个当前内容tokens经32×64内容代码本同样soft attention，得到c_i及其mean c_bar。
输入concat(d32,d_hat32,c_bar64)=128，MLP128→64 GELU→32，输出o16及R16的raw。R=diag(1e-4+(10-1e-4)*sigmoid(raw))。
代码本/投影只生成观测，不覆盖原分割器的患者结构特征，不将c_i保存至下一患者。不宣称代码本已经实现真实因果结构/风格可辨识分离。

## A3. 因果过滤
state=(m16,P16×16)，初始m=0,P=I。
F=.95 W/max(1,||W||2)，W为可训练完整16×16矩阵，初始化W=(.9/.95)I使F=.9I。Q=diag(1e-4+(1-1e-4)*sigmoid(q))，初始化.01。
m^- = Fm；P^- = FPF^T+Q。
P^+ = [(P^-)^{-1}+R^{-1}]^{-1}。
m^+ = P^+[(P^-)^{-1}m^-+R^{-1}o]。
用Cholesky解，不显式inv实现。保留完整P（不能默认只存diag）；R可diag，但完整F的非对角耦合允许后续P产生相关结构。目标输出只用m^+，不Monte Carlo融合。

这里是明确假定的高斯信息过滤；识别头用source proxy targets和任务监督训练。不得称为原RP-GSSM的精确likelihood实现。F谱半径<1不证明观察器反馈+分割器不崩溃；R量是source proxy误差模型，未自动校准目标域覆盖率。

## A4. 源训练、校准和STATIC
共同1000步task训练，loss=query Lseg +.1 obs_MSE +.01 posterior proxy NLL +.05 style reconstruction +.05 content term。
obs_MSE=mean((o-z*)²)；NLL=.5[(z*-m)^TP^{-1}(z*-m)+logdet P+r log(2pi)]/r。
style reconstruction=mean((d_hat-d.detach())²)。content term=mean((c_styled-E_clean.detach())²)+mean((k_styled-k_clean.detach())²)，clean/styled同support已对齐、无几何变换；除2使两项平均。

训练后冻结全部任务参数，source cal学习一个tau=.25+3.75sigmoid(a)，使R_cal=tau² R的状态proxy NLL最小；256步4样本（将4样本视为一个四时刻cal序列）。最终权重不按target挑选。
A_STATIC独立训练同架构，但每图将m,P清零/单位阵，再执行同样过滤；source/target都如此。
A_ISO_OBS是附加部署消融，R→trace(R)/16 I，其他不变；这不是把P替换为标量，也不是重训STATIC。

## A5. 审阅必须看到
基构建与小矩阵直接SVD参考一致；B原始hash；cov/probe source分组不重叠；float64精度；过滤与独立batch precision解一致；P严格SPD；高R时依赖prior的极限；零噪声下界和无未来测试；static确实不携带历史；患者tokens释放。
零FiLM状态保持输出但导数非零；state存储/加载保持下一图结果；完整随机网络source训练梯度能到观测/transition/codebooks但不到冻结主干。

## 研究否证
相同source预算STATIC已解释全部收益；posterior方差无法反映held-out source proxy误差；同style换患者引起系统有害状态改变；都需要如实报告。没有规定必须通过一个固定Dice增益才允许讨论。
