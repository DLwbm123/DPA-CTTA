# 逐通道中位数共识：真实资格 A01

冻结假设：不同亮度视图在原分类头前可能产生通道特异的异常响应，逐通道中位数可能抑制这些响应。信号来自内部通道次序不一致，不声称增加外部语义信息。此前均值特征融合与仿射头后logit均值等价；中位数不与多通道加权读出普遍交换，因此有独立可反证机制。若gamma下各通道单调，中位数可能就是SOURCE，这种无差异不当成功。

原8登记内容×4外观、512原处理、原checkpoint/全部权重BN冻结。对每个当前x使用固定identity、minmax(x^.9)、minmax(x^1.1)，取seg_head输入三组32维特征，逐位置逐通道中位数交原seg_head，主CONSENSUS。无梯度、无伪标签、无历史模板、无方向删除、无重算残差、无decoder有限门差分。不扫描gamma/层/分位数/融合。与gate-curvature使用相同探针以固定信息预算，但算子为末端非线性次序统计，不是旧差分改符号或强度。检索现有公开protocol未发现该逐通道中位数模块，不能据此声称全局原创。

六臂SOURCE、CONSENSUS、LOGIT_MEDIAN（三组logit中位数）、MEAN（三组logit均值）、SHIFT、B。SHIFT仅将两个非中心视图特征水平方向循环半幅，保留中心及边缘分布后用相同中位数和原头；空间对应是主空对照。原阈值0无后处理。特征均值仅做仿射等价工程检查，不列为独立新控制。两worker合计192完整网络、130额外head卷积（每输入2、首输入各1等价检查），无decoder-only。

SOURCE封存位图一致；原权重BN及无梯度、生成拒绝GT/旧评价/未登记资产/网络；两worker完整logits/masks/diagnostic精确重演后独立GT评分。原合取门全部保留：主−B OC≥.30pp且95%内容CI下界>0；主对SOURCE/LOGIT_MEDIAN/MEAN/SHIFT差下界均>0；OD−B下界≥−.10pp；最差头损失≤1pp；无新增空头/嵌套；全部表面/vCDR定义；OC ASSD/HD95比B的CI上界≤1.05；vCDR差CI上界≤.005；主与SHIFT logit差>1e-3比例≥.1。8内容等权、外观[2,1,1,1]/5、bootstrap2000/seed20261018。失败关闭本配置，不扫探针/层/分位数/空对照位移、换主臂或挑样本。8反复使用开发内容不是独立患者验证。

my-gpu GPU1原UUID，NFS/probe/GPU余量/中性父子argv检查；root基线0预计768MiB预留1280MiB hard1GiB，clock3600/guard3540/stage3530，首输入成本门；项目18GB工作/20GB硬限+aux1GB。阶段wall非GPU kernel。结果自有源码协议聚合公开，原图GT权重张量私有。

工程修复：首worker在CUDA median索引确定性检查处中止，未产出封存预测或GT评分。改为逐元素max(min(a,b),min(max(a,b),c))，对有限实数严格等于三数中位数，无需索引；不关闭确定性。保留原始时钟、首次失败费用和身份回执，方法/阈值不变。
