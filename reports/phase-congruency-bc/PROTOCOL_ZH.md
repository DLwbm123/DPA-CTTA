# Phase-Congruency-BC-v1：真实读取前完整冻结

本轮为一个独立CPU B/C资格实验，0新增encoder/模型/权重/训练/GPU。固定原8开发内容、32输入与原S/B位图，不扩图像/seed/候选空间。阶段A的phase.py原样复用；阶段A只证实现，不给本轮信号或收益背书。不是P1整数平移phase，不复活血管U/r3/q或历史selector/crop/800。

## 信息与相位场

每独立worker重新读取8原RGB，PIL bicubic→512、float32/255，原四外观identity/gamma.7/RGB(1.15,1,.85)/7×7 reflect Gaussian σ1及clamp，再每图全通道minmax；取当前绿色通道转float64×255。两worker使用相同CPU公式，逐项核输入/全部场/支持/分数/影子mask exact；不声称与旧GPU浮点输入exact或作者Matlab parity。

phase.measure只接受该二维灰度，不收OD/OC、条件标识/参数、ROI或identity参考。固定4尺度、波长3、倍率2.1、sigmaOnf.55、k3、noise−1、cutOff.5、g10、deviationGain1.5、epsilon1e−4、lowpass.45/order15及周期FFT分解；不扫描参数/通道/尺度/窗。先于任何OC候选提取场并只读。字段：PHASE=阶段A PC；AMPLITUDE=四尺度单基幅度和；ENERGY=累计实/两Riesz分量的模；GRADIENT=灰度Sobel x/y（reflect，各除8）的模；SHIFT=完整PC场沿宽循环移W//2+1（512时257），不移动图像或mask。原始单位保留，不用GT归一或择向；AUC和匹配排名均逐输入计算，不需把幅度与PC数值尺度硬配齐。

这是同RGB确定性结构信息，不是独立语义观察。PC包含相位偏差、频宽、噪声权，胜幅度也不能单独证明纯相位因果；血管、纹理、反光可能主导。

## 空间、外轮廓、支持与读出

S/B取原my-gpu迁移E8封存SOURCE/TOPO_SMOOTH。联合差异8连通，OC-only定义OD无差/OC有差；逐项重核3028登记块/query，固定legacy共同2881块、6997 OC编辑像素，147非共同保留覆盖记录但不增空间。原direction是SOURCE→B，实际动作B→S，配额仍沿旧direction标签。OD固定B。

外轮廓C(M)：最大8连通OC分量（同面积扫描顺序首个），4背景填孔，取8邻域内边界；小岛/洞边不作为杯缘。固定B与每个单块B→S候选分别求C，允许缓存B边界但新合成check核缓存/直算同值。

支持U：两轮廓均非空，且存在C(B) XOR C(candidate)位置使**未移位PHASE>0**。没有阈值搜索或半径/容差扩张，0是阶段A噪声/相位权给出的精确零；不按q符号、强弱分位或GT支持。该U只限定本协议的单块影响动作空间，不是所有联合相位方法的普遍上界。若只增加零PC位置导致轮廓平均分母变化、但没有正PC成员改变，则该块仍不支持，所有臂置0；限制明确不暗借周长作为支持。

五场固定同构读出q_F=mean(F[C(candidate)])−mean(F[C(B)])，高分支持撤回。对照PERIMETER=−|len(C(candidate))−len(C(B))|/len(C(B))，用于周长变化偏好控制；它不构成幅度匹配或语义真值。空轮廓/无U统一0弃权，真正计算错误硬停。常数场轮廓均值差应0，即使长度变化；专门合成核查。没有额外ROI、边界窄带或距离权。

9臂固定顺序PHASE/AMPLITUDE/ENERGY/GRADIENT/SHIFT/LOGIT_Q/RAW_MARGIN/GEOMETRY_ONLY/PERIMETER。后三个旧分数来自独立sealer对白名单无GT E8 worker缓存的导出，只含块索引/公共元数据和三列控制；不读旧效用或结果。exact LOGIT_MATCH未保存，不假称具备，不新增模型取它。所有臂只同U，外部共同0平分，保2881完整AUC分母。

SHIFT非退化预冻门：PHASE/SHIFT非零位置并集上数值不同位置比例≥.5，且两分数(|q|>1e−12)非零块并集上|qPHASE−qSHIFT|>1e−12比例≥.5；分母0门失败。报告最终影子动作实际改变帧数，数值改变不冒充动作改变或GT收益。

## 无GT影子联合决策与配额

worker阶段在GT前封存唯一PHASE联合mask：同U且q>1e−12、每个单块候选OC非空/在B OD内/8连通1组件/无4连通孔；同时安装全部符合块。联合不合法则整帧回退B。不得按预算、方向或条件挑选。对每一控制按PHASE回退后实际选块配额(exact size,旧direction,distance_bucket)，从同U的单块合法候选内取本控制分最高者（同分block_id升序），不要求对照正分；联合同样合法性回退。若实际配额因回退不再精确，匹配门失败，不称严格等预算。

两独立worker各重新解码、重测32视图，比输入/五场/支持/9分数/全部影子mask/配额exact。worker只8登记RGB、旧封存bits、新导出cache与本轮private；拒GT/效用/旧结果/模型root、禁止网络。分别触发3文件负测与socket构造负测，不发连接；不删旧失败。两worker OS0且精确PID/start退休后scorer才8GT原登记SHA（原协议必需）并nearest512：OD<255、OC==0。旧SOURCE/B迁移不证明原旧服务器bit parity。

## 第一门B1：支持受限精确联合上界

先且仅算共同全集、U内各32个联合OC Dice最优放宽上界。复用Fraction比值优化2(T+Σt)/(P+G+Σa)，不相加单块utility；从实际oracle mask核64个精确比例和OD固定。忽略拓扑/嵌套，因此标RELAXED_UPPER_BOUND；合法影子规则是其子集。共同全集原+.4721342181562814pp重现差≤1e−12，基线B四条件87.8052924668%容差1e−8；不套identity90.98749%。

固定[2,1,1,1]/5四条件，8内容等权；bootstrap2000、seed20261018。若U上界均值<+.30pp，B2/C标NOT_RUN_SPACE_GATE，立即止：不构造单块效用标签/算AUC/评价影子mask/局部GT或子组。≥门仅免空间否决，不证明合法可达，也不是独立患者总体推断。

## 第二门B2：辨识

仅B1过后，在2881完整共同块计算精确单块B→S Dice效用，0效用不参与正负配对但保计数，不支持仍分数0平分。每帧无正或无负AUC=NA；内容内按可定义条件的固定权重归一，内容CI仅8内容均定义。全部输入/内容/NA分母保留，不后选子组。

唯一主：PHASE−AMPLITUDE BLOCK AUC均差≥.05且配对CI下界>0；PHASE BLOCK AUC≥.60；相对ENERGY/GRADIENT/SHIFT/LOGIT_Q/RAW_MARGIN/GEOMETRY_ONLY/PERIMETER各BLOCK差CI下界>0；定义输入≥24/32且8内容全；|utility|配对加权主差≥0；SHIFT两个非退化门均通过。全部合取，失败C NOT_RUN_SIGNAL_GATE，不翻q/改主/去负条件/选臂/调参数。

## 第三门C：联合收益与损害

仅B1+B2全过，评价已封存PHASE联合输出；不得GT后重建决策。相对B OC≥+.30pp且内容配对CI下界>0，SOURCE完整对照；OD exact B，每帧每头不新增空、每帧不新增嵌套，表面两头64/64及vCDR32/32定义；任一内容/条件/头Dice下降≤1pp。OC ASSD/HD95内容加权均值之比bootstrap97.5%分位≤1.05；vCDR误差增量CI上端≤.005。对全部8个实际配额匹配控制的OC收益CI下界>0且配额无破坏。所有分头Dice/ASSD/HD95/vCDR、NA、空/嵌套和完整分母均报。不能将继承B的收益算模块收益。

## 资源与完成

自有源码+本协议commit冻结后才触真实。CPU新独立root baseline0/NFS容量与小写读/预计峰值480MiB、预留640MiB、输出hard512MiB（两进程exact的32视图/五场及9影子缓存需要此空间，禁止另存原始RGB/模型）；总≤1800秒、guard1740秒保存停、controller剩余1730秒。首次实际执行UTC/boottime唯一，所有工程修复共享clock；<50GB/≤24h及20GB/18GB/aux1GB继续。只my-gpu CPU、CUDA_VISIBLE_DEVICES=''，不查GPU/旧server，父子neutral argv核ps，guard独立于SSH且只管本轮精确身份；保留失败和历史资产。

实际四stage check→worker1→worker2→score完整退出、原身份退休、guard退出、未知父OS码null；finalledger/访问/重演/覆盖/空间/成本齐备才可完成。新检查仅caller比值暴力对照及9个读出调用检查，不重跑阶段A。完成失败均中文REPORT/AUDIT、自有源码匿名聚合+无路径回执，经代理非force push核账号remoteSHA/匿名200；禁真实RGB/GT/weights/bits/fields/概率/coords/逐块逐内容值/身份/私有路径/第三方全文。8重复开发内容非独立患者或临床成功，非CTTA持续收益。唯一ctta30分钟，不逐轮审核，单唤醒最多本有限新独立轮；结果后自主分析且保所有停止条件。
