# H1：源保持特征历史 V1 冻结协议

## 假设与适用依据

独立GAIN启发变体，不是原论文复现。已读GAIN30页，原B1/长流分类有正证据；数学迁移及反例见已交付post-q1-source-preserving-history。当前真实源模型输出第三项32维head_input，两个1x1线性头有bias；正式CPU检查加载既有checkpoint、合成RGB一次forward并要求源logit与重调用seg_head(features)逐元素相同，GPU资格再核32个真实冻结帧。

核心假设是冻结源logit之外的跨图特征关联能提供持续读出收益。与Q1同图probe一致性不同，信息来自历史特征；与G1/M1/V6不同，不更新模型/BN/Adam，固定编码器坐标保持可比。源错误、伪关联、首图几何/组成与不确定性错设仍可能损害，不能将源保持初始或不更新网络称为安全/TMI创新。

只检验完整32维direct-q相对仅源margin的一维direct-q，不把32维r维数惩罚或binary clip本身当新机制；不按结果选择r/clip/HALF为新主臂。R75/R76旧全类内白化及均值修正负结果保留，只读复用别的聊天证据，不重跑其控制、不混成本。本轮不将整个特征白化，首图V固定而历史状态是源校准的类一阶矩；仍改变有效判别几何，必须有1D、打乱、RESET及结构控制。

## 固定状态、坐标和读出

全部512x512点参与统计，无抽样网格/裁剪/额外增强。source标准BN/编码器/seg_head都冻结，权重不训练；OD/OC独立Bernoulli。特征与logit在原生FP32源forward输出，之后统计/工作读出使用FP64。最终full/residual logit>=0生成bit，不用q8。无参数更新，因此不虚构.01L2、Adam年龄或匹配norm。

每头以首个当前到达的冻结特征建立mu和逐维总体方差V，之后固定。活动维精确定义FP64 V>0，不加floor/ridge；全无活动维停止。零方差维原源直通，以后变动也保留源贡献。rank1输入为同一source head不含bias的margin=w·features，自身1D mu/V，权重1；其全部零方差也停止，无临场替代。两头分别保持状态。

在活动u=f−mu上，中心偏移先验m0=−Vw/2,m1=Vw/2、kappa=3。后验中心=(3prior+累计加权u)/(3+n_class)；共享固定对角V下似然差g_q，候选q_logit=z_source+g_q−w_active·u。零方差坐标不进入似然而源logit保留，有直通维时只是条件工作模型，不声称真实源类中心/完整联合生成后验。无历史显式q/r=源，不让初始不确定性温度改首次读出。

当前图用冻结源p=sigmoid(z_source_FP64)、可靠性max(p,1−p)分配两类mass=[1−p,p]*reliability/N。每图两类新增支持之和<=1，无独立像素或患者假设；n和一阶矩同样归一化。保留实际软类质量，不强制每类计1，不用原反支持频率先验。只用source分配，不用adapted反馈，不读标签/未来图。

r_c工作似然包含Mahalanobis distance/h_c与d_active*log(h_c)，h_c=1+1/(3+n_c)，然后同源残差校准。距离通过二次型展开，FP64负数舍入截到0，非finite停止；源保持校准不为风险安全界。clip在logit中将r裁剪到source与q区间，等价二元概率区间裁剪；HALF_Q=(source_logit+q_logit)/2为固定几何混合。direct-q/r/clip/half不反馈更改历史，状态完全共享。

类关联打乱SHUFFLE仅将两类source mass在全部像素的一维展开上循环平移N//2+1，特征、首图几何和每类支持不变。没有随机seed新增；它摧毁当前特征与分配的对应，不宣称保留所有空间结构。FULL/RANK1/SHUFFLE累积n每次必须一致atol1e−12、rtol0，RESET FULL/RANK1同样配对，否则停止。

每到达先emit PRE，再按当前source分配更新，再emit诊断POST。RETAIN保留首图几何、n、矩；RESET每图重建几何/状态，PRE始终源、POST仅当图诊断。所有核心首个到达PRE和每个RESET PRE都须与源FP64转换逐元素相同。当前图复制与源赋值无oracle保证；真实图resize不受复制不变性证明。

## 冻结矩阵和缓存边界

同8个重复开发内容，两顺序、每流40到达，5块[0,1,2,3,0]；不扩数据/seed/流/模型/源训练/权限。核心8个读出工作：FEATURE_Q RESET/RETAIN，LOGIT_Q RESET/RETAIN，SHUFFLE_Q RETAIN，FEATURE_R RETAIN，FEATURE_CLIP RETAIN，HALF_Q RETAIN。

每个顺序共享一个source-only流：3个持久状态FULL/RANK1/SHUFFLE和2个每图重建RESET bank；80个共享历史到达、400个实际bank状态更新，640个逻辑核心读出arrival。真实CW1/CW2各RESET/RETAIN另320arrival；五静态SOURCE_EVAL/CURRENT_BN/旧D/R/平移R保留。总24个读出轨迹、960逻辑科学arrival、2320预测、29绝对聚合行。多个读出不是独立学习轨迹/患者；state updates与readout arrivals必须分开报告。

32个已登记case/condition源帧只forward一次并以FP32缓存source logit与32维feature在RAM/GPU，复用到各相同内容/条件的到达。预计算是固定源的只读推理，不用于提前校准或更新任何历史；Stream只接收当前帧，首图mu/V只从该顺序当前首个到达建立。缓存不落盘，不上传。每帧源head重构精确、源shape(1,32,512,512)与head(2,32,1,1)检查；模型权重及buffers整个worker不可变。source冷静态与native currentBN语义差仍披露。

worker白名单8RGB、既有checkpoint及旧冻结头；拒绝Q1整个root/scorer/旧评分。不读取旧预测或标签，静态旧参考只重算固定模型，标签仅GPU退役后独立CPU评分。完整访问/所有预测identity覆盖检查。

额外正序40共享到达重演：FEATURE_Q RETAIN PRE/POST共80个完整logit tensor逐元素相同，三个持久history完整mu/V/active/n/s/images状态亦精确。使用相同缓存冻结features，不重跑encoder；仅认证该缓存历史流同进程，不认证原生CW/RNG/跨进程/encoder重推。额外200bank更新独立计费，不加入科学400。STRICT不warn_only/回退。

## 主比较与推进

唯一主FEATURE_Q_RETAIN−LOGIT_Q_RETAIN:全流PRE Dice，均值>=+.5pp且8内容绑定条件/顺序/通道的2000次描述配对bootstrap95%下界>0。与SHUFFLE_Q PRE下界>0；首块PRE及全流PRE对SOURCE下界>0；最差PRE对源>=−1pp；返回PRE对源均值>=−.2pp。

主臂POST须超过五静态、FEATURE_Q RESET、LOGIT_Q RESET/RETAIN、SHUFFLE_Q RETAIN与CW1/CW2各RESET/RETAIN；r/clip/half仅同状态预定副读出，不依其表现改主结论。无新增空/嵌套，ASSD/HD95/vCDR及定义覆盖不劣源；源不变/head重构/support一致/零模型更新全部通过。任一失败advance=false。

全部通道/条件/顺序、首块与返回、PREPOST、源参照的纠正/损害、函数变化及历史活动维/支持范围都报告。diagnostics的history_images_before是到达PRE计数，在POST行复用；shared_full_*是共享FULL更新后状态元数据，不代表RESET/rank1的状态。不能据此把POST支持量解释为PRE信息。

首图源码等价与数学合成可行性不代替真实推进；8内容患者/预训练成员未知，无独立患者/临床/TMI成功。报告中逻辑960次不能冒充960独立图像。

## 工程、预算和停止

先9项CPU检查，包含与标准库逐点公式对照、坐标变换/零维直通、图复制、shuffle质量、先预测后更新/RESET、逐元素重演/clip、数值退化/配置，以及**实际既有checkpoint+合成RGB** CPU forward的head精确重构。CPU会读取既有权重、进行1次模型forward，不读真实图像/标签或CUDA；这与之前纯代数检查的0模型读取不同，单独报告成本与范围。

正式GPU前NAS实际NFS/容量小写读、free>=8192MiB、中性入口和父子ps+nvidia准入。GPU1200s、CPU评分900s、输出2GiB，一次GPU派发无重试。首正序2个共享到达保守投影=已用时+(80+40−2)*均时*1.35+320native+60收尾<=1180；均时包括8读出/5bank实际工作。数值/shape/ownership/access/覆盖/重演/预算失败保留停止，不改V阈值/加floor/砍控制/延预算重跑。

历史GPU3530.1142151355157、ledgerCPU1652.9929370585148，另既有本地body约.631及上次公式主体约.0010669171单列。旧.0084569158基数差、未计时导入错误/阅读编排继续披露不清零；本轮准备及实际进程另计。R75/R76其它聊天成本不混入。完成或工程停止都中文报告/审计及匿名聚合/无路径receipts公开GitHub，经代理核账号/分支/远程SHA/匿名HTTP200。禁图像标签权重逐例预测概率评分身份私人路径第三方全文。
