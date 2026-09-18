# R7 三组方法的共享协议 v1

本文件与 specs/COMMON_PROTOCOL.json 是新设计；原R6所有配置、结果与失败记录不变。先实现，审阅后分阶段授权真实计算。**这不是R6-B/R6-E的重启，也不是把论文模块直接拼接后宣称新方法。**

## 1. 当前边界与独立性
当前任务仅Stage I：代码、随机初始化完整网络/程序化CPU检查、元数据dry-run及审阅材料。不读取真实RGB、mask、源/目标checkpoint，不查询GPU，不进行真实源端训练或目标推理。注册元数据可按已有权限读取；不能把元数据路径当作允许解码资产。

三组可独立达到READY和审阅，不因一组失败而重写另一组科学定义。共享目录只由协调任务维护；并行执行使用独立worktree，禁止多个agent同时改同一文件。Stage I不下载/训练基础模型，不运行上游脚本。仅依赖已固定CTTA/GraTa实现和已有PyTorch环境；新增依赖需记录版本/许可证/用途，不能安装“最新版”悄悄改变环境。

必须独立标记 implementation_ready、source_binding_ready、source_training_started、target_execution_started。缺少真实源数据只阻塞后续SOURCE_PREP，不自动阻塞已经能验证的程序化实现；没有源数据时不得用目标数据替代。科学冲突或核心实现不完整则不能READY。

## 2. 共同分割器、适配接口与观察器
- 现有ResUNet34、原登记RIM_ONE_r3 checkpoint不重训backbone。目标输入512×512，两个独立OD/OC logits通道。
- 新方法和C0统一用current-batch/spatial BN统计，track_running_stats=False；BN affine与其他分割权重冻结。C_BASE保持历史C的一切更新规则。这个选择不等于原始source-eval N。
- 只在已存在的up1、up3输出增加FiLM，各256通道，ambient v为1024维，顺序up1 gamma256/beta256/up3 gamma256/beta256。
- 调制：h' = h + expm1(0.1*tanh(v_gamma))*h + 0.1*tanh(v_beta)。v=0必须精确恢复原输出，同时训练/Jacobian在0处导数不能被if分支切断。不得改seg_head、重写BN、增加输出融合。
- 每次新方法在线先做一次v=0观察前向，再根据当前观察及已有状态更新latent，第二次原图前向输出。两个forward不是“单forward”或“零计算”。模块MLP、矩阵分解、ISTA/IRLS另计成本。
- 观察器d_raw=RGB均值/总体标准差6维 + res.conv1通道均值/总体标准差128维=134。RGB用原预处理后的0..1值；标准差unbiased=False。fit-only归一化max(std,1e-6)。MLP 134→64 GELU→32。不用dropout、新BN或目标running scaler。
- 当前患者tokens：up3先按每通道空间均值/总体std规范化eps1e-6，adaptive-average-pool到8×8，固定seed20260918的256×64正交投影，再LayerNorm(eps1e-6)；64×64。投影使用QR，R对角为负时同步翻转相应Q列，固定生成器不耗全局RNG。
- 完整原始特征仍在分割路径中；字典只参与状态观测，不让字典完全重建/覆盖患者结构。tokens、概率图、患者标识不得跨visit保存。保持变量见各组spec。
- 临时code更新先完成数值检查和最终预测，再提交持久状态、释放evaluator。任何失败hard-stop，无自动重试和静默回源。

## 3. 源端分组与信息预算
保持已有效的新模块fit/cal/val subject-disjoint划分。没有现成划分时，按SHA256('R7_SOURCE_SPLIT_V1|'+group_id)排序，floor(.70n) fit、floor(.15n) cal、其余val，先冻结manifest再读取像素。至少fit32/cal4/val4组。group不能从文件名猜患者身份；只有content/eye ID时明示依赖风险，不能叫独立患者检验。原checkpoint是否见过这些source患者单独审计：这里是新模块划分，不是源主干的独立盲测。

源端可用标签；目标阶段不能挂载源RGB/mask或使用目标label。部署允许源训练得到的基、字典、scaler、模块和参数，不能因不保存原图而宣称隐私保证。额外源端学习必须与C的低准备成本同时披露，并与各组同预算STATIC比较。

共享源光度模拟器：9参数顺序 brightness, contrast, gamma, gain_R/G/B, blur_sigma, noise_sigma, vignette；范围见JSON。
操作精确顺序：
1) x=(x-.5)*contrast+.5+brightness；clip[0,1]。
2) x=x**gamma，再逐RGB乘gain。
3) Gaussian blur：sigma=0时identity；否则半径ceil(3sigma)，离散核exp(-k²/(2sigma²))归一化，reflect padding，逐通道可分离卷积。
4) 加独立N(0,noise_sigma²)像素噪声。
5) vignette：坐标u,v在线性[-1,1]，r²=(u²+v²)/2，乘(1-vignette*r²)。最后clip[0,1]。
不改变几何和mask，不引入target style donor。参数与noise realization分开；同style的support/query用不同噪声像素。

fit128/cal32/val32个固定style anchor，各fold的anchor0为identity。fit/cal其他anchor交替选1和2个非identity因素；val其他anchor选3个因素。每参数均匀采样范围，生成seed为SHA256('R7_STYLE_V1|20260918|fold|index')前8字节big-endian低63位。这是源光度外推探针，不能叫临床domain shift全覆盖。

## 4. 共享source proxy oracle——新增准备阶段，不是新TTA模块
每个anchor选同fold两个不同group的support图，冻结主干，v初始化0，Adam16步(lr.03,betas.9/.999,eps1e-8,wd0)，最小化两图Lseg平均+1e-3*mean(v²)。Lseg=普通mean BCE +0.5*两个通道softDice loss平均；Dice=(2sum py+1e-6)/(sum p+sum y+1e-6)。support pair固定、图像噪声由anchor/group键固定。每步2次forward、**一次合并loss.backward和一次Adam**。

192anchors对应6144 F、3072 backward calls、3072 Adam。这里还未包含oracle query评价、source observer缓存、Jacobian、fit/cal等成本，禁止只报此数当全部准备成本。

得到v*_s只是小support集合的监督干预代理，不是真实域状态或最优器。不允许用query标签优化该v*。train/cal/val的oracle按fold隔离；只有fit oracle能拟合任务模块、基或字典。cal/val oracle仅评价/校准。检验proxy oracle在不同query患者上相对v0的迁移性；失败是研究证据，不用target结果挑新oracle步数。

B/C共享U32：对1024×128的fit v*矩阵做未中心化SVD，取前32左奇异向量，确定符号并冻结。秩不足32则报SOURCE_PREP_INCOMPLETE；不补随机列。A按独立方法文档构建B16。oracle投影误差逐anchor报告，不假定降维后仍等于oracle。

## 5. 源端方法训练与同预算STATIC
每组训练FULL和STATIC两个独立副本；组内相同初始化seed、fit数据调度和优化预算。STATIC不是仅推理时清历史的版本：训练也每visit清历史，任务方法其他机制保留。临时函数移除的ablation另标，不冒充重训STATIC。

每模型1000个AdamW steps，lr3e-4、wd1e-4、betas(.9,.999)、eps1e-8、global grad norm clip1；无scheduler/early stopping，固定step1000为发布权重。source fit只更新新模块，原主干/scaler/已冻结基不变。每step一条4时刻序列，完整unroll后一次loss.backward；每t有clean support观察、styled support观察、同style不同group query调制输出，共12个backbone forward。源query可微，用标签评价状态后果；query不得更新该episode状态，下一时刻仍从support提交状态继续。

每step根据step%3选择schedule，索引由局部RNG固定：
- abrupt：[a,a,b,b]，a!=b；
- gradual：[a,n1,n2,n3]，在9参数标准化空间逐次选择最近的未用fit anchor，距离并列取anchor ID较小者；
- recurrence：[a,b,a,b]。
每个时刻source support/query不同group；episode 8个图像角色尽量无重复，按fold group数量无放回可满足则必须无放回。不足8但≥4时保证每对不同和相邻support不同，明示复用；训练不把同subject伪装不同patient。每次support/query用各自键控光度noise。

6模型名义fit：72000 backbone forwards、6000合并backward/AdamW步；所有真实计数以hook为准。预处理、codebook、oracle、Jacobian和校准另计。

各组损失见spec。所有均值明确：code MSE按latent维度mean，观察MSE按元素mean，NLL除latent维度，coherence均值只含off-diagonal，内容项对tokens和维度mean。所有时间点平均而不是求和。没有目标数据参与loss选系数。

校准每模型256个steps，batch4 source-cal support观测，每步4backboneF，仅校准参数反向。A学一个R温度，B学一个残差温度kappa，C学独立噪声meta-model；task模块全部冻结，cal图像不用于任务fit。6模型名义6144 F、1536小模块backward/Adam steps；不能称目标在线计算。校准optimizer Adam(lr1e-3,betas(.9,.999),eps1e-8,wd0)。固定最终step256，不挑target最优。

source val用固定64个4步episode（允许锚点复用但组内分组约束保持），每模型；同时报告proxy latent error与query分割，不能用它再搜索系数。审阅时另核source prepared STATIC是否已经提供全部增益。

## 6. 精度、数值、计算与随机性
主干/MLP float32、无AMP；小型过滤/ISTA/IRLS及基构建float64，source可微类型转换必须保留梯度。在线no_grad下显式矩阵运算，不建Adam、不调用autograd.grad、不执行模型backward。源码层明确区分“latent更新”与“权重SGD”。A Cholesky失败、非有限数或形状错误即停止；固定方差下界是模型定义，不允许失败后随机增jitter/裁state救结果。

数学float64比较rtol1e-8/atol1e-10，float32端到端比较rtol1e-4/atol1e-5。零调制identity需CPU bitwise且导数可达。ISTA/IRLS次数固定，不按Dice、能量或运行中表现挑步数。

所有新模块状态随轨迹顺序延续；不接收domain、GT、未来图像；SOURCE/ TARGET接口分离。每独立target轨迹从同组已冻结source权重与初始state开始。observer/style/scaler冻结，不跨臂共享在线state。

model forward、backward调用、VJP/JVP、optimizer step、小矩阵solve、ISTA/IRLS迭代、wall/CPU/GPU/peak memory均分别记录。图像预处理/模块计算不伪装成0 FLOP。旧C仍8/1/1/0；新方法2/0/0/0，C0为1/0/0/0。

## 7. 目标实验与研究决策
首批SCREEN：C_BASE、C0、三组FULL/STATIC共8臂×orders0/1/4=24条，全部1951内容逐图处理；remaining_dev1695按图内OD/OC同权、四域同权、order0/1同权评分；回访4单列。

名义SCREEN：46824评分访问，122913模型F，5853模型backward/Adam（仅C）。源端计算、smoke另计。不复用历史C分数冒充新同期控制。新方法可能输C但赢STATIC，或相反，必须区分。

针对性机制阶段：A_ISO_OBS、B_PRED_ONLY、C_CONST_R各三流，最多9条，35118F、0模型backward/Adam；同FULL checkpoint的部署消融，不是最优重训替代算法。须单独审阅理由，不自动补跑。

扩展：最多2组入选，C_BASE/C0+每组选FULL/STATIC在orders2/3新增最多12条，66334F、3902backward/Adam。旧流数据仍完整保留。新顺序不是独立患者，新训练seed也不是新患者。是否做完整重训消融/多seed/新患者确认另设后续协议，本包不自动开启。

无新的+.5pp自动科学gate。每组报告FULL-C、FULL-STATIC、最差域/顺序、回访、OD/OC、配对尾部、ASSD共同有效分母与undefined、源端/在线成本。评审允许“无净均值优势但机制线索明确”的组获得有界下一实验，不能把探索资格叫成功。也允许全部NO_NEW_EXPERIMENT。

最终替代C仍需稳定、有实际意义的收益和风险解释；不能把本提案或审阅PASS当TMI足够性、显著性、非劣/等效或临床安全证据。

## 8. 授权与交付
- IMPLEMENTATION审阅前：真实source和target计算均disabled。
- SOURCE_PREP：审阅实现通过+单独用户receipt，绑定源码、split、模拟器、源checkpoint、GPU与有限caps。训练完锁定oracle/basis/module/calibration hashes并交付source报告。
- TARGET_SCREEN：源包完整性/无目标访问检查通过、冻结推理配置后独立receipt。不能沿用SOURCE_PREP或R6授权；不能让训练后脚本自动跑target。
- 所有实验失败保留证据且停止，不自动重试；纯代码/合成fixture调试不算已授权真实运行。
- 不要求为此重跑全部166旧模型测试。Stage I需数学单元、源/目标隔离、随机完整ResUNet接入和旧C窄保持性测试。
- 只读读取旧聚合要按current_result版本普通文件映射，链接仅审计元数据；不得调用历史recompute/invalidate/publish或破坏旧结果。
