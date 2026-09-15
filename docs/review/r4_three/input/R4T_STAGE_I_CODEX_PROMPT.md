# 给 Codex：保留 R4D 两方向，增加原第五条 G，合并为 R4T

阅读`R4T_COMBINED_EXPERIMENT_PLAN.md`、`R4T_SCIENCE_PROPOSAL.json`及`PRESERVATION.json`。当前仅授权阶段 I：代码、程序化CPU检查、现有登记元数据解析、70轨迹dry-run、GitHub去身份公开交付。

**不授权GPU、真实目标RGB/mask、源数据/源原型/源代理、额外预训练模型、源端训练、正式实验、后台等待或自行生成review通过。** 科研配方是提案，程序化数学核通过不是完整host审阅通过。

## 1. 增量范围，先明确继承

A教师时间尺度×RP六臂，B的KDG及三控制四臂全部保留科学定义；新增C方向的G_BOUND、G_CONST、G_SHUFFLE、G_GLOBAL。共14臂，每臂既有四主序+回访流，共70条完整轨迹。

本版替代旧R4/R4D尚未执行的排程，不先运行旧50条再运行70条。如果旧A/B已在实现，直接继承，不为换名称重做；如果已经发生真实运行、未披露修改或与本提案的科学冲突，先明确报告，不擅自取消程序、改写结果或合并为新批次。

从`2f90a6a0933cec3a25337a0d46ee632f8d772840`建`experiment/r4-three-track-teacher-kernel-boundary-v1`，保留实际执行`185fd440b16920f367c1f5e3096ab495bd85c0ec`的审计/NFS/cuBLAS/回收修补。禁止改main、历史配置与结果、固定依赖模型源码。

Registration：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`；回访流：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。阶段I不解码像素。路径优先解析旧私有配置；没有真实路径不阻塞其余代码和toy tests。

## 2. A/B不可暗改

A：C、RP、MT、MT_RP、FT、FT_RP。所有教师采用C的当前输入统计，FT不是source-BN的N。EMA只更新affine、m=.99，无warm-up，在本图最终预测和memory commit后更新。RP维持原32维/rank8/四bank/λ=.05/累计统计；标签和可靠性来自该教师，写入的是学生pre原图特征，不是教师或post特征。MT_RP和FT_RP为取学生pre特征多一次前向，共9；其余8。

B：KDG、K_ALL、K_MAG、K_FREE。严格按原§5在`res.conv1`和`res.layer1.2.conv2`接入，不扫描层。Cayley、DC/detail、幅度、自由DC、精度和参数规模都不改。保留source底层weights，更新的是生成有效kernel的坐标。一个Adam更新原BN+新增坐标，B没有RP、EMA、图或额外loss。

新增K零初始化只保证更新前函数与C对齐；其首步更新后可以不同于C。不得为满足错误等价断言断开新增坐标梯度。

## 3. C方向明确继承原第五条G，不是新增不相关路线

新增G_BOUND为“当前图边界锚定图监督”。继承G的图teacher→C BCE接口，但不是原SPEGC完整复现。四个G臂都只有原BN适配，不含PCA bank、EMA、KDG或router。

输入仅当前raw pixel_rgb[1,3,512,512] [0,1]、原C六个已对齐概率图[6,2,512,512]和q。图用4x4概率/RGB平均池化到128x128，不对logits池化，不借用RP的32x32中心采样。保留所有teacher张量detach。

可靠点定义(q<=.1或q>=.9)且六视图hard一致；图网格和完整512网格分别计算。轮廓为q>=.5的真实四邻接类别变化的两个端点，不虚构padding背景，膨胀半径2格。

G_BOUND/CONST/SHUFFLE自由域=轮廓带∩不可靠；GLOBAL自由域=所有不可靠。所有固定点保持原soft q，不round到0/1。无轮廓/空自由域不报错，q*=q仍执行C；不得临时扩带或修改阈值。

边为四邻接32512条。原RGB图网格逐通道标准化，std floor=.05；w=max(exp(-||ΔRGB_std||²/(2*.5²)),1e-4)。CONST按水平/垂直方向取相应原边权均值；SHUFFLE保留各方向边权multiset，仅重排边权位置，局部hash seed按主计划，绝不改变全局RNG；GLOBAL保留原RGB边。

能量=0.5 Σ_free(Q-q)²+0.5 Σ_edges w(Qu-Qv)²，λ=1。CPU float64做64步同步Jacobi：Qnew=(q+ΣwQold)/(1+degree)，固定点每步原样拷回q。不得分配16384²稠密矩阵，不得改成Gauss-Seidel、增加闭环效果停止或按GT选迭代。

把Δ=Q-q双线性上采样到512；可写支持为nearest(free)∩NOT reliable_full。q*=where(支持,clamp(q+upΔ,0,1),q)。支持之外与可靠区必须原样保留q。**没有全图包含投影、锐化或log-odds替换。** 两个通道不强制包含，记录违反但不据此改规则。

strong仍全图普通BCEWithLogits(strong,q*)，不额外按band均值重加权、不增加图loss；最后原图网络post输出才是正式预测。每图8forward、1backward、1Adam、0VJP。图teacher有限支持不意味着学生最终变化也只在该支持内，不作安全保证。

## 4. 辅助评价不能回流

全部14臂保留原q teacher Dice/Brier/GT前背景计数，只有三个RP臂评实际写入token质量；无memory臂NOT_APPLICABLE。

G多评q*相对q的错误→正确、正确→错误、其余两类；分别记录全部像素、允许支持和实际hard-flip域的分母。GT只在学生最终预测和全部无标签状态commit后进入evaluator，绝不回传图、host、teacher、optimizer或采样。当前q/q*/mask评后释放，只保存标量。

保留主学生Dice/ASSD及尾部报告；teacher改善不等于学生改善。G能量下降或固定区不变也不等于方法有效。

## 5. 唯一矩阵和预算

14臂顺序固定为C,RP,MT,MT_RP,FT,FT_RP,KDG,K_ALL,K_MAG,K_FREE,G_BOUND,G_CONST,G_SHUFFLE,G_GLOBAL。

正式70轨迹、136570评分/Adam/backward、1112070forward、0VJP。主序56条109256记录；回访14条27314记录，分开汇总。每实际GPU smoke为32Adam/backward、260forward，阶段I不得运行。

C/RP只各运行五流一次。不得新建各方向专属重复C，不按中间分数决定另一路是否运行。资源上限沿用原R4D，但实际GPU/并发/后台权限仍待审阅后用户确认。

## 6. 实现与测试

复用原A/B代码和已验证基础设施，只加boundary_graph、辅助评价和14臂薄入口。不能改R3的85job或旧R4D的50job常量来绕过授权/分析器，新science与run binding要如实生成。

保留旧回归。新检查以本包reference为参考：图边/可靠性/轮廓无虚构边框、恒等退化、常数权重总量匹配、shuffle多重集合与RNG隔离、solver与显式小系统、范围与固定点、512/128支持回写、detach、当前状态与评价隔离、实际8/1/1及70job计数。不要求模型先涨Dice或图必需翻转像素才准入。

本包reference是纯张量数学代码，不是已完成的host；测试记录不能冒充你的集成测试。初次失败日志和已知EIO风险如实保留，不因为反复通过便宣称根因解决，不无期限加诊断。

## 7. 交付后停止

推送新分支，交付完整implementation SHA、新science字节SHA、窄patch、真实CPU日志、A/B保持性、70job实际元数据dry-run、生命周期、来源/变更说明、预算/未运行项和REVIEW_INDEX。

只公开去身份材料，禁止真实路径/样本身份/权重/RGB/mask/论文PDF/设备授权。状态写：

**R4T_IMPLEMENTATION_READY_FOR_REVIEW**

停止等待外部review。不得自行审批或后台等待后自动开跑。实现审阅通过且用户明确资源授权后，才进入一次性三方向完整实验。
