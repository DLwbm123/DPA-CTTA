# 空间logit梯度方向 V1：冻结协议

本轮为CAREM启发的独立双Bernoulli分割变体，不是原文逐样本多类几何或冲突投影的复现。用户明确允许机制改造。只改变既有自然对数Bernoulli熵损失的输出梯度方向，不叠加Utility/MARS/LAST/结构loss/教师/源KL/RDC/ENGRAM。无新源训练、标签或未来图用于适应。

## 假设与精确定义

此前普通熵持续适应弱于强源，MARS固定软标签能减害但未证条件化持续收益。新假设：整图logit的置信度径向放大和其余方向对后续图像的效用不同。单像素二类切向为0，不妨碍每头整图向量一般非零；不把非径向当结构正确性。

每图各头空间logit z_c，基础g=dL_entropy/dz；沿用既有概率clamp1e-6、自然对数、两头全图均值。分别逐头以FP64计算 g_rad=<g,z>/||z||²*z，g_tan=g−g_rad；零能量头radial为0。原梯度总范数≤1e-20/非finite停止。四个新方向臂若整体||g_tan||/||g||≤1e-6停止，无替补方向、不改阈值；零/非finite所选方向也停止。投影写回FP32后TANGENT逐头径向余弦绝对值须≤5e-6。

TANGENT用g_tan；RADIAL用g_rad；RANDOM_ORTH在各头减去固定高斯空间随机向量上的分量；NORM_ONLY保留原g方向。四臂总输出梯度范数统一为||g_tan||，ENT_NORM保留原g。固定随机方向由既有SEED+case/condition派生，同内容/条件返回重复，不增加独立seed试验。因此TANGENT对NORM_ONLY检验方向而非仅梯度缩小。

只更新82个BN仿射张量/19136坐标，源runningstats/卷积/head冻结。每arrival一步Adam lr1e-4、beta(.9,.999)、eps1e-8；所有core最终实际FP32参数L2=.01±5e-7、与原Adam提案cos≥.999999，无zero。网络Jacobian及Adam之后实际logit方向不保证仍正交或norm守恒；逐步记录实际Δlogit径向余弦、梯度保留比例/余弦与函数变化，不把输出梯度几何当真实模型安全证明。

## 范围、矩阵和资格

同8个已暴露内容、两顺序、五块[0,1,2,3,0]，每流40到达；不扩图/seed/模型/流长。TANGENT RESET/FRESH；RADIAL FRESH；RANDOM_ORTH FRESH；NORM_ONLY FRESH；ENT_NORM RESET/RETAIN/FRESH，共8 core工作/640arrival/640匹配更新。RESET参数和Adam均清；FRESH只保留仿射；RETAIN均保留。

真实native CW1/CW2各RESET/RETAIN共320arrival，原BN语义和步数不改。SOURCE_EVAL/CURRENT_BN/旧D/R/平移R五静态强参考保留。总24轨迹/960科学arrival/2320读出/29绝对聚合行。同冻结源BN的core是主比较；native currentBN语义差必须披露。

额外TANGENT RESET正序完整40arrival重演，80对PRE/POST logits及末model.state_dict逐元素一致；STRICT deterministic不warn_only/回退。仅认证这条流同进程，不认证所有Adam/RNG或跨进程。额外40更新计费不计科学矩阵。

worker沿用现有白名单/scorer隔离，并拒绝历史M1整个root；只8张登记RGB/现有模型及旧冻结头资产，内容清单无mask路径。标签仅GPU完全退役后CPU scorer读取。全精度logit≥0生成bit预测；q8概率误差≤1/510，仅辅助校准，不驱动更新/主Dice，不发逐例结果。静态缓存不参与学习。

## 唯一主比较与完整门槛

TANGENT_FRESH−NORM_ONLY_FRESH:全流PRE Dice，均值≥+.5pp且绑定8内容、条件、顺序和通道的描述性95%配对bootstrap下界>0；不是独立患者推断。

推进还须PRE对RADIAL/RANDOM_ORTH/ENT_FRESH下界均>0；首块/全流PRE对SOURCE下界>0；最差单元≥−1pp；返回PRE对源均值≥−.2pp。TANGENT FRESH POST高于五静态/ENT_FRESH/TANGENT RESET/nativeCW1/2的RESET和RETAIN；无新增空/嵌套；ASSD/HD95/vCDR与定义覆盖不劣源；实际规范更新全匹配且无zero。任一失败advance=false。源前景/边界corrected-spoiled、全条件/通道/顺序、校准和函数变化完整报告；不能只胜负熵基线或当图POST就称持续收益，不按标签选方向/幅度/阈值。

## 资源与停止

GPU1200秒/CPU评分900秒/输出2GiB，一次派发无重试，每唤醒至多一个有据新轮/同聊天最多一活跃GPU。先CPU合成几何/toyBN真实更新/退化重置/配置分组检查，再NAS实际挂载容量读写、GPU free≥8192MiB、正式中性入口及父子ps+nvidia-smi准入。

首TANGENT RESET两arrival投影=已用时+(640+40−2)*均时*1.35+320(native预留)+60收尾，须≤1180。不准入/数值/ownership/access/strict/范围失败保存停止，不砍控制/调阈值/延预算重跑。2320份q8约1.22GB、bit/gate约.24GB及元数据余量，实际超2GiB停止。

历史累计GPU2374.4118444919秒、ledgerCPU1260.4551422301秒不清零；既有.0084569158基数差与未计时阅读/准备错误保留。已完成本地4测试body .631秒（三位精度，启动/导入未计）；后续远程CPU/正式进程另计。准备期间曾误用服务器无python退出127、找不存在文件、一次本地协议写入cwd前缀错误，无CUDA/图像标签权重读取，未计主体时间，保留披露，不算GPU失败。本地早期测试命令含测试名且已结束；正式主子入口全部中性。

8内容患者/预训练成员未知，无独立患者/临床/TMI成功结论。完成或工程停止均中文分析并公开GitHub交付；只自有源码/配置/匿名聚合/无路径receipts，禁私有图像标签权重概率逐例评分身份路径与第三方全文。
