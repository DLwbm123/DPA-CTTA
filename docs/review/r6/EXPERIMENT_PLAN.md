# R6：有界区域重加权的一致性适配

**状态：DESIGN_PROPOSAL_NOT_IMPLEMENTED。当前仅提供设计及阶段 I prompt，不授权 GPU、真实目标流或源端训练。**

本轮重新登记为 R6，不复活 R5-B。基线发布为 `e271098e2a12baa166fc7b77848af4a2300d9cc3`；R5-A 的真实执行 SHA 仍为 `b4b71601a5bdf87bb3a7e5d3db352610adcdff74`。

## 1. 研究决策

保留 C；关闭 R5 的接受/拒绝路线。R5-A 两主序影子收益为 −0.007689pp，连同接受率的随机影子参考也没有显示有用选择。[S1]

更关键的是，两主序的同路径 GT 即时选优机会分别为 0.0060949285 和 0.0075332988pp，均值约 0.006814pp。它约束的是既定 C 状态路径上的 pre/trial 图像级二选一，不是所有新 loss 或新长期轨迹的上限。不能调门控阈值去追求原 +0.20pp，也不能因此断言所有 CTTA 更新没有空间。[S2]

**本轮改变“哪些像素贡献多少监督”，不改变教师给出的软目标，也不决定是否更新。** 核心问题是：同样的六视图 q、BN affine、Adam 和每图一次更新，按伪前景/背景重新分配像素损失的贡献，能否形成优于 C 的完整适配轨迹？

假设不是既成事实。前景面积小并不证明背景梯度主导：大量容易背景可能几乎没有梯度。R6 必须记录 strong 分支的实际残差和梯度贡献，用结果判断这个假设，而不是由面积直接宣布原因。

这是有依据的简单损失重加权研究对照，不是“首次发明类别平衡”或保证发表的算法。已有眼底 SFDA 工作讨论此类问题，但其教师、数据集级预测库和训练协议不能直接转移到本项目。[S6]

## 2. 与已结束路线的区别

| 已结束路线 | R6 不做什么 | R6 只改变什么 |
|---|---|---|
| R5 更新门控 | 不拒绝、不回滚、不改提交率 | 每张图仍提交一次 Adam |
| R4 教师/图 | 不换 q，不做 EMA、固定教师或 q* | 对同一 q 的像素 BCE 加权 |
| R4 kernel/R1–R3 PCA | 不增加参数、记忆库或正则项 | 仍只更新 19,136 个 BN affine |
| B2/B3 interval | 不用分歧半径，不投影目标或截断残差 | 保留普通 BCE，只乘正像素权重 |

B2 已做过分区均值/打乱的区间半径控制，不等于本轮的普通 BCE 正权重重分配；不要把旧 interval 负结果改名后重跑。[S4]

## 3. 固定共同协议

只用原 Fundus ResUNet34 checkpoint；源图像、标签、query、代理、原型、源重训及额外预训练模型均禁止。阶段 I 连该 checkpoint 也不加载。

每轨迹 1,951 内容，全部参与适配；remaining_dev 1,695 用于主评分。四域原内容数与等权政策不变。512×512、OD/OC 独立 sigmoid、原 0.5 评分阈值和 mask 映射不变。每条独立初始化，seed=20260907；不按域 reset，不向 host 提供域、ID、subset、未来图或 GT。

C 当前输入 BN 统计，41 层、19,136 affine；Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、wd=0、float32、AMP 关闭。六弱视图实际 q + 一强视图 loss + 一次更新后原图：每图 8 forward、1 backward、1 Adam、0 VJP。[S3]

## 4. 唯一主候选 R_BAL

设每通道 N=512×512，q 是实际传给 C loss 的 detach 软目标。仅用于生成权重的硬分区为 h=1[q>=0.5]，并不是把 q 换成 hard target。

对 OD/OC 各自求 n_fg、n_bg。当两者均非零：

rho = clip(n_bg / n_fg, 1/8, 8)
w_bg = N / (rho*n_fg + n_bg)
w_fg = rho*w_bg
w_u = w_fg if h_u=1 else w_bg

若某通道只有一类，该通道 w 全部为 1，不制造不存在的另一类。数学上通道内 mean(w)=1，1/8<=w<=8。比值上限 8 是本轮事前选定的保守工程限幅，不来自目标结果，不是理论最优值，不扫描 4/8/16。

未触发限幅时两分区各占该通道权重总量的一半；触发限幅时只是部分平衡，不得称严格 50/50。限制的是两个区域的权重比，而非对目标面积施加约束。

L_BAL = (1/(2N)) * sum_(c,u) w_(c,u) * BCEWithLogits(z_(c,u), q_(c,u))

权重同时乘 BCE 的正负两项；所有权重 detach。**禁止用 pos_weight 替代。** 对固定 q、正 w，单点最优仍是 p=q；而只放大正项通常会把最优值改成 rho*q/(1-q+rho*q)。这里保持的是单次固定目标的最优点，不保证多步自训练无偏或校准。[S5]

## 5. 两个必要控制：不能把幅度改变当机制成功

四臂为 C、R_BAL、R_SCALE、R_SHUFFLE。在每个臂自身的当前 z、q 状态上，计算 hypothetical 的语义权重 w（不进行额外模型求导）：

d = sigmoid(z.detach()) - q.detach()
S0_c = sum_u d_cu^2
Sw_c = sum_u (w_cu*d_cu)^2

所有 S 在 CPU float64 归约，但 d 先按实际 loss dtype 计算，w 先按实际施加到 loss 的 float32 取值还原到 float64，避免审计使用理想权重、实际训练使用另一组取整权重。

**R_SCALE：通道内均匀缩放控制。** a_c=sqrt(Sw_c/S0_c)，loss 为 OD/OC 各自的 mean BCE 乘 detach(a_c) 后再等权平均。它不区分该通道中的前景或背景，仅匹配 R_BAL 的逐通道 logit 梯度范数。

**R_SHUFFLE：打乱权重位置控制。** 在每个通道的全部 H×W 位置独立打乱 w 得 w_perm；不能只在前景内/背景内打乱，因为区内权重常数，那样没有真正的控制干预。不打乱 q、z、图像、特征或 GT。Sperm_c=sum(w_perm*d)^2，b_c=sqrt(Sw_c/Sperm_c)，loss 为 mean(detach(b_c)*w_perm*BCE)。

打乱前后基础 w 的直方图不变；乘 b 后只保留相对直方图形状，绝对权重总量不再保证为 1。不要声称最终权重直方图逐值相同。

S0=0 时，正权重使 Sw=Sperm=0，两倍率均设 1；仍执行唯一 Adam，保留其历史状态效应。不设分母 epsilon，不根据效果裁倍率；非零残差却出现非正加权能量为数值错误。

用同一局部 z、q 比较，三种非基线臂的每通道输出 logit 梯度 L2 范数匹配。倍率必须 detach，否则其导数会破坏控制。**这不等于 BN 参数梯度范数、Adam 位移或整个轨迹匹配。** 各臂状态分叉后各自计算，不能用另一个臂的未来/同步状态标定本臂。

打乱使用专用 CPU torch.Generator；seed 定义为 SHA256(`R6_WEIGHT_PERM_V1|20260907|visit|channel`) 前 8 字节的大端整数截取低 63 位，visit 从 1、channel 从 0 开始。只用到达序号，无 ID、域或全局增强 RNG。

## 6. 最小分阶段矩阵：12 → 新增8，总计20

| 阶段 | 轨迹 | 意图 | 停止状态 |
|---|---:|---|---|
| I | 0 真实轨迹 | 四臂实现、CPU 数学/完整 host/继承回归、metadata dry-run | R6_IMPLEMENTATION_READY_FOR_REVIEW |
| A | 四臂×order0/order1/recurrence=12 | 比较真实完整轨迹，不是单步影子选择 | R6A_COMPLETE_ELIGIBLE_FOR_REVIEW 或 R6A_COMPLETE_NO_ADVANCE |
| B_NEW | 四臂×order2/order3=8 | 复用 A 的 12 条，补齐五流20条 | R6_EXPERIMENT_COMPLETE |

本轮明确重跑同批 C 控制；R4/R5 C 只作历史背景，不静默充当新的同期控制。重新运行的理由是当前 loss 实验需要同批完整状态路径与诊断，预算已计入。

A 中所有12条必须完成，不能按中间分数跳臂。A gate通过仍停止，B要求单独的新授权。B不得再运行A的12条；复用必须绑定原SHA、science、数据、流、生产行为指纹和标量账本。

## 7. 晋级：看完整轨迹及匹配控制，不降低R5门槛重新找正数

R6不是R5影子门控，故不使用mean(max(0,-L))作为上限或+0.20单步门槛。A的差值来自每臂自己的完整适配轨迹，最终学生原图预测，主子集不变。

### A 资源筛选条件（全部满足）

两主序等权平均：R_BAL−C>=0.50pp；R_BAL−R_SCALE>=0.20pp；R_BAL−R_SHUFFLE>=0.20pp。这三个配对在两个主序中分别都不低于0。recurrence对三个控制的差分别不低于−0.10pp。每域两主序平均R_BAL−C不低于−2.00pp。加上全部机械、身份、计数、CPU核验有效。

### B 完整开发比较

四主序平均阈值保持0.50/0.20/0.20pp，三个配对各至少3/4主序严格正向；最差同序R_BAL−C>=−0.50pp；四域各自四序平均R_BAL−C>=−2.00pp；recurrence三个配对仍>=−0.10pp。

这些是新实验的描述性资源标准，不是显著性或临床安全。若只能超过C、不能超过SCALE/SHUFFLE，不支持区域权重位置的必要性。若简单控制更好，如实记录，不从本轮重新搜索其参数或自动晋级新路线。

## 8. 机制记录与尾部

不额外反向：在实际 strong 分支记录每通道 n_fg/n_bg、rho、w_fg/w_bg、fallback；BCE两区域求和、未加权与加权残差平方贡献、S0/Sw/Sperm、a/b、权重与残差的夹角/范数关系（可由标量内积计算）。记录实际 loss、strong logit 梯度 hook 的范数、BN梯度范数及真实Adam affine位移；每次仍只有一次模型 backward。

原图 pre/q/post 可只读暂存，并在状态提交后由 evaluator 同图评分。q是诊断，不作为最终输出。评价GT只能用于评分，不能反馈权重、比值上限、分区或更新。已保存标量可验证代数关系；不宣称重建协方差、梯度张量或ASSD几何。

报告主指标、OD/OC、逐域逐序、真实接受损伤不存在（本轮每图都更新）、配对正/零/负、最差ceil(10%)、最差单内容、共同有效ASSD及undefined。不要只报被加权前景局部改善，掩盖误检或OC退化。小Drishti_GS仍占25%权重，并附内容加权和逐域敏感性分析，但不替换主指标。

## 9. 成本、授权与历史保护

| 范围 | 轨迹 | 访问 / backward / Adam | forward | VJP |
|---|---:|---:|---:|---:|
| A | 12 | 23,412 | 187,296 | 0 |
| B_NEW | 8 | 15,608 | 124,864 | 0 |
| A+B | 20 | 39,020 | 312,160 | 0 |

每个实际设备每阶段一次程序化smoke：原C4步+四臂各4步=160 forward、20 backward/Adam；这不计入正式表。只有原C与新C要求parity；新增loss从第一步就可能不同，不能要求其post等于C。新增臂的正确性来自CPU数学/梯度/完整host检查，不以toy效果改善验收。

本轮无训练前置成本；有CPU权重计算、置乱、归约和评价开销，必须实测。不宣称与C等耗时，更不能把相同网络调用数称相同FLOPs。

所有设备保持null。资源只是拟议硬上限：最多3 worker、每卡1个、每worker2 CPU线程；6h/trajectory、24h/stage wall、48h/stage累计worker、8GiB新私有输出。它们不是预计耗时。真实运行需单独绑定设备、caps、SHA、science/data/stream、stage和smoke配方。

继承R5修复：像素TP/FP/FN/TN可实现性、失败live计数及下界标记、保留原异常/首失败、只有新R6输出可invalidate、NFS/中性入口/有限监督器/清理先于落盘。不要修改R5已结束状态或旧科学文件。

## 10. 验收和最终研究边界

阶段I只用程序化图像张量和随机初始化完整ResUNet，禁CUDA初始化、真实像素/GT/checkpoint/source读取。新C与旧C多步精确匹配；新loss在统一权重时退化C；非均匀权重下与独立数学及梯度参考匹配；RNG、计数、GT时序、非法指标和失败记账均验收。

当前随附数学参考最终8/8通过，只涉及小型程序化logit/概率张量。首次检查因torch.where接收Python浮点先形成float32再赋double造成均值误差；已改为直接向float64张量写入标量，未放宽断言。首失败和最终日志均保留。这些不是R6实现/完整模型/GPU测试成绩。

若R6通过仍只得到开发阶段候选。必须冻结方案，在真正未用于本轮选择的新内容队列上确认；新顺序或新seed不能创造独立患者。没有新内容时如实限定证据，不把已暴露内容重新命名为盲测。

Science提案文件：`R6_SCIENCE_PROPOSAL.json`；原始字节SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`。这是真实提案文件摘要，不是尚未出现的实现SHA或运行receipt。

## 来源与核对边界

[S1] R5-A 结果报告，固定发布提交：
https://github.com/DLwbm123/DPA-CTTA/blob/e271098e2a12baa166fc7b77848af4a2300d9cc3/docs/results/r5a_diagnostic_v1/REPORT.md

[S2] R5-A 公开聚合与其实际分析定义：
https://github.com/DLwbm123/DPA-CTTA/blob/e271098e2a12baa166fc7b77848af4a2300d9cc3/docs/results/r5a_diagnostic_v1/PUBLIC_AGGREGATE.json
https://github.com/DLwbm123/DPA-CTTA/blob/b4b71601a5bdf87bb3a7e5d3db352610adcdff74/src/dpa_ctta/r5_update_acceptance/analyze.py

[S3] R4 完成报告及原 C 接口：
https://github.com/DLwbm123/DPA-CTTA/blob/e271098e2a12baa166fc7b77848af4a2300d9cc3/docs/results/r4_three_track_v1/REPORT.md
https://github.com/DLwbm123/DPA-CTTA/blob/e271098e2a12baa166fc7b77848af4a2300d9cc3/src/dpa_ctta/b1_host.py

[S4] B2 interval 结果：
https://github.com/DLwbm123/DPA-CTTA/blob/e271098e2a12baa166fc7b77848af4a2300d9cc3/results/b2_interval_consistency_v1/B2_EXPERIMENT_REPORT.md

[S5] PyTorch 官方 BCEWithLogitsLoss 数学定义；只用于解释公式，不授权升级软件：
https://docs.pytorch.org/docs/main/generated/torch.nn.BCEWithLogitsLoss.html

[S6] 已有相关工作：Source-Free Domain Adaptation for Medical Image Segmentation via Mutual Information Maximization and Prediction Bank，Electronics 14(18), 3656, 2025。它讨论眼底前景/背景不平衡，但包括教师和数据集级 prediction bank，不是本轮严格因果的一图一步方案。本轮不复现其完整算法、不引用其增益作为本项目预期。
https://www.mdpi.com/2079-9292/14/18/3656

本设计核对公开报告、代码和相关一手资料；未访问私有逐内容账本、服务器、真实数据或给定 checkpoint。数学参考只作程序化张量检查，不能替代阶段 I 完整 host 验收。
