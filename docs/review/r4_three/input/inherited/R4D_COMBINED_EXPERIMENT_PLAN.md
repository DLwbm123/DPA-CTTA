# R4D 联合实验计划
## 方向 A：教师时间尺度 × 区域 PCA；方向 B：卷积核空间细节保持的几何适配

**状态：方案提案。尚未完成项目代码实现、尚未通过实现 review、未授权 GPU 或后台执行。**

本文件取代尚未执行的原 R4 排程，形成一次联合实验。原 R4 的六个算法臂不变，只新增一个独立方法方向及三个控制。不是先运行原 R4 的 30 条轨迹，再重复运行本文件的 50 条。

## 0. 一次回答两个问题，不将两条路线提前叠加

**方向 A（延续性研究）：** 当前学生、EMA 教师、固定初始教师三种监督时间尺度，是否影响原区域 PCA 的有效性？重点不是“EMA 本身新不新”，而是区域 memory 能否提供超出教师本身的增量。

**方向 B（方法学候选）：** 在不改变源训练、没有源数据的前提下，是否可以从给定卷积核中划分“允许适配的空间均值项”与“保持不动的零均值空间细节项”，再以受约束的通道几何更新前者？暂记 **KDG：Kernel-Detail-preserving Geometric adaptation**。

KDG 是一个新实验设计，不是已确认首创、已验证有效或已达 TMI 标准的方法。它借鉴 PAID 的角度保持思想，但既不复制其全套算法，也不使用其预提取源统计。两条路线共享同批 C 基线、同一 checkpoint、输入和评价；B 不接 EMA、RP、密度教师、图、router 或恢复控制。本批不运行 KDG+MT/RP。

## 1. 依据与研究边界

R3 的 85 条轨迹已完成。五个主候选中，M_TRANSPORT 的约 +0.166 pp 被原 RP 和 M_IDPOST 覆盖；U_PCA、G_PCA 未超过各自简单控制，T_LR、S_JOINT 出现较明显退化。这些是冻结实现和开发流的证据，不是方法家族普遍无效。[S1]

因此本批不继续修改 R3 的密度温度、VJP 行数、专家容量、迁移矩阵或图参数。A 保留原 RP 的简单公式，研究其上游监督；B 改变适配参数化，不再用伪标签驱动的 PCA 来决定保护方向。

用户的 CTTA 文献总结 A16 将 PAID 定位为“幅度/方向分解、角度保持的参数几何”路线，同时提醒参数角度不等于解剖结构。[S2] 补查 PAID v2 §3.3 可见完整方法需要从 500 张源图像取得统计。[S3] **这些统计不在普通 checkpoint 中，不能使用。** 本批只借鉴可由权重本身定义的几何约束，目标函数仍为 C 的无标签一致性。

本项目固定 ResUNet 的解码器主要使用 1×1 卷积和转置卷积，不应臆造两个不存在的 decoder 3×3 refine 层。已核对的实际参数入口是 `res.conv1` 和 `res.layer1.2.conv2`，见 §5。[S4]

## 2. 固定资产、数据和合法信息

- 基础结果提交：`2f90a6a0933cec3a25337a0d46ee632f8d772840`。
- 继承实际 R3 运行修补：`185fd440b16920f367c1f5e3096ab495bd85c0ec`。不要退回此前进程审计修补前的版本。
- 新分支建议：`experiment/r4-dual-teacher-kernel-v1`。不改 main、历史配置或旧结果。
- 只有给定 Fundus ResUNet34 checkpoint；禁止源 RGB/mask、source query、源代理/原型/额外校准统计、源端训练或外部预训练模型。
- 原登记：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`。
- 回访流摘要：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。
- 全流 1,951 组；主要子集 remaining_dev 1,695 组。它们已用于开发，不称盲测。
- seed=20260907。所有臂独立从同一源 checkpoint 开始；跨域不 reset，不读取未来图像、域标签或样本身份来做适配。
- 原预处理、512×512 输入、OD/OC 两个独立 sigmoid 通道、0.5 分割阈值、GT 解释与 evaluator 不变。
- C 的 BN 当前输入统计政策不变；41 层 BN、19,136 个 affine 标量仍在线更新。源 running-stat 推理 N 不混入任何本批教师。
- 学生 Adam：lr=1e-4，betas=(0.9,0.999)，eps=1e-8，wd=0，float32，AMP off。每图一次 backward 和一次 Adam。
- K 系列在同一 Adam 调用中额外更新 §5 的 kernel 坐标；源存储权重不改写，但有效卷积核会改变。不得宣称其实际预测网络完全冻结。
- 所有臂最终输出均为更新后的学生原图预测；无输出平均、教师替代输出或额外后处理。

### 五条流

| id | 顺序/构造 | 用途 |
|---|---|---|
| 0 | REFUGE → ORIGA → REFUGE_Valid → Drishti_GS | 主序 |
| 1 | Drishti_GS → REFUGE_Valid → ORIGA → REFUGE | 主序 |
| 2 | ORIGA → Drishti_GS → REFUGE → REFUGE_Valid | 主序 |
| 3 | REFUGE_Valid → REFUGE → Drishti_GS → ORIGA | 主序 |
| 4 | R3 原 64 张域块轮换回访流，域内顺序不变 | 单独压力测试 |

每条流内每组只出现一次。回访的是环境，不重复回放同一病例。四主序与回访结果不合成五序均值；同内容不同顺序不是独立患者重复。

## 3. 合并矩阵：10 臂，50 条轨迹

| 臂 | 方向 | 教师 | RP | 新卷积参数化 | 前向/图 |
|---|---|---|---:|---|---:|
| C | A/共同基线 | 当前学生 | 否 | 无 | 8 |
| RP | A/共同参照 | 当前学生 | 是 | 无 | 8 |
| MT | A | EMA affine, m=0.99 | 否 | 无 | 8 |
| MT_RP | A | EMA affine, m=0.99 | 是 | 无 | 9 |
| FT | A | 固定初始 affine | 否 | 无 | 8 |
| FT_RP | A | 固定初始 affine | 是 | 无 | 9 |
| **KDG** | **B 主候选** | 当前 KDG 学生 | 否 | DC 分量正交旋转+正幅度；细节固定 | 8 |
| K_ALL | B 控制 | 当前本臂学生 | 否 | 相同几何作用于全部空间系数 | 8 |
| K_MAG | B 控制 | 当前本臂学生 | 否 | 只缩放 DC，不旋转 | 8 |
| K_FREE | B 控制 | 当前本臂学生 | 否 | DC 自由加性更新 | 8 |

C/RP 本批各跑五流一次，同时服务两方向，不重复计算。原因仍为需要同口径教师与采样质量标量，而不是制造额外独立重复。两方向全部完成后统一选择；A 的效果不决定 B 是否运行，反之亦然。

## 4. 方向 A：保留原 R4 科学定义

### 4.1 教师与学生

学生记作 f_(w0,ψ)，教师记作 f_(w0,ψbar)。教师与学生都使用 C 的当前输入统计。教师无 optimizer，全部参数不参与反向。

q_t = sg[(1/6) Σ_j T_j^(-1) sigmoid(f_(w0,ψbar_(t-1))(T_j x_t))]。

六个原有几何视图分别前向；不堆成 batch=6；不锐化、不混源概率、不做 hard target。

- C/RP：ψbar_(t-1)=ψ_(t-1)。
- MT/MT_RP：ψbar_0=ψ0；学生最终输出、memory commit 后，ψbar_t=0.99ψbar_(t-1)+0.01ψ_t。
- FT/FT_RP：ψbar_t=ψ0。

FT 是“初始 C0 前向规则的六视图教师”，不是 No Adapt source-BN。EMA 不加 warm-up、偏置校正、域相关动量或阈值门控。

学生目标：

L_A = BCEWithLogits(z_student_strong, q_t) + λ_RP L_RP。

无 RP 时 λ_RP=0，有 RP 时固定为 0.05。

### 4.2 RP 的公式和坐标不变

复用原绝对重建函数：

L_RP = Avg_(r,u) ||(I-U_r U_r^T)(v_s-μ_r)||² /
                   (sg[||v_s-μ_r||²]+ε)。

保留原函数的 epsilon、归约与空区域处理。32 维 `seg_head` 输入；32×32 网格；rank≤8；OD_bg/OD_fg/OC_bg/OC_fg 四 bank；每区域每图最多 32 个 token；16 张贡献图/128 个向量后允许使用；每 16 张非空贡献刷新；累计 Welford，无遗忘、迁移或路由。

BCE q、hard 分区、可靠性和实际待写入位置均来自本臂教师。可靠性仍为 q≤0.1 或 q≥0.9 且六视图 hard 一致。可靠性只控制写入，不能暗中改变原 RP loss 的全部区域归约。

**存储特征来自学生更新前原图，不是教师或 post 特征。** MT_RP/FT_RP 额外做一次学生原图无梯度前向取得 f0_student，所以是 9 次前向。其余 A 臂 8 次。

### 4.3 A 时序

当前 RGB → 过去 memory 快照 → 教师六视图 q → 必要的学生 pre 特征/待写入选择 → 学生 strong loss → 一次 Adam → 学生原图最终预测 → memory 合并 pre 特征 → EMA 状态更新 → evaluator 才读取 mask。

不能用当前图更新后的 EMA 重新产生本图目标，不能将本图提前并入自己的 PCA。

## 5. 方向 B：KDG 卷积核空间细节保持的几何适配

### 5.1 假设与真实创新边界

假设：除 BN 通道缩放/偏置之外，还需要有限的跨通道响应调整；但不必自由改写卷积核中对空间差分的响应。使用源权重可解析的结构来限制这部分适配，可能优于伪标签决定的保护方向。

原 C 已经冻结全部卷积权重；所以 KDG 相对 C 是**受限地增加卷积可塑性**，不是比 C 冻结更多内容。它要证明这种新增自由度值得开放，而不只是证明自己能保持已规定的 kernel 不变量。

这不是“低频一定是风格，高频一定是解剖”的定理。空间 DC kernel 也不是严格截止频率的图像低通滤波器。这里的可证明性质仅是**选定卷积核的零均值空间部分保持不变**，不等于整网边界、拓扑或旧域准确率保持。

### 5.2 两个预先固定的实际入口

| 路径 | 源权重 OIHW | stride/padding | 理由 |
|---|---|---|---|
| `res.conv1` | [64,3,7,7] | 2 / 3 | 最早局部 RGB 响应 |
| `res.layer1.2.conv2` | [64,64,3,3] | 1 / 1 | 首个残差 stage 的末端空间卷积，影响高分辨率 skip |

均 groups=1、bias=False。路径与结构已从固定依赖源码核对。[S4] 实现时程序化枚举确认；若不匹配，不按得分改选层，不臆造等价路径。只报告真实结构差异并保留其他可实现工作。

本批不扫描插入层，不触碰 1×1 segmentation head，不更换 BN 统计，不加入新的分割头或边界网络。

### 5.3 空间分解

设一个 kernel 为 W0[o,i,s]，s 遍历 K=kh×kw 个空间位置。定义单位 DC 基：

b(s)=1/sqrt(K)，A0[o,i]=Σ_s W0[o,i,s] b(s)，H0=W0-A0⊗b。

于是 Σ_s H0[o,i,s]b(s)=0，W0=A0⊗b+H0。A0 是 O×I 通道矩阵；H0 是本计划所称的“空间细节项”，不是人为解剖标签。

### 5.4 KDG 有效权重

新增 I(I−1)/2 个坐标 u，填入 S 的严格上三角，下三角为其负值，对角为 0。令：

Q = solve(I + S/2, I − S/2)，D=diag(exp(a))。

u 和每个输出通道的 a 均初始化为 0，故 Q=I、D=I。不要用零态 Python 分支直接返回原权重，否则会切断初始梯度。

A_t = D_t A0 Q_t，W_eff = W0 + (A_t−A0)⊗b。

每次前向生成有效 kernel，不改写底层源 `.weight`。小矩阵运算 float64、同模型设备；有效 kernel 转回 float32，保留经过 cast 的梯度。使用残差形式，避免 H0+A_t⊗b 重建在初始化时改变源 kernel。

**不对 gain 进行隐含 clipping，不做额外正则或恢复。** exp(a)>0，但不宣称幅度有界。非有限值按已有工程错误处理，不临时夹断后继续。

### 5.5 两个局部代数性质

1. **空间细节保持：** H_eff=H0，kernel 任意两个空间位置的差保持不变；ΔW 只存在于 spatial-DC 模式。
2. **DC 行夹角保持：** 对非零 A0 行，cos(A_t[o],A_t[p])=cos(A0[o],A0[p])，因为右乘 Q 正交且每行仅有正缩放。

不得把第 2 条扩大为“整个 flattened kernel 的夹角保持”：KDG 只保护 DC 矩阵的角度。K_ALL 才有完整 kernel 行角度保持性质。

对同一个局部输入 patch x，新增输出为 Δy_o=Σ_i ΔA_oi Σ_s b(s)x_i(s)。因此，对每输入通道都具有零空间均值的 patch 扰动 δx，(W_eff−W0)δx=0。这个恒等式条件是同一个局部输入；上游 BN 或其他层改变输入后，不可推成整网函数不变。

### 5.6 三个控制

**K_ALL：** 使用与 KDG 完全相同的 u/a 数量，在每个空间位置上执行 W_eff[o,:,s]=exp(a_o) W0[o,:,s]Q。不保留 H0；用于检验“只开放 DC、保留细节”的必要性。这是 PAID 思想的受控卷积化对照，不是完整 PAID。

**K_MAG：** 只学习 a，Q 永远为 I，其余为 KDG；检验是否只需要均值项幅度调整，而不需要跨通道方向学习。

**K_FREE：** 学习 E∈R^(O×I)，E0=0；A_t=A0+(||A0[o,:]||/sqrt(I)) E[o,:]。不额外学习 gain，不约束角度；仍只更新 DC。检验简单的通道均值项自由修正是否已经足够。源 DC 零行在所有 DC 臂保持零，不添加人工最小源范数或随机源方向。

KDG 与 K_ALL 参数坐标数相同；K_MAG 更少，K_FREE 更多，学习率相同不意味着函数空间步长相同；后续当前统计 BN 还可能抵消 K_ALL 的部分整通道幅度变化，因此等参数坐标不等于等有效自由度。必须报告实际有效 kernel 位移，不能伪称所有控制等参数/等 FLOPs，或从 K_FREE 一次失败直接证明角度约束的因果必要性。

### 5.7 参数与优化

| 臂 | 新参数 | 原 BN affine | 合计可学习标量 |
|---|---:|---:|---:|
| KDG | (3+64)+(2016+64)=2,147 | 19,136 | 21,283 |
| K_ALL | 2,147 | 19,136 | 21,283 |
| K_MAG | 64+64=128 | 19,136 | 19,264 |
| K_FREE | 64×3+64×64=4,288 | 19,136 | 23,424 |

先按原 C 配置 BN，再安装新参数化并组装确定顺序的 optimizer 参数列表。一个 Adam 同时更新 BN 与新增坐标，全部 lr=1e-4、其余设置相同。原始源卷积、bias、分类器和其他非 BN 参数全部不被 optimizer 拥有。底层源 tensor 不变与有效权重变化分别记录。

B 的六视图目标来自本臂当前更新前模型（包含该臂当前有效 kernel），strong 分支仍只优化原 BCE。无 EMA、无 PCA、无额外 loss、无 Jacobian。

B 每图 8 次网络前向、一次 backward/Adam。虽然调用次数不增加，参数化矩阵求解和卷积 weight-gradient 会增加工作量；不得声称与 C 等 FLOPs 或无额外耗时。

### 5.8 B 时序与冷启动测试

当前 RGB → 当前有效 kernel 下六视图 q → strong BCE → 一次 Adam 更新 BN+kernel 坐标 → 新有效 kernel 下学生原图预测 → 固定在线状态 → evaluator 读取 mask。

新增参数从第一张图即参与更新，无 ready/warm-up。**只要求零初始化的更新前函数与 C 一致；第一步更新后的 K 输出不应被强制等于 C。** 另在新坐标强制冻结为零的程序化测试中验证完整 C 轨迹退化对齐。不能照搬 R3“未 ready 的所有新臂首步 post 必须等于 C”的断言。

## 6. 共同评价与辅助质量证据

主评价严格继承：每图 OD/OC 平均 → 域内均值 → 四域等权 → 四主序等权。记录所有既有 subset，remaining_dev 为主。输出每域/序 OD、OC、macro、配对正负比例、中位数、最差 10%、最差同序差以及共同有效 ASSD；undefined 不填零。REFUGE_Valid OC 单列，不由小域收益覆盖。

### 教师质量（全部 10 臂）

只利用正常前向已经得到的 q。当前最终预测和全部在线状态 commit 后，在 evaluator 用同一当前 mask 算教师 OD/OC Dice、Brier、GT 前景/背景分项与分母。教师指标是辅助，不另算一条评分轨迹，不替代学生主输出。q 评价后立即释放，不长期存储 dense 图。

### Memory 写入质量（仅 RP、MT_RP、FT_RP）

使用实际写入且已去除零向量的采样 token ID。网格位置沿用 `r1.region_memory.grid` 的中心索引，不用 nearest resize 另创坐标；512 栅格中心为 8,24,…,504。[S5] 在既有 evaluator 将 mask 转成与预测相同栅格后按相同坐标评分。报告 OD/OC 前景/背景正确及错误 token 数、分母，不将多个 token 视为独立患者。

无 RP 臂记为 NOT_APPLICABLE，不伪造 shadow memory 或填成“0 错误”。接口只能将当前 detached q/选择索引单向送 evaluator，GT 和评价结果不得进入 host、teacher、optimizer、memory 或 RNG。

### K 几何与成本证据

按层记录源 weight 不变性、实际 DC/detail 位移、Q 正交误差、非零 DC 行的归一化 Gram 差、gain 范围、有效 kernel 位移和 adapter 坐标更新范数。K_ALL 同时记录 full-kernel Gram；K_FREE 不要求满足角度不变量。统计可由小矩阵完成，不重新跑模型，不做 mask-based 风险门控。

全部臂记录网络前向、loss backward、Adam、EMA 次数、kernel 构造/求解时间、host 时间、实际 peak allocated 与保留状态；两个核函数运算不能计成两个网络前向。参考实现的完整 buffer 副本若被保留，也应计入实际内存，不仅报告可学习参数。

局部代数测试通过不是临床边界保持证据。回访流不是固定旧域 probe 的前后遗忘矩阵，本批不作“零遗忘”结论。

## 7. 完整计算预算

| 部分 | 臂数 | 流数 | 完整轨迹 | 正式评分/Adam/backward | 网络前向 |
|---|---:|---:|---:|---:|---:|
| A：原 R4 教师×RP | 6 | 5 | 30 | 58,530 | 487,750 |
| B：KDG 及控制 | 4 | 5 | 20 | 39,020 | 312,160 |
| **联合总计** | **10** | **5** | **50** | **97,550** | **799,910** |

主序 40 条、78,040 记录；回访 10 条、19,510 记录。源训练、DD、VJP 均为 0。A 两个 MT 臂无梯度 EMA 更新正式共 19,510 次，独立计数，不冒充 optimizer 更新。

每实际 GPU 一次 smoke：10 臂各 2 次＋旧 C/RP 各 2 次，共 **24 Adam/backward、196 forward**。A 的 RP 臂第二次可用程序化 ready memory；B 第二次测试已有 adapter 状态，不加人为效果阈值。Smoke 不进入主表。

| GPU 数 | 含 smoke Adam/backward | 含 smoke forward |
|---:|---:|---:|
| 1 | 97,574 | 800,106 |
| 2 | 97,598 | 800,302 |
| 3 | 97,622 | 800,498 |

资源上限：最多 3 个独立 worker，每卡最多一个本作业 worker，每 worker 2 个 CPU 线程；单轨迹 6 小时、墙钟 72 小时、合计 active-worker 96 小时、新私有输出 6 GiB。这些是上限不是耗时预测。按既有 `(arm_index+order_index)%workers` 轮换，不固定方法绑定设备。

偏好 GPU 6/7 不构成权限。实际设备、最大并发和后台允许与否在实现审阅通过后重新授权。程序只能清理本次自有进程，不能停止他人任务。工程故障遵循原有限监督器，不自动重试、追加组合或超预算扩展；失败前缀和额外 smoke 如实单列。

## 8. 两条路线独立选择，不为保留 PCA 或 KDG 叙事而改判据

A 必报：MT−C、FT−C；RP−C、MT_RP−MT、FT_RP−FT；MT_RP−RP、FT_RP−RP；MT−FT 与 MT_RP−FT_RP。描述性交互 `(MT_RP−MT)−(RP−C)` 不称显著协同。

B 必报：KDG−C、KDG−K_ALL、KDG−K_MAG、KDG−K_FREE；K_ALL/K_MAG/K_FREE 各自−C。A/B 的总体排名可列，但不将不匹配的跨方向差解释为某个机制的因果贡献。

沿用筛选参考：相对 C 四主序平均至少 +0.5 pp，至少 3/4 主序改善。域均值低于 C 2 pp 或最差同序低于 C 0.5 pp 要明显警示，不靠任务均值掩盖。这不是统计、临床或 TMI 阈值。

- MT 改善但 MT_RP 无增量：保留简单 MT，不强留 PCA。
- 固定教师更好：不主张 EMA 必要。
- K_MAG 与 KDG 相当/更好：不主张旋转必要。
- K_ALL 更好：不主张固定细节的价值。
- K_FREE 更好：保留自由 DC 控制的有效性，不主张角度约束必要。
- KDG 要作为完整机制候选，除净增益外，应对关键匹配控制表现出跨序一致的优势；不以极小均值差作成立证明。
- 两方向均有信号：本批停止后再讨论新组合或未见验证，不自动跑 KDG+MT_RP。
- 两方向均无清楚收益：保留 C/已验证简单参照，关闭本批固定配置，不搜索层、学习率、教师动量和 RP 系数直到正结果。

这些仍是固定 checkpoint、已暴露内容的开发证据。真正正结果之后才考虑另一个已给定 checkpoint 或未参与设计的数据；本文件不预授权后续实验，也不要求重新训练源模型。

## 9. 实现、CPU 检查、GitHub 交付

### 实现范围

新增 `src/dpa_ctta/r4_dual/`，建议分为 teacher、kernel_geometry、host、evaluation_aux、plan、run/analyze。C/RP 精确委托或严格对齐已审 R1；不要修改旧科学文件来让 50-job 排程通过旧 85-job 验证。新薄适配层自行绑定 10 臂/50 轨迹，复用数据读取、NFS、cuBLAS 阶段政策、有限监督与原子发布，不重写共享基础设施。

核参数化应维持原源 tensor 身份与名称映射，明确新增坐标的所有权。可以继承 Conv2d 或建立薄 wrapper，但不得 in-place 改 `.weight.data`、共享学生/教师的可变参数、修改既有依赖源码。调用 `_conv_forward` 或等价实现时保留 stride/padding/dilation/groups/padding_mode，不改变算子几何。

附带 `reference/kernel_geometry.py` 是可运行的数学核参考，不是完成的 host。它用于验证公式与避免零初始化断梯度，不要求原样复制其 buffer 布局。真实 host 的权重加载、RNG、计数、tensor 生命周期仍需 Codex 实现并审阅。

### 必要的新测试

A 保留原 R4 的 EMA/固定教师、当前统计、学生 pre 坐标、时序、辅助评价隔离、m=0 退化与 8/9 前向检查。

B 测试：零初始化 pre 函数等于原 C、零初始坐标梯度可达；source weights 始终不变；有效卷积变化；DC/detail 分解、正交/角度恒等式与明确不成立的 K_ALL 细节不变量；有限差分；零源 DC 行行为；参数数量、无别名和 optimizer 所有权；强制冻结新增坐标时回到 C；当前图最终输出前模型状态正确；8 前向/1 backward/1 Adam。

验证 50-job 矩阵、A 六臂科学语义与原 R4 一致、主/回访分开。保留已有共享层回归，不按测试数量另建门槛。第一失败日志保留，明确是程序化 CPU/随机权重；已有 EIO 历史未知原因仍保留，不凭多次绿灯改写为根因解决。

### 交付与阶段

阶段 I 只实现、程序化 CPU 测试、元数据 dry-run 和公开去身份材料。禁止 GPU、真实目标 RGB/mask、源数据、正式实验或后台任务。已知私有路径优先从既有配置解析，缺少真实路径不阻止其余实现。

推送完整 implementation SHA、新 science 原始字节 SHA256、窄 patch、真实 CPU 日志、50-job dry-run、teacher/kernel/memory/Adam/RNG 生命周期、来源与本设计差异、实际参数枚举、预算、未运行项及 REVIEW_INDEX。状态：**R4D_IMPLEMENTATION_READY_FOR_REVIEW**，随后停止。

阶段 II 是外部 review 固定提交及科学配置后，再明确用户资源授权。一次执行两个方向所有轨迹，不能根据先跑结果跳过另一方向或改配方。代码 review 不评价方法是否先涨分。

## 10. 来源和证据性质

[S1] DPA-CTTA 结果提交 `2f90a6a0933cec3a25337a0d46ee632f8d772840`，`docs/results/r3_five_frameworks_v1/REPORT.md`。实际运行代码为 `185fd440b16920f367c1f5e3096ab495bd85c0ec`。

[S2] 用户附件《CTTA 2024–2026 文献检索与研究综述》，A16（PAID）、第 1.3 节（摘要证据等级）；《Zotero ICLR 2026 CL 论文总结》阅读说明。摘要级导读不代表全部论文全文或最新录用核验。

[S3] Wang et al., PAID: Pairwise Angular-Invariant Decomposition for Continual Test-Time Adaptation, arXiv:2506.02453v2，§3.2（Householder 参数几何）、§3.3（500 源样本统计）。https://arxiv.org/html/2506.02453v2 。本 KDG 的卷积空间 DC/detail 分解、Cayley 实现、两层入口、四臂对照和 C 目标是新提案；不称原 PAID 复现。

[S4] DLwbm123/CTTA 固定依赖 `dbff0d985c6c95345d9fb78f5b1daef57b392564`，`VPTTA/OPTIC/networks/ResUnet.py` 与 `networks/resnet.py`。

[S5] DPA-CTTA 固定运行提交 `185fd440...`，`src/dpa_ctta/r1/region_memory.py` 的 `grid/tokens/Memory`。

[S6] 先前提供的 `R4_EXPERIMENT_PLAN.md`（SHA256 `121b9cf03802f9b6c86e180002f64ad45f757f3878cc8b431d7d24ef680e79be`）与 `R4_SCIENCE_PROPOSAL.json`（`22e59034737ab37404b4647edb6fd7186ec19613ddb0666ddb5258877a387ac5`）；本包保留 A 六臂算法，只统一扩展排程。

[S7] Tarvainen & Valpola, Mean Teacher, arXiv:1703.01780；Wang et al., Continual Test-Time Domain Adaptation, CVPR 2022 / arXiv:2203.13591。EMA 和增强平均是已有机制；本批当前统计固定教师与学生坐标 RP 是受控实验设置。

本包数学参考已执行 12 项程序化 CPU 检查，详见附带真实日志；它不包含完整模型、真实 checkpoint、GPU 或目标数据的运行证据，不预先给 Codex 实现盖章。
