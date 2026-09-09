# B1：结束当前 DD／融合扩展，开展匹配的医学 TTA 更新范式对照

**研究优先级：有效性第一，创新性第二，SOTA 第三。**

本文件是一轮新的有限实验建议，可作为提交给 Codex 的任务说明。它不重启 M1–M4、SA 或 P2，不声称新实验已获独立评审通过。提交本任务时需明确本轮执行与 GPU 使用授权；不得沿用已结束任务的 receipt。

## 1. P2 的结论与本轮目的

P2 已完成 3,753 个内容组、52,542 条评分、22,518 次正式在线更新。实际执行提交为 `d3ee6901379be293f47abd1687808caaf5b04266`，结果发布提交为 `2fcbc0a69645e34da42b97830935f0e290e488aa`。

主要子集 remaining_dev：

| 任务／顺序 | A−N | EA−A | O2−A | EO2−EA | ED4−EA |
|---|---:|---:|---:|---:|---:|
| Fundus／0 | +6.058697 | −0.832674 | −1.367558 | −0.614094 | −0.418791 |
| Fundus／1 | +6.342363 | −0.729359 | +0.880559 | +1.126876 | +0.225092 |
| Polyp／0 | +2.918063 | +0.260689 | −0.694424 | −0.339991 | −0.088949 |
| Polyp／1 | +2.594980 | +0.150959 | −1.007334 | −0.581998 | −0.577527 |

单位为 Dice 百分点，来自 P2 公开报告。部分展示值末位可能受舍入影响，最终比较以私有逐图标量的重算值为准。

这些证据支持：保留原生 A 作为有效参考，结束当前固定 K=4 代理目标、SA 和固定融合的连续修改；不支持所有 TTA／DD 无效。原 source-clean 与 target fresh-host 表也不是目标适配后的遗忘测量。

**本轮不再设计一个新的附加 loss。直接检验另一种完整、已发表的适配范式：在相同源权重和目标流上，更新网络 BN 的 affine 参数、使用多视图一致性与 GraTa 的梯度对齐更新，是否比当前低频 prompt 路线更有用。**

本轮不是新方法创新声明，也不是声称“参数维度导致旧方法失败”的因果诊断。与 A 的比较会同时改变更新参数、目标函数、归一化规则及状态机制；它是算法整体的实际效能比较。G 与 C 的比较更直接检验发布版 GraTa 更新规则的增量，但不单独拆分梯度对齐和动态学习率的作用。

## 2. 固定源码与范围

- 项目基线：`DLwbm123/DPA-CTTA@2fcbc0a69645e34da42b97830935f0e290e488aa`。
- 新分支建议：`experiment/b1-grata-matched-baselines-v1`。
- 已有模型依赖：`DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`。
- GraTa 官方依赖：`Chen-Ziyang/GraTa@33ae20d664f305af34739ec54a5bec7da53ffa0b`。
- 先读官方 `GraTa-master/TTA.py`、`custom_optimizers/grata.py`、`dataloaders/aug/`、`networks/ResUnet.py`、`run.sh`，以及当前项目 P2 的 source-only model、预处理和评分入口。
- 不运行官方 `run.sh` 或完整 `TTA.py`。它们含作者路径、GPU 设置、读取 target mask 的 loader、不同 checkpoint 入口以及域间重建模型的外层循环。只接入需要的更新和增强函数，隔离其 IO、环境变量和评估路径。
- 保留第三方许可证和来源，不改写旧提交、旧配置和固定依赖。
- **只做 Fundus。** GraTa 发布入口直接对应 ResUNet34 和 OD/OC 双通道；本轮不把 Polyp/PraNet 移植伪装成现成官方复现。选择由实现／协议匹配决定，不因 Fundus 新结果好坏改变范围。Polyp 的现有结果归档，不开启新的融合搜索。

## 3. 为什么必须叫 G-CTTA，而不是“完整论文复现”

官方 `TTA.py` 在每个目标域创建新的 `TrainTTA`，而项目要求跨域持续适配。本轮只在整条任务流开始时重置，跨域保留 BN affine 参数及 Adam 状态。

因此：

- 命名 `G-CTTA`：GraTa 发布更新规则在本项目同权重、连续流协议下的移植。
- 命名 `C-CTTA`：同参数化、同一致性目标、同增强与连续历史的固定学习率对照。
- 不与原论文表格数值直接相减；不称原作者完整协议复现、SOTA 或首次创新。
- 若代码与论文存在差异，以本轮明确冻结的发布代码为准，并在报告中记录差异，不暗中“纠正”方法。

## 4. 新实验臂与旧对照

本轮只新增两条适配轨迹：

| 臂 | 可更新参数 | 更新目标／规则 | 学习率 |
|---|---|---|---|
| C-CTTA | 分割前向实际使用的全部标准 BN weight/bias | 官方多视图一致性目标；直接一次 Adam | 固定 1e−4 |
| G-CTTA | 与 C 完全相同 | 官方 aux=`ent`、pse=`consis`、临时参数扰动与恢复、动态 LR；一次 base Adam | 初始 1e−4，官方 cosine 映射 |

复用 P2 同一完整 Fundus 流上的 N、A、EA、O2、D4 作为历史冻结对照。若本地已经有本轮所要求的相同权重／流／代码协议的 C/G 完整运行，不重复启动；不能用另一个 checkpoint 或逐域 reset 的旧报告充当本轮结果。

不新增以下实验：新 DD、SA、EMA teacher、ensemble-C/G、entropy 系数搜索、额外数据增强搜索、norm 插值扫描、router、域名选择器、新 source 训练、Polyp 移植。

## 5. 源权重和模型接口

1. 使用 P2 封存的同一个 RIM_ONE_r3 checkpoint，不下载作者另一组 source 权重，不为 GraTa 重训 backbone。
2. 用项目现有标准 source-BN 分割模型作为规范分割前向，包装成官方优化器需要的 `(logits, feature)` 接口。`ent/consis` 路径不需要训练 reconstruction、rotation、denoise、super-resolution 辅助 head，不能补随机 head 后使其影响输出。
3. 在启用 C/G 的适配配置之前，标准 BN eval 的图像预处理和 logits 应与 N 在预登记容差内一致。
4. 只接受严格完整的分割子网络状态映射；有名称差异时给出显式一一映射及完整覆盖。禁止用任意 `strict=False` 掩盖未加载的分割参数。
5. 整个 source 文件不改变。在线允许改变的仅为已登记的 BN affine 参数。其他卷积、decoder、segmentation head 等参数必须不变。
6. 官方配置把标准 BN 切为当前 batch 统计，`track_running_stats=False`、running stats=None；C/G 采用同一规则。它是本方法的公开组成部分，不等同于重启 H1/H2 的 norm 专项研究。
7. 记录真实可训练参数数量与层清单，不预先编造数量，不要求它等于 VPTTA 的 prompt 参数量。

## 6. 准确冻结发布版目标

### 6.1 C-CTTA

在当前输入 x 上，按官方 `cal_consis_loss`：原图加五个固定几何变换产生六个无梯度预测；逆变换到同一网格，sigmoid 后平均为软目标。当前图像的一次官方强外观增强作为可微输入，以 BCEWithLogits 对齐该软目标，然后执行一次 Adam，最后在原图上返回更新后的预测。

六视图来自当前同一张图，不是读取六张未来图像。几何逆变换、增强参数、归一化次序全部沿用官方代码，不另外加入 target 风格 donor。

### 6.2 G-CTTA

按官方 `GraTa.step`：

1. 计算当前图的 aux=`ent` 梯度。
2. 临时将 BN affine 参数按官方规则扰动（发布代码是减去 auxiliary gradient，不自行乘上另一学习率）。
3. 在扰动后的模型上计算同一 `consis` 目标及 pseudo gradient。
4. 计算两种梯度 cosine，恢复原参数值。
5. 用官方 `custom_activation(cosine)=(cosine+1)^2/4` 乘初始学习率，使用 pseudo gradient 执行一次 base Adam。
6. 在原图上返回最终预测。

发布版 `cal_ent_loss` 使用 `−sum_c p_c log(p_c+1e−6)`，并不是完整独立 Bernoulli 熵。**保留发布式并明确标注**；不能私自补 `(1−p)log(1−p)` 后继续叫同一个发布方法，也不能把本轮包装成改进熵公式。

C/G 共用：Adam betas=(0.9,0.999)、eps=1e−8、weight_decay=0、batch_size=1，初始 lr=1e−4。aux/pse 分别 ent/consis，无其他 auxiliary heads。G 不强行使用 VPTTA 的 lr=0.05。

合法的零梯度或 G 的动态学习率恰为零，是需要记录的算法行为，不是必须人为构造正梯度才能通过的性能条件。非有限数值与实现错误单独处理。

## 7. 数据、标签和连续协议

完全复用 P2 的 1,951 个 Fundus 内容组、四域、三个子集标签及两个域序：

- Order-0：REFUGE → ORIGA → REFUGE_Valid → Drishti_GS。
- Order-1：反转域块，域内 manifest 顺序不变。
- remaining_dev=1,695；legacy_dev=128；p1_extension_dev=128。
- 不筛选新样本、不因旧结果好坏选择域，不重建 target split。
- 每方法、每顺序分别从相同 source 状态开始；跨域不 reset，不读取域切换信号。
- C/G 的 teacher-like 弱增强软目标来自当前模型，不另加 P1 的固定 source teacher 或平均输出。
- 在线函数只接收当前图像；不得把包含真实 mask 的完整官方 data dictionary 传入 optimizer。数据增强字典仅含图像和必要非标签字段。
- 真实 mask 由 evaluator 在最终预测固定后读取。禁用、隔离官方 `cal_groundtruth_loss` 与其他不需要的标签入口。
- Fundus mask 继续使用项目的 OD/OC 独立二值通道与 nearest resizing；主指标和阈值不改。
- 所有目标内容均已用于开发；本轮不能把 remaining_dev 称为盲测，也不能把两个顺序称为独立患者重复实验。

## 8. 随机性与公平性

每条方法／顺序开始前 seed_all(20260907)，固定 Python、NumPy、Torch、CUDA 的使用方式。C/G 使用相同的官方增强机制与初始随机种子；单独隔离各自 RNG，避免共享调度造成互相消耗随机数。

不根据分数重抽强增强，不尝试多组 seed 取最好。两种顺序使用不同内容位置，不强行宣称增强序列与所有内容逐一匹配；记录配置和所需复算证据。

沿用 P2 正式计算环境。配对 smoke 可以复用既有严格确定性上下文并恢复；不能将 smoke 的严格确定性结果泛化为正式全 CUDA 路径逐比特确定。

## 9. 一次实现检查，一次 smoke，一次完整比较

### CPU／程序化检查

只增加本次路径必要的回归，旧测试不删除。至少检查：

- 标签改变不会影响 C/G 预测与参数历史；
- 启用适配前 source 模型与 N 一致；
- BN affine 参数清单和 optimizer ownership 正确；
- 六个几何视图及逆变换一致；
- C 使用官方一致性值和梯度；G 的扰动、恢复、cosine、LR、base optimizer step 顺序一致；
- 对比 G 是发布式，不把两个梯度直接相加冒充发布版；
- convolution/head 参数和源权重文件不变；
- 真正的 finite/state/coverage 错误能保留前缀并退出。

可以使用外部只读依赖与隔离解释器处理命名冲突。必要依赖在不污染其他工作的前提下一次性处理，记录版本；不能为了满足旧环境说明而整体降级运行中的 CUDA/PyTorch。

### GPU smoke

每个新臂，在同一四张程序化图上运行一条忠实发布函数参考路径和一条新入口路径：

`2 arms × 2 paths × 4 images = 16 base-Adam calls`。

两条配对路径使用相同源状态、增强随机数和适配配置，比较 logits、BN affine、optimizer state、LR 和 RNG。容差 rtol=1e−4、atol=1e−5；离散值、身份、步骤精确相同。比较不得因分割得分被放宽。

未出现参数非零变化不一律视为故障，先按确定的输入／解析梯度测试确认实现；真正非有限值则停止。

### 正式执行

- 两个新臂、两个域序、每序 1,951 组：**7,804 条新评分／7,804 次 base Adam 调用**。
- 含 smoke：**7,820 次 base Adam 调用**。
- G 临时参数扰动与恢复单独计数，不冒充两次正式 Adam；C/G 的 backward 次数也分开记录。
- 新 source/DD/外层训练均为 0。
- 发布实现下，C 每张通常为六弱视图+一强视图+一最终预测=8 次网络前向，G 再加一条 auxiliary 前向=9 次。正式预期为 C 31,216、G 35,118，共 66,334 次前向。若接口封装需要额外校验前向，单独登记，不能悄悄计入方法推理效率。
- 只在已授权 GPU 中使用一张，优先 7；不停止其他进程。活跃 GPU 阶段硬上限 6 小时，新私有输出上限 1 GiB；这是限额不是预计耗时。资源不足保留状态说明，不自行扩大预算。
- 不新建无限后台 watcher 或自动重跑；采用当前任务可用的受控执行机制，退出后不自行进入下一实验。

## 10. 报告主表与匹配问题

主要子集沿用 remaining_dev，同时报告 legacy、P1 extension、all_dev。每域先平均 OD/OC，再等域平均。两个顺序分别展示。

主表：N / A / EA / O2 / D4 / C-CTTA / G-CTTA。

冻结比较：

- G−C：发布梯度对齐与动态学习率组合的增量；
- C−A、G−A：不同完整适配方案的实际效能，不是纯参数量因果效应；
- C−N、G−N：相对无适配的净收益；
- G−O2、G−D4：是否超过旧复杂代理对照；
- 两个顺序中同一差值的变化，不按序选赢家。

使用配对均值、中位数、正/零/负、最差10%及最差单图；ASSD 使用共同有效 cohort，空/满和未定义数量显式报告。没有像素图就不声称独立重算像素级 ASSD，CPU 复核标量与计数即可。

旧结果只有在实际样本身份／顺序、checkpoint、预处理、N/A/O2/D4 原代码和 evaluator 一致时复用；不能以不同域序或相似均值凑表。新 C/G 不需要重复所有 P2 轨迹。

准确率和成本同时看。GraTa 多视图成本远高于单步调用数暗示的成本；不能只写“一步更新所以一样快”。分别报告每图前向／反向／优化器调用数、耗时、峰值显存、变化参数量。新方法没有源样本 rehearsal；旧 O2/D4 保留源 mask 与凝缩图像，部署条件也需标明。

## 11. 去留决策：不把本轮又变成训练准入 gate

C/G 一次完整跑完，不要求 C 先高于 A 才执行 G，也不要求 source 分数先提高。没有源端训练分数筛选和 target checkpoint 搜索。

- 若 G/C 在同流下明显优于 A 且代价可接受：得到可用的新参考范式，仍不能记为本项目原创方法。
- 若 C 已取得全部收益而 G 无增量：保留简单 C，不强推复杂梯度对齐。
- 若 A 仍更好：接受原生 VPTTA 是该协议下较强基线，结束这一轮移植，不立即启动 norm／loss／lr 补丁搜索。
- 若两序和域间差异明显：报告混合适用性，不用 GT 逐域选臂形成虚假“统一方法”。

不设置数值效果门槛来阻止计划内实验。最终结合效应量、方向一致性、尾部风险和成本作资源决策。单 seed、两个相同样本顺序，不作统计显著性、未见泛化或临床声明。

下一篇主方法的贡献需要重新定义；本轮只补足外部完整方法在同协议下的实证参照。不将 GraTa 更名为 Ours，也不在成功后自动接回原 DD。

## 12. 交付与状态

建议输出：

- `configs/b1_grata_matched_baselines_v1.json`
- `docs/B1_GRATA_TRANSFER_CONTRACT.md`
- `results/b1_grata_matched_baselines_v1/B1_EXPERIMENT_REPORT.md`
- `results/b1_grata_matched_baselines_v1/public_aggregate.json`
- `results/b1_grata_matched_baselines_v1/execution_audit.json`

执行 commit 与发布 commit 分开记录。输出目录0700、私有文件0600，源码、匿名聚合和许可证可公开；真实图像、mask、身份、路径、checkpoint、概率图不公开。

独立 CPU 重算完成后记为 `B1_MATCHED_BASELINE_COMPARISON_COMPLETE`。工程故障记为 INCOMPLETE 并保留前缀；不能临时换算法、换参数或自动续跑形成一个伪完整实验。

该状态仅表示比较完成，不等于已建立新的有效方法。

## 参考与已核验源码

1. P2 最终报告：
   https://github.com/DLwbm123/DPA-CTTA/blob/2fcbc0a69645e34da42b97830935f0e290e488aa/results/p2_frozen_factorial_full_stream_v1/P2_EXPERIMENT_REPORT.md
2. GraTa 论文（AAAI 2025）：
   https://ojs.aaai.org/index.php/AAAI/article/view/32244
3. 官方冻结源码：
   https://github.com/Chen-Ziyang/GraTa/tree/33ae20d664f305af34739ec54a5bec7da53ffa0b
4. 训练／测试入口：
   https://github.com/Chen-Ziyang/GraTa/blob/33ae20d664f305af34739ec54a5bec7da53ffa0b/GraTa-master/TTA.py
5. 梯度与一致性实现：
   https://github.com/Chen-Ziyang/GraTa/blob/33ae20d664f305af34739ec54a5bec7da53ffa0b/GraTa-master/custom_optimizers/grata.py
6. 官方运行默认任务：
   https://github.com/Chen-Ziyang/GraTa/blob/33ae20d664f305af34739ec54a5bec7da53ffa0b/GraTa-master/run.sh

以上用于精确说明算法来源与实现边界，不构成本环境已执行新实验的声明。

## Implementation binding

Runtime imports the unmodified pinned GraTa functions. It does not execute TTA.py or run.sh. The canonical P2 segmentation state loads strictly with identity name mapping. Unused auxiliary heads are absent. The feature tuple is unused by ent/consis. All adaptation is standard BN affine, with current batch statistics and no running-stat updates. Published entropy and perturbation/LR math are preserved.

Strong augmentation uses batchgenerators 0.25.2 in an isolated dependency directory on the server; this pins the missing transitive augmentation version for both arms without replacing PyTorch. The published augmentation includes a constructed but unused Bezier object; retained. No new donor image. The image-only numpy dictionary may be mutated by the augmentation library, so the original final-inference tensor is kept separate, as in the official CUDA entry point.

Scheduling is one arm at a time, order0 C/G then order1 C/G. RNG is privately restored/saved by each host; each starts seed 20260907. Across-domain host/Adam state is continuous. Six-hour budget includes smoke and formal active-stage wall time. Long run uses a detached one-shot launcher; run failure stops, no retries or watcher.

Official dependency: https://github.com/Chen-Ziyang/GraTa/tree/33ae20d664f305af34739ec54a5bec7da53ffa0b . MIT copyright/license reproduced in docs/GRATA_LICENSE.txt. Upstream model/augmentation/update sources remain in an external pinned checkout.
