# P2：冻结方法的全量连续流与匹配集成比较

> 用途：可直接作为 Codex 的下一项有限实验任务。研究优先级为有效性第一、创新性第二、SOTA 第三。本文不是新的算法有效性声明，也不是已经发生的实验授权或执行记录；用户将本文作为任务指令发出后，按以下范围执行一次。
>
> **本轮不训练新模型，不修改 SA，不学习融合权重，不新增 router，也不重新启动 DD。** 把已有适配方法放在同样的预测层对照下，在现有合法清单的完整内容上比较；完成后停止。

## 1. 为什么做 P2，而不是继续修改 SA

P1 的主要终点是新增内容组 `extension_dev`。其公开结果如下，Dice 单位为百分比，差值为百分点 [S1]。

| 任务/顺序 | A | ENS_A | SA | ENS_SA | O2 |
|---|---:|---:|---:|---:|---:|
| Fundus/order0 | 75.530124 | 74.557334 | 75.694456 | 74.584981 | 75.862848 |
| Fundus/order1 | 74.417744 | 74.092643 | 74.563188 | 74.181860 | 74.706328 |
| Polyp/order0 | 80.035920 | 81.802933 | 80.056420 | 81.805723 | 79.850722 |
| Polyp/order1 | 81.606976 | 82.868578 | 81.579378 | 82.836592 | 81.293925 |

P1 支持：Polyp 的任务均值收益主要来自固定概率平均，不是 SA 更新；Fundus 上平均有明显代价，SA 只有小幅增量且未超过 O2。融合损害了 Fundus 的 ORIGA 和 Polyp 的 ETIS，不能称为所有域都更安全。P1 不是泛化成功或新方法创新的证明。

据此关闭 SA 当前配置的继续扩展。P2 只回答：

1. 固定概率平均的 Polyp 收益，在更完整内容及长流上是否保留？
2. Fundus 的平均代价，是否在相同条件下依然存在？
3. 对 A、O2、D4 同样加平均后，旧 DD 相对最简单的 ENS_A 是否仍有增量？

第 3 问是预测层的公平对照，不是“再给 DD 一次训练机会”。不得把 ENS_A 对 raw O2/D4 的优势直接归因为无需 DD，因为前者包含额外 source 预测。源知识保持/集成已有相关方法，本文不将固定平均包装为新理论 [S5–S6]。

## 2. 基线、分支与范围

- 项目：`DLwbm123/DPA-CTTA`。
- 基线发布提交：`18b8f32186678e8930a6c540f96941365fd511b9`。
- P1 实际执行提交：`f65e119016f2d6a32cd0cffbee2bb2e94f570c2d`。
- 固定外部依赖：`DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`。
- 新分支建议：`experiment/p2-frozen-factorial-full-stream-v1`。
- 新配置：`configs/p2_frozen_factorial_full_stream_v1.json`。
- 完成状态：`P2_FROZEN_FULL_STREAM_COMPLETE`。该状态只表示覆盖完整，不表示有实质收益。

先读 P1 报告、配置、运行/评分代码及既有 registration；定位并复用封存 O2-600 与 D4-150 的模型无关代理文件及其身份记录。旧运行代码、配置和结果不覆盖，固定依赖不编辑。

本轮明确不做：SA2、lambda 搜索、平均系数搜索、温度校准、熵/置信度门控、按域选择最佳方法、oracle 输出、source 训练、新 DD、代理继续训练、改变 K、改 norm、添加边界项或风格迁移。也不新增 source-clean 或源域回访流。

## 3. 七个输出、三条适配轨迹

设 `q0 = sigmoid(source_logits)`，source 路径沿用 P1 的标准 source BN、无 prompt、相同 checkpoint、相同预处理。

| 输出名 | 定义 | 独立在线更新 |
|---|---|---|
| N | q0 | 无 |
| A | 原生 VPTTA 的最终预测 pA | 有 |
| EA | `(q0 + pA) / 2` | 无，绑定 A |
| O2 | 原封存 O2 代理下的最终预测 pO2 | 有 |
| EO2 | `(q0 + pO2) / 2` | 无，绑定 O2 |
| D4 | 原封存 D4 代理下的最终预测 pD4 | 有 |
| ED4 | `(q0 + pD4) / 2` | 无，绑定 D4 |

- 所有平均均在概率空间按每个像素/独立通道进行，权重固定 0.5，hard 阈值固定 `>=0.5`。
- 不引入 SA、L4 或其他新臂；O2 代表已经封存的适配目标代理，D4 代表 GM 对照。两个任务使用同一组对照，不看域分数选代理。
- 平均输出不能影响父轨迹的 loss、prompt、Adam、memory 或下一次访问；不得为平均分支额外 step/push。
- `EO2` 必须用它自己的 O2 轨迹预测，不能用 A 或其他臂的状态替代。
- 保留 K=4、proxy weight=0.1、boundary=0、identity proxy、完整 batch、原 clone 和原生学习率等设置；O2/D4 不重新训练，不改像素。
- 不将 teacher q0 的计算写成免费；配对研究可共享前向，但每个独立部署方法都应承担其实际需要的 source 前向和驻留模型成本。

### 3.1 解释性数学事实，不是代码修改

对有限二值 logits a、b，在实数算术下：

`(sigmoid(a) + sigmoid(b))/2 >= 0.5` 当且仅当 `a+b >= 0`。

因此在这个特定权重和阈值下，hard 决策等价于两个 logit 求和后判符号。概率值本身不等价于 `sigmoid((a+b)/2)`；概率损失、校准和有限精度行为也不相同。生产评分继续使用原概率平均实现，不能借此静默替换算法。

平均可以修复一条路径的错误，也可能将另一条路径的正确前景压回背景；不将其描述为自动识别真值或保证安全。

## 4. 数据：从已合法清单一次性扩展到完整开发池

### 4.1 固定域范围

Fundus 顺序 0：
`REFUGE → ORIGA → REFUGE_Valid → Drishti_GS`。

Polyp 顺序 0：
`CVC-ClinicDB → ETIS-LaribPolypDB → Kvasir-SEG`。

顺序 1 仅反转域块；每域内部依然是原 manifest 播放顺序，不倒序图像，不按分数重新排序。

历史 H2 inventory 为 3,759 行，是去重、保护角色检查之前的上限，不是当前已核验的有效组数 [S4]。本轮使用这些已有清单中满足既有合法性规则的全部内容组，不再每域挑 32 张。

### 4.2 可执行的资格规则

1. 复用当前 source 身份、目标根目录、编码契约及 P1 已核验 registration；只对新增选中内容追加必要字节/几何检查。
2. 根据既有全 source image digest 排除 source/proxy 内容重叠。
3. 显式 sealed-final、禁止开发的资产不得纳入。
4. 同一图像的 mask 身份冲突遵循既有排除规则并记录，不根据分数解决冲突。
5. 跨域同内容只保留预登记较早域的代表；同域 alias 采用既有最小 sample_id 代表规则。顺序 1 不重新做去重归属。
6. 已知患者/视频关联用于既有禁止重叠约束；缺失字段记录 UNKNOWN。不得又以缺少患者/视频登记作为探索性研究的硬停止条件，不额外扫描 NAS 寻找不存在的登记。
7. `train/test/combined` 原始字段保留为 provenance，不视为自动的开发/盲评角色。既有 combined 内容允许按此前明确的探索性开发协议纳入。
8. 不因某域预测差、mask 前景小、平均带来退化或特殊病灶而剔除样本。
9. 资格筛选后某域数量变少按实际值使用，指标继续域等权；不重复补齐、不换域。如果某域零合法资产，明确缺域，不伪造七域完整评价。
10. 不能拿“零分”替代资产无法读取或指标实现异常。登记之外的工程故障保留前缀并停止，不跳图继续污染后的轨迹。

### 4.3 固定三个内容标签

- `legacy_dev`：M1–M4 已使用、P1 也保留的原 224 组；
- `p1_extension_dev`：P1 新增的 224 组；
- `remaining_dev`：本轮合法内容中不属于上述两类的组。

`remaining_dev` 是本轮主要内容子集；不是“全球从未见过/全项目盲评”，因为这些域可能早已有整库历史评价。所有样本都是探索性开发。本轮的子集标签只用于评分分层，不能传入在线方法。

如果某些组因真实身份异常不再合法，记录精确原因；不悄悄把它们从旧 448 中删除却继续声称完全对应。

### 4.4 全流与子集的关系

三类内容在完整 manifest 顺序中自然混合。先运行整条连续流，再根据标签分层统计，不能各子集单独 reset 后拼接成全流。

更完整的历史会改变旧 448 组的适配预测。旧 A/O2/D4 的逐图指标不得直接填入新表。历史 P1 仅作不同流下的参考；方法比较使用 P2 同一条流的记录。

## 5. 执行协议

- 每任务、每域序、每父方法独立从原 source checkpoint、空原生 history 和 seed=20260907 开始。
- 跨域不 reset、不清 Adam、不提前查看后续图像统计。
- 两个域序之间重新构造 host，不沿用前序状态。
- 线上输入是一张当前图像；不提供 task 内域 ID、subset ID、标签、候选方法分数。
- N 前向冻结、不更新 BN；A/O2/D4 各自按原生方式初始化 prompt、adapt/predict/push。
- 目标 mask 在相应所有待评分预测固定之后，才交给 evaluator；不得用于融合系数、方法选择、回滚或梯度。
- 最简单调度是逐访问依次运行三个父 host，共享本次 q0；各 host 的 RNG 独立保存/恢复。
- 如显存不宜同时驻留三个 host，允许在执行前冻结为逐父轨迹顺序运行，在线重新计算 q0，不保存目标概率图。这只改变调度和 teacher 前向次数，不改变科学设置。报告真实前向数，不能将额外调度代价计为方法优势。
- 不升级环境、不重写 native 算子。复用 M2 修复后的严格配对 smoke 作用域，正式训练/评分仍保持 P1 的既有设置。不声称相同 seed 保证全部 CUDA 逐比特确定性。

## 6. 一次必要验收，不开新诊断轮次

### CPU 相关测试

保留已有测试；优先运行与此次注册、视图、评分和父轨迹相关的测试，不强制无关历史全套重新执行作为额外 gate。必须覆盖：

- 0.5 概率平均、阈值和独立 OD/OC 通道；
- 平均前后父轨迹状态不变，ensemble 行物理 update/push=0；
- 七个输出共享正确 sample/order/parent 身份；
- 更改 evaluator mask 只改变评分，不改变父轨迹与平均预测；
- 全量选择/去重/保护角色与三个 subset 标签；
- 缺失行、重复行、错序、错误计数时拒绝 COMPLETE；
- 下述混淆计数可以恢复 Dice；ASSD 的未定义值不作零填充。

### 单次 GPU smoke：12 次 online Adam

两个任务，各比较 A/O2/D4 的既有入口与 P2 新入口。每个父方法两条隔离路径各执行一个预定程序化访问：

`2 tasks × 3 parents × 2 implementations × 1 step = 12`。

使用已封存权重/代理与合法的程序化状态，按既有严格确定性上下文比较；不能用正式目标分数调整容差。N 和三种 average 做前向/评分接口检查，不额外 step。

检查 logits、prompt、Adam、counters、memory、RNG 的作用域一致性。离开 smoke 后恢复原有 backend 设置，异常也恢复。

smoke 一次通过后直接完成本轮全部正式评价；没有任何分数门槛。

## 7. 预算：零新训练，按实际 G 计数

设资格筛选后两个任务合法组数之和为 G，历史 inventory 上限为 G<=3,759。

| 项目 | 精确公式 | 使用历史 inventory 的最大值 |
|---|---:|---:|
| 七输出 × 两序评分行 | 14G | 52,626 |
| A/O2/D4 × 两序 online Adam | 6G | 22,554 |
| smoke online Adam | 12 | 12 |
| 总 online Adam | 6G+12 | 22,566 |
| 新 source / proxy / selector 训练 | 0 | 0 |
| 新外层 Adam / 可微训练 inner | 0 | 0 |

资格登记完成后写入实际 G 及精确 counts，不能按常量宣称已完成。G 的增加必须在旧清单范围内，不能临时加入新数据集。

teacher 前向：共享逐访调度时正式为 2G；逐父轨迹调度时可为 6G。N 行和平均视图不引入新的可适配状态。独立部署每个融合方法均应计入一个 source 前向。

一次有限执行，GPU 活跃阶段上限 6 小时，新私有磁盘上限 1 GiB；这是资源配额，不是时间估计。只使用本轮任务明确许可的 GPU，建议沿用物理 3–7 中一张、优先 7，核对 UUID，与其他程序共存，不停其他进程。若新任务授权范围更窄，以新任务为准；不得将旧轮许可伪装为本轮已经发生的授权。

不重训代理，不创建新的有限差分/梯度诊断，不重建 source state history 库。资源不足则保存前缀并明确 PARTIAL，不建后台等待器、不自动重试、不按效果追加实验。

## 8. 评分与少量配套证据

### 8.1 主评价

- 两任务分别统计，Fundus 在每域先 OD/OC macro，再跨域等权；Polyp 跨域等权。
- 同时报告 `remaining_dev`（主要）、`legacy_dev`、`p1_extension_dev`、`all_dev`。
- 两种域序分别报告。可以附同一内容在两序下的差异，但不得将 2G 当作 2G 个独立患者。
- 每组保留 N/A/EA/O2/EO2/D4/ED4 绝对分数；不能用 target 域真值选“最佳方法”拼成可部署表。

必须报告的 paired differences：

`EA−A`、`EA−N`、`EO2−O2`、`ED4−D4`、`O2−A`、`D4−A`、`EO2−EA`、`ED4−EA`、`EO2−ED4`。

额外报告交互差：

`(EO2−O2)−(EA−A)`、`(ED4−D4)−(EA−A)`。

这些是同一内容流内的匹配比较。P2 与 P1 的 raw 差不能简单归因于某一模块，因为内容量与适配历史也变了。

### 8.2 分布和风险

对每个 paired comparison 保留：均值、中位数、p10/p90、未舍入正/零/负数量、最差 ceil(10%n) 的均值和最差单图。Dice 为 fraction 存储，展示为百分比/百分点。

ASSD 继续使用共同有效 cohort 和原单位，报告 empty/full 与未定义数量；不把 undefined 填成零，不构造 OD/OC macro ASSD。

不要求额外患者登记来计算描述性结果，也不以当前两序声称统计显著。默认不做把访问当独立样本的显著性检验。

### 8.3 同次评分顺带记录错误互补，不另起实验

在 N、父方法 B、平均视图 E 的预测已固定后，按每通道记录 16 格像素计数：

`count[GT_bit, N_hard, B_hard, E_hard]`，每个维度为 0/1。

由此可还原三者 TP/FP/FN、hard disagreement，以及：
- N 错而 B 对时，平均保留了多少正确判断；
- B 错而 N 对时，平均修复了多少错误；
- 平均新增的 FN/FP 与修正的 FN/FP。

前景/背景分别报告，不能只报全像素 accuracy 让大量背景淹没小结构。不同臂不能被 GT 选择或加权；这些计数只是预测后的解释材料，不是在线输入。

硬预测一致的像素在精确算术下平均也一致；有限精度/等号边界按固定生产实现记录，不为通过一个代数不变量悄悄改阈值。

这些计数不保存目标图像、标签图、logits 或概率图。Dice 可以从计数独立重算；ASSD 仍只能在 live scoring 时计算，CPU 标量重算不能冒充重新计算了被丢弃的像素距离图。

## 9. 结果解释与结束动作

本轮必须完整运行所有预定方法，不以 EA 已提高或降低作为提前停止/进入下一阶段的 gate。非有限、身份异常、真实实现错误或资源配额耗尽才是执行停止原因。

预登记解释：

| 结果 | 结论 |
|---|---|
| Polyp EA 在 remaining_dev 与两序均有清楚收益，Fundus 仍退化 | 保留任务特异的实用经验，不能称统一 CTTA 方法；不按域 GT 动态切换 |
| EO2/ED4 不再高于 EA | 历史 DD 在相同额外 source 前向下未体现必要性；停止新 DD 训练 |
| EO2 或 ED4 稳定高于 EA，且超过单条原始方法 | 只支持封存 DD 与 source 的互补性，不宣称原训练目标普遍更好；后续是否验证另行决定 |
| 平均在完整流或剩余内容上失效 | P1 的局部平均收益不能泛化到本协议；不自动搜索系数或 learned gate |
| 方法仍仅交换微小均值排名且负向尾部无改善 | 不为维持 Ours 名称继续追加模块；将当前路线作为已完成探索性基线与消融材料保存 |

报告“至少多少差值值得下一轮资源”只能作为完成后的参考，不能调到刚好跨线；本轮没有新增效果准入门槛。评价重点是相对相同计算/输出条件的简单基线，而不是跟一个变弱的臂比。

不能称为新 SOTA、临床安全、患者级独立或全项目未见测试。源模型保持/集成已有文献；任务特异的选择若未来用于部署，应提前固定且另做不参与设计的验证，不从本轮逐域真值产生在线 oracle。

## 10. 交付、安全与执行边界

- 私有目录 0700、文件 0600；原文件只读复用，不删除。
- 仅保存小型标量日志、上述像素计数、身份和资源记录。O2/D4 代理只读加载，不发布到 GitHub。
- 每次完成的记录 flush/fsync；发生失败保留精确阶段、已完成计数和成功前缀。
- 接口简单复用 P1，不再搭一套与科学任务无关的层层审批框架。receipt 绑定实际干净执行提交、配置、资格登记、GPU UUID、一次 smoke 和固定代理摘要即可。
- 独立 CPU 重读全体标量日志，检查 parent/view、subset、域序、覆盖、online 计数、Dice 与比对项；ASSD 的重算边界如实披露。
- 结果发布提交不同于执行提交，不把执行 checkout 中途更新到发布 HEAD。

公开交付建议：

```
configs/p2_frozen_factorial_full_stream_v1.json
docs/P2_FROZEN_FULL_STREAM_CONTRACT.md
results/p2_frozen_factorial_full_stream_v1/P2_EXPERIMENT_REPORT.md
results/p2_frozen_factorial_full_stream_v1/public_aggregate.json
results/p2_frozen_factorial_full_stream_v1/execution_audit.json
```

一次任务内完成：必要实现、一次验收、完整评分、独立重算、报告。结束时明确列出执行/发布 commit、各阶段退出码、实际 G、14G/6G 计数、主要比较与预算，随后停止。不要承诺当前会话以外的自动后续研究，不自动开启 P3。

## 11. 来源和状态说明

以下链接用于核对设计背景；本文件中的新实验尚未执行。P1 报告与配置已经读取，1.78 MB 的完整聚合 JSON 未在本次方案撰写中逐项重算；私有逐图数据也未访问。

- [S1] P1 结果报告（固定发布版本）：
  https://github.com/DLwbm123/DPA-CTTA/blob/18b8f32186678e8930a6c540f96941365fd511b9/results/p1_no_dd_current_image_v1/P1_EXPERIMENT_REPORT.md
- [S2] P1 配置：
  https://github.com/DLwbm123/DPA-CTTA/blob/18b8f32186678e8930a6c540f96941365fd511b9/configs/p1_no_dd_current_image_v1.json
- [S3] P1 执行审计：
  https://github.com/DLwbm123/DPA-CTTA/blob/18b8f32186678e8930a6c540f96941365fd511b9/results/p1_no_dd_current_image_v1/execution_audit.json
- [S4] 历史完整清单 inventory（不是当前有效组数）：
  https://github.com/DLwbm123/DPA-CTTA/blob/348f13ade87e4f7b7197ac6638bd38f4b528bdd0/docs/H2_SOURCE_STATS_TARGET_DEV_REPORT.md
- [S5] RMT, CVPR 2023；师生适配近邻，不等同于本轮固定概率平均：
  https://openaccess.thecvf.com/content/CVPR2023/html/Dobler_Robust_Mean_Teacher_for_Continual_and_Gradual_Test-Time_Adaptation_CVPR_2023_paper.html
- [S6] ROID, WACV 2024；权重集成等近邻，不等同于本轮预测概率平均：
  https://openaccess.thecvf.com/content/WACV2024/html/Marsden_Universal_Test-Time_Adaptation_Through_Weight_Ensembling_Diversity_Weighting_and_Prior_WACV_2024_paper.html


## 实现绑定

用户本轮“请执行”授权 P2。新代码复用 P1 的 update/teacher、概率平均、像素评价、代理加载以及原生 Observed；P1/M1–M4 科学文件和结果不改。shared_current_visit 调度同时驻留 N/A/O2/D4（O2/D4 各有原 proxy clone），每个父方法独立保存 RNG。七个输出全部固定后读取当前 GT；三种平均只保留父输出视图，无额外 Adam/push。

新增内容完成字节/容器/几何检查；P1 已验证文件只复核登记身份及 size/mtime，不再全量重新哈希。16 格表只随融合行存储，CPU 复核其 N/父/融合边缘计数与 Dice。背景/前景分列，ASSD 不从丢弃的像素图重建。独立部署 resident_models 为 N/A=1、O2/D4/EA=2、EO2/ED4=3，prompt 不计为分割模型；实测 allocated peak 是共享调度峰值，不冒充各方法单独峰值。

相关 CPU 测试覆盖完整选择/冲突/保护角色/三标签、冷轨迹 label/view 隔离、七输出错序/缺失/重复/父绑定/错误计数拒绝、16格及 Dice/ASSD 边界、完整标量重算和确定性异常恢复。无新 DD 二阶诊断。CPU 首次夹具中的 alias mask 与原图不符、以及将 legacy 标签改成相同值的问题已纠正，无 GPU 消耗或数值容差放宽。

长期实验遵守用户当前运行约定：单次 smoke 通过后可靠后台执行，确认正常日志前缀后可结束会话；不持续轮询或自动重启，完成后查询时再校验与公开交付。

成本口径补充：pipeline_elapsed_seconds 是共享访视开始至当前输出评分的累计时延，不可跨臂相加充当物理墙钟。独立部署时延使用 parent host-step 加所需 source 前向，评价器时间单列在共享 pipeline 中。

实际登记：G=3753（Fundus 1951，Polyp 1802），remaining_dev=3305，legacy_dev=224，p1_extension_dev=224。旧 P1 448 组均完整保留。正式 records=52542、online=22518、teacher forwards=7506；含 smoke online=22530。ETIS 合法组为190，其余域与历史清单上限相符。复用 P1 字节验证记录896个文件，新验证6610个文件；具体排除与关联信息仅存私有 registration。
