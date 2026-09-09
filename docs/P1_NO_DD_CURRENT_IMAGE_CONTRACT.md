# P1：结束当前影像 DD 主线，验证无 DD 的当前图像源预测约束
## 实验计划与可交给 Codex 的有限任务说明

**状态：待执行的新方案；不是已运行实验，不继承 M4 的执行授权。**

研究优先级：**有效性第一，创新性第二，SOTA 第三。**

## 0. 研究决策：不再继续修饰 M1–M4

M4 的匹配对照 T4−L4 在 Fundus/Polyp、原域序/反域序四种设置下均为负，且 T4 的训练耗时更高。当前证据支持停止“固定 K=4 合成图像 → 原生低频 prompt 的间接监督 → 不断增加外层目标复杂度”这条具体实验主线。保留全部实现、代理和负结果，不删除历史，不把 GM-DD 改名为 Ours。

本轮不做：
- 新的数据蒸馏、代理训练或延长 M4 窗口；
- 新的 episode sampler、FDA/频带/混合强度搜索；
- H_source_stats、norm 扫描或 source 分数修复；
- learned router、critic、按目标 GT 选择方法；
- 更换分割 backbone、source checkpoint 或引入大基础模型；
- 任何“某个子模块先涨分，另一个实验臂才运行”的性能 gate。

本轮直接执行一项**无新离线训练的部署方法比较**：把冻结 source 模型在当前图像上的预测作为软约束，并用固定概率平均对照，拆清“改变更新”与“仅集成输出”的贡献。

这不是原 DD 方法的成功续章，也不预先命名为已建立创新的新 SOTA 方法。源模型约束、师生一致性和集成均有已有研究；本轮首先检验它们在相同医学分割协议下是否有可用价值。

---

## 1. 基线与必须阅读的材料

仓库：`DLwbm123/DPA-CTTA`

- M4 结果发布提交：`fab4ce4f7733e54258d577fe5e22cea8a562b6d6`
- M4 实际执行提交：`f88e99d212e914b74a7c521bac8c418fc73e5d6c`
- 固定外部依赖：`DLwbm123/CTTA@dbff0d985c6c95345d9fb78f5b1daef57b392564`
- M2 O2 使用各任务已封存的第 600 步代理。
- M4 D4/L4 使用各任务已封存的第 150 次外层更新代理。

新分支建议：`experiment/p1-no-dd-current-image-v1`

先读：
- `results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md`
- `configs/m4_trajectory_distillation_v1.json`
- M1/M2 的实际数据注册、target 选择、评分和历史结果格式；
- `src/dpa_ctta/hosts/vptta.py`；
- 原生 `VPTTA/{OPTIC,POLYP}/vptta.py` 和 AdaBN/Prompt/Memory；
- M2 后已解决的 smoke 严格确定性作用域及环境恢复逻辑。

旧配置、旧分支和外部依赖不修改。新增独立配置 `configs/p1_no_dd_current_image_v1.json` 和独立入口。

---

## 2. 本轮假设及其边界

### 2.1 要检验的假设

对于一张当前无标签图像，冻结 source 模型的预测与原生适配模型的预测具有不同的错误模式。将 source 预测用于同图、像素对齐的软约束，可能减轻有害适配，并保留部分有用的跨域修正。

### 2.2 不能预先当作事实的内容

- Source 模型可能在目标域很差，甚至自信地错误。
- 源预测不是目标真值，本轮损失属于自蒸馏/软预测约束，不是真实目标监督。
- 固定概率平均不保证优于两个组成模型。不能从已有 Dice 数值平均推算 ensemble Dice。
- 该约束仍使用原生 prompt，并未证明或消除适配空间的容量限制。
- 固定系数的无增益只能否定本轮具体配置，不证明所有源约束或全部 TTA 无效。

本轮完全不搜索这些问题的“最佳解”，通过一个预定 2×2 设计得到完整结果。

---

## 3. 方法与 2×2 对照

设：
- `q0 = sigmoid(z_source(x)).detach()`：同一 checkpoint、标准 source BN、eval、无 prompt 的预测；
- `z_pre`：原生适配前向所得、Adam 更新之前的当前图 logits；
- `pA`：原生 A 的更新后预测；
- `pSA`：新 SA 分支的更新后预测。

Fundus OD/OC 使用两个独立 Bernoulli 通道；Polyp 使用一个。不得使用类别互斥 softmax。

### 3.1 A：原生对照

`L_A = L_host`

保持 native Prompt、AdaBN、warm-up、Adam、memory、初始化、最终推理和 push 顺序。

### 3.2 SA：当前图像源预测约束

定义像素/通道平均的 Bernoulli KL：

\[
L_{\mathrm{anchor}}
=\frac{1}{CHW}\sum_{c,u}
D_{\mathrm{KL}}\{\operatorname{Ber}(q_{0,c,u})
\Vert\operatorname{Ber}(\sigma(z_{\mathrm{pre},c,u}))\}.
\]

唯一新增 loss：

\[
L_{\mathrm{SA}}=L_{\mathrm{host}}+0.1L_{\mathrm{anchor}}.
\]

- 使用稳定的 `BCEWithLogits(z_pre, q0) - H(q0)` 实现。
- `H(q0)` 为 stop-gradient 常数，可用正确处理 0/1 的 xlogy/等价实现。
- 对整个 B/C/H/W 求均值，B=1。不得引入新的前景重权、置信度筛选、Dice 伪标签项或阈值。
- 若实现只用 BCEWithLogits 而不减去常数，梯度虽相同，但报告必须明确记录 BCE 与 KL 的差异；推荐同时正确记录 KL。
- 系数 `0.1` 为本轮预先指定值，不依据 target 指标校准。
- teacher 完全冻结，无 EMA、无 teacher optimizer、无 target label 输入。
- teacher 在原 RGB 上走 N 的原预处理，不能使用已适配 prompt 或某种“选得更好”的源模型。
- student loss 使用已有的第一次当前图适配前向，不额外做一次 student forward 来得到这个 loss。
- host 与 anchor loss 在同一 backward 内合并，仅执行一次原生 Adam step。
- 原生 memory 每张图只 push 一次；存储语义不变；不存 mask 或 teacher prediction 到 memory。
- 不更新 source/teacher 权重及 running buffers。
- 不改变 norm 的前向规则，SA 的 student 仍保留原生 AdaBN。
- 不增加代理 clone、proxy image、DD artifact 或历史轨迹 meta-gradient。

在固定 teacher 下，anchor 对 logits 的梯度是 `sigmoid(z_pre) - q0` 再除以平均分母；用独立张量测试验证。

### 3.3 ENS_A：纯输出平均对照

\[
p_{\mathrm{ENS\_A}} = 0.5q_0+0.5p_A.
\]

### 3.4 ENS_SA：源约束更新加相同输出平均

\[
p_{\mathrm{ENS\_SA}} = 0.5q_0+0.5p_{\mathrm{SA}}.
\]

- 两个系数固定 0.5，不搜索。
- 平均概率，不平均 logits，也不平均 hard masks。
- evaluator 统一在概率 >=0.5 处生成 hard mask。
- ENS_A/ENS_SA 是父轨迹的输出视图，没有独立 optimizer、独立 memory 或额外更新。
- ENS_A 不能改变 A 的轨迹，ENS_SA 不能改变 SA 的轨迹。
- 不根据域名、图像内容标签或评价分数选择 q0/pA/pSA。
- 不作 OD/OC 后处理修补以抬高分数。

### 3.5 完整结果表的八个输出

`N / A / ENS_A / SA / ENS_SA / O2 / D4 / L4`

O2、D4、L4 在两个任务上都保留，不能先按各任务的旧分数挑一个对照，然后声称打败了整个历史路线。

无条件完成所有计划输出；不要求 ENS_A 或 SA 先有效。

---

## 4. 固定宿主、状态与计算条件

除 SA 新增同图 soft-KL 外，其余设置沿用：

- Fundus ResUNet34、512；Polyp PraNet、352；
- 原 source checkpoint；
- native prompt alpha=0.01；
- lr=0.05 / 0.01；
- Adam betas=(0.9,0.99)，eps=1e-8，weight_decay=0；
- 每图一次更新；
- warm_n=5，neighbor=16，memory_size 参数=40；
- 保留原生实际容量/重复 key/淘汰语义，不偷偷“修正”；
- seed=20260907；
- N 标准 source BN；A/SA 原生 AdaBN；
- 每个 task/order/method 初始化独立；
- 跨域无 reset；跨任务/顺序重新开始；
- 两个 ensemble 完全复用对应父轨迹。

D4/L4/O2 的在线设置和代理均保持其历史原样，按同一新流重新执行。不得仅将旧均值填入新的内容/顺序表。

---

## 5. 不再只围绕同一 224 组评价：固定一次开发扩展

### 5.1 复用与扩展

保留 M1–M4 使用过的每域 32 组，标为 `legacy_dev`。
每域从原数据清单的剩余合法内容中，再选最多 32 组，标为 `extension_dev`。

名义范围：
- Fundus：REFUGE、ORIGA、REFUGE_Valid、Drishti_GS；
- Polyp：CVC-ClinicDB、ETIS-LaribPolypDB、Kvasir-SEG。

最多 7×64 = 448 个不同内容组。原样本共 224 个，扩展最多 224 个。

这里的“扩展”只表示没有用于 M1–M4 的逐图评分；不保证整个项目历史上从未评价过。不得宣称 untouched final test、患者独立或盲评。

### 5.2 确定性选择

使用已有完整 manifest、M1 注册中的角色与排除信息；不重新扫描整个 NAS，不重新选择旧 32 组。

候选必须：
- 不与已有 M1–M4 图像内容重复；
- 不与该任务完整 source 图像内容重叠；
- 不属于明确封存/禁止开发资产；
- 无同图不同 mask 的未解决冲突；
- 按已知关联排除明确的禁止跨角色重叠；
- 继续执行跨目标域图像去重，保持原域优先顺序。

从合格候选中按：
`SHA256("P1:20260907:" + task + ":" + domain + ":" + image_sha256)`
排序；平局按 image_sha256；代表样本按已有规则。

选择至多 32 组后，旧组与新组共同恢复原 manifest 顺序；已知视频相对时序不打乱。
这一步不读取任何模型分数，也不以 mask 面积或类别难度选样本。

已明确的 train/test/combined provenance 原样保留。REFUGE_Valid combined 和缺失 patient/video linkage 继续按 M1 已采纳的探索性开发协议处理：
`UNKNOWN` 是披露限制，不再触发 H2 式无限登记阻塞。

某域不足 32 个新增合法组，就使用全部合法新增组并报告实际数；若为零，保留旧组，明确该域没有扩展证据。不可用重复图补齐，不换域，不自动扩大数据搜索。
配置和预算的确切 expected stream 在注册后、模型运行前生成。

选中文件的字节身份与编码检查针对本轮选中资产完成，不将旧元数据检查冒充新字节验证。

### 5.3 两种连续流顺序

Order-0：
Fundus `REFUGE → ORIGA → REFUGE_Valid → Drishti_GS`
Polyp `CVC-ClinicDB → ETIS-LaribPolypDB → Kvasir-SEG`

Order-1：反转域块顺序，保持每个域内图像顺序不变。

同域加入新图会改变历史。因此，本轮 legacy_dev 子集也处于新轨迹中；不能将其与旧 32 图流的差异全归因于新算法。

### 5.4 目标标签与数据使用

在线算法仅接收当前图像。q0 是当前图像的冻结源模型预测，不是 label。

A、SA 的预测及两个 ensemble 必须固定后，evaluator 才读取该图 GT。原生旧代理对照也遵守相同边界。
不读取未来图像用于 teacher 或统计，不依据标签重置状态，不用本轮 target loss/指标修改系数。

本轮不再新增 source-clean 评分或 post-target source 回访。已有 source 结果可作背景；不得称为新方法的保持性证据。

---

## 6. 实施次序：一次实现、一次 smoke、一次完整评价

### 6.1 CPU 必要正确性检查

保留历史测试，不必为“审计数量”重复所有不相关 DD 二阶测试。
运行受影响的：
- 新 KL 的独立梯度与 q0=0/1 数值边界测试；
- probability-ensemble 的算法和 >=0.5 阈值测试；
- λ=0 的新入口对原生 A 退化等价；
- teacher params/buffers/grad 冻结；
- evaluator 标签变化不影响主轨迹；
- ensemble 不改变父轨迹；
- 三个冻结 DD 代理的加载身份和 shape/mask 绑定；
- 扩展选择与去重、空新增候选、combined/UNKNOWN 正常处理；
- expected stream、记录覆盖和独立重算。

这是一轮实现正确性测试，不是新的 norm 或 source 效果诊断。
不得放宽旧数值断言掩盖路径错误。

### 6.2 GPU smoke：12 次真实在线 Adam

每任务共 6 步：
1. 原生 A 一步；
2. SA 新入口 λ=0 一步，与 1 从相同状态比较；
3. SA λ=0.1 一步；
4. 冻结 O2 一步；
5. 冻结 D4 一步；
6. 冻结 L4 一步。

两个任务合计 12 次。
使用预定程序化 query，源 checkpoint 为真实注册资产；无需真实 target GT。
配对 smoke 继续使用已验证的严格确定性上下文，完成后恢复；正式计算环境沿用 M4，不偷偷改变全局数学路径。
λ=0 的配对可使用已有 index=16 原生状态；只在内存恢复，不重跑历史。状态资产若缺失可用预先定义的冷启动 CPU/程序化替代 smoke，并明确记录 smoke 覆盖减小，不为补 16 步临时启动新诊断。

不能要求 SA 必须比 A 产生更高程序化 Dice。
检查更新、prediction 和 source invariants 有限/正确即可。
零 anchor 梯度在 q0 已等于 student 的输入上是合法的；非零梯度连通性用预先指定的非匹配 toy 张量验证。

### 6.3 正式评价

先冻结代码、配置、注册和全部六个历史 DD 代理（两个任务×三方法）的摘要。

建议按 task/order：
- A 与其 N teacher 产生 N、A、ENS_A；
- SA 与其 N teacher 产生 SA、ENS_SA；
- O2、D4、L4 各自独立连续运行。

或者使用等价的顺序/共享 teacher 计算方式，但不能污染方法间 RNG、状态或结果。
若共享同一个 q0 前向节省总计算，报告部署独立成本时仍给每个需要 teacher 的方法计入 source forward。

每条结束的记录持久化，所有臂/顺序按配置完成后统一重算。性能不好不中途停止，也不增跑新权重系数。

---

## 7. 计算预算和计数口径

设实际不同内容组总数为 `G`，最大为 448。

### 7.1 正式评分与更新

- 两个顺序；
- 八个输出：N、A、ENS_A、SA、ENS_SA、O2、D4、L4；
- 五条有更新的独立轨迹：A、SA、O2、D4、L4。

因此：
- 新评分记录：`8 × 2 × G`，最大 **7,168** 条；
- 正式真实在线 Adam：`5 × 2 × G`，最大 **4,480** 次；
- 加 smoke：最大 **4,492** 次；
- 新的图像外层更新：**0**；
- 新的可微内层/轨迹反向训练：**0**；
- 新的代理训练：**0**；
- source 模型训练：**0**；
- 源图像 rehearsal 仅限冻结旧 DD 对照自身既有的 synthetic/mask 路径，SA 无代理。

N 在不同顺序的行是同一无状态方法的重复展示，不是新增独立样本。
ENS_A/ENS_SA 是从父轨迹得到的不同预测，不多计 Adam 和 memory push。

旧 M4 记录只作为历史背景，不计入本轮新记录，也不直接补齐扩展流缺失行。

### 7.2 资源范围

- 一张当前任务明确授权的 GPU，优先 7；计划范围 3–7，使用前登记实际 UUID；
- 允许已有进程共存，不停止其他程序；
- 不升级环境，不同时占用多张 GPU；
- GPU 活跃阶段上限 6 小时，不因结果不佳追加时长；
- 新私有标量/注册/报告输出上限 256 MiB；
- 已有代理和 checkpoint 原位置只读引用，不复制完整模型；
- 不保存 target 图像、mask、概率图或 logits；融合所需预测仅在当前访问内存中使用并及时释放；
- 无后台轮询、自动重启或额外参数运行。

实际时间/显存观察不能冒充预先保证。此处 6 小时是资源上限，不是完成时间预测。

---

## 8. 评价与结果判定

### 8.1 主终点

分别报告：
1. extension_dev（未用于 M1–M4 的新增内容）；
2. legacy_dev；
3. combined。

**主要方法判断优先看 extension_dev 的两种顺序结果，combined 为整体开发表现。**
若部分域新增为空，extension 表列出实际覆盖域，不悄悄填旧组或零分。
Fundus 域内先 OD/OC macro 再等域平均；Polyp 等域平均；两个任务不合成一分。

所有平均概率后的 Dice/ASSD 都必须由真实像素预测计算。
`0.5 Dice(N)+0.5 Dice(A)` 不是 ENS_A 的 Dice。

### 8.2 核心对照

- `SA − A`：同图源预测约束是否改善更新；
- `ENS_SA − ENS_A`：在相同集成规则下，新更新是否有额外贡献；
- `ENS_A − A`、`ENS_A − N`：固定平均本身的作用；
- `ENS_SA − SA`：对新轨迹做相同平均的作用；
- SA/ENS_SA 分别对 N、O2、D4、L4 的配对差；
- 每个域的 A−N 背景，尤其不能只减少坏域损失而抹去 ETIS/ORIGA 的有效适配收益后再只挑坏域报告。

不可用 GT 在方法间逐图/逐域选最优结果当作可部署方法。
“历史最强对照”仅是完整方法的指标比较，不是组成一个 label-informed oracle。

### 8.3 指标

每任务/域/通道/顺序/子集：
- Dice（0–1 原始、% 展示）、差值 pp；
- 配对中位数、正/零/负数量、p10/p90；
- 最差 ceil(10%n) 配对差均值和最差单条，说明内容组非独立患者；
- ASSD common-valid 数量、均值/中位数差和正误差尾部；
- GT/pred empty/full，未定义项不置零；
- N 对比任何适配方法的严重退化，不被任务均值掩盖；
- pipeline/host-step、N teacher 前向、反向、memory push/retrieval、allocated peak。

两个顺序用同一批内容，不把它们当两个独立患者队列。单 seed 不报告缺乏依据的显著性。
如做重复样本重采样，不能声称其包含重跑整条流所产生的顺序不确定性；本轮不新增统计重跑模型。

### 8.4 冻结的解释规则（不是运行 gate）

- 若 SA−A 和 ENS_SA−ENS_A 在新增内容、两个顺序上均有有用的正向差异，才支持更新层面的新信号。
- 若 ENS_A 有收益，而 ENS_SA 不优于 ENS_A：收益属于固定集成，不归功于新增 loss。
- 若 SA 更接近 N，但没有优于 N/A 或融合对照：属于损害抑制，不称“学到更好的适配”。
- 若增加源约束主要损害本来受益的目标域：如实记录保守偏置的代价。
- 若旧 D4/L4/O2 更好：保留这个事实，不把旧 DD 负结果扩大成“DD 永远不值得”。
- 若所有增量仍为小幅、方向不稳：停止本方案，不自动搜索 λ、融合系数或 learned router。

可将“新增内容任务均值比最强简单可部署对照高至少 0.5 pp，两个顺序均为正”作为研究资源优先级的参考，且同时查看任何域是否损失超过 1 pp。它不是统计/临床阈值，不触发自动追加训练。
任意结果都必须完整交付；不存在某臂先涨分才评分下一臂的条件。

---

## 9. 故障、复现与提交

保留原始失败日志。只有身份错误、真正的实现错误、非有限值、OOM、资源或写盘失败阻断执行；低分、零净增益、UNKNOWN linkage 不是工程故障。

一次正确性验证完成后执行固定全流程。若正式阶段工程失败，保存已完成前缀并报告准确次数，不静默换系数/图像/硬件条件续跑；修复执行必须有单独记录，不伪装一次无失败实验。

只复用本次任务授权的资产和 GPU，receipt 绑定实际干净执行 commit、配置、流注册、proxy 摘要和 smoke 摘要。
没有独立外部 review 时，不写虚构 CODE_REVIEW_PASS。必要检查属于一次普通工程验收，不再要求多轮审查后才能比较方法。

独立 CPU 进程重读本轮所有私有行：
- 校验真实 expected stream；
- 验证八输出/两序覆盖；
- ENS_A/ENS_SA 父轨迹绑定；
- 验证真实更新计数不含 ensemble 重复；
- 重算所有表与差值；
- 在明确父输入身份/公式的情况下检验导出标量一致性。仅存标量不能重建未保存概率图，应如实说明“重算标量汇总”，不声称离线再做了像素级推理复现。

最终交付：
- `configs/p1_no_dd_current_image_v1.json`
- 必要实现与测试；
- `docs/P1_NO_DD_CURRENT_IMAGE_CONTRACT.md`
- `results/p1_no_dd_current_image_v1/P1_EXPERIMENT_REPORT.md`
- `public_aggregate.json`、`execution_audit.json`；
- 执行 commit 与发布 commit 分开；
- 列出扩展实际组数、新记录/更新次数、计算成本、失败历史和限制；
- 公共内容不包含患者/视频 ID、真路径、逐资产 digest 或私有数据。

完成状态：`P1_NO_DD_COMPARISON_COMPLETE`。
科研结论单独写：更新有增量 / 仅集成有益 / 混合 / 未见收益。
完成后停止，不自动进入 P2。

---

## 10. 参考材料与事实来源

1. M4 结果与执行证据：
   https://github.com/DLwbm123/DPA-CTTA/blob/fab4ce4f7733e54258d577fe5e22cea8a562b6d6/results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md

2. M4 冻结配置：
   https://github.com/DLwbm123/DPA-CTTA/blob/fab4ce4f7733e54258d577fe5e22cea8a562b6d6/configs/m4_trajectory_distillation_v1.json

3. 原生 VPTTA / Each Test Image Deserves a Specific Prompt：
   https://arxiv.org/abs/2311.18363

4. Robust Mean Teacher for Continual and Gradual Test-Time Adaptation（师生一致性已有工作；本方案不是 RMT 的忠实复现）：
   https://openaccess.thecvf.com/content/CVPR2023/html/Dobler_Robust_Mean_Teacher_for_Continual_and_Gradual_Test-Time_Adaptation_CVPR_2023_paper.html

5. ROID / Universal Test-time Adaptation through Weight Ensembling, Diversity Weighting, and Prior Correction（源知识保持和集成已有工作；权重集成与本方案概率平均不是同一算法）：
   https://arxiv.org/abs/2306.00650

本方案的数学定义和预算是新的预定实验设计，不是引用文献宣称已验证有效。


## 实施绑定补充

用户本轮“执行”授权 P1。旧科学源文件与配置均保持原样；新入口为 scripts/run_p1_no_dd_current_image.py，后台串联为 scripts/launch_p1_once.py。SourceAnchorHost 只在隔离编译的原生 image block 中捕获第一次 forward 的输出，并替换唯一 loss 赋值；λ=0 直接返回原生 host loss。标准 BN teacher 从原 N 入口构造。A/SA 采用独立 RNG/state，同图共享一个只读 teacher 前向。先完成五个预测，再由 evaluator 获取 GT；三个历史 DD 对照另走完整独立流。

概率评价直接在 >=0.5 处阈值，保留像素计数以便独立复算 Dice；没有概率转 logits 再取 sigmoid 的舍入路径。ASSD 沿用既有四连通表面/像素距离定义。CPU 重算不保存/重建概率图，只核对标量、计数、身份和父输出公式；实际融合在当前图内存完成。共享 bundle 峰值按共享分配报告，部署独立成本给需要 teacher 的方法计入一次源前向。

本地受影响检查共 7 项，覆盖两个任务 18 步冷/热 λ=0 对齐、teacher/ensemble/label 隔离、KL 边界梯度、概率阈值与旧像素指标一致性、固定代理绑定、扩展角色/冲突/空集，以及八输出两序独立重算。不以重复 DD 二阶测试数量替代这些验证。不声称独立外部代码审查。

长时间运行遵循用户当前约定：完成 GPU smoke 后可靠后台启动，低成本确认日志与正常前缀即可结束会话；不持续轮询。下一次查询确认运行结束后完成公开结果交付。
