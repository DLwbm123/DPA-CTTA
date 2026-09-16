# R5 阶段 I Codex Prompt：更新收益诊断与验证后提交

请在 DLwbm123/DPA-CTTA 中实现 R5。当前任务仅为阶段 I：实现、程序化 CPU 验收、元数据 dry-run 和审阅材料交付。不得启动 GPU smoke、真实目标图像/标签/给定 checkpoint 的模型运行、R5-A 或 R5-B 正式实验。R4 的执行授权和 review waiver 不延续到 R5。

## 1. 固定来源与工作范围

从以下发布提交建立新分支 experiment/r5-update-acceptance-v1：
- 基础发布 SHA：b2bfce6cb29cea2df026194120f45b4f7252d53f。
- R4 实际执行 SHA：2377505819ca9be6658b4f5b34f49dac3bf67889；不要把发布 SHA 当作执行 SHA。
- registration digest：8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf。
- recurrence digest：cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db。
- 固定 CTTA 依赖：dbff0d985c6c95345d9fb78f5b1daef57b392564。
- 固定 GraTa 依赖：33ae20d664f305af34739ec54a5bec7da53ffa0b。

先阅读以下真实文件，并沿 import 追踪 C 的实际调用路径，不把非 C 分支当作 C 的实现：
- docs/results/r4_three_track_v1/REPORT.md
- docs/review/r4_three/STATE_LIFECYCLE.md
- docs/review/r4_three/REPRODUCE.md
- src/dpa_ctta/r4_three/host.py、execution.py、evaluation.py、analyze.py、plan.py
- src/dpa_ctta/r1/host.py
- src/dpa_ctta/b1_host.py
- scripts/run_r4t.py 及其 neutral_subprocesses 入口保护
- results/b4_frozen_c_transfer_v1/B4_EXPERIMENT_REPORT.md

检查当前工作区已有 R5 工作，保留用户改动；不得覆盖、重置或重复已有正式结果。不修改 main、R1–R4 科学配置/结果、固定依赖或旧 C 的行为。优先新增 r5_update_acceptance 包、独立入口和测试。必要的共享改动必须最小化、单列，并证明旧入口行为保持不变。

本轮只研究原 C 更新是否应当提交。不加入 RP、EMA/固定教师、kernel 适配、图优化、额外 loss、输出融合、周期回源或多状态路由。不重启其他已结束路线，不宣称首次创新、显著性、临床安全或泛化。

## 2. 共同科学协议

只使用既有 Fundus ResUNet34 给定 checkpoint；禁止源 RGB/mask、源 query、源代理/原型、源训练、额外预训练组件。阶段 I 连给定 checkpoint 也不得加载；只允许程序化输入与随机初始化的固定完整网络。

保留原 1,951 内容/轨迹、remaining_dev=1,695、512×512、OD/OC 独立 sigmoid、原 0.5 阈值判定、原 mask 映射/resize/ASSD 实现及空预测约定。算法每次只接收当前 RGB，不接收 domain、真实域边界、sample/group ID、subset、GT、未来图像或评价结果。到达计数 t 合法，从 1 开始。四主序与回访流沿用已登记元数据；不得重建回访序列。

可更新参数仍为 C 的 41 层 BN 的 19,136 个 affine 标量；保留 C 的当前输入统计政策。Adam lr=1e-4、betas=(0.9,0.999)、eps=1e-8、weight_decay=0；使用已有环境、既有 optimizer backend、float32、AMP 关闭，不升级依赖。每图六弱视图、一强视图、一次候选 backward/Adam、一次候选原图预测，共 8 forward、1 backward、1 Adam、0 VJP。

每轨迹独立初始化，seed=20260907；跨域不 reset。增强生成器沿用 C 的顺序与隔离方式；额外控制随机数不得消耗增强 RNG。GT 仅在最终在线状态提交、所有决策及缓冲区更新完成后由 evaluator 读取。

## 3. 冻结矩阵与分阶段授权

R5-A：只运行带只读诊断的 C，流为原 order0、order1、recurrence，共 3 条完整轨迹。影子规则不改变 C 的实际参数、Adam、输出或增强 RNG。不得根据前两条效果取消第三条。

R5-B：仅作为后续待授权方案。四臂 C、C_HALF、C_RANDOM、C_VERIFY × 原四主序和回访流，共 20 条正式轨迹。R5-A 的 3 条合格 C 计入该矩阵，不重复运行；B 新增 17 条：C 的 order2/order3，加另外三臂各 5 流。复用须科学指纹、数据/流、C 路径和记录完整性相符，并保留原 execution SHA；同源重复不是独立确认。

阶段 I 可以实现和测试四臂，但所有真实入口默认 disabled。A 与 B 使用不同授权 scope。A 完成后无论 gate 通过与否都退出；不得自动执行 B、追加 GPU 诊断、追加 seed、修改阈值或组合模块。

## 4. 三个预测与零额外前向的观察接口

每次访问捕获：
- p_pre：原有六弱视图中的原图预测概率，来自本次候选更新前状态。
- q：实际传入原 C BCE 的六视图软目标，不另换设备/归约顺序重算后替代它。
- p_trial：候选 Adam 更新后的原图概率。
- p_emit：最终已提交学生状态的原图概率；接受时为 p_trial，拒绝时为缓存的 p_pre。

模型/evaluator 之间保持原生 logits 接口；缓存相应 pre/trial logits，选择返回的 logits 与 p_emit 对应。概率只用于门控与诊断，禁止将概率交给会再次 sigmoid 的旧评分入口。不得因缓存改变原 dtype、shape、通道顺序或输出设备契约。

另保留六个已逆变换对齐的弱视图概率 p_j，供当前图的无标签可靠区计算。捕获只读、detach，不能改变前向返回、原 loss、求导路径或 RNG。不得额外调用模型获取 pre，不得用 strong 预测冒充 pre，也不得把更新后 q 或教师集成作为 p_emit。

稠密概率只在当前 step/evaluator 生命周期存在；评价完成后清空，不跨图保存、不落盘。历史算法状态只包括已提交模型/Adam、至多 128 个风险标量、到达计数及独立 RNG。当前事务快照不属于历史图像记忆。

## 5. 唯一冻结的无标签规则

计算仅使用 detach 后的当前概率；标量归约使用 CPU float64，模型与 loss 计算保持原样。以下量均在完整 512×512 栅格计算，不借用 RP 的 32×32 token 采样或 PCA bank。

e_pre = mean((p_pre - q)^2)
e_trial = mean((p_trial - q)^2)

对 OD、OC 分别建立可靠前景和背景，共四个区域：
R_c_fg = (q_c >= 0.9) AND (全部六视图 p_j,c >= 0.5)
R_c_bg = (q_c <= 0.1) AND (全部六视图 p_j,c < 0.5)

对每个非空区域 R：
r_R = mean_R((p_trial,c - p_pre,c)^2)
r = max_R(r_R)

空区域不参与 max，记录计数与 null；全部四区为空时 r=null，不能填 0。概率、梯度、loss、参数、optimizer 或已定义标量出现非有限值是运行错误，立即停止该轨迹，不作为普通拒绝。

缓冲区保存至多最近 128 个有效 r，包括 warm-up 访问及之后被拒绝的候选。计算当前门限只读过去缓冲区，不含本图。历史分位数固定为 q=0.90、线性插值，h=(n-1)*0.90；使用排序后的相邻值插值。完成本图接受/拒绝并恢复必要状态后才追加本图有效 r，之后 evaluator 才读 GT。

以下情况强制接受，按顺序记录原因：t<=32；当前四区全空；过去有效 r 少于 32 个。其余记 eligible=1，并执行：
accept = (e_trial <= e_pre + 1e-12) AND (r <= Q90_past + 1e-12)

eligible=0 时 accept=1。1e-12 是固定比较容差，不是搜索参数。没有额外可调阈值、按域门限、置信阈值扫描、learned selector 或 GT 反馈。

C：实际全部提交；只记录上述 shadow_accept。
C_HALF：从第一张开始仅将 lr 改为 5e-5，其余原 C，全部提交。
C_VERIFY：实际使用上述规则；拒绝后输出 pre，不重新前向。
C_RANDOM：采用相同 warm-up/可观察性强制接受条件；eligible=1 时不查看 e/r 的数值大小，仅按常数 p_accept Bernoulli 决策。独立随机流采用新建 random.Random(20260908)，每次访问恰好消耗一个 random()，即使本次强制接受也照常消耗。

p_accept 不是现在填写的猜测值。冻结派生程序：A 三流全部 1,951 内容中的 sum(eligible * shadow_accept) / sum(eligible)。仅从无标签 trace 计算，不能按 Dice、域、subset 或搜索结果确定。分母为零则 A 资格不通过。A 结束后把值、分子/分母、输入摘要写入 label_free_calibration.json，绑定到 B receipt；在此之前 B 参数为 null、不可运行。RANDOM 在自己的状态上计算 eligibility；只保证同计算配额与近似接受率，不声称实际提交次数或轨迹完全匹配。 明确披露：p_accept 是 A 完整开发流提供的离线无标签标定，RANDOM 是有此标定的开发对照，不是零目标预扫描对照；其标定成本计入 A。C_VERIFY 不使用这个常数。B 内不得重新标定；未来新内容确认也只能使用已冻结的开发值，不能先扫描确认流。

## 6. 事务状态与物理计数

C_VERIFY/C_RANDOM 在候选计算开始前，为全部可变模型状态与 Adam 状态建立无别名快照。快照包含 BN affine、确有可变的 buffers、每参数 step/exp_avg/exp_avg_sq 及实际存在的其他 optimizer 字段、param-group 配置、延迟初始化状态的存在性。保持 Parameter 对象与 optimizer ownership 不变；不改 source 冻结张量。

接受：保留候选模型/Adam，输出 p_trial。
拒绝：恢复模型/Adam 的全部事务状态，清除候选梯度，输出 p_pre。候选中新出现、而快照不存在的 optimizer state 也必须删除。不能只恢复 weight/bias，不能把拒绝改成恢复到 source checkpoint。

拒绝时不得回滚：已消费的增强 RNG、RANDOM RNG、到达计数、候选尝试计数、物理 forward/backward/Adam 计数、过去风险缓冲区。候选图的有效 r 仍在决策结束后加入缓冲区。

分别记录 n_visits、n_candidate_adam、n_committed、n_rejected、n_forced、n_eligible、实际参数恢复次数。候选 Adam 的物理调用不能因拒绝从预算中减去；已提交 Adam step 应与 n_committed 一致，不再与 n_visits 强行相等。旧 C 的原断言不修改，新 host 使用自己的提交计数。允许合法零梯度/零位移，不以 toy 指标必须改善作为机械通过条件。

状态机应显式区分 IDLE、CANDIDATE、DECIDE、COMMIT/ROLLBACK、EVALUATION_RELEASE。前一张评价 payload 未移出不得进入下一张；失败 host 不可原地继续。

## 7. A 的诊断定义与唯一资格 gate

真实评分只由 evaluator 产生。Dice 转为百分制，差值为 pp，按同内容同顺序同通道配对：
L_t = D(p_trial,y) - D(p_pre,y)
G_s = MacroAvg_s((1-shadow_accept) * (-L_t))

MacroAvg_s：仅 remaining_dev，先每图 OD/OC 平均、域内均值、四域等权。A 主结果为 order0/order1 两序等权；recurrence 单列。全部 1,951 内容均参与真实 C 状态更新和无标签缓冲区，不能只对 remaining_dev 运行方法。

令 a_s = sum(eligible * shadow_accept) / sum(eligible)，使用该流全部内容的无标签计数。相同 eligible 范围、相同平均接受率的均匀随机影子参考为解析期望：
G_random_s = MacroAvg_s(eligible * (1-a_s) * (-L_t))

不需要追加模型轨迹或多组随机种子求这个期望。它是事后同路径诊断，不是在线已知未来接受率的真实方法。

A 合格必须同时满足：
1) 三条轨迹、计数、数据绑定、CPU 标量核验和 C 保持性均有效。
2) 三流 eligible 分母均非零、eligible 内至少发生一次影子拒绝，且 eligible 接受率各不低于 0.50。
3) G_order0 >= 0，G_order1 >= 0，两者平均 >= 0.20pp。
4) G_recurrence >= 0。
5) 两个主序分别满足 G_s - G_random_s > 0。

这些是描述性的资源筛选条件，不是统计显著性、安全或临床门槛。未满足则报告 R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE 并停止，不调整规则后重跑。满足则报告 R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW，仍须新的 B 授权。缺失/损坏/机械不一致不能写作科学负结果。

G_s 仅是在完整 C 状态路径上的影子输出变化，不是回滚后未来轨迹的效果或长期上界。可以补充 mean(max(0,-L_t))，但明确其为使用 GT 的同路径即时选优机会量，不是可部署方法。

## 8. 可选历史分解与应报告的诊断

若已有私有 C0 逐内容分数可只读取得，必须核对同一 checkpoint、content、预处理、统计政策、评分实现/阈值与分母，然后可计算：
H_t = D(p_pre,y) - D(p_C0,y)
D(p_trial,y) - D(p_C0,y) = H_t + L_t

C0 缺失、不匹配或只有聚合均值时，H_t=null，说明原因；不得按表格均值、行位置或跨内容拼接，不自动补跑 C0。此项不阻塞 A 的核心 gate。H_t 叫历史适配分数差，不叫固定旧域遗忘量。

报告 pre/q/trial/emit 的同图 Dice；L_t 的均值、中位数、正零负计数、最差 ceil(0.1*n) 配对均值；eligible/accept/reject 的分母、被拒绝与保留更新的 L_t；e/r 与区域计数分布；OD/OC 分列；逐域、逐序和固定到达窗口 1–32、33–128、129–512、513–1024、1025–1951 的描述性结果。

ASSD 至少覆盖 pre/trial/emit，使用既有 evaluator；按共同有效配对报告，undefined 不填 0，单位为原评估栅格像素。不把 q 的内容加权全图统计与 remaining_dev 域等权主表直接相减。

## 9. B 的预注册比较

主终点沿用四主序 remaining_dev 四域等权；recurrence 不与四主序合并。开发阶段晋级需同时满足：
- VERIFY-C >= +0.50pp，至少 3/4 主序为正。
- VERIFY-HALF、VERIFY-RANDOM 各 >= +0.20pp，且各至少 3/4 主序为正。
- VERIFY-C 的最差同序差 >= -0.50pp；每个域的四主序均值差 >= -2.00pp。
- recurrence 的 VERIFY-C >= -0.10pp。

尾部损伤、OD/OC、共同有效 ASSD 和全部缺失分母必须报告，不能用上述数值合格自动宣称安全。若只有 HALF 或 RANDOM 同样有效，报告简单控制解释，不归功于验证机制。正式完整矩阵内不按中途分数跳臂；全部完成后分析。全部仍为已暴露开发数据，A 参与选择的流与 B 新增流需标注，新 seed/顺序不是独立患者或盲测。

## 10. 记录与独立 CPU 核验

每访问一条私有 ledger，绑定实现/science/registration/stream 摘要与唯一访问身份；包含物理计数、提交计数、eligible/forced/reason/accept、过去缓冲区长度、Q90、r、四区 SSE/count、e_pre/e_trial 的 SSE/count、影子和真实决定，以及由单独 evaluator 生成的指标。

建议无标签 trace 与评价 trace 分开，独立 analyzer 才按身份联结。算法不能读取评价 trace。历史 C0 仅供 analyzer。公开导出不包含逐内容行、ID、路径、图像、mask、稠密概率、activation、适配权重或 optimizer。

CPU analyzer 从标量复核 visit/计数、分母、SSE/count 归约、过去窗口与 Q90、强制接受原因、影子/真实决策、p_accept 派生、主指标、配对差和 gate。要有重复/缺失/乱序/篡改即失效测试。只能称标量关系重算，不能称重建了概率、梯度、协方差或 ASSD 几何。

## 11. CPU 验收必须覆盖

- CUDA 未初始化；真实 target/source 像素、给定 checkpoint 读取数为零。只读已授权 registration 元数据不算像素运行；禁止任意 NAS 扫描。
- CPU 程序化完整 ResUNet + 原固定依赖的多步匹配：旧 C vs C_DIAG；VERIFY 强制全接受 vs C；RANDOM p=1 vs C。比较预测、可训练参数、Adam、增强 RNG、物理计数；同 CPU 后端要求逐值一致，不能只比较末尾 Dice。强制测试开关不可通过生产配置启用。
- 六视图顺序/逆变换、pre 为原图、q 为真实 loss target；detach 和只读 hook 不改梯度或额外前向。
- 已有 Adam 状态及首步尚未初始化状态的回滚；拒绝后参数/矩/step 恢复，下一张结果与正确回滚参考一致；物理计数与 RNG 不倒退。
- 访问序列超过 32 与 128，覆盖 warm-up、历史不足、全部/部分空可靠区、恰等门限、线性分位数、当图不得提前进入窗口、拒绝候选仍进入窗口及非有限值硬失败。
- GT 改变、评价延迟/省略（在合法 payload 生命周期下）、domain/ID 无法进入 host，不影响预测与状态轨迹；evaluator 不消耗算法 RNG。
- 全部四臂每图 8/1/1/0 的物理计数；attempt/commit/reject 守恒。
- 标签无关 p_accept、解析随机影子期望、域等权与内容加权差异、空分母、主要 gate 的边界值、3/17/20-job 生成及禁止 A 自动启动 B。
- 继承 IO/NFS ENOENT 处理、其他 IO 错误传播、neutral_subprocesses、进程清理、receipt scope、历史结果保护和 CPU analyzer 污染测试。

保留首次失败、修复及最终日志。明确实际通过/失败/跳过数与环境；依赖缺失不能伪造全套通过。可运行的部分完成并交付，缺失真实资产/授权不得静默替换。

## 12. 预算、后续运行入口与交付

A 正式预算：3×1,951=5,853 访问；46,824 forward；5,853 backward/候选 Adam；0 VJP。
A+B 正式总预算：20×1,951=39,020 访问；312,160 forward；39,020 backward/候选 Adam；0 VJP。
B 新增预算：17 条、33,167 访问、265,336 forward、33,167 backward/候选 Adam。
以上不含 CPU 程序化测试和机械 smoke，它们必须另记物理计数，不得隐藏或混入效果表。C0 新运行默认 0。VERIFY/RANDOM 不因拒绝更新节省前向或反向，不声称加速。

未来 A 机械 smoke 建议固定为每实际设备：旧 C 4 次访问与 C_DIAG 相同 4 次访问的独立初始化匹配，共 64 forward、8 backward/Adam。只用事前登记的内容与容差，全部单列；阶段 I 仅生成配方，不执行。正式运行的设备列表、GPU 容差、资源上限、smoke 内容和新 receipt 要在执行前冻结，不能把 R4 的 GPU 编号当作当前可用资源。最多 3 worker、每卡 1 worker，不获取新源资产。B 的执行配方须单独审阅绑定。

保留已有有限监督器、失败清理、NFS/进程审计修复；禁止未授权后台等待、自动 retry 或失败前缀混入正式结果。A 科学配置与规则不得在看到标签结果后修改。工程修复保留 provenance，不能伪装成从未失败。

交付至少包括：新代码/测试；默认 disabled 的 science 配置；本 prompt；EXPERIMENT_PLAN.md；IMPLEMENTATION_REPORT.md；STATE_LIFECYCLE.md；PARITY_CONTRACT.md；REVIEW_INDEX.md；真实 CPU 日志/JSON；metadata dry-run 的 3/17/20-job 矩阵及预算；窄代码 patch；去身份 DELIVERY.json。

PARITY_CONTRACT.md 在真实运行前说明 CPU 精确匹配、GPU smoke 继承的数值容差及出处、与历史 R4 C 逐内容/逐序核对的可用性和边界。不能看到差值后放宽容差，也不能用不受控共享 GPU 的时间证明速度优势。

默认入口只能输出计划/元数据，真实运行要求精确 implementation SHA、新 science SHA256、registration/stream 摘要、显式阶段授权和新 receipt。外部 review pass 与用户明确 waiver 必须分字段记录，不能自行签发或复用 R4 waiver。

若全部必需实现和 CPU/元数据验收通过，最终状态只写：
R5_IMPLEMENTATION_READY_FOR_REVIEW
同时列出 execution_started=false、external_review=NOT_RUN、R5A=NOT_RUN、R5B=NOT_RUN。未通过则写实际 BLOCKED 状态与已有证据，不宣称完成或启动计算。当前任务到此结束。
