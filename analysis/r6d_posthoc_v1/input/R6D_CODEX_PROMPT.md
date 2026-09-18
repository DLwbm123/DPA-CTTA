# Codex 执行 Prompt：R6-D 只读事后诊断

你正在仓库 DLwbm123/DPA-CTTA 中工作。请完成 **R6-D：既有 R6-A 标量账本的只读、事后机制诊断**。这不是 R6-B，不是新 GPU 试验，不是改 gate 后重判旧结果。

本任务允许：编写隔离的分析工具，执行程序化 CPU 测试，读取你已有权限访问的既有运行标量账本和注册 JSON 元数据，在全新私有目录完成离线分析，发布去身份聚合与结论。禁止：GPU 查询/初始化/smoke、原始 RGB/mask/source 数据和 checkpoint 读取、模型构造/前向/反向/Adam/VJP、真实轨迹、后台轮询/自动重试、修改旧结果或生产代码。既有评价计数包含 GT 信息，允许离线使用；不能宣称本分析不使用任何标签信息。

## 一、输入文件与优先级

随附同版本材料：
- R6D_EXPERIMENT_PLAN.md
- R6D_ANALYSIS_SPEC.json
- README.md
- MANIFEST.json

先按 MANIFEST 验证文件原始字节、长度及 SHA256。计划与 JSON 必须一致；真正冲突应报告，不能选择方便的版本。无需等待另外的 science proposal、math_reference 或 math_history：本包不依赖这些额外文件，不得据文件名臆造它们。

本分析明确为 **已见到原 R6-A 汇总结果后的 post hoc exploratory analysis**。现在冻结分组和公式不等于结果未知时预注册。所有分析分支都须交付，不能只展示支持新实验的结果。

## 二、固定历史绑定

原结果发布提交：aa732b42b03d378149028a3770879b72ebc55db3
原模型执行提交：c94fff7c05cec38d541c441b62a9381ee46ba076
原science SHA256：2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff
原run_id：a70dc79f23db47289532792141b61f57
registration digest：8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
stream digest：cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
production fingerprint：a6ef161066d3560dc2dd540cb507bff2e4a7cadf94ab711545877c525dee2f2e
原结果：R6A_COMPLETE_NO_ADVANCE
原R6B：NOT_RUN

先检查现有工作区和已有 R6-D 实现，保留用户改动。不覆盖 main、旧分支、原生产 Python 或任何历史结果。建议新分支 analysis/r6d-posthoc-v1，新代码在 analysis/r6d_posthoc_v1/。分析提交 SHA、分析配置摘要和原执行 SHA 分开记录。

## 三、资产发现、只读快照与失败边界

1. 从当前工作区已有的去身份交付指针和已获权限的本地/服务器配置解析原账本位置。不猜测 /Users/... 或服务器路径，不全盘扫描，不读取不相关账户或密钥。公开 GitHub 只用于报告/源码/聚合，不能宣称获得私有账本。
2. 使用既有标量 JSON/JSONL 与注册元数据白名单，记录每项真实来源、长度和SHA256。在新的私有分析工作目录创建普通文件快照；禁止源符号链接、硬链接、写入源目录、chmod源目录和改旧结果指针。拒绝与源目录存在重叠的输出根。PYTHONDONTWRITEBYTECODE=1。
3. 源只读字节核对在开始/结束分别做；目标以新文件写入，不覆盖以前失败记录。临时输出也只能进入新工作目录。公开包不包含私有逐内容行和路径清单。
4. 不调用原 recompute、invalidate、publish。优先使用经过源码检查确认为只读的 completed/replay/join/validate_metric 等逻辑。如必须加读取适配层，保留原校验强度、单独测试，不修改原实现。
5. 缺失私有账本时，仍完成能从公开聚合恢复的分析以及工具的CPU测试；联合分布/逐内容联结/时序字段为NOT_AVAILABLE，状态R6D_PARTIAL_AGGREGATE_ONLY。不得用边际分布生成伪配对行，不得以补充新推理解决缺字段。
6. 已有账本身份、完整性或只读保护失败属于 R6D_INCOMPLETE，不是科学负结果。保留原异常、已完成项和实际成本；不自动重跑。

## 四、数据核验与主指标保持

- 只用 C、R_BAL、R_SCALE、R_SHUFFLE × streams 0/1/4。
- 每组合1,951个唯一内容、1,951条unlabeled和1,951条evaluation；两类文件各总计23,412行。
- 核验receipt/scope/run/science/code/registration/stream/production fingerprint以及worker/job binding、退出及smoke/completion记录。
- 当前代码的原8/1/1/0访次与累计计数、pixel计数可实现性、跨预测/跨臂/跨流GT一致性、q前景数与trace.n_fg一致性全部保留。
- 主指标只用remaining_dev的1,695内容；每图OD/OC平均、四域等权、order0/order1等权；stream4单列。全部1,951内容用于时间轴、先前曝光和chunk位置。
- 私有逐图配对键为(stream,group_id,channel)，用sample_id/metadata/visit作为一致性核对。跨流必须按(group_id,channel)配对，禁止按visit、排序后的Dice或任意行号配对。
- 用完整精度复算原主结果、逐序和gate，不改变旧NO_ADVANCE。对原REPORT六位小数仅使用规格内的显示舍入核对容差，不能拿它放松底层核验。
- 不将内容group_id称为患者ID，不把同一内容跨流/跨臂的重复访问视为独立患者。

## 五、分析定义（与计划一致）

### A. 精确路径/当前步分解

每图每通道使用百分制Dice：
L_m = post_m - pre_m
Delta_post = post_BAL - post_control
Delta_pre = pre_BAL - pre_control
Delta_step = L_BAL - L_control
要求 Delta_post = Delta_pre + Delta_step，controls为C/SCALE/SHUFFLE。

对OD/OC/macro，三流、四域和注册段分别计算。均值在同一行集同一权重下可相加；中位数、分位数、worst-decile必须从配对行算，不能从两个边际摘要相减。

每条主评分流行权重omega=1/(4*N_domain_remaining)，两个主序合并再乘1/2；通道贡献入macro再乘1/2。报告signed net、positive mass、negative mass，各域/通道/段贡献要加回同一个主指标。净差接近零时不输出以其为分母的解释占比。

Delta_pre不是C0 H_t或遗忘因果量；Delta_step也不是同一个模型状态下两个loss的反事实比较。

### B. 同内容跨流与注册切换

对BAL−C的pre/step/post分差分别计算1−0、4−0、4−1的同内容对照，按域/通道展示分布与贡献。

使用既有注册chunk_schedule与contiguous_domain_segments；不得另选最坏窗口。相邻同域chunk不是新切换，初始段也不是切换。真实切换后固定第1–8访问与第9及以后比较；所有到达图决定位置，评分过滤随后进行。

记录segment/chunk、先前同域曝光数、同域出现次数、距上次同域访问间隔及唯一内容分母。域内内容排序由原registration确定，不从评分大小推出。

回访1,951内容均唯一；禁止把不同内容早晚分数叫遗忘。跨流同内容也同时改变历史、位置及访问相关增强分配，不叫纯顺序因果效应。

### C. 监督贡献与伪目标误差

从既有trace恢复：pi、BCE前景占比、f0=residual_fg_sse/S0、fw=weighted_fg_sse/Sw、fw−f0、a以及两个局部梯度余弦：
cos_BAL_C = residual_weighted_dot / sqrt(S0*Sw)
cos_perm_C = residual_shuffled_dot / sqrt(S0*Sperm)

分母0给null和原因。所有量属于该臂自身局部状态；C/SCALE/SHUFFLE里基础区域权重量不冒充实际施加的权重；上述余弦不冒充BN参数梯度方向。

从q评价的P/G/I/N恢复TP/FP/FN/TN，计算precision/recall、FP/N、FN/N、(P−G)/N：
E_plain=(FP+FN)/N
E_regional=(applied_w_fg*FP + applied_w_bg*FN)/N

C/SCALE/SHUFFLE的E_regional标为该臂局部状态下的hypothetical_base_regional_hard_error，不冒充实际loss；BAL也只能称hard-error质量，不是BCE或错误梯度。

不得把整个伪前景residual_fg_sse当作FP残差。没有按GT与soft residual联合分区记录的量全部缺失，不推断/合成。既有评价GT只用于离线分析，不生成在线标签选择器。

### D. 预先列明的事后分层

- 主分层用C锚定的同图区域五分类：EMPTY_FG；0<nfg<N/9；N/9<=nfg<=8N/9；8N/9<nfg<N；EMPTY_BG。使用整数边界检查。
- 在各stream/domain/channel内，C的a、fw−f0、Adam displacement分别独立分四分位；只由C无标签量计算边界，重复边界允许空组，等于边界归较低组，不用ID打破ties，不按结果合并组，不做变量交叉网格。
- 保留BAL自身变量的次要描述并标为处理后变量；GT误差不是主要分层依据。
- 分组mean不是新四域主指标；同时报告组内分母和按原行权重计算的主指标贡献。某域缺失时不悄悄重归一化成3域平均。
- 每个预定单元全部导出；n_unique<30标LOW_SUPPORT而非删掉。不能仅凭低支持单元提出新GPU实验，但它仍进入原四域主指标。
- 用域内全部内容顺序的前后半作一次时间混杂描述；mid=floor(N_domain_all/2)。不是新训练/验证集，不要求每半都赢。
- 不做p值筛选、最优阈值、可训练gate、反向门控、eta/cap扫描、隐含多个比较后只留赢家。

## 六、必须回答的竞争解释与停止规则

报告逐项列“支持 / 反证 / 不能识别”：
1. pre已经存在的历史路径差距；
2. 当前一步加权是否额外损伤，或在缓解既有差距；
3. FP/FN及基础加权hard-error质量是否提供错误监督风险证据；
4. C锚定的较强干预状态是否与损伤相伴，是否只是单域/低支持/时间位置构成；
5. 单纯域/通道交换，尚不足以识别机制。

不要求必须提出新假设，也不添加一个必须涨分的新gate。

只有“干预强度过大”获得可检验的描述性支持时，允许在HYPOTHESIS_DECISION.md提出（不是实施）：
R6-E: w_half=0.5+0.5*w_BAL，固定eta=0.5；C/原R_BAL/R_HALF×0/1/4，共最多9条新完整轨迹。
这是权重向1收缩，不是LR/ratio_cap减半，不是预测融合。formal拟议140,472 forward、17,559 backward/Adam、0VJP，未来smoke另计。

不能为了推进这条预留提案选择性解释数据。若证据不支持强度解释，记录不建议运行R_HALF；至多提出一个更明确的新问题，不在本轮新增模型实现、science或GPU授权。R6-E的成功/推广门槛须在后续独立方案中说明，不由本分析自动给PASS。

## 七、实现、测试与成本

- 工具尽量只依赖标准库及已有numpy/pandas；不引入模型训练依赖。确需import torch读取冻结纯validator时仍禁止模型与GPU接口、autograd和optimizer调用。
- 用程序化标量fixture测试，不调用旧完整ResUNet套件。按实际报告新测试数量，不提前宣称通过多少项。
- 必需覆盖：identity/duplicate/missing；精确分解；domain/channel/order权重；段贡献；跨流group配对；相邻同域chunk边界；null/ties/稀疏组；q计数下界；GT一致；原文件字节不变/结果指针不变；错误输出路径；模型/GPU入口拒绝；公开去身份。
- 可以先开发/修复程序化CPU测试，再在已有权限环境执行一次固定分析。保留首失败日志，不把本轮代码测试称为重跑原166项或复现模型。
- 单进程最多2 CPU线程；单次分析wall cap1800秒、新输出cap2GiB；快照I/O和磁盘另外计账并事先检查。失败不后台重试，不不断轮询。模型调用预算永远0。
- 准备完成后无需再为“允许读旧标量”制造一轮GPU式审阅授权流程；但材料/权限确实不足时用PARTIAL或INCOMPLETE如实结束。

## 八、交付

REPORT必须开头说明：POST_HOC_EXPLORATORY；原R6-A仍NO_ADVANCE；原R6-B未启动；本次无新模型计算。

交付：
1. 上述全部CSV及机器聚合，保留分母、null原因、对照标签、贡献权重、LOW_SUPPORT和全表。
2. HYPOTHESIS_DECISION.md：证据支持/反证/缺失、至多一个下一次干预问题。
3. ANALYSIS_BINDING.json：历史执行绑定、analysis SHA、analysis spec SHA、实际输入摘要；不修改原science。
4. READONLY_AUDIT.json：源文件before/after、快照摘要、原指针不变、0资产/模型/GPU接口调用与局限。
5. 真实CPU测试、运行日志、成本、FIELD_AVAILABILITY及DELIVERY。
6. 私有逐行联结和源文件列表不公开；公开包不能含group_id/sample_id、患者映射、绝对私有路径、PID、GPU UUID或资产字节。

最终状态按实际结果：
R6D_COMPLETE_HYPOTHESIS_PROPOSED
或 R6D_COMPLETE_NO_NEW_HYPOTHESIS
或 R6D_PARTIAL_AGGREGATE_ONLY
或 R6D_INCOMPLETE

共同字段：
new_model_execution_started=false
new_model_forwards=0
new_model_backward_calls=0
new_adam_calls=0
new_vjp_calls=0
R6A=R6A_COMPLETE_NO_ADVANCE
R6B=NOT_RUN
next_gpu_execution_authorized=false

analysis_executed与analysis_status按实际填写，不能把已经完成的R6-A写成NOT_RUN。完成后停止，不进入R6-E或任何模型运行。
