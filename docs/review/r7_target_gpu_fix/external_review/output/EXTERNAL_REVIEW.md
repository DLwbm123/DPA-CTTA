# R7 GPU SOURCE_PREP、完成证据与 TARGET_SCREEN 联合外部审阅

审阅日期：2026-09-19。仓库：DLwbm123/DPA-CTTA。本记录不涉及 SSL_CL_seg/AGMS_OBSERVER，也不沿用该项目权限。

## 1. 结论及准确范围

| 对象 | 固定版本 | 本轮结论 |
|---|---|---|
| SOURCE_PREP GPU 实现 | ba04ca46a6bae5df4f4ebb272ef4045f5ee400e5；证据 ea47b81727115d50abe4824d3d2ed1327377f3ab | PASS：回顾性代码差异及提交资格证据审阅，不是补签历史启动授权 |
| SOURCE_PREP 完成证据 | 1936775cf7bc25dee5bcc658480da45e11e63236 | PASS_SUBMITTED_COMPLETION_EVIDENCE：预算及终态记录支持一次完整源准备；不要求重训 |
| 六份产物身份及 release 证据 | 同一完成发布 | 接受已声明的六份文件/context 身份；真实落盘包 fresh-loader roundtrip 未验证，目标部署验收未关闭 |
| TARGET_SCREEN CPU 执行壳 | 9de62090fb921695aacd7d68027cf7ec853bcc47；证据 f5d1ef75859cf82cf039819c50732584812f4d9d | NEEDS_FIX：失败分支缺少模型计数保存；另有 GPU context 集成缺口 |

没有签发 TARGET_SCREEN 执行 PASS、GPU 操作许可或用户启动确认。当前三项 TARGET scope 均保持 NOT_RUN；next_scope_authorized=false。

## 2. 本轮实际工作及限制

通过 GitHub 连接器读取固定 ref 的索引、源码、差异、发布计数和相关绑定材料。Git 比较确认 GPU 实现相对旧 CPU 实现仅变更五个路径：GPU 检查脚本、共享 context/network、source runner、source 测试。source 后续两次提交及 target 发布追加的均是审阅文档/证据，不是被测生产代码变更。

独立工作包括：从此前已归档且完整 SHA256 一致的 CPU runner 中仅提取未改变的 expected_counts 函数，按 111/23/25 重算七类预算；重算已发布验证均值差值与资源上限；对逐字保存的 execute_job 函数片段执行三个替身控制流情形。未导入 Torch、未加载模型、未创建子进程、未查询 GPU、未读取真实资产、未访问服务器，模型及 optimizer 调用均为零。没有重跑 90/15/114 验收套件。

只有用于探针的函数片段已保存；不声称重新哈希了完整 target runner。RESULTS 数字从连接器读取后转录，不声称重新哈希原始完整 RESULTS 字节、私有 phase 记录或私有模型。匿名 HTTP 200 是用户提供的核验，本轮未独立证明；本地普通下载因 DNS 失败，连接器读取成功。

## 3. GPU 与完成证据

GPU 路线保留 CPU 预处理、局部随机数、方法模块、FP64 潜状态和 CPU loss；仅 FP32 主干执行迁至 GPU。FiLM 的 .to(h) 及输出 .cpu() 没有新增 detach。该补丁没有引入 AMP；configure_backend 禁用 TF32、要求确定性 cuBLAS 配置并启用确定性算法；GPU context 额外记录 execution_backend。旧 CPU PASS 不能直接打开 GPU 路线，新的用户过渡授权及资格文件另有检查。

提交资格记录包含 15 项检查，日志/FiLM 梯度比较、同 GPU 重复性、完整 16-step 合成 oracle、六份模型各一次 fit/cal/package 及两次源 VJP。该记录的 133 forwards、22 source backwards、6 calibration backwards、16 source Adam、6 AdamW、6 calibration Adam、2 source VJP 是生成资格成本。比较中的另外两次 autograd.grad 已被文字单列。不得把微步资格当作 CPU/GPU 完整训练等价或模型效果证明。

七类完整预算独立重算吻合：135328 forwards；9072 source backwards；1536 calibration backwards；3072 source Adam；1536 calibration Adam；6000 AdamW；1024 source VJP。提交报告声明 45 phases 完成，六模型各 1000 fit、256 cal、64 validation episodes；父子退出 0、源 after-check UNCHANGED、completion 已发布。supervisor 8650.284795 秒及输出 11602322 字节低于 259200 秒/536870912 字节上限。审阅不是实际服务器复验。

用户主动中断的 CPU 尝试另有 FAILED/KeyboardInterrupt、4335.490738 秒、4136 forwards、2067 backward/Adam 记录。不能把它写成已完成，也不能删除成本。源 GPU 使用的是新的用户过渡授权；本次不追认或改写过去的授权时间线。

## 4. TS-COST-1：TARGET 失败分支遗漏模型成本计数（阻塞）

位置：target runner 的 online()/execute_job()。当前顺序是 online 返回完整计数，验证总数，关闭模型，执行 posthoc，最后才 out.write('counts.json', counts)。异常分支只保存 first_error 和含 image_IO/mask_IO 的 worker.json。online 的局部累积计数不会在异常时返回；host 的已观测计数也未被该 finally 分支持久化。

独立探针运行的是保存的 execute_job 片段，在线训练、评分、模型、文件校验全部替换为标准库对象，没有真实 15608 次模型调用。三个情形如下：

- 成功对照：替身返回 F=15608、B=1951、Adam=1951，counts.json 正常保存。
- 在线已完成、posthoc 首次 mask 解码失败：首异常正确保留，但 counts.json 缺失，worker.json 没有模型计数字段。
- 在线部分执行失败：替身 host 已有 F=8、B=1、Adam=1 的观测状态，仍没有对应模型计数落盘。

这是失败成本记录缺口，不是已经发生的真实 target 失败，也不是源端结果作废。修复应在评分前保存完整 online 计数，在失败路径保留观测到的物理调用与已提交 visit 数，避免把 nominal budget 或预测文件数量当作精确成本。硬杀尾部未知时标记 lower bound/unknown，不能填写零。成本证据写失败不得覆盖原始首失败。

新增回归应覆盖评分失败、部分 online 失败、调用后预测写失败、统计记录写失败与源首异常组合。不能只把 counts.json 移到评分前就声称所有 partial-failure 分支闭合。

## 5. TS-BACKEND-1：旧 CPU TARGET 与新 GPU context 不兼容（集成阻塞）

旧 target device_policy 明确只接受 CPU；make_host 对 C_BASE 硬编码 CPU、对 C0/新方法调用缺省 CPU 的 load_model。GPU 源 context 则包含 execution_backend。共享 require_context 用完整 actual!=expected 拒绝不一致。即使参数 tensor digest 相同，也不能从原 CPU 路线得到匹配的 GPU context。

还需注意：现有 preflight 校验 context 的格式、摘要和 source/method 绑定，但不据 environment.execution_backend 与 CPU 资源进行早期拒绝。C_BASE 不消费六包 context，因而从调度调用顺序推断，错误后端包不一定在整个矩阵读取真实 target 之前暴露；可能直到 C0/新方法实例化才失败。这是静态路径推断，不是真实目标运行观察。应补充整批次资产/后端就绪检查，避免先消耗基线运行再发现其余臂不兼容。

这不是源包损坏，也不是在 receipt 里改 device 就能完成的工作。建议复用已验证的“GPU 主干、CPU 方法/FP64 state”桥接路线，做独立 target 窄补丁并保留原六个产物/context 字节。禁止删掉 backend 字段、重算原 context 来掩盖差异，禁止从 candidate 自造 expected。

GPU target 资格必须覆盖八个 arm，包括 C_BASE 的 8F/1B/1Adam、C0 的 1F、六个 FULL/STATIC 的 2F/0B/0Adam；资格成本与正式 24 jobs 分开。主方案采用相同已批准后端进行对照，不能不披露地把 C_BASE 留 CPU 而对新方法声称纯 GPU 成本优势。若确需不同后端，应另外明确比较范围及资格，不能隐式切换。

## 6. ART-ROUNDTRIP-1：真实落盘包部署证据待补（不是重训要求）

release 在序列化前对六份内存 weights 调用 inference_from_tensors；收尾检查了落盘文件长度、文件 hash 与 context 身份。真实 .pt 经 load_artifact 重新读取、weights_only 反序列化、重建方法及上下文匹配的 roundtrip 未被本次材料声明。

在明确的资格/只读加载授权下，对同一匹配 GPU backend 执行六次 fresh-loader 检查即可。读取独立保留 inventory/context，绑定原 checkpoint 与 effective tensor identity，不读取真实 source/target pixels，不训练，不改包。先记录 load-only 成本；若要做生成图像前向，需独立列出固定额外计数与用途。该证据可以与新 TARGET GPU 资格一起交付，不要因缺 roundtrip 重跑整个 SOURCE_PREP。

不得在执行前把 source_release.trusted_loader_verified 填成“真实落盘包已验证”；应按验收证据的实际语义填写。当前记录的产物身份接受与真正的部署就绪分开。

## 7. 科学解释与下一步

FULL-own-STATIC 的 source validation soft Dice OC 差值，按 100×原始 Dice 差计算：A +0.081081 个百分点、B +0.003190 个百分点、C +0.001102 个百分点。OD 差分别为 A -0.000560、B +0.000986、C +0.000153 个百分点。没有在这六个重复 query 均值上进行统计显著性检验；不得称为显著收益或显著无效。

没有 C_BASE/C0 的源端对照结果，也没有目标效果数据，因此没有 FULL-C 结论。不按 B 的源均值最高挑胜者，不删减 A/C，不调 rank/损失/seed。患者依赖及 checkpoint 暴露未知继续披露。

下一步仅为 TARGET 计数修复、GPU context 桥接、有限生成资格与六包加载证据准备；代码修复后用受影响的程序化检查验收。真实 GPU 资格和私有包加载需要单独用户明确许可，真实 24-job SCREEN 更不得自动运行。合并后的新 target SHA、source 六包身份、设备资源与登记/输出等绑定一次性交付复审。SOURCE_PREP 不重训；旧166不重跑；MECHANISM/EXTENSION不启动。

证据来源路径及固定 ref 见 SOURCES.json；机器审阅结论见 R7_EXTERNAL_REVIEW_RECORD.json；探针完整结果见 control_flow_probe_results.json。
