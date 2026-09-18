# R7 SOURCE_PREP 外部审阅：NEEDS_FIX

## 结论与范围

审阅实现：`9fa2ce37149909687ef5be2e08aa2b32a5887903`。证据发布：`ad95f212b8770e5006e0b1ea3520717767c7413f`。

本次判定为 **R7_SOURCE_PREP_REVIEW_NEEDS_FIX**，只针对新增 SOURCE_PREP 执行层。原 Stage I PASS 保留为既有外部结论，不重新签发、不重新打开已关闭的历史 F1。本文以 SP1–SP3 命名本轮新问题，避免与历史编号混淆。

来源为固定提交上的公开源代码、准备契约、CPU_RESULTS、测试文件、绑定记录及提交比较。原始 17+59 验收没有在本环境重跑；私有源清单和用户电脑上的交付 ZIP 没有读取。公开仓库无法通过本环境的 git 网络下载，源代码读取由 GitHub 连接器完成。

另行执行了标准库隔离复现：路径检查片段，以及按公开内容复制的 supervise_one 函数。进程、cleanup、输出失败、时钟使用明确的测试替身。未导入仓库训练代码，未启动真实源进程，未读取真实源图像/checkpoint，未进行 GPU 查询。复现说明的是相应控制流，不是完整真实 SOURCE_PREP 运行结果。

没有写回仓库，没有授予执行权限。以下阻塞项已足以拒绝当前实现的执行层 PASS；不宣称已穷尽所有问题。

## SP1：输出路径规范化缺口

位置：`src/dpa_ctta/r7_source_prep/runner.py:50–54`，以及 `src/dpa_ctta/r7_shared/io.py` 的 Output 初始化。

preflight 对 output、source、storage 等使用 Path.absolute()，随后进行词法的 is_relative_to / parents 判断。该操作没有消除 `..`。Output 后续的 mkdir 没有额外的规范化隔离校验。

临时目录复现：批准存储是 `approved_storage/`，保护源目录是 `source/`，输出写成 `approved_storage/../source/new_run`。当前三个相关路径 guard 都不拒绝，实际 mkdir 落在 source 内，并在批准存储外。不是 symlink 或 hardlink 替换；不需要并发攻击。

建议修复：对所有参与比较的路径一致地处理规范化；最小策略可以直接拒绝 `..`。保留在规范化之前的既有 symlink 拒绝，不要用 resolve 掩盖链接。对现有父目录进行严格验证后，以规范化路径核对存储包含关系及 source/code/checkpoint/metadata 的双向不相交关系，创建前完成检查。

回归应覆盖：普通合规输出、storage 的 `..` 逃逸、通过 `..` 落入 source/code/metadata、source/storage 自身带非规范化分量，以及既有 symlink/hardlink 拒绝。被拒绝时不能产生输出文件或进入资产读取。

## SP2：父监督器的证据写入异常覆盖首失败

位置：`runner.py` 的 supervise_one finally 部分。

首失败被存入 first，但随后调用 out.evidence 没有保护。若证据写入抛出 OSError，控制流不会到达最后的 `if first: raise first`，原始失败被替换，后续终态证据也可能不再尝试写出。

隔离复现：合成子进程 exit=3，预期首失败为 RuntimeError("source worker nonzero exit 3")；让 evidence 抛出合成 OSError，外部实际收到 OSError，异常 context 为 null。不是原 17 项测试的重跑。

建议修复：父进程也使用不会替换既有 first 的证据落盘辅助函数。执行失败、cleanup 失败、evidence 写入失败分开记录；磁盘证据不可写时保留受限 stderr 回退。已有 first 时继续抛出它；没有已有 first 时，写证据失败必须令运行失败。不能为了保留首失败而吞掉所有证据故障，也不能重试真实训练。

回归应覆盖：启动失败/非零退出/超时与 evidence 写入失败的组合；cleanup 失败与 evidence 失败的组合；无先前失败但 evidence 失败的情形。检查抛出的主异常、后续证据尝试、cleanup 和 retry=false。

## SP3：子进程退出后的资源终态检查缺失

位置：`runner.py` 的 supervise_one 轮询循环，以及 main 的 completion 发布路径。

时限和输出目录总量仅在 `while process.poll() is None` 内检查。子进程已经退出，或在最后一轮轮询后写出大量内容并退出时，这些检查可能不再执行。main 后续检查 worker execution.status 和 source_after_check，但不补全树资源终态校验。

隔离复现 A：cap=200000 bytes，合成 worker 日志和父进程记录共 220188 bytes，supervisor 仍写 complete=true。

隔离复现 B：cap=1 秒，测试时钟在 start 返回时为2秒，supervisor 报 wall_seconds=2 且 complete=true。这是测试替身时间，不是真实训练计时。

这些结果只证明监督器的终态判定缺口，不声称已经在真实训练中发生超额，也不声称复现了完整 main 的 SOURCE_PREP_COMPLETE。

建议修复：退出状态判定与资源检查不能互相跳过。子进程退出后、成功终态发布之前，必须核对耗时和整个输出树（包括 worker.log、父子元数据和诊断）的资源总量。为后续所有终态证据预留全局额度，不能只依赖各自 BudgetOutput.used。超额应失败并保留证据，不能截短模型预算或重试。

回归应覆盖：已经退出的 child；最后一轮写入突增后 exit0；时限边界；总输出包含日志和父子记录；终态证据预留；已有 worker 失败不能被后续资源审计异常改写。

## 已有工作如何处理

提交的 CPU_RESULTS 记录了两端各17新增+59回归零失败，这仍是既有用例的验收证据，不等同于覆盖上述边界。新的问题不要求删除或重写旧 PASS 日志；应追加 before-fix 失败/暴露记录和修复后的最终 SHA 验收记录。

源码中的授权 scope 检查与 Stage I PASS 分离；默认资源 disabled，GPU 路径拒绝；注册内容组不冒充患者。这些设计应保留。

按固定111/23/25拆分，源码预算的算术复核为135328 forwards，9072 source backwards +1536 calibration backwards，3072 oracle Adam +1536 calibration Adam，6000 AdamW，1024 VJP。这只是预算算术，不是本次模型调用，不能据此或据短合成验收推算真实训练耗时。

CONTENT fallback 属于既有协议路径，不应因缺少患者关联而重新设计算法；但内容组不证明患者独立，注册图像hash不重叠也不证明患者无重叠，checkpoint文件hash不证明预训练无暴露。未知字段继续如实保留。未来的风险接受、文件绑定、资源批准和用户执行授权应分别说明，不能靠把 PENDING 改成 BOUND 来替代。

运行后有效tensor与训练产物hash只能在相应获批操作后产生；不能为了准备状态完整而伪造，也不能把尚未训练的产物hash要求成训练启动前已存在。

## 下一步任务边界

仅进行 SOURCE_PREP 执行层定点修复及程序化验收。保持 A/B/C 公式、四份 science 原始字节、FULL/STATIC 同预算独立训练、fit/cal/val 角色、查询不更新状态、24/9/至多12目标矩阵以及历史结果不变。

将上述回归加入原执行层测试，在修复后的准确实现 SHA 上重跑扩充后的执行层测试与59项R7回归；记录实际数量、耗时、失败、跳过及合成调用，不预定虚假的“新增测试数”。没有依据要求重跑旧166项，亦不得把缺少真实数据作为扩展算法工作的理由。

更新实现绑定、窄patch、预算/失败契约和新增日志，旧开发失败及既有审阅材料保持。新的纯证据发布 SHA 继续与实际验收实现 SHA 分开。

交付修复后重新等待外部执行层复审。不得真实源训练、加载真实checkpoint、解码真实图像、GPU查询/smoke、目标实验或自行启动后台任务。后续真实SOURCE_PREP也必须另有精确绑定的新授权，不自动承接本次审阅。

## 本次状态

```text
R7_SOURCE_PREP_REVIEW_NEEDS_FIX
stage_I_review=PASS
source_prep_execution_layer_review=NEEDS_FIX
source_binding_status=PENDING
real_source_training_started=false
real_target_execution_started=false
SOURCE_PREP=NOT_RUN
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
execution_authorized=false
```

## 包内文件与运行方法

`review_probes.py`：独立标准库控制流复现；`probe_results.json`：本次实测输出；`REVIEW_RECORD.json`：机器可读结论；`MANIFEST.json`：包内文件SHA256。

```bash
python review_probes.py
```

该程序仅在 TemporaryDirectory 内写入合成小文件，并使用进程测试替身；不要求源数据、仓库依赖、GPU或执行receipt。

## 固定源码与材料索引

实现提交下：
- src/dpa_ctta/r7_source_prep/runner.py
- src/dpa_ctta/r7_source_prep/registry.py
- src/dpa_ctta/r7_shared/io.py
- src/dpa_ctta/r7_shared/preparation.py
- src/dpa_ctta/r7_shared/source.py
- tests/r7/test_source_prep.py
- configs/r7_source_prep.defaults.json

证据发布提交下：
- docs/review/r7_source_prep/REVIEW_INDEX.md
- docs/review/r7_source_prep/IMPLEMENTATION.md
- docs/review/r7_source_prep/IMPLEMENTATION_BINDING.json
- docs/review/r7_source_prep/CPU_RESULTS.json
- docs/review/r7_source_prep/SOURCE_BINDING.json
- docs/review/r7_source_prep/RUN_CPU.md
- docs/review/r7_source_prep/input/R7_SOURCE_PREP_PREPARATION_PROMPT.md

提交比较确认：发布提交比被测实现多一个证据提交，所列变更位于该审阅证据目录；不能把证据提交声称为方法重跑的实现SHA。
