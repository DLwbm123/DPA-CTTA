# R5 外部代码审阅

## 结论

**R5_REVIEW_NEEDS_FIX**

本结论针对 Implementation SHA `0c08cece6a91bdf5e3d06c9862dcd192df7787d8`，审阅材料发布 SHA `ef06c90b3306d01f0f35e313e803f1fb03c50c31`。文档及代码声明的 Science SHA256 为 `89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5`。

核心候选规则和事务回滚未在本次检查中发现阻断性算法缺陷；两个已复现的异常处理/标量审计缺口应在 R5-A 前修复。它们不是已发生的正式结果错误：R5-A/B 均尚未执行。

本审阅不构成 GPU 执行授权，也没有改写远端仓库、历史交付状态或任何实验记录。

## 审阅范围与证据边界

通过 GitHub 连接读取固定 SHA 的 host/rule/analyze/evaluation/execution/plan/run、CPU 检查入口与测试，以及所依赖的 B1/R1 调用链、指标校验器、配对统计与有限监督器。核对实施报告、DELIVERY、PARITY_CONTRACT 和服务器最终测试日志。

发布日志报告本地及服务器最终 R5 套件均为 22/22，通过、零失败/错误/跳过。本次没有在原服务器环境复跑该完整套件。完整仓库 Git 下载未成功；本次有界 CPU 探针使用连接读取后转录的源文件及函数摘录。host.py / rule.py 经 Git blob SHA-1 验证与所审实现完全匹配；没有把函数摘录称为整个仓库副本。

独立探针环境：Python 3.13.5，Torch 2.10.0+cpu。未初始化 CUDA，未读取给定 checkpoint、真实图像、真实 mask 或源数据。该环境不同于原服务器；探针仅支持其注明的局部结论。

## 通过的有界检查

1. 四臂各 400 次规则判定，共 1,600 次，与独立历史列表、NumPy 线性分位数及独立随机数参考一致。包括越过 32/128 访问、空风险、候选拒绝仍入窗、每次 RANDOM 到达消耗一次随机数。
2. 完整 512×512 栅格的全空可靠区及部分非空可靠区计算符合定义。
3. 尚未初始化 Adam 时，候选更新后回滚可恢复参数、buffer、param groups、optimizer 空状态和 Parameter ownership。再执行下一步，与从未执行被拒绝候选的参考一致。
4. 已初始化 Adam 时，同样通过参数、矩、step、额外状态删除及下一步精确一致检查。参考分支没有调用被测 snapshot/rollback 函数。

这些不替代完整 ResUNet、官方增强和 GPU smoke。仓库原四步完整模型测试全处于 warm-up，不能当作完整网络在真实 eligible 接受/拒绝轨迹下的长期验证。

## F1：标量指标校验缺少可实现性下界

优先级：**P1，R5-A 前修复**。

位置：`src/dpa_ctta/r5_update_acceptance/analyze.py::join` 调用 `src/dpa_ctta/p2_analysis.py::validate_metric`。

该 validator 检查 intersection 的非负与上界，以及由记录的像素计数重新计算 Dice，但没有检查：

`intersection >= max(0, pred_pixels + gt_pixels - total_pixels)`。

复现：total_pixels=262144、pred_pixels=262144、gt_pixels=100、intersection=0，Dice=0、pred_full=true、其他 flags 与所记计数一致、ASSD=1.0。预测覆盖全图时 intersection 必须为 100，因此这个记录不可能来自任何二值预测/GT 对。现有函数仍然接受。

进一步使用原 replay/join 函数摘录和 33 条合成 ledger，在最后一条 eligible 访问将 trial/emit 换成上述不可能记录、pre 保持 Dice=0.9，join 仍返回 OD/OC 均为 -90pp 的 L，并保留 shadow_accept=false。本探针没有运行完整 recompute，也没有生成正式 gate 结论。

风险是坏标量可能成为看似有效的即时损伤证据，而不是已经证明 evaluator 会产生这种记录。

修复：新增 R5-local 包装校验器，保留旧 validate_metric 并增加四个混淆矩阵单元（TP、FP、FN、TN）均非负的检查；不要改历史 P2 行为。pre/q/trial/emit 全部走该校验。加入边界合法与非法计数测试，以及完整 A fixture 污染后 valid=false 的测试。

## F2：smoke 步内异常漏记已完成计算

优先级：**P2，按本轮审计要求在 R5-A 前修复**。

位置：`src/dpa_ctta/r5_update_acceptance/execution.py::smoke`。

当前 physical 累加发生在 `h.step()` 成功返回后。step 内部异常时，except 只写已完成访问累加值，没有并入本次失败访问的 live core counters。

有界故障注入：保留原 smoke 编排函数，使用程序化 CPU failing host，在真实完成三个小型 Linear 前向后抛异常。故障被正确重新抛出，但 smoke.failure.json 的 network_forwards 仍为 0；探针已完成前向数为 3。这是隔离编排层探针，不是原 CTTA 模型或 GPU 执行。

修复：使用调用前后 live core counter delta 或等价的可验证累计方式；异常路径也收集当前 host 已完成的 forward/backward/Adam。不能依据成功访问数乘 8/1/1，也不能重复累加已记步骤。若进程中断使计数不确定，明确记录可证实下界/未知项，不把缺失表示为精确 0。保持失败停机、清理和无自动重试。

补测至少覆盖三个故障点：若干 forward 后；backward 后、Adam 前；Adam 后、step 返回前。成功 A/B smoke 的冻结计算配额不变。

## 为什么既有 22 项通过未覆盖这些缺口

`test_r5_analysis.py` 包含直接修改 Dice 的污染测试，但这种修改会违反现有 Dice 重算关系；它没有覆盖“Dice 与所记计数一致、但计数本身违反集合可实现性”的情况。

现有 fixture 构造 smoke completion；R5 测试未实际覆盖 R5 smoke 步内失败的物理计数路径。原测试通过和本次补充发现不矛盾，不应删除或改写原日志。

## 修复后的验收与下一阶段

保持 science 原始字节、阈值、预算、四臂目标和 A/B 边界不变。只做 R5 局部审计/异常处理与新增测试。补入非自引用的回滚下一步参考测试。最终候选修复 SHA 上重跑完整 R5 CPU 套件与必需继承回归，再取得服务器 CPU 证据；不得把修复前的 22/22 改称修复后结果。

交付新 implementation SHA、窄 patch、首次失败与最终日志、DELIVERY，以及 F1/F2 逐项解决说明。最终停在 `R5_IMPLEMENTATION_READY_FOR_REVIEW`，执行仍 disabled，R5A/R5B 仍 NOT_RUN。

复审通过后，另行冻结实际设备、资源上限、新 receipt 和 A smoke，再单独授权 R5-A 三条轨迹。不得运行 B_NEW；A gate 通过也不自动授权 B。

## 源码证据

- 审阅入口：`https://github.com/DLwbm123/DPA-CTTA/blob/ef06c90b3306d01f0f35e313e803f1fb03c50c31/docs/review/r5/REVIEW_INDEX.md`
- R5 analyzer：`https://github.com/DLwbm123/DPA-CTTA/blob/0c08cece6a91bdf5e3d06c9862dcd192df7787d8/src/dpa_ctta/r5_update_acceptance/analyze.py`
- inherited metric validator：`https://github.com/DLwbm123/DPA-CTTA/blob/0c08cece6a91bdf5e3d06c9862dcd192df7787d8/src/dpa_ctta/p2_analysis.py`
- smoke：`https://github.com/DLwbm123/DPA-CTTA/blob/0c08cece6a91bdf5e3d06c9862dcd192df7787d8/src/dpa_ctta/r5_update_acceptance/execution.py`
- host / rule：上述 implementation 下 `src/dpa_ctta/r5_update_acceptance/host.py`、`rule.py`。
- tests：上述 implementation 下 `tests/test_r5_host.py`、`tests/test_r5_analysis.py`。

探针输出见 `probe_results.json`、`join_probe_results.json`。gap probe 的 passed=true 表示问题已按预期复现，不代表被审生产代码通过审阅。
