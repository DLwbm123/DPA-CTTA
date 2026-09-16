# R5 外部审阅修复

本轮只修复 R5 局部审计/异常路径及测试，不启动真实模型实验。旧审阅结论 R5_REVIEW_NEEDS_FIX 保留在 external_evidence/，当前等待复审，不自行签发 PASS。

Implementation SHA：b4b71601a5bdf87bb3a7e5d3db352610adcdff74。基线文档发布 SHA：ef06c90b3306d01f0f35e313e803f1fb03c50c31；被审实现 SHA：0c08cece6a91bdf5e3d06c9862dcd192df7787d8。新分支 review/r5-audit-fix-v1。后续文档提交不改变已测试实现。

Science 原始字节 SHA256 保持 89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5。完整 metadata dry-run 与原交付 JSON 逐结构相同，四臂、3/17/20 条、预算、阈值、规则、默认 disabled 与 C 指纹保持不变，见 PRESERVATION.json。未改 host/rule/evaluation/plan、旧 C、P2/R1–R4、固定依赖或历史结果。

## F1：R5-local 指标可实现性

analyze.validate_metric 先调用原 P2 validator，再重建 TP=intersection、FP=pred-TP、FN=gt-TP、TN=N-pred-gt+TP，要求四者均为非负整数。join 原有 pre/q/trial/emit 路径统一调用此包装器；历史 P2 validator 原样保留，evaluator 和 Dice/ASSD/GT/阈值均未改。

新增边界测试覆盖 pred_full/gt_full 不合法交集、交集下界恰等合法及低一像素非法、空/非空/合法满图、非整数和直接 Dice 污染。33 条合法 C 影子 trace 的第 33 条 eligible/shadow_reject 访问，可用完全自洽的 flags/Dice 但 TN<0 污染每种预测；trial/emit 同时污染以保留输出分支相等，join 仍拒绝。完整 3×1951 的 A fixture 先取得有效结果再污染，recompute 抛异常，current_result.valid=false、status=INCOMPLETE，并移除 current 指针；不把损坏记录发布为科学负结果或晋级结论。

## F2：异常访问的物理计数

smoke 不再依据成功返回的每步 trace 累加。每个已构造 host 的 finally 块读取 live core 累计 forward/gradient/Adam-post counters 并仅累加一次；前面成功的访问、前面完整的臂与当前失败访问都进入总数。旧 C 直接读取自身 counters，四个 R5 臂读取 core counters。拒绝不会减去物理调用，成功配额仍单独核对。

finally 在正常和异常路径都卸载当前 host/core 的 hook、释放局部引用并调用 gc；既有有限监督、自有进程组清理和 worker 失败停机逻辑未改。异常继续原样传播，无下一访问/下一臂或自动 retry。失败 JSON 使用原有排他写入：已有首失败不会覆盖；如果写入遇到 OSError，stderr 记录落盘问题，仍传播原计算异常。

所有失败计数明确标记 OBSERVED_HOOK_COUNTS_LOWER_BOUND_ON_FAILURE，说明 forward/gradient/Adam-post hook 之前或构造中断的计算不可观测；没有把它表述为完整精确成本，也没有以成功访问数乘 8/1/1 填补。VJP 在本 smoke 实现中不存在，因此仍为 0。正常完成的 live 计数必须精确匹配冻结配额。进程被不可捕获信号杀死时无法保证写出新 JSON，仍由既有监督器判为失败；没有伪造成功或精确零。

CPU 故障注入走真实旧 C/R5 类、固定 GraTa 和程序化 Toy，只替换构造网络，不以 mock step 模拟计数。每个故障点覆盖 OLD、C、C_HALF、C_RANDOM、C_VERIFY，第 2 次访问注入以同时检查此前计数。独立 hook 观测的当前臂计数分别为：前向第 3 次后 11/1/1/0；backward 后 Adam 前 15/2/1/0；Adam 后 step 返回前 15/2/2/0。第 k 个完整先行臂额外贡献 32/4/4/0，最终计数不漏不重。A/B 成功编排分别实测 64/8/8/0 与 160/20/20/0。这些为 CPU fixture，不是 GPU smoke。

## 非自引用回滚参考

新增参考直接构造原 R1 C，参考从未接收被拒绝图，也不调用 R5 snapshot/rollback；用调用即失败的 patch 明确保护该边界。被测 VERIFY 执行候选和回滚后，仅复制其已经消费的增强 RNG 到参考，下一图逐值比较输出、参数、完整 Adam 与 RNG。初始空 Adam 和已有 moments 均覆盖；参考实际 forward 比被测少 8，证明未执行该候选。没有生产强制接受/拒绝开关，保留原回归。

首版新测试错误地直接调用包在 R5 host 内的 core，违反 R5 的 CANDIDATE hook 生命周期；因此两个子例报错。该错误属于测试参考构造，已改为完全独立原 C，不改变生产 host 或门控以使测试通过。red01 与 fix01 日志均完整保留。

## 验证与证据边界

新 R5 完整套件 31 个 test methods（原 22 + 新 9）；继承回归 112 个，合计 143。最终本地和服务器均在上述 implementation SHA 上执行完整 143 项，本地 143/143 通过、零失败/error/skip，311.89 秒（Python 3.12.9 / Torch 2.6.0）；服务器同样 143/143 通过、零失败/error/skip，841.42 秒（Python 3.10.6 / Torch 2.2.1+cu121），检查后源码工作区干净。两端 B1 范围实测均为 2466 forward / 306 gradient-hook / 301 Adam-post / 0 VJP；该计数不覆盖继承非 B1 路径，不能当作所有方法总成本。完整原始证据见 logs/INDEX.json 和 JSON。不同运行相互重叠，不将次数相加成独立覆盖。unittest 的 failures/errors 包括 subtest，首次 9 个 methods 的失败事件数可以大于 9。

red01：修复前生产代码、新测试首版，复现 F1/F2，另含上述独立参考测试错误及旧日志排他写掩盖原异常的错误。fix01：生产 F1/F2 已修，0 assertion failure，但独立参考测试仍有 2 个 errors。fix02：纠正参考后 9/9 通过。最终日志另外保留，不拿旧 22/22 代替本轮结果。继承测试中的刻意注入 IO 异常栈仍保留，按 unittest 结论区分预期故障。

所有像素/模型权重仅为现场程序化 fixture 或随机初始化网络；原完整 ResUNet 四步 CPU 匹配仍在完整套件中。CUDA 初始化被禁止，真实 RGB/mask/checkpoint/source 读取为 0。服务器源码依赖 SHA 和 clean detached checkout 已核对，中性 CPU argv 已检查。没有查询/使用 GPU，没有后台任务、正式 A/B、source 训练或新组件。外部证据来自不同 Python/Torch 的有限探针，按其 README 原边界保存，未作为本轮完整套件结果。

真实 R5-A/B 仍 NOT_RUN。历史 R4 逐内容 parity、真实 GPU 数值或长期 eligible 轨迹并未因此获得验证。可选 C0 匹配仍未核实，H_t=null。本修复只关闭两项审计缺口，不能据此宣称科学有效性或外部复审通过。
