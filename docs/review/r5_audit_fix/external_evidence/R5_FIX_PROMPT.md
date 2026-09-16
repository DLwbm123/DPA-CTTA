# R5 外部审阅后修复 Prompt

请依据随附 R5_EXTERNAL_REVIEW.md 修复 R5。当前仅授权代码、程序化 CPU 验证和交付；不启动 GPU smoke、真实图像/mask/checkpoint 模型运行、R5-A、R5-B 或后台任务。

## 固定基线

仓库：DLwbm123/DPA-CTTA。
被审 Implementation SHA：0c08cece6a91bdf5e3d06c9862dcd192df7787d8。
被审文档发布 SHA：ef06c90b3306d01f0f35e313e803f1fb03c50c31。
Science SHA256 必须保持：89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5。
建议新分支：review/r5-audit-fix-v1。

先检查已有修复和工作区，保留用户改动。只修两个审计/异常路径问题及必要测试，不改变科学定义，不修改旧 C、P2/R1–R4 逻辑、旧结果或固定依赖。

## F1：R5 标量指标可实现性

R5 analyze.join 目前直接使用 p2_analysis.validate_metric。该函数允许以下不可能记录：N=262144、pred_pixels=N、gt_pixels=100、intersection=0、dice=0，且所有 flags 与记录计数一致。原因是没有验证 TN=N-pred_pixels-gt_pixels+intersection>=0。

在 R5 中新增局部 wrapper，先执行既有 validator，再要求 TP/FP/FN/TN 均为合法非负整数。等价必要下界为 intersection>=max(0,pred_pixels+gt_pixels-N)。pre/q/trial/emit 全部必须调用新 wrapper。不修改历史 P2 validator，不改变 evaluator 的 Dice/ASSD/阈值或 GT 解释。

测试至少包括：
1. pred_full 且 intersection<gt_pixels 必须拒绝。
2. gt_full 且 intersection<pred_pixels 必须拒绝。
3. pred+gt>N 时 intersection 恰好位于下界合法，低一像素非法。
4. 各类空/非空、合法满图边界维持原行为。
5. 直接 Dice 污染继续拒绝。
6. 构造33条有效影子trace，污染最后trial及对应emit的计数后，join必须失败；不能只测试修改Dice造成的不一致。
7. 完整A fixture中的不可能指标必须让recompute失败且current_result.valid=false，不能发布科学负结果或晋级结论。

## F2：smoke 异常时物理计数

当前 execution.smoke 在 h.step 成功返回后才累加 physical；若 step 内已执行计算后抛异常，smoke.failure 会漏计。

修改为从 live core counters 获取准确累计或调用前后delta，覆盖当前失败访问；保证旧C与R5各臂适配，已完成访问不重复累加，候选被拒绝也不减计数。不得用成功访问数乘8/1/1替代测量。

异常时先确保既有自有进程/资源清理约定不受破坏，再保留可证实计算计数与原异常。若确有未完成/in-flight计数不可确知，显式记录下界或未知，而非伪装精确0。不得自动重试，不覆盖首失败日志。

程序化CPU测试至少在下列时点注入异常：若干前向完成后；backward后Adam前；Adam后step返回前。验证异常仍传播、无下一步继续、failure物理计数不漏不重、不会发布成功completion。测试成功A smoke仍为64/8/8/0，B既定smoke仍为160/20/20/0；测试不加载真实checkpoint或像素。

## 非自引用回滚回归

补一个真正独立的下一步参考：参考分支从未执行被拒绝候选，也不调用被测snapshot/rollback。被测分支执行候选、回滚后，二者在同样的下一步输入/已消费增强状态上比较参数、完整Adam与输出。覆盖初始空Adam和已有moments两种状态。保留现有测试，不新增生产强制接受/拒绝开关。

## 验证与交付

在最终修复候选SHA上运行完整R5 CPU套件和原任务要求的继承回归，取得服务器CPU检查。所有像素/权重均为现场程序化fixture或随机模型；CUDA仍不得初始化。若依赖或服务器权限缺失，如实给出未运行项，不把旧22/22当作新结果。

保存首次失败、修复、最终完整日志，区分测试次数与相互重叠的套件。检查Science原始字节SHA不变，四臂/3-17-20矩阵/预算/规则/门限/C路径与默认disabled行为不变。

交付新Implementation SHA、F1/F2解决说明、窄patch、CPU日志/JSON、新测试清单、DELIVERY和REVIEW_INDEX。旧外部审阅指出NEEDS_FIX的历史保留，不自行签发复审PASS，也不复用R4 waiver。

若必需检查全部通过，最终仅写：
R5_IMPLEMENTATION_READY_FOR_REVIEW
execution_started=false
external_review=AWAITING_REREVIEW
R5A=NOT_RUN
R5B=NOT_RUN

若检查未全部通过，写实际BLOCKED状态及证据。到此停止。
