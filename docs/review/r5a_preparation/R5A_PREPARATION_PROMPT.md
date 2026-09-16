# R5-A 启动准备 Prompt（复审通过，不代替执行授权）

请读取随附 R5_REREVIEW.md 与 REVIEW_DECISION.json。本次外部代码复审结论是 R5_REVIEW_PASS_A_ONLY。F1/F2 已关闭；不要再次修改算法或扩展实验矩阵。

当前这份 prompt 允许整理启动材料，**本身不是 GPU 或真实资产执行授权**。只有用户另外明确授权 R5-A、实际设备及资源上限，才进入执行。已有 CPU fixture 卡号、R4 授权或旧 waiver 均不可沿用。

## 固定绑定

仓库 DLwbm123/DPA-CTTA。
实现 SHA：b4b71601a5bdf87bb3a7e5d3db352610adcdff74。
修复材料发布 SHA：b50348776c03aeb621cfbe9cfeac1068c4683412。
Science SHA256：89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5。
C fingerprint：71c8be1e77e1fe45eb9d9c84ba2c46edbc3936720117399eb342de4a89ccb829。
Registration：8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf。
Recurrence：cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db。
Scope：A；approved_trajectory_count=3。

## 准备工作

1. 检查工作区和已有 R5-A 产物，保留用户改动与旧结果，不覆盖、不重复执行已有合格轨迹。本次审阅结果只适用于上述实现 SHA；优先建立该 SHA 的干净 detached execution worktree。文档提交与执行提交分别保留。
2. 归档本次复审报告，登记 external_review.status=PASS、code_sha=上述实现、scope=A，reference 指向真实归档的报告。不要把这个代码 review 字段当成设备使用授权。
3. 不改科学文件、旧 C、依赖或阈值。固定 CTTA 依赖 dbff0d985c6c95345d9fb78f5b1daef57b392564；GraTa 依赖 33ae20d664f305af34739ec54a5bec7da53ffa0b。
4. 创建新的 A authorization/receipt 草案，默认 enabled=false。绑定真实已授权 registration、scope、SHA、fingerprint、三条矩阵、设备、worker 数、caps、私有输出与 smoke 配方。calibration=null、reuse_A=null；不要猜测 RANDOM p_accept。
5. 若当前没有用户独立的 R5-A 执行授权、允许的实际卡号或四项 caps，就交付 disabled 草案并停止，报告 R5A_PREPARED_AWAITING_EXECUTION_AUTHORIZATION。不初始化 CUDA、不加载真实 checkpoint/像素，不在后台等待。

## 仅在用户明确授权完整绑定后执行

- 只执行 C/order0、C/order1、C/order4，共 3 条，每条 1,951 内容。不得加入 HALF、RANDOM、VERIFY 正式轨迹。
- 每个实际设备一次原 A smoke：PROCEDURAL_OLD_C4_DIAG_C4_V1；旧 C 四步和 R5 C 四步；程序化 pixels 0/1/2/3、seed=20260907；64 forward、8 backward/Adam、0 VJP；rtol=1e-4、atol=1e-5。不得看结果后放宽容差。
- 完成所有实际设备 smoke 后才运行正式流。正式 46,824 forward、5,853 backward/候选 Adam、0 VJP。Smoke、失败前缀和其他真实物理成本单列，不伪装只用了正式预算。
- A 每图实际完整提交 C。记录 pre/q/trial 和影子规则；shadow_accept 不得改变参数、Adam、输出或增强 RNG。
- 所有 GT 只在状态完成后由 evaluator 读取。不得用标签、域名、ID、未来图像选择更新；subset 仅用于事后汇总。
- 保持中性入口、有限监督、自有进程组清理和 IO 错误政策。不自动重试、不按中间分数跳流。失败记录保留原异常和 live hook 成本下界，首次失败不覆盖。
- CPU 验证完整三流、身份/计数/metric 可实现性、decision replay、校准摘要及冻结 A gate。不加入阈值/学习率搜索，不新增 C0。
- 三流全部机械完整且 CPU 核验有效后，按真实 gate 输出 R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW 或 R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE。机械或审计失败输出 INCOMPLETE，不伪装科学负结果。
- 无论 A gate 如何都停止，next_execution_authorized=false、R5B=NOT_RUN。不启动 B_NEW 的 17 条。

## 交付

准备阶段交付：真实 review 归档引用、固定 checkout 记录、三条 metadata matrix、disabled A 授权草案及缺失的用户授权项；不自行填充批准。

若另外取得完整执行授权且已实际运行，则交付实际 implementation/receipt/binding、每设备 smoke 证据、三流逐轨迹成本、全部无标签/评价账本的去身份聚合、CPU 核验、A gate 每项检查、固定校准分子分母与停止状态。公开包不泄露原始像素、标签、权重、私有路径、逐内容身份或机器身份。

阶段 B 的审阅与授权留待真实 A 结果后单独处理。此次不再要求修改科学设计，也不把 CPU 通过写成方法有效。
