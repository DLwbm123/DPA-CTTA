# R5 修复复审：R5_REVIEW_PASS_A_ONLY

日期：2026-09-16  
审阅性质：本次对话中的外部代码复审、提交证据核对及有界独立 CPU 探针。不是第三方机构认证、完整环境复现或 GPU 验收。

## 1. 结论与绑定

**结论：F1、F2 已关闭，独立回滚参考已补齐；在此次修复复审范围内，没有发现阻断 R5-A 的问题。代码审阅通过仅覆盖 R5-A。**

- 仓库：`DLwbm123/DPA-CTTA`
- 本次通过的 Implementation SHA：`b4b71601a5bdf87bb3a7e5d3db352610adcdff74`
- 本次查阅的证据发布 SHA：`b50348776c03aeb621cfbe9cfeac1068c4683412`
- 上轮被审 Implementation SHA：`0c08cece6a91bdf5e3d06c9862dcd192df7787d8`
- Science SHA256：`89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5`
- C fingerprint：`71c8be1e77e1fe45eb9d9c84ba2c46edbc3936720117399eb342de4a89ccb829`
- 审阅 scope：`A`。不批准 B_NEW、自动晋级、搜索或组合。
- `execution_authorized=false`；本次未运行 GPU、R5-A 或 R5-B，也未写入远端仓库或服务器状态。

通过代码审阅不等于用户已授权使用设备、读取真实资产或启动实验。实际执行必须另有绑定上述实现 SHA、science、registration、stream、C fingerprint、3 条轨迹、设备与资源上限的新 A 授权。

## 2. 修复范围与发布一致性

核对的差异仅涉及：

1. `scripts/check_r5_cpu.py`：加入新增测试模块。
2. `src/dpa_ctta/r5_update_acceptance/analyze.py`：增加 R5-local 可实现性包装器。
3. `src/dpa_ctta/r5_update_acceptance/execution.py`：修复 smoke 累计计数、清理与失败证据处理。
4. `tests/test_r5_audit_fix.py`：新增 9 项测试方法。

差异中未修改 host/rule/evaluation/plan、旧 C/P2/R1–R4、science、固定依赖或历史结果。保持性文件声明 science 原始字节、C 指纹和 3/17/20 metadata dry-run 不变；该声明与核对的代码差异一致。[R1–R3]

另外直接读取了实现提交和证据发布提交的 Git 根树；两者的 src/scripts/tests/configs 以及其他非 docs 根项完全一致，只有 docs 树变化。[R4–R5]

| 树 | 两个提交共同的 tree SHA |
|---|---|
| src | `4c1542cad4638f4e60adf9109abd10fc4c167e00` |
| scripts | `2c4f67c150cd857e978c687d5fb82c6e2f00fb56` |
| tests | `304cb3022159484fc42d8258109c987bcd5f57d5` |
| configs | `1fb9f76adbd682bca7588dc6325f49027e78eb54` |

本次授权绑定建议仍使用被测试的 implementation SHA。不要把文档发布 SHA 静默替代为执行 SHA。

## 3. F1：像素计数可实现性——CLOSED

R5 包装器先调用既有 P2 validator，再检查：

`TP=I; FP=P-I; FN=G-I; TN=N-P-G+I`

四格均须为非负整数。这补上了 `I >= max(0, P+G-N)` 的下界，且 pre/q/trial/emit 统一经过包装器。历史 P2 validator 和 evaluator 评分逻辑未改。[R2]

本次独立探针在 N=1..8 的小栅格上枚举所有二进制集合对，直接建立真实可实现计数集合，再验证 2,024 个候选计数组合：494 个合法计数被接受，1,530 个非法计数被拒绝；全部与集合枚举参考一致。另验证原 512×512 反例及 6 个满图/下界边界情况。

在 33 条程序化有效 trace 上，依次污染 pre/q/trial/emit，均被明确以 `R5 infeasible TP/FP/FN/TN` 拒绝。trial/emit 同步污染，避免仅被输出分支不一致拦住。

完整 A fixture 的 `valid=false / INCOMPLETE` 行为由提交的新测试与最终两端日志支持；本次独立探针没有重跑完整 A recompute。[R6–R8] 计数可实现性是必要一致性条件，不等于重建了原图、真实 mask 或概率图。

## 4. F2：smoke 步内失败成本——CLOSED

修复后每个已构造 host 在 finally 中读取 live core 累计计数，且每个 host 只入账一次，覆盖已成功访问、先行完整臂和当前失败访问。正常配额仍必须精确匹配，失败则明确标为 `OBSERVED_HOOK_COUNTS_LOWER_BOUND_ON_FAILURE`。[R2]

失败过程中没有继续下一访问或下一臂；已有首失败不覆盖；失败证据写入 OSError 时保留原计算异常。正常/异常退出都会移除该 host/core 注册的 hooks。[R2]

本次独立探针使用原 smoke 函数摘录、受控 host 替身与真实 CPU Linear/autograd/Adam 操作，在五种 host 角色中分别注入三个故障点，共 15 个场景，观测计数与记录一致。每次均在第 2 次访问注入：

| 当前角色内部故障点 | forward | gradient hook | Adam-post hook |
|---|---:|---:|---:|
| 本访问第 3 个前向之后 | 11 | 1 | 1 |
| backward 之后、Adam 之前 | 15 | 2 | 1 |
| Adam 之后、step 返回之前 | 15 | 2 | 2 |

每个先行完整角色另贡献 32/4/4。成功 A/B 编排分别得到 64/8/8/0、160/20/20/0；另通过已有首失败保留和失败日志写入 EIO 的两个场景。所有这些是本地受控 CPU 探针，不是实际旧 C/GraTa 全路径或 GPU smoke。

提交的新测试另使用真实旧 C/R5 类、固定 GraTa 与程序化 Toy，覆盖对应故障点；最终两端完整日志均包含这些测试。[R6–R8]

Hook 未触发前或构造期间的中断仍可能存在不可观测计算，故失败计数的“下界”标记必须保留。PyTorch 的 step post-hook 定义为在 optimizer step 之后调用，也不能据此宣称所有中途硬终止计算均被精确记账。[R9]

## 5. 非自引用回滚参考——补充项完成

新增测试直接构造原 R1 C 参考。该参考从不接收被拒绝图，也不调用 R5 snapshot/rollback；仅同步已经消费的增强 RNG，随后比较同一下一图的输出、参数、完整 Adam 和 RNG。空 Adam 与已有 moments 均覆盖；参考比被测分支少 8 次前向。[R6]

本次独立探针还对未改动的 snapshot/rollback 函数重新运行了两个小型 CPU 参数/Adam 测试；独立参考没有执行候选、也没有使用被测回滚。两种状态下回滚与下一步均逐值一致。该探针不是完整 ResUNet/GraTa 的长期拒绝轨迹。

## 6. 测试证据分层

| 证据 | 结果 | 环境与边界 |
|---|---|---|
| 发布的最终本地完整日志 | 143/143，0 failure/error/skip，exit 0；311.89 s | Python 3.12.9 / Torch 2.6.0；提交方执行，本次读取核对 |
| 发布的最终服务器完整日志 | 143/143，0 failure/error/skip，exit 0；841.42 s | Python 3.10.6 / Torch 2.2.1+cu121；提交方执行，本次读取核对 |
| 本次独立函数级探针 | 10/10 组通过 | Python 3.13.5 / Torch 2.10.0+cpu；本次实际执行 |

143 项是 31 项 R5（原 22+新 9）与 112 项继承回归。两端跑的是同一套测试，不能相加为 286 项独立覆盖，也不能把本次 10 组与 143 项直接相加。[R7–R8]

本次没有 SSH 到服务器，也没有在当前容器中重跑整套 143 项。容器没有可用的完整固定 GraTa/batchgenerators 环境；直连获取整仓不可用。对远端代码使用 GitHub connector 阅读；独立探针采用明确标注的函数摘录和依赖替身。

host.py/rule.py 使用上轮留存副本，重新核对 Git blob SHA；修复差异没有修改这两份代码。新 wrapper/smoke 使用此次 connector 所读补丁的函数摘录；源码片段、脚本、输出和边界随证据包提供。所有独立探针均 `cuda_initialized=false`、真实 RGB/mask/checkpoint 读取为 0。

## 7. R5-A 的下一步与停止条件

进入的只有 C 在 order0、order1、recurrence(order4) 上的三条只读诊断轨迹。A 实际始终提交 C 更新；shadow_accept 只做诊断，不能真实拒绝。

正式预算：5,853 次访问，46,824 forward，5,853 backward/候选 Adam，0 VJP。每个实际设备先执行原样 A smoke：旧 C 四步+C_DIAG 四步，64 forward、8 backward/Adam、0 VJP；rtol=1e-4、atol=1e-5。若实际使用 g 台设备，计入 smoke 后预算为 `46824+64g` forward 和 `5853+8g` backward/Adam，g 必须由新的用户授权确定。

设备列表、最大 worker 数、caps、新私有输出、实际 checkout SHA 和执行授权必须全部绑定；不要沿用 R4 卡号或测试 fixture 内卡号。缺任何必需授权则只准备 disabled 配置并停止，不轮询等待。

三条轨迹完整结束后 CPU 核验与冻结 A gate；无论 gate 通过与否都停止。通过状态是 `R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW`，未通过是 `R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE`；机械失败仍为 INCOMPLETE，不能算科学失败。B_NEW 的 17 条不自动开始。

B 的正式状态拒绝、GPU rollback、单独 smoke 配方与执行许可不在此次 PASS 范围。历史 R4 逐内容 parity、C0 匹配和独立泛化也仍未被本次复审验证。没有这些数据时 H_t 保持 null，不补跑 C0。

## 8. 来源索引

所有仓库链接均锁定提交，不依赖浮动分支。

- [R1] 修复交付：[REVIEW_INDEX](https://github.com/DLwbm123/DPA-CTTA/blob/b50348776c03aeb621cfbe9cfeac1068c4683412/docs/review/r5_audit_fix/REVIEW_INDEX.md)
- [R2] 窄补丁：[IMPLEMENTATION.patch](https://github.com/DLwbm123/DPA-CTTA/blob/b50348776c03aeb621cfbe9cfeac1068c4683412/docs/review/r5_audit_fix/IMPLEMENTATION.patch)
- [R3] 保持性：[PRESERVATION.json](https://github.com/DLwbm123/DPA-CTTA/blob/b50348776c03aeb621cfbe9cfeac1068c4683412/docs/review/r5_audit_fix/PRESERVATION.json)
- [R4] 实现根树：[Git tree](https://api.github.com/repos/DLwbm123/DPA-CTTA/git/trees/b4b71601a5bdf87bb3a7e5d3db352610adcdff74)
- [R5] 发布根树：[Git tree](https://api.github.com/repos/DLwbm123/DPA-CTTA/git/trees/b50348776c03aeb621cfbe9cfeac1068c4683412)
- [R6] 新增测试：[test_r5_audit_fix.py](https://github.com/DLwbm123/DPA-CTTA/blob/b4b71601a5bdf87bb3a7e5d3db352610adcdff74/tests/test_r5_audit_fix.py)
- [R7] 本地日志：[final-local.log](https://github.com/DLwbm123/DPA-CTTA/blob/b50348776c03aeb621cfbe9cfeac1068c4683412/docs/review/r5_audit_fix/logs/final-local.log)
- [R8] 服务器日志：[final-server.log](https://github.com/DLwbm123/DPA-CTTA/blob/b50348776c03aeb621cfbe9cfeac1068c4683412/docs/review/r5_audit_fix/logs/final-server.log)
- [R9] 官方 hook 文档：[Optimizer.register_step_post_hook](https://docs.pytorch.org/docs/main/generated/torch.optim.Optimizer.register_step_post_hook.html)
- [R10] 本次新生成的本地独立证据：`probe_rereview.py`、`probe_results.json`、`probe_run.log`、`sources/`。具体替身与验证边界见输出 JSON。
