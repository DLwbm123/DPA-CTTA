# R3 外部差异复核：R3_REVIEW_PASS

## 1. 结论与范围

**外部结论：`R3_REVIEW_PASS`。**

- 本次通过的完整实现：`d601496a0827af3e1fe728612a4b9f17a613955d`。
- 上轮被审实现：`6d8fc7317506400b039d3d2ed41bca18523beb27`。
- 本次证据发布版本：`8b0102884c2aee6d61f152c6933ef13a6b81ff3b`，不是实际运行版本。
- 上轮唯一阻断项 `R3-ENV-01` 已关闭；本次差异未发现新增阻断项。
- 五个框架、17 臂的上一轮方法审阅结论保持，不重新开展公式、参数、阈值或效果审查。

这是代码审阅通过，不是已经完成 GPU smoke、真实 checkpoint 验收或正式性能验证。设备、并发和后台运行仍需用户对本批明确授权。本报告本身不启动任何作业。

## 2. ENV-01 的闭环

### 创建子进程之前

`src/dpa_ctta/r3/execution.py:33–39` 的 `worker_environment()` 对传入环境复制后处理：smoke 强制 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；formal 删除该键；未知阶段拒绝。

`launch.start` 在调用 `Popen` 前使用这个函数。父环境不被修改，CUDA 可见设备、job、packet、worker 和入口等原字段保留。既不使用 `setdefault`，也不依赖用户 shell 预先 export。

| 父变量 | smoke 子进程 | formal 子进程 |
|---|---|---|
| 不存在 | `:4096:8` | 键不存在 |
| 空字符串 | `:4096:8` | 键不存在 |
| 非法字符串 | `:4096:8` | 键不存在 |
| `:16:8` | `:4096:8` | 键不存在 |
| `:4096:8` | `:4096:8` | 键不存在 |

### worker 二次核验

`worker()` 先走原 `context()` 完成授权及绑定，再核验阶段环境，之后才加载 checkpoint、查询 backend 或开始模型工作。直接 `RUN_MODE` 入口不再静默接受不合规环境。未经授权的请求仍优先按原授权规则拒绝。

### 记录真实运行政策

`backend_policy()` 读取实际 `deterministic_algorithms`、`warn_only` 与 workspace，不查询或初始化 CUDA。

smoke 保留进入比较上下文前的 `backend`，另外在 `deterministic_smoke_pair()` 内记录 `paired_comparison_backend`。两者不再混用。正式阶段记录自己的实际开关，不把 smoke 的严格确定性政策加入正式算法。原上下文在成功和异常退出时均恢复调用前的两个 PyTorch 开关。

CPU 直接调用 `smoke(...,device='cpu')` 不需要 cuBLAS 环境前提。这与实际 GPU worker 必须符合环境政策是两个不同层面的条件。

PyTorch 2.6 文档对 CUDA ≥10.2 的严格确定性 `mm/mv/bmm` 列出了合法 workspace 配置；本修复解决的是该入口前提，不构成整个程序跨硬件逐比特可复现的保证。

## 3. 本次代码变更没有改变方法

核对窄 patch 和提交差异：一份 R3 运行文件与三份测试文件变更：

- `src/dpa_ctta/r3/execution.py`：阶段环境构造、入口断言、backend 证据。
- `tests/test_r3_execution.py`：六项新环境回归与对应 fixture。
- `tests/test_r3.py`：原 38-update 程序化模型检查增加 backend/恢复断言，无新增模型调用。
- `tests/test_r1_fixes.py`：仅增加 IO 故障测试的清理观测、PID/returncode/wait 证据与 traceback 输出，原断言保留。

共享 supervisor、IO/NFS/evidence、T/U/S/M/G 数值实现、source checkpoint、85-job 计划、四主序与 secondary 规则未因本修复而改写。冻结材料记录了对应不变性；本次本地另核对 science 的完整 Git blob 与 SHA256。

## 4. 测试证据分层

### 交付环境中的完整回归

公开日志 `env_fix/logs/cpu-full-01.log` 记录：104 tests，0 failures/errors/skips，exit code 0，约 253.28 秒；Python 3.12.9、PyTorch 2.6.0，CUDA 未初始化。104 个不同测试为此前 98 个加上 6 个新测试，targeted 检查与其重叠，不能相加为新的独立测试数。

输入是程序化张量、随机模型权重，IO 检查只用新建程序化文件。未读取登记的真实 checkpoint、目标 RGB/mask 或源数据。本次外部复核没有在本环境重跑完整 104 项。

公开程序化 ResUNet CPU 证据中的计数仍是 316 次前向、38 次 loss backward、38 次 Adam、18 次实际 VJP、5 次参数替换。它们不是 GPU smoke 结果。

实际 backend 证据为：调用前 `False/True/None`，比较上下文内 `True/False/None`，退出后恢复为 `False/True/None`，依次对应 algorithms/warn-only/workspace。workspace 为 None 合法地反映本次为 CPU 检查。

首个 targeted 失败来自新测试误取 `trajectory.call_args.args[6]`（checkpoint IO）而非 `[5]`（backend）。失败日志确实显示 `{'bytes': 7}` 与 backend 比较；当前测试已取 `[5]`。不需要据此修改生产函数或科学配置。失败和后续通过的文件都保留。

### 本次外部独立检查

**9/9 通过，0 failures/errors/skips。** 本地 Python 3.13.5、PyTorch 2.10.0+cpu，CUDA 未初始化。

这些检查直接编译被审 `execution.py` 的原 AST 函数，未改写待检验函数。完整 execution 源字节从用户提供的上一轮审阅归档与公开窄差异恢复，并核对当前 Git blob `7b2266d647111f816c8faff6507f7070a7e3f9eb`。网络原始字节下载失败，未把恢复文件描述为成功网络下载。确定性上下文使用通过连接器读取的原函数片段，明确不把片段摘要说成整份 m2_run.py 的本地字节核验。

覆盖内容：

1. execution/science 字节身份。
2. 十种阶段环境组合、父环境及其他字段不变、未知阶段拒绝。
3. 原 `launch.start` 的十次记录型 Popen 调用，确认环境在 spawn 前构造。
4. 原 worker 对八种错误环境在资产/设备阶段之前拒绝。
5. 授权检查优先于环境检查的控制流顺序。
6. 实际 PyTorch API 与原上下文的四种开关初态、正常/异常退出共八种恢复场景。
7. 原 smoke 在第一个模型构造之前注入异常，确认严格开关、失败记录、退出恢复。
8. 正式 worker 对实际调用者政策的记录和保持。
9. 根据冻结 science 重新计算 17×5 规模及各卡数预算。

资产、模型、授权内容验证和设备查询的外部协作者使用明确替身；这些协作者的内部逻辑不是本次独立检查对象。没有真实子进程、模型前向、GPU、私有数据或 checkpoint。详细代码、日志与 JSON 见 `evidence/`，不与交付方的 104 项重复或合并计数。

## 5. 保留的非阻断风险

### 历史 EIO 单次异常

**本次未重现，根因未知。** 本轮公开全套中的 persistent-EIO 检查，四个自有子进程在测试兜底清理前已返回且回收；无关对照仍存活。随后兜底清理结束了对照。故障产生非零退出是该测试的预期结果，不是 unittest 失败。

新观测器仍调用原 stop_owned，记录此前缺少的异常和退出状态；没有修改共享监督器。不要将最新通过改写为根因已定位，也不需要为了获得重复绿灯继续添加诊断轮。本次外部复核未重复进程压力测试，沿用上一轮有限复现和这次真实回归证据。

### 图零边计账

上一轮披露的 G 零边短路时 iterations 字段可能表示计划迭代数的非阻断限制保持。结果解释时结合 active_edges 与实际短路逻辑披露，不把未执行的迭代当成实际计算。本次不要求重新修改 G 数值路径。

## 6. 通过的固定绑定与预算

```text
EXTERNAL_REVIEW_STATUS = R3_REVIEW_PASS
IMPLEMENTATION_SHA = d601496a0827af3e1fe728612a4b9f17a613955d
SCIENCE_SHA256 = 73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e
REGISTRATION_DIGEST = 8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
SECONDARY_STREAM_DIGEST = cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
```

science 本地完整 SHA 已核验；registration 和 secondary 摘要沿用公开冻结材料，本次未读取私有身份序列独立重算。运行时由既有入口核验。

正式 85 条轨迹，165,835 条评分/Adam/loss backward，1,385,210 次网络前向，额外 VJP 至多 234,120 次。每块实际设备的 smoke 为 38 次 Adam/loss backward、316 次前向、VJP 至多 24 次。

| 实际设备数 | 正式评分 | 含 smoke Adam/loss backward | 含 smoke 前向 | 含 smoke VJP 上限 |
|---|---:|---:|---:|---:|
| 1 | 165,835 | 165,873 | 1,385,526 | 234,144 |
| 2 | 165,835 | 165,911 | 1,385,842 | 234,168 |
| 3 | 165,835 | 165,949 | 1,386,158 | 234,192 |

四主序与 secondary 单独汇总，不能变成五序平均；1,951 个内容每条流各出现一次，不能把 165,835 条评分说成同等数量的独立患者。

## 7. 下一步

代码修复与审阅阶段结束。在用户明确本批物理 GPU、并发和后台权限后，使用清洁的上述实现版本执行每卡一次既定 smoke，再自动完成原有限矩阵，最后 CPU 汇总和公开交付。不增加效果准入，不调参、不自动组合、不增加任务或重试。

## 来源

- 当前索引：https://github.com/DLwbm123/DPA-CTTA/blob/8b0102884c2aee6d61f152c6933ef13a6b81ff3b/docs/review/r3/REVIEW_INDEX.md
- 修复报告与窄 patch：https://github.com/DLwbm123/DPA-CTTA/tree/8b0102884c2aee6d61f152c6933ef13a6b81ff3b/docs/review/r3/env_fix
- 当前生产入口：https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py
- 当前环境测试：https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/tests/test_r3_execution.py
- 冻结 science：https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/configs/r3_science_v1.json
- PyTorch 2.6 对应规则：https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html
- 本地独立证据：`evidence/INDEPENDENT_CHECKS_RESULT.json`、`evidence/INDEPENDENT_CHECKS_LOG.txt`、`evidence/SOURCE_IDENTITY.json`。
