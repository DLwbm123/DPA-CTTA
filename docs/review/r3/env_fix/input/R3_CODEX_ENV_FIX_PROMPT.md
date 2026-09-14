# 发给 Codex：R3 仅修复 smoke／formal 环境边界

## 当前任务

外部审阅状态：`R3_REVIEW_CHANGES_REQUIRED — R3-ENV-01 ONLY`。

被审代码：`6d8fc7317506400b039d3d2ed41bca18523beb27`。

五个框架及匹配控制已完成方法检查，本次只修复执行层 `CUBLAS_WORKSPACE_CONFIG` 的阶段边界，保留算法和所有科学配置。当前仍为阶段 I：不使用 GPU，不读真实目标 RGB／mask，不读源图像／mask，不启动正式实验或后台等待。

## 唯一必修问题

`src/dpa_ctta/r3/execution.py` 的 `launch.start` 仅继承父环境；`worker()` 也未设置／核验 cuBLAS 变量。与此同时，`smoke()` 使用 `deterministic_smoke_pair()`，该上下文启用严格确定性，却不设置环境变量。ready RP／U 中存在 CUDA 矩阵运算。

对照原 `r1.run.worker/formal_worker`：原 smoke 显式设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，formal 显式删除。R3 遗漏了这段已存在的执行政策。

PyTorch 2.6 官方文档：

```text
https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html
```

其中说明，CUDA ≥10.2 严格确定性模式的 mm／mv／bmm 需要合法配置，否则 RuntimeError。

外部证据 `smoke_environment_witness.py` 用原 `start` AST 和记录型 Popen 替身确认：缺失、非法或有效父变量都被原样传入两种阶段。该脚本没有运行 GPU，断言的是当前缺口；修复后不应为了继续满足这些旧断言而保留错误。

## 需要实现

### A. 明确的子进程阶段环境

在 Popen 前、子进程解释器／CUDA 初始化前设置：

```python
if phase == "smoke":
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
elif phase == "formal":
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
else:
    raise ValueError("unknown execution phase")
```

推荐把纯环境构造封装成 R3 本地小函数供测试。不要改共享 R1/R2 科学路径。不要使用 `setdefault`，不要依赖人工 shell export，不修改父环境。

### B. worker 入口保证

通过现有授权后、任何真实 CUDA 设备查询／工作前，核验当前 worker 阶段与环境一致。

- smoke 必须是 `:4096:8`。
- formal 必须移除该变量，保持原正式政策。
- 绕过 launch 的 `RUN_MODE` 入口不能在不一致的环境里静默执行；可以清楚拒绝，或在确认未初始化 CUDA 且足够早时规范化。优先使用从进程创建时已经配置、worker 再断言的简单路径。
- 不改变既有未经授权时先拒绝的顺序。
- CPU 数学测试直接调用 `smoke(...,device='cpu')` 不必新增 cuBLAS 前提。

### C. 如实记录 backend

记录机械比较上下文中的实际 `deterministic_algorithms`／`warn_only`／workspace 值，与进入上下文前 `environment()` 返回的设置分开；formal 记录自己的实际设置。不要用上下文外捕获的开关代表 smoke 内部。

不需要改变 science JSON；这些属于执行证据。若扩充 smoke JSON，只同步对应的 CPU fixture／验证断言，不放松原有配对或物理调用数。

### D. 针对性 CPU 测试

只需围绕上述边界补齐小型单元测试：

1. 父变量缺失、非法、`:16:8`、`:4096:8`，smoke 子环境均强制 `:4096:8`；formal 均不含变量。
2. 父环境未被修改，CUDA_VISIBLE_DEVICES、RUN_MODE、精确执行身份与其他既有设置保留。
3. 用记录型 Popen 或纯环境构造断言，环境在 spawn 前确定；禁止真实 GPU 查询及初始化。
4. 不合规的直接 worker 入口在设备工作前拒绝；默认禁用授权仍优先拒绝。
5. 实际 smoke 确定性上下文的设置与退出恢复正确；仍是 38 次 Adam／loss backward、316 次前向、最多 24 次 VJP，不新增模型调用。

保留当前测试和断言。当前材料是 97 项完整套件＋4 项 focused（其中 3 项重复），共有 98 项不同测试；不要把它们误计为 101。新增测试数量按实际情况报告，不规定一个人为的通过数量。

## EIO 残留：保留事实，不凭猜测改监督器

原 CPU05 的 persistent-EIO 测试曾失败一次；九次后续诊断及最终回归通过。外部审阅对当前精确 supervisor/evidence 又做了正常、超时、ENOSPC、EIO 四项程序化检查，均能回收自己的子进程并保留无关进程，但没有定位旧失败。

- 不能把它改写成“已修复”或“已确认测试假阳性”。
- 本次不因猜测修改共享监督器，不删除／弱化该测试，不增加自动retry。
- 正常完整 CPU 回归仍运行该测试。再次失败时保留该次清理异常、PID所有权、returncode和wait结果，不要靠重复运行覆盖失败；测试兜底必须清理所有自有程序化进程。
- 未再次出现就记“本次未重现，根因未知”。不要发明新的效果 gate 或长时间后台诊断。

## 禁止改动

- 不更改 T/U/S/M/G 的公式、阈值、rank、协方差快照、slot 容量、memory 时序、teacher或最终输出。
- 不关闭严格 smoke 确定性，不设 warn_only 来绕过问题，不放松配对容差，不删 ready 分支。
- 不向正式流添加新的确定性政策或保留 smoke workspace。
- 不修改 source checkpoint／训练、原 registration、四主序和 secondary 序列。
- 不使用真实 target RGB／mask、源数据或 GPU 进行本轮修复。
- 不重跑旧方法、不新增实验臂或组合。

可选的非阻断计账备注：`teachers.graph_target` 对零边短路时实际没有执行 32 次迭代，当前 `iterations` 字段可能仍记 32。可将 planned／executed 区分清楚，保持数值路径完全不变；无需为此单独开启修复轮。

## 不变绑定与预算

```text
SCIENCE_SHA256 = 73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e
BASE_REGISTRATION_DIGEST = 8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
NEW_STREAM_DIGEST = cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
```

17臂×5流=85条正式轨迹；165835条效果评分／Adam／loss backward；1385210次正式前向；额外VJP最多234120。每块实际GPU的既定smoke为38次Adam、316次前向、最多24VJP。当前无GPU／后台授权。

## 交付与停止

在当前实验新分支提交修复，发布：

- 新 implementation SHA；相对 `6d8fc731` 的窄 patch。
- 新环境入口及测试的具体位置；修改前后环境矩阵。
- 真实完整 CPU 日志、任何首失败日志；独立列出新增检查，避免重复计数。
- science 字节未变、基础登记与secondary摘要未变的说明。
- EIO残留状态，不将未复现描述成已解决。
- 更新审阅索引，不公开权重、真实数据、私有路径／身份、设备UUID或论文PDF。

最终状态：

```text
R3_ENV_FIX_READY_FOR_REVIEW
```

随后停止。不要自行生成 `R3_REVIEW_PASS` 或执行授权。下一次外部审阅仅检查本次环境入口及相关证据差异；通过并获得本批资源授权后，再运行原85轨迹矩阵。
