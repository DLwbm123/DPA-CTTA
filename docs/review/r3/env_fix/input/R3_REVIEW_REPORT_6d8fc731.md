# R3 外部代码审阅：6d8fc731

## 结论

**R3_REVIEW_CHANGES_REQUIRED — R3-ENV-01 ONLY**

五个框架的核心方法、匹配控制和 85 条轨迹设计可以保留。本次识别出一处需要在正式 GPU 验收前修复的执行入口缺口：R3 没有继承原有 smoke／formal 的 `CUBLAS_WORKSPACE_CONFIG` 分阶段政策。它使严格确定性 GPU smoke 依赖调用者的偶然环境变量。

这不是效果诊断，不要求修改算法、重建数据登记或改变实验范围。目前不签发 `R3_REVIEW_PASS`，也不授权 GPU 或后台运行。修复后只做环境入口相关差异复核，不重新设计五个框架。

## 1. 被审对象与冻结身份

| 项目 | 身份 |
|---|---|
| 仓库 | `DLwbm123/DPA-CTTA` |
| 实现提交 | `6d8fc7317506400b039d3d2ed41bca18523beb27` |
| 基础提交 | `f52f132e4be576ea871467432a6f4b12337209a5` |
| 材料发布提交 | `98e18ae4b8b8320a53943dce98950e131f9178ad` |
| Science SHA256 | `73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e` |
| 基础登记摘要 | `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf` |
| 新回访流摘要 | `cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db` |

Science 字节与原任务包一致，并与远程文件 Git blob `3b23d466bb79b9ece61ccc90196e0b7c8082e18a` 对齐。两个登记／流摘要来自公开交付的私有元数据核验，本次没有访问私有 registration，未独立重算其中的真实身份序列。

以原任务包的 `01_MASTER_PROMPT.md`、`02_METHOD_SPEC.md`、`03_EXPERIMENT_PLAN.md` 和 science 为实现依据；没有将论文摘要中的未知机制补入本轮算法。

## 2. 核心方法检查

| 范围 | 已核对的关键语义 | 结论 |
|---|---|---|
| C／RP | 实际委托给未修改的 R1 C／REGION；不是重新手写一个近似基线 | 无阻断项 |
| 共同宿主 | 当前图六视图原 CPU 归约；共享增强顺序；一次 Adam；更新后原图预测；预测和状态固定后才读 mask | 无阻断项 |
| T | 同 checkpoint 的冻结 source-BN 测量器；前景／背景密度成对比较；收缩、LR／ISO／DIAG 区别；log-odds correction 上采样；零修正精确保留原 q | 无阻断项 |
| U | 第一原图前向真实连到 BN affine；最多八个 VJP；实际 Adam proposal 位移变换；moments 仍是一次原梯度 Adam；随机方向和同范数控制 | 无阻断项 |
| S | 最多三个 affine／Adam／memory 条目；一个活动模型和一个冻结测量器；未选条目不被更新；slot 切换不回放全局增强 RNG | 无阻断项 |
| M | 同位置 pre／post；正交映射方向；live 与 cached 均值／散布／基底一起迁移；当前 post token 后插入；identity-post 与 shuffle 控制齐全 | 无阻断项 |
| G | 1,984 条局部边；独立 Bernoulli KL；32 次固定求解；全分辨率 log-odds 修正与包含投影；最终原图预测不做强制嵌套 | 无阻断项 |
| 计划与数据 | 17×5=85；四个主序不变；新增回访流每个内容只出现一次、保留域内顺序；算法只收当前图，不收域名 | 无阻断项 |
| 汇总 | 主四序与 secondary 分开；同批 C／RP 与候选完整配对；有限计数及旧基底时序核对；失败使旧完成结果失效 | 无新阻断项 |

### 已披露、可以接受的实现解释

密度协方差／均值采用固定贡献刷新周期上的旧快照，不读取本图更新后的统计。零 raw-PCA rank 在达到样本支持条件后，仍可通过收缩得到正定密度模型；不能把原始 PCA 的非零 rank 与密度可计算性混为一谈。此解释已经在 `METHOD_PROVENANCE.md` 明示。

T 不使用源预测充当新标签；S_NOPCA 保留 shadow memory 以控制容量／路径，但不把 RP 项加到其 loss；U_SCALE 在自己的轨迹上匹配位移范数，不与另一条独立轨迹共享更新。

密度可能不准确、router 可能只产生一个 slot、正交迁移可能不适配真实漂移、PCA 探针可能保护错误方向、图可能过平滑，都是需要由冻结实验回答的科学风险，不据此临时加效果 gate 或修改超参数。

## 3. 唯一必修：R3-ENV-01

**级别：正式 GPU smoke 前必须处理的执行正确性缺口。**

### 3.1 位置

`src/dpa_ctta/r3/execution.py`：

- L33–70：`smoke()` 在 L47 进入 `deterministic_smoke_pair()`。
- L110–133：`worker()` 没有设置／验证 smoke 与 formal 的 cuBLAS 环境。
- L160–164：`launch()` 内的 `start()` 直接继承 `os.environ.copy()`，没有按阶段控制该变量。

该文件 Git blob：`3e2c438a6e92d8c5766e3eb0490914e163a6a4de`。

### 3.2 问题链条

```text
新 worker 继承调用者环境，未显式保证 cuBLAS workspace 配置
    ↓
smoke 打开 torch.use_deterministic_algorithms(True, warn_only=False)
    ↓
ready RP 及 U 等路径使用 CUDA mm / mv
    ↓
没有合法 workspace 配置时，触发 PyTorch 确定性 CUDA 运行限制
```

`m2_run.deterministic_smoke_pair()` 只管理确定性算法开关，不替调用者设置 cuBLAS 环境。`source_pilot_release.environment()` 设置 TF32／cuDNN 等并查询设备，但同样不设置该变量；`seed_all()` 只设 Python、NumPy、Torch seed。

**原 R1 已有正确的阶段区别：** smoke worker 在 CUDA 工作之前设置 `:4096:8`，formal worker 移除该变量。R3 薄入口未复用这段设置。这是遗漏的执行边界，不是新的科研设置。

PyTorch 2.6 官方 `torch.use_deterministic_algorithms` 文档规定：CUDA ≥10.2 时，严格确定性模式下的 mm／mv／bmm 需要合法的 `CUBLAS_WORKSPACE_CONFIG`（`:4096:8` 或 `:16:8`），否则报 RuntimeError。

官方依据：

```text
https://docs.pytorch.org/docs/2.6/generated/torch.use_deterministic_algorithms.html
```

### 3.3 本次实际验证与推断边界

本次从被审 `execution.py` 提取原 `launch.start` AST，未修改函数内容，以记录型 Popen 替身执行环境构造。没有创建 GPU worker，也没有调用 torch CUDA。

| 调用者变量 | smoke 子进程变量 | formal 子进程变量 |
|---|---|---|
| 不存在 | 不存在 | 不存在 |
| 非法字符串 | 原样继承非法字符串 | 原样继承非法字符串 |
| `:4096:8` | `:4096:8` | `:4096:8` |

六种组合确认了“入口完全继承环境”的事实。**这里独立复现的是环境控制缺口，不是 GPU RuntimeError。后者是实际调用路径与 PyTorch 2.6 官方契约共同支持的运行风险判断。** 当前没有访问服务器的真实启动环境，不能说其环境必然未设置。

### 3.4 期望修复

在 `Popen` 之前构造明确的阶段环境：

```python
if phase == 'smoke':
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
elif phase == 'formal':
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
else:
    raise ValueError('unknown execution phase')
```

不要依赖操作者先在 shell 中 export，也不要只用 `setdefault` 保留非法值。worker 在真实设备／模型工作前做一致性核验，防止绕过 launch 的 `RUN_MODE` 路径静默运行在另一套环境下。CPU 直接调用数学 smoke 的测试仍可以原样运行，不要求 CPU 具有 cuBLAS 配置。

记录 smoke 确定性上下文内部的实际开关和环境值，区分 `environment()` 在上下文外捕获的 backend 与机械比较期间设置。formal 继续原有正式数值政策，不强行打开严格确定性，也不保留 smoke 变量。

不能为了让 smoke 通过而关闭确定性、改 warn_only、放松数值容差或删去 ready 分支。不会增加任何模型调用。

## 4. 已披露 EIO 残留的处理

CPU 05 的旧测试确实出现了 `all_owned_returned=false` 与 `all_owned_reaped=false`；异常退出，测试兜底清理了自己的进程，无关对照进程仍存活。后续九次检查和最终回归通过，不能抹掉第一次观察，也不能推定已定位为系统调度或测试假阳性。

本次对当前提交的 `supervise.py`／`evidence.py` 核验 Git blob 后，又独立运行了以下四个 CPU 场景：

| 场景 | 监督器返回时自有进程已返回／回收 | 无关进程保留 | 监督器进程退出码 |
|---|---|---|---:|
| 正常完成 | 是／是 | 是 | 0 |
| 可写目录下超时 | 是／是 | 是 | 1 |
| 失败报告写入 ENOSPC | 是／是 | 是 | 1 |
| 两个 formal 启动后持续 EIO | 是／是 | 是 | 1 |

观测均在检查脚本的兜底清理之前取得；故障场景未派发后续任务，所有检查进程最终回收。

**结论：本次未复现，不等于旧异常已修复。** 保留为未解释的工程风险。目前没有新的确定复现或可归因代码差异，因此不要求凭猜测重写共享监督器，也不将它变成必须连续重复若干次直到变绿的程序。

下一次修复的正常完整 CPU 回归仍保留该测试及断言。再次出现时，应保留当次原始异常、各自有进程的清理动作／returncode／wait 结果，避免仅重复直到成功；没有再次出现则如实记“未重现，根因未知”。

## 5. 独立 CPU 审阅证据

### 5.1 与 Codex 交付测试分开

公开最终套件实际为 97 项全部通过；最终 focused 4 项中有 3 项重叠，材料说明共有 98 个不同测试。不是 101 或 121 个不重复测试。

Codex 的真实 ResUNet34 结构检查使用程序化随机权重，并对 head 做程序化缩放以激活可靠性路径，记录 38 次 Adam、38 次 loss backward、316 次网络前向、18 次 VJP。它不是已登记源 checkpoint 的 GPU 验收。

本审阅没有重跑其完整项目测试、真实模型或真实 checkpoint。

### 5.2 本次本地独立检查

- `independent_checks.py`：12/12 通过。包括真实被审核函数的密度、参数 Jacobian 有限差分、Adam 位移／moments、三 slot 状态隔离、统计迁移、图教师，以及矩阵／回访组合检查。
- `supervisor_checks.py`：4/4 程序化进程场景通过。
- `smoke_environment_witness.py`：6 个环境组合完成，确认当前缺口；这是修复前 witness，不是修复接受性测试。

本地 Torch 为 `2.10.0+cpu`，与交付套件的 `2.6.0` 不同。所有本地检查均不使用 GPU 或实际源／目标资产；数学检查只替换 trace 分布格式化依赖；监督器检查只隔离 binding／显式程序化资源上限；环境检查仅提取原函数 AST 并替换 Popen。范围在脚本中明示。

12 个本地保存源码文件与远程 Git blob 一致，详见 `evidence/SOURCE_IDENTITY.json`。它们是审阅摘取源码，不是完整仓库归档。

## 6. 不改变的实验范围

17 臂、四个主序＋一个独立 secondary 回访流，共 85 条完整轨迹。

| 项目 | 正式预算 | 每张实际 GPU 的 smoke |
|---|---:|---:|
| 效果评分 | 165,835 | 不计入 |
| loss backward | 165,835 | 38 |
| Adam proposal | 165,835 | 38 |
| 网络前向 | 1,385,210 | 316 |
| 额外 Jacobian VJP | ≤234,120 | ≤24 |

各臂仍使用同一个原始 checkpoint；不加源训练、Polyp、新 seed、超参数搜索或五框架组合。回访流仍独立于四序主平均。

修复只限 R3 环境入口、对应 CPU 回归和必要的 backend 证据说明。可顺带如实区分图分支的“计划迭代数”与零边短路时的“实际迭代数”，但不修改图数值路径；此项是非阻断计账备注。

交回新 implementation SHA、相对 `6d8fc731` 的窄 patch、science 保持性证明、环境边界测试和真实回归日志。状态应为 `R3_ENV_FIX_READY_FOR_REVIEW`，不是自行签发 review pass。随后停止等待差异复核。

## 7. 可追溯源码入口

代码均按本报告指定实现提交读取：

```text
src/dpa_ctta/r3/{host,kernels,teachers,stats,displacement,contexts,transport,plan,execution,analyze}.py
src/dpa_ctta/r1/{run,supervise,evidence,region_memory,streaming_pca}.py
src/dpa_ctta/m2_run.py
src/dpa_ctta/source_pilot_release.py
src/dpa_ctta/source_pilot.py
scripts/run_r3.py
src/dpa_ctta/r3/run.py
tests/test_r3.py
tests/test_r3_execution.py
tests/test_r1_fixes.py
```

交付材料按 publication SHA 读取：

```text
docs/review/r3/REVIEW_INDEX.md
docs/review/r3/IMPLEMENTATION_REPORT.md
docs/review/r3/METHOD_PROVENANCE.md
docs/review/r3/DEVELOPMENT_LOG.md
docs/review/r3/logs/cpu-05.log
docs/review/r3/logs/cpu-06.log
```
