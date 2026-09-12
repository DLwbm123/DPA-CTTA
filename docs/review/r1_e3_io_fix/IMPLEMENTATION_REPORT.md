# R1 E3-IO 修复报告

状态：**R1_E3_IO_FIX_READY**。本轮只修复外部差异审阅中的 E3-IO；不生成外部审阅通过结论或阶段 II 授权。

完整 implementation SHA：`54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b`。审阅基线：`c93018c0edca2bbe2475117ab0edf6adeece6ff2`。分支：`experiment/r1-recovery-target-subspace-v1`。

## 缺陷与修复

原 `finish()` 先写失败记录，后终止子进程；普通 ENOSPC/EIO 异常可使正常路径和 finally 同时跳过终止操作。新实现把停止派发、失败状态和首个异常保留在内存。`finish()` 只处理状态与进程回收；最终收尾先逐组尝试所有已登记自有进程，再逐份尽力写证据。单组清理或日志错误不阻止其余组的清理尝试。

普通运行中的写入错误仍向外抛出，立即停止新派发并进入清理；证据持续不可写时仍非零退出，尽力输出 stderr。此时不保证失败 JSON 能落盘。写盘错误不会被吞掉后继续实验；失败不会返回 COMPLETE，也不会调用后续实验重算或自动重试。

保持 SIGINT/SIGTERM handler 只记录请求，继续在安全边界处理中断，收尾期间忽略再次中断。Popen 返回后立即登记所有权，现在登记也位于父日志句柄关闭之前，以覆盖关闭句柄报错的同类路径。没有修改 `stop_owned()` 的有限 TERM/KILL/wait 语义、`evidence.write()`、模型、资产 reader、分析器或分配算法。

`dispatch.stopped.json` 与逐任务 supervisor 失败记录改在进程清理后写入；内存中的停止派发立即生效。运行时的 `usage.json` 和已启动进程记录仍照常写入，任何普通 IO 异常均中止矩阵。

## 真实测试证据

新增两个接受性回归首先在旧 c930 实现上执行：**2 失败，0 error，退出码 1**。两种故障均观察到自有 formal 子进程未回收；测试脚本的 finally 随后清理了自身创建的进程。保留[初次失败日志](INITIAL_ACCEPTANCE_FAILURE_LOG.txt)，仅替换本地绝对路径。另独立运行包内缺陷复现器的三项短 CPU 检查，结果见[基线复现结果](BASELINE_REPRODUCTION_RESULTS.json)；其 source 字段来自原脚本，本次实际输入是用户提供的包内源码快照。

修复后的本地 12 项进程检查通过，覆盖两个新故障、正常成功、超时、SIGINT/SIGTERM、Popen 登记及 mkstemp/fd 等旧回归。该中间本地检查之后只收紧了新测试 finally：已回收的直接子进程不再次发送组信号；最终完整提交由下面的服务器套件验证。

最终完整提交在服务器现有 Python 3.10.6 / Torch 2.2.1+cu121 环境运行：**37 通过，0 失败，0 错误，0 跳过，exit 0**；unittest 60.576 秒，包装器 60.970 秒。`cuda_initialized=false`，`checkpoint_mode=programmatic_random_weights`。详见[完整原始日志](CPU_TEST_LOG.txt)与[机器摘要](CPU_TEST_RESULTS.json)。

| 新故障回归 | 自有子进程 | 无关对照 | 监督器退出 | 汇总 |
|---|---|---|---|---|
| 仅失败记录 writer 抛 ENOSPC | 2 smoke + 2 formal 均结束并已回收；无后续派发 | 仍存活 | 1 | INCOMPLETE |
| 两个 formal 启动后所有证据 writer 持续抛 EIO | 同上 | 仍存活 | 1 | 无文件；不要求写盘恢复 |

两项均在独立真实 CPU 监督器进程内注入，测试在自身兜底清理**之前**检查 Popen 返回状态及 waitpid 的已回收状态，并从外层实测非零退出。观察完成后，测试 finally 也清理无关对照进程。详见[机器结果](IO_REGRESSION_RESULTS.json)和[完整日志](CPU_TEST_LOG.txt)。这些是程序化 smoke/formal 命名的短时 CPU 子进程，不是实际设备 smoke 或真实流实验。

既有 35 个测试方法完整保留，AST 比较确认方法体及全部断言未改变；仅新增两项。本轮没有删除或放宽测试，也没有增加效果 gate。

复现命令沿用 `scripts/check_r1_cpu.py`，使用现有环境、固定 CTTA/GraTa 代码依赖和程序化权重。`CHECKPOINT` 未设置；`CUDA_VISIBLE_DEVICES=''`，入口禁止 CUDA 初始化；解释器、依赖和输出路径经环境变量提供，实际进程命令行保持中性：

```sh
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
```

`RUN_FILE` 指向上述测试入口；`PYTHONPATH` 包含项目 src/tests 及已有依赖；`CHECK_OUTPUT` 和 `TMPDIR` 指向本次私有审阅目录。CPU suite 未安装依赖、未创建常驻或等待授权的任务。

## 不变配置与未运行项

- science SHA256：`e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2`。
- registration 摘要：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`；本次仅重新读取既有私有登记 JSON 并核对摘要，未打开其中指向的资产。
- 六臂、四序、1,951 组、seed 20260907；正式 46,824 评分/Adam/backward、398,004 前向；每实际设备 smoke 14 Adam/backward、118 前向，均未变化。
- C、SENS、PCA 的全部科学参数和 checkpoint-only 边界保持不变。execution 默认禁用、审批为空、GPU 列表为空、workers=0、后台关闭。

[配置与保留性摘要](UNCHANGED_CONFIG_SUMMARY.json)包含实际校验结果；[science config](../../../configs/r1_science_v1.json)及[原冻结 dry-run 矩阵](../r1_fix/DRY_RUN_MATRIX.json)继续有效。

未运行：GPU 初始化与机械 smoke、真实目标 RGB/mask 读取、真实源 checkpoint 内容读取/重验、源图像/mask/代理/原型访问、真实 24 条实验轨迹、实际模型效果和真实历史指标重算。未创建后台等待器；外部窄差异复核仍待进行。这里修复的是已返回的普通 Python IO 异常，不声称能解除内核不可中断 IO 或进程信号 API 本身失效。

## 交付范围

[窄 patch](implementation.patch)只包含相对 c930 的 `supervise.py` 与 `tests/test_r1_fixes.py` 两个文件，保持审阅范围；完整 implementation commit 保留仓库历史及此前 cafb 材料。新报告与真实日志在单独的交付提交中发布，避免报告自引用。

公开源码、窄 patch、配置摘要和去身份 CPU 证据；不发布私有登记、服务器路径、凭据、checkpoint、原始图像/mask、患者/受试者身份或第三方 PDF。此前已关闭的 E1/E2/E4 和方法设计未重开。

最终状态：**R1_E3_IO_FIX_READY**。停止，等待外部窄差异复核。
