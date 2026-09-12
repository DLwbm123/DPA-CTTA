# R1 E3-IO Review Index

状态：**R1_E3_IO_FIX_READY**；等待外部窄差异复核，阶段 II 未授权。

- 完整 implementation：[54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b](https://github.com/DLwbm123/DPA-CTTA/commit/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b)。
- 外部审阅基线：`c93018c0edca2bbe2475117ab0edf6adeece6ff2`。
- [实施报告](IMPLEMENTATION_REPORT.md)：根因、修改、真实验证、未运行项及公开边界。
- [相对 c930 的两文件窄 patch](implementation.patch)、[diff stat](DIFF_STAT.txt)。仅监督器与 CPU tests，原方法/配置/reader/分析器未改。
- [最终 37 项完整 CPU 日志](CPU_TEST_LOG.txt)、[测试摘要](CPU_TEST_RESULTS.json)、[两项 IO 回归机器结果](IO_REGRESSION_RESULTS.json)。
- [旧实现的两项接受性失败日志](INITIAL_ACCEPTANCE_FAILURE_LOG.txt)：本地实测，唯一脱敏为 checkout 绝对路径。
- [包内三项缺陷复现器的本地输出](BASELINE_REPRODUCTION_RESULTS.json)、[中间本地 12 项进程检查](LOCAL_PROCESS_CHECK_LOG.txt)；最终判据以服务器完整提交的 37 项为准。
- [不变配置与 35 个原测试方法 AST 保留性摘要](UNCHANGED_CONFIG_SUMMARY.json)。
- [交付清单](DELIVERY.json)。

## 重点入口

- [`stop_owned`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L22)：`src/dpa_ctta/r1/supervise.py:22`。
- [`supervise`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L42)：`src/dpa_ctta/r1/supervise.py:42`。
- [`halt`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L59)：`src/dpa_ctta/r1/supervise.py:59`。
- [`failure`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L66)：`src/dpa_ctta/r1/supervise.py:66`。
- [`write_failure`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L69)：`src/dpa_ctta/r1/supervise.py:69`。
- [`finish`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/src/dpa_ctta/r1/supervise.py#L73)：`src/dpa_ctta/r1/supervise.py:73`。
- [`io_failure_probe`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/tests/test_r1_fixes.py#L229)：`tests/test_r1_fixes.py:229`。
- [`test_failure_record_ENOSPC_reaps_all_owned_without_new_dispatch`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/tests/test_r1_fixes.py#L299)：`tests/test_r1_fixes.py:299`。
- [`test_persistent_EIO_reaps_all_owned_and_exits_nonzero`](https://github.com/DLwbm123/DPA-CTTA/blob/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b/tests/test_r1_fixes.py#L301)：`tests/test_r1_fixes.py:301`。

finally 先尝试全部已登记进程组的清理，再独立尝试失败日志；`remember()` 保留首个异常并停止派发。日志持续不可写不阻止收尾或非零退出。请以“缺陷不存在”为接受性断言；原包内复现器的反向缺陷断言只用来复现旧版。

## 沿用的冻结材料

- [science config](../../../configs/r1_science_v1.json)、[execution 默认禁用配置](../../../configs/r1_execution.defaults.json)。
- [原冻结 dry-run matrix](../r1_fix/DRY_RUN_MATRIX.json)：全部真实任务仍 NOT_RUN；1/2/3 槽分配及预算未改变。
- [method provenance](../r1/METHOD_PROVENANCE.md)、[state lifecycle](../r1/STATE_LIFECYCLE.md)：本轮不改方法、状态或 source-only 边界。

没有 GPU/真实资产/效果实验，没有后台等待任务或自动重试；不自行生成外部 review pass。
