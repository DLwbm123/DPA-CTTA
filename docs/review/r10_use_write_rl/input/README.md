# R10_USE_WRITE_RL_V1

一轮独立于R9的有限实验计划：当前使用码与历史写入分开，源端比较普通可微训练、当前奖励、序列奖励、保留奖励及持续EMA，目标端冻结策略。

## 转交方法

将本目录交给Codex，并发送 `CODEX_EXECUTE_R10.md` 的执行任务。该prompt覆盖实现、自动验收、有限真实profile及准入通过后完整运行，不要求在普通阶段结束后再人工审阅。

- `R10_EXPERIMENT_PLAN.md`：完整数学、来源与迁移区别、训练/选择/数据/预算/恢复/报告定义。
- `CODEX_EXECUTE_R10.md`：可直接转交的实施与条件执行指令。
- `R10_SPEC_AND_MATRIX.json`：25个训练任务、340核心目标槽位及70个终点敏感性槽位。
- `reference_math.py` / `test_reference_math.py`：独立的CPU数学参考及10项检查，不是完整CTTA runner。
- `verify_plan.py`：计数与名义目标调用检查，不是实测资源PASS。
- `build_plan.py`：重建机器矩阵；不是实验入口。
- `REFERENCE_TESTS.log` / `PLAN_VERIFICATION.json`：本次已运行的本地检查结果。
- `FILE_SHA256.json`：包内文件完整性摘要。

本包没有读取私有医学数据、加载父checkpoint、访问远端GPU、修改仓库、启动训练或创建监测。新R10运行器仍需Codex实现并在真实环境验证；代码起点是13bd6a8，不是声称该提交已有本轮RL实现。

检查：

```bash
python verify_plan.py
python -m unittest test_reference_math -v
```

25训练任务=5个2000步共享warmup+20个4000采样轮post任务。各post轮2次optimizer更新，4候选×4时刻。最多410个目标评估槽位；可复用项不冒充新的物理轨迹。

独立预算：256 GPU-worker小时、64GiB、12M forward、2M backward、1M optimizer；有限profile最多8 GPU小时且计入总包。真实profile能否满足完整矩阵仍待验证；本地计数检查不作此保证。
