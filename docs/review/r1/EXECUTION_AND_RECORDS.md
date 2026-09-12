# R1 可审阅入口与未执行项

阶段 I 只允许CPU程序化测试和metadata读取。没有GPU授权文件，也不生成外部review结论。所有GPU smoke、目标预测和正式分数均为 NOT_RUN。

使用已有环境设置 PYTHONPATH、DPA_CTTA_BASE_ROOT、DPA_GRATA_ROOT。命令形式：

```sh
python scripts/run_r1_matrix.py
# 默认只输出24条任务的JSON；不读取目标或checkpoint，不查询/初始化GPU。
python scripts/run_r1_matrix.py --run
# 默认 execution.enabled=false；立即拒绝，先于权重和GPU访问。
```

未来阶段II：显式 --run --authorization --assets --output。真实运行在中性 python -c + 环境变量入口下，主/子进程不携带项目/方法路径参数。私有assets提供唯一checkpoint与过滤后的target metadata；原旧登记proxy/history/identities/source query不进入新schema。路径缺失只影响未来执行，不能阻止当前实现。

授权检查是防误运行，并非密码学签名或审阅真实性证明；用户转发的真实review和新执行指令仍是必要条件。科学配置保留任务包原始字节，enabled/GPU/并发完全分离。例行测试中的TEST_FIXTURE_ONLY和假SHA只是函数测试，不是GPU授权。

每GPU未来smoke：旧C两步、新六臂各两步=118前向/14 backward/14 Adam。程序化controller输入fixture触发恢复，程序化PCA统计预置ready；fixture随测试host销毁，正式状态从空开始。C四类状态对齐用旧确定性作用域，正式backend使用新进程。现在没有运行它。

固定正式矩阵=24×1951条。前向398004；backward/Adam各46824。每worker两CPU线程；GPU列表必须重新显式给定，最多三卡；临时保守内存要求4GiB free属于可执行资源预留，不是科学筛选。实际延迟/显存未测量，不承诺符合24小时预算；阶段II按既定smoke和真实资源核验，不缩减矩阵或调科学参数。

每条成功记录保存：opaque身份、域/序/子集（仅runner）、global_visit、segment_age、reset标量、physical counts、原C loss、实际affine更新L2、PCA输入配额/版本/rank/计数/字节/subloss、OD/OC计数Dice、ASSD/empty/full、host/pipeline时长和allocated peak。不持久化真实图像、mask、预测图、checkpoint副本或PCA状态。

CPU closeout：所有24条完成且覆盖/身份/时序/调用数/GT像素计数一致才生成报告。Dice由计数还原，ASSD仅用保存标量；逐域/通道/四子集的paired分布与尾部保留。A/C0旧标量只在指定历史文件存在且身份匹配时作为次要参照，缺失明确UNAVAILABLE，不补跑。

预登记0.5pp/3序/域损失2pp/最差序0.5pp只生成解释标记；不能中断、追加、选GT最优臂或启动组合。候选建议仍需审阅全部负向尾部，不等于显著性、临床获益或独立泛化。
