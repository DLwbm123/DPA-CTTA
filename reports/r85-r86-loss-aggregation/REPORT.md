# R85/R86：损失聚合因子实验

状态：**QUALIFICATION PASS / R85 REPLAY RUNNING**。尚无新的性能结论。协议见 [PROTOCOL.md](PROTOCOL.md)。

这次只改变 BCE 和原方向 LSO 的聚合位置，主要候选 VB，反向对照 BV；保留原 VV/BB/NATIVE 对照。先 fresh 资格，再六条 VV 参考重演和48个同状态四格探针，之后12条新候选连续轨迹，最后全部GPU退休后统一CPU评分。

前置元数据审计确认当前注册1,951项与历史R30身份集合一致，192项为其子集，所有患者/视频链接为UNKNOWN。其余1,759项已进入历史流，不能当作未暴露确认集。因此R87在当前已注册内容内为NOT_AVAILABLE；新seed也不能消除这项限制。SOURCE_ONLY使用相同checkpoint及192项预处理补齐静态推理，与current-stat P0区分。

计划3616个新物理更新（资格16、参考1152、丢弃探针144、候选2304），另400次诊断反向遍历和192个SOURCE_ONLY静态forward，均计账。协议的多重比较、种子/顺序稳健性、域伤害门槛已冻结。历史R82/R83/R84负结果与R84工程失败保持原结论。

本实验所有阶段/失败attempt共享24小时硬截止；独立watchdog及root写入限额不依赖心跳。复用唯一r52半小时轻量跟进，结果出来后按当前持续自主授权分析并决定下一步。

## 已核实启动状态

科学源码提交 b26cd602b6e2b18aa58db9b5e4188d109276a8b2。CPU准备PASS（16.241517秒）；四格fresh资格16/16更新全部PASS，156F、16诊断反向遍历，output/gradient/BN/Adam/RNG核心state与独立oracle exact，PRE全状态恢复、q/W来源和连续Adam通过。SOURCE_ONLY已完成192个静态forward，资格及静态worker均exit0、注册PID消失。

首次正式检查两参考worker实际RUNNING，各完成16个到达；合计41物理更新，其中9个丢弃探针更新，对应3个同状态四格探针。旧VV PRE/K逐到达exact与梯度/分量恒等式目前通过。该检查仅为启动证明，不是1296个参考更新完成证明，也不是R86候选性能结果。STARTUP_AUDIT.json保存该时刻匿名状态。

阶段事件ledger的all_workers_retired字段是在下一批worker启动前的快照，可能陈旧；本次已以现场startticks/中性argv确认两参考worker活跃。活跃成本须叠加elapsed，不能读作费用只有ledger已收尾的60.427488 GPU进程秒。GPU进程费用为spawn到控制器观察退出的墙钟上界，最多包含约5秒采样迟延；worker内部body另记，是其子集，不重复相加。不是设备能耗/利用率积分。

24小时共享硬截止为北京时间2026-10-21 13:04:01.100432；独立watchdog已核实存活，中性入口一致。当前根写入限额2GiB并预留128MiB终止余量，2.2GB完整reservation；旧保守累计空间reserved(含local/headroom)13GB<20GB。此处仍仅交付启动状态，后续全部GPU退休后CPU评分及正式交付仍待完成。唯一r52已原位更新为R85/R86及后继实验的半小时跟进，没有增加监测任务。
