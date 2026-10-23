# r05 退出审计与独立超时修复

本目录只修复运行控制与状态对账。没有重跑 r05，也没有启动后继实验。原 r05 supervisor 与原始 EXIT 保留，不能将本目录测试当作医学证据。

## 退出与预算

r05 原始硬预算 900 秒，起点到 EXIT 结束时间为 1173.257 秒；EXIT 中 seconds=989.187 在同步退出存储扫描前取值，扫描路径额外耗时 184.070 秒。worker 精确停止时刻没有独立本地回执，故无法确认 worker 是否按时停止，端到端预算不能标为合规。退出占用 73728 字节低于限额，但不是连续峰值证明。

原始退出码 −9，reason=null。合成复现证明 GNU timeout 的 TERM 后二次强杀可以产生 −9；这只是相容解释。内核日志无权限、journal 不可用，无法确证历史 SIGKILL 来源，也不能仅凭 −9 诊断 OOM。事后结果读取曾卡在 RPC 等待，不能倒推为原 worker 的唯一死因。没有完成或可用部分结果，r05 科学判据保持未检验。

## 已确认的控制缺陷

1. 相对 timeout 时长在存储扫描、网络盘写入和真正启动之前计算；这些延迟未从剩余时间扣除。
2. 循环先检查时间，却在发停止信号前同步扫描网络盘。扫描卡住会阻止该层停止动作。
3. 启动/结束回执依赖同一网络盘，没有独立本地的终止确认。
4. timeout 的 −9 被写成一般失败且 reason=null，丢失超时上下文。

## 修复后的控制路径

`src/guard.py`、`src/probe_storage.py` 先放在服务器本地磁盘的中性 /tmp/eNNN.py 入口；用隔离系统 Python `-I -S` 执行，工作目录 /tmp。job.json、CLOCK、LAUNCH、SIGNALS、EXIT 和有大小上限的 stdout.log 均位于 uid 私有、非符号链接、经 mountinfo 确认的本地 /tmp/n... 目录。私有工作路径通过配置/环境传递，不放在可见 argv 中。

CLOCK 将首次起点固定映射到本次启动的单调时钟，保留 boot id；过期拒绝启动，已有 LAUNCH/EXIT 拒绝复用，不能通过重试重置时钟。预检通过后才按绝对截止重新计算 native timeout 的剩余时长。原生 timeout 独立于 Python guardian 和 SSH；worker 留在独立进程组中，子进程一并终止。

存储扫描是独立子进程，自带原生 timeout；卡住、报错或容量不足均不启动/停止 worker。guardian 的定时循环不读网络盘。时间 TERM 只由 native timeout 发出，避免两层重复信号破坏保存宽限期；其他故障 TERM 只发给组内 worker，随后按独立上限强杀。结束回执区分已确认终止与未确认终止，不把发出 SIGKILL 当成进程已经退出。

## 使用接口与边界

配置字段示例见 config.example.json，仅用于解释接口，绝不是 r05 重启配置。入口环境变量为 CONTROL_ROOT；worker/probe 均须为中性 /tmp/eNNN.py。测试 driver 中的秒级参数仅为故障模拟。不要用模拟参数运行科研任务。

Linux 内核不可中断 I/O、整机失联/掉电无法由用户态脚本保证即时退出；本修复会保留未确认状态。worker 不得 daemonize、setsid 或逃离受控进程组。存储保护是采样与保守余量，不是内核配额；应用写入必须有已知上界，预期峰值和本地控制日志均计入预算。网络盘恢复后才单独对账，不允许为补回执阻塞终止路径。

## 验证

tests/validate.py 仅在本地 /tmp 创建合成进程/小文件，0 科研数据读取、0 模型调用。共 13 项，覆盖预检容量不足与运行中增长触发软停、正常退出、忽略 TERM 的后代、保存宽限、预检/运行中扫描卡住、过期时钟、日志洪泛、拒绝复用、guardian 被强杀后的 worker 与扫描器原生兜底，以及旧 timeout 的 −9 复现。VALIDATION.json 为实际 Linux 运行回执。

远端科学状态对账单独记录在 RECONCILIATION_STATUS.json；只有原 EXIT 校验、原 STATE 备份、关闭状态写入及最后提交回执读回全部完成，才能标 COMPLETE。修复不改变 r05 原始时钟、原始 EXIT、科学协议或未完成事实。
