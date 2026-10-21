# R95 文档预算与未来确认资源草案

## 1. 本次R95实际范围

仅本地读取已有公开文档/源码，编写小文档和Git交付；无模型导入、GPU、统计评分、私有资产读取、图像/标签读取或SSH。没有新CPU/GPU实验进程，不宣称已部署或验证运行hardguard。

新增本地预留256KiB（262144 bytes），远端0；文档目标<64KiB，余量含版本控制和小收据开销，Git开销不逐项计量。R94后保守总19996917760，加本次后**19997179904 bytes**<20GB，剩**2820096 bytes**，不释放历史预留、不删失败腾空间。文档准备时段目标60分钟以内，不把它当未来GPU实验T0或运行授权。

旧campaign T0=1792492974.1323855、deadline1792536174.1323855已过期；R94独立T0=1792549120.201643/deadline1792552720.201643也不延长。旧4253updates/43700F与747/6300历史余额不变，本次不使用。未来真正独立确认须另核具体资源与新时钟，不借编号为旧失败扫描续时。

## 2. 纠正NATIVE费用的符号预算

令N为未来新cohort内容数，A为预定seed×order轨迹数。N、A均未选择，不代表重新使用旧192。以下沿旧迁移实现工作负载口径，**每arrival包含一次只读原图PRE**，其fullsnapshot须恢复；用于计划，不代替未来实测计数资格。

|臂|updates|training BW|F（含PRE）|说明|
|---|---:|---:|---:|---|
|VV K|AN|AN|10AN|原step9F+PRE1F|
|E-VV K|AN|AN|10AN|同一步配方另计状态恢复/构造墙钟；不能只恢复参数|
|NATIVE K|AN|AN|9AN|原step8F+PRE1F；这是动态适应臂，不是静态模型|
|三动态臂小计|3AN|3AN|29AN|不含资格/重演/失败|
|Q0|0|0|6N|确定性六弱每内容一次；若广播A条轨迹不虚增推理或独立性|
|P0独立推理（保守预留）|0|0|N|若明确从Q0 identity只读导出且资格exact、实现冻结，可省N；不得先假定免费|
|SOURCE_ONLY，可选|0|0|N|辅助stored-stat参照，若选择须计入|

当A=6时，三动态臂为**18N updates/BW、174N F**，加Q0和独立P0后为**181N F**；若再运行SOURCE_ONLY为182N F。仅用N=192作算术示例：动态3456updates/33408F，Q0=1152F、P0=192F，合34752F；再选SOURCE_ONLY为34944F。**这些不是新运行预算或样本规模建议，更不是重新用旧192独立确认。**

前次只计VV+E-VV的2304updates/23040F（A6,N192）遗漏了NATIVE的1152updates/10368F；表中已纠正。[旧计数依据](../r85-r86-loss-aggregation/migration/PROTOCOL_AMENDMENT.md)显示BB10F/arrival与NATIVE9F/arrival的合计，VV/E-VV配方计数另需资格确认。

总预算还必须加：全部模型构造/冷启动/profile、资格与独立原实现参照、PRE只读恢复验证、诊断反向、失败/修复/迁移、数据I/O、CPU评分/封存/报告、状态恢复开销及日志。F/BW/updates分别计，不以物理更新数覆盖静态推理成本；GPU进程秒不是能耗，body不与process重复计。正式计数与上述公式不符就NOT_EVALUABLE，不能改期望绕过。

## 3. 尚未准入的运行要求

当前没有新的运行root、显存/profile或存储许可，不能在约2.82MB的保守总余量内默认创建模型输出目录。需明确独立的可用预留与预计实际峰值；原有资产只读复用、新副本全部计，不复制环境/data/model，不移除历史预留或删除历史。每服务器新增实际严格<50GB，实际总历时<=24h，具体更严限需在启动前冻结，失败/修复/迁移共享首次T0。

先确定具体N/A、完整公式和资格/失败余量，再用包含静态与CPU收尾的实际profile准入；不以未实测速度承诺可完成。独立deadline/storageguard在启动前就绪，提前停止写入/合作退出，TERM/KILL仅自有startticks匹配组；半小时r52不能充当hardguard。

若以后取得运行授权，仅既定my-gpu GPU0/1固定UUID，启动前核挂载/容量/小写读probe/每卡>=10GiBfree、8GiB allocator与neutral主子argv/ps/nvidia-smi。大文件按既定远端路径；本次不访问服务器或GPU核这些条件，状态明确为NOT_CHECKED_NOT_ADMITTED。
