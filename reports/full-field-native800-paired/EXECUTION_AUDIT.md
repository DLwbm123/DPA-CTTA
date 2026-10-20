# 执行与报告审计

- 运行source1c72d30fede699abf536b55109407b8d6542d249；协议、四臂、独立重演、插值/dtype顺序与G0-G3均在运行前冻结。
- 原始UTC2026-10-20T12:09:59.374097Z；check/worker1/worker2/score共享boottime，600秒预算、独立guard540秒/32MiB、controller剩余530秒。实际93.005814797秒，无科学阶段重试。
- check OS0/5.807709455秒；worker1 OS0/40.457283020秒；worker2 OS0/37.149294138秒；score OS0/9.364512682秒。全部精确身份退休，guard OS0，controller父OS未知null，无活跃本轮进程。
- 批准物理GPU1，启动free32758MiB；两worker自身nvidia UUID/中性进程通过，主子ps中性。峰值torch分配1006253056B。未干预其他任务，未访问旧server。
- 合成CPU映射/常数/step/重演通过；48完整encoder/48head/16SOURCE bit核对，24跨独立进程输入、logits、features摘要相同、24mask逐位相同；两遍重新解码8RGB。各进程参数缓冲exact不变、无梯度/0更新；41标准BN eval+track_running_stats=true，跨进程配置一致，无改BN行为。
- 生成阶段各2禁止路径负检查；0GT/utility/旧scores读取，网络禁用。两worker OS0且精确退休后才读取8GT核登记摘要；32mask及全部4臂8内容分母齐，全部边界16/16、vCDR8/8，空头/嵌套0仍不等于安全。
- G0=true；G1/G2/G3=false。完整NA/负结果保留，统计单位8内容，不把重演或条件当新患者。800首次实际前向计入正式预算，无免费profile或尺寸替换。
- NFS容量与小文件探针通过，基线0/预留64MiB、hard32MiB。新增1708189B/61文件，含固定4096B空间receipt，无符号链接遍历；预留退0。项目knownactual327940278B+aux1GB+旧server未知335544320B=1663484598B。
- 传输前占位符错误仅造成远端解析失败，无语句执行/科学stage或root；原始错误记录保留，修正传输后才开始本轮原始clock。报告方向字段校正只澄清误差指标正差是变差、最差为最大差；35对比计数断言通过，统计值/CI/门不变。远端原始stage输出与私有bundle保留，未重跑任何科学阶段；完整features/logits不落盘。
- 输入/状态/模型输出摘要仅用于本轮预登记独立重演，保留私有；公开无原像素/GT/权重/位图/features/logits/真实坐标/身份/逐内容值/私有路径/第三方全文。仅自有代码、协议、中文报告审计和匿名聚合receipt。报告后代码方向标签修正未用于已结束运行；执行版本始终标为上述source。
- GPU完整worker历时累计77.606577158秒，CPU阶段15.172222137秒；进程历时不冒充纯GPU/CPU核心时间。原累计和未计时准备/报告处理/历史差/旧server未知成本均保留。
