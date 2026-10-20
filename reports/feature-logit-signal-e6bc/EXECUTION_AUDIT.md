# E6-B/C 执行审计

执行source：`b7ae86fd3a7dfcf60dbc49285d956eb4f9ad5e0c`，branch `experiment/feature-logit-signal-e6bc`。科学配置此前已冻结；本提交只实现，不根据本轮结果调整。

- 合成CPU 9/9 PASS：符号/零与非有限范数、锚点排序/不足、常特征、移位坐标、分层置乱、AUC ties/加权/缺定义、有理效用、访问隔离、存储与进程身份。
- 32输入/64源encoder前向/64head重构/64source bit parity。全部score/anchor/uid/coverage同进程精确重演，非跨进程认证。
- CPU评分3028单块精确效用、18 AUC、15配对对比、共同/完整受限空间；8标签只在worker退出码0且原身份不存活后读取。
- 0参数更新、0 backward、0 selector安装、源全state_dict不变，32维原生坐标未插值。
- 首次实际CPU前保存UTC/BOOTTIME起点，独立guardian与controller均用中性入口；子进程ps与真实GPU nvidia中性条目/UUID通过，启动free24124MiB。硬保存停止86340s/24h共用原始时钟，256MiB水位/320MiB reservation。
- guardian采用已有核验Linux pidfd ABI只绑定登记start_ticks身份，未触发停止；空间safe walk不follow symlinks，仅忽略ENOENT。
- 准备文件名padding和NAS回执变量错误单独披露，无实际CPU/GPU重试、无预测改变；未计时编排不计入worker成本。
- worker/scorer/check exit0与退役核实，controller与guardian原身份退役，OS码未知保持null。见RETIREMENT与RESOURCE_LEDGER。

公开仅本项目源码、冻结协议、匿名聚合与无路径回执。private scores/anchors、GT效用、端点bits、RGB/权重、逐块/逐内容结果均不发布。没有新性能预测，报告不制造方法Dice或结构指标。
