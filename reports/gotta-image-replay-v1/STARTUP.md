# G2 启动与工程审计（不是最终结果）

已冻结并派发：执行源码 `af86e905d16b28496a8e2d65b6342f775175e9f1`，配置 `configs/gotta_image_replay_v1.json`。正式GPU worker在NAS与显存准入后唯一启动，指定GPU启动前空闲23858MiB≥8192MiB。主子进程和GPU命令行已核中性入口，没有用户/项目/方法名称。

唯一9秒启动检查是 `QUALIFYING`、0 arrival/0读出的早期快照，worker active，尚无 `QUALIFICATION.json`。**不是本次发布时的实时进度，不代表预算、完整重复性或科学门槛已通过。** 不为等待结果重复短检或重启任务，后续沿用户已授权的半小时监测处理。

CPU检查：新增9项全通过；共15项最终PASS，完整进程4.606180667877197s。第一次15项运行中，新9项及其它5项通过，一项旧回归的配置读取因部署archive未包含旧flip配置而FileNotFoundError，完整进程6.403694152832031s。保留 `CPU_CHECK_FAILED_PACKAGE.json` 和原私有日志；仅补齐已存在于同执行commit的旧配置，再整套检查。没有改算法、阈值、科学协议或GPU重试。准备CPU合计11.009874820709229s；不删除初失败、不把打包问题称科学负结果。

合成CPU检查真实RGB/标签/权重读取均0，未初始化CUDA。完整科学矩阵2000读出/800到达/20轨迹/25绝对行尚待worker和独立CPU评分完成；额外DIVERSITY FRESH全40到达重演还须核80对logits及model/memory/Adam/RNG state。GPU1200s/CPU评分900s/输出2GiB，不砍对照或延预算。

本次发布源码、冻结协议、测试与去路径准备回执。最终结果报告、资源账本、覆盖与退役审计尚未产生，不称实验交付完成。RDC/ENGRAM关闭，8个反复开发内容限制和患者/预训练未知保持。
