# V4执行审计

RUN_STATE GPU_COMPLETE/128arrival；REPEATABILITY PASS/strict_exact=true/128paired读出/16组合汇总；SCORE_COMPLETE COMPLETE/engineering_only=true/evaluation_label_files=0。worker与score均exit0/inactive，最终ledger all_workers_retired=true，controller /proc不存在。10项CPU检查PASS，真实数据0、CUDA未初始化。

GPU进程87.9062025547秒（内部85.1021232605秒）<180；CPU汇总2.8530201912秒<30；峰值allocated1581094400字节；仅小型JSON输出，无逐例张量保存，低于64MiB。历史GPU972.9505324364秒，累计1060.8567349911秒。此前已计CPU531.0013813972+本轮准备3.3542213440+汇总2.8530201912=537.2086229324秒；旧未计时CPU错误/只读诊断仍未计，不声称全墙钟精确覆盖。

只读取两个既有RGB、源checkpoint和既有实现；适应标签0、旧预测0。不发布图像、标签、权重、逐例预测/评分、身份或私有路径。源码、冻结配置、CPU检查、匿名重复性汇总和无路径receipts公开；GitHub推送SHA及匿名HTTP由私有DELIVERY_V4回执确认。
