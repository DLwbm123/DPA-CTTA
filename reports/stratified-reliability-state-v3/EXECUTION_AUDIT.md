# V3执行审计

源码8ce67302d7962297dd5b650591273eddfad63e26；GPU worker exit0/inactive、GPU_COMPLETE、3600/1600/128完整；score exit1/inactive；最终ledger all_workers_retired=true；现场controller的/proc不存在。SCORE_COMPLETE和STATE_COMPARISON均不存在。最终资格NOT_EVALUABLE，不能仅用GPU_COMPLETE或部分RESULTS.status称成功。

V3 GPU进程472.6916074753秒，内部469.8153898716秒；CPU score进程253.6882219315秒，未重试。历史GPU500.2589249611秒，累计972.9505324364秒。历史CPU准备及V2评分274.0750441551秒，加V3准备3.2381153107秒与此次评分后为531.0013813972秒；两次旧未计时CPU脚本准备错误和本次只读诊断未单独计时，不声称覆盖全部墙钟成本。输出407583013字节，峰值CUDA allocated1514289664字节，均低于本轮1200秒GPU/900秒CPU/2GiB。

2320控制逐位检查失败；新增一次完整只读差异定位，无新模型/标签读取/适应，无GPU启动，138读出/238像素位差异。全量差异只公开按方法/状态/时点聚合；未发布图像、标签、权重、逐例预测评分或私有路径。源码、协议、CPU证据、资格失败与匿名部分聚合公开；部分聚合明确标记不具本轮科学资格。GitHub公开交付需远程SHA和匿名HTTP验证，由私有交付回执记录。
