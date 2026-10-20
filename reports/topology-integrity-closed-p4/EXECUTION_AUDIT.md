# P4 完成与独立评分重建审计

- 原科学执行source e5aba97c11e52c47175cea8d0ac77c4644b50290；恢复执行source 2f24c3b8738bc0d166cdd185388e493c521b2ccd。原worker未重启，科学模块、目标与矩阵未改。
- 原controller空间递归扫描临时目录删除竞态崩溃；原日志/代码/身份保留，退出码未知不填1。独立CPU守卫接管，扫描只忽略ENOENT、不跟随符号链接，未触发空间终止。
- GPU完成状态写在RUN_STATE.status=GPU_COMPLETE，未另生成GPU_COMPLETE.json；32输入/384输出，11520科学+360重演=11880核心step。私有JSONL11880条顺序核过，384位图齐。MECHANISM/ACCESS通过，源参数和全部buffer不变、目标网络BNbuffer不变，worker标签/旧评分读取0、未来图用于更新0。
- 首输入四臂90步全encoder/logit/末model/Adam精确重演PASS，仅同进程首输入，非全部32输入/native/跨进程。资格首图四臂PASS。REPEATING的遗留case/condition元数据不代表实际x，实际是首输入condition0，范围文件保留。
- 原worker由pidfd证实退役，但OS退出码未捕获，保持null。original_protocol_exit0_gate_met=false；不生成原SCORE_COMPLETE、不声称原完整执行资格恢复。完整推理/重演/机制/access/索引证据齐后，按预冻结恢复协议独立CPU重建，未绕过失败证据、未改变统计/主比较/预测。
- RECONSTRUCTION_ADMISSION PASS；RECONSTRUCTED_SCORE_COMPLETE：8GT、384metric、12绝对、96区域、96条件头、54对比。重建scorer exit0/inactive，原worker/原controller/新guardian/重建scorer原身份均退役。160静态bit parity PASS。结果execution_exit0_certified=false、independent_scoring_reconstruction=true。
- 核心科学11520+重演360更新；原生另64arrival/96backward/800forward，RESULTS.parameter_updates=11880只含核心。核心model forwards12176/static forwards32，原生计数不能混入字段。无新源训练，未新增权重或添加模型。
- 原worker完整kernel寿命GPU 28289.745191941969s，起点tick分辨率；主体28286.285861492157s。独立CPU评分完整81.315892696381s/主体76.40546655654907s。guardian elapsed 22796.888897657394s单列为监测跨度，不能当连续CPU计算时间，也不重复加GPU。
- 累计GPU 34773.396795879060s；已计CPU 2698.706994742460s，已含P4准备4.453143119812012及恢复两次CPU5.098078727722168。历史.0084569158基数差、近似body、未计时安装/阅读/编排保留，不清零或混其它聊天成本。
- 完成RUN_STATE算法输出62763386B；全窗口已登记actual150423215B、reservation0、辅助reserve1GB，总计1150423215B，低于18GB工作水位/20GB硬限。全root实际含code/archive/log，与算法输出数不混；未删除历史资产。峰值GPU1002023424B。
- 未按原1200/900秒限制终止：用户已取消未来时间/GPU配额，仍守新增空间限制。无GPU重试/评分重试。
- 公开自有源码/协议/中文分析/匿名汇总/无路径回执；不公开RGB/GT/权重/特征/位图/密集概率/私有trace/逐内容值/身份/私有路径/第三方全文。
