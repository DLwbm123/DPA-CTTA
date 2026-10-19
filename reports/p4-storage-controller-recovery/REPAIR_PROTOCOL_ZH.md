# P4 控制器独立恢复协议（冻结后执行）

## 故障与授权范围

原GPU worker仍运行、计数推进；原controller扫描正在删除的临时依赖目录时遭FileNotFoundError退出，故空间轮询和自动CPU评分接力中断。保留原代码、原controller.log/身份/进程文件；不杀/重启/注入/修改GPU worker，不改模型、loss、数据、seed、90步、384输出、主比较、统计或安全门槛。本修复只恢复监督并如实记录退出证据。原已分配root/reservation内增加少量code/log，空间上限不变，无新GPU/显存准入。

空间扫描用os.walk/lstat且只跳过ENOENT（同时删除的目录/文件），不跟随symlink；权限/I/O等其它错误不吞掉。PID+start_ticks验证原worker，pidfd绑定身份、后台等待退出，原有2GiB全root守卫继续；超限仅向该pidfd发TERM/KILL，不影响其它任务。

## 退出证据与分支

pidfd可确认进程退出，但非父进程不保证获得退出码。等待线程即时尝试读取/proc/pid/stat字段52的wait状态，只在相同start_ticks且terminal/Z/X状态时解码；运行态字段0绝不当exit0，已被PID1回收则exit_code=None。参考 [Linux proc stat](https://man7.org/linux/man-pages/man5/proc_pid_stat.5.html) 和 [pidfd](https://man7.org/linux/man-pages/man2/pidfd_open.2.html)。不申请额外权限、不ptrace、不改调度器或注入handler。

1. 若实见exit0且GPU_COMPLETE384/32/11880、首图资格、四臂logit/model/Adam重演、source/BNbuffer/ACCESS、384索引/位图、11880顺序JSONL全部完备，启动**原冻结CPU scorer**，原评分公式不改；原控制器故障仍单独披露。
2. 若实见非零exit或完成证据不全，不启动标签评分，保留停止/成本。
3. 若pidfd证实退出但OS码不可得、上述全部完成证据齐全，另立**独立CPU评分重建**：worker exit_code保持None，原exit0资格门不恢复。重建scorer仅改退出准入与结果回执名（RESULTS_RECONSTRUCTED/RECONSTRUCTED_SCORE_COMPLETE），Dice/区域/结构/CI/主比较/全部安全门逐行一致并CPU校验。原SCORE_COMPLETE不生成、不冒称原资格PASS；可报告重建任务指标及其明确执行证据限制，不能把原exit0字段填零。

GPU耗时从原kernel start_ticks到pidfd退出CLOCK_BOOTTIME计算，含已跑部分，不从监控接管时清零；起点只有一个时钟tick精度，原controller内存中的wallclock起点无法恢复。guardian elapsed单列；CPUscore仍用直接子进程wait记录实际exit/耗时。原controller退出码未观察不伪造，traceback和原身份退役单独留证。

## 校验和最终交付

8项CPU：临时路径/文件消失、其它错误不忽略、symlink不跟随、带空格comm解析/身份、运行态0不当成功/信号退出、缺证据拒准入、重建函数仅准入和回执差异、非子进程pidfd退出观察。合成临时文件/中性短子进程，无RGB/标签/权重/模型/CUDA。内核已回收时测试允许unknown而非假设0，不声称保证捕获退出码。

单次中性恢复入口，验证guardian/原worker身份与ps；恢复controller退休后再核完整ledger、原/修复controller和worker/scorer身份。公开自有修复代码、中文审计、匿名回执，不发布原临时目录名字/私有路径/进程身份/逐内容数据。20GB新增硬限/18GB水位/辅助1GB、无源训练/RDC/ENGRAM和已停路线约束全部保留。新空间守卫不回改旧运行失败。
