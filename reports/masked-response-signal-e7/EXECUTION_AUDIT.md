# E7 遮挡响应信号：执行审计

执行source与结果见本目录receipts。独立科学定义在 `../masked-response-signal-qualification/PROTOCOL_ZH.md`，配置 `configs/masked_response_signal_e7.json`。本轮E7指遮挡响应信号资格，并非Pro早先建议的“E6通过后安装零阈值规则”的性能轮；E6未通过的判定不变，没有安装任何选择器。

## 已冻结的实现与准入

- 源eval、no_grad、所有参数requires_grad=False、BN buffers冻结；模型、源训练、数据内容、seed和权限不扩大。RDC/ENGRAM关闭。
- 每输入一次源+八次mask，32输入全部独立重演，同一过程总576完整encoder；64次源head重构是额外head-only计算，不作为额外完整encoder。
- RNG、rho及coarse mask索引都独立重建，实际4×4 patch铺放、归一化输入乘0；不变换标签或裁剪，不按query可见性筛块。
- 原E6共同集合现场重算，与仅含block_id/common布尔的封存对照逐块核验；准备进程从既有E6输出只提取该无标签元数据，不向新worker提供旧分数、效用或oracle选择。
- 所有源/原shift/logit范数、锚点/uid/score与mask索引在全32输入重演。dense features/logits/probabilities/masked RGB不落盘；score和mask坐标私有保留。
- 熵采用 `sigmoid(z)*softplus(-z)+sigmoid(-z)*softplus(z)`，无epsilon；合成极端logit与有限概率公式对照。负熵融合高分接近零，不偷偷反转符号追分。
- 16项CPU全执行PASS，无skip：E6九回归，新增mask坐标/RNG/数量、熵极限、null/方向、均值与shift关系、共同集合/score重演、toy源不变/无grad/CUDA未初始化、当前内核自身pidfd+signal0。
- CPU stage实际开始前保存原始UTC/BOOTTIME，controller和独立guardian共用起点；86340s保存停止/86400s硬退休，256MiB root水位/320MiB预留，修复共用原clock。guardian基于pidfd绑定登记PID/start_ticks，空间safe walk忽略ENOENT且不follow symlink。
- NAS新root写前检查实际NFS/剩余容量并小写读，登记reservation后才复制64端点/code/archive。GPU准入现场核UUID/free≥8192，父子argv与worker首次真实nvidia中性条目核验；不干预其它聊天。
- worker只允许8RGB/已有checkpoint/自有64bits与无标签共同集合ID；拒旧根/标签/scorer模块，包括pyc。worker exit0原身份退休后，独立CPU scorer才读8GT。

完成、退出、成本、是否触发守护及所有覆盖以本目录最终JSON为准；controller/guardian未实捕OS退出码时保留null，不能用运行态stat中的0冒充退出成功。统计仅8重复开发内容描述bootstrap，无独立患者或临床结论。
