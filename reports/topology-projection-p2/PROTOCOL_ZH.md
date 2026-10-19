# P2冻结：直接二值拓扑投影与固定平滑强基线

用户授权12小时自主推进性能/修复，不再问例行授权，取消未来时间与GPU配额；新增空间硬限20,000,000,000B，工作水位18,000,000,000B并预留辅助文件空间。不新增源训练/数据/模型/seed/RDC/ENGRAM，BN尺度和Calibration Maps不做。P1及旧冻结结果不改。

## 信息依据与改造

已读TopoSculpt §3.2用解剖Betti先验与完整性项测试优化；本轮只抽取二维盘/杯单主体、无孔的结构先验，改成不训练的直接二值投影。原文是3D管状PH/90步等优化，不能拿其成绩认证此简化。LCC/填洞是成熟后处理，不是原创ICLR/TMI贡献，也不是原PH算法复现。

P1已公开结果后发展的重复开发轮：固定源概率3×3 binomial平滑有双头+.2841/OC+.4670pp正信号，ASSD仍劣源。只读P1原始二值输出、不读标签的新census发现32输入源两头均有额外组件/孔洞；平滑后仍有OD9/9、OC16/19输入存在组件/孔洞缺陷。计数提供实际可修对象，不证明修后Dice一定更好。原先空/嵌套0不足以表示拓扑正确。

本轮不用图像/梯度/教师/概率阈值搜索；输入仅P1已冻结SOURCE/SOURCE_SMOOTH的64bit mask和索引。不重推模型。禁读P1 scorer/public与本轮scorer；独立worker完成并退役后CPUscore才读8标签。事前已见P1总体结果，不声称独立验证。原无标签census和CPU成本保留。

## 确定投影与八臂

前景8邻接、背景4邻接。每头先保留最大面积8连通组件，精确等面积tie取扫描顺序首个；再4背景填洞。空源保持空，不造结构。两头独立，不加包含裁剪、面积/病理比例或GT约束。可能删错主体，可能破坏包含，必须报告。

SOURCE；SMOOTH（P1固定平滑不重算）；TOPO_SOURCE；TOPO_SMOOTH；LCC_SOURCE（只最大组件）；FILL_SOURCE（只填洞）；MASS_MATCH_SOURCE；MASS_MATCH_SMOOTH。后两臂分别精确保留对应投影的每头删除/填入像素数：在源0与1两组的行优先索引内，将投影所选位置序号平移N//2+1，保总面积/改动量，破空间位置。非新随机种子，可能偶然重合/退化，报告交集，不强称每帧全破坏。

## 唯一主与推进门

唯一主TOPO_SOURCE−SOURCE:OC_DICE均值≥+.5pp且8内容绑定条件/通道的2000次seed20261018描述bootstrap下界>0。须另胜SMOOTH及MASS_MATCH_SOURCE的OC区间下界>0；TOPO_SOURCE双头源比较CI>0、最差条件头≥−1、原条件静态再读均值≥−.2；OC真值前景及3px边界净纠错CI>0；ASSD/HD95/vCDR点估计均不劣源、覆盖不降、空/嵌套不增。任何失败advance=false。

TOPO_SMOOTH−SMOOTH OC是预定次要，不替代主，不按结果选择LCC/填洞/tie/连通轴。所有子臂/控制完整报告。静态没有持续状态，continual_benefit_tested=false；原条件再读不是源返回/遗忘。

## 执行与空间

32固定输入×8臂=256bit预测，8绝对64条件头64区域35对比；五块权重[0,1,2,3,0]/两静态重读=640逻辑读出。SOURCE/SMOOTH64bit与P1逐位一致，所有256数组在同一冻结输入上重算精确；无网络前向/权重/RGB读取、无CUDA/参数/optimizer/backward。C+H=D完整，worker与scorerexit0 inactive、controller原身份退休/finalledger后交付。

CPU检查先过；新root写入前登记18GB工作水位内的空间reservation，NAS实际挂载/容量/小写读。输出根（含代码/cache/log）限2GiB是空间约束，不是时间/GPU配额。CPU控制器不设运行超时，监督自己子进程/文件大小与已登记空间；不终止其它聊天任务。旧失败/数据保护，不删历史结果腾空间。仅自有代码、匿名聚合及无路径receipts经代理push/SHA/匿名200，隐私原始物料不公开。
