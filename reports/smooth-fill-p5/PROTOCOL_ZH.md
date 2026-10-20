# P5 冻结协议：固定平滑后仅填孔的有限融合

用户12小时自主窗口内有据推进性能，不新增源训练、不扩数据/模型/seed，RDC/ENGRAM关、BN尺度/Calibration Maps停止。未来时间/GPU配额取消，新增20GB硬限/18GB水位/辅助1GB保留。本轮CPU-only，不用GPU/RGB/权重/模型前后向；所有参数/optimizer/backward=0。

## 为什么有依据，为什么仍可能失败

已读TopoSculpt的解剖结构先验在P2直接投影/填孔中有有限正证据，而P4全网络PH+自参考完整性虽目标下降，OC从源87.28降至38.10，背景与边界严重受损。因此本轮不重跑/调P4，而检验固定输出操作的融合：P1平滑有OC+.467pp，P2源填孔有OC+.344pp及前景/边界净纠错阳性，合并未必可加。P2的TOPO_SMOOTH同时删除非最大组件并填孔，本轮FILL_SMOOTH保留所有组件，仅将不连到外边界的4连接背景孔置1，能避免该删除操作，但也可能保留假阳性小岛。它保证原平滑mask的1不被删除、外连背景不改变，并不保证Dice/结构安全。

这是成熟平滑/形态学基线的预定融合资格，非新TMI方法/TopoSculpt复现/CTTA，不将局部阳性事后换为P4成功。已看P1/P2/P4结果做开发选择，不声称独立验证或新语义证据。若只是重现已有输出或未胜强控制，关闭此固定融合，不扫核大小/连通性/阈值或选择有利条件。

## 输入与六臂

只用P1封存SOURCE及SOURCE_SMOOTH64个位图及其索引。P1平滑是既定概率3×3 binomial后.5阈值，直接复用bits，不重新平滑二值或改阈值。worker禁P1标签/评分/public结果，scorer独立在worker exit0/inactive后才读8GT。8内容4条件32输入；两头独立，无硬包含裁剪/面积界/新seed。

六臂SOURCE、SMOOTH、FILL_SOURCE（源仅4背景填孔）、TOPO_SMOOTH（前景8最大组件后4背景填孔，等面积首扫描tie）、FILL_SMOOTH（平滑后仅4背景填孔，保留所有前景组件）、MASS_MATCH_SMOOTH。最后控制在平滑mask的0组行优先序号内将实际新增孔像素序号平移N//2+1，精确保总添加数、面积和改动量、无删除；可能重叠或完全退化，要报告，不假称每帧均破坏。所有操作deterministic，无参数搜索。

## 主比较/门槛与评分

唯一主FILL_SMOOTH−SOURCE OC Dice≥+.5pp且8内容绑定条件2000次seed20261018描述bootstrap下界>0。还须OC对SMOOTH/FILL_SOURCE/TOPO_SMOOTH/等质量错位各CI下界>0；双头源CI>0，最差条件头≥−1pp、原条件再读双头均值≥−.2pp；OC真值前景与3px边界净纠错CI>0，ASSD/HD95/vCDR点值不差源、定义覆盖不降、空/包含不增。任一失败advance=false。不换主/选副臂/看GT调操作。所有六臂全部报告。

C/H唯一输入原始总数与按[0,1,2,3,0]宏率分别报，C+H=D核；区域有重叠不可相加。两重复逻辑顺序共480逻辑静态读出，不是独立图或持续适应；原条件再读非遗忘/持续返回，continual_benefit_tested=false。患者/预训练成员未知，无独立患者/临床/TMI结论。

## 执行/停止与交付

新独立branch/source/config/root，CPU6项新检查加旧P2五项回归须通过；合成检查不读取RGB/GT/权重/CUDA。NAS容量挂载小写读/写前空间reservation；CPU-only不做无关显存检查。控制器复用原CPU阶段逻辑、采用已核os.walk/lstat只忽略ENOENT的空间计数，防P4同类扫描竞态。不重启旧任务，不干预其它聊天。

32输入×6=192bit输出；64source/smooth bit parity，192数组同输入重算exact；worker ACCESS 0RGB/GT/权重/前向/更新，MECHANISM完整新增数匹配。独立CPUscore8GT/192metric/6绝对48条件头48区域25对比；worker/scorerexit0inactive、controller原身份退休/finalledger/空间转actual后交付。任一工程失败保部分/成本/源/日志，修复另source不涂改；科学失败不盲调。无时间cap、root2GiB是空间分配，写前reserve2.5GiB并计所有code/archive/log；总新增守20GB，不删历史。

仅自有代码/中文报告/匿名聚合/无路径回执，经指定代理GitHub push、远程SHA/匿名HTTP200确认；禁止位图/RGB/GT/权重/特征/概率/逐内容值/身份/私有路径/第三方全文。
