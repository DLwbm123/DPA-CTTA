# FOV800-Paired-v1：完整视野800单点资格协议

本轮沿持续自主授权，在首次实际运行前冻结。原固定crop路线已关闭，本轮是独立的完整视野观测资格，不是上下文因果实验或新CTTA方法。固定既有8开发内容、identity、原权重、seed20261018；无新数据/外模型/训练/梯度/BN适应/融合/selector或尺寸搜索。

四臂：SOURCE为原512全图；B为既有identity封存静态TOPO_SMOOTH；N为完整800RGB；R为旧PIL RGB BICUBIC512重建完整800。PIL原RGB解码float32/255，lo/span来自同一旧512全图；SOURCE/N/R均使用同仿射再显式clamp[0,1]，不分别归一化。R使用torch bilinear align_corners=False antialias=False，先重建后归一化。SOURCE512直接输入模型，N/R800直接输入同模型，不能先缩回512。所有输出为两独立头logits，N/R logits用相同bilinear align_corners=False antialias=False回512后>=0；SOURCE logits直接>=0，不softmax、不嵌套修复。真实原文件800不是传感器原生分辨率证据，N−R不是纯高频因果效应。

CPU check仅合成：常数、双坐标平面800→512 half-pixel映射，512→800→512固定内域剔2像素误差<1e-9（float64），完整域边缘差单列；边缘step的范围/单调性及重复构造exact。静态800/32整数格可拼接只作为形状依据，实际800首次前向属于正式预算，输出形状/finite/head/资源失败硬停，不换768、padding、decoder或滑窗。空预测或面积收缩仅记录，不依诊断挑选内容，工程门未触发则完成8图。

两遍使用独立worker进程，每遍8×(SOURCE512+N800+R800)=24完整encoder，总48；第二遍重新读原RGB bytes、重新解码/重建、重新建模载入同权重。比较原文件bytes、输入tensor、原始logits和features的dtype/shape/连续字节SHA256摘要，以及最终512 mask逐位一致。摘要是本轮预登记的独立重演证据，非例行扩展哈希；不保存完整features/logits。每遍SOURCE核封存8bit、head重构24、参数/缓冲前后exact无变、无梯度；记录全部归一化层类/实际training/track_running_stats/eps等配置，并比较两进程一致，不修改它们。记录原800输出及回映512输出的面积比例、空头和logit分布匿名汇总，不把诊断当安全证明。

worker禁止GT、utility、旧scores/scorer及网络，允许同8RGB、8SOURCE bit、原权重/代码/环境。两worker分别OS0且精确身份退休后，独立scorer才可读原8GT核既有登记摘要，并读8B位图；32mask齐，不读取旧效用/结果挑规则。

统计：8内容等权、identity唯一条件、同一组seed20261018的10000次配对bootstrap、percentile95%CI。唯一主比较N−R的全图OC Dice pp。完整报告N−SOURCE、N−B、R−SOURCE、R−B及N−R的全部OD/OC Dice、分头ASSD/HD95、vCDR误差，胜/平/负计数、最差内容差和留一均值范围。NA保留8分母，不足8则完整CI=null。无多重检验后选、无新主比较。

四层预冻门（新增数值为本开发轮研究推进标准，不是临床安全界值，不能倒套旧实验）：
- G0：全部输入/形状/finite/状态/独立重演/GT隔离/完整32评分通过。
- G1观测：N−R OC均差CI下界>0。
- G2竞争力：N−SOURCE和N−B OC均差CI下界都>0，且N−B OC均差≥+0.30pp。
- G3完整性：N不得相对SOURCE或B逐内容新增任一头空预测/嵌套像素；N双头边界16/16、vCDR8/8完整。OD相对SOURCE和B均差CI下界≥−0.20pp；N相对B任一内容任一头Dice差≥−1.00pp。OD/OC的ASSD、HD95四个“内容均值之比N/B”的配对bootstrap CI上界各≤1.05；分母为0或不足8为未通过，不加epsilon。N−B平均vCDR绝对误差增量CI上界≤0.005。
仅G0/G1/G2/G3全过才推进固定静态候选。G1通过而G2/G3失败，保留观测信号但输出操作不推进，最多对既有输出做一次独立CPU错误账本，不启动适应；N/R都严重落后SOURCE则关闭本固定直接800操作，不扫分辨率/BN/层或融合救分；N/R难区分但都优于B，保留静态操作证据、不称原文件独有增量。全过后才考虑既有四条件一致性复核，仍需下一独立协议；持续适应必须以后独立面对强静态教师以及RESET/CONT/PRE/POST，当前不执行。

仅my-gpu物理GPU1指定UUID，free≥8192MiB准入、缺则WAITING_GPU不启动原始时钟。批准NFS根内新root基线0，预留64MiB、输出硬32MiB，不复制RGB/权重。CPUcheck首次实际运行起600秒总预算，共用UTC/boottime；guard540秒、controller等待剩余530秒，工程修复不重置。中性主子argv与自身nvidia核验，独立guard不依赖SSH/ctta。每唤醒最多一个有限独立轮。公开仅自有源码、中文报告审计、匿名聚合/无路径receipt；输入/输出摘要与逐内容值、真实坐标、原像素/位图/features/logits/权重私有。8重复开发内容/患者独立性与预训练覆盖未知，bootstrap不消除开发偏差。
