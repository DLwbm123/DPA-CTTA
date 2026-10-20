# Vessel-Bend512-BC-v1：首次真实测量前完整冻结

本轮仅8个既有开发内容×identity/gamma.7/RGB(1.15,1,.85)/blurσ1共32输入；不增加图像、模型、训练、seed、encoder或GPU。CPU既有依赖。A算子bend.py原样复用，不改变σ2/β.5/c.03/高.10低.03/h4/10°/r3/轮廓规则。测量不是原论文r-bend算法复现，也不代表真实语义。证据API只接当前绿色通道与B OD，不接收条件标识、OC或同内容参考。

输入逐文件PIL RGB bicubic512，float32/255；按原外观公式变换并clamp[0,1]，然后全图全通道共同minmax，取绿色通道。blur7×7、reflect、σ1。两独立进程重新读取8文件并重建32视图，比较输入数组、点、各权重、支持、分数、联合mask逐项exact；这是当前CPU重演，不声称与历史GPU浮点输入逐位相同。0模型、权重、特征/logits新读取、更新。GT/旧效用/旧结果不进入worker；只预处理原E8无标签scores缓存，白名单导出block_id/query/size/oc_pixels/原direction/distance_bucket/common及LOGIT_Q/RAW_MARGIN/GEOMETRY_ONLY。旧exact LOGIT_MATCH未保存，明确用上述三个现有强控，不冒充它或机械重编码。受限文件访问审计及网络阻断覆盖worker，负向测试必须通过。

SOURCE/B沿现有封存位图：3028 OC-only块全登记，147非共同保留为覆盖说明；分析共同2881块、6997 OC编辑像素。每帧SOURCE/B联合差异8连通分量逐项重算，query须精确对应OD不变/OC变化的全分量；不得新增空间。实际行动B→SOURCE，旧direction按SOURCE→B，仅作配额标签。OD永远B。所有证据在访问OC候选前提取并只读。

U定义：正BEND权重点上min(d²/9,1)在B与该单块候选间至少一处差>1e-12；不是q>0，也不依GT。无弯折/主轮廓为空统一弃权0，数值/访问/计算错误必须raise。BEND/VESSEL_ONLY/SHIFT及三个缓存对照全部同U，其余块共同0平分；2881完整分母不可删。q=E(B)-E(candidate)，BEND权a*k、VESSEL_ONLY权a、SHIFT逐路径半周循环a*k；支持与q分开。SHIFT预冻非退化门：在bend/shift非零权重位置并集上权重不同位置比例≥.5，以及在两者非零分数(|q|>1e-12)块并集上|qB−qSHIFT|>1e-12比例≥.5；分母0门失败。报告实际分母、变化数和最终动作变化。保持同U可使shift无效，不能把退化对照当成功。

影子决策必须无GT预生成：只选U且q>1e-12的单块合法B→S撤回；合法要求OC非空、位于B OD、8连通单分量、无4连通孔。同时安装所有正分合法块，联合不合法则整帧回退B。不扫预算或阈值。对照按BEND实际回退后选块数，以(exact size,旧direction,distance_bucket)逐桶从同U的单块合法集合取最高对照分（同分block_id升序），不要求对照正分；联合同样安全回退，若回退导致配额不再exact则匹配门失败。决策两遍exact封存；GT后不再生成或修改mask。

两worker均OS0且PID/start精确退休后，独立scorer才核8GT原登记SHA256并读取，nearest512、OD<255/OC==0。来源摘要校验是原协议必需。内容绑定：4条件权[2,1,1,1]/5，8内容等权；bootstrap2000、seed20261018。NA不填0；每帧AUC无正或无负则NA，内容内按可定义条件权重归一，CI仅8内容全定义时提供；完整输入/内容分母明确。

第一门B1：只算U内联合OC Dice最优上界。以B面积P、GT面积G、交集T及每个不相交B→S块的面积变化a/TP变化t，复用精确Fraction比例优化器solve，联合最大化2(T+Σt)/(P+G+Σa)，绝不加单块utility。每帧重建oracle mask并核Fraction结果，共64核对（共同全集及U各32）。忽略拓扑/嵌套约束，必须标RELAXED_UPPER_BOUND；包含空集因此上界≥0。重算共同全集，与原公开未舍入+.4721342181562814pp的同权重结果差≤1e-12；B基线复核87.8052924668%（1e-8容差），不是identity90.98749%。若U上界平均<+.30pp，硬停B2/C，标NOT_RUN_SPACE_GATE；不构造效用标签、不算AUC、不做局部GT错误或子组追分。≥门仅避免空间否决，不证明合法可达。

第二门B2仅在B1过后：为2881共同块计算精确单块Dice utility符号，零utility不参与正负配对但保留计数。BEND BLOCK AUC≥.60；唯一主差BEND−VESSEL_ONLY均值≥.05且配对CI下界>0；对SHIFT、LOGIT_Q、RAW_MARGIN、GEOMETRY_ONLY四强控BLOCK差全部CI下界>0；定义输入≥24/32、8内容全定义；utility绝对值配对加权主差≥0；上述shift两个非退化门均须过。失败C标NOT_RUN_SIGNAL_GATE，不翻转q/调方向/窗口/分层救分。

第三门C仅在B1+B2过后评价已封存mask：BEND−B OC均值≥+.30pp且CI下界>0；SOURCE对照完整报告；OD exact B，每帧每头不新增空、每帧不新增嵌套，OC表面/vCDR32全定义；任一内容×条件×头Dice下降≤1pp。OC ASSD和HD95先计算加权内容均值，bootstrap均值比的97.5%分位≤1.05；vCDR误差增量CI上端≤.005；所有实际配额匹配对照的OC Dice差CI下界>0且配额没被回退破坏。报告分头Dice/ASSD/HD95、vCDR、空mask/嵌套、全部NA/分母；不把B继承收益算模块收益。

预算：CPU单独新轮≤1800秒，独立守护1740秒保存停、controller阶段剩余1730秒，输出hard128MiB/预留256MiB，预计≤100MiB；新root运行前baseline0/NFS容量与写读探针。原始UTC/boottime从首次实际运行起唯一，修复共享clock。每服务器新增<50GB/≤24h及更严20GB/18GB/aux1GB继续；不删旧文件。中性父子argv、CUDA_VISIBLE_DEVICES=''，只my-gpu，不查询GPU/旧服务器。代码与本协议先commit冻结，之后才准入。失败和工程异常同样交付，父OS未知保留null。

所有历史E8/selector/crop/800负结果不变。本实验开发内容重复使用/患者独立性/预训练未知，不称CTTA泛化或临床成功；公开仅自有源码、中文匿名聚合报告和无路径receipts，私有数据不发布。唯一ctta30分钟原位维护，结果自主分析，单唤醒最多本有限B/C轮。
