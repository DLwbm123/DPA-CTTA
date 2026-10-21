# 粗网格全局语义与细网格图像证据：真实资格 A01

运行前冻结。互近邻传播未优于SOURCE/B，故不再调整其图/层/强度。新假设：单图512网格的全局语义与同一原图1024网格的细节可在原解码器内分工，减少高分辨率直接推理因有效感受野变小产生的语义漂移。此为独立提出的多网格内部推理资格试验，不声称原创或有效。与旧E8裁剪信号预测编辑效用不同，本轮无裁剪、无旧B→S局部动作选择，直接产生完整掩码；也不是旧gamma深浅配对的层/系数扫描。

资产仍限原8登记图像、原checkpoint；仅同一图像字节分别BICUBIC缩放512和1024，未读取额外内容或训练。SOURCE的512预处理/4外观保持逐位原逻辑。高网格按相同gamma.7和通道[1.15,1,.85]外观；模糊sigma2/radius6对应512 sigma1/radius3的物理尺度，离散卷积/缩放不严格交换，此差异在全部高网格对照中共享。两路各自minmax。高网格可能本身是原图上采样，不能无条件称新增原始信息；原图尺寸保持既有私有记录。

每输入先执行512和1024原模型，获取deep和4级skip。把512 deep用bilinear/align_corners=False映射到1024 deep的空间尺寸，保持全部通道，连接原1024 skip，经原decoder输出1024 logits，以area回512后阈值0。无额外归一化、范数调整、融合强度参数、训练或BN改动。

七臂：SOURCE原512；MULTIGRID上述低网格deep+高网格skip；HIGH原1024输出area回512；ENSEMBLE=(SOURCE+HIGH logits)/2；RESAMPLE仅由当前512输入bicubic上采样1024并clamp[0,1]、经原encoder获得skip，配相同低网格context解码；SHIFT将context沿宽循环半幅后配原高网格skip；B封存TOPO_SMOOTH。HIGH控制高分辨率本身，ENSEMBLE控制同信息普通集成，RESAMPLE控制纯插值网格变化，SHIFT控制内部空间对应。不同网格卷积未必与插值交换；不把工程对齐当科学证明。

每输入3完整网络+3decoder-only，两worker首输入各增加3原生decoder重建核，共192完整网络+198decoder-only；两worker全量logits/masks/诊断exact重演。SOURCE封存bit parity、state_dict不变/无梯度、生成禁GT/旧评价/网络/未登记资产；全部生成退休后独立CPU评分。插值常数保持自检只用于接口，不当科学结果。

合取门不改：MULTIGRID−B OC≥.30pp且内容配对95%bootstrap下界>0；对SOURCE/HIGH/ENSEMBLE/RESAMPLE/SHIFT差下界全部>0；OD−B下界≥−.10pp；最差逐输入头下降≤1pp；无新增空头/嵌套；全部表面/vCDR可定义；OC ASSD/HD95比B比值CI上界≤1.05；vCDR差上界≤.005；与SHIFT logit差>1e-3比例≥.1。8内容等权、外观[2,1,1,1]/5、2000重采样/seed20261018。失败关闭固定512→1024多网格方案，不扫分辨率、deep/skip层、插值或融合救分。8为反复使用开发内容，非独立患者结论。

预算仅my-gpu指定GPU1，原clock3600/guard3540/stage3530，新root基线0/预计768MiB/预留1280MiB/hard1GiB。启动NFS/probe/容量/GPU≥8192MiB余量，父子中性argv核，首输入cost门。192完整forward含高网格成本而非等价512F计费；stage wall如实报，不能只拿forward次数声称同算力。项目18GB工作/20GB硬限与aux1GB保留，历史清理不冲抵新费用。结束后公开自有源码/协议/聚合；图像/GT/checkpoint/张量保持私有。
