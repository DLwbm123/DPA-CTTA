# 成对扰动下的解码器激活切换偏差修正：真实资格 A01

预冻假设：此前大幅空间/表征改写破坏原任务信号，本轮保留原表示中心，只测局部成对扰动引起的非线性响应。若外观扰动在深层产生近似对称的不确定性，解码器的有限尺度激活切换可能使平均输出偏离中心；减去该对称残差可能修正预测。假设未证实，不保证当前域偏差满足噪声模型，不宣称原创。冻结ReLU/BN解码器是分段仿射，普通Hessian几乎处处为零；这里是跨激活区域的有限差分，不伪称在固定区域内测得光滑曲率。

仍限原8登记内容×4外观32输入、原checkpoint和512预处理，全部权重/BN冻结。当前x及minmax(x^.9)、minmax(x^1.1)各执行原网络，取deep d0,da,db及原skip s0。delta=(da-db)/2，仅将其全局L2范数上限设为0.1||d0||（小于上限则保持），不删除方向、不改skip、不改变原表示中心。原decoder D下S=(D(d0+delta,s0)+D(d0-delta,s0))/2，主DEBIAS=2*z0−S。仿射区域内主结果等于SOURCE；跨激活门才可能有差异。alpha=1固定，不根据标签调修正幅度。

六臂SOURCE、DEBIAS、SMOOTH=S、ENSEMBLE=(z0+za+zb)/3、RANDOM、B。RANDOM用固定seed20261022标准正态方向（每输入同形状同方向），归一到delta完全相同全局L2能量，再作相同成对残差修正。它不匹配逐位置/通道能量，保留该局限。SMOOTH控制普通随机平滑方向，ENSEMBLE控制同信息外观集成，RANDOM控制与真实外观方向无关的等能量响应。logit阈值0不后处理。与旧rank8子空间删除和gamma.7跨配对的机制不同：无投影删除、无深浅跨配对、无参数/层/强度扫描。

每输入3完整网络+4decoder-only，两worker首输入各额外1原生decoder精确重建，共192完整网络+258decoder-only。SOURCE封存parity、模型/BN不变、无梯度、GT/旧评价/非登记资产/网络拒绝，两worker完整logits/masks/诊断exact重演后独立GT评分。仿射抵消与ReLU门切换的代数检查只是工程自检，不当科学效果。

原合取门不改：DEBIAS−B OC≥.30pp且95%内容bootstrap下界>0；对SOURCE/SMOOTH/ENSEMBLE/RANDOM差下界全>0；OD−B下界≥−.10pp；最差逐输入头损失≤1pp；无新增空头/嵌套；全部表面/vCDR可定义；OC ASSD/HD95与B比值CI上界≤1.05；vCDR差上界≤.005；DEBIAS与RANDOM logit差>1e-3比例≥.1。8等权、外观[2,1,1,1]/5、bootstrap2000/seed20261018。若修正退化或损害，就关闭本固定残差修正，不扫gamma、cap、alpha、层、随机种子或反向后选救分。8反复用开发内容非独立患者验证。

仅my-gpu GPU1指定UUID，新root0/预计768MiB/预留1280MiB/hard1GiB，原clock3600/guard3540/stage3530、首输入成本门；NFS/容量/probe/显存/父子中性argv核。项目18GB工作/20GB硬限+aux1GB，历史清理不冲抵新费用。阶段wall如实记，非GPU kernel。完成/失败公开自有源码协议聚合，原图/GT/checkpoint/张量私有。
