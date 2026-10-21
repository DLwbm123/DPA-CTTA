# 冻结残差块复用：真实资格 A01

动机与边界：此前替换分类器造成显著损害。本轮保留原分类头及全部权重/BN，改为增加已有表征计算，不以伪标签生成新权重。参考 ICLR27 本地目录 KdExztBpoT 同名论文 Architectural Sampling，arXiv https://arxiv.org/html/2610.01687v1 已读§3.1–3.3与§4.1。原文复用VLM decoder窗口并报告pass@k，不能当作密集分割收益证明。本轮仅迁移冻结层复用原理，是独立宿主试验，不是原文复现、原创主张或保证决策边界不变。

冻结假设：再次应用浅层同维度残差块，重新计算依赖当前特征的残差，可能比重放第一次残差提供有用表征；源码、配置与停止门先提交，GT仅在全部预测封存后独立评分。原8登记内容、4外观、512处理、原checkpoint，不加数据/训练/梯度。8仍为反复使用开发内容，不是独立患者验证。

固定首同维块res.layer1[0]，总应用2次，主REUSE=(z_source+z_repeat)/2。经forward hook替换单次输出为block.forward(out)，直接forward不递归触发hook；assert仅一次hook。全部后续层与skip遵循实际修改后的原网络路径，不混原skip。无伪标签/缓存/参数更新，无窗口或次数扫描。

七臂SOURCE、REUSE、SINGLE=z_repeat、REPLAY、LATE、AUGMENT、B。REPLAY在同一浅块输出relu(2*B(h)-h)，重放旧残差而不重新计算；LATE改为末同维块res.layer4[-1]复用2次；AUGMENT使用原模型minmax(x^.9)。后三臂均与SOURCE等权logit平均；SINGLE测融合必要性。阈值0，无后处理。强控信息/成本并非完全相等：层位置计算量不同，AUGMENT另一次完整前向，残差重放较便宜，如实报告，不以相同候选数声称等FLOP。

每输入5完整网络+2额外残差块，每worker32输入，首次各加1 identity-hook完整前向检查，合计322完整网络+128额外block，无decoder-only。SOURCE封存bitparity、identity-hook精确一致、模型/BN不变、无梯度、拒绝GT/旧评价/未登记资产/网络，两worker完整logits/masks/诊断exact后独立GT评分。

原合取门：REUSE−B OC≥.30pp且内容bootstrap95%下界>0；对SOURCE/SINGLE/REPLAY/LATE/AUGMENT差下界均>0；OD−B下界≥−.10pp；最差输入头损失≤1pp；不增空头/嵌套；全表面/vCDR可定义；OC ASSD和HD95比B的比值CI上界≤1.05；vCDR差CI上界≤.005；REUSE与REPLAY logit差>1e-3比例≥.1。8等权，外观[2,1,1,1]/5，bootstrap2000/seed20261018。失败关闭固定早块二次复用，不扫层/次数/融合强度/激活修正或挑控制臂变主结果。

my-gpu GPU1固定UUID，原时钟3600s，独立guard3540/stage3530；新root基线0、预计768MiB、预留1280MiB、hard1GiB；首输入成本门。写前NFS容量probe、GPU显存余量、父子中性argv/ps/nvidia；项目18GB工作/20GB硬限+aux1GB。wall阶段费用非GPU kernel；失败也记账。完成公开自有代码/协议/聚合/报告，不公开样本/GT/权重/张量。
