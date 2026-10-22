# r09：区域干预选择的翻转对齐专家

运行前冻结。r08完整完成：FULL−DIRECT+0.418006/+0.214570pp，FULL−独立CONTROL+0.183140/+0.323795pp，但两种子未过原门。第二种子clean−0.423321、尾部−2.133627pp，残差不确定度支持没有完全解决问题。原canonical候选持续弱于DIRECT，可能使区域选择可用的纠错内容有限；这是机制假设而非已确认因果。

唯一主要变更：四候选由SOURCE/DIRECT/CANON_SOURCE/CANON_DIRECT变为SOURCE/DIRECT/FLIP_SOURCE/FLIP_DIRECT。对512输入水平翻转，真实重跑冻结骨干及对应r03冻结DIRECT纠错器，再将256预测水平翻转回原坐标。没有插值对齐误差，不把翻转前掩膜当作新专家预测。canonical颜色变换和fit颜色参考不再使用。此改动旨在提供几何视图互补，不声称翻转专家更强或来自新模型。

其余保持r08：0.5固定候选干预幅度、整头Dice区域替换信用、无收益源区域DIRECT目标、4p_DIRECT(1-p_DIRECT)残差支持、双视图几何/一致性训练、源划分和评价选择。两种子CONTROL/FULL各2000步，400评价，独立CONTROL同样接收翻转候选作为网络输入、同辅助信用和训练资源，但固定DIRECT混合加同残差。FULL必须胜该CONTROL，不能把输入信息或残差容量收益归给选择。另报告FLIP_SOURCE/FLIP_DIRECT/MEAN/NO_RESIDUAL/DIRECT_WEIGHTS。冻结专家不额外训练，模型调用预算与r08相同。

只有源fit训练/源dev选择，真实目标继续封存；原门和内容级证据限制不变。无部署梯度/持续状态。2GB/4h且原窗2026-10-23T04:00:43Z硬截止，总预留24.5GB。禁止根据结果扫翻转组合或为候选表现而调baseline。若翻转自身解释全部收益，应明确承认简单视图集成效果。
