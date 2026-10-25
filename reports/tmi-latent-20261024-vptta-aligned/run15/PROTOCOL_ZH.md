# run15 源监督逐图动作选择（运行前冻结）

run12/14风险损失系列关闭不续调；diag11 G−I源条件43.01%下降，检验收益是否可预测。冻结run01base/run06 I/run10 G，仅学习选择两已有动作，不训练/微调分割或gamma网络，无跨图状态。

source fit沿用80/20，base见过source/dev多轮使用，非独立泛化。每fit图clean及2次独立随机光度（seed92515各源重置，run06增强分布）收集I/G硬Dice平均差；特征为run10冻结115维features、G−I loggamma差1维、I/G平均二元熵2维，共118。熵来自各自分割概率。预测器Linear118→32ReLU→1末层零初始化；特征仅用fit均值/标准差归一化（std下限.001），标签clip(DiceG−DiceI,−5,5)/5。AdamW .001wd.0001，1200stepsbatch64 seed92515，MSE。阈值固定0不选参，预测>0选G否则I；缓存只私有特征/标签不复制图像。

全部五源训练后dev clean/fixed_appearance/color/blur评价I/G/S。源门：两已知扰动S−I源等权≥.1且每源非负、S−G平均≥0、clean S−I≥−.05、pooled p10(S−I)≥−.1。全过才原五源目标S一次，I/G复用run06/10；任何失败不目标不改阈值/门。标签仅训练/源验证，目标回调只用图像和预测。固定两动作选择不代表解决所有域移位，源安全门不保证目标安全。

3h硬截止TERM300KILL60，1GBsoft2GBhard预计200MB，总新增<50GB45GB预停。11:37汇总不强停健康任务。公开匿名结果源码协议，不公开数据权重逐图或fit特征。
