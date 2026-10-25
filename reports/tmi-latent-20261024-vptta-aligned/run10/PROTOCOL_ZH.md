# run10 gamma残差估计（运行前冻结）

diag09预设规则仅gamma分量合格，故冻结run06 RGB估计与全部网络、run01基础模型，新增gamma残差头。输入冻结网络128输入64维池化特征，以及每RGB通道17个强度log分位数（q=.05到.95均匀，64x64下采样，减本通道中位数，共51维）；log中心化在无裁剪时消去乘性增益，但不保证分离内容。MLP115→64ReLU→1，最后层零初始化；输出原loggamma+.25tanh(residual)，裁剪[-.5,.5]，RGB完全不变。

同run06源fit/dev划分，base已见全部source且dev反复开发，非独立验证。每源1200步batch8，seed92510，AdamW .001 wd.0001，clean与随机已知gamma/RGB各一半；唯一损失loggamma MSE，不使用目标数据/结果或分割梯度训练，区分估计精度与分割效果。所有五源训练后再源dev clean/fixed_appearance/color/blur评价I原估计、G新gamma、ZERO；blur无真实gamma，不用于MSE。无时间状态。

源gate：两已知扰动G−I五源等权平均≥.1pp、每源≥0；clean G−I≥−.05pp；扰动pooled p10≥−1pp；三已知条件gamma MSE较I≤.8。全部过才五源原协议目标G一次，I复用run06，ZERO复用run01，不重跑选参。失败不目标不改门。公开全部阴性结果；成功也仅为独立外观校正，不能称持续学习或安全成功。

硬截止2小时、TERM提前300秒/KILL60秒；1GBsoft2GBhard预计<200MB，新增总<50GB、45GB预停，无额外GPU小时配额。唯一30min心跳，健康静默，完成后自主分析推进。
