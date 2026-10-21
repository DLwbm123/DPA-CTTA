# 图像引导预测重建真实资格 A01

运行前冻结。LQN原版不直接迁移：§3.2/算法1需要统一D的多层token与坐标学生；本宿主up1–4为256维、末层head为32维。改变投影、尾部或幅值约束会成为新的适配版本。余弦PAD不识别范数，而当前固定仿射sigmoid头对范数敏感。该问题不是断言所有LQN迁移都不可能；本次不另建缺少规格的学生、不运行LQN，也不回到旧BN路线。

改检验当前RGB是否提供有益空间约束。依据He等Guided Image Filtering (ECCV2010) §3.1公式5–8：https://people.csail.mit.edu/kaiming/publications/eccv10guidedfilter.pdf 。该文证明局部回归算子，不证明本任务收益。本轮是经典组件资格、静态读出，非原创或持续适应成果。

原8登记内容×4外观、原checkpoint、seed和minmax预处理均沿用illumination-real-a01；只当前无标签图像，模型eval且参数/缓冲冻结。每输入一次原模型前向，共32/worker、双worker64。独立生成结束再读8登记GT评分。原SOURCE逐位匹配迁移基线；不新下载数据/模型、不访问旧192内容、不改变数据范围。

固定灰度I=.299R+.587G+.114B。对SOURCE两头raw logits p逐通道引导：a=(box(Ip)-box(I)box(p))/(max(box(I²)-box(I)²,0)+.001)，b=box(p)-a box(I)，q=box(a)I+box(b)。box半径8、17×17、scipy reflect半样本对称边界，float64计算，结果float32。参数一次选择，不使用评分调参。五臂SOURCE、GUIDED、BOX=box(box(p))、SHIFT=引导图宽轴循环257像素后同公式、B=原TOPO_SMOOTH。BOX等于常量引导的退化控制；SHIFT保留灰度直方图但破坏空间对应。它们不保证改动量匹配。所有新臂阈值0logit，不混合旧B、不强制嵌套。

最小工程断言：常量保持、常引导等于双box、输入仿射等变、错位非恒等。独立worker完整logits/masks exact replay只验证确定性。生成deny GT/网络/非登记数据模型；score在两worker退出后执行。沿用固定内容权重[2,1,1,1]/5、8内容等权、2000内容配对bootstrap、seed20261018。重复使用的开发样本不等于新患者验证。

推进门：GUIDED−B OC均值≥.30pp且95%下界>0；对SOURCE/BOX/SHIFT差下界>0；OD−B下界≥−.10pp；最差任一输入头较B≥−1pp；不新增空头、不增加嵌套违规；全部表面/vCDR定义；OC ASSD/HD95比B之比的CI上界≤1.05；vCDR误差差上界≤.005；GUIDED和SHIFT logits差>1e-3的比例≥.1。全门合取，失败就关闭固定配置，不调半径/epsilon/颜色权重、不改阈值救分。记录全部负项。

预算：仅my-gpu指定GPU1，root基线0，预计768MiB、预留1280MiB、hard1GiB；原clock3600秒、独立guard3540、stage3530。首次输入完成估计总成本超限即停。项目18GB工作/20GB硬限、aux1GB以及单服务器<50GB继续。启动核NFS/probe/GPU，父子中性argv并验证。CPU/GPU按阶段进程wall时间记账非kernel。失败也发布自有源码和聚合，原图/模型/第三方论文不发布。
