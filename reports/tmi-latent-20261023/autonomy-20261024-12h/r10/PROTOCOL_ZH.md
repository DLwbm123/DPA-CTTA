# r10：影像条件解剖生成的离散混合路径

运行前冻结。复用ICLR2025 Generator Matching的条件概率路径与离散生成算子原则（方法段核查，非全文代码复现），不复用直接对比方法核心。不同于r06任意SDF加噪及r07展开残差训练；不再拟合影像纹理。不是CTTA或性能成功声明。

## 网络与训练

冻结r01最终两seed基础网。pOD/OC=sigmoid(-16d)，q杯=min(pOC,pOD)，q盘缘=pOD−q杯，q背景=1−pOD。数值floor1e-6后归一，固定不调。STRUCT_ZERO为q.argmax再按OD=非背景/OC=杯读出，原ZERO也保留。

终点网络输入RGB3、固定c4特征16、当前状态onehot3、q0三类3、时间1，共26通道；64分辨GN U形32/64/96，末层3通道零初始化，插值到256后加log(q0)，输出CE logits。两seed10225/10226，各1000次Adam lr.001/batch4/clip5，sampler seed+70000，同原源增强。只79fit训练，20重复dev最终一次，target0。每图t均匀[0,1)，先从q0逐像素采样z0，再以t概率取真实标签，否则取z0；监督终点真实三类标签CE，无额外Dice/SDF混合以避免改变后验学习目标。医学源基础头已有监督不再更新。单标注/重复开发集/患者UNKNOWN/额外源预算/r01 RNG限制保留。

条件路径每坐标的跃迁率向终点为1/(1−t)，学习终点分布替代未知终点，有限步更新核K=(1−a)I+a*qθ，a=dt/(1−t)。固定t=0,1/3,2/3，dt=1/3，a=1/3,1/2,1。自转移在核中计入；同步多坐标三步只是近似，不能声称精确CTMC或校准后验。

## 固定读出和对照

每seed20内容×4固定条件，一次最终ZERO/STRUCT_ZERO/DIRECT1/ENSEMBLE3/CHAIN3/CHAIN3_SAMPLE。链与独立预测共享第一个起点及第一次网络输出；两个额外独立起点在t0各做一次预测，与第一次概率平均为ENSEMBLE3。链按上述核更新两次状态，在第三次输出端用类别argmax作主CHAIN3，另固定随机终点样本CHAIN3_SAMPLE辅助；后者不是择样或新增网络调用。链与ensemble各3调用，共享1，实际每图5调用。完整每图固定独立Generator(seed+80000+condition_index*1000+content)，不由前图耗随机数改变，训练RNG单独保存。STRUCT_ZERO须拆开类别argmax与包含变化，不将包含恒成立算机制成功。无长期状态、无测试权重更新、无标签择步。

主门CHAIN3−原ZERO两seed>0且均值>=.2pp，各seed OD/OC>=0、clean>=−.3、内容q05>=−2，无新增空头。机制门CHAIN3−ENSEMBLE3两seed>0且均值>=.1pp；结构拆分门CHAIN3−STRUCT_ZERO两seed>0均值>=.2pp。CHAIN3−DIRECT1和样本读出完整报告，不替代同调用控制。全15组成对差、四条件/单头/vCDR/共同有限HD95 ASSD及空头/杯越盘事件；3类包含本身不证明准确。训练CE下降不替代医学效果。若主门或机制门失败，停止本固定混合路径配置，不搜步数/调度/噪声/宽度/CE权重/种子挽救。

## 预算与保护

GPU1；首次实际运行含合成/读取/训练/评价/退出最多2h，提前120秒停，硬截止min(7200秒,1792815004)，kill余量5秒。campaign远端软28GB硬29GB、额外1GB本地/控制总30GB。本轮预计<50MB，旧权重只读不复制。独立guard/native timeout/有界NFS probe。合成一次检查q0合法、核概率守恒/端点、固定RNG可重现、基础梯度冻结与可训练梯度；不是科学证据。每25步保存网/optimizer/CPU/CUDA/sampler RNG但入口禁止已有runs自动重放；故障不补已完成臂，重试原时钟。完成仅收小JSON、不收权重图像，公开匿名聚合/协议/源码/去标识回执。
