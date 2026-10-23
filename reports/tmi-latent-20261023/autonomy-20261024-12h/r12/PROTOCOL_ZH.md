# r12：预测初始化下的联合轮廓生成与同预算直接预测

运行前冻结。r11仅真值表示资格通过，r10像素混合路径失败；本轮改变医学状态为联合130维，非相对r10单变量，禁止复活r10调度搜索。源r01冻结网给出影像特征及预测初始化，真值仅训练监督和最终评分，决不部署真值中心。

## 表示和异常

64射线/.25间距/双线性.5第一出界，周期线性解码，与r11保持。中心预测cup质心，若不在OD和cup交集，则取交集EDT最大点；交集空则取OD EDT最大点，OD空则(127.5,127.5)。每异常计fallback，非正常模板，若最终dev任何fallback则主门失败。空/不含中心的头半径取.25，全部半径保护[.25,hypot(256,256)]；杯半径不超过盘。目标表示若需要fallback则停止科学运行不补样。中心编码logit(c/255)裁到[1/512,511/512]；OD log(r/64)，杯盘比logit裁[1e-4,1-1e-4]。解码同区间保护，记录裁剪坐标数。r11不自动证明这种变换，合成检验往返。

## 配对训练

固定c4与RGB各8×8池化，拼接anchor130/current130/t1，共1477输入，MLP512/GELU/256/GELU/130末层零初始化。DIRECT/FLOW同初始化、同source批、同增强、同Adam.001/batch4/clip5，每seed各臂1000更新，seed10225/10226，总4000；sampler seed+90000，无best。纯坐标MSE，不加Dice/SDF改理想速度匹配，不称医学loss创新。目标来自源真实形态的中心和半径。z0=预测anchor+.1Gaussian；DIRECT输入z0,t0预测target-anchor，读出anchor+输出；FLOW输入zt=(1-t)z0+t*target,t~Uniform学习target-z0。网络共享全空间条件和全部坐标，不逐像素独立。原基础网全冻结，只读旧权重。79fit/20重复dev固定末步/target0。

## 固定读出

ZERO/STRUCT_ZERO/DIRECT1/DIRECT3/FLOW1/FLOW3。每图固定Generator(seed+100000+condition*1000+content)采3个起点，DIRECT3平均三次坐标预测后解码；FLOW3从第一个起点做3步Euler，t0,1/3,2/3，dt1/3；FLOW1为同首次速度一步直接走到终点。各主要臂3调用，总每图6调用；两单步共享相应首次前向，不另调用。STRUCT_ZERO解码anchor。模型测试全冻结、无持续状态、不做标签择样。

主门FLOW3-ZERO两seedDice>0均值>=.2pp，每seedOD OC>=0、clean>=-.3、内容q05>=-2，无新增空头、dev fallback为0。机制门FLOW3-DIRECT3两seed>0均值>=.1pp；结构拆分FLOW3-STRUCT_ZERO两seed>0均值>=.2pp。全部15组差/四条件/单头/vCDR/共同有限边界/空头杯越盘/裁剪/fallback均报；FLOW1比较仅辅助。失败停止此配置，不搜射线/hidden/噪声/步数/loss权重/seed挽救。三步有限精度、真值一次标注、反复dev、患者UNKNOWN、源预算和r01 RNG限制保留；不称CTTA/后验校准/超论文。

## 执行预算

GPU1，首次合成/加载/训练/评价/退出≤2h且deadline<=1792815004，提前120秒停/kill余量5秒，原时钟不重置。campaign累计远端软28GB硬29GB+本地小控制1GB合计30GB，本轮预计<50MB。独立guard/native timeout/有界storage probe；每25步保存双方网/optimizer/CPU CUDA sampler RNG，入口拒绝runs重放，无精确恢复不重跑完成部分。合成仅工程并计入首时钟，源模型冻结及读出权重不变核验。结束仅收小JSON不收模型图像，按全局公开源码协议匿名聚合去标识回执。
