# R83：固定一次OD半反馈折中

## 授权、动机与冻结

沿用用户2026-10-19的12小时自主性能优化授权，至2026-10-20 11:33北京时间；窗口内无需再请求许可，无GPU/时间上限，但所有诊断、资格、失败、正式运行和CPU准备/评分成本照实计账。新增空间总量≤20,000,000,000字节。旧R75–R82已交付退休，不原地改写或重跑。

R82硬固定OD q并保留当前OC q，VERTEX相对原方案macro下降0.335251pp、D1 OC下降1.516356pp；D3 OD/OC改善0.319132/0.246793pp。共享BN/连续Adam导致跨通道轨迹耦合，与结果相容但未唯一识别。只检验一个在R82启动时已经提出的折中，不扫描系数：qOD=(初态六弱OD+当前六弱OD)/2，qOC=当前六弱OC，W=原完整当前六弱W。数值实现先逐弱视图OD概率相加乘0.5再六弱平均；独立公式oracle匹配该浮点运算次序。初态和当前完全相同的首步应与原方案exact。

## 保持不变

192个既有开发内容、既定2顺序、3种子20260907/17011/29009；BN-only19136、current-stat无running-stat、非BN和head冻结；连续Adam1e-4、betas .9/.999、eps1e-8；原六弱、W(.1,2)方差/供体边界/归一化、BCE及0.1 orientation LSO公式不变。BCE和LSO用同一half q；W来自本格当前六弱，不能将边界改随half q。NATIVE/VERTEX/BARY原完整损失定义；原strong与确定水平flip逆对齐、分开batch1；SeedSequence([既定seed,原内容ordinal,79])增强跨方法/顺序/旧对照相同。无新数据、seed、模块、教师、EMA、输出平均或FULL。

3方法×3seed×2order×192=3456新增物理更新、18独立连续轨迹，PRE/K=6912预测。全部旧R82评分5格只读复用，包括原R81四格和R82硬SPLIT，总90旧轨迹/34560预测，不是新重复验证。HALF评分index5，HARD_SPLIT index4，ORIGINAL index3；job字符串01仅是实现标识，不是旧q0/W1。

## 资格与准入

CPU数值/通道/计数/评分分解自检先通过，再fresh GPU资格36物理更新：每方法两个对角与旧R80对应host各2arrival×2host=8；HALF对独立原公式2arrival×2host=4，共12/方法。要求完整state/output对角exact，HALF独立输出/梯度/BN/Adam/buffer/RNG核心state exact；q来源OD_HALF_OC1与Wcurrent exact；初态首步与对角exact；PRE/初态六弱observer完整恢复；连续Adam age、BN冻结、一步一次backward/Adam、物理F计数均满足。半混合与硬拆分物理调用相同，NATIVE每arrival含PRE15F、其余16F；正式54144F，资格438F，总3492更新/54582F。失败保留证据与费用，明确工程bug可另attempt修复，不能将性能负结果当bug。

GPU仅注册0/1 UUID，启动≥10GiB空闲，allocator8GiB，不动他人进程。先将新轮2.2GB加入同一窗口reservation；保留既有R82 2.2GB、本地1GB、全局余量1GB，合计6.4GB，小于20GB。每轮输出2GiB包含64MiB元数据余量；不复制旧数据、模型、环境。CPU/GPU进程无预算截止；测得projection仅规划，不用于旧资源门槛。全部新GPU实际exit0/GPU_COMPLETE后才允许CPU读取labels和旧scorer。进程argv保持中性。全部费用记录，主体时间不可与进程时间重复计算。

## 评分与预设成功门槛

逐内容配对H=PRE−P0、C=K−PRE、T=H+C。报告HALF−ORIGINAL、HALF−HARD_SPLIT、HALF−全初态q/currentW、HALF对NATIVEoriginal及HALF方法间结果。所有OD/OC、各域、seed/order、FP/FN整图像素比例、前景、内容median/正负覆盖、真实较早较晚位置都保留。

主要候选固定为VERTEX_HALF，性能门槛沿用R82的原尺度而非放松：macro对VERTEXoriginal≥+0.1pp且95%内容bootstrap下界>0；D1 OC对VERTEXoriginal≥−0.2pp；任域×通道对NATIVEoriginal均值≥−0.5pp，任域×通道×order跨seed均值≥−2pp。仅比失败硬拆分改善不算超过原方案。所有方法结果保留，不挑checkpoint，不改阈值。

2000次按域分层contentbootstrap六anchors耦合、无多重比较校正。已暴露开发集，不是患者独立或FULL验证；后续选择进一步增加开发适应偏差。连续BN/Adam内生分化，不能称同入站state纯梯度效应；不拆q在BCE/LSO或W子成分。折中效果不保证线性插值，也不保证性能改善。本轮之后不再扫描OD混合权重；若无合理新证据则转结果/代码分析，不能为保持运行空烧GPU。

## 完成交付与监测

每30分钟低成本监测，无重要变化安静；真实终态核18/18、3456attempt=complete、资格36另计、6912新预测、online labels/source0、实际退休与评分屏障、全部费用与新增空间。只下载匿名public，完整中文报告和全部负单元与源码推送对应公开GitHub分支，代理连接、核remoteSHA/匿名REPORT200并保存新local+remote DELIVERY。隐私数据、身份、mask、weights、raw private logs、机器路径不公开。12小时窗口内不因一轮交付机械暂停；窗口后不新开阶段，健康已运行任务可完成后交付再暂停。
