# 空间照明校正真实图像资格实验 A01

运行前冻结。矩运输A01/A02失败后换路线；取消原A03合成因子诊断。本轮使用已登记真实RGB和原冻结分割模型，所有科学结果来自真实图像，不新增数据、模型或源训练。

## 依据与定位

[IPOL Multiscale Retinex](https://www.ipol.im/pub/art/2014/107/)的算法2提供强度通道多尺度局部照明校正，算法3提供对称边界Gaussian卷积。已读取低分辨率完整论文；此前高分辨率下载不完整，未用于证据。

[SGRIF](https://arxiv.org/html/1805.06625)提供眼底成像退化与视杯任务的关联依据；它是另一算法，使用结构迁移/边缘保持与去雾模型，本轮不复现它，也不将其结果当MSRCP有效证据。这里检验一个尚未证明的假设：当前RGB空间照明场能提供有益输入校正。当前旧ColorSharp是全局颜色/锐度策略学习，本轮是固定空间场直接读出，不用旧B→S块选择器，不恢复旧失败配置。

本轮是已有算法组件的静态资格测试，不是原创CTTA方法或临床证据。若无真实任务收益，停止此固定配置，不以图像观感/熵/对比度改为成功；若通过，才另论证因果持续状态和跨数据资格。

## 资产与固定映射

仍原8登记内容×4已登记外观32输入，原seed20261018和原checkpoint；不读取旧192内容、其它数据/权重，不按类别/病理选择输入。与旧基线完全相同PIL RGB bicubic512、float32/255、四外观identity/gamma.7/通道乘(1.15,1,.85)/7×7 sigma1 Gaussian，再全图min-max。SOURCE logits>=0必须逐位匹配既有迁移SOURCE；不冒称旧服务器原sealed bit parity。模型eval、全部参数冻结、BN缓冲逐位不变，无优化器/反向/BN更新。

候选MSRCP-derived使用float64 RGB x；强度i=mean_RGB(x)，p=max(i,1/255)。s=[15,80,250]像素固定，等权log差r=log(p)-mean_s log(max(G_s*p,1/255))。G_s以DCT-II正交规范、频率πk/H和πl/W、乘exp(-s²ω²/2)并逆变换实现half-sample对称边界。固定1%/99%线性插值分位点裁剪并映到[0,1]为i1；gain=i1/p；RGB同乘min(gain,1/max_c x_c)，黑点分母下限1e-12。末尾按宿主全图min-max并转float32。分位跨度或最终跨度<=1e-12时该映射恒等回退；非finite硬停。float输入、1/255地板、分位插值及宿主最终归一化是本轮明确的迁移约定，不冒充IPOL原二进制逐字复现。RGB共同乘增益保比例只成立在末尾全局仿射归一化之前。

五臂全部预定：SOURCE、MSRCP、GLOBAL、SHIFT、B。GLOBAL对原强度i直接做同一1%/99%平衡，并用同一gain/gamut/宿主归一化，不使用空间surround。SHIFT使用MSRCP未截断gain宽轴循环257像素移位，再用当前像素gamut限制；未截断gain集合相同，最终剂量可能因限幅不同，不称严格匹配。B为原TOPO_SMOOTH静态强基线。SOURCE与三变换每输入4前向、128/worker、两worker256前向。新预测使用两独立sigmoid头，阈值0logit，不强行互斥、不做GT辅助阈值/融合/拓扑修复。

## 预定主指标、控制与停止

独立评分在两个worker OS0且身份退休、全32输入/logits/masks跨进程exact replay后执行；生成器deny GT、旧评价数据、网络和其它资产。评分仅8登记GT、nearest512、OD<255与OC==0，无GT反馈。两头Dice/ASSD/HD95、vCDR绝对误差、空mask/嵌套及最差逐输入头损害全部报告。比较采用原[2,1,1,1]/5条件权重、8内容等权、内容绑定bootstrap2000/seed20261018；重复开发内容不等于独立患者，区间不作确证性统计证据。

主推进合取：MSRCP−B OC>=0.30pp且配对95%区间下界>0；对SOURCE、GLOBAL、SHIFT的OC差下界均>0；OD−B区间下界>=−0.10pp；任一输入/头较B下降不超过1pp；不新增空头或增加嵌套违规像素；所有表面/vCDR指标32输入均定义；OC ASSD/HD95逐输入比B之比的内容均值bootstrap上界<=1.05；vCDR误差增量区间上界<=.005；MSRCP与SHIFT实际输入像素差>1e-6比例>=.5。任何一门失败不推进此配置，不看子组改门、不扫sigma/分位/场尺度，不继续合成诊断找解释。

与旧撤回块不同，本轮产生新完整掩码，旧块oracle空间门不适用；直接有限完整对照即可证伪。这不是将旧空间不足门调低或改写旧实验。

## 执行与资源

先最小算子断言（常量Gaussian、finite/范围、RGB共乘比、退化回退），它是工程检查，不单独报告科学合成成绩。只my-gpu指定GPU1 UUID，新root基线0、预计768MiB、预留1280MiB、hard输出1GiB；总3600秒原clock、独立guard3540秒、stage截止3530秒，首次正式输入投影余量检查不通过即停。父子中性入口、ps及GPU自身nvidia核；NFS容量/探针先验；不干预他人进程。<=24小时、每服务器新增<50GB及项目20GB/18GB/aux1GB继续。两worker共享实现、重演只证明确定性，不冒独立算法验证；controller父OS码未知保持null。失败也整理公开聚合与自有源码代理push，无真实样本/权重/私有路径外发。
