# 固定视野原文件观测配对推理（首次运行前冻结）

唯一主问题：在既有8内容、原模型、identity、固定同物理ROI下，原800文件N是否较512重建R提升全图OC Dice。L直接低分辨率裁剪、SOURCE全图源输出、既有静态TOPO_SMOOTH B为强控制。没有训练、梯度、selector、Q、预算/窗口/层搜索；不是新CTTA方法性能宣称。

数据与输入沿用已交付CPU资格：仅8登记RGB；由identity封存SOURCE OD bbox中心生成256窗，clip[0,256]后最近16对齐，tie向上，原文件400窗位置乘25/16。N=原800裁400→512，R=旧PIL bicubic512重建800裁400→512，L=旧512裁256→512；bilinear align_corners=False。共同旧512全图lo/span，随后三路**显式共同clamp[0,1]**，记录裁元素，不分别归一化。SOURCE为既有512全图identity归一化。

三crop输出logits bilinear降256，各自阈值>=0替换SOURCE的两个独立头ROI；ROI外逐位保持SOURCE。不softmax，不做OC嵌套修复、不融合B。B只作独立控制。完整全图指标计入ROI可能带来的损害，不只报告局部变化。

运行前CPU合成25窗几何/常数/边缘以及拼接support与阈值检查。模型生成8×4×2遍=64完整encoder；逐次验证seg_head(feature)==logits，源位图两遍共16次核封存SOURCE，32输出logits及32features同进程exact replay，参数/缓冲全程不变、无梯度、0更新。原文件像素每图加载一次，重演使用同输入，不能冒充独立重新解码重演。只保存第一遍40个最终位图与私有定位/裁剪计数，不保存features/logits。

worker禁GT、utility、旧scores/scorer、网络；只可读8RGB、8SOURCE位图、原权重与既有代码/环境。worker exit0且精确身份退休后，scorer读原8GT并核既有登记摘要，另读8静态B位图；评估40mask。GT使用既有nearest512和OD<255/OC==0定义，不反馈到生成。

唯一主统计N−R全图OC Dice百分点：8内容等权，固定seed20261018 bootstrap10000、percentile95%CI，下界>0为主门。N−L、N−SOURCE、N−B同法，全部下界>0才资格通过；任一未过即不推进这个固定方案，不挑控制或改主。所有5臂OD/OC Dice、分头ASSD/HD95、vCDR绝对误差、空mask/嵌套违例及4个N对比全部报告。NA保留完整8分母，若不足8则CI为null，不因剔除困难内容改善门。无多重比较后选。8重复开发内容/患者与预训练未知，不作为独立患者或临床泛化证据。

独立总预算600秒，从首次CPU check起UTC/boottime冻结，guard540秒/输出32MiB硬停止门，controller剩余530秒约束；修复共用原时钟。root基线0、64MiB预留，预计<32MiB；不复制图像/权重。仅my-gpu物理GPU1指定UUID，启动显存free≥8192MiB，无则WAITING_GPU不启动计时；主子argv中性，自身nvidia/UUID检查。独立guard不依赖SSH/ctta。每唤醒只这一有限独立轮，结果后分析并交付，后继新实验留后续唤醒。
