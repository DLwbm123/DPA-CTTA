# r13：源训练跨内容作用迁移
运行前冻结。独立源表示前置检验，非生成模型/CTTA，不加载任何旧权重，不复活冻结后纠错。借用EbC的support-query作用迁移，不声称复现或满足群可识别定理。

数据保持79fit/20重复dev/两seed10225、10226/256/target0，原整图minmax预处理保持，非blindspot。源每批8不同内容，原有随机水平垂直翻转，逐通道gamma=exp(U[-.2,.2])为整批共享，x'=x^gamma；无gain/shade/clamp/再量化。gamma在数学上保持各通道排序及像素坐标，取值0/1保持；不因此保证临床充分性，不声明有限群理论。固定四条件最终评价clean/gamma/color/blur，后两类不在训练作用族，不能掩盖其退化。

同标准Res34规模GN encoder+原c4/8/16投影和SDF decoder，删除renderer/relative_style，style改为全局平均e1->Linear64到4。分割只经content空间图；contrastive使用c4平均16维和style4维，共20维。三个臂同初始化、批次/增强/医学loss/AdamW lr.0003 wd.0001 clip5；每seed每臂2000更新，总12000。医学loss原BCE+Dice+.2边界SDF+.01包含，原图/变换图平均。加.05对称contrastive，原向变换/反向各query2，候选为该方向全部8个内容，目标为索引6、7，距离为20维平方距离均值/温度.1。没有原图自身作为同方向额外负样本。

前6内容support、后2 query。style矩阵R=solve(S^T S+.01I,S^T S'+.01I)，row-vector约定，ridge向I。content不变。SHARED正确support配对；SHUFFLED将support目标固定roll1的无自配对排列；INVARIANT用I。逆向另拟合，均计算矩阵，INVARIANT不使用拟合输出（故梯度算力略不同，不能称严格FLOP相同）。不裁矩阵、不选rank，记录奇异值/rank/迁移误差，非医学证据。岭解始终可算但不证明数据满秩，不按rank跳过/补episode。sampler seed+110000，gamma独立Generator seed+120000，所有臂共享。无best/超参搜索。

每seed三臂完成2000后固定末读各一次20x4，无测试权重更新/持续状态。主SHARED-INVARIANT两seedDice>0且均值>=.2pp，每seedOD/OC>=0 clean>=-.3 内容差q05>=-2，无新增空头/杯越盘>1%；配对机制SHARED-SHUFFLED两seed>0且均值>=.1pp。各臂clean Dice须>=75，否则标训练未合格，任何主线支持结论失败。完整3组成对差、四条件/单头/vCDR/共同有限边界/结构事件全报。原r01只是历史位置，非同预算主对照。失败不搜维度/ridge/gamma/温度/权重/步数/seed。

GPU1，从首次合成、载入、训练到评价退出最多2h且不晚于1792815004，提前120秒停止+5秒kill；独立guard/native timeout/有界storage probe。每100step保存3模型+Adam+所有RNG及累计计数，最新文件原子替换；预计峰值<3GB（6最终臂约1.6GB+单seed临时~.8GB），campaign软28GB硬29GB+1GB本地控制，累计30GB。不删历史、不复制旧权重。入口拒绝已有runs，无准确恢复不重放，失败保留首次时钟。合成2forward/1backward无optimizer单列，工程自检不算医学证据。预期训练编码192000图像、末评价480，合成另计；12000updates。完成正常退出退役后仅JSON聚合与公开源码/协议/匿名聚合/去标识回执。

重复dev/患者UNKNOWN/单源模拟扰动/额外源预算限制保持；不是独立患者、目标域或超过论文的证据。该试验检验一个源训练机制，不通过改名绕过旧停止族。
