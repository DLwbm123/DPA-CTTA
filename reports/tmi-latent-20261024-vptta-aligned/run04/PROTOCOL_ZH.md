# run04 图像-边界条件能量

源依据：diag02重建梯度不稳定对齐分割；run03源监督梯度校正未优于直接控制且损伤clean，未测试目标。本轮改变无标签优化目标，不做旧潜变量权重/步长网格。

冻结run01基础分割模型，训练独立小能量E(image, softOD/OC)。输入统一缩至128，5通道CNN：conv5 stride2 32/GN8/GELU，conv3 stride2 64/GN8/GELU，conv3 64/GN8/GELU，conv1标量图均值。源正样本为GT SDF经sigmoid(-16d)，负样本为同图GT SDF各头偏移±[.03,.12]或平移±[3,12]像素（随机符号），柔化相同，避免把硬/软边界差当正确性。图像外观变化时正确边界保持不变。

固定种子92504、按源CSV随机80fit/20dev，基模型已见全部源，仅能量模型留出；每域1500步batch4 AdamW lr.0003 wd.0001，损失softplus(1+Epos-Eneg)+.001(Epos²+Eneg²)。clean和随机gamma/gain/shade组成同批。固定末步、无搜索。

测试只更新当前预测logits，不更新网络：三步能量梯度，每通道RMS归一后步长.25，偏离初始logits最多1；objective=E+.01*mean((z-z0)²)，每步允许1/.5两档回溯，下降才接受。不使用标签，无持续状态；只检验图像条件边界能量是否能作为适应目标，不宣称持续性贡献。

五源训练完成后源dev clean/fixedappearance(gamma.7/gain1.15,.9,1.05)/blur9比较ZERO与E。机制负控制E_mismatch把能量条件图像换成同源下一dev图，基模型初始预测与标签仍本图；仅源dev使用该控制，不是部署方法。另固定正/负边界排序检查。源gate：每源正确边界排序率≥75%；扰动均值较ZERO≥.2pp、较mismatch≥.05pp；每源扰动增益≥0；clean均值≥−.1pp；扰动差值p10≥−1pp。任何未过不测试目标。通过才五源E一次VPTTA完整流，ZERO复用run01，禁止目标选参。

共享起点4小时硬限，300秒预停/60秒KILL；2GBsoft3GBhard预计<500MB，只读复用数据和base，不删除旧证据。所有阴性完整报告；源dev反复开发及base源暴露如实披露，不称独立测试。
