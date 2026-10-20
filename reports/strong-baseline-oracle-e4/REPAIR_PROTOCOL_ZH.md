# E4 独立工程修复R1

初版source7f8b884758d48a7a54a8cb326bdc66157c9d5129执行失败：配置口径清单错误假设CPU-only P2存在checkpoint_path，KeyError停止。完整CPU进程5.476994752883911s，初准备.8347280025482178s，均保留；无公开RESULTS、未变更预测/模型。另有部署字符串替换导致的远程解析错误，发生执行前，保留说明，不伪造耗时或算法失败。

修复只沿P2明确input_root读取P1注册checkpoint_path，与E3配置比较；不重新加载权重，也不把路径相同当作逐字节验证。新增CPU-only血缘合成检查，旧oracle/C/H/候选/权重/门槛完全保留。独立修复root与源码，初root/log/code/ledger不覆盖；正式计算和独立进程重演完整执行，原失败不改称PASS。私有逐输入矩阵仍不公开。
