# R76 启动状态

2026-10-18 UTC08:28左右，固定均值位移分解实验已后台启动。源码9175de6；CPU协方差／均值分解／完整导入检查PASS；真实模型原生输出、完整学习状态及读出后BN／Adam／RNG精确一致检查PASS。中性主／子进程命令和GPU显示已核验。

这是SCREEN启动，尚无科学结果。唯一候选为MEAN_PRESERVE；另外输出NATIVE、ORIGINAL_WITHIN、SHIFT_ONLY，原生连续学习轨迹保持相同。CPU评分必须在GPU成功退休后执行，并验证NATIVE和ORIGINAL_WITHIN逐mask精确重现R75。

继承R75累计355.588795185 GPU-process秒及25.273327827 CPU-process秒，本轮CPU准备新增4.381698132秒；总GPU上限14400秒和北京时间19:56:29截止不变。保持每半小时监测。效果门槛与预算均通过才进入同方法固定FULL；本次机制修正之后不自动开其他科学变体。
