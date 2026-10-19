# R84：第二强视图来源最小对照

状态：**v2 FORMAL RUNNING**。修复后的fresh资格8/8物理更新PASS，两个正式worker已各32次更新，合64/1152次；尚无科学评分结果。执行科学源码`e89a31c`；冻结协议及明确工程修正见[PROTOCOL.md](PROTOCOL.md)。

唯一候选为VERTEX：第一strong不变，第二由同strong的flip改为NEXT既定seed的既有strong的flip，逆对齐后完整原W/0.1LSO loss各一半；当前q/完整当前W、BN-only19136、continuous Adam、192内容/2顺序/3seed不变。仅新增6条轨迹、1152更新/2304 PREK预测，原NATIVE/VERTEX端点评分只读复用；不再重跑NATIVE/BARY或q候选。当前增强替换证据只支持提出假设，并未证明性能或梯度variance改善。

修复前v1资格collapsed首arrival失败，正式0，2次attempt=complete/19F，GPU进程15.106729秒、CPU准备5.300599秒，失败现场与费用保留。[FAILURE_AUDIT.json](attempt1-failure/FAILURE_AUDIT.json)记录原因：第二strong遗漏native原本就有的normalize_image_to_0_1。v2另root/entry/空间预留，复用原归一化函数（不是新模块），候选与独立oracle均恢复该步骤。没有绕过失败检查或将其解释为科学性能结果。

v2 fresh资格：8次更新/76F，collapsed第二输入与第一实际strong exact，collapsed对原R79 VERTEX核心完整state/output exact，真实NEXT候选对独立原公式output/gradient/BN/Adam/buffer/RNG核心exact；PRE完整恢复、q/W当前来源、BN冻结/连续Adam/一步一次backward和Adam/物理9F均通过。这是工程资格，不能替代科学得分。

v2 CPU准备4.299914秒，资格GPU进程25.151868秒；旧失败GPU和CPU费用另计，终态将报告跨attempt总量，主体时间不双算。规划projection1061.966368秒仅规划。正式start epoch 1792431435.344691，deadline/GPUcap均null；两worker实际RUNNING、中性argv和GPU0/1固定UUID绑定已核验。

成功门槛仍对原VERTEX：macro≥+0.1pp且95%内容CI下界>0，D1 OC≥−0.2pp，并通过对NATIVE原方案的域/顺序伤害线；不改阈值。全部GPU实际退休后才CPU读labels和旧score，保留所有负单元/缺失和开发集限制，不保证性能改善。

窗口空间：R82/R83/R84-v1失败/R84-v2各2.2GB预留，加本地1GB、余量1GB共10.8GB<20GB。失败空间不删除或遗漏。继续每半小时监测，结束后完整匿名结果和中文报告proxy push并核验；启动交付不代表正式结果交付。
