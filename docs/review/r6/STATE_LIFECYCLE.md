# R6 状态生命周期

IDLE → WEAK（原 C 六次，其中首个原图 logits 只读捕获）→ STRONG_LOSS（第七次、实际 criterion q）→ UPDATE（实际 logit hook 已触发一次，BN backward 已完成）→ POST（Adam 后原图第八次）→ COMMIT（验证物理计数、Adam、frozen source 和有限状态，原 C 已提交 augmentation RNG）→ EVALUATION_RELEASE。

只有当前 RGB 进入 host.step。visit 是本地递增整数，不接受 ID、域、subset 或 GT。criterion 保留完整软 q；C 调用原标量 BCEWithLogitsLoss。其余臂只有同一 BCE 的正权重，不执行拒绝、回滚、reset 或额外求导。即使梯度为零仍调用 Adam，已有 moments 可产生非零位移。

每次返回原生 post logits。take_evaluation 将当前 pre/q/post payload 移出，host 回到 IDLE；evaluator 此后读取 mask，评价结束在 finally 中清空 payload。未移出 payload 前不能下一步；移出后不保存历史图、概率或特征。每次 logit hook 随当前图移除，临时权重/残差只在 criterion 中存在；日志只有标量及权重 checksum。任何硬失败令 host 永久 FAILED，清空当前 payload，不能原地续跑。

现场 model forward、BN gradient hook、Adam post-hook 计数为计费来源。smoke 每个已构造 host 在 finally 中只计一次 live 累计数，清理 host/core hooks。失败记录标明可观测下界，构造或未触发 hook 的工作未知；不以完成访问数乘固定配额替代。首失败与原异常保留，无自动 retry。
