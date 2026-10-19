# Q1 执行审计

- 执行源：7b1b7509e5add5dda86fbe3b1328ef715e01f060；源码冻结后无GPU重试/改配置/砍控制/放宽门槛。
- GPU_COMPLETE：1520预测、560科学arrival；14轨迹，静态400+core480+native640读出。
- MECHANISM：240/240科学规范更新匹配、zero0；额外40更新；82仿射张量19136坐标；源anchor不可变、RESET返回mask差0。
- 探针：每个core/额外到达含前/后/源各4前向，总core前向4200（含原图源/候选），真实native前向4160/反向480。core真实损失反向280；probe无反向、不参与提交，额外source anchor梯度反向0。
- FULL_REPEATABILITY PASS40到达/80对logits+末model_state精确，范围仅ENT_NORM RESET正序同进程；不认证全Adam/RNG/其它策略/跨进程。
- QUALIFICATION PASS，首2到达均时.8595724106s，保守投影714.7669039726s≤1180。原cap GPU1200/CPU900/输出2GiB保持。
- ACCESS：8登记RGB，适应阶段source/eval标签读取0、旧评分读取0、未来图用于更新false、新源训练false；拒绝上一轮根目录及评分边界。scorer只在GPU退役后读取8标签。
- SCORE_COMPLETE：1520读出、878个唯一指标计算（仅CPU缓存去重，不少读出）、各状态160风险头事件、19绝对聚合行；bootstrap各2000有效。
- 退出：worker exit0 inactive，score exit0 inactive，最终ledger all_workers_retired true/no_retry true；现场按原controller身份确认不存在。公开retirement删除PID/命令行/私有路径。
- CPU：早期源6项PASS完整5.6825993061s，增强控制/空区停止后最终源6项PASS5.0774729252s，均派发前。初次本机环境缺numpy在导入阶段失败，无数据/CUDA，未计时；py_compile/准备/阅读编排未计时。
- 成本：Q1完整GPU 573.6666119098663s，完整CPU评分126.5166585445404s，CPU准备10.760072231292725s。累计GPU3530.1142151355157s；CPU按旧ledger基数+准备+评分1652.9929370585148s，另既有本地body约.631s（约1653.623937058515s）。旧基数比V6对账高.0084569158s，继续披露，不改原冻结账本；其它研究成本不混入。
- 输出132448727B/峰值allocated1523415552B，无q8概率；无图像/标签/模型/逐例预测评分公开。
- 科学门槛：primary/utility_correlation/harmful_discrimination false；both_heads/matching true；advance_to_new_controller_protocol false。不是工程停止；未测试在线控制器或建立持续收益。
- SOURCE_EVAL冻结BN，native CW为current-image BN，比较必须披露语义混杂。八重复开发内容不是独立患者证据。

RAW receipts和匿名聚合保留原值；START_RECEIPT是派发时历史快照，CPU_CHECK_INITIAL是派发前初版，不当当前状态或最终运行源。本次结论来自完成receipts与RESULTS。

## 次要诊断偏差

离线配额排序基于按内容分组展开的列表，稳定排序打破切点并列时实际按case→order→block，而不是协议全流到达顺序。4行RESET（OD负pre源漂移；OC正pre分歧/负pre分歧/负pre源漂移）受影响；原始CSV/tie标记保留，不作选择效果解释，未重评分或改冻结源码。FRESH/RETAIN无切点并列，主Spearman/所有AUROC采用平均秩与并列0.5，与此遍历顺序无关。见OFFLINE_QUOTA_SCOPE。该偏差不恢复已失败的主门槛，但必须披露，不称全部诊断均严格符合协议。
