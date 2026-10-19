# E2-A 独立CPU聚合修复：先冻结，后评分

初始GPU预测完整96/32、全32输入重推logit精确、参数/buffer不变、无backward/optimizer，worker exit0。首CPU评分exit1，原因是复用了paired_state.summary只接受(8,5,2,2)/(8,2,2)的持续流接口；本轮合法静态区域与单头差是(8,5)，ALL是(8,5,2)，返回是(8,2)，被错误拒绝。初CPU耗24.781968355178833秒，初RESOURCE_LEDGER与失败log保留不改，首轮score不冒称PASS，也不重跑GPU或更改任何预测。

用户已授权完成分析交付，工程修复限于使既定静态数组能进入相同cluster bootstrap。另冻结本修复source：仅本模块summary形状改为以上3种，调用原同一interval函数、同seed、同2000次8内容抽样/绑定条件通道、同均值/最差/负单元定义；不修改旧paired_state或原冻结实验代码。新增1项CPU回归检查静态形状/重复相同顺序不改变区间，原5测试保留。不看标签调阈值、融合权重、区域或主比较。

本次是单独CPU结果重建，不是重试原GPU资格。原协议的失败停止/无重试记录保留；在另立scoring_repair目录，只以symlink读取同一96个已封存bit预测、原manifest和8标签，重建结果不覆盖初失败private评分文件/源码/tar/ledger。新的CPU_CHECK/SCORING_REPAIR_LEDGER、过程身份与成本单列；初score与修复测试、重建CPU合计不超过原900秒上限，无GPU或模型前向。任何修复失败停止，不再无限重建。

公开最终报告须明确“初评分失败；独立CPU接口修复后结果重建”，不将原controller/score exit1改成0。科学原比较、统计规则、预测集合和静态范围全部保持；结果重建完成后核3绝对/24区域/24条件头/8对比、修复scorer exit0/inactive及修复controller退休、完整成本。没有新的参数实验或部署成功结论。
