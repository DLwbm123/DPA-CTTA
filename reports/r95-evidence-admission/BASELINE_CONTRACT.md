# 确认实验基线合同草案

状态：DRAFT_NOT_EXECUTABLE。以下是拟冻结的操作定义；具体版本、字段分类与资格尚未实施。不得据此启动。

## 1. 输出与学习合同

|臂|学习状态|当前内容操作及输出|跨内容状态|
|---|---|---|---|
|VV K|原VV，19136维BN affine、current-stat、无running-stat，非BN/head冻结，原continuous Adam|原六弱q/current whole W(.1,2)、原strong及同strong的flip逆对齐、逐view损失均值，真实一步更新后原图K|保留完整学习状态|
|E-VV K|同VV的网络/参数集合/损失/优化器超参|每内容学习前恢复完整S0学习状态，执行与VV同语义的一步更新，输出同一时点原图K|不保留跨内容学习状态；每内容局部Adam age从0到1|
|NATIVE K|**原适应基线，执行更新**；方法定义固定，不是参数冻结|原单strong分支损失与原更新流程，输出当步原图K；不得加VV额外flip分支改变NATIVE|按原方法连续保留状态，不擅自改为episodic|
|Q0|初态affine、current-stat，无optimizer更新|同固定六弱对齐logits→sigmoid→mean顺序，独立双头>=.5，内容一次；不跨图学习|无学习；重复anchor只是广播参考|
|P0|初态affine、current-stat，无optimizer更新|原图只读输出；若从Q0 identity视图导出，必须先资格exact并冻结实现|无学习|
|SOURCE_ONLY（可选辅助）|初态参数、stored-source running statistics，eval|无增强原图只读输出|无学习；不能与P0混称|

原VV损失为BCE+.1方向LSO、原BCE全局numel/LSO联合分母不变，Adam lr1e-4、betas(.9,.999)、eps1e-8；native normalize及所有实际增强保留。动态臂各自生成其当前q/W，不把VV缓存借给E-VV或NATIVE。初始checkpoint与库/环境版本锁定，strict model-only，不读source图像/额外模型/在线标签或未来图像。

Q0是已有强静态参照，不是穷尽搜索得到的最优静态模型；NATIVE是原适应方法参照，不是冻结源模型。不能因确认结果不利而删除Q0或NATIVE。

## 2. E-VV 完整恢复与不可回退部分

|类型|要求|
|---|---|
|参数/梯度|初态BN affine及应为初态的grad状态；核非BN/head不变|
|优化器|Adam完整m/v/step及状态是否未初始化的语义恢复，不只设梯度0；不在恢复参数后继续推进旧moments|
|学习缓存|GraTa/目标或权重缓存/weak或strong缓存/feature hooks等所有影响后续学习的状态恢复或清空到S0|
|外生流|内容arrival、真实全局visits、已发生读取、费用/前向/更新计数绝不回退|
|随机数|按预先锁定seed×canonical内容ordinal的原增强规则匹配；不因恢复S0而每图重放同一随机序列；相关native计数字段先区分学习年龄与外生访问语义|
|只读观测|PRE、Q0/P0导出前后fullsnapshot与下一步state/output必须exact，不改变主轨迹|

具体snapshot字段映射需在实现后、无标签资格阶段逐项审查，目前未验证，不标PASS。无法匹配的RNG或计数语义必须披露；不能以“同样更新数”掩盖。CPU/GPU费用是外部不可逆账。

## 3. 重复性与数值资格先于正式阶段

E-VV整体去除跨图学习状态，不单独识别Adam/qW/某层。它与旧R79逐图初态一步的概念有重合，不宣称新方法；未来需核方法版本、内容、输出时点、增强RNG、状态恢复、硬件与已有资产。若已有严格等价且可核结果，不重复运行；旧开发重跑不能变成新确认。

拟资格：VV/NATIVE对冻结原实现短序列fullcore/output exact；E-VV对独立S0单图参照exact、处理A→重置→B与独立B一致、local Adam age=1且globalvisits单调；Q0重复/identity对P0exact；只读观测和post不改下一步。物理计数含参考调用、资格、丢弃分支与失败。资格失败为NOT_EVALUABLE，保留现场，不放宽exact为Dice接近。
