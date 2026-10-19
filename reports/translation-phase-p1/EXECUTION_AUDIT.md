# P1执行审计

## 授权及冻结

用户明确不做Calibration Maps、停止BN尺度，允许在已读机制上改造、有依据小轮与GitHub交付，不新增源训练/RDC/ENGRAM。P1只隔离HRAC几何对应启发的四个冻结网络平移相位读出，不做原翻转KL、蒸馏、记忆、参数或BN统计更新，不宣称原论文复现/CTTA/TMI。

初冻结c85b60bb35115f0ef133bf6d0345882a3b3bc7b7；CPU测试中一个“错位必改变hard mask”的假设因阈值tie不成立。源码算法/科学协议未改，仅改为验证连续预测场错位，当前GPU执行source **7325b2bb08cff41f18bbc4595e220381fd64c269**。原CPU_CHECK_FAILED_TEST_ASSUMPTION、失败log、code_initial_cpu及source_initial_cpu.tar保留，不覆盖失败。初CPU3.6263699531555176秒，正式5PASS2.2224881649017334秒（body .027三位打印精度），总5.848858118057251秒，均无真实RGB/标签/checkpoint/CUDA。

CPU验证点模型下精确平移反移、step采样模型非零相位响应、边缘回退/无roll插值/输入不变、双独立头/非finite规则、固定配置与静态评分形状。它们不证明当前checkpoint有任务收益。

## 资源准入与执行

NAS实际NFS挂载容量>4GiB、小创建写读通过；新root/中性入口独立，不覆盖历史。GPU6原UUID核对，启动free23858MiB≥8192MiB。唯一9秒短检主子ps与nvidia GPU条目中性，8输入64输出/GENERATING，预算投影106.49047392606735秒PASS；worker首四相位后另核自身GPU可见项/UUID。无其它聊天进程干预。

本轮首2完整四查询束均时.5189722776412964秒，投影包括剩余62束及全部32输入重演。按当时已冻结GPU1200/CPU900/输出2GiB完成，无GPU或评分重试，没有削臂/减数据/改平移/权重/阈值。用户随后取消未来轮次时间/GPU配额，未追溯改变本轮协议或gate。新增空间20GB约束管理独立，不删除旧证据。

## 信息隔离与覆盖

固定原预处理后reflect向右/下整数平移，不再normalize，不roll/缩放/插值。所有干预只在H−1×W−1共同域安装，最后一行/列回退源。reflect影响可能经感受野进入内域，未宣称整体无边界效应。四视图同一当前图，无新增图或跨样本支持。

GPU只登记8RGB/已有checkpoint，拒本轮scorer、上一B1整个root和旧评分；source参数/所有buffers精确不变、无grad、optimizer/backward0。256唯一bit预测、32输入全四相位encoder重推，全部256最终数组/诊断exact，256forward；严格确定性无回退，限同进程，不承诺跨进程。logits仅RAM保重演，不落概率/features。

GPU worker exit0 inactive退役后无CUDA CPU才读取8标签；GT前景/边界只评分，不参与相位/权重定义。8绝对、64条件头、64区域、34对比/256metric覆盖完整，C+H=D核过。worker/scorer exit0且/proc不存在，controller原身份退休，finalledger all_workers_retired/no_retry。

bootstrap绑定8内容的条件/通道，640静态逻辑读出含重复，不是独立患者/CTTA流。SOURCE_SMOOTH是预先对照，正结果不改写P1主比较或主门槛。所有控制完整报告，无按结果挑子组、去掉有害视图或换概率/logit均值作主。

## 成本

|过程|完整进程秒|主体秒|
|---|---:|---:|
|初CPU假设失败|3.6263699531555176|初log保留|
|修测试后的正式CPU|2.2224881649017334|.027（打印精度）|
|唯一GPU worker|21.197056531906128|18.770238876342773|
|唯一CPU score|44.84848475456238|42.781972885131836|

输出17369663B（不含当前/初CPU源码目录及archive），torch peak464847872B。累计GPU5239.803265571536秒、已计CPU2522.6499504748053秒；旧单列body与CPU基数.0084569158差、未计时阅读导入/打包编排等保持，不清零不混其它研究成本。不将共享源查询/重演总时间作公平方法速度排名。

## 交付

一次必要覆盖/隐私检查，只发布自有代码配置/冻结协议/报告、匿名聚合与无路径回执；禁止逐内容身份、RGB标签权重features概率逐例评分、私人路径和第三方全文。全部GitHub经有效代理、NO_PROXY不绕GitHub，核账号/branch/remote、push远程SHA及匿名HTTP200；最终提交及访问回执为交付依据。
