# R82：通道拆分q反馈的性能与机制结果

状态：**COMPLETE，性能门槛NOT_PASSED**。18条新连续轨迹全部完成，3456次正式更新、36次资格更新，未发生失败重试。执行科学源码`36ca0c89208307fddb495275423654051a81b400`；协议见[PROTOCOL.md](PROTOCOL.md)。

**固定OD伪目标并保留当前OC伪目标，未保住D1收益。VERTEX整体比原方案下降0.335251pp，95%内容bootstrap区间[−0.457075,−0.222761]；D1 OC下降1.516356pp。其他域伤害门槛通过，但这不足以宣称性能改善。** 三方法对各自原方案的macro均下降。负结果保留，未调整门槛、挑选checkpoint或覆盖旧端点。

## 配对设计与边界

q=(初态六弱OD均值，本格当前六弱OC均值)，W仍完整取本格当前六弱，方差/边界/归一化与原W公式不变；BCE和0.1 orientation LSO使用同一split q。无需域标签或在线真值。19136个BN affine及原连续Adam保持；current-stat BN、无running-stat，原head和非BN冻结。原strong和确定水平翻转分开batch1，NATIVE/VERTEX/BARY完整损失定义不变。192既有开发内容、2既定顺序、3既定种子不变；内容绑定增强与R79–R81相同，没有新数据、模块、教师或输出平均。

新增3456更新/6912 PRE、K预测；R81四格72条旧轨迹/27648预测只读复用，不能视为本轮独立重复。SPLIT是新候选（评分index4）；旧01是全初态q/currentW，不混淆命名。H=PRE−P0，C=K−PRE，T=H+C；P0为零更新评分参考，不是真值教师。下列pp均为Dice百分点。

## 最终K总体得分

|arm|cell|macro_Dice_percent|OD|OC|imageweighted_Dice_percent|
|---|---|---|---|---|---|
|NATIVE|ORIGINAL|+75.838678|+84.191921|+67.485434|+75.941705|
|NATIVE|SPLIT|+75.728186|+84.075952|+67.380420|+75.848029|
|VERTEX|ORIGINAL|+76.820033|+84.991714|+68.648351|+77.128472|
|VERTEX|SPLIT|+76.484782|+84.702057|+68.267507|+76.757364|
|BARY|ORIGINAL|+75.961377|+83.846440|+68.076314|+76.124668|
|BARY|SPLIT|+75.777074|+83.749495|+67.804654|+75.904245|

同方法SPLIT−ORIGINAL的总体变化：

|contrast|macro_delta_pp|OD_delta_pp|OC_delta_pp|
|---|---|---|---|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|-0.110492|-0.115969|-0.105014|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|-0.335251|-0.289657|-0.380845|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|-0.184302|-0.096945|-0.271660|

## VERTEX各域：D1损失与D3缓解并存

|domain|channel|mean_delta_pp|median_delta_pp|positive_contents|negative_contents|
|---|---|---|---|---|---|
|D0|OD|-0.162901|-0.165296|1|21|
|D0|OC|-0.149529|-0.094973|3|19|
|D1|OD|-1.168479|-0.547567|1|56|
|D1|OC|-1.516356|-0.418320|5|52|
|D2|OD|-0.146379|-0.205643|11|46|
|D2|OC|-0.104287|-0.143828|22|35|
|D3|OD|+0.319132|+0.311305|45|11|
|D3|OC|+0.246793|+0.255465|53|3|

D1 OD有56/57内容下降，OC有52/57内容下降；D3 OD有45/56内容改善，OC有53/56内容改善。因此均值并非单个极端内容造成。预先固定门槛要求macro≥+0.1pp且区间下界>0、D1 OC≥−0.2pp；这两项失败。对NATIVE原方案的各域均值≥−0.5pp、域×顺序跨种子均值≥−2pp两项通过，不替代前两项。

## 历史与当前一步分解

|contrast|domain|channel|mean_delta_pp|
|---|---|---|---|
|VERTEX/HISTORY/SPLIT_MINUS_ORIGINAL|D0|OD|-0.161876|
|VERTEX/HISTORY/SPLIT_MINUS_ORIGINAL|D1|OD|-1.112626|
|VERTEX/HISTORY/SPLIT_MINUS_ORIGINAL|D1|OC|-1.459757|
|VERTEX/HISTORY/SPLIT_MINUS_ORIGINAL|D2|OD|-0.144550|
|VERTEX/HISTORY/SPLIT_MINUS_ORIGINAL|D3|OD|+0.320250|
|VERTEX/CURRENT/SPLIT_MINUS_ORIGINAL|D0|OD|-0.001025|
|VERTEX/CURRENT/SPLIT_MINUS_ORIGINAL|D1|OD|-0.055854|
|VERTEX/CURRENT/SPLIT_MINUS_ORIGINAL|D1|OC|-0.056599|
|VERTEX/CURRENT/SPLIT_MINUS_ORIGINAL|D2|OD|-0.001829|
|VERTEX/CURRENT/SPLIT_MINUS_ORIGINAL|D3|OD|-0.001118|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|D0|OD|-0.162901|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OD|-1.168479|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OC|-1.516356|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|D2|OD|-0.146379|
|VERTEX/TOTAL/SPLIT_MINUS_ORIGINAL|D3|OD|+0.319132|

直接证据是更改OD的q来源改变了整个连续学习轨迹，连未更改供体规则的OC也受影响。OC规则仍为本格当前预测，但共享BN参数和Adam在不同梯度历史下分化，当前OC供体本身也会分化；因此“保留OC当前q”不保证保留OC得分。大部分差异已在PRE的历史项出现，不能把最终变化都说成当前图像直接纠错。共享参数/优化器引起跨通道耦合是与代码和结果相容的解释；本轮没有分离共享BN层、Adam矩、q在BCE与LSO中的作用，不能断言唯一原因或单独梯度机制。

BARY与NATIVE的同方法域变化及SPLIT相对NATIVE原方案的伤害：

|contrast|domain|channel|mean_delta_pp|
|---|---|---|---|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|D0|OD|-0.256153|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OD|-0.363381|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OC|-0.356887|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|D2|OD|-0.164312|
|NATIVE/TOTAL/SPLIT_MINUS_ORIGINAL|D3|OD|+0.319969|
|VERTEX_SPLIT-NATIVE_ORIGINAL/TOTAL|D0|OD|-0.402037|
|VERTEX_SPLIT-NATIVE_ORIGINAL/TOTAL|D1|OD|+1.980772|
|VERTEX_SPLIT-NATIVE_ORIGINAL/TOTAL|D1|OC|+3.006351|
|VERTEX_SPLIT-NATIVE_ORIGINAL/TOTAL|D2|OD|-0.109926|
|VERTEX_SPLIT-NATIVE_ORIGINAL/TOTAL|D3|OD|+0.571736|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|D0|OD|+0.170631|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OD|-0.926525|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|D1|OC|-1.117361|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|D2|OD|+0.099641|
|BARY/TOTAL/SPLIT_MINUS_ORIGINAL|D3|OD|+0.268473|
|BARY_SPLIT-NATIVE_ORIGINAL/TOTAL|D0|OD|-0.647579|
|BARY_SPLIT-NATIVE_ORIGINAL/TOTAL|D1|OD|-0.353183|
|BARY_SPLIT-NATIVE_ORIGINAL/TOTAL|D1|OC|+0.877445|
|BARY_SPLIT-NATIVE_ORIGINAL/TOTAL|D2|OD|-0.684625|
|BARY_SPLIT-NATIVE_ORIGINAL/TOTAL|D3|OD|-0.084319|

不能由R81的BARY相对OD机制，推断所有方法或通道固定后都会改善。R82同时取消一条有益反馈路径；D1损失并不表示原W/LSO实现有bug。SPLIT−全初态q/currentW以及SPLIT方法间全部H/C/T列于[FOCAL_MECHANISM.csv](FOCAL_MECHANISM.csv)和[CONTRASTS.csv](CONTRASTS.csv)。

## 错误类型、位置与稳定性

|domain|channel|metric|mean_delta_pixel_pp|
|---|---|---|---|
|D0|OD|FN|+0.040499|
|D0|OD|FOREGROUND|-0.040369|
|D0|OD|FP|+0.000130|
|D1|OC|FN|+0.075764|
|D1|OC|FOREGROUND|-0.031647|
|D1|OC|FP|+0.044117|
|D1|OD|FN|+0.195386|
|D1|OD|FOREGROUND|-0.113268|
|D1|OD|FP|+0.082119|
|D2|OD|FN|+0.040637|
|D2|OD|FOREGROUND|-0.055071|
|D2|OD|FP|-0.014435|
|D3|OD|FN|+0.003211|
|D3|OD|FOREGROUND|-0.059433|
|D3|OD|FP|-0.056223|

误差分母为整幅图像像素，非FPR/FNR；均值FP/FN不能代数重建均值Dice。不能仅据这些总量命名某种空间形变。

|domain|channel|earlier_delta_pp|later_delta_pp|later_minus_earlier_pp|
|---|---|---|---|---|
|D0|OD|-0.012335|-0.311417|-0.299082|
|D1|OD|-0.869472|-1.355779|-0.486307|
|D1|OC|-1.201638|-1.717876|-0.516237|
|D2|OD|-0.008471|-0.280629|-0.272158|
|D3|OD|+0.107621|+0.532879|+0.425258|

真实较早/较晚位置也改变前序内容、BN与Adam历史，不是独立时间因果。全部方法、内容覆盖、中位数、六锚点最差seed/order保留于匿名表。

|domain|channel|seed|order|delta_pp|
|---|---|---|---|---|
|D0|OD|29009|0|-0.563923|
|D1|OC|29009|1|-1.890038|
|D1|OD|29009|1|-1.554005|
|D2|OD|29009|1|-0.408613|
|D3|OD|20260907|1|+0.044639|

共63个对比、3024个域×通道×seed×order单元，其中1217个负值均公开；没有协议内缺失。2000次按域内容bootstrap耦合六锚点，未做多重比较校正。192个内容是已暴露开发集，非患者独立或FULL验证；旧端点复用不增加独立样本。

## 资格、退休、成本与空间

36/36 fresh资格PASS：两个对角完整state/output与旧host exact；SPLIT独立公式的输出/梯度/BN/Adam/buffer/RNG核心状态exact；qOD0OC1与Wcurrent来源、PRE/初态只读恢复、首步一致、BN冻结、连续Adam、一次backward/Adam和真实F计数均通过。这些是实现资格，不是性能证据。

两个formal worker各1728次attempt=complete、9条COMPLETE，各27072物理F；正式共54144F，加资格438F，共3492次新GPU物理更新/54582F。三个GPU及CPU scorer实际exit0、注册PID均消失且GPU无该PID；CPU评分开始晚于所有GPU退休，在线标签/源图像读取均0。控制器COMPLETE且PID消失，但退出码未被独立记录。详见[TERMINAL_AUDIT.json](TERMINAL_AUDIT.json)。

累计GPU进程4980.716647秒，包含资格75.193553秒、formal0 2457.913005秒、formal1 2447.610089秒；CPU准备10.688980秒+评分25.032910秒=35.721891秒。worker主体时间是进程子集，不重复计账；没有失败attempt。正式墙钟2483.254秒，于2026-10-20T00:15:52.317077+08:00完成。按用户最新授权无GPU/时间上限。

新增NAS本轮一次解析路径统计为563560KiB（577085440字节），在2.2GB预留内；窗口仍预留NAS2.2GB+本地1GB，外加1GB余量，小于20,000,000,000字节硬限。统计之后仅增加小型匿名报告/收据，保留预留不缩减。

## 下一步的依据与限制

R82没有证明硬性固定OD q可提高性能。D3改善与D1大幅损失支持仅评估一个事先说明的折中：qOD=(q0OD+qHOD)/2、qOC=qHOC、Wcurrent，其余完全不变。该候选只是可能缓解硬切换的假设，不保证总体超过原方案；不得把连续轨迹效果按权重线性插值预测。后续必须另冻结协议、保留本轮负结果并独立资格检查，在同一开发集上的选择偏差明确保留。12小时授权窗口内继续有证据的小步优化，不因本次交付机械暂停；不做无上限系数搜索。
