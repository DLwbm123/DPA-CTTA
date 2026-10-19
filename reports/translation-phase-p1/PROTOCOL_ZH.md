# P1冻结协议：四平移相位直接读出

## 范围和冻结定义

不做Calibration Maps，不重启BN尺度或B1调参。不新增源训练/标签/图像/模型/种子/流/权限，RDC/ENGRAM关。保持既有8内容、4条件、源checkpoint与预处理。单图独立双sigmoid，无softmax或杯盘比例目标。

每个当前图先按原方法归一化，然后以原尺寸整数(dy,dx)={(0,0),(0,1),(1,0),(1,1)}查询同一冻结源模型。输入左/上reflect补齐再截到原尺寸；不卷绕、不重新归一/缩放/插值。正确反移取每个输出的[dy:dy+H-1,dx:dx+W-1]。所有干预只安装到原图[0:H-1,0:W-1]，最后一行/列精确保持原源logit；所有臂同一共同域。

八臂SOURCE、PHASE_PROB（四个反移独立Bernoulli概率等权.25）、NOALIGN_PROB（同四输出不反移）、PHASE_LOGIT（反移logit均值）、SOURCE_SMOOTH（仅源概率reflect 3×3可分[1,2,1]/4平滑）、SHIFT_X/Y/XY（单个反移视图）。概率均值以FP64计算，转logit仅为统一存储/评分表示，clamp[1e-12,1-1e-12]不作为预测阈值搜索。SOURCE不改变原logit。任何非finite/形状/边缘回退/源状态/重演失败硬停。

## 主比较和推进

唯一主PHASE_PROB−NOALIGN_PROB:OC_DICE，均值≥+.5pp且内容绑定的2000次seed20261018描述bootstrap下界>0。五块条件权重[0,1,2,3,0]，两重复顺序只重读静态结果，不增加独立样本。

还必须：PHASE_PROB对SOURCE_SMOOTH与三个单视图的OC区间下界全>0；对源OC均值≥+.5pp且CI下界>0、双头CI下界>0、最差条件头≥−1pp、原条件再读双头均值≥−.2pp；OC GT前景和3px边界净纠错CI下界>0；ASSD/HD95/vCDR三项点估计均不劣源、surface/vCDR覆盖不下降、空/嵌套不增。PHASE_LOGIT是预定副读出，不替换主。任何门失败advance=false；不事后改平移量/相位/权重/阈值或挑条件。

各独立视图的Dice/结构/净纠错全报，以免重演E2-A只看学生不看教师错误。NOALIGN与PHASE共享完全相同4次网络查询，仅对应不同；SOURCE_SMOOTH不需新网络查询，属于更便宜的平滑解释控制，不宣称函数幅度匹配。静态没有RESET/参数状态，SOURCE就是零干预；不机械重复CW学习流或用旧currentBN较低分数支持成功。

## 覆盖与访问

32输入×8臂=256唯一bit预测/metric；8绝对、64条件头、64区域、34对比。640逻辑静态读出不是独立患者/CTTA轨迹。32输入完整重推4相位，全部256最终数组和诊断精确重演，总256模型前向；同进程限定。源全state_dict不变/0optimizer backward。标签只GPU worker exit0退役后由独立CPUscore读取；GT区域不回传。GPU拒本轮scorer、上一B1整个root和旧评分，只8登记RGB/既有权重。CPU合成检查与NAS/GPU/中性进程准入必须先过。

## 预算与停止

一次GPU派发≤1200s、独立CPU评分≤900s、输出≤2GiB，不重试。首两完整4查询束现场投影=已用+(64−2)*均时*1.35+60≤1180才准入，含32输入全重演；不砍臂/放宽/扩数据或延预算。源不可变/正确反移/全覆盖/重演/访问/worker scorer与controller退役/finalledger齐备才科学可评价，失败也保留部分与成本。

前轮累计GPU5218.60620903963s/已计CPU2471.9526076021857s继续累加；旧单列body与成本差、未计时错误/阅读/编排保留。本轮只静态教师信息资格，continual_benefit_tested=false，不作独立患者、临床/TMI成功推断。完成或停止写中文分析/执行审计，仅自有代码、匿名聚合、无路径回执经代理push并验证远程SHA/匿名200；原图标签权重features概率逐例评分身份私人路径及第三方全文不发布。
