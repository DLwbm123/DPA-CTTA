# E7：冻结源随机遮挡响应的信号资格（科学设计冻结，未启动）

## 对象与访问

保持E6-B/C原联合8连接OC-only块、B=TOPO_SMOOTH默认、OD-active锁B和全部32输入。共同集合按完全相同E6无标签规则重新生成：末32维原生512网格、原/shift特征和两源logits查询/锚点范数非零、半径16、排除分歧≤3、S/B同意且p≥.9/≤.1、各8最近锚点、原tie。不能从E6私有效用/评分选择共同集合。非finite停止，zero norm不可评分，无epsilon/floor。源bit parity及共同集合计数与E6封存无标签输出核对；不读取E6 public结果来适应分数。

源eval/no_grad，模型参数、BNbuffers不变。每输入源1次+mask8次，全部32输入全encoder重演一次，共576完整encoder前向；无backward/optimizer/训练、无新源数据。worker只8登记RGB、既有checkpoint、新root限定64端点bits；拒旧scorer/效用/E5/E6A/E6BC结果和GT。worker封存所有分数与mask生成元数据后exit0退休，独立CPU才读8GT。稠密表示/概率/遮挡RGB不落盘。

## 干预固定

采用MARS默认K8、patch4、rho独立uniform[.2,.4]；512²分为128²patch，用NumPy default_rng(20261018 + case*4 + condition)，依次每视图先取rho再choice无放回floor(16384*rho)块，4×repeat。已归一化输入选中处乘0，不另填颜色/裁剪/翻转。相同源输入的掩码全部控制共享；重演重置相同rng，实际rho/patchindex精确重演，mask仅私有元数据。并不声称视图是独立患者或mask独立于解剖正确性。

所有query都评分，包括被mask覆盖的位置，保持原文全图扰动语义，不根据结果挑可见query。报告每块query平均遮挡率的匿名分布，不能借此后选子集。

## 分数固定（越高越支持撤回B→S）

每个OC-only块在实际OC变化像素取平均；p为源OC概率、q_bar为8mask前景概率均值，H为自然对数Bernoulli熵，0log0按连续值0，禁止epsilon。c=p*q_bar+(1-p)*(1-q_bar)。

1. MARS_FD（唯一主臂）：`−H(p)*(1−c)`。
2. RAW_MARGIN：`abs(source_OC_logit)`。
3. NULL_FUSION：`−H(p)*2*p*(1−p)`，即q=p时的完整原融合，隔离纯源置信度作用。
4. CONSISTENCY_ONLY：c。
5. MASKED_MEAN_DIRECT：`(2*S_OC−1)*(q_bar−.5)`，只看遮挡后的源方向支持，非主臂。
6. GEOMETRY_ONLY：完全E6锚点距离差/16及delta聚合。
7. RESPONSE_SHIFT：保p原位，仅q_bar场宽roll W//2+1后计算同MARS_FD，破坏对应，q边际不变。
8. STRATIFIED_ROLL：共同块内按E6输入/源→B增删方向/size1、2–4、≥5/预测边界距离≤1、(1,3]、>3分层，scanblockid roll1，singleton不动。分数位置移动≥.5，否则控制不足门失败。实际值变化另报。

熵必须稳定计算并与高精度合成对照，不能把浮点sigmoid舍入导致0log0当错误，不能因此改变源logit值；实现使用logsigmoid恒等式。分数均CPU FP64，masked q来自固定FP32模型输出转FP64 sigmoid，操作定义固定。

## 标签效用、统计与门槛

沿E6精确单块撤回Dice效用，不用GToracle选择当标签；正/负分开、零效用排除AUROC、无一类时None不补0/.5。BLOCK主、|utility|和实际编辑PIXEL双重加权辅助；逐输入AUC→固定条件权重[2,1,1,1]有效项重归一→8内容等权，2000次seed20261018内容绑定描述bootstrap。

唯一主MARS_FD−RAW_MARGIN BLOCK AUC均值≥.05且95%下界>0；MARS_FD AUC≥.60；对NULL_FUSION/CONSISTENCY_ONLY/MASKED_MEAN_DIRECT/GEOMETRY_ONLY/RESPONSE_SHIFT/STRATIFIED_ROLL的BLOCK配对差都须均值>0且CI下界>0。绝对效用加权对RAW_MARGIN差≥0。共同oracle≥.2pp且CI正、主32中至少24个有正负类、8内容均有定义、分层置乱位置移动≥.5。全部门合取，任一失败不推进selector。该协议以强控制为准，不仅赢坏错位分数。

报告全部AUC/配对差/共同空间、score ties/size与方向匿名分层、正负零效用数；不挑子条件/方向/大小或反转分数救主。无方法预测安装，不报新方法Dice/ASSDHD95/vCDR/源返回；原oracle空间仅供资格，不证明结构安全。不是临床、独立患者、原文复现或持续收益。

## 工程停止与预算

先合成二Bernoulli恒等/熵极限、mask数与坐标/RNG、源immutable、q=p null、view同均值异方差、score符号、AUCties/缺类、共同集合/访问隔离/重演。独立24h clock自首次实际CPU起、修复共享；86340s保存停，86400s原身份全部退役，不能靠heartbeat/SSH限制。256MiB root输出/320MiB含code archive cache logs预留，累计20GB/工作18GB/辅助1GB。NAS实际NFS容量小写读、GPU6 UUID/free≥8192/中性父子ps和自身nvidia检查通过后直接运行，不等审核。一次只一本聊天GPU，旧退休PID不扫描。

完成须32输入/576完整encoder、全输入全部分数/掩码/锚点/uid精确重演、源参数buffers不变、ACCESS/bit parity/score冻结、独立CPU8GT完整24AUC/21配对行、worker/scorer exit0退休、controller/guardian退休与最终ledger。工程失败保留原source/log/cost、独立修复协议且共用原始clock；科学负结果不调比例/patch/K追分。完成失败都中文报告/匿名聚合/无路径receipt，代理核账号remoteSHA/匿名200才称push完成。
