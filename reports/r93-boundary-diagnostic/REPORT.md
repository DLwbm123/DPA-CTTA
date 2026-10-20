# R93：域边界历史隔离的三臂机制筛查

**COMPLETE，冻结推进门 NOT_PASSED。保留原VV，关闭这项边界历史隔离，不调分组、不训练路由器、不扩大seed矩阵。** S相对伪分组J有优势，但两者均差于原VV；D3OC的小幅改善不能抵消D1有益历史的大幅丢失及结构恶化。

## 设计及可解释范围

本轮仅原seed20260907、两既定order、全部192开发内容，V/S/J各2条独立轨迹。V原VV全流共享；S依既有域metadata离线预构造四bank；J用固定rng键93分配canonical内容，四bank同最终22/57/57/56更新数。所有臂保留原VV19136 BN/currentstat/原q/W/strong配对/损失/Adam与原图K，不部署Qt，不做融合/teacher/EMA。

两order每域只出现一个连续块，没有重访，所以S严格等价于域边界恢复初态affine+空Adam，**未测试状态检索**。S与J的最终age及全流localage multiset相同，但order0/1分别189/183次arrival的localage不同；S四段，J137/139段，S−J不能唯一归因于域语义。metadata只用于CPU预构造匿名索引，模型不读域名、标签、旧评分或未来图像；这是oracle域信息定义的离线干预，不是domain-agnostic部署算法。[冻结协议](PROTOCOL.md)、R92准入及旧reset非严格等价审计均在GPU前记录。

## 主要结果

以下绝对分数只对应本轮一个seed的两order；不与R91三seed总均值混作时间趋势。

|臂|raw macro Dice %|固定后处理后 %|
|---|---:|---:|
|V 原VV|76.734295|76.865502|
|S 域边界隔离|75.732163|75.882783|
|J 固定伪分组|75.501274|75.671983|

|预定raw比较|macro差 pp|两order差 pp|Bonferroni双侧97.5%内容CI pp|主门|
|---|---:|---|---|---|
|S−V|−1.002132|−.773069 / −1.231195|[−1.380773,−.654052]|失败|
|S−J|+.230889|+.245343 / +.216435|[+.094325,+.394497]|通过|

J−V为−1.233021pp（算术描述，未新增CI）。S比J好仅说明这两种分组操作有条件差异，不能宣称S优于保留原历史。post的S−V为−.982719pp、S−J为+.210800pp，不挽救raw失败。所有区间按域内容bootstrap2000次，两anchor耦合；不是患者独立或seed总体区间。

## 保护门与全部域头

|域|S−V OD pp|S−V OC pp|S−J OD pp|S−J OC pp|
|---|---:|---:|---:|---:|
|D0|−.262000|−.258744|−.182925|−.027557|
|D1|−2.989289|−4.315135|+.658618|+.375166|
|D2|−.400481|−.417860|−.088873|+.149981|
|D3|+.226517|+.399937|+.832784|+.129921|

D3OC相对V +.399937pp达到预定+.3pp；56内容50正/6负，中位数+.376649pp，改善有较广覆盖。可是D1OD/OC远低于−.2pp保护门：57内容分别46负/11正、44负/13正，中位数−1.579927/−1.553101pp，最差内容跨order均值−16.240057/−24.857772pp，不是仅一个异常值造成。不能挑D3结果忽略D1代价。

S相对同seedNATIVE的所有域头均值仍在−.5pp限制内（最差D0OD−.488262pp），所有域头order也在−2pp内（最差D2OD/order1−1.033519pp）。这两项通过不能抵消S−V主门与D1保护失败。它说明原VV的额外历史收益在S中大量丢失，而不是S一定比所有基线都差。

## 误差、结构和NA

S−V raw总体整图FP+.055793、FN+.100137、前景−.044344个百分点，误报与漏报均增加；不是FPR/FNR。ASSD+1.119626输出网格像素、HD95+3.130449像素，结构恶化。总体vCDR误差−.000206507略改善，但D1该误差raw+.007146155、post+.004274417，不能以总体均值掩盖。

raw/post共38项结构/空头/NA检查，16项失败，包含总体ASSD/HD95、D1双头距离与vCDR、raw D3OD距离；全部评分NA计数0，空头不增加。结构距离不是mm，误差指标不构成临床安全保证。固定后处理保持M0的8-LCC→4背景填洞→OC∩OD，未按本轮分数调整。

全部2112行绝对指标、704行配对单元、352行内容分布与逐项门见ALL_SCORES.csv、ALL_CONTRAST_CELLS.csv、CONTENT_DISTRIBUTIONS.csv、PERFORMANCE_GATE.json，保留所有正负结果/NA而非只展示D3收益。

## 完整性、费用与资源

fresh资格9updates/90F/9BW通过：原V重演对旧参照与fullsnapshot exact、bank0/1/0对独立bank参照fullstate/output exact、global visits/RNG/counters不回退、local ages一致。正式3臂×2order×192=1152更新；V两流逐PRE/K对同硬件R85mask exact，未复制V大预测。所有臂globalvisits192；V最终age192，S/J四bank各22/57/57/56，与冻结路由一致。含资格共1161updates/11610F/1161BW，全部完成，无GPU执行失败attempt。

三GPUworker及CPUscore全exit0、PID实际消失；controller COMPLETE/PID消失但独立数值退出码未记录；watchdog正常all_owned_children_retired。GPU真实退休后CPU先封存4条S/J raw/post，再读既有192标签；在线labels/sourceimages0。V/NATIVE旧指标仅在新V exact后复用，不当独立重复。

GPU进程秒：资格50.598338、formal0 768.468633、formal1 874.648934，合1693.715905。CPUscore进程481.616402秒，body479.505692为子集；CPU fixture body5.755130秒另列，其余准备未独立计量。首neutral入口literal newline SyntaxError发生在imports前，保留原entry及失败证据，修复仅入口；一次GO收据相对路径scp失败已重传，无科学/资格放宽。未量准备不记零，body不双算，进程秒不是能耗或CPU核时，spawn到观测退出可含约5秒poll余量。

原共享T0至controller结束17011.438590秒，原12h绝对deadline未重置。本轮控制器历时1417.520783秒。含旧R90失败与R91/R93资格后共享累计4253updates/43700F/4253BW，GPU5844.183504进程秒；已测M0/M1/R91/R92/R93 CPU进程3780.470235秒，自检与未量准备另列，不称完整CPU总费。更新/前向尚余747/6300仅为账面余额，不作为再开实验理由。旧服务器R85未知尾账独立保留，不混入完整累计。

一次du实root为12,099,584bytes，父root行87,064,576排除已计该child，campaign实际snapshot合99,164,160bytes。最终小报告/DELIVERY在恢复余量内，不重复du。低于R93 32MiB/campaign1.5GiB cap；总保守预留19,980,140,544bytes<20GB，不释放历史预留、不删除旧结果腾空间。完整计数、费用、缺失和空间解释见TERMINAL_AUDIT.json、COMPLETION.json、SEALED.json。

## 决策

原有跨块历史对D1有重要净价值；在本次一个seed两order中，清除跨域前缀可改善D3OC，却不能保住整体与D1收益。它不证明所有域共享都好或所有隔离都无效，也不识别唯一Adam/参数机制。**关闭本项边界隔离，不调伪分组/边界/seed/门槛、不训练router，不扩展成大矩阵。** 保留原VV，当前无GPU作业。

全部192/1951内容历史暴露，patient/video/source overlap UNKNOWN；R87未暴露确认不可用、R88前置未满足。一个seed两order只机制screen，不能冒充六anchor复核、患者独立确认或部署可识别性。持续研究授权仍有效，但后继必须有区别于已否决操作的新证据与冻结协议，不为用满余额空烧GPU。
