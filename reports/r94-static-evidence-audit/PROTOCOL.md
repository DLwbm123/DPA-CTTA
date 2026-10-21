# R94：强静态对照与证据准入审计（CPU-only）

冻结日期：2026-10-21；基准版本 e9059b4737323aadd7b21cfd59df3d4670f323e1。
用户明确授权执行该审计、单列预算与硬超时、报告并 push；禁止新增 GPU 实验，不重置旧 campaign 时钟。

## 问题与范围

只补齐原 VV K−已有强静态 Q0 的配对证据，审计是否出现新确认数据或机制依据。SOURCE_ONLY/P0/Q0/NATIVE/K 定义和原 raw/post 固定；不修改方法、不重评分 R90–R93 晋级、不调阈值或新增选择器。K−Q0 为完整输出方案比较，不是单因素历史因果；同读出历史量 P−P0、Qt−Q0 沿用旧结论。

## 资产先行

只读检查发现 R91 已有内容级 values.private.npz（669152 bytes，values/domains 两数组）与 M0 已有内容级 m0_values.private.npz（661696 bytes）；已有 R91 COMPLETE/SEALED 收据保留 exact 和先封存再标签事实。优先直接使用这些已评分数组，不读取封存预测、图像、标签或模型。

R91 values 轴固定 content192 × anchor6 × arm6(P0,P,K,Qt,Q0,NATIVE) × raw/post2 × head2 × metric11。M0 为 content192 × anchor6 × arm4(SOURCE_ONLY,P0,NATIVE,VV) × PRE/K2 × raw/post2 × head2 × metric11。两者内容顺序均由同迁移 order0 canonical panel 绑定，已有代码可追溯；本轮核 domains 一致、quota22/57/57/56、静态广播一致，并将关键已有域头/anchor聚合与原公开表核对，不从汇总均值重建内容。

Q0 每内容一次的六 anchor 广播不是独立观测。复用的 K 是旧 exact 主线，不计独立重复。若数组/映射或旧汇总不符，停止且记 NOT_EVALUABLE，不用近似或补 GPU 绕过。

## 唯一新增统计

对 K−Q0 同内容配对差 raw/post，给出：
- 四域双头等权 macro、六 anchor、两 order/三 seed；四域双头各 anchor 和均值；LODO 仅事后描述。
- 各域头内容跨六 anchor 均值的正/负/零覆盖、median/p10/p90/min/max，保留全部负结果；不发布逐内容身份或可连接行。
- Dice、整图 FP/FN/前景、ASSD/HD95（输出网格 pixel）、vCDR 绝对误差、嵌套、空头/NA。vCDR 总体只计一份，不将双头复制当新观测。结构差使用共同 finite 配对；NA 各侧另报。沿用原距离/空头规则，不重新计算结构。
- K−Q0 raw 为主要事后比较；post 是固定敏感性描述。两者分别报告 95% 及 97.5% 配对内容百分位区间；不设新晋级门。域头区间为未校正探索描述，不称同时覆盖。

bootstrap 固定 rng20261021、2000 次，各域内部按内容有放回，每个 sampled content 同步带入全部 seed/order/head/method。宏平均四域等权；同一抽样同时用于 raw/post 与域头描述。六 anchor 不单独重采样，不能相减旧 CI 端点。缺失使用共同 finite 配对并披露有效分母；Dice 应全 finite，否则停止。

必要自检仅覆盖本轮新逻辑：常数/配对共同扰动对消；域均衡与 anchor 耦合；逐内容恒等 K−Q0=(Qt−Q0)−(Qt−K)；已有聚合 parity。不得重做 R92 全量编辑账或历史 GPU 重演。

## 数据与证据准入

只检查 R87/R88 以来本会话实际收到的新授权内容、关联身份或源成员证据；没有新增则沿用已记录阻塞，不全扫旧 dataset、不重划1951、不换seed冒充确认。分别记录：未参与方法选择、患者/眼别/重复拍摄独立性、源/预训练成员重叠；未知分别保留 UNKNOWN。

状态仅可为：有合格新确认条件→只冻结确认协议并资源准入；有真正新机制依据→只进协议审查；两者均无→保留 VV，目前无合格可运行 GPU 后继。本 CPU 审计不授权自动 GPU 阶段。当前没有提交新的合格 cohort、关联身份/源训练成员清单或新的在线机制信号证据。

## 独立小预算与原时钟

旧 campaign 原 T0=1792492974.1323855、deadline=1792536174.1323855 已过期，累计4253updates/43700F保持不变。R94 是用户新授权的独立文档/CPU审计，不延续或重命名旧性能运行。

新审计保守 T0=1792549120.201643（首份预算收据前回溯300秒覆盖初始准备；精确首工具时刻未独立计量），绝对deadline=1792552720.201643；全流程60分钟，失败/修复共享。CPU外部 timeout 最多900秒（TERM后5秒KILL，且不越绝对截止），内部alarm840秒，CPU核秒硬限840/845；单线程 BLAS。worker写前检查；独立wrapper每2秒检查空间与截止，必要时仅终止自有 startticks 匹配进程组。主子均中性argv，GPU可见设备空；不调用nvidia-smi或查询旧GPU/PID。

远端8MiB、本地8MiB新增预留，总保守19,996,917,760bytes<20GB，余3,082,240bytes。远端正常写入上限7MiB、总cap8MiB含1MiB恢复；以max(logical,allocated)+目录块计，写前保守块预估，wrapper独立存储guard；本地代码/报告/私有收据同时计入本地8MiB。新root另立，不复制大资产，不删除历史、不释放历史预留。RAM仅加载既有小数组；预计新增输出<2MiB。新增模型/GPU F/BW/update/图像/标签读取均0。

## 完成与公开边界

CPU exit0、timeout/wrapper/worker真实退休，覆盖/NA/parity/访问审计、进程费用及未量准备、一次实际新root空间核齐。公开源码、冻结协议、匿名汇总、资产角色/证据边界和继续/停止决策；不公开私有路径、逐图身份/预测/labels/weights。中文报告commit/push原branch，代理连接，核remoteSHA与匿名REPORT200；local+server DELIVERY。只有本次审计输出，不能覆盖旧报告/状态。唯一r52沿用，不新建监测。
