# A0执行审计

## 冻结、实现与资格

执行源21f5ab6a6fcc47f28178ebba32897ab449b217c8，branch experiment/single-bn-rb-a0。SOURCE/ADABN/VAR_ONLY/RB_BN四组，固定res.bn1(64ch)，普通eval BN，所有上游/下游参数与BN buffers冻结。没有新增源训练、优化器、反向传播、参数位移或记忆；RDC/ENGRAM关闭。评分前没有改变源码/config/protocol/层/主比较/阈值/预算。

5项CPU合成检查通过，完整进程4.477785348892212秒，unittest body .221秒（打印精度）；无真实RGB/标签/checkpoint/CUDA。覆盖源输出一致性/当前图总体矩/复制位置不改矩/零方差无额外floor/补偿恒等式/无buffer改动/同输入重复/配置与静态汇总形状。py_compile及阅读打包编排没有单独计时，不计入CPU worker伪造精确成本。

实际NAS为NFS，容量>4GiB及小创建写读探针通过；启动GPU6 UUID在私有回执核对，free23858MiB>=8192。使用中性入口/解释器，唯一9秒短检主子ps完整命令均中性。**该短检时GPU compute-apps列表尚无本worker，不能说已采样确认GPU列表中的名称。** 后续完整GPU执行与源不变、全重演已证实；可见入口无方法/项目/用户名称，未为补观察重启实验。

首2输入实际四分支完整均时 .4577007293701172秒，投影104.72703485488893<=1180 PASS。GPU硬cap1200/CPU900/输出2GiB；未重试、未改准入系数或减少覆盖。

## 完成与退役

GPU_COMPLETE32输入/128bit预测、256forward，全部128个logit数组经全encoder重推逐元素一致；仅同进程、不认证跨进程。所有分支指定层输入逐元素相同，源state_dict参数/buffer逐元素不变，grad为空。局部恒等式使用真实FP32安装后的输出通道均值，误差记录在DIAGNOSTICS；没有把公式输出冒充实测。

worker退出0并且active=false、/proc不存在后，独立CPU评分才开始。SCORE_COMPLETE8标签/128metric/4绝对/32区域；聚合CSV另外32条件头及14预定对比已核，所有C+H=D。32唯一输入采用5块条件权重/两重复顺序只形成320逻辑静态读出，未增加独立样本数。

worker/scorer均exit0 inactive且/proc不存在，controller原身份/proc不存在；RESOURCE_LEDGER all_workers_retired/no_retry。私有PID/实际路径不公开，仅公开退役结论。scorer未修复、GPU未重跑、预测/门槛/标签时序没有改变。

## 成本与历史对账

|项目|完整进程墙钟秒|主体墙钟秒|
|---|---:|---:|
|GPU worker|21.523701667785645|19.118167877197266|
|CPU scorer|21.41064476966858|19.188894510269165|
|CPU准备测试|4.477785348892212|.221（日志三位精度）|

峰值torch GPU已分配508889088B；worker生成结束计8427018B，完成评分/小回执后的运行输出8691405B（不含code/source archive）。没有落盘dense特征/logits/概率；128bit文件私有。read/normalize/forward/写盘均在相应worker计费内，跨分支共享的执行不能用于方法部署延迟排名。

历史GPU5143.519649743976秒，加A0后 **5165.043351411761秒**；历史已计CPU2367.7230074110694秒，加准备和评分后 **2393.611437529630秒**。旧本地.631近似、公式body及.0084569158历史基数差、未计时导入/阅读/编排等继续保留，不混其它聊天成本，不清零既有失败。见PRIOR_RESOURCE_LEDGER与ANALYSIS_SUMMARY。

## 科学与公开边界

mechanism_primary_pass=true与advance_to_history_protocol=false分别保留；不是用主机制阳性忽略对源和结构的失败。无PREPOST学习/持续记忆，原条件再读不作遗忘结论。8重复开发内容、患者及预训练成员未知；所有区间描述性，次要比较未校正。

发布前一次必要检查：本分支/远程/账号/待提交范围、聚合覆盖与C+H=D、隐私关键词；只发自有代码协议结果与无路径回执。Github经有效代理操作，不绕过NO_PROXY。最终提交/远程SHA及匿名HTTP访问结果以交付回执和最终回复为准，冻结执行源不因报告提交而改变。
