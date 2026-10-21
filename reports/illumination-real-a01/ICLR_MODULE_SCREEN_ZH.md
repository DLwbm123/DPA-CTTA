# ICLR 2027 模块复查与本轮决策

使用2026-10-18公开投稿目录快照，非录用名单、非当前版本核定。标题关键词命中213个TTA相关条目和27个光度相关条目；只选读候选摘要及已有关键全文，不声称240篇全文已读，也不声称穷尽所有可用方法。

- [ISD](https://openreview.net/forum?id=a79fe9gjJE)：本次复核已有全文与旧接口审计。同维中间头与最终分类头共享是关键，本宿主up1–4为256维、最终头32维；BN前后同维不自动成为不同语义深度。没有新的兼容证据，不重复旧试验。
- [OriTTA](https://openreview.net/forum?id=GFQstEABV5)：已有全文审计仍有效；分类合成原型/每类记忆不能直接当dense二头语义锚点。
- [HRAC](https://openreview.net/forum?id=fb1drQMQmW)：原全文和此前翻转负结果均保留。嵌套关系可作检查，但不能把脑肿瘤固定比例阈值移植到病理杯盘；不重开E1/E2/P1。
- [EXACT](https://openreview.net/forum?id=mtgeo9qBIo)：摘要层面有按支持量启用prototype的可复用思想；本次官方全文入口要求浏览器验证，未获新增全文。没有完整二头契约，不据摘要启动。
- [Drift-TTA](https://openreview.net/forum?id=jXacgqRokv)、[Transporting Flow](https://openreview.net/forum?id=vJicORaQ93)：当前只读目录摘要；分别依赖离线adapter或预学习源密度，未登记资产，不能按“source-free”字面去掉前提。
- [ASPM](https://openreview.net/forum?id=fUUytHZl3e)、[T3LIE](https://openreview.net/forum?id=U6Ddri6MmB)、[Diffusion Photometric Priors](https://openreview.net/forum?id=Ke9xZ8xIbw)：摘要提示空间光度与内容分离、局部响应是潜在模块方向，但涉及学习模块或外部扩散模型；不冒称本轮已复用实现。

决策：ICLR新条目暂未解决当前直接准入缺口；不为了贴ICLR模块名称拼接依赖。先用已读完整算法的MSRCP-derived组件，在真实图像上检验空间照明校正是否有超出全局映射的任务价值。它是低成本前置资格实验，与ICLR论文原结果严格区分；若无收益即换路线，不继续运输/光度参数扫分。
