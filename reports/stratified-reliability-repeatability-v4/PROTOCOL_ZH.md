# V4 无标签 GPU 重复性诊断（独立工程资格轮）

2026-10-19。依据V3冻结parity失败（2320控制中138读出/238像素位不同），V3已NOT_EVALUABLE公开交付，不重跑、不放宽、不覆盖。现有seed固定/禁TF32/禁benchmark不等于严格确定性；非确定反向核只是候选原因，尚未证明。

唯一主要干预：DEFAULT为原设置，STRICT设置torch.use_deterministic_algorithms(True,warn_only=False)及cudnn.deterministic=True；其余相同。CUBLAS_WORKSPACE_CONFIG沿用:4096:8。禁止warn_only、吞掉不支持的算子错误或自动回退；任何不支持/严格差异/预算/访问失败即停止。

只用原8内容中的前2个RGB，不读标签/旧预测/旧评分，worker访问守卫禁止V3整个root和新scorer。图像来自原private注册表，不新增成员。固定4arrival：(内容0,C0)、(内容1,C1)、(内容0,C2)、(内容1,C0)，两内容反复出现；仅工程测试。STRAT、RANDOM、原生CW1/CW2各RESET和RETAIN，在每种模式内从同一source和seed执行两条完全相同轨迹。此处core RETAIN仍保留参数和Adam，如V2，用于隔离重复性，不执行V3的optimizer reset干预。共享访问守卫在配置含state_prior_root时即拒绝该整个目录，与optimizer reset开关解耦；V3原语义不变。两步core/CW2与一步CW1原实现不变，RDC/ENGRAM关闭，无新源训练。

共2模式×4方法×2状态×2重复×4arrival=128arrival，256个PRE/POST前向读出组成128对。逐元素比较logits及每条轨迹末模型state_dict（参数/缓冲），同时单列硬阈值预测位差及最大logit差；仅聚合保留，不保存逐例结果。主资格为STRICT全部8组合logit/state完全一致，允许DEFAULT诊断不一致，不能因DEFAULT偶然一致就宣布根因已定位。CPU scorer只在worker正常退役后审核工程汇总，不加载任何分割标签或评估Dice。

先通过CPU比较器边界测试及原9项回归；新GPU进程cap180s、CPU汇总cap30s、输出64MiB，实际GPU显存准入至少8192MiB，NAS探针和中性入口检查必做。一次唤醒只启动这一项，不自动重复资格或科学轮。V1至V3 GPU累计972.9505324364秒、已计CPU531.0013813972秒继续保留；本轮资源另计并累加，旧未计时错误/只读诊断仍披露。

本轮只检验同一进程中两个内容/四arrival的重复性；即使成功也不能认证跨进程/完整40arrival、更不能恢复V3资格或支撑TMI方法。下一轮若需科学比较，应先提出新的同进程配对设计及强控制，另冻协议；不能沿用V3跨轮差值并悄悄放宽parity。
