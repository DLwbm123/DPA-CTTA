# 原生解码路径交互分离：运行前冻结

上一轮向量负投影发生了真实激活变化，却只改变5个OC像素。新假设不是将该投影放大：原网络非线性地混合上下文与跳连，某些不可靠的共同激活可能通过分类头形成错误证据；以四种固定路径组合分离混合交互，检验相加证据是否更可靠。此为未经验证的模型内假设，不是额外监督或语义真值。训练得到的交互很可能有益，主臂因此输给SOURCE/B也是明确反证。

固定原up4两半128通道，经up5对应无bias卷积及冻结BN尺度形成d/s，bias与BN偏移形成b。u=b+d+s须与原生bnout输出allclose(rtol2e-5/atol2e-4)。定义f(v)=seg_head(ReLU(v))，z为原生SOURCE；zD=f(b+d)，zS=f(b+s)，z0=f(b)。唯一主ADDITIVE=zD+zS-z0。交互I=z-ADDITIVE。固定系数1、原末级接口，不扫层/系数/基线/方向，不改上一轮投影配置。对仿射读出，混合有限差分恒零；ReLU下可正可负。保留原模型权重/BN/输入，无梯度、历史或新模型。

完整臂：SOURCE、ADDITIVE、DEEP=zD、SKIP=zS、REVERSE=z+I、SHUFFLED=z+roll(ADDITIVE-z,半幅水平方向)、强静态B。DEEP/SKIP为未校准单路径消融，不是假定有真值意义的教师。REVERSE与SHUFFLED在每头全图匹配主残差L2；错位不匹配逐像素幅度或边界支持，不能单靠胜过它声称因果识别。SOURCE/B与全部四控制必须同时通过，失败不后选副臂。

只原8内容×4条件、原checkpoint/GPU1，正反各32输入；两独立worker共128输入。每输入1full、2pathconv、3head，合128full/256pathconv/384head。双序无状态只测顺序不变，不称持续学习。SOURCE位图一致、state不变/无梯度、独立精确重演、访问负测通过；生成全部退休后独立评分读GT。使用既有内容bootstrap及条件权重，不增分母。

主OC−B≥.3pp且CI下界>0，对SOURCE/DEEP/SKIP/REVERSE/SHUFFLED的OC差CI下界>0，两个顺序均过主门。OD差下界≥−.1pp，最差头≥−1pp，无新增空头/嵌套；表面/vCDR完整，OC ASSD/HD95对B比例上界≤1.05，vCDR误差差上界≤.005；主与SHUFFLED logit差>1e−3比例≥.1。保持全部门，不以更多编辑像素等同收益。

预约1.25GiB、输出1GiB硬限，总时钟3600s（回溯10min覆盖准备）、独立guard3540/stage3530，重试共享原钟。首输入估价须满足总预算，写前NFS/容量/probe、GPU≥8GiB余量，中性父子argv及nvidia核。禁止删除历史资产。完成公开自有源码/聚合报告，不发布原图/GT/身份/权重/逐样本预测。
