# 深层语义流与skip流的跨外观配对：真实资格 A01

运行前冻结。新假设是同一外观下深层与skip流的联合偏差可能被非线性解码器放大；在同图两外观的深层/skip完整2×2配对上平均，可能改善任务输出。不是删除敏感子空间，不改层/rank/强度救旧配置，不使用经典滤波；所有原参数/BN buffer冻结。允许假设失败，不要求事前已证明有效。

仅原8登记内容×4既定外观32输入、原checkpoint及预处理。当前归一化输入x与g=minmax(x^.7)各执行原模型；保存res返回的deep和4个skip，获得z00,z11。解码D严格重用原F.relu(deep)、up1–4、up5、bnout/ReLU、seg_head。交叉z01=D(deep_x,skip_g)，z10=D(deep_g,skip_x)，不混合不同图像、不采未来。两外观几何一致，不用原图GT或对齐标签。

五臂：SOURCE=z00；CROSS=(z00+z11+z01+z10)/4；INTACT=(z00+z11)/2；SHIFT=(z00+z11+D(deep_x,shift(skip_g))+D(deep_g,shift(skip_x)))/4；B原TOPO_SMOOTH。shift每级skip沿宽循环半幅，保留其值与形状但破坏深浅空间对应。所有平均均logit平均，独立二头阈值0，无额外后处理。INTACT是关键同信息双外观集成控制，CROSS−INTACT来自混合配对与解码非线性交互，不能把超过SOURCE的收益直接归给内部机制。SHIFT只控制空间配对，不保证最终改动能量相同。

每输入2完整原网络前向及4decoder-only前向；每worker第一个输入额外一次D(deep_x,skip_x)与原z00逐位核对。两worker共128完整网络+258 decoder-only，不把复用编码视为免费。模型state逐输入不变检查、SOURCE迁移bits parity、GT/非登记资产/网络deny、全部logits/masks/诊断跨worker重演后独立GT评分。工程代数检查可加分离解码器的交互抵消，但不当科学合成结果。

固定主门：CROSS−B OC均值≥.30pp且95%内容配对bootstrap下界>0；相对SOURCE/INTACT/SHIFT的OC差下界均>0；OD−B下界≥−.10pp；最差逐输入头下降≤1pp；不新增空头/嵌套；全部表面/vCDR定义；OC ASSD/HD95比B之比CI上界≤1.05；vCDR误差差上界≤.005；CROSS与SHIFT logits差>1e-3比例≥.1。8内容等权，条件[2,1,1,1]/5，2000重采样seed20261018。失败关闭本固定跨外观配对，不扫描gamma、配对层、融合系数或集成视图救分。8反复使用开发内容不代表独立患者，不外推真实持续适应/新颖性。

预算仅my-gpu指定GPU1。新root基线0/预计768MiB/预留1280MiB/hard1GiB，总原clock3600秒、独立guard3540/stage3530，首输入成本投影门。NFS/GPU/中性父子argv核查；项目18GB工作/20GB硬限、aux1GB不变。完成或失败都交付自有源码/协议/聚合；私有图像、mask、checkpoint和张量不公开。阶段wall含CPU和加载，不称kernel时间；controller父OS若无回执保持未知。
