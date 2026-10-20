# R91 CPU 重复性与资产审计

**ADMISSIBLE：没有找到同硬件、同192内容/三seed两order、同VV写入下，入站六弱概率均值Qt、初态同配方Q0与实际K的完整相同实验或可复用Qt/Q0资产。** 这不是声称所有不可访问历史私有文件均不存在；旧服务器不可用资产不作为本轮对照，也不重启旧任务。

审计覆盖本仓库 tracked reports/src 中 weak/六弱、readout/读出、output averaging、Q0/q0 等相关协议与实现，并检查当前服务器已有实验根目录及迁移源的资产记录。不读取图像/标签、不构造模型、不调用GPU、不hash或全扫dataset。

|已核材料|与拟议对照的关系|处理|
|---|---|---|
|R31 readout|调整最终分类器参数，非原六弱入站概率均值|不重复|
|R73 crop mechanism|48内容、两个既有anchor，目标与裁剪机制，非192完整VV历史/同硬件Q0/Qt/K|不复用为匹配控制|
|R75/R76|稠密白化/均值保持特征读出，非原六弱均值|已否决路线不重跑|
|R78 PROTOCOL与method.readouts|更新后O与水平翻转原图等概率平均A；不是入站六弱Qt，且旧硬件|不将其失败改名重跑；本次不同因素|
|R79|PRE/K、替代强增强/episodic诊断，无六弱输出候选|旧reset不恢复|
|R80/R81 PROTOCOL与method|初态六弱参与q/W反馈，改变学习轨迹；未交付同一原VV历史的只读Qt/Q0/K面板|不混合轨迹，不恢复反馈候选|
|R82–R86|q/损失/配对干预，原图K为输出，无六弱部署读出|保留全部负结果|
|R85迁移、R90 M0 ASSET_AUDIT|同硬件P0/PRE/K/NATIVE存在；固定六弱静态与概率输出NOT_AVAILABLE|只读复用原PRE/K参照及M0指标|

精确公式核验：原cal_consis_loss在normalized输入上依次identity、horizontal flip、vertical flip、90/180/270 rotation，各视图logits逆对齐，CPU torch.stack(...).sigmoid().mean(0)送至原criterion。无随机弱变换；当前q由原Host.target_cache捕获，已经验证等于该均值。Q0按同一模型/原变换/同精度顺序计算，初始BN affine与currentstat，无跨图学习、无新教师。

源定义证据：r20_model_only_search/method.py::_weak_hook、r85_loss_aggregation/method.py::_criterion、r78_aligned_barycenter/method.py::readouts。第三方实现仅私有读取，不公开复制。

CPU准备中inspect.getsource对动态载入类失败（TypeError），改为读取已定位源文件；无模型/GPU/图像/标签调用，其短CPU时间未独立计量。初次猜测源路径不存在，随后按文件名定位；不影响科学准入。
