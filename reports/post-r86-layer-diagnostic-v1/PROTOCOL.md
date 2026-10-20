# R86后既有向量的全BN组CPU描述分析（冻结）

状态：FROZEN_BEFORE_ANALYSIS。独立的终态后离线分析，不续跑或重置R85/R86实验；没有新性能测试、模型构建、前向、反向、Adam更新、图像或标签读取。输入仅原48固定同H探针的4梯度/4实际delta/8分量梯度和对应6条VV终态snapshot的参数名称/shape；终态Adam状态不得用于探针分析。模型调用和GPU更新预算均为0。

原19136维向量按b1_host.configure真实参数序及终态named parameter顺序切分；全部6份映射名称/shape/order须一致、82参数tensor总数19136，所有probe tensor须同维度且有限。固定41个BN模块、10架构阶段（stem、res.layer1–4、up1–4、bnout）及全BN；每组分别all/gamma/beta，共156个组，全部报告，不按结果挑top组。阶段分组仅按名称结构，不依结果或Dice。gamma=weight，beta=bias。

48=4匿名域×两个机械位置×3seed×2order；按既有公开表(seed,order,probe_number)映射域，同一域probe_number小/大对应原预设25%/75%位置。每个域×seed×order必须恰有2位置，不新选内容。不能把48当独立患者/重复，不能把稀疏delta求和当192步漂移。

所有组、4cell均报告实际delta绝对L2、每参数RMS、组内能量占本cell全BN能量比；VB/BV/BB−VV实际delta差的L2、RMS、占全BN差能量比例、该cell实际delta与VV的局部余弦及范数比；VV自对照明确为0差。按全部seed/order/domain/position及整体提供描述汇总，原probe详细行全部保留。

通道梯度g_ch=B_cell,ch+.1 S_cell,ch，原BCE全局numel及LSO全通道联合分母不改。报告局部OD/OC余弦、各通道与实际delta_VV的内积；负内积只表示该代理损失的一阶下降方向，不等于Dice收益。另报告R_delta=delta_BB−delta_BV−delta_VB+delta_VV的组L2/RMS/能量比例及相对尺度L2(R)/sum_cell L2(delta_cell)，以及同组梯度四格残差。固定共同m/v/t与固定预条件分母时更新对g仿射，实际分母依g导致非线性；残差不是历史动量致害或全轨迹Dice交互的独立证明。

所有已保存向量在CPU转float64再聚合，原delta是float32 post−pre参数差。没有逐probe入站参数、m/v/t，无法恢复相减精度或给出可靠量化下界，所有残差标记quantization_bound=NOT_IDENTIFIABLE。不能用终态状态补充，也不为此自动复跑GPU。精确零分母/零向量输出NA并计数，不加epsilon；非零但小的分母直接报告分母和未经精度保证的比值，不据比值排序选层。舍入余弦保留原计算值，不裁剪隐藏。

自检：切分覆盖与分组分区完备、6映射一致、48probe全覆盖、有限值、全BN分量梯度和与完整梯度在原atol2e-6/rtol1e-4一致；简单合成零向量NA/正交余弦/仿射delta四格抵消/能量分区恒等式。不增加仅复述实现的测试。

硬预算：首次CPU分析进程开始T0（导入前设置时钟），所有修复共享T0；最多600秒原始历时，外部timeout TERM于590秒、5秒后KILL，内部570秒停止；CPU线程1，shell/子进程中性argv；输出独立批准NFS根，仅匿名汇总，无私有副本。远端预留128MiB，root写入硬cap64MiB（文件大小逐行增量检查，最后2MiB恢复余量），本地预留128MiB，保守总envelope17.669GB<20GB，各服务器新增<50GB。已有只读输入不复制；不hash、全扫dataset或重复旧du。正常失败保留状态/费用/NA，不覆盖失败文件。

出口：只定位聚合改动敏感组，不能称VV有害层。稳定性看全部seed/order/机械位置的描述，不做显著性、不用一步Dice选组/调尺度。没有充分且可解释依据则交付负诊断并结束层干预分支，不扫描层组合。有充分依据也须另冻最小因果协议、VV和确定性参数规模匹配控制，区分改变层位置与削弱总更新，明确参数/优化器操作和D1 OD/OC保护等门槛后才可启动。R87当前NOT_AVAILABLE、R88条件未满足。仅复用唯一r52，无第二监测。
