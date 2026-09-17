# R6 外部审阅报告

## 结论：R6_REVIEW_PASS_A_ONLY

本次已审范围内未发现阻断 R6-A 的缺陷。通过的是固定实现的阶段 A 代码/协议审阅，不是效果确认、GPU parity 通过，也不是用户 GPU/正式实验授权。R6-B 不在本次通过范围内。

- Implementation SHA：`c94fff7c05cec38d541c441b62a9381ee46ba076`。
- Evidence publication SHA：`70e48d5da9737f806bd797b5c1752bd5b50a8e20`。
- Science SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`。
- 原数学参考 SHA256：`cbf47ef84616c445e53bc52182ac6848e666b1810ff1dc3ccd712d09e2c93cb6`。
- Scope：`A`，四臂 × orders 0/1/4，共 12 条。
- execution_authorized=false；R6A=NOT_RUN；R6B=NOT_RUN。

这份文件记录本次审阅，不表示已修改远端仓库状态或签发实际执行 receipt。

## 1. 审阅及证据边界

通过 GitHub 只读接口查看固定实现的 loss、host、analyze、plan、execution、入口，原件对应表、验收汇总、原始日志末段及相关测试代码。核对生产公式、logits/payload 生命周期、标量可实现性检查入口、累计调用数、scope 与 A/B 复用和结果发布路径。GitHub compare 显示实现到证据发布的变化只在 docs/review/r6；未观察到发布时变更生产 src/scripts/tests/configs。

提交方两端完整套件各166/166通过，是读取并交叉核对过的原始日志证据，**不是本审阅者重跑166项**。本审阅者未连接用户服务器，未运行真实 GraTa/ResUNet host、GPU smoke、实际 checkpoint 或私有目标记录；未独立重新提取/逐一重验全部11项归档。已核对其原件对应报告，另外独立验证科学原件SHA256及其Git blob，与固定实现config的Git blob一致。

本次实际执行的是下述五组有界 CPU 函数级探针。环境 Python 3.13.5、Torch 2.10.0+cpu；CUDA未初始化、网络forward=0、模型/optimizer调用=0、真实数据/给定checkpoint读取=0。autograd只用于程序化logit张量的验收，不进入生产路径。

完整 loss.py 本地副本的Git blob为 `67f8e6e5d946c7d31ef43529f01b1029d40fd9ab`，与接口返回值一致。analyze中的audit_channel、gate仅作明确标注的函数摘录；它们不是完整模块或完整结果重算。原数学参考由只读导入执行函数，不以脚本入口运行，因此未覆盖历史数学日志。

## 2. 科学实现核对

### 区域权重与目标

R_BAL从实际q>=0.5形成权重分区。前景/背景均非空时rho=clip(n_bg/n_fg,1/8,8)，w_bg=N/(rho*n_fg+n_bg)，w_fg=rho*w_bg；一方为空时整通道权重为1。通道平均权重在实数算术中为1，应用float32舍入值另外记录。

权重乘普通BCE的完整正负两项，不是pos_weight；q仍为原多视图软目标。对于固定正权重，每个独立logit的驻点仍为sigmoid(z)=q。该数学事实不保证有限参数模型的适配性能。

### 控制臂

残差先在loss dtype计算，再CPU float64归约；Sw、Sperm使用实际float32权重提升后的数值。R_SCALE/R_SHUFFLE倍率不参与反向传播。SHUFFLE在每通道全部位置置乱，只改变权重位置；专用CPU Generator的seed由固定ASCII编码及visit/channel确定，不消耗全局增强RNG。

匹配对象仅为同一局部z/q上的逐通道strong-output logit梯度范数，不能改称BN参数梯度、Adam位移或跨臂轨迹匹配。

### 已披露但不阻断的差异

R_SCALE生产用“通道mean BCE后乘a”，原数学参考用“逐像素乘a后全局mean”。实数公式一致，float32加法顺序可能不同。本次168共同输入比较最大loss绝对差1.1920928955078125e-7，最大逐像素梯度绝对差4.656612873077393e-10，符合原冻结容差；C本身仍逐值一致。不为追求SCALE bitwise一致而修改生产或放宽容差。

smoke配方名称有已披露别名，实际顺序OLD/C/R_BAL/R_SCALE/R_SHUFFLE、各4步、seed/像素索引/配额不变。执行receipt必须用代码检查的 `R6_ALL_ARMS4_OLD_C4_V1`。

低层weights接受requires_grad的q并仅detach权重构造，与参考的非法输入拒绝范围不同；生产固定GraTa接口提供no_grad q。这个边界已披露。本次通过限于该生产入口，不承诺任意外部q输入API行为等价。

## 3. 本次独立探针结果

| 检查组 | 实际范围 | 结果 |
|---|---|---|
| 权重代数 | 总像素数1..128的所有前景计数，共8,384种分区；检查mean、正性、限幅、空分区 | 通过 |
| 原参考与生产loss/autograd | 两dtype、7种分区、3个visit、4臂，共168组 | 通过 |
| 完整512栅格运行时标量审计 | 10类float32输入×4臂×2通道，共80次audit；含单像素前景、零/微小残差、饱和和恒定残差 | 通过 |
| 驻点/零能量/非法输入 | 12个病例；包含实际autograd及非有限/越界拒绝 | 通过 |
| 原JSON与生产gate函数 | 1,016组A/B边界及随机有限差值，另3个非法输入拒绝 | 通过 |

完整栅格控制臂相对BAL的局部梯度范数最大相对误差为 3.58202747277261e-08。这些是函数级代数/实现证据，不是完整轨迹或GPU数值结果。五组不能与提交方166项相加成新的“171项完整套件”。

探针源码与JSON/log均附在证据包。gate探针的passed仅表示测试通过，不代表R6-A科学晋级。

## 4. 提交方测试证据核对

| 项目 | 本地 | 服务器 |
|---|---|---|
| 最终完整套件 | 166/166，fail/error/skip均0，exit0 | 166/166，fail/error/skip均0，exit0 |
| 环境 | Python3.12.9 / Torch2.6.0 | Python3.10.6 / Torch2.2.1+cu121 |
| 套件记录时间 | 432.382483秒 | 1417.386201秒 |
| CUDA / 真实RGB、mask、给定checkpoint | 未初始化 / 读取0 | 未初始化 / 读取0 |

两端另有原数学参考8项新运行与3项共同输入/协议比较；历史数学8项与本次新运行没有混写。旧BLOCKED状态及失败日志保留。suite预置jacobian_vjp_calls=0不能代表所有继承回归的VJP总数；ORIGINALS_ACCEPTANCE已经明确该总量未知，不据此声称完整166项总VJP=0。

## 5. 生命周期、审计与阶段隔离

Host保持原C调用链，在原criterion接入新loss；pre来自已有原图弱前向，q为实际criterion target，post为最后原图输出。强logit只读gradient hook验证范数并返回原梯度；单图hook在结束/异常时移除。参数、Adam、RNG提交后才移出payload并读取GT；实际每图仍8F/1B/1Adam/0VJP。零梯度仍调用Adam，不能把历史动量产生的位移误判成错误。

analyzer保留R5局部指标可实现性检查；重算分区、权重、归约、scale、seed、当前q评价前景数量以及单/跨轨迹GT计数。它不从标量重建空间位置、梯度或ASSD几何；runtime证据与CPU关系检查不能混为完全独立图像复算。

已读失败路径使用live计数并保留观测下界说明，首失败与原异常不主动覆盖。scope只可A/B_NEW，AB仅元数据；默认disabled；A完整12条后才有科学gate；B要求完整合格且可重验的A12条和新授权；下一阶段执行授权始终false。

## 6. R6-A启动边界

只进入启动准备，不改科学配方、不追加测试矩阵。实际运行另需用户对本次A明确授权，并绑定实现SHA、science摘要、production fingerprint、登记/流摘要、设备、worker数、caps及新私有输出。

- 四臂：C/R_BAL/R_SCALE/R_SHUFFLE。
- 三流：order0/order1/order4；每条1951内容，全12条完成。
- 正式：23,412访问、187,296F、23,412B/Adam、0VJP。
- 每实际设备smoke：160F、20B/Adam、0VJP；只要求旧C与新C parity，不要求新臂等于C。
- A gate保持mean差+.50/+.20/+.20pp，每主序三个配对均>=0，recurrence三个配对均>=-.10，每域两主序BAL-C>=-2。
- A无论是否晋级均停止；B_NEW8条不在本次审阅通过范围内。

结束状态只可能为：机械有效且通过gate `R6A_COMPLETE_ELIGIBLE_FOR_REVIEW`；机械有效但未通过 `R6A_COMPLETE_NO_ADVANCE`；机械/身份/计数/核验失效 `INCOMPLETE`。不调cap、不换seed、不自动retry、不直接进入B。

## 7. 来源索引（均固定提交）

- 审阅入口： https://github.com/DLwbm123/DPA-CTTA/blob/70e48d5da9737f806bd797b5c1752bd5b50a8e20/docs/review/r6/REVIEW_INDEX.md
- 原件对应： https://github.com/DLwbm123/DPA-CTTA/blob/70e48d5da9737f806bd797b5c1752bd5b50a8e20/docs/review/r6/CORRESPONDENCE.md
- 验收汇总： https://github.com/DLwbm123/DPA-CTTA/blob/70e48d5da9737f806bd797b5c1752bd5b50a8e20/docs/review/r6/ORIGINALS_ACCEPTANCE.json
- 本地原始日志： https://github.com/DLwbm123/DPA-CTTA/blob/70e48d5da9737f806bd797b5c1752bd5b50a8e20/docs/review/r6/logs/originals-redelivery/suite-local.log
- 服务器原始日志： https://github.com/DLwbm123/DPA-CTTA/blob/70e48d5da9737f806bd797b5c1752bd5b50a8e20/docs/review/r6/logs/originals-redelivery/suite-server.log
- 源码目录： https://github.com/DLwbm123/DPA-CTTA/tree/c94fff7c05cec38d541c441b62a9381ee46ba076/src/dpa_ctta/r6_regional_consistency
- 配置： https://github.com/DLwbm123/DPA-CTTA/blob/c94fff7c05cec38d541c441b62a9381ee46ba076/configs/r6_science_v1.json
- 完整host测试： https://github.com/DLwbm123/DPA-CTTA/blob/c94fff7c05cec38d541c441b62a9381ee46ba076/tests/test_r6_host.py

本审阅没有推送代码、修改远端状态、登录用户服务器或启动实际实验。
