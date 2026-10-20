# ColorSharp-BC-v1：真实读取前完整预注册

状态：协议冻结，执行source/新root尚无，未执行、未获模型工程资格。此文固定科学设置；下一先实现并commit执行source，再工程准入。不是通过合成就允许任意真实模型试跑。只同8内容、原模型、既有seed20261018；旧192内容、BN尺度/Calibration Maps、crop/800/fCE/gamma组合保持关闭。

## 一主假设与信息限制

唯一候选：冻结模型在学习式Color/Sharpness难视图上的SOURCE-vs-B logit支持，比同一模型原始视图的支持更能辨认既有B→S撤回效用。TeSLA的学习式难视图提供相关机制依据，数值A仅给实现证据；本假设不是原论文已证明的块级语义结论，不是完整TeSLA，不新增真实语义观测。

不是GT风险估计：模型仍可能确认错误SOURCE，熵增大/策略参数移动不等于正确。CLEAN和旧LOGIT_Q/GEOMETRY是必须战胜的同信息强控，不能只胜弱随机便推进。

## 数据、空间和预处理

仅原8登记RGB×原4外观32输入：PIL RGB bicubic→512、float32/255，identity/gamma.7/RGB(1.15,1,.85)/7×7 reflect Gaussian sigma1，按原顺序clamp后每图全通道min-max。恒定span真实输入硬停。不得重抽样、读其它图/源训练样本或按条件另设策略。GT在所有生成成功退休前不开放。

S/B固定my-gpu迁移E8 SOURCE/TOPO_SMOOTH位图；OD锁B，OC-only3028登记，legacy共同2881块/6997编辑像素及147非共同完整保留，不创造新ROI/候选空间。重新核block_id、query/size/direction/distance配额与原登记。每个干净原模型输出阈值>=0须逐位SOURCE核对；不把迁移parity说成原旧server封存parity。原模型调用只此512输入契约。

## 学习式光度策略：一次固定

两操作顺序Color、Sharpness。沿A已冻定义：Color灰度(.2989,.5870,.1140)，Sharpness复制边界3×3核[[1,1,1],[1,5,1],[1,1,1]]/13；v=.1+1.8sigmoid(m)，T=clamp(base+v*(x-base),0,1)，随后全图min-max。无空间变换、几何逆映射或重新调相位/边界参数。非几何意义是坐标未重排，不声称语义严格不变。

每输入独立重置a=(0,0)、m=(0,0)，20次Adam更新；概率softmax只对两操作，绝不对OD/OC。学习率a .001/m .01，betas(.9,.999)、epsilon1e−8、无权衰减。无跨内容policy记忆，不称原TeSLA持续策略复现。

原模型eval、全部参数requires_grad=false、无模型optimizer/权重或BN更新。原始输入clean前向固定p0=独立两头sigmoid；参考detach，不以B/S/块/条件标签参与策略学习。对全部现有BatchNorm2d输入空间均值作hook，记录标准eval模式/41层预期；仅测量，不更新统计。每个增强前向得到p_j及各层均值；D_j为与clean均值的层内通道MSE再跨层平均。

策略最大化J=sum_j softmax(a)_j * [mean(H_Bernoulli(p_j))+mean(KL_Bernoulli(p0||p_j))−D_j]。熵/KL概率统一clamp[1e−6,1−1e−6]，系数1；两头与像素等权。精确两操作期望，不采样REINFORCE；固定clean参考，不用顺序avg。与A不同的是解析预测器/RGB均值替换为原模型/全部BN输入均值，故A不提供这部分工程证书。

政策变量float64，输入/模型/卷积核float32，模型输出转float64计算熵/KL和最终读出；显式记录混精度，不将A float64差分误差套给真实网络。真实model对policy计算输入梯度，无模型参数梯度；所有非finite、状态变化或未接通的策略梯度硬停。允许有限步里某幅度梯度为0，不按幅度变化量挑输入或救分。

## 只读原模型工程门G0与成本准入

原8RGB登记摘要/模型/代码按既有完整性要求核验，不扩大数据读取。每worker独立加载同原权重，固定既有seed20261018、deterministic/关闭TF32/CUBLAS工作区配置；只GPU1指定UUID，CUDA_VISIBLE_DEVICES映射后自身核UUID与nvidia中性argv。全部模型state含BN缓冲逐位前后相等、模型参数grad始终None、41层eval/track_running_stats=true、无新模块/任务头/训练。允许输入图的autograd，不说无任何反向。

每输入1 clean +20×2学习 +2 final +2 frozen=45完整encoder，每worker32输入1440，双进程计划2880完整encoder。原模型/优化器都不更新；策略每输入20步只4变量。不能把encoder数当GPU秒，也不能遗漏反向成本。

第一worker首输入完成后，以其完整用时×剩余输入数×两遍余量1.5，加已耗时/预留CPU180秒必须能落入原3600秒总线；否则保存WAITING/工程未准入并结束，不缩steps/输入/精度、换卡或重置clock。该首输入属于正式worker，不另反复预跑。独立第二worker重新读bytes/解码/学习，全部32归一化输入、clean/final/frozen logits、policy轨迹、支持/读出/影子mask exact；源位图共64次核。若跨进程精确重演失败，不读GT，不调容差/换精度救门。工程失败不叫信号阴性。

## 冻结支持与读出

对每输入学习后两final logits zColor,zSharp，终点权重w；zP=sum_j w_j*z_j，zU=(zColor+zSharp)/2。两frozen logits在初始m=0（v=1）生成，以终点w相加为zF；浮点运算可能不逐位等于clean，实际差保留，不谎称identity端点exact。zC=clean。随机对照zR在同终点两logits中均匀选1，local numpy default_rng(20261018)按固定case/condition顺序生成32个选择，两个worker重置同序，不新增seed或多次挑最好；不多调用模型。zShift=zC+roll(zP−zC,257,width)，只移位响应场，不移动图像/mask。

共同块k的编辑指示d=S_OC−B_OC，仅xor像素取平均：q_z(k)=mean(d*z_OC)。这是独立Bernoulli下候选相对B的平均loglikelihood差，正分固定支持B→S；不按结果翻向/温度/平均方式。不是无偏utility、真实深度或解剖置信。

U固定为共同块xor像素至少一处|zP_OC−zC_OC|>1e−6。该阈值仅定义可测模型响应支持，不是解剖阈值，无GT或预算，所有臂U外q=0平分。它可能很宽；上界PASS只排除空间不足，不保证合法可达或信息有效。

8臂固定顺序POLICY/CLEAN/UNIFORM/FROZEN/RANDOM_VIEW/SHIFT/LOGIT_Q/GEOMETRY_ONLY。后两列仅独立sealer从旧无GT worker缓存白名单导出，不读旧效用/结果；当前CLEAN由新前向直接重算，不假称与旧RAW_MARGIN公式完全同一。所有臂同U/原2881完整分母。

SHIFT非退化：响应场(delta/roll delta)的|.|>1e−12并集上数值差>1e−12比例>=.5；qP/qShift非零并集上差>1e−12比例>=.5；空分母失败。另报joint mask实际变化数，不将数值差冒称收益。

## GT前预封存联合动作与公平配额

固定POLICY在同U/q>1e−12且单块撤回后OC非空、位于B OD内、8连通1分量/无4连通孔的块上全部撤回；全部合装后不合法则整帧回退B。无top-k/10%或方向窗口搜索。

7控制按POLICY最终实际配额(exact size,旧direction,distance_bucket)同U、单块合法内取本分数最高，block_id升序破同分；不要求控制正分。联合同样合法性回退；若回退后配额不等，匹配门失败，不称严格匹配。SOURCE/B完整强基线必报。worker完全看不到GT/效用/旧结果/第三方网络。

## 评分门顺序：G0→B1→B2→C

全部生成worker OS0/精确身份退休/重演PASS后独立CPU scorer才核8GT原登记摘要/nearest512，OD<255、OC==0。GT不能回流到生成/支持/选择/协议。评分使用原[2,1,1,1]/5条件权重、8内容等权，bootstrap2000、seed20261018；8反复开发内容不代表独立患者。

B1先且只算共同全集与U的联合放宽最优OC Dice上界，Fraction比值优化并从实际64 oracle mask核，不相加单块utility。共同全集+.4721342181562814pp容差1e−12和B四条件87.80529246678334%容差1e−8须重现；U上界均值>=+.30pp才过。忽略拓扑/嵌套，标RELAXED_UPPER_BOUND。失败B2/C NOT_RUN_SPACE_GATE，不构造单块utility/AUC、不评价影子收益/子组；不按CI上端过门。

B2仅B1过：原2881块精确单块B→S效用；零效用保计数但不成正负对，无正/负帧AUC=NA，不支持分数0保平分。唯一主POLICY−CLEAN BLOCK AUC>=.05且内容配对CI下界>0；POLICY AUC>=.60；相对UNIFORM/FROZEN/RANDOM_VIEW/SHIFT/LOGIT_Q/GEOMETRY_ONLY各BLOCK差CI下界>0；可定义输入>=24/32且8内容全；|utility|配对加权POLICY−CLEAN差>=0；SHIFT双非退化门均过。合取失败C NOT_RUN_SIGNAL_GATE，不以任何辅助格/权重/条件换主。

C仅前门全过：只评价GT前封存联合mask。POLICY−B OC>=+.30pp且内容CI下界>0；与SOURCE完整比较，但不把继承B收益算模块收益。相对全部7配额匹配控制OC差CI下界>0且配额全等；OD逐位B、不新增任何帧头空mask/嵌套、两头表面64/64和vCDR32/32定义，任一帧/头Dice下降<=1pp；OC ASSD/HD95相对B内容加权均值之比bootstrap97.5%上分位<=1.05；vCDR误差增量CI上端<=.005。所有分头Dice/ASSD/HD95/vCDR/NA/空/嵌套/分母完整报，不称临床安全。

## 资源、完成与停止

一个新root先NFS基线0/容量/小写读，预计峰值768MiB、预留1280MiB、输出hard1024MiB含code/cache/log/replay，不另复制RGB/权重。总clock<=3600秒，独立guard3540保存停、controller stage截止3530，原时钟从首次实际执行起修复/重演共享；<50GB/24h及项目20/18GB/aux1GB不变。初始GPU1指定UUID可用显存>=8192MiB否则WAITING_GPU不启动GPU、不换卡、不干预别人；只my-gpu、不查旧server。

所有父子中性argv核ps，GPU自身nvidia/UUID准入。final完整4stage check→worker1→worker2→score退出/身份退休、guard退出、ledger访问重演覆盖/空间实际齐才完成；父OS退出未知null。完整GPU进程含导入/CPU准备、CPU/未计时阅读编排等分别记，不把body重复加或未知造0。

执行source须在真实读取前另commit冻结，本协议不声称该source已存在。任何科学失败关闭此固定配置，不改支持/q方向/操作/步数/参数/预算/主比/条件救分；纯工程修复记录故障且共clock，不能因此读GT或降低资格。完成失败匿名中文REPORT/AUDIT、自有源码聚合无路径回执经代理非force push核账号remoteSHA/匿名200；真实RGB/GT/权重/bits/features/logits/probs/坐标/逐块内容/身份/路径/第三方全文禁止公开。唯一ctta30分钟，不逐轮问，本轮只有协议冻结、无科学执行。
