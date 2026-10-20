# R91：历史状态 × 固定六弱读出（GPU前冻结）

## 资格、唯一候选及控制

承接用户2026-10-20明确授权与[CPU重复性审计](REUSE_AUDIT.md)。R90 M1已NOT_PASSED，M2关闭；本轮不改少写信号，不改任何VV学习写入。原192暴露开发内容、两既定order、种子20260907/17011/29009、512输入、六弱q/wholeW(.1,2)、原strong及其同实际strong水平flip逆对齐、BN affine19136/currentstat无runningstats、冻结非BN/head、连续Adam1e-4(.9,.999)eps1e-8、BCE+.1方向LSO及分母不变。

唯一候选Qt为原VV本步更新前已有六弱概率均值q，逐头>=.5阈值输出；仍完成原训练并提交全部学习状态和K。原q顺序：normalized原图、水平flip、垂直flip、90/180/270旋转，logits逆变换后CPU stack→sigmoid→mean(dim0)。不改logit/prob顺序，不混Qt/K、不改视图、阈值或按域选输出。

P0为初态currentstat原图，P=原入站PRE，K=原更新后输出。Q0为初态currentstat同六弱读出：确定性变换、同输入/顺序/精度，每内容一次共192，向六anchors广播只作匹配，不声称重复推理。使用同一Host临时初态参数及完整状态恢复，不增加教师模型；无参数更新、无跨图学习。SOURCEONLY running-stat不能代替P0。

新Qt只从原target_cache只读clone，不新增前向。Q0六前向。PRE/K每步必须对迁移同硬件packed reference exact，任何parity失败NOT_EVALUABLE，不能用Dice接近代替。Q0 identity logit须对已留存P0二值exact。

## 资格与不可逆计数

CPU fixture检验六视图变换/逆对齐、重复同视图退化均值/阈值/双头独立、判据算术。GPU fresh qualification两原Host×前三arrival=6更新/60F；一条导出Qt/临时Q0，另一条不导出，PRE/K与完整学习核心状态逐步exact。Qt须等于原q且导出前后完整snapshot exact。Q0第一内容两次、已有历史后对第一内容再一次=18F，重复输出exact，初态Qt==Q0，临时只读后参数/梯度/Adam/counters/RNG/modes/诊断恢复exact。资格合6updates/78F/6BW；无额外反向。独立CPU变换自检验证inverse，无任务评分。

正式静态Q0=192×6=1152F/0updates；主VV=1152updates/11520F，两个worker每个576updates/5760F/576BW。从源空Adam开始6连续轨迹。含资格本轮1158updates/12750F/1158BW；新独立路线硬限1250updates/20000F含失败修复。仍核R90累计1934/19340，加本轮预计3092updates/32090F，受原5000/50000更严共享限。只读前向的计数不随快照恢复，实际attempt先charge、complete另记。

按资格除冷交易外maxsteady×1152×1.5/2，加static max单图秒×192×1.5，再加1800秒CPU及600余量，必须小于原deadline剩余；不满足不删控制/门槛。

## 共享预算与隔离

原保守T0=1792492974.1323855，绝对deadline=1792536174.1323855（北京时间2026-10-21 06:42:54.132385），不因改名/attempt重置；取本轮最多12h与原deadline更早者。原准备首命令精确时间未量，原15min回溯已保留。独立watchdog截止−90TERM/−85KILL仅核身份自有进程组，worker−180合作停；CPUscore独立1740秒alarm及共享期限。

仅my-gpu GPU0/1固定原UUID，启动各>=10GiBfree，allocator8GiB；主子中性argv。所有新文件位于remote-home下原campaign新子根，不复制data/model/env或旧PRE/K。不读取线上labels/sourceimages/未来图像/旧评分。模型guard拒绝scorer和M0衍生评分，允许当前图像及已读图像收尾readonly复核。

新增remote256MiB（含32MiB恢复余量）+local32MiB保守预留；原19,602,653,184加301,989,888=19,904,643,072bytes<20GB。新root实际cap256MiB，同时原campaign全部实际cap1.5GiB继续生效；max(logical,allocated)+目录块统计，预留输出块后才写，定期复核且独立存储watchdog，不能靠心跳。大资源只读，无删除旧结果腾空间。实际新增每服务器严格<50GB。

## 标签屏障与指标

qual/source/formal两worker全部exit0/GPU_COMPLETE且PID/startticks真退休后，CPU只处理预测生成固定M0后处理（8-LCC并列first-raster→4背景填洞→OC∩OD），全部raw与post封存后才读labels。新Qt/Q0存packed二值gzip，不存六视图概率。可复用M0已完成的P0/PRE/K/NATIVE指标（主线parity先成立），保持同内容、metric定义和分母。固定后处理全对照一致，不以其挽救raw主门。

四对比逐内容形成：READ=Qt−P；DEPLOY=Qt−K；HISTORY=Qt−Q0；INTERACTION=(Qt−P)−(Q0−P0)。另报Q0−P0、Qt−NATIVE和全原样/后处理正负单元、内容median/覆盖/FPFN整图分母/前景/真实earlierlater/D1与余域/LODO、结构NA。交互不是唯一BN/Adam机制，数据非独立验证。

## 固定晋级门（无任务评分前冻结）

raw双头四域macro DEPLOY>=+.1pp、HISTORY>=+.3pp同时成立；按域内容bootstrap2000次，六anchor耦合，两主要比较Bonferroni双侧97.5%CI下界均>0。各比较两order均正、各seed两order均值正、至少5/6正anchor。

相对VV K，D1 OD与OC分别均值>=−.2pp；D3OC均值>=+.1pp。相对NATIVE任何域通道跨anchor均值>=−.5pp、任何域通道order跨seed均值>=−2pp。后处理DEPLOY/HISTORY宏均值均>0，不救raw失败。

结构门raw与post都检查：总体四域等权双头ASSD/HD95及vCDR绝对误差均值，及D1/D3逐头ASSD/HD95、各域vCDR误差均值相对K不得恶化；距离容差1e-6输出网格pixel、vCDR容差1e-9，仅数值舍入，不作实质容忍。vCDR按每图一个比值（双头表重复不计独立）；用双方共同finite内容配对，并在每scope要求每metric NA数不增加。每scope/每头空预测数亦不增加。只一方空边界按网格对角惩罚，双空0；OD空vCDR NA沿用M0。通过只表示未观察均值恶化，非临床安全。

任何门未过，完整保留负结果且关闭该固定读出假设，不扫描融合、EMA、教师、q权重或新选择器。全部通过仅开发读出候选，非自动新机制/患者独立确认。192及1951注册内容历史暴露，patient/video/source overlap UNKNOWN，R87 NOT_AVAILABLE/R88条件不变。
