# R93：域边界历史隔离的三臂离线机制筛查

**须R92 GO、fresh资格和profile全部通过才运行正式。** 当前192流每域只有一个连续块，因此S每域bank在本流严格等价于域边界恢复初态affine+空Adam。不是状态检索试验，不是domain-agnostic CTTA，不是S能作为所有路由方法上界。不恢复Qt、q混合、EMA/teacher或R90少写信号。

## 固定三臂和状态

唯一seed20260907、原两order、192既有内容，V/S/J每order各独立源初始化，3×2×192=1152正式更新（每次PRE+原VV9F=10F）/11520F。V原全流共享VV并逐PRE/K对同硬件旧mask exact，不复制V预测。S根据既有域metadata CPU预构造四个匿名bank整数；J同四bank、同最终22/57/57/56更新数，canonical内容ordinal由np.random.default_rng(93)一次permute后按四quota分配，跨order同一内容同bank，不依据图像/标签/结果或扫描随机划分。93是固定伪分组随机键，不新增模型seed。

GPU只读冻结当前索引的bank整数，不读域名/target labels/旧评分/未来图像。使用域metadata定义的干预只作offline机制诊断，不是部署输入，不能依据结果训练路由器。两order每一步bank localage不同于J，但终态计数和全流localage multiset相同；审计中明确披露，不能唯一隔离域语义。

Host复用当前VV原19136BN affine/currentstat无runningstats，head/nonBN冻结，六弱currentq/wholeW(.1,2)、原strong及同实际strong flip逆对齐、原BCE+.1orientationLSO分母、continuous per-bank Adam1e-4(.9,.999)eps1e-8不变。所有臂交付原图K。bank状态含affine/梯度/Adam m/v/step/GraTa optimizer及完整学习缓存；切bank前恢复指定bank学习状态，global visits/native counters/外部RNG/native RNG取当前全局值，外部physical charges绝不回退。原strong content-keyed SeedSequence([20260907,ordinal,79])不受bank切换改变。首次bank为空源态，bank恢复可在CPU RAM缓存，不将模型/数据/状态大副本写磁盘。

## fresh资格、预算与运行

资格9updates/90F/9BW：原V前2arrival存state/output，再恢复初态重演前2（4updates），对同硬件PRE/K和fullsnapshot exact；两个原Host机械bank序列0/1/0（3updates）与独立bank0两步参照（2updates），对同当前外生RNG/visits/counters下完整state/output exact，验证bank ages2/1、globalvisits3、冻结/一次BW+Adam/onePRE计数。机械bank序列仅测试切换，不作为新科学候选。CPU merge fixture确认所有学习状态按bank、globalRNG/visits/counters不回退；路由fixture检查两order quota/localage histogram exact、bankage差异；评分合成正负门PASS。资格失败NOT_EVALUABLE保留场景，不放宽。

含资格计划1161updates/11610F；原共享3092/32090不清零，预期4253updates/43700F，仍受5000/50000硬限。新本轮限1250updates/12500F含失败修复。formal每worker576updates/5760F，所有attempt累计先核账才修复，不自动重新完整跑。profile maxsteady×1152×1.5/2+2400秒须小于原剩余时间。

保留T0=1792492974.1323855、deadline=1792536174.1323855（CST2026-10-21 06:42:54.132385），12h共享，不改名重置。独立deadline/storagewatchdog在−90TERM/−85KILL，仅startticks匹配自有组；worker−180停。GPU只my-gpu0/1固定原UUID，每卡启动≥10GiBfree、8GiBallocator，主子中性argv，ps/nvidia-smi确认，不动他人。CPUscore独立1740秒alarm及共享deadline。

R92后总预留19,938,197,504，R93新增remote32MiB（含4MiB恢复）+local8MiB，共19,980,140,544<20GB。新root实际cap32MiB且原campaign1.5GiB同时约束；只存S/J packed gzip、必要计数/小表，V旧参照只读。保持每服务器新增实际<50GB、所有大文件remote-home、不复制model/data/env、不删旧/失败、不默认hash/全扫dataset/重复du。独立watchdog与写前余量检查，半小时心跳不当hardguard。

## 标签屏障、指标与退出

qual+formal0+formal1三个GPUworker真实exit0/GPU_COMPLETE/身份退休后，CPU先封存4条S/J raw及固定M0 post，再读既有192标签。V指标复用M0先须同硬件mask exact；NATIVE同seed两order复用匹配M0。保留PRE/K、raw/post、各域头/顺序、整图FPFN/前景/ASSD/HD95/vCDR/空头NA。新结构/评分不重裁旧R91。

两主比较S_K−V_K和S_K−J_K：raw四域双头macro均>=+.1pp，两order均正，2000按域内容bootstrap、两anchor耦合，两比较Bonferroni双侧97.5%CI下界均>0。不是三seed六anchor推进门，不宣称独立确认。

S相对V D3OC均值>=+.3pp，D1OD及OC各>=−.2pp；S相对同seedNATIVE任何域头均值>=−.5pp、任域头order>=−2pp。post两主macro均正。raw/post结构总体及D1/D3沿R91同scope/分母/共同finite/容差（距离1e-6px/vCDR1e-9）、NA和空头均不恶化。

任一门失败关闭这种边界隔离干预，不调分组/边界/seed/阈值救结果；S只优于V但不优于J不能支持域相关特异性；全部通过也只允许另行完整复核，不自动开发路由器/训练专家/扩大矩阵。当前及1951内容全历史暴露，patient/video/source overlap UNKNOWN，R87不可用/R88条件未满足。
