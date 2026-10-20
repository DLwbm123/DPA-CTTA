# R85/R86：BCE 与方向 LSO 聚合位置的冻结二因子实验

状态：PREPARED；资格与正式结果尚未产生。用户于 2026-10-20 授权按修正后的 Pro 计划继续，现行全局授权允许持续自主研究。旧 R75–R84 终态不覆盖、不重跑。

## 唯一科学改动

原两个 strong 视图：第一次原有 strong 经 native normalize 后，第二次取同一实际输入的水平 flip，再逆 flip 对齐 logits。保留原六弱当前 q、完整当前 W(variance_temperature=.1,boundary_boost=2)、原 BCE、方向 LSO 的 .1 系数、BN affine 19136/current-stat/no-running-stat、非 BN/head 冻结、batch1、连续 Adam(lr=1e-4,betas=.9/.999,eps=1e-8)。原 content-keyed SeedSequence([seed,ordinal,79])；种子 20260907/17011/29009 和两既定顺序、192 开发内容不变。没有 teacher、模块、EMA、输出平均、q 混合或新增强。

令 B_V/S_V 为两视图各自 BCE/LSO 的均值，B_B/S_B 为对齐 logits 均值上的 BCE/LSO。VV=B_V+.1S_V，VB=B_V+.1S_B，BV=B_B+.1S_V，BB=B_B+.1S_B。VV/BB 保留旧 VERTEX/BARY 的浮点运算顺序，VB 为预设主要候选，BV 为反向对照。不扫描系数或配对。

## 先验资格与 R85

CPU 合成检查：同视图退化、固定 W 的 BCE Jensen 非负和 q 显式抵消、四格梯度恒等式、LSO 通道贡献和原损失的一致性、H/C/T 代数。CPU 不初始化 CUDA。

fresh GPU 资格：四格各连续 2 arrival，新 host 和独立 host 各更新一次，共 16 物理更新；VV/BB 对旧原实现，VB/BV 对独立直接原公式 oracle。完整核心 state（参数/梯度/BN/buffer/Adam年龄及矩/RNG）和 output exact；PRE 全状态恢复 exact；当前 q/wholeW 来源 exact；每 step 9F、一次训练 backward/Adam；非 BN 冻结。VV 两个 arrival 额外各 8 次诊断 autograd，须不改变核心更新。SOURCE_ONLY 对前两项各两次静态推理 exact/frozen，资格共 156F、16 training backward、16 diagnostic backward。

独立公式同运算顺序的资格用 exact。独立梯度代数容差预设 atol=2e-6,rtol=1e-4；梯度四格最大残差阈值 2e-6+1e-4*四格最大绝对分量。不得因失败放宽阈值。Adam 实际 delta 非线性，不要求满足梯度线性恒等式。

R85 重演原 VV 六条完整连续轨迹。每个 order 每域按该域在流中的到达次序取 floor((n-1)*.25) 和 floor((n-1)*.75) 两位置；4域×2位置×3seed×2order=48 inbound 状态。选择只用离线域元数据、无标签/结果，worker 仅拿整型位置、不拿 domain/router。每处 snapshot 全状态，依次 VB/BV/BB 丢弃更新后恢复，VV 最后提交；共享相同 q/W/strong 和 inbound BN/Adam/RNG。实际 1152 VV +144 discarded=1296 physical updates。重演每 arrival PRE/K 的 packed masks 对旧 R79 main 的 PRE/K exact；旧 U 不复跑。工程 parity hash 只对这些必要预测块，非全资产扫描。

记录四格实际 BN 梯度/Adam delta 的向量私有证据和匿名范数/余弦、D_B/D_S、虚拟 K/PRE。仅 VV 的每个探针做 B_V/B_B/S_V/S_B × OD/OC 的 8 次 autograd，384 次全计量。LSO 两通道贡献保持原全通道联合分母；BCE 保持原 numel 全局分母。分量梯度相加核对四格梯度，诊断 hook 计数隔离但物理 backward 独立累计，不当免费。固定状态梯度解释不外推整条轨迹。

## SOURCE_ONLY 与 R86

SOURCE_ONLY 原 source eval/stored-running-stat/no augmentation，原 checkpoint 和相同 model_input_from_pixels，192 额外静态 forward，无训练。另记录 P0 current-stat，二者不混称。旧 R30 全域 aggregate 不直接当当前 192 基线。

R86 仅新增 VB/BV 各 3seed×2orders×192=2304 updates，12 新连续轨迹，PRE/K4608 predictions。每条从共同 S0、空 Adam 启动；不从 VV 中途状态分叉。复用已交付 VV/BB/NATIVE original 的同内容18端点轨迹/6912 PREK预测；新增 R85 replay PREK2304、探针虚拟K192、SOURCE_ONLY192单列。旧 values 坐标校验，R85 mask parity 必须通过，不以旧负候选替代原基线。

合计资格16 +R85 1296 +R86 2304=3616 新物理更新；training backward3616，诊断400，physical forwards=36204（资格156+source192+R85 12816+R86 23040）。每次准备、失败、资格、训练、评分的进程时间、低成本分析时间分列，worker body 是进程子集不双算。无 GPU-hour 硬限；规划不等于实测。

全部本轮 GPU 进程真实 exit0/GPU_COMPLETE/PID退休后，CPU scorer 才读取 mask/旧 values。在线标签及 source image read=0。H=PRE−P0，C=K−PRE，T=H+C。全轨迹四格差分为算法效果，不当同 H 纯梯度因果。

主指标：域宏平均 OD/OC Dice；全域/通道/seed/order/负单元、imageweighted、内容 median/覆盖、FP/FN/foreground（整图分母）、真实 paired earlier/later、最差 anchor/seed/order；D1贡献、其余三域及 leave-one-domain-out 为描述性，不更改主宏指标。48探针按匿名编号和域报告虚拟当前收益，不以一步 Dice 为晋级条件。不追加未经冻结的临床边界指标。

预设晋级：相对 VV macro>=+.1pp，内容bootstrap区间下界>0；D1OC>=−.2pp；相对 NATIVE 任域×通道平均>=−.5pp，任域×通道×order跨seed平均>=−2pp；两个 order macro 均正、至少5/6 anchor macro正、每seed两order平均正。两个候选对 VV 使用同 seed 的 2000 次按域 content bootstrap、六 anchors 耦合；Bonferroni 两个双侧97.5%区间为晋级判据，旧95%另呈现。选择优先级 VB 通过则 VB，否则 BV 通过才 BV，否则无候选升级。内容CI不是seed总体或患者独立CI。

## R87/R88 前置审计与条件

旧1951内容（SEARCH1017/legacyREVIEW678/CONTEXT256）已经历史暴露；剩余于192之外不等于 fresh。现有可访问注册 manifest 的身份与历史暴露审计，不按模型结果筛选。患者/ROI/源域重叠未知则明确 UNKNOWN，不能证明患者独立。

只有确认未暴露内容存在，才另冻结 R87：每域最多48个独立单元、不得重复填充；5新种子和两固定顺序，NATIVE/VV及最多一个预选候选，SOURCE_ONLY静态/当前P0分开。新种子只能称随机性复核，不能替代 fresh cohort。若无合格新数据记 NOT_AVAILABLE，不自动对开发192换seed冒充确认。

R87 若实施，VV−NATIVE macro>=+.3pp、候选−VV>=+.1pp且CI下界>0、保留域伤害门槛、>=8/10正anchors/每seed均正/两order均正；是否主张超过SOURCE_ONLY必须事前纳入多重比较。R88 仅在有效R87通过后比较 continuous/episodic reset，五seed、每内容重置 S0/空Adam，无 order 重复；不能用旧 R79 开发 reset 当未暴露证明。

## 资源与停止

R85/R86 属同一个实验，第一次 CPU 准备开始 T0，资格/重试/子阶段共享 24h 绝对截止，不拆名续命。后台独立 watchdog 在deadline−90秒开始终止，仅身份匹配的自有 process groups，5秒后强杀未退，controller/SSH死亡不影响；worker 在deadline−180秒合作停止，controller不能在剩余不足180秒时新开子任务。新根独立2.2GB reservation、实际写入 cap2GiB、保留128MiB终止余量，整个root含source/private/failed/log/scorer；已有数据/model/env只读复用。保守继续旧累计20GB空间账，同时满足每实验每服务器<50GB；不删除失败/旧结果腾空间。

固定GPU0/1 UUID，启动>=10GiBfree、allocator8GiB，不动其他任务、不换卡。CPU/主/子/watchdog均中性可见入口/argv。watchdog、注册startticks、所有费用可审计。进程超时、存储、非有限、访问边界、parity、资格失败都保留 NOT_EVALUABLE，不当科学负结果。工程修复另attempt保留成本及原T0；性能失败不重试或改门槛。

按现行持续自主授权复用唯一 r52 半小时跟进；健康安静。完成后公布全部匿名代码/结果/中文报告/费用/负结果/失败证据，代理push并验证remoteSHA及匿名报告，不发布图像、标签、身份、权重、原始私有日志、凭证或私有路径。之后依据证据选择合理独立阶段，不因一轮完成机械暂停。
