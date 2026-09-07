# DPA-CTTA：面向可比性能的研究收敛与实验计划
日期：2026-09-07
状态：方法修改建议；不是已实现、已训练或已达到 SOTA 的声明。

## 1. 本次决策

保留 DLwbm123/DPA-CTTA 仓库及本地已审计提交，不再新建仓库、不删除旧方案。
旧的 Distilled Potential Atlas 保存为 potential_atlas 研究对照；暂不把自由 chart 参数、状态回归、
闭式 proximal update 和 contraction 作为冲击性能的默认主线。

优先主线：
已验证的医学 CTTA 宿主
→ 可选医学损失
→ 已知标注的源代理数据提供适配监督
→ 只有代理监督有价值才做 dataset distillation
→ 检验“面向适配结果的蒸馏”是否优于普通 DD。

核心研究假设：
小型图像—掩码集合能否保留单图在线适配真正需要的区域/边界修正信息，
在目标外观条件下提供比纯置信度自训练更有用的更新？

不是以下未经证明的结论：
- 任意合成图像有梯度，就说明 DD 有贡献；
- 稳定/收缩的 latent state，就说明分割正确；
- 达到一个 source-proxy 阈值，就说明目标上 SOTA；
- 当前有害更新比例为零，就应人为制造有害样本以“通过”门。

## 2. 当前证据及边界

A. 已提供的复现报告：
VPTTA 已覆盖 Fundus/RIM_ONE_r3 与 Polyp/BKAI 的完整 source-A 流。
报告的 legacy DSC 分别约 73.910316 与 81.000943。
Polyp 同一结果的 standard nearest-GT DSC 约 82.245766，二者不可混用。
这些是起点，不是截至今天所有方法的统一排行榜。

B. CRISP 的有限方向/半径诊断：
已有约 0.2 DSC point 的 supervised-direction headroom 是特定 basis、注入点、
方向和半径下测得的值，不是任何低维适配模型的全局最优上界。
不重复 proxy、inverse、router rescue。

C. 当前 DPA 仓库：
本次 GitHub 读取显示 public、size=0、branches=[]、commits API 返回 empty repository。
尚不能审查本地未推送源码，不能签发 CODE_REVIEW_PASS。

D. 当前上传计划：
包含 multi-scale FiLM、synthetic anchors、自由 m/B/precision、对角 proximal state update。
其数学可实现，但 DD 的不可替代性、状态跨患者迁移和源外推仍待证。

## 3. 必须补齐的最近邻研究

[1] DO-ALL，用户提供的 arXiv:2606.20196v2，2026-06-30。
    静态蒸馏锚点 + replay/MixUp/MMD/blending；原文实验是分类 CTTA。
    不据此宣称医学分割已经有效。Table 6 同时给出不小的计算/存储开销。
    https://arxiv.org/abs/2606.20196

[2] Kang et al., Leveraging Proxy of Training Data for Test-Time Adaptation, ICML 2023.
    已有 style-normalized condensation + test-style injection + supervised proxy adaptation。
    “把测试风格注入凝缩源图像”不是本项目新贡献。
    https://proceedings.mlr.press/v202/kang23a.html

[3] Chen et al., Each Test Image Deserves A Specific Prompt, CVPR 2024 (VPTTA).
    成熟宿主和已验证工程起点，不作为新增创新。
    https://arxiv.org/abs/2311.18363

[4] Chen et al., Gradient Alignment Improves Test-Time Adaptation for Medical Image Segmentation,
    AAAI 2025 (GraTa).
    与任务更新方向相关，应纳入公平对比。
    https://ojs.aaai.org/index.php/AAAI/article/view/32244

[5] Liu et al., Efficient Deformable Convolutional Prompt ... , AAAI 2025.
    医学 prompt 方法近邻，避免把“加一个轻量适配器”当创新。
    https://ojs.aaai.org/index.php/AAAI/article/view/32591

[6] Wu et al., SicTTA, Medical Image Analysis 108:103859, 2026.
    https://pubmed.ncbi.nlm.nih.gov/41207141/
    https://github.com/HiLab-git/SicTTA

[7] Kervadec et al., Boundary loss for highly unbalanced segmentation, MIDL 2019 / journal extension.
    已知掩码的 signed-distance boundary supervision 来源。
    https://proceedings.mlr.press/v102/kervadec19a.html

[8] Zhao et al., Dataset Condensation with Gradient Matching, ICLR 2021.
    https://openreview.net/forum?id=mSAKhLYLSsl

[9] UniDD, CVPR 2025; D3S2, arXiv:2605.25022.
    Dense DD 的图像-mask对应、类别不平衡、资源问题参考，不直接复制 diffusion pipeline。
    https://openaccess.thecvf.com/content/CVPR2025/html/Qi_Towards_Universal_Dataset_Distillation_via_Task-Driven_Diffusion_CVPR_2025_paper.html
    https://arxiv.org/abs/2605.25022

[10] MedSeg-TTA benchmark, arXiv:2512.02497.
     用于审查统一协议，不代表它的所有任务都与当前 Fundus/Polyp CTTA 可直接横比。
     https://arxiv.org/abs/2512.02497

扩展对照候选：TTDG-MGM (CVPR 2025)、TACT (CVPR 2026)、TANGO (CVPR 2026)。
逐项核验是否需要额外 source training / VFM / different architecture / batch/future data，
不混入同一张“仅替换适配算法”的表。

## 4. 最小主方法：不再堆叠 Atlas/Router/Critic

### 4.1 固定宿主

首个 pilot 使用已验证 VPTTA 控制流和同一 source checkpoint：
- Fundus: ResUNet34；
- Polyp: PraNet；
- prompt、warmup、memory、optimizer 状态生命周期沿用已封存实现；
- main network 权重不更新；
- 不把“权重冻结”误写成“不需要输入到 prompt 的反向传播”。

新方法只额外提供一种 task-supervised proxy gradient。
模块关闭时必须精确回到宿主，不消耗额外 RNG、不多写 memory、不改 native counter/BN。

### 4.2 模式拆分

mode=base:
    原 VPTTA，无 DD。

mode=medical_no_dd:
    base + 一种独立医学辅助项；与后续 DD 分开评价。
    可优先测试同图几何等变一致性；Fundus inclusion 只作为独立 ablation。
    不强制该分支一定有效。

mode=proxy_rehearsal:
    base + 小型已知 mask 的代理图像监督；
    先用真实 coreset 仅作“特权机制对照”，不可标成 source-free 部署结果。

mode=condensed_rehearsal:
    用相同 mask/layout 预算的凝缩图像替代 coreset；
    标准 DD 与 adaptation-oriented DD 单列。

不要把 coreset 直接称为数据蒸馏，也不要把 target pseudo-label knowledge distillation
与 dataset distillation 混成同一件事。

### 4.3 代理数据

首版 K=8 个 anchor，只在 source 数据上准备。
每个 anchor 保存图像、固定 dense mask、必要的 mask-derived distance map。
不自由拟合 m_k、B_k、P_k；图像必须实际进入分割模型并产生在线梯度。

为降低 dense DD 难度，首版允许“固定源 mask、只优化图像”：
- 这属于图像凝缩、保留少量源标注，不是全部图像/标注都无来源的生成；
- 公开说明 mask 来源与隐私限制；
- 不在 GitHub 发布患者图像、个体掩码或默认发布 synthetic 医学数据；
- 无正式隐私保证，不写匿名/隐私安全。

真实 coreset 与 DD 尽量共享相同 layout 选择，避免 mask 多样性混淆 DD 增益。
缺失类别不构造假的前景，小目标与边界 layout 不能全部被大目标替代。

### 4.4 条件外观变换

T_s(anchor_image) 保留 anchor mask 的空间位置，只注入 donor 的外观统计。
第一版固定一种已有方法，例如小幅低频 amplitude transfer；不可同时搜索 FDA/AdaIN/PIPDE。
目标 donor 只提供 detach 后的统计，不提供 label，不读取未来图。

保留相位/不做空间 warp 并不严格证明标签语义不变：
- 先在 source-only donor/query 上检查 transform 质量；
- 不设“必须让 DSC 掉到某个区间”的前置条件；
- 不继续逆变换数值救援；
- 变换关闭时直接 identity path，作为计算/精度对照。

直接把两个患者 RGB 图像线性叠加的 MixUp 不进入首版。

### 4.5 在线更新

令 phi 为原宿主的可适配 prompt；o 为原 optimizer 状态：

L_total = L_host(x_t; phi)
        + lambda_replay * mean_k L_med(f_theta,phi(T_s_t(anchor_k)), mask_k)
        + lambda_optional * L_medical_unlabeled(x_t; phi)

(phi_next, o_next) = U_native(phi, o, L_total)

固定：
- extra branch 共用同一个 phi，不建第二组默默更新的 prompt；
- anchor forward 不修改 native sample counter、warmup、BN running buffer 或 memory；
- current image 的最终预测之后，只按原宿主规则提交一次；
- source/proxy labels 不等于 target labels，日志和 API 必须可区分；
- lambda_replay=0 应精确退化到宿主。

本式包含原宿主的测试时 backward，因此不再声称 backward-free 或沿用原 Atlas contraction theorem。

## 5. 医学任务损失：只保留有正确监督来源的项

### 5.1 Region loss

OD、OC 使用独立 Bernoulli 通道，不能 softmax 为互斥类别。
每通道先平均，再平均通道，防止全图像素统计淹没小结构。

有正负标注支持时：
L_balBCE,c =
  0.5 * mean_positive BCEWithLogits
+ 0.5 * mean_negative BCEWithLogits

若真实/固定代理 mask 某类为空，则该类用合法背景监督，
不能把“没有可靠目标伪前景则禁更”的旧规则错误套到真实负样本。
不虚构正像素。

L_region = mean_c [L_balBCE,c + (1 - softDice_c)]

### 5.2 Boundary loss

对有界非空/非满的固定已知 mask，计算 signed distance：
inside < 0, outside > 0；按图像对角线归一化。

L_boundary,c = mean_u dbar_y,c(u) * (p_c(u) - y_c(u))

与经典 signed-distance boundary loss 仅差与参数无关的常数，梯度一致。
该写法在正确符号约定下便于检查 GT prediction 的零值。
空/满类别只跳过无法定义的 distance 项，仍保留 region loss。

L_med = L_region + beta_boundary * mean_defined_classes L_boundary,c

该 loss 用于已知的 source/proxy/synthetic mask，
绝不能直接以未知 target GT 构造 signed distance。
初始 beta_boundary=0.1 仅是待 source-validation 冻结的工程起点，不是文献最优值。

### 5.3 Fundus inclusion（可选单独消融）

L_subset = mean relu(p_OC - p_OD)^2

前提：确认标签语义为 OC subset OD。
该项只编码必要关系，不能确保非空、正确边界或正确杯盘比。
空杯预测会使该项很小，所以不能作为防坍塌唯一目标。
默认关闭，先分别观察其收益。

### 5.4 首版不加入

- polyp 必須单连通/圆形/固定面积；
- 把视杯面积拉回健康均值；
- 全局 TV/最小周长作为主要适配目标；
- clDice 作为 OD/OC/实体 polyp 的默认 loss；
- 未经训练的 boundary head；
- 用不可靠伪 mask 生成 target signed-distance 硬监督；
- 同时堆 Dice+BCE+Hausdorff+clDice+topology+shape critic。

clDice/TopoTTA 的血管等管状结构动机不能直接移植成实体器官/病灶普适先验。

## 6. 真正值得研究的 DD：优化“适配之后的任务结果”

先让普通代理监督工作，再研究下面的蒸馏；不先搭复杂 atlas。

每个 source episode：
1. 取 source support 与不同图像 group 的 query。
2. 对两者使用同一风格变量 xi；没有 target 数据。
3. 从登记的宿主 prompt/optimizer 状态开始，在 styled synthetic anchors 上做一次更新。
4. 在同风格、不同图像内容的真实 source query 上评价 region+boundary task loss。
5. 仅通过这条真实更新链优化 synthetic image 参数。

形式：
phi_plus(S, xi) = U(phi, o, L_host + lambda * L_med(T_xi(S)))
S_star = argmin_S E_episode L_med(f_theta,phi_plus(S,xi)(T_xi(x_query)), y_query)

首版单步；跨样本迁移有效后才评估长度 2–4 的 episode。
不优化“合成图像看起来像原图”的指标作为主目标。

关键实现问题：
- native Adam 与离线 differentiable update 必须一致，含 m/v/step、epsilon、bias correction；
- 或明确另注册 SGD 宿主版本，并让全部对照使用相同版本；
- 不能 offline SGD、online Adam 却宣称优化了同一更新算子；
- 不能 detach(phi_plus) 后继续声称 task loss 更新 synthetic images；
- frozen source weights 不等于对 synthetic inputs 使用 no_grad；
- 基础 source checkpoint 本身不因 DD 更新；
- 模型中的 BN/custom statistics 要显式控制副作用；
- phase 尚未进行，不预写训练结果。

这是拟议贡献，不是已证“首次”。最近邻包括 ICML2023 proxy TTA、梯度/轨迹匹配 DD、
DO-ALL 及 dense DD；需证明超出这些简单组合。

## 7. 必要对照与因果问题

最小机制试验按顺序执行，不能一开始跑全部 source 矩阵：

A. 原 VPTTA。
B. A + medical_no_dd（与 DD 完全分开）。
C. A + 真实 coreset rehearsal，仅特权机制对照。
D. C + target-style conditioning。
E. 与 D 相同在线方法，用普通 dense DD 替代真实 coreset。
F. 与 E 相同，改为 adaptation-oriented DD。
G. F 去掉 boundary loss，其他不变。

D−C：style conditioning 是否必要？
E−D：是否只是压缩交换性能？真实 coreset不是公平 source-free 主方法。
F−E：面向在线更新结果的蒸馏是否优于普通 DD？
F−G：boundary 机制是否改变了结果，而非单纯多数据？
B−A：无需 DD 的医学约束是否已足够？

初次快速实现只需要 A/B/C/D 的模块接口；E/F 真正训练在代码审查和 source pilot之后。
若 C/D 都无源域跨图改善，不投入重型 DD；
但不能据此数学断言“DD永远无用”，只是当前项目资源分配不继续。
若 B 已更好、更快而 F 不提供增量，主方法不保留 DD。

## 8. 实验协议：三项科学问题，不再设十余项串联门

第一阶段：发布与代码审查
- 推送已有代码，保存旧 Atlas 状态；
- 只用 CPU/synthetic 检查新增数学原语；
- 无真实图像、无 checkpoint、无 GPU 训练；
- 固定 commit 后独立审查。

第二阶段：source-only 机制 pilot（审查后另行授权）
- 固定 source group split 和少量已知 photometric shifts，不构造 PIPDE；
- 每个任务最多两轮预先列出的 source-only 配置；
- 主要观察：实际 hard Dice、边界指标、同风格下一张不同患者/图像收益、
  source/anchor/target-style 梯度夹角以及计算量；
- 使用真实 source labels 的控制标注为 supervised diagnostic；
- 不要求一定有10%–70% harmful样本，不强制25%图像落入某个Dice区间；
- 若精度变化近零，可检查连续优化损失、输出差异，不把“不显著”自动当bug。

第三阶段：DD价值试验
- K=8 起步，固定总图像/掩码字节预算；
- 普通 DD 和 adaptation-DD 训练预算、初始化、mask layouts尽量匹配；
- 同样的 source-validation选超参；
- 留出整类shift及query图像group评价；报告不是新的 source-model-independent holdout，
  除非 source checkpoint训练成员能够确认；
- 记录DD离线GPU小时、合成数据字节、在线每图耗时/显存；
- 允许NAS私有保存必要的synthetic tensors/小checkpoint，预注册配额；
  不能沿用“任何模型/合成图像都不能保存”的诊断限制使蒸馏无法重现。

第四阶段：冻结后的目标评价
- 首先 mechanical smoke；不由分数选择checkpoint；
- mini32/24只能是开发/快速回归，不是可靠SOTA判断，不以tiny mini独立选赢家；
- 配置冻结后完整 source-A，两任务共1951/1808次原冻结访问；
- 若根据这些目标结果修改方法，标为development，不能再宣称这些是blind test；
- 最终增加未用于设计的source/外部domain或独立患者队列；同一stream新seed不是新患者；
- 固定域顺序/混合/复现域/3轮，重复轮次不视为独立seed。

## 9. SOTA 判定与资源公平

设定一张“same source weights / backbone / input / metric / stream / label access”表。
source训练改变或额外基础模型的工作另列增强设置。

最低比较清单：
- No Adapt、VPTTA、MGIPT；
- GraTa、DCP、SicTTA 可复现/协议支持部分；
- TTDG-MGM、TACT、TANGO需完整核对 source prep、weights和batch协议后纳入。
不能把缺源码/不同骨干的方法记为零分，更不能用已有复现负结果替代作者结果。

主评价：
- 每图DSC，Fundus OD/OC分别；
- 每域均值和等域宏平均，跨source再宏平均；
- legacy与standard分开；
- HD95/ASSD给出单位、空预测/未定义规则与数量；
- 预先定义surface Dice tolerance，不能看target结果选；
- small-target组由source规则或预先任务定义，不事后挑子集；
- 边界分数提高不能掩盖漏检率增加；
- method seed / stream order seed / source training seed分开；
- paired bootstrap以patient为单位；patient未知只能image-content group，并披露限制。

“SOTA”只在对应可比协议完整比较后使用。
先称“相对可比baseline有增益”；不预报成功概率或保证在两周内SOTA。

## 10. 发布阻塞的最小修复

GitHub OAuth的workflow scope控制workflow文件的创建/更新；
不是训练或审查科学代码的前提。

首选最小权限路径：
- 保留原本地已审计commit；
- 在新临时目录从该commit导出公开安全快照；
- 不包含任何 .github/workflows/ 文件，也不继承含该文件的提交历史；
- workflow模板可移动到ci_templates/且明确标为未启用；
- 新建publication root commit，只向已确认空的远端main推送；
- 若远端出现commit立即停止，不能force；
- 源码逐文件SHA与原已审计commit对比，只允许声明的非科学差异；
- 重新运行本地审计，CI状态写NOT_CONFIGURED。

仅在用户另行同意权限扩展时：
gh auth refresh -h github.com -s workflow
这是交互授权，不修改workflow YAML里的permissions来掩盖token scope问题。
官方：
https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps
https://cli.github.com/manual/gh_auth_refresh

## 11. 当前停止点

本轮仅发布已有实现、增加研究决策文档、给出可审查commit。
不删除原Atlas代码，不自动实现全部新分支，不训练。
下一轮基于实际commit做逐文件审查，再给出最小implementation patch与source pilot授权。
