# 给 Codex：R7 TARGET GPU 桥接与失败成本修复；不启动真实 target

本任务仅属于 DLwbm123/DPA-CTTA，不属于 SSL_CL_seg/AGMS_OBSERVER。先读取附带 R7_EXTERNAL_REVIEW_RECORD.json 和 EXTERNAL_REVIEW.md；按原字节归档，保留所有旧审阅和失败记录。本记录没有 TARGET_SCREEN PASS 或用户启动确认，不能制造 enabled receipt。

## 固定锚点与唯一任务范围

源 GPU 实现 ba04ca46a6bae5df4f4ebb272ef4045f5ee400e5；GPU 证据 ea47b81727115d50abe4824d3d2ed1327377f3ab；完成及六产物身份 1936775cf7bc25dee5bcc658480da45e11e63236。
原 TARGET CPU 实现 9de62090fb921695aacd7d68027cf7ec853bcc47；发布 f5d1ef75859cf82cf039819c50732584812f4d9d。

在新分支/独立工作树完成 TS-COST-1 修复和 TS-BACKEND-1 集成。不重训 SOURCE_PREP，不触碰已完成运行目录、原权重/context、原 source split、science、算法公式、学习率/rank/seed、历史 C 或历史账本。不得覆写 GPU 源实现，新的 target 合并实现使用自己的最终 SHA。

## TS-COST-1

成功 online 后先保存可验证计数，再进入 posthoc。对部分 online 失败、host.step 失败、预测写失败、评分失败及成本记录写失败分别保留：观测的 F/B/Adam/VJP/其他计数、已提交 visit 数、失败阶段、首异常、次级错误。不要只依赖成功返回 trace。

不能用固定预算、成功预测文件数或完成 visit 数反推所有物理调用，尤其 C_BASE 一次 visit 内可在多次 forward 或 optimizer 后失败。精确计数与硬杀尾部 lower bound/unknown 明确区分；不对不可观测成本填零。不得为了保存成本替换首失败，不加入 retry/resume。

至少补充：成功对照；online 完成后 mask/posthoc 失败；部分 online 失败；模型调用后 prediction 写失败；计数持久化失败同时保留首异常。附带 review_probes.py 的复现结果是修复前缺口，不是修复后的 PASS。

## TS-BACKEND-1

复用源端已审 GPU backbone + CPU method/FP64 latent bridge，仅新增 target 所需设备、context 和资源绑定。不删除/伪造 GPU execution_backend，不重写原 context/hash，不从候选模型生成 expected。旧 CPU 路线继续兼容；给 GPU 包走 CPU 路线必须在正式 target 读取前拒绝。

八个 arm 都保持原算法：C_BASE 8F/1B/1Adam；C0 1F/0B/0Adam；A/B/C FULL/STATIC 各2F/0B/0Adam。GPU 资格不能只覆盖六个新方法而遗漏 C_BASE/C0。保持图像独立输入、state 提交规则、STATIC reset、每 job 独立模型/进程/输出、在线结束后才评分、首失败清理和全树资源终态检查。

24 jobs、seed20260907、orders0/1/4、每 job1951 arrivals与1695 remaining_dev 不变。总预算122913F/5853B/5853Adam；46824是全部 visits，实际 remaining_dev scalar rows为40680，不将其当作独立患者。最多3 worker只有在资源批准后启用，不能因实现调度器就默认占3张GPU。

## 资格与真实落盘产物：先计划、后在明确许可下执行

这段指令授权代码/公开元数据工作、相关有界程序化CPU回归和合成边界检查；不授予GPU查询/使用、真实checkpoint/.pt加载、真实图像读取或TARGET实验权限。禁止重跑旧166或整个source训练。

生成一个有限的 GPU_TARGET_QUALIFICATION_PLAN，预先列明每一项的模型大小、steps、各arm调用、VJP/optimizer/序列化成本和总上限。应覆盖八arm后端执行、FULL/STATIC、context错误拒绝、确定性政策、模型状态/梯度界限和成本记录。未实跑部分为NOT_RUN；不得声称CPU资格等于GPU资格。

另生成只读 ARTIFACT_ROUNDTRIP_PLAN：在与源context匹配的后端，对原六份磁盘.pt调用既有load_artifact，从独立inventory/context核验真实字节、反序列化、effective backbone、group/mode和method digest。只加载不训练，不读真实source/target pixels；若增加合成前向另列精确计数。不得预填trusted_loader_verified=true。

若本轮另有用户明确给出的、覆盖该资格与私有包只读加载的授权，才按其范围执行相应资格，并将许可原文、预算、真实调用、环境和六包结果单列。否则只交付禁用的资格计划；不得解释旧SOURCE GPU授权为TARGET GPU授权。

## 一次性交付与停止

交付新准确代码SHA、patch、CPU测试与成本、GPU资格计划/实际结果、六包roundtrip计划/实际结果、原六包及context身份、禁用TARGET授权模板、精确24-job清单和资源提案。不要把元数据发布SHA伪装成已测试实现。

提交审阅时清楚区分 source code review、source completion evidence、真实serialized loader证据、新target execution-layer review、用户target launch权限。新target未审前不生成PASS/enable，不读取真实target。

结束状态：
R7_TARGET_GPU_BRIDGE_AND_COST_FIX_READY_FOR_REVIEW
SOURCE_PREP=COMPLETE_PRESERVED
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
execution_authorized=false
next_scope_authorized=false

若无法完成，报告具体工程缺口，不反复生成内容相同的旧启动绑定包。
