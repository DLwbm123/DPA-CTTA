# 发给 Codex：R3 审阅通过后的固定矩阵执行

## A. 已完成的外部审阅，不由 Codex 重新批准

```text
EXTERNAL_REVIEW_STATUS = R3_REVIEW_PASS
CLOSED_ITEM = R3-ENV-01
REVIEWED_IMPLEMENTATION_SHA = d601496a0827af3e1fe728612a4b9f17a613955d
SCIENCE_SHA256 = 73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e
BASE_REGISTRATION_DIGEST = 8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
SECONDARY_STREAM_DIGEST = cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
APPROVED_TRAJECTORIES = 85
EXTERNAL_REVIEW_REFERENCE = R3_REVIEW_PASS_d601496a.md（本包 REVIEW_PASS.md 的同内容副本）
```

生产代码是 implementation SHA，不是资料发布提交 `8b010288…`。已审 science 中的 PROPOSED 字样只是冻结文件中的历史文字，不修改该文件来标记通过；审阅记录和授权放在独立私有 JSON 中。

此次外部复核通过五个框架的原方法审阅及有限环境修复。不需要再改方法、不需要再次发代码等待同一项审查，不设置先获得正向 Dice 才运行下一臂的条件。

## B. 用户必须明确的本批资源字段

```text
AUTHORIZED_PHYSICAL_GPUS = <用户本批明确选择，最多3张；6/7仅为既有偏好>
MAX_CONCURRENT_WORKERS = <用户明确选择，1至实际授权设备数，最大3>
ALLOW_BACKGROUND = <用户明确 true 或 false>
PRIVATE_ASSET_CONFIG = <沿用原服务器已验证资产配置>
PRIVATE_OUTPUT_DIR = <已有私有存储根内的全新目录>
```

资产路径优先从既有 R1/R2/R3 私有配置解析，不要求用户重新上传或重建登记。设备/后台授权不能由偏好、旧实验授权或本报告推断。未填资源权限时只准备私有授权材料并询问缺少的权限，不运行 GPU、不建立等待批准自动启动的后台进程。

用户可在发送本文件时加一句：

> 授权按已审 R3 方案执行全部 85 条轨迹；使用物理 GPU 6、7，最大并发 2；后台运行允许。

上句只是示例，不是本文件替用户作出的授权。用户也可以明确选择前台或其他合法设备组合。`EXECUTION_AUTHORIZATION_TEMPLATE.json` 保持 disabled，待真正获得授权才生成启用副本，不修改仓库中的 disabled defaults。

## C. 运行范围与执行顺序

使用既有 `scripts/run_r3.py` 入口和干净隔离 checkout。沿用原 Python、固定模型与 GraTa 依赖，不重新安装环境，不做源训练，不扩大数据范围。

17 臂固定顺序：

```text
C, RP,
T_LR, T_ISO, T_DIAG,
U_PCA, U_RAND, U_SCALE,
S_JOINT, S_SHARED, S_NOPCA,
M_TRANSPORT, M_IDPOST, M_SHUFFLE,
G_PCA, G_ISO, G_ORDER
```

每臂四个主序加一个固定 secondary 回访流，每条 1,951 组，共 85 条完整轨迹。C/RP 本轮同批运行，不用旧结果顶替。四个主序 68 条轨迹/132,668 评分；secondary 17 条/33,167 评分。secondary 是不同内容的环境块回访，不是重复同一图，不与四序主终点合并。

先按已经实现的资源检查与授权绑定核验本批运行，然后每块实际设备做一次既定 smoke。所有设备 smoke 成功后自动进入同一个有限正式队列。无需把 smoke 效果发回等待批准，也不新增额外 warm-up。

保持既有 `(arm_index + stream_id) % workers` 调度，一卡最多一个本作业 worker；一条轨迹内部按已登记顺序串行。同一轨迹不能中途移卡/续接另一轨迹的 affine、Adam、memory、router 或 RNG。

## D. 本次修复后的环境政策必须生效

由被审 `launch.start` 在 Popen 前构造环境，不依赖用户 shell：

- smoke：`CUBLAS_WORKSPACE_CONFIG=:4096:8`。
- formal：删除该环境键，沿用原正式数值政策。
- 授权后的 worker 再核验阶段环境，错误时在模型/设备工作之前拒绝。
- 严格 smoke 保留 `deterministic_algorithms=True`、`warn_only=False`，不降低配对容差，不移除 ready 分支。
- 分开记录 smoke 外部 backend 与内部 `paired_comparison_backend`；formal 记录自己的实际开关。
- 不从资料中的 CPU workspace=None 推断 GPU 也可缺失配置。CPU 测试与实际 GPU 环境前提不同。

## E. 预算与状态约束

| 项目 | 固定正式预算 | 每块实际 GPU smoke |
|---|---:|---:|
| 评分 | 165,835 | 不进入效果表 |
| loss backward | 165,835 | 38 |
| Adam proposal | 165,835 | 38 |
| 网络前向 | 1,385,210 | 316 |
| 额外 Jacobian VJP | 至多 234,120 | 至多 24 |

两卡全部成功时：165,835 条正式评分，含 smoke 165,911 次 Adam/loss backward、1,385,842 次前向、VJP 至多 234,168 次。VJP、参数替换不是额外 Adam；不能混在一个模糊的更新数里。

原资源上限不变：单轨迹 6 小时，矩阵墙钟 96 小时，累计活跃 worker 160 小时，新增私有输出 6 GiB，每 worker 最多 2 个 CPU 线程。上限不是完成时长预测。显存和共享资源按既有入口检查，不停止其他用户或其他作业，不自动占用额外 GPU。

每条正式轨迹都从规定的源初始化与空 memory 开始，不带入 smoke 的程序化统计。只有给定 checkpoint 及其已有 buffers 可用，禁止 source images/masks/query/proxies/prototypes 或新的源模型训练。按当前到达只读当前 RGB；host 返回固定预测并提交本图在线状态后，evaluator 才读取当前 mask。

router 不扩容、density/basis 尚未 ready、无图边或候选落后，均正常记录并完成固定方案，不以效果改变参数、rank、可靠性阈值或顺序。G 零边短路的迭代字段按现有代码如实解释为计划/实际差异，不宣称短路也实际求解 32 次。

## F. 失败处理：不自动试到成功

普通运行期间的 IO/资产/非有限/超时或调用预算错误，沿用被审有限监督器政策：停止新的任务派发，按故障范围处理在途任务，必要时只回收本作业登记的进程组；绝不为了写入失败日志成功而阻止进程清理。

历史 EIO 状态保留：本次回归未重现，根因未知。不能宣称根因已经修复；但不新增重复 EIO 压测或等待连续绿灯的门槛。若真实运行出现故障，保存真实异常、已完成和未完成 job、前缀计数以及进程退出/清理证据。

不自动 retry，不合并残缺前缀冒充完整轨迹，不把已完成但分数差的 job 作废。确需工程续跑时，先报告原因、完整/残缺任务和新增物理成本，等待明确授权。本文件不授权自行修改代码后继续运行。正常成功路径一次完成全部框架。

## G. CPU 汇总、公开交付与最终选择

全部轨迹结束后执行独立 CPU 标量重算，拒绝缺失/矛盾证据，原子发布成功结果；失败重算不能留下有效 COMPLETE。保持 104 项检查和本次外部审阅边界，不把独立标量重算说成重新计算了未保存的特征、Jacobian 或 ASSD 距离图。

交付完整17臂，不只展示赢家。报告四主序等权结果及独立 secondary 结果，包含 OD/OC/macro、逐域配对、正负比例、最差十分位、共同有效 ASSD 及 undefined 分母、调用数、VJP、峰值 allocated 显存、实际墙钟/host耗时与状态存储。

分别比较五个主候选与 C、RP 和本家两个控制：

- T_LR 对 T_ISO/T_DIAG；
- U_PCA 对 U_RAND/U_SCALE；
- S_JOINT 对 S_SHARED/S_NOPCA，另分析 secondary 复用；
- M_TRANSPORT 对 M_IDPOST/M_SHUFFLE；
- G_PCA 对 G_ISO/G_ORDER。

按既定 `08_RESULT_SELECTION.md` 解释结果：平均相对 C +0.5 pp、至少3/4主序同向只是资源选择参考；同时检查关键 OC 负向风险和成本，不视为统计/临床/录用阈值。简单控制胜出就如实推荐简单控制，不为保留 PCA 而改评价。

报告只使用真正保存的证据；没有 teacher 真值对照或几何缓存时不得补造该项分析，不为补齐论文式图表追加 GPU 推理。四序共享已暴露内容，不是四次独立患者验证。

公开推送代码、配置、去身份结果、报告与真实执行/失败记录，验证远程提交和匿名可读。不得公开原图、mask、checkpoint、私有路径/身份、设备 UUID 或论文 PDF。实际执行 SHA 与结果发布 SHA 分开写。

只有全部85条完整轨迹和CPU汇总通过才写：

```text
R3_EXPERIMENT_COMPLETE
```

随后结束本次有限任务。不自动组合、不启动 Polyp、其他 seed、阈值搜索或新一轮研究。
