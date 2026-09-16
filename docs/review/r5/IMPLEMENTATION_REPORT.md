# R5 阶段 I 实现报告

当前状态：R5_IMPLEMENTATION_READY_FOR_REVIEW。这里只交付实现和 CPU/元数据证据，external_review=NOT_RUN、execution_started=false、R5A=NOT_RUN、R5B=NOT_RUN。程序化 fixture 的 gate 值不是目标数据效果结果。

## 来源、范围与冻结

- 基础发布：b2bfce6cb29cea2df026194120f45b4f7252d53f；历史 R4 实际执行：2377505819ca9be6658b4f5b34f49dac3bf67889。
- 实现：0c08cece6a91bdf5e3d06c9862dcd192df7787d8；分支 experiment/r5-update-acceptance-v1。后续交付提交仅增加文档/证据；真实执行必须绑定实际 checkout SHA，不能把文档提交或旧授权静默当成已批准执行。
- science SHA256：89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5。
- C 指纹：71c8be1e77e1fe45eb9d9c84ba2c46edbc3936720117399eb342de4a89ccb829。
- registration 与 recurrence 摘要分别为 8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf、cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db。

改动全部是新增 R5 文件。未修改共享 C、数据读取、NFS ENOENT/其他 IO 异常行为、监督器、main、历史科学配置/结果、锁定依赖。窄 patch 只包含 src/scripts/tests/configs，附件和审阅材料分开交付。

## 已实现行为

C 使用原 B1 C 候选链，并只读捕获原始 pre logits、六个逆对齐弱视图和实际 BCE q。C_HALF 仅 lr 减半；C_VERIFY 依据固定风险门控；C_RANDOM 使用独立每访问一次的随机抽样和待 A 派生的 p_accept。无额外 loss、模型前向、teacher、融合或跨图稠密记忆。

门控按完整栅格 CPU float64 SSE/count、四可靠区、过去 128 风险与线性 Q90 执行。强制接受原因有优先序；有效拒绝风险仍在事务完成后入历史；非有限量进入失败状态。VERIFY/RANDOM 恢复 affine、buffers、完整 Adam 字段、param groups 与延迟状态，不更换 Parameter 对象，不回退 RNG/物理计数。输出始终是对应 pre/trial 的原生 logits；GT 只在事务完成并移出 payload 后读取。

独立 CPU analyzer 重算标量关系、窗口、决定、身份/覆盖/计数、无标签校准、主指标、解析随机影子参考和 A/B gate。它没有重建概率、梯度或 ASSD 几何。A 三流即使 gate 不通过也必须机械完整，科学失败与缺失/损坏分开；通过后仍停止。B_NEW 只增加 17 条并重验 A 的原始 SHA、trace、gate 和冻结校准，不重跑 3 条 C，不自动衔接 A→B。

未来入口沿用有限监督、自有进程清理和 neutral_subprocesses，默认 disabled。显式新 scope、完整绑定、新 review 或明确 waiver、设备、资源 caps 与 smoke 配方缺一不可。B 配方另行审阅；当前没有设备、caps、calibration 或 GPU 授权。新 R5 输出有专属归属 marker，analyzer 在任何失效/重命名动作前拒绝历史外部目录。

## CPU 验收与日志出处

见 logs/INDEX.json 和原始文本/JSON。每次检查均保留；路径替换是唯一去身份编辑。早期一次 R5_FAST 运行明确跳过完整网络；后续完整运行无跳过。final01 为最终定稿前快照的 133 项（21 项 R5 + 112 项继承回归），之后补充历史输出保护测试、严格 Q90/跨轨迹 GT 核验、三 worker 分配和全部参数 finite 检查；final02 为定稿实现的 22 项。不要把 133 项误称为最终版本同一次运行。

最终 R5 套件包括：固定完整随机 ResUNet 四条四步逐值匹配；真实 q/六视图顺序；初始与已有 Adam 状态回滚及下一步参考；参数/buffer/额外嵌套 Adam/group 字段与 ownership；GT 改变、延迟/省略、无 ID/domain 接口；超过 32/128 访问；空/部分区域、门限恰等、非法数值；各臂实测 8/1/1/0；A 校准及 gate 边界；独立构造 A 3×1951 与 B_NEW 17×1951 的标量 fixture 完成/污染/复用测试。全部 fixture 当场程序化生成，无真实目标或给定 checkpoint。

最终本地 22 项：22 通过、0 失败、0 error、0 skip，55.24 秒，Python 3.12.9 / Torch 2.6.0。实测 B1 路径共 368 forward、46 backward、46 Adam、0 VJP；其中完整网络匹配为 128/16/16/0。继承回归日志包括刻意注入的 IO 异常栈，unittest 的最终结果为 133 通过；这些预期异常不得删去，也不等于正式运行失败。服务器固定 Python 3.10.6 / Torch 2.2.1+cu121 的最终检查同样 22/22 通过、0 失败/error/skip，161.81 秒；计数也是 368/46/46/0。独立 detached worktree 精确为实现 SHA，检查结束 git status 为空，两个源码依赖 SHA 与冻结值一致，进程完整 argv 已确认中性；未查询或使用 GPU。详见 cpu-server-01 原始日志。

所有最终检查均阻断 CUDA 初始化及真实 asset loader；CUDA initialized=false、真实 RGB/mask/checkpoint/source 读取数均为 0。旧 IO 回归允许的图像/权重仅为独立临时目录内现场生成的小 fixture。CPU 测试物理计数和未来 GPU smoke 预算不混入正式矩阵；包含旧回归时计数范围明确限制为 B1 派生路径，不能声称覆盖其全部非 B1 方法。

## Metadata dry-run 与阶段预算

| scope | 条数 | 访问/候选 backward/Adam | forward | VJP |
| --- | ---: | ---: | ---: | ---: |
| A | 3 | 5,853 | 46,824 | 0 |
| B_NEW | 17 | 33,167 | 265,336 | 0 |
| A+B | 20 | 39,020 | 312,160 | 0 |

Dry-run 只读取已授权 registration JSON 并复用登记 recurrence，未打开 RGB/mask/checkpoint；公开件只含矩阵/摘要，不含逐内容身份或路径。每条仍为 1951 内容，remaining_dev 为 1695。C0 新运行预算 0，p_accept=null。A 与 B 的先后/复用暴露明确记录；新增序列不是独立患者确认。

## 未执行与边界

未执行 GPU smoke、A/B、后台等待、source 训练/代理/原型或额外 seed。没有方法效果、GPU parity、外部审阅通过或新执行授权结论。未来正式 GPU smoke 有事前继承容差，不能根据差值放宽。历史 R4 C 逐内容运行一致性尚未重新核验；C0 逐内容匹配未核实，H_t=null，详见 PARITY_CONTRACT.md。当前仅支持标量审计，不能据此主张统计显著性、临床安全或泛化。
