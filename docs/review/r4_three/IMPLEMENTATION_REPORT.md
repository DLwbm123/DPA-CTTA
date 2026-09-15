# R4T 阶段 I 实现交付

**R4T_IMPLEMENTATION_READY_FOR_REVIEW**。14 臂、70 条轨迹的代码、冻结 science、真实 metadata dry-run 和程序化 CPU 检查已完成。未启动真实目标流，不宣称 GPU 验收、效果提升或外部 review 通过。

实际 implementation SHA：`36ae77f7629150b244a57c1a888c7577b38e7f97`。基础提交：`2f90a6a0933cec3a25337a0d46ee632f8d772840`。新分支：`experiment/r4-three-track-teacher-kernel-boundary-v1`。

science 原始字节 SHA256：`3b4db63b60cb483c91329c87cafd91931f316b4cb94a1d5aeb6c08c236a64096`；registration：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`；回访流：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。

## 最终实现

| 路线 | 臂 | 实现与测试 |
|---|---|---|
| A 教师×RP | C, RP, MT, MT_RP, FT, FT_RP | C/RP 旧路径完整状态、输出逐值对齐；MT=.99 且晚于最终输出/memory；FT 当前输入统计；RP 存学生 pre 特征，额外前向计入两教师 RP 臂 |
| B kernel | KDG, K_ALL, K_MAG, K_FREE | 固定两个层位，底层源权重不变；新增坐标同一 Adam；零初始化 pre 教师对齐，冻结坐标时两步完整退化 C；实际新增参数和有效位移有完整模型证据 |
| C 边界图 | G_BOUND, G_CONST, G_SHUFFLE, G_GLOBAL | 原 raw RGB 与六概率图，128 网格/64 Jacobi/32,512 边，支持外与可靠区精确不变；无 RP/EMA/额外图 loss，最终输出仍为学生 |
| 延后评价 | 全 14 臂 q，三 RP 实际 token，四 G 的 q* | Dice/Brier/前背景分母，纠错四分类的 all/allowed/hard-flip 三类分母；GT 只在状态 commit 后进入 evaluator；不落盘稠密概率或 mask |
| 运行及核验 | 70-job 唯一矩阵 | 新 science/run binding；旧 IO/NFS/审计/cuBLAS/监督器保留；有限队列、失败回收、不自动重试；独立 CPU 重算与原子有效性 |

本轮只增加新文件。没有修改 main、R3 算法/85-job 常量、历史配置/结果或固定模型依赖。前十臂与核心科学块相对提供的 R4D JSON 未变，详见 [PRESERVATION](PRESERVATION.json)。生产图核与提供 reference 字节相同；kernel 的唯一数学实现细节修正为等价的 DC 求和次序，另增诊断量，详见 [来源与差异](METHOD_PROVENANCE.md)。

## 实际 CPU 证据与首失败

- 轻量 host 首轮：6 个测试方法，5 通过，完整模型检查在此 focused 运行明确跳过；不将其称为 6 项完整通过。
- 第一轮数学+完整 host：33 项，1 失败、2 错误。保留 [host-01](logs/host-01.log)：提供的 test_reference 搬入 repo 后相对 R4D JSON 路径需适配；模型工厂实际返回 (model,metadata) 而非 model；浮点带权求和让应为零的 DC 行出现极小残差，导致逐值零态断言失败。
- 修复不改科学配方：解包模型工厂返回值、修正测试文件定位、将常数基提出 DC reduction。没有放宽断言、人为 epsilon 源阈值、裁剪 gain 或断开初始导数。随后数学/host 33/33 通过，见 [host-02](logs/host-02.log)。
- 独立分析器/授权/70-job 检查 4/4 通过，见 [execution-01](logs/execution-01.log)。
- **最终本地完整套件 145/145 通过，零失败、零错误、零跳过，249.49 秒**。包含 38 项本轮测试与 107 项继承回归，前述 focused/host/execution 运行与最终套件重叠，不能相加当成独立测试数。真实 [完整日志](logs/cpu-full-01.log) 与 [JSON](logs/cpu-full-01.json)。
- 本地环境 Python 3.12.9 / Torch 2.6.0；CUDA 未初始化，真实目标 RGB/mask、给定 checkpoint 和源数据读取均为 0。仅旧 IO 回归允许自己创建的临时程序化 IO fixture。
- 随机权重完整 ResUNet 实际执行 R4T **260 forward、32 loss backward、32 Adam、0 VJP**。包含旧 C/RP 参考和 14 臂各 2 步。kernel 真实可学习标量为 21,283 / 21,283 / 19,264 / 23,424；当前检查不要求新 K 首步 post 等于 C，也不要求图一定翻转或提高 toy Dice。见 [完整 CPU traces](FULL_MODEL_CPU_TRACES.json)。

这些是程序化测试和环境证据，不是实际 target 流效果；历史 EIO 风险仍保留未知根因表述。服务器原 Python/Torch 环境的独立 CPU 检查结果在 [服务器核验](SERVER_CPU_CHECK.md) 单列，不能与本地 backend 混淆。

## 元数据与冻结预算

已解析原私有 registration JSON 的 1,951 条元数据，匹配原内容/子集/顺序、registration 和 recurrence digest；不解码像素。生成的 [70 条实际绑定 dry-run](DRY_RUN_MATRIX.json) 不再只是包内算术 reference。

- 主序 56 轨迹、109,256 条记录；回访 14 轨迹、27,314 条记录。
- 正式合计 136,570 评分/Adam/backward、1,112,070 forward、0 VJP；每实际设备 smoke 32 Adam/backward、260 forward。
- 三条路线共享同批 C/RP 结果，各轨迹仍独立初始化；不先执行旧 50-job 排程，不按中间效果跳臂。
- 运行上限：最多三 worker、每卡一个、每 worker 两 CPU 线程；6h/trajectory、72h wall、96h active-worker、新私有输出6GiB。上限不是预计耗时。

## 交付与边界

[完整 patch](IMPLEMENTATION.patch) 包含实现提交相对基础的全部变更，包括提供的规范材料；[窄代码 patch](IMPLEMENTATION_CODE.patch) 仅 src/scripts/tests/configs。[状态生命周期](STATE_LIFECYCLE.md)、[方法来源](METHOD_PROVENANCE.md)、[未运行项](UNRUN_ITEMS.md)、[CPU/metadata 复现](REPRODUCE.md)、[机器可读交付](DELIVERY.json) 均已提供。

实现和证据分别提交，implementation SHA 固定为上方值，后续证据提交不冒充运行代码。只发布去身份代码/配置/标量与日志，完整权重、RGB/mask、逐内容身份/路径及实际设备授权留在私有位置。

尚未产生外部 review 通过结论。执行入口默认禁用；真实外部 pass 与用户明确免除 review 是分开的状态值，且都必须绑定具体 SHA/新 science 和完整70轨迹。代码提供显式 waiver 路径不构成本轮 waiver。收到需要的 review 条件后才使用已确认资源，不能靠后台等待自动开跑。
