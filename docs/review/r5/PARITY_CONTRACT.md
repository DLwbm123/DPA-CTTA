# R5 事前保持性约定

## CPU 精确性

同一 CPU 环境，随机初始化固定完整 Fundus ResUNet34，seed=20260907。每条轨迹重新初始化，四步依次使用 tests/test_vptta_host.py 的 pixels('fundus',0..3)。旧 R1 C 与 R5 C_DIAG（生产臂名 C）、VERIFY 的强制接受期、RANDOM p=1 精确比较：每步 logits、BN affine、完整 Adam.state_dict、增强 RNG；rtol=atol=0。每图实际 8 forward、1 backward、1 Adam、0 VJP。完整网络四条四步共 128 forward、16 backward/Adam，与未来 GPU smoke 分开计数。

Toy 测试另验证首个空 Adam state、已有 moments/step 的拒绝恢复与下一张输出、额外嵌套 optimizer 字段与 group 元数据、buffers、对象 ownership、payload 生命周期和标签隔离。测试可以 monkeypatch 决策强制拒绝；生产 science/CLI 没有此开关。合法零梯度和零位移不要求额外涨分。

## GPU 配方（仅提案，未执行）

A 每个实际设备：旧 C 四步与 C_DIAG 同四步独立初始化，共 64 forward、8 backward/Adam、0 VJP。程序化 pixels 索引固定为 0/1/2/3，seed=20260907；不从真实标签挑 smoke 内容。实际 source checkpoint、设备与新授权都留到 A 阶段。

rtol=1e-4、atol=1e-5，来自既有 src/dpa_ctta/source_pilot_release.py:TOLERANCE，并由 host_diagnostic.close 使用；Adam step 与 RNG 仍精确。配方必须原样绑定 auth/receipt，不能看见差值后调宽。沿用 deterministic_smoke_pair 和 R3 修复的 smoke cuBLAS 环境；正式阶段删除 smoke 专用 cuBLAS 配置。当前不宣称 GPU parity 已通过。

B 配方须单独审阅绑定：提议旧 C 与四臂各四个 warmup 访问，共 160 forward、20 backward/Adam。C/HALF/RANDOM/VERIFY 的边界与角色不同，HALF 不断言与 C 数值相同；其他臂只比较这四个强制接受访问。该配方不构成 B 权限。science 中 B_smoke_recipe=null，只有另行显式 receipt 才能提供受检查的配方。

## 历史 R4 C 与 C0

R4 发布 SHA b2bfce6cb29cea2df026194120f45b4f7252d53f 与实际执行 SHA 2377505819ca9be6658b4f5b34f49dac3bf67889 分开保留。阶段 I 没有重读/重跑真实目标，也没有宣称完成 R5-A 对历史 R4 C 的逐内容一致性核对。当前 CPU 证据证明同后端旧 C 调用链的保持性；历史 R4 五条 C 的私有逐内容记录可在后续授权下按 checkpoint、registration/stream、身份、预处理/统计政策/阈值核对。共享 GPU 的旧运行不等于位级重现，不据共享设备耗时声称加速。

旧 B4 公开报告含 C0 聚合，但本次未取得并核实匹配的私有 C0 逐内容输入。因此 H_t=null，明确标为未核实匹配，不声称 C0 文件不存在，也不以聚合均值或行位置补数、不自动补跑。可选 H 不参与 A 的核心 gate；当前交付不实现未经证明的跨记录历史分解。

## 后续 A→B 复用

三个 A C 的原 execution SHA 必须保留。reuse_A 重放完整 A 的无标签及评价 ledger、gate、校准摘要，要求相同 science/data/stream/C fingerprint。指纹覆盖旧 B1/R1 C、R5 host/rule/evaluator 和 science 原始字节。严格匹配可能拒绝仅工程性变更；此时需单列审阅证明，不静默放宽。B 每次只用 A 冻结的 p_accept；RANDOM 自己判定 eligible，不保证实际接受次数完全相同。
