# B1 GraTa matched baselines：修复版启动报告

**B1_RUNNING：正式运行已启动，完整评分及 CPU 汇总尚未完成。**
快照时间：2026-09-09T12:14:43.898Z。
执行提交：029a79340057b58b8074ccf86c7b5671b5fdbd1c；发布分支：experiment/b1-grata-matched-baselines-v1。
P2 基线：2fcbc0a69645e34da42b97830935f0e290e488aa；CTTA 依赖：dbff0d985c6c95345d9fb78f5b1daef57b392564；GraTa 依赖：33ae20d664f305af34739ec54a5bec7da53ffa0b。
运行期间执行 checkout 保持不变。

## 修复、授权与验收

首轮在第一个 C 参考弱前向后因返回值接口不匹配停止，完成 GPU base Adam=0。
失败 checkout、receipt 和前缀保留；[首轮报告](https://github.com/DLwbm123/DPA-CTTA/blob/ee4b434f016ce121e835359a7427931d151e72b4/results/b1_grata_matched_baselines_v1/B1_EXPERIMENT_REPORT.md)仍可访问。
用户随后明确批准“允许修复版执行”；本次使用独立目录及新 receipt。

修复将 P2 的 (logits, skips, head_input) 包装为官方需要的 (logits, feature)，不增加参数、辅助 head 或前向，不改变 logits、BN、增强、目标、扰动及 Adam。
7 项相关 CPU 回归本地及服务器通过，包含实际完整模型的 C/G 官方参考、梯度/状态/RNG、标签隔离、严格加载、几何变换及错误前缀检查。本次复用修复提交上已通过的 CPU 记录。

修复版 GPU smoke **16 次 base Adam 全部通过**；C/G 各四个访问的配对 logits 最大差均为 0，affine、Adam、LR、RNG 比较通过，source 对齐差也为 0。
耗时 15.921 秒，C/G allocated 峰值分别为 615612928 / 615775744 bytes。
容差固定 rtol=1e-4、atol=1e-5，离散状态和 RNG 精确比较。
正式阶段恢复 P2 backend 设置，不宣称正式 CUDA 路径全部逐比特确定。

## 冻结协议与预算

仅 Fundus，复用 1951 组：REFUGE=400、ORIGA=650、REFUGE_Valid=800、Drishti_GS=101。
remaining_dev=1695、legacy_dev=128、p1_extension_dev=128。
旧 P2 七输出的身份/顺序/覆盖/父视图/计数/Dice 已复核，本轮复用 N/A/EA/O2/D4 五个冻结对照，不重新推理。

C：六弱视图软目标 + 一强视图 BCE，一次固定 lr=1e-4 Adam。
G：发布版 ent 辅助梯度、直接减梯度扰动、consis 梯度、恢复参数、cosine 动态 LR，一次 base Adam。
保留发布版不完整 Bernoulli 熵，不修改算法公式。
顺序为 C/order0 → G/order0 → C/order1 → G/order1。每条轨迹从相同 source 状态、seed=20260907 开始，跨域不重置，各自隔离 RNG；预测固定后读取 mask。

实际更新 **19136 个标量参数**，来自 **41 层 BN / 82 个 affine 张量**，完整名称见 execution_audit.json。
其他卷积、decoder/head 参数冻结，source 文件只读复用。

正式计划：**7804 条新评分、7804 次 base Adam、66334 次前向、11706 次反向**；G 扰动和恢复各3902次。
加本次 smoke 后计划 base Adam 总计7820；新 source/DD/outer 训练均为0。
本次 smoke 另有2次 source 对齐前向；首轮失败的2次 source对齐与1次弱前向单独保留，不计作正式方法效率。
6小时活跃阶段、1GiB新私有输出是配额，不是预计耗时。
C/G 每图分别8/9次前向、1/2次反向；不能因一次 Adam 就声称与原 A 同等成本。
旧 O2/D4 需要凝缩图像及源 mask，C/G 不使用源样本 rehearsal。

## 已确认启动状态与剩余工作

GPU 7 启动前可用16666MiB，准入要求12288MiB；NAS 挂载、容量和写读探针通过。
后台 launcher PID=621353，不依赖 SSH/Codex 会话持续开启。
快照时 C/order0 已写入 **42 条记录**，Adam step=42，每图8次前向/1次反向/1次Adam；
allocated 峰值=617054208 bytes，运行环境与 P2 一致，未见立即失败。
上述仅为启动快照；完整 C/G 结果仍待定。

四条轨迹结束后由后台启动独立 CPU 重算，核验完整记录及旧对照配对，生成聚合与报告。
失败即停止并保留前缀，不自动重试、不创建持续监测器、不自动开启下一实验。
当前不能写 B1_MATCHED_BASELINE_COMPARISON_COMPLETE 或方法有效性结论。

公开源代码、配置、许可证和去标识启动审计；私有 registration、资产身份/路径、图像、mask、checkpoint、概率图和逐图原始记录不公开。
ASSD 保留共同有效 cohort 与未定义计数，不构造 OD/OC macro ASSD，CPU 标量复算不冒充像素 ASSD 重算。
这是同权重连续协议的 G-CTTA 移植，不是原论文完整复现、原创方法或未见泛化/临床结论。
