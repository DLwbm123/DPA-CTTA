# P1 无 DD 当前图像源预测约束：启动报告

状态：**P1_RUNNING**，检查时间 2026-09-09T09:47:57+08:00。正式比较仍在运行，尚无效果结论；不是 `P1_NO_DD_COMPARISON_COMPLETE`。

实际执行提交：`f65e119016f2d6a32cd0cffbee2bb2e94f570c2d`。本报告在单独的发布提交中更新。基线：`fab4ce4f7733e54258d577fe5e22cea8a562b6d6`；固定外部依赖：`dbff0d985c6c95345d9fb78f5b1daef57b392564`。

## 已固定的比较

输出为 N / A / ENS_A / SA / ENS_SA / O2 / D4 / L4；五条独立适应轨迹，两个 ensemble 只复用父轨迹输出。SA 使用同 checkpoint、标准 source BN、eval、无 prompt 的当前图 teacher 概率，对 student 原生第一次当前图 forward 施加 0.1 × Bernoulli KL；融合在概率空间固定 0.5/0.5，再直接以 >=0.5 阈值评价。

A、SA 与 teacher 在当前访问中产生五个固定预测后，评价器才读取 GT。A/SA 的 RNG、Adam 和 memory 独立；teacher 只读前向共享一次，独立部署成本仍为相关方法计入源前向。三个封存 DD 代理全部在新流上重新执行，无旧均值填表。没有新增 DD、source 或轨迹训练。

## 实际扩展与预算

| 任务 | 域数 | 旧组 | 新组 | 合计 |
| --- | ---: | ---: | ---: | ---: |
| Fundus | 4 | 128 | 128 | 256 |
| Polyp | 3 | 96 | 96 | 192 |
| 合计 | 7 | 224 | 224 | 448 |

每域均为 32 legacy_dev + 32 extension_dev，旧/新一起恢复原 manifest 顺序。扩展按计划的 SHA256 排名，未读取模型评分、mask 面积或难度用于选择。选中资产完成新字节身份与编码/几何验证；6 个封存代理摘要与历史值相符。源域完整内容、跨域重复、冲突及受保护角色均按登记规则排除。UNKNOWN linkage 和 combined provenance 明确作为开发集限制保留。

正式计划：7,168 条新评分、4,480 次 online Adam；加 smoke 为 4,492 次。outer、可微 inner 训练、新代理训练与 source 训练均为 0。主解释优先 extension_dev；legacy_dev、combined 和两种域顺序全部报告。

## 验收和已确认运行状态

本地与服务器均通过 7 项相关 CPU 测试：包含两任务各 18 步冷/热原生等价、KL 独立梯度与 0/1 边界、teacher/label/ensemble 隔离、概率阈值和原像素指标一致性、代理绑定、扩展去重/空新增，以及八输出两序的独立标量重算。

GPU 7 完成唯一一次 GPU smoke，实际 12 次 online Adam，退出码 0。Fundus 与 Polyp 的 λ=0 预测对原生 A 最大绝对差均为 **0.0**，prompt/Adam/memory/counter 通过原容差；SA 及全部历史代理输出和状态有限，teacher/source 权重与 buffers 保持不变。没有用程序化 Dice 作为运行 gate。

使用既有 Python 3.10.6 / Torch 2.2.1+cu121，正式环境与 M4 相同；严格确定性仅限配对 smoke，随后恢复。启动时 GPU 7 空闲 24,124 MiB；采用 8,192 MiB 准入余量，smoke allocated 峰值为 Fundus 3,628,422,656 bytes、Polyp 2,315,915,264 bytes。正式共享 bundle 的峰值另行记录，不把 smoke 峰值冒充全部运行峰值。

后台 launcher 已确认存活，Fundus/order0 的 N/A/ENS_A/SA/ENS_SA 各写出 **52 条**完整记录，A/SA 各完成 52 次更新；没有即时失败。之后按固定计划继续历史对照、另一顺序及 Polyp，最终串联独立 CPU 重算。当前 run/recompute 退出码为 null，效果指标为 pending。

运行不依赖 SSH 或当前会话持续连接。阶段工程失败会保留前缀并停止；没有自动重启、参数搜索、持续轮询或 P2。

## 准备阶段问题与公开边界

第一次登记在遍历 M4 包装结构时出现 KeyError: tasks；已改为读取 tasks 成员并完成登记。该次错误在选中资产字节验证和 GPU 更新之前发生，online/outer/inner 均为 0；原始准备失败记录保留。CPU 初版夹具的 clone 对象身份假设和观察器安装后重载状态问题也已修正，未放宽数值容差。没有外部独立 review，不声称 CODE_REVIEW_PASS。

公开源码、配置、测试、本报告及去标识[执行证据](execution_audit.json)。原图/标注、checkpoint、代理、路径、内容组 ID、逐资产 digest 和私有 receipt 留在 NAS；不保存本轮 target logits 或概率图。独立 CPU 过程重算的是标量汇总、Dice 像素计数和父输出绑定，不能离线重建未保存的像素预测或重新计算像素 ASSD。

完整运行后还需核对覆盖、尾部与所有对照，再单独给出“更新有增量 / 仅集成有益 / 混合 / 未见收益”的科研判断。两个顺序共享样本、单 seed、已暴露开发域及未知患者/视频关联限制不变。
