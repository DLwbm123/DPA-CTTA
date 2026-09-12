# R1 registration v2：只恢复metadata绑定

审阅基线为`f67a4d91c4337fe53e350cc2cf621ab703b9d2c1`。本修复不改变目标选择、顺序、分组、子集或方法参数。

`from_b3`仍只解析既有Fundus登记metadata；不调用旧source/proxy验证器，不遍历旧源资产。

| 字段 | v2内容与作用 |
|---|---|
| schema_version | 2；不是新的实验协议或执行许可 |
| checkpoint | 原有path、sha256、bytes、mtime_ns；SHA是内容依据，mtime不替代验证 |
| target | 保留原sample_id/group_id/domain/subset、路径、几何、manifest_index和linkage字段；恢复既有image_sha256、mask_sha256、original_split、split |
| historical_scalars | 原A与C0的metadata路径，不在本阶段读取真实分数 |
| historical_bindings | 对应旧receipt路径、receipt内容摘要、registration路径和旧commit/config/registration/device绑定字段；缺失项不阻止其余实现 |
| registration_digest | 对整个新私有登记计算原规范化JSON摘要。schema恢复后摘要必然更新，后续授权必须绑定新摘要 |

每个smoke进程和每个formal轨迹进程从唯一源checkpoint路径读取一份字节，核对既定SHA和bytes，再从该内存字节流执行`torch.load(weights_only=True, map_location='cpu')`。错误内容在GPU环境查询和模型构造前拒绝。不会以当前文件重新计算的摘要覆盖原预期值。

正式当前RGB到达时读取并校验其既定SHA；原PIL reader接受该已验证的BytesIO，RGB转换、BICUBIC缩放及像素缩放不变。host返回已固定预测之后，才读取/校验当前mask，再以原灰度/NEAREST/OD-OC规则解码。没有提前扫图或扫mask。每条记录保留RGB与mask读取、校验、解码的字节数和耗时；checkpoint读取/校验/加载成本记录在所属completion中，不折算成网络前向。

历史A按receipt绑定的原登记流逐项检查task/arm/order/visit和身份；C0先验证stateless canonical来源、原流和visit，再允许按身份映射至其他order。通道集合必须为OD/OC，GT计数与本次记录匹配。缺少文件标UNAVAILABLE；存在但不匹配或无法绑定标UNVERIFIED并保留具体错误；不阻断新主表，不补跑副对照。

公开解析摘要见交付的`ASSET_RESOLUTION.json`。私有登记含身份和绝对路径，只保留在服务器；不纳入源码归档或公开patch。当前阶段没有读取真实RGB/mask，也未对其内容做重新哈希。
