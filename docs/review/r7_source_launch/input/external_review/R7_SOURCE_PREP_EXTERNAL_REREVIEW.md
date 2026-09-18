# R7 SOURCE_PREP 执行层外部复审：PASS

## 1. 结论与精确绑定

**R7_SOURCE_PREP_REVIEW_PASS_EXECUTION_LAYER**。

SP1、SP2、SP3 在本轮约定的 CPU 执行层审阅范围内均可关闭；未发现需要继续阻塞这一修复版本的新增问题。原 Stage I PASS 保留为既有结论，不重新签发、不重开已关闭的方法/F1 审查。本结论不是安全性穷尽证明，也不是执行授权。

- 被审实现：`f719c703087b38c07bdfbe7ce9dcfa62d88a12d9`。
- 证据发布：`ecde8ba63940eb5c660f4de01bd7a9300b18c15b`。
- 修复基线：`ad95f212b8770e5006e0b1ea3520717767c7413f`，旧被审实现 `9fa2ce37149909687ef5be2e08aa2b32a5887903`。
- `source_binding_status=PENDING`，`execution_authorized=false`；SOURCE_PREP 与所有 TARGET/GPU 阶段继续 NOT_RUN。

未写回 GitHub。本文与机器记录是外部复审材料，不是 `R7_SOURCE_PREP_AUTH_V1`，不可直接当作可执行 receipt。

## 2. 证据来源与核验边界

通过 GitHub 连接器读取固定 SHA 的代码、测试、FIX_REPORT、CPU_RESULTS、LOG_MANIFEST、PRESERVATION、BUDGET_AND_FAILURE_CONTRACT、DELIVERY，以及四份最终日志的完整内容或结果段；另核对两个提交区间。[R1–R10]

基线到实现的改动为三份生产代码（共享 io、registry、runner）、一份执行层测试和归档的旧外部审阅。实现到发布只有本轮证据目录的新增文件。未把纯证据发布 SHA 当作被测实现 SHA。

本地保存的完整 `runner.py` 与 `io.py` 经独立 SHA256 计算，分别匹配发布绑定中的 `f160e2ab99a899ff2994a32cdc6db2322a73df0ca4df896d41f1a0f05f848a5c` 和 `dfbbdb9a9c2a6f99af449451f9dc690502af900f23a9e832c03594a934300daa`。详情见 SOURCE_IDENTITY.json。没有声称对所有仓库文件和所有日志完成了本地逐字节重哈希。

用户电脑上的新交付 ZIP、私有源清单、患者成员表和未脱敏原始日志未读取。普通网页访问和容器 raw 网络下载失败；连接器读取成功不等同于独立证明匿名公开访问。本轮不否认用户已完成的公开访问验证，也不冒充自己完成了该验证。

## 3. SP1–SP3 关闭依据

### SP1：路径边界 — CLOSED

`checked_path()` 在转换绝对路径之前拒绝 `..`，逐级拒绝软链接并要求父目录实际存在且为目录。`Output` 和输入 ordinary-file 检查共用它；`preflight` 对 output/source/storage/code/checkpoint/metadata 进行一致校验，再检查 storage 包含关系及受保护路径双向不相交。[R2–R4]

独立探针覆盖合法 preflight、不创建输出、输出逃逸/落入 source、非规范化 source/storage/checkpoint，以及 manifest/split/target/config 各自的非规范化路径、软链接父目录、缺失父目录和直接 Output 构造。元数据路径探针每次使用独立、hash 正确的配置夹具，避免被其他错误提前拒绝而冒充路径覆盖。

原 `approved_storage/../source/new_run` 形式不再通过。此结论不扩展为对恶意并发目录替换的 race-free 保证。

### SP2：首失败保留 — CLOSED

父监督器的 `persist()` 不再让证据写入或 stderr 回退异常替换既有首失败。它继续尝试必要终态记录，各文件不重试、不覆盖；若之前无失败，最后一份 supervisor 摘要写入失败也会令函数失败，并尝试尚未写过的错误记录。[R2,R5]

独立探针组合了 start/非零退出/timeout/cleanup 故障与磁盘写失败，另叠加 stderr 故障；预期主异常仍保留。摘要单独故障不能变成成功。合法监督路径仍能完成，并保持 `retry=false`。

### SP3：退出后资源审计及完成发布 — CLOSED

退出、cleanup 后的资源检查不再依赖轮询是否进入循环；父输出以真实全树字节而非本地 used 计数作判断，并保留六个 16 KiB 全局终态额度。日志、子目录 payload、父子记录均纳入树检查，非普通文件/硬链接输出被拒绝。[R2,R6]

`publish_completion()` 先审计，再写 completion.pending.json，再审计终态写入所耗的时间/字节，成功后才改名为 completion.json。worker 完成状态或 source_after_check 不合格、超时/超额时不发布完成文件。[R2,R6]

独立探针覆盖首次观察已退出时的超额、最后一次 poll 写入突增、恰好达到/超过时限、预留额度不足、全树与 local-used 不一致、nonzero+审计/证据故障、完成写入引起超时或超额及合法成功发布。特别使用同一 cap 约束 BudgetOutput 和监督器，验证无剩余磁盘证据额度时仍保留首失败。

## 4. 验收结果与独立复核分开计量

下表来自提交的固定版本公开日志与 CPU_RESULTS，不是本环境重跑结果。wall_seconds 为验收脚本记录，可能略大于 unittest 自身打印的用例计时。[R7]

| 环境 | 执行层 | R7 回归 | failures/errors/skips | 执行层 wall | 回归 wall |
|---|---:|---:|---|---:|---:|
| 本机 | 32 | 59 | 0/0/0 | 1.228053 s | 33.657539 s |
| 服务器 | 32 | 59 | 0/0/0 | 2.256901 s | 109.791883 s |

两端各 91 项；不是 182 种独立测试。每端提交的程序化成本为 497 forwards、36 backwards、2 VJPs、24 Adam 和 6 AdamW；其中新增执行层套件只包含 2 Small forwards、5 次合成 checkpoint 反序列化和 2 次合成图像/mask decode。日志中的 SOURCE_PREP_COMPLETION_WRITE_ERROR 对应故障注入用例，不是未报告的真实源训练失败。[R5,R7]

本轮另行运行 **40 个独立边界情形，最终 40/40 通过**。程序从完整 hash 匹配的源文件 AST 中只选择路径、preflight、监督、预算、发布等边界定义；没有导入仓库/torch，没有启动真实子进程，没有模型 forward/backward、真实 checkpoint 加载、图像 decode、真实源/目标资产读取或 GPU 查询。时间、进程、signal、常规 cleanup，以及 preflight 的 code identity/metadata audit 使用显式测试替身。它们不能冒充原 32+59 套件或完整 SOURCE_PREP 执行。

### 复核自身的首轮夹具错误（保留，不隐藏）

首轮独立探针结果为 39/40；失败项的夹具在 process.json 写入前就放入了超额文件，却预期先观察到 worker 非零退出。实际代码正确地先拒绝 process.json 写入，其 ValueError 成为更早的首失败，因此探针期望错误。

仅将超额写入移到首次 poll 的退出观察时刻，使其发生在 process.json 写入之后，以验证本来要测的“worker 失败 + 退出后超额”顺序。未修改被审源码；最终 40/40。`review_probes.initial.py` 和 `probe_results.initial.json` 保留原始过程，最终脚本与输出另存。

## 5. 伴随改动与科学协议保留

macOS EPERM fallback 新增的一次最多 0.5 秒 owned-child wait 没有重新发信号或重启 worker；成功仍要求 child 已回收且 ps 确认 PGID 不存在。独立替身分别验证 reaped、alive 和 group-present；这不是本环境在真实 macOS 上的运行证据。历史 stop_owned 文件未改动。[R2,R5,R8]

代码 diff 与 PRESERVATION 支持：A/B/C 方法公式、四份 science 字节、FULL/STATIC 独立同预算、fit/cal/val 职责、查询不更新状态、24/9/至多12 目标矩阵和历史记录未在本修复中改动。[R8,R9]

独立调用纯预算函数复核，111/23/25 对应 135328 forwards、9072 source backwards +1536 calibration backwards、3072 oracle Adam +1536 calibration Adam、6000 AdamW、1024 VJP。该项只有算术，无模型调用，不能用于推断真实运行时长。

## 6. PASS 的限制与下一道门

当前通过的是默认禁用、CPU-only 的执行层代码审阅。输出 cap 是轮询/终态核验与失败控制，不是物理文件系统硬配额；允许突发写入一度超额后被检出、保留并失败，不允许截断、删除证据或标记成功。时限从监督器开始，含 child/cleanup/终态写入，不声称覆盖 Python import 或元数据 preflight。[R6]

`supervisor.complete=true` 单独不能证明 SOURCE_PREP 成功；还需有效 worker 记录、source_after_check=UNCHANGED、通过资源审计的 completion.json 和父进程零退出。这些边界在消费结果时也必须保持。

下一步应是 **源绑定、明确风险解释、有限 CPU 资源与精确代码授权材料闭合**，不是新增算法实验，也不是立刻开跑：

1. 保留 111/23/25 冻结拆分，核对合法已登记 source/checkpoint/metadata 的文件身份。仅在既有明确权限下读取已知源文件原始字节；不得用未知身份、目标数据或目录猜测补齐。患者/眼别关联和预训练暴露无证据时继续 UNKNOWN；CONTENT 分组不证明患者独立，也不证明无预训练暴露。任何未知风险的接受都须显式记录，而非由文件 hash 推导。
2. 定义训练前可绑定字段与运行后才产生字段。manifest/split/checkpoint/config/目标登记等身份与资源可先绑定；effective tensor hash 和 trained asset hash 必须待相应获批加载/训练后生成，不能伪造，也不应要求训练产物 hash 在训练启动前存在。PENDING 不得仅因本代码 PASS 自动变成 BOUND。
3. 新 receipt 必须绑定实际启动的准确 code SHA、外部代码审阅及源/资源审查、manifest/split/target/config/checkpoint 摘要、CPU/线程/有限时限/输出/存储和唯一 SOURCE_PREP 用户授权。不能沿用旧 R6、Stage I 或 TARGET 授权。采用 f719c703 或其他纯文档后继版本时，都须明确实际 code_sha 与审阅覆盖关系；不能将发布 SHA 冒充已重跑测试的 SHA。

72 小时 CPU cap 和 512 MiB 输出 cap 仍只是既有拟议值，不是实际 ETA 或本次资源批准。当前无 GPU 支持/资源/资格放行，也不新增 GPU 迁移任务。代码没有再改且仅归档复审材料时，无需为了文档而虚构测试重跑，更不要求重跑旧166项。

## 7. 外部复审状态（未写回仓库）

```text
R7_SOURCE_PREP_REVIEW_PASS_EXECUTION_LAYER
reviewed_implementation_sha=f719c703087b38c07bdfbe7ce9dcfa62d88a12d9
reviewed_publication_sha=ecde8ba63940eb5c660f4de01bd7a9300b18c15b
stage_I_review=PASS
source_prep_execution_layer_review=PASS
SP1=CLOSED
SP2=CLOSED
SP3=CLOSED
source_binding_status=PENDING
real_source_training_started=false
real_target_execution_started=false
SOURCE_PREP=NOT_RUN
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
GPU_SMOKE=NOT_RUN
execution_authorized=false
```

## 8. 复现与包内容

```bash
python review_probes.py > probe_results.new.json
```

运行只使用标准库及临时合成文件；源文件 hash 不一致则拒绝继续。不使用实际 receipt、不导入完整 runner。包内 evidence/runner.py 和 evidence/io.py 是审阅输入副本，不应作为运行入口。最终/首轮探针、结果、机器审阅记录、源身份和 SHA256 清单均保留。

## 固定来源

[R1] 发布的审阅索引、FIX_REPORT、IMPLEMENTATION_BINDING；[R2] 实现 runner.py；[R3] 实现 io.py；[R4] 实现 registry.py；[R5] 实现 test_source_prep.py；[R6] BUDGET_AND_FAILURE_CONTRACT；[R7] CPU_RESULTS 及四份最终日志；[R8] FIX_REPORT；[R9] PRESERVATION 与提交比较；[R10] DELIVERY、LOG_MANIFEST。

固定 URL 与文件清单见 SOURCES.json。上述源材料事实以固定版本为准；本复审的独立测试结果以包内 probe_results.json 为准。
