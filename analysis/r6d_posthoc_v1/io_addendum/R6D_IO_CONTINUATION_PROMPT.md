# Codex 续办 Prompt：R6-D 固定版本输入适配与一次只读续跑

本任务是对已停止的 R6-D 输入预检失败进行局部工程修复，并在修复测试通过后，执行一次新的前台只读标量诊断。它不是原任务的后台自动重试，不是 R6-B，不是新增模型实验，也不是修改 gate。

本 Prompt 作为用户新下达的续办任务时，允许：修复隔离分析层、程序化 CPU 验收、读取已有权限下的既有标量账本和注册元数据、在全新私有目录做一次固定诊断。此文件本身不是远端已经执行或测试通过的证据。

## 1. 输入、固定绑定与保留事项

本补充包包含本 Prompt、R6D_IO_ADDENDUM.json、R6D_IO_DECISION.md、README.md、SOURCE_REFERENCES.json 和 MANIFEST.json。先核对补充包 MANIFEST；不等待新的数学参考或额外未交付材料。

已有原始分析材料在仓库 analysis/r6d_posthoc_v1/input/；其原 MANIFEST、原计划、原 Prompt 和原 R6D_ANALYSIS_SPEC.json 必须原样保留。原规格 SHA256：
ffd3c648931e27130c2c275da7ca83f7bac62906b9e9e3d51ec4b3fd62d07368

旧分析实现：ad75672e2931c5dd09ffb19f3249d6b65ec6aa0d
旧分析失败发布：fad933d9c49dd1ae034a9368785381fb89c32cf4
原 R6-A 结果发布：aa732b42b03d378149028a3770879b72ebc55db3
原模型执行：c94fff7c05cec38d541c441b62a9381ee46ba076
原 run_id：a70dc79f23db47289532792141b61f57
原 science SHA256：2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff
原 registration digest：8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf
原 stream digest：cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db
原 production fingerprint：a6ef161066d3560dc2dd540cb507bff2e4a7cadf94ab711545877c525dee2f2e
原 scope：A

原 R6-A 的 EXECUTION_AUDIT.json 已记录 current_result.result_directory 为：
results/7b2e24f345f0463fa8ee451888650e66

先检查工作区，保留用户改动；建议新分支 analysis/r6d-pinned-input-fix-v1。不得重写 main、历史提交、原科学配置和旧运行结果。原失败尝试保持 R6D_INCOMPLETE，旧报告、日志、NOT_AVAILABLE 表原样保留；新结果另建目录并引用 prior_attempt。

## 2. 根因与本次唯一协议修订

原发布器 src/dpa_ctta/r1/evidence.py::publish 使用：

    source/public_aggregate.json -> current/public_aggregate.json
    source/current -> results/<version>
    source/current_result.json.result_directory = results/<version>

旧分析 run.py::allowed_sources 把顶层 public_aggregate.json 当成普通数据文件，再用 regular() 拒绝了正常发布入口。

修订仅限输入适配：两个已知发布别名允许做 lstat/readlink 元数据检查；分析内容从权威指针绑定的普通版本文件读取。快照仍禁止符号链接和硬链接，其他来源文件仍受原普通文件规则约束。

不能把“任意跟随符号链接”作为修复。不能删除 regular()、移除 O_NOFOLLOW、使用递归 cp -L、修改原链接、chmod 源目录，或调用原 recompute/invalidate/publish 重建结果。

这是明确的 I/O 规则补充，不是科学公式修改。保留原分析规格字节和摘要，新增 input_adaptation_addendum_sha256；不得宣称原 Prompt 的输入规则从未调整。

## 3. 读取已绑定的固定版本，不读取动态别名的内容

A. 从既有授权配置获取 source、code、全新的 work；不要猜测私有绝对路径或扫描不相关目录。确认 source 根与 code、work 的实际关系，work 不得与 source/code/以前尝试重叠。所有新文件仅写 work。

B. 在内容读取前建立最小标量白名单及失败遥测。current_result.json 必须先作为普通、单硬链接文件读取，记录其原始字节 SHA256、长度及文件身份，保留原始字节，不重新序列化代替原件。校验：
- valid is True；status 恰为 R6A_COMPLETE_NO_ADVANCE；binding 与上列完整历史绑定相同。
- result_directory 是源根内相对版本路径，符合 results/<32个小写十六进制字符>，且必须与已发布原 EXECUTION_AUDIT 指定的版本一致。
- 禁止绝对路径、..、额外路径分量、跨源根路径；不能扫描 results/ 后按 mtime 或分数另选“最新/最好”的版本。

C. 仅记录和检查两个发布别名的元数据，不经它们打开内容：
- lstat(source/public_aggregate.json) 为 symlink，readlink 原文为 current/public_aggregate.json。
- lstat(source/current) 为 symlink，readlink 原文为上述已绑定 result_directory。
- 指针、两个链接和实际目标必须一致；悬空、循环、越界、目标不符均失败，不自动切换版本。
- 链接类型、readlink 原文、身份、mtime/ctime/mode 等单独存入 link audit；readlink 字符串摘要不是目标内容摘要，不混用。

D. 实際数据源为：

    source/results/7b2e24f345f0463fa8ee451888650e66/public_aggregate.json

从已绑定 source 根向下的 results 目录、版本目录不得是链接；最终目标必须普通文件、st_nlink==1。保留每项白名单源文件原有普通文件检查。不得只用 is_file() 或 resolve() 判断“源是普通文件”。

使用只读打开和不跟随链接的检查；在已支持的 Linux 环境可从源根目录描述符逐级 O_DIRECTORY|O_NOFOLLOW 打开，到最终文件用 O_RDONLY|O_NOFOLLOW，并用 fstat 确认实际打开对象。保留现有只读策略，不允许无保护的静默回退。记录实际采用的保证与平台限制；不要声称单次路径检查能防住所有并发替换。

从同一个打开的源文件描述符流式计算 SHA256 并复制到快照普通文件。复制前后核对 fstat，再对路径身份复核；大小、mtime/ctime、inode/device、nlink 等发生改变时拒绝。原 atime 排除说明保留，不通过修改源时间戳来“恢复”审计。

E. 维护显式映射：

    logical_name = public_aggregate.json
    source_relative_path = results/<已绑定version>/public_aggregate.json
    snapshot_relative_path = public_aggregate.json

新 snapshot/public_aggregate.json 是独立普通文件，包含上述版本聚合的原始字节，不是链接，也不与源共享 inode。顶层逻辑名称只供旧纯标量验证器读取，不能用它冒充真实源物理路径。真实映射只放私有记录，公开用 file_token、摘要和长度。

完整快照白名单、完整输入行数、原生产指纹、原 source_output_bytes 的元数据口径均保留；不得漏掉顶层聚合后假称完成主结果核对。install_guard 的允许读取集合只加入指针和已绑定的普通版本目标，不把整个源目录或任意链接加入内容读取白名单。

## 4. 前后只读核对与异常处理

在读取版本前、快照完成后、分析结束时核对指针和别名仍一致；所有白名单源负载都做分析前/后字节核对。要求：

    原版本负载before SHA256 == 快照SHA256 == 原版本负载after SHA256
    原current_result原始字节before == after
    两个发布别名的readlink和被保护元数据before == after

不访问动态别名以获取聚合内容，不改变原 current_result/current/public_aggregate。源目录其余符号链接只按既有 inventory 做元数据记录，不遍历目标；不得把白名单字节核对写成全目录每个字节都已核对。

将预检也纳入 wall cap、阶段标识、异常捕获与成本记录。这是与本次阻塞直接相关的记账修复，不扩大分析范围。

主异常与 finally 中复核异常分开保存。source_record、指针/链接检查或 after-check 再出错时，不覆盖第一个异常，也不能使已有失败被最终“成功”文件掩盖。未取得的 before/after/hash/资源记录填写 NOT_CAPTURED/NOT_PERFORMED/UNKNOWN，不补0、不填true。保留原先失败尝试的未知成本；本次新增遥测不能追溯补齐旧尝试。

只有全量分析、只读核对和最终解释都完成，才发布完整完成状态；中间 public/ 输出仍不等于终态。失败保留本次独立尝试，禁止自动重跑或切换到公开聚合伪装成功。

## 5. 最小修改范围与验收

修改仅限：analysis/r6d_posthoc_v1 的输入适配、run 的预检/异常记账、新 I/O helper、相应测试/README，以及新增本补充规范。

core.py 的科学分析公式、配对、分组、窗口、权重、空值规则及原分析规格不变。只读 helper 确需调整可局部改动并列出；不能借 I/O 修复修改 validators 容差、原 src/、模型路径或统计结论。

先运行已有25项标量测试，再增加以下覆盖；实际最终数量据日志报告，不预填通过数：
1. 与旧 publish 等价的双层别名布局；从固定版本得到相同字节、普通独立快照，并能进入已有完整分析 fixture。
2. 原普通文件保护仍拒绝内容源/快照中的符号链接、硬链接、FIFO和目录；不能为兼容 fixture 降低生产合同。
3. 指针版本、current 和 public_aggregate 别名一致性；悬空、越界、循环、..、绝对路径、错误 run/status/binding 明确失败。
4. results 或版本目录为链接、最终目标为链接/硬链接时失败。
5. 多个版本同时存在时只选择已绑定版本；不能按最新时间选另一个。
6. 指针/链接在读取阶段发生切换，或源文件复制时改变，快照被替换/污染时拒绝。
7. 内容读取白名单与真实映射一致；源写入、调用旧mutator或模型/GPU路径仍被拒绝。
8. 预检失败也产生日志/成本边界；主异常加 after-audit 异常同时存在时首错保留。

使用程序化标量/目录 fixture，不调用生产 publish 来改任何真实源目录；可复制其布局语义构造 fixture。测试隔离 audit hook 的辅助进程只能用于CPU测试；真实固定分析仍单进程。

在最终修复候选 SHA 上完成本地与已有服务器环境的标量套件；记录真实环境、退出码、失败与跳过。不要重跑166项完整模型回归，也不要重新索要原本已具备的只读账本权限或制造一轮GPU式READY/PASS流程。

固定候选 SHA 后，校验原analysis spec SHA和原science SHA不变；新I/O补充摘要独立记录。任何科学冲突停止并报告，不自行选一个版本。

## 6. 一次新的前台固定诊断

修复测试和只读预检通过后，按原 R6D 分析规格在一个全新私有 work 中执行一次。保留旧失败目录，不就地复用。

预算保持：单进程、最多2 CPU线程、wall cap1800秒、新输出cap2GiB（快照I/O/存储单独记录并预检）。启动计时与错误保护要覆盖预检，不在失败后补估耗时。使用既有权限配置，不轮询、不后台重试。

仍复核12条轨迹×1951内容，两类日志各23412行，再完成原定：
- Delta_post = Delta_pre + Delta_step；
- 同内容跨流配对、注册段和固定切换窗口；
- 监督贡献、q误检/漏检、区域加权硬错误质量；
- C锚定固定分层，全反例、空值与分母；
- 支持/反证/不可识别，至多一个新问题。

原 gate 只复算不更改，R6-A仍NO_ADVANCE。新表全部标POST_HOC_EXPLORATORY。只有诊断实际支持时可提R_HALF，不实现、不运行、不授权R6-E。

若发现新的真实完整性或权限问题，保留首错，仍为R6D_INCOMPLETE；本次已知私有账本存在，不能为绕开失败而改报aggregate-only。任何新科学结论均须等待真实表。

## 7. 交付与终态

建议新公开结果目录 docs/results/r6d_posthoc_v1_continuation_01/，保留原 docs/results/r6d_posthoc_v1/ 不变。报告关联旧失败尝试和此次修复，实现SHA与证据发布SHA分开。

交付：窄patch、IO_ADAPTER_REPORT、原规格摘要与新补充摘要、真实测试日志、READONLY_AUDIT、ANALYSIS_BINDING、字段可用性、全部原计划表和HYPOTHESIS_DECISION、DELIVERY。链接元数据和真实来源映射私有保存；公开不含逐内容ID、私有绝对路径、PID或设备UUID。日志去身份不删除异常。

成功且解释已完成后按实际选：
R6D_COMPLETE_HYPOTHESIS_PROPOSED
或 R6D_COMPLETE_NO_NEW_HYPOTHESIS

失败：R6D_INCOMPLETE。
工具测试通过不等于真实诊断完成；ANALYSIS_TABLES_COMPLETE_INTERPRETATION_PENDING也不等于最终结论。

始终保留：
new_model_execution_started=false
new_model_forwards=0
new_model_backward_calls=0
new_adam_calls=0
new_vjp_calls=0
R6A=R6A_COMPLETE_NO_ADVANCE
R6B=NOT_RUN
next_gpu_execution_authorized=false

到此停止，不进入R6-E、新轨迹或阈值搜索。
