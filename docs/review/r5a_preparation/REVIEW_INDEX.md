# R5-A 启动准备

**R5A_PREPARED_AWAITING_EXECUTION_AUTHORIZATION**

代码复审 PASS 仅覆盖 A；execution_authorized=false、execution_started=false、R5A=NOT_RUN、R5B=NOT_RUN。没有 GPU 查询/初始化、真实 checkpoint/像素读取、后台任务或正式运行。

实现 SHA：`b4b71601a5bdf87bb3a7e5d3db352610adcdff74`。本次仅新增文档/草案，不改代码或 science；执行必须使用该干净 detached checkout，不以本次文档发布 SHA 替代。

- [真实外部复审报告](R5_REREVIEW.md)、[准备 prompt](R5A_PREPARATION_PROMPT.md)、[审阅登记](REVIEW_REGISTRATION.json)。授权草案的 external_review.reference 已锁定真实报告归档提交。没有自行生成复审 PASS；该字段也不授予设备权限。
- [disabled 授权草案](AUTHORIZATION.disabled.json)、[receipt 草案](RECEIPT.draft.json)、[三条矩阵](A_MATRIX.json)。只含 C/order0、C/order1、C/order4，每条 1951 内容；calibration=null、reuse_A=null，未填写 RANDOM 参数。receipt 是未签发的草案，run_id=null、devices=[]、schedule=null。
- [固定 checkout 记录](CHECKOUT_RECORD.json)、[准备检查](PREPARATION_CHECKS.json)、[交付清单](DELIVERY.json)。本地新建执行工作区，服务器复用已测试的干净 detached 工作区；两个固定源码依赖 SHA 与 cleanliness 已检查。已知项目 jobs 目录仅发现两次 CPU 检查，未发现正式 A 候选；这不是全服务器扫描。

尚缺用户独立授权 R5-A、实际 GPU 卡号、worker 数（最多 3、每卡 1 worker）和四项正数 caps：trajectory_seconds（单轨迹秒数）、wall_seconds（整批墙钟秒数）、active_seconds（累计进程活动秒数）、bytes（输出容量字节）。均未猜测或沿用旧 R4/fixture 数值。background_allowed=false，不表示批准后台运行。

私有草案另行绑定真实已授权 registration、固定依赖/解释器路径和未创建的新私有输出路径。公开件对机器身份、私有输出路径和 registration 逐内容身份去除；公开 receipt 的 private_output=null 是隐去该私有绑定，不是授权缺口。未来启动前须再次确认目标未被占用，不能覆盖既有结果。

A 正式预算 5853 访问、46824 forward、5853 backward/Adam、0 VJP。每实际设备的原样 smoke 另计 64/8/8/0，seed=20260907、pixels 0/1/2/3、rtol=1e-4、atol=1e-5。设备数 g 尚未授权，总成本表达式留作 46824+64g forward、5853+8g backward/Adam；不执行 smoke 来决定卡号或放宽容差。全部设备 smoke 通过后才可正式执行，A 无论 gate 结果如何都停止，B 不获授权。

本次验证为 metadata 和 disabled 入口检查，两处均在真实资产/设备调用前拒绝执行。science SHA256 与 C fingerprint 保持冻结值；没有重跑 143 项，也没有将历史 CPU 日志冒充新运行。

附件提及 REVIEW_DECISION.json，但未随本次文件提供且 Downloads 文件名搜索未找到。已如实登记缺失；机器可读的 REVIEW_REGISTRATION.json 只是对已提供 Markdown 报告的转录，不是冒充外部审阅方补签的决策文件。独立探针 bundle 也未随本次附件提供，其结果仅按报告引用，不声称亲自复跑。

当前准备任务到此停止，不轮询等待授权。
