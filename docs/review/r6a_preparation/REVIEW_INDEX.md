# R6-A 启动准备交付

**R6A_PREPARED_AWAITING_EXECUTION_AUTHORIZATION**。

external_review=PASS，仅限scope=A及实现 `c94fff7c05cec38d541c441b62a9381ee46ba076`；execution_authorized=false、execution_started=false、R6A=NOT_RUN、R6B=NOT_RUN。本次只归档审阅、核对固定对象和生成元数据草案，没有GPU查询/初始化、模型/真实像素/mask/checkpoint/source读取、CPU测试重跑或后台任务。

## 固定对象与审阅来源

Science SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`。
生产fingerprint：`a6ef161066d3560dc2dd540cb507bff2e4a7cadf94ab711545877c525dee2f2e`。
阶段I证据发布：`70e48d5da9737f806bd797b5c1752bd5b50a8e20`。

- [外部审阅报告原件](R6_EXTERNAL_REVIEW.md)、[外部机器记录原件](R6_REVIEW_RECORD.json)、[本次准备prompt原件](R6A_PREPARATION_PROMPT.md)：三文件均原样归档，没有重新序列化JSON或改写审阅结论。
- [已审对象核对](OBJECT_VERIFICATION.json)：服务器既有执行checkout为干净的c94fff7，science原字节/两个依赖提交/生产fingerprint均匹配；本地文档工作区基于70e48d5，src/scripts/tests/configs与已审实现无差异。
- 原实现config与已归档原提案逐字节一致。没有把此次文档发布SHA转移到review.code_sha；未来实际run必须使用已审干净checkout，不能用本准备分支HEAD代替。

PASS来自用户提供的外部记录，不是本助手签发。审阅者明确只执行五组函数级CPU探针，未重跑166项/完整模型/GPU。本次三附件没有包含所提探针源码/JSON/log证据包；本交付仅引用该报告，不声称取得或独立复跑探针。历史阶段I文档、失败/BLOCKED记录和测试日志全部保持不变。

## A的固定计划与未签发模板

- [A_MATRIX：12 jobs](A_MATRIX.json)、[A计划/预算/gate](A_PLAN.json)、[去身份回访流摘要](STREAM_SUMMARY.json)。由既有plan和原registration JSON生成，只读元数据；四臂C/R_BAL/R_SCALE/R_SHUFFLE×orders0/1/4，每条1951访问，remaining_dev1695只作主评分。没有追加B或其他矩阵。
- [disabled授权模板](AUTHORIZATION.disabled.json)：enabled=false；review为PASS/A/c94fff7，reference指向本包实际附带的报告文件，不虚构外部review URL。approved_*键名沿运行接口，预填值仅为冻结对象绑定，绝非已经获得用户GPU执行授权。
- [receipt草案](RECEIPT.draft.json)：DISABLED_DRAFT_NOT_EXECUTION_RECEIPT，run_id=null、devices=null、schedule=null、private_output=null、output_created=false。它不是实际receipt；本次没有创建正式私有输出目录。
- [交付状态](DELIVERY.json)、[准备元数据记录](PREPARATION_CHECKS.json)。不修改已审生产代码、science或execution defaults，不重新设计算法，不增加CPU数学/模型测试。

正式预算：23,412访问、187,296 forward、23,412 backward/Adam、0 VJP。每个实际设备smoke另计160/20/20/0，配方名称严格为R6_ALL_ARMS4_OLD_C4_V1，OLD/C/BAL/SCALE/SHUFFLE各4步，seed20260907、pixels0..3。只要求新旧C parity，GPU容差rtol1e-4/atol1e-5；未执行GPU资格验证。设备数g未授权，因此总预算仅记187296+160g forward、23412+20g backward/Adam，不填实际g。

A gate保持原阈值和完整12条先验要求。A结束无论通过与否均停止；可用终态为R6A_COMPLETE_ELIGIBLE_FOR_REVIEW、R6A_COMPLETE_NO_ADVANCE或机械失效INCOMPLETE。所有next_execution_authorized=false，B不获本次审阅或执行授权，不自动重试、换seed、调cap或扩展矩阵。

## 当前未填字段

需另行明确的新A执行授权，以及实际物理GPU列表、max_workers（最多3、每卡1worker）、四项正数caps、全新私有输出路径和后台运行策略。上述字段全部未填，不从旧R3/R4/R5的GPU567或waiver推断。

拟议caps仍为21600秒/trajectory、86400秒/wall、172800秒/active-worker、8589934592字节，仅供未来明确批准，既非批准值也非预计完成时间；每worker2CPU线程保持固定方案。private_output=null确实表示尚未选择/授权路径，不是隐藏已填写的私有绑定。

已审服务器checkout路径仅记录在本地私有准备说明，不在公开材料披露机器路径。CPU依赖核对未加载给定checkpoint或探测GPU；未来授权后的设备/存储启动检查仍须当时执行。本次准备至此结束，不后台等待授权。
