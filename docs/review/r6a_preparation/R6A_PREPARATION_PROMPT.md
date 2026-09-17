# R6-A 启动准备 Prompt

本次外部审阅结论为 R6_REVIEW_PASS_A_ONLY，报告随附。
本任务仅整理启动准备，不是GPU/正式实验授权；未取得用户明确的新A执行授权前保持disabled。

## 固定对象

仓库 DLwbm123/DPA-CTTA。
Implementation SHA：c94fff7c05cec38d541c441b62a9381ee46ba076
Science SHA256：2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff
证据发布SHA：70e48d5da9737f806bd797b5c1752bd5b50a8e20
审阅scope=A；R6-B不在此次通过范围。

## 当前立即执行的范围

1. 阅读并原样归档 R6_EXTERNAL_REVIEW.md、R6_REVIEW_RECORD.json 及其证据范围。不改历史审阅/失败/BLOCKED记录，不把函数级探针称为完整166项重跑或GPU验收。
2. 核对干净的固定实现checkout、science原始字节、固定依赖与生产fingerprint。只能做只读元数据核对，不读取真实像素、mask、checkpoint，不查询或占用GPU。
3. 生成A的12-job metadata计划、设备及caps待授权字段、审阅引用和执行receipt模板。scope仅A；review.code_sha必须绑定上述Implementation SHA。用户未确认的设备/worker/caps不得从历史运行猜测。
4. reviewed code必须是上述固定实现。若归档文档产生新发布SHA，不自动把review.code_sha转移过去；实际run用已审干净工作树。不要为写入review而修改已审生产代码或science。
5. 不再设计新算法或增加不必要的CPU检查。出现实际SHA/hash不匹配时停止并报告，不自动修补/重跑。

外部review字段可以据本报告记录PASS，但execution.enabled仍为false。本prompt未提供用户执行授权、GPU卡号或有效执行receipt。

## 在用户另行明确授权A后，唯一允许的执行范围

- 使用同一既有checkpoint、registration与recurrence，固定四臂C/R_BAL/R_SCALE/R_SHUFFLE，orders0/1/4，共12条，各1951访问。
- 每图所有内容均适配，8forward/1backward/1Adam/0VJP；R5保持NO_ADVANCE，不复用旧C充当本次同期C。
- 正式总计23412访问、187296F、23412B/Adam、0VJP。
- 每实际设备一次已审smoke：OLD/C/BAL/SCALE/SHUFFLE各4步，160F/20B/20Adam/0VJP，程序化pixels索引0..3，seed20260907。
- 实际receipt中的配方名称为R6_ALL_ARMS4_OLD_C4_V1。只比较新旧C的parity，GPU rtol1e-4/atol1e-5、RNG/step按原合同。新臂不得要求与C相同，不根据toy Dice决定通过。
- 无source数据/代理/原型/训练，不加teacher/PCA/graph/kernel/回滚/融合。cap8、seed、loss、gate均不改。
- 用户明确设备、maxworkers<=3、每卡1worker、每worker2CPU线程、新私有输出和caps后才启动有限队列，不后台等待、不自动retry。拟议caps21600s/trajectory、86400s/wall、172800s/active-worker、8589934592bytes只是待批准配置，不是已授权值或预计时间。
- 保留已有neutral入口、NFS错误规则、进程组清理、失败live观测计数下界、首失败和原异常。
- 全12条完成后才做CPU标量核验/gate，不能按中间效果取消臂。缺失或坏记录为INCOMPLETE，不是科学NO_ADVANCE。
- A判定：BAL-C均值>=.5pp，BAL-SCALE/BAL-SHUFFLE各>=.2；两主序每个配对均>=0；recurrence三个配对各>=-.1；四域两主序BAL-C各>=-2。
- 输出全部子集/逐域/OD/OC/主序和回访分列、直接控制配对尾部/ASSD及分母、实际loss/残差/权重/梯度/Adam位移与成本，不能只挑平均正数。
- A结束无论结果如何都停止：R6A_COMPLETE_ELIGIBLE_FOR_REVIEW或R6A_COMPLETE_NO_ADVANCE；机械失败INCOMPLETE。全部next_execution_authorized=false，B_NEW不启动。

## 当前交付

只交付已审对象核对、12-job计划、待授权receipt模板与明确未填字段：
external_review=PASS（scope=A、固定code SHA）
execution_started=false
execution_authorized=false
R6A=NOT_RUN
R6B=NOT_RUN

不得把当前prompt自动转换成“用户已授权执行”。
