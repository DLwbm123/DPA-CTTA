# E1 翻转对应蒸馏：冻结小规模强基线协议

## 唯一主要变化与数据

2026-10-19。新独立实验，HRAC启发的Bernoulli翻转蒸馏代替熵，非HRAC复现。8个既有重复开发RGB、既有checkpoint、seed20261018、两种顺序、每序5块[0,1,2,3,0]与40到达均不扩。患者与预训练成员未知，无独立患者结论。RDC/ENGRAM关闭，无源训练、标签、未来图或新投影/额外模型用于适应。

令z=fθ(x)，t=stopgrad(fθ(flip(x))反翻)。两头独立，L为2×512×512上自然对数Bernoulli KL(sigmoid(t)||sigmoid(z))均值，FP64计算稳定logsigmoid，教师仅no_grad前向，原图分支唯一反传。翻转作用于现有归一化输入宽轴，无重新归一化/crop。对应控制ROLL在反翻后再沿宽轴roll(W//2+1)，每头边际概率保持，空间关联破坏，不随机选偏移。SOURCE_FLIP使用固定源模型同一翻转输出，只有该控制访问固定源教师，不增加源KL。ENT_NORM使用原自然熵，无翻转教师。

每图1次Adam(lr1e−4,betas .9/.999,eps1e−8)，仅82个标准BN仿射张量19136坐标。源runningstats、conv/head固定；所有core实际FP32位移L2=.01±5e−7且原proposal方向cos>=.999999，沿用既定ULP修正，无zero。无参数范数相等即函数范数相等的声称。L非finite/翻转KL<=1e−12/BN梯度norm<=1e−12/原proposal<=1e−12/方向或半径不通过则硬停，不fallback、不强造方向。无熵/边界/包含损失/伪标签门控/源锚组合。

## 矩阵、状态与覆盖

FLIP RESET/FRESH/RETAIN，ROLL FRESH，SOURCE_FLIP FRESH，ENT_NORM RESET/FRESH/RETAIN，共640core到达。RESET每图清仿射与Adam；FRESH留仿射每图清Adam；RETAIN两者保留。真实CW1/CW2各RESET/RETAIN另320到达，原生currentBN语义与冻结源不同，须披露。5静态为SOURCE_EVAL/CURRENT_BN/旧D/R/平移R。总24轨迹、960科学到达、2320PRE/POST/静态读出、29绝对聚合行。不是独立患者数。

额外FLIP RESET正序完整40到达，80对PREPOST logits及末model.state_dict逐元素一致。STRICT deterministic warn_only=False，不支持算子或不一致停。仅同进程这一流，不认证全部轨迹/Adam/RNG/跨进程。源模型最终参数/buffer完全不变；RESET源返回掩码一致。GPU前CPU6项资格覆盖公式/布局/detach/冻结/实际规范步幅/Adam重置及配置拒混版，合成输入非科学效果。

## 主比较与推进门槛

唯一主FLIP_FRESH−ROLL_FRESH:PRE，均值>=+.5pp且描述性95%区间下界>0。固定seed2000次8内容bootstrap绑定所有条件/顺序/通道，非独立患者推断。还须FLIP_FRESH PRE对SOURCE_FLIP_FRESH和ENT_FRESH下界>0；首块与全流PRE对SOURCE下界>0，最差>=−1pp；源返回PRE对SOURCE均值>=−.2pp。POST均值须高于五静态、ENT_FRESH、FLIP_RESET及nativeCW四臂；无新增空/嵌套，ASSD/HD95/vCDR与定义覆盖不劣源；640规范更新全匹配无zero。任一失败advance_development_gate=false。

各头/条件/顺序、首曝/返回/PREPOST、前景与3px边界纠错损害、实际函数变化/参数norm/Adam年龄全报。q8概率只作近似Brier/ECE辅助，误差<=1/510，不驱动优化/Dice、不能替代主终点。对源ROI宏平均不做像素独立推断。fixed_teacher_loss_post只指对这次已detach教师，不是重算新的翻转一致性、不是语义正确性。对ROLL大幅胜出不能替代对源/RESET正益，不选SOURCE_FLIP或其他副臂事后换主。

## 资源、安全与停止

唯一GPU派发，硬1200秒、独立CPU评分900秒、输出2GiB。首2FLIP RESET实测每到达均时（含所有前向/实际步幅检查）投影=已用+(640+40−2)×均时×1.35+320native+60，须<=1180；未准入则保留停止，不砍控制/降系数/延预算/重跑。所有core与重演同计费。GPU前CPU/NAS实际挂载容量小读写/free>=8192MiB/中性入口父子ps及nvidia检查，不能干预其它任务。

历史累计GPU3932.1413059234033秒、已计CPU1854.4658777418797秒，再加本轮准备及完整ledger；旧本地.631近似、公式.0010669171、历史CPU基数.0084569158差/未计时阅读编排单列，不清零。适应worker拒H1整root及scorer/旧分数，仅8登记RGB/既有checkpoint和旧冻结head。标签只在GPUworker exit0退役后独立CPU读8份。

完成需GPU_COMPLETE2320/960、FULL_REPEATABILITY PASS40/80/model_exact、完整MECHANISM/ACCESS、640匹配无zero/源不变、worker与score exit0 inactive、SCORE_COMPLETE8标签29行、最终ledger/controller原身份退休。工程停止不是算法负结果。匿名聚合/无路径receipts/自有源码/中文报告推送公开GitHub，验证remoteSHA与匿名HTTP200；不公开图像标签权重概率逐例评分身份私有路径或第三方全文。不调翻转/阈值/步幅扫参追分。
