# TeSLA学习式策略：官方源码与历史组件覆盖核查

结论：**学习式策略不等同于已关闭R71，保留一个有限迁移契约问题；当前不准入合成或真实性能执行。** 不能把此前组件阴性解释成完整TeSLA或所有增强学习失败，也不能以完整组合救旧分数。

## 原文和源码范围

本轮读[CVPR官方摘要](https://openaccess.thecvf.com/content/CVPR2023/html/Tomar_TeSLA_Test-Time_Self-Learning_With_Automatic_Adversarial_Augmentation_CVPR_2023_paper.html)，官方PDF本次403、arXiv指定v2不存在，未称重新读完全篇。项目R70/R71留有此前全文公式审阅记录，作为历史定位。

本轮通过规定代理读取作者仓库固定提交`69870cc7d6f1caee48a6264ad54135224cb54c69`的5个文件：`networks/augmentation/{segoptaug,aug_predictor,med_seg_policies,rgb_seg_policies}.py`及`test_time_trainers/tesla_seg.py`。只作私有源码审阅，不执行、不发布第三方文件。以下为源码行为摘要，不是作者实现性能复现。

## 已核事实

- PolicyPredictor由零初始化的策略选择权重和幅度权重构成，不是必须下载预训练外模型的图像编码器；在线策略学习也不能误称新增源训练。是否准入当前新参数仍需完整事前协议。
- 策略Adam中概率与幅度学习率分别为.001、.01；选入子策略的学习时采样、幅度梯度和概率更新有自己的状态，不是对固定候选一次取最大值。
- 官方预测使用类别softmax；学习含熵、归一化层输入均值差及与顺序累计预测的KL。代码虽记录方差，所核特征正则实际使用均值。测量这些均值不等于获得BN尺度更新准入。
- 医学/RGB策略表不相同，包含多种光度与空间操作、幅度范围、clamp以及逆几何。当前输入预处理与支持处理必须另核，不能原样套用到独立sigmoid两头，也不能移除不便操作后冒称完整原算法。

## 历史覆盖不是整体否证

| 历史组件 | 封存结论与未覆盖部分 |
|---|---|
| [R70 flipped-CE](../r70-flipped-ce/REPORT.md) | FCE−NATIVE −.212879pp，门失败；更换强视图损失位置，不是完整TeSLA。保持关闭，不调epsilon/LR或并入策略救分 |
| [R71 entropy-view](../r71-entropy-view/REPORT.md) | SELECT−NATIVE −.023882pp，CI跨0；SELECT优于退化RANDOM不改主门失败。测试固定gamma集合的有限选择，不含原版持续概率/幅度学习。禁止gamma/系数/种子扫描 |
| [R72 crop-target](../r72-crop-target/REPORT.md) | 固定两crop目标路线关闭；不能将学习策略当借口重开crop/ROI/尺度搜索 |
| R20/R61/R62及P1/E8 | 已涉及教师/视图/目标/平移或特征读出，不由此推出所有顺序策略等价；但候选若仅重命名这些操作不获得新依据 |

上述数据来自本地公开封存报告，无新评分、轨迹或GT访问。原旧192内容/两顺序只是历史证据，不能自动恢复为当前数据权限。

## 下一有限契约问题

只核一个“学习式光度策略组件”能否被完整定义，原模型和当前8内容不扩。首先核信息流：当前全图min-max前后，候选操作是否仍有非平凡作用；正比例亮度在无截断时可能被消去，截断造成的变化也不能冒称真实结构。必须从已核源码与既有预处理定义确定，不看8图或GT找设置。

契约必须一次说清：几何不变的操作及幅度来源、归一化/clamp顺序、二头Bernoulli熵/KL、running-average的顺序/重置、概率与幅度优化/RNG、encoder/BN/梯度/优化器状态隔离、策略实际执行预算。任何必要删减或损失替换应明示原创组件，不能称原TeSLA复现。

有完整独立设计依据才另冻纯合成资格与 matched-random/frozen-policy/identity强控，资格只核导数/状态/参数边界/信息流而非任务收益；再另立真实性能协议。当前无操作子集/数值协议/source/root/派发，不声称正在训练或通过资格。若与旧已关配置等价或无法定义，记录缺口关闭，不扫描光度/空间参数，也不加入teacher/fCE/marginal组合救分。
