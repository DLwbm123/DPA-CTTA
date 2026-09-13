# 阶段 I 交付与外部 review 范围

目标是对一个固定提交做集中方法审阅，不重复搭建工程gate。审阅先检查五个机制是否真实、对照是否匹配、是否符合checkpoint-only和因果数据流，然后才批准GPU机械验收。

## 必须回传

| 材料 | 内容 |
|---|---|
| 完整 implementation SHA | 包含全部17臂；不得只给分支名或材料发布SHA |
| `IMPLEMENTATION_REPORT.md` | 实现了什么、哪些实际运行、哪些未运行、残留差异 |
| `REVIEW_INDEX.md` | 逐方法关键函数、行号和测试入口，完整代码归档/patch入口 |
| science JSON与摘要 | 本提案无变化则字节保留；有冲突必须明示，不自称已批准 |
| 基础登记与新stream摘要 | 基础registration不变；新stream单独digest及纯元数据覆盖证明 |
| `STATE_LIFECYCLE.md` | 参数/Adam、measurement clone、bank、slot、snapshot/frame、RNG生命周期 |
| 完整 patch | 相对f52f132基础；共享执行层任何改动单列 |
| 真实CPU日志 | 包内参考测试＋仓库集成/相关回归；首失败及修复记录不删除 |
| dry-run JSON/CSV | 85唯一job、每job调用数及1/2/3worker轮换 |
| `METHOD_PROVENANCE.md` | 原论文内容、摘要支持内容、本项目改造、未知项分开 |
| `UNRUN_ITEMS.md` | GPU/真实目标/正式实验都未运行；不能用toy通过代替 |

发布源码与审阅文档到新公开分支，不上传权重、论文PDF、私有数据、路径/身份。原始CPU日志可能含实路径，保留私有原版，公开去身份副本注明去掉什么；不能为了匿名删除错误事实。

## 方法审阅重点

### 共同

C/RP是否对齐旧实现；新介入不改变弱视图/增强随机顺序；标签只到evaluator；每轨迹独立初始化。只有当前图tensor进入算法，图像ID/域名不得进入router。

### T

冻结测量器真冻结，使用给定checkpoint源buffers；q标签来自active C而不是source label；前景/背景密度比较；ISO/DIAG/LR协方差区别；logdet、收缩、证据尺度、回退明确；delta上采样而非q降采样；未ready原样C。

### U

first-original graph连到真实BN参数，非叶子feature重开requires_grad不算；Jacobian方向和参数次序；实际Adam proposal仅一次；projection是参数位移、moments按原g；随机与范数控制正确；额外VJP全部计账。

### S

只保存3份小状态而不是3个完整模型；global RNG不随slot切换；每slot optimizer步数/数据支持计数正确；新建不读取域边界；S_SHARED确实共享参数，S_NOPCA不把RP项留在loss；只有被选slot修改。

### M

正确同位置配对；pre选择/post特征存储；identity-post控制必有；Q左右乘方向；live和cached mean/basis一起迁移；frame与eig版本分开；当前图不反向改变自己的loss。

### G

独立Bernoulli KL方向、graph与包含projection公式；q、graph、basis无反向；1984边而非稠密N²；可靠位置权重；全分辨率teacher处理保持高分辨率原q；最终输出未强制嵌套；zero-edge ORDER匹配。

## 非阻断的科学风险

密度可能错误，正交迁移可能不拟合真实表示变化，probe不能代表所有历史样本，router可能只激活1个slot，图可能过平滑。只要代码忠实实现本方案，就让固定实验回答，不加“效果必须先变好”的审查。

源码确实有错误时提交窄修复；不借review修改seed、lambda、rank、门控或数据范围。review是对指定code/config的外部意见，Codex不能自签通过。
