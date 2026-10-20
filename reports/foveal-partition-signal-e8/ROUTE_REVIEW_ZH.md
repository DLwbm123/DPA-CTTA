# SphereTrust与FoRE全文审查及E8迁移依据

## 阅读与来源

SphereTrust：arXiv 2609.30080v1官方PDF29页，正文及A–G科学附录/全部表格已读，参考文献浏览，p4公式目视核对。FoRE：OpenReview wlT0Oy0zY1官方PDF47页，正文及A–V科学附录/全部表格已读，参考文献浏览，p3几何目视核对；未逐页视觉检查全部彩图。命令行Sphere下载成功，FoRE命令行403保留，随后正常浏览器官方Download PDF成功；未导出凭证、无第三方代码执行/权重加载。全文与图片仅保私有研究目录，不公开。

[SphereTrust](https://arxiv.org/abs/2609.30080)，[FoRE](https://openreview.net/forum?id=wlT0Oy0zY1)。以下为本项目推导与设计，不继承论文增益。

## SphereTrust到底提供什么

§3.1/angular contrast把单位特征分为掩码内外两群，比较两群归一化和向量方向；coverage使用候选池共识core与边框背景prototype建立至多4个模式，frame-contact是边框占比，最后每图候选池z-score等权融合。A.1明确DINOv2 ViT-B/14、350输入/25网格和候选面积/至少3patch等过滤。选择阶段免训练；§3.2的student/跨fold prototype/完成mask与训练配置不免训练，配置还读1085标注验证图，不能把其完整下游收益搬来。

原文也披露：F不区分mask与其补集，C增加前景时单调不减，可奖励泄漏；B.1的coverage优势随候选池改变符号，四个医学集不在本论文主scope。B.2中SAM自有encoder明显弱于DINO，不支持“任意分割器特征都一样有用”。F.3统一prior尺度后对面积控制优势在部分任务反转。E6的原始cosine失败因而必须保留，不能以此论文承诺源32维特征有效。

本项目B加每个单块撤回构成的pool，若有至少2个不相交块，逐像素多数core就是B本身（每个改变只出现在一个候选里）；如此直接搬coverage会让待审核基线参与定义参考。OC微编辑通常不碰图像外框，E项差也可能恒0。故不复制完整F+C−E或事后调权重；只取明确的候选相对角分区差，并让旧表示/插值/源与裁剪logits作强控制。

## FoRE到底需要什么

§2–3、B明确DINOv3-L/16与9,207,393可训练调制参数、15000无标签图/24000步；换骨干要重训调制器。它虽不训练backbone/不读标签，仍有新模块与训练，不符合当前边界。few-shot/probe readouts还需标注support/拟合头，不能当作我们免费已有的纠错监督。

可复用的是§2.1 control teacher Tv：只裁当前全局view里的像素、同一冻结骨干重新编码，M表33把它与高原始分辨率teacher区分；不必增加模型或训练。论文的teacher/students相对优劣随readout与任务改变，R节列出teacher不优于bilinear的自然图像情形；L节条件/均值化率含假设，S/同场景匹配还会反转，不能保证眼底的固定head或单图相似性更好。

E6已原生512特征，问题不是网格插值。E8问改变分析尺度/上下文后，是否提供比原f、纯重采样及crop logits更强的有限动作判断。裁剪不新增像素、仍可产生分布外语义误判；不能把crop输出直接当teacher接管全图。

## 具体迁移与资格顺序

使用现有8×4输入、checkpoint不变，原OC-only联合块/B默认/OD锁定。逐合法块固定256方窗裁归一化x，bilinear放大到512，同一模型forward，局部输出bilinear回256。比较候选B与只撤回当前块的S时的角分区差。没有新DINO/SAM、调制器/头、训练/BN统计替换、RDC/ENGRAM。

这不是原SphereTrust或FoRE复现，是两项思想的独立有限变体。先用共同集合、原始精确单块Dice效用验证AUROC与原表示/插值/crop-logits等强控制，原样报告空间/无效覆盖。信号不通过不安装selector/不换层尺度/不追加历史。当前E8不产生新方法预测，不能借oracle结构安全称性能成功。
