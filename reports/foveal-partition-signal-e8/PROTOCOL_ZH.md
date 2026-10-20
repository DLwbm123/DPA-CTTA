# E8：候选相对角分区与真实局部重编码信号（执行前冻结）

## 固定计算

保持8内容×4条件、原checkpoint/预处理/BN eval/原联合8连接块与B=TOPO_SMOOTH默认、OD-active锁B。先按E6全部固定无标签规则现场生成OC-only块与锚点/legacy分数，仅legacy common块进入crop计算，不读旧效用/分数或GT。crop中心为块bbox像素中心，`start=floor((min+max+1)/2)-128`，各轴clip[0,256]，窗口256方形；块全部像素不在窗口则不可评分，不改尺度。

裁已归一化512原输入x的256方窗，torch bilinear align_corners=False升至512，现有同一源模型eval/no_grad一次；末head_input32和两logit原生512。回256用同bilinear align_corners=False，即每2×2中心采样，不插值mask/GT，B和单块撤回S用原256格坐标。全局f和z也取同窗；RESAMPLE只对原f窗升512回256，隔离插值。CROP_Q_SHIFT为回256 crop f场宽roll129；不移动B/S。

每个表示每像素单位化，无epsilon/floor；非finite硬停。共同方向有效像素V为crop/global/resample f、global/crop两logit及shift crop f所有norm严格非零的交集。被编辑的全部query必须在V；否则当前块所有分数不可评分。对B及单块撤回后mask，前景与背景在V均至少3像素，单位向量和norm均严格正，否则全臂不可评分。允许共同集合小于E6，报告每原因/编辑像素与common oracle门，不补分。

`Q(M)=1-cos(sum_{p in V & M} unit(v_p), sum_{p in V & ~M} unit(v_p))`。

每个Q分数=`Q(B only_this_block→S)-Q(B)`，高分支持撤回源。可用sum增量精确等价计算，与合成bruteforce对照。该Q对补集对称，不提供新的语义真值；端点源方向仅限定动作。独立OC评价不拼互斥softmax，两头仍sigmoid。固定11臂：CROP_Q/GLOBAL_Q/RESAMPLE_Q/LOGIT_Q/CROP_LOGIT_Q（上述各表示同构Q）、原E6_FEATURE、RAW_MARGIN=mean abs源OCz、GEOMETRY_ONLY=原E6值、AREA_DELTA=单块源−B面积/65536、CROP_Q_SHIFT、STRATIFIED_ROLL（E6输入×direction×size×pred-boundary-distance，common块scanid roll1/singleton不变）。不加C/触边/zscore组合；未按真实评分调规则。

## 评价与主门

源32forward和全部实际crop编码完整重演；N为第一遍legacy common且crop完全包含query的块数，总encoder=64+2N≤5826，额外head重构同次数但不计encoder。每输入流式、稠密f/z不存储。全部共同布尔、crop bbox/无效原因/锚点/score/uid精确重演；全源state_dict/BNbuffers不变。无梯度/backward/optimizer/新模型训练/历史或selector。worker只8RGB/已有权重/新64端点，拒旧根/scorer/GT。原身份exit0退休才独立CPU读8GT。

效用沿E6精确单块撤回Dice差，用实际mask逐块核验，不把GToracle决策当标签、不加总多块效用。AUROC零效用排除、oneclassNone。BLOCK主、|utility|和PIXEL辅助，逐输入→[2,1,1,1]固定条件→8内容等权/seed20261018/2000内容绑定描述bootstrap。唯一主CROP_Q−GLOBAL_Q BLOCK≥.05且CI下界>0；CROP_Q AUC≥.60；对另外9控制的BLOCK配对差均>0且CI下界>0，效用加权主差≥0。common oracle≥.2pp且CI正、≥24/32输入有正负/8内容均有效、strat位置移动≥.5。所有门合取，任何失败signal_qualified=false，不能用crop logits/子条件替主或调window/layer/epsilon救分。报告所有33AUC/30paired、size/direction与有效pixel覆盖/空间。不是超所有logits条件信息的证明。

## 完成与停止

先合成窗口/映射、Q补集/常特征/无效norm/三像素、delta vs brute、纯插值/zero editing、源head映射/no-grad/访问隔离/AUC与重复等CPU检查；CPU前记UTC/BOOTTIME统一24h时钟，86340s独立保存停、24h全原身份退出，修复共享。NAS实际NFS容量小写读、写前320MiB含code/archive/cache/logs reservation/root256MiB/累计20GB水位18GB辅助1GB，GPU6现场UUID/free≥8192/中性父子ps+自身nvidia后直接运行。至多本聊天1GPU，不干预其它任务。

完成须32输入/全部N crop与全量重演、64源bitparity/全部head重构/source不变/ACCESS、独立CPU8GT/3028效用/33AUC/30paired、worker/scorerexit0退休/controller guardian退休与finalledger。工程失败保留原source/log/cost独立修复；科学失败不扫尺度。此轮不安装任何规则/输出，无新方法Dice/结构/返回成绩，GToracle只界定空间。完成失败中文报告/审计/自有代码匿名聚合无路径receipt；第三方PDF/稠密/位图/坐标/逐块值不公开。8重复开发内容非独立患者验证，不称临床或TMI成功。
