# E2-A 执行审计：GPU预测完整，初评分失败，独立CPU重建完成

## 来源与完整性

GPU冻结source ff85bdca89ff80e7159459dcb68fcd7df3bef0aa；独立CPU修复source38830afc8c21ce67add4ba6f82769aea899c5c20。初协议、首轮源码/tar、RESOURCE_LEDGER、失败log和private初评分文件保留，没有覆盖或把exit1改0。SCORING_RECONSTRUCTION_ZH.md在重建前冻结。

唯一GPU派发，GPU_COMPLETE32唯一输入/96预测；全32输入重推source与反翻teacher共64logit数组逐元素一致PASS，128模型前向（含重演）、optimizer/backward/参数更新0，源参数/buffer不变。首次预算投影89.99379487037659秒PASS。NAS实际NFS容量小写读、启动free23858MiB/中性父子及GPU进程检查通过。

初5CPU PASS2.221911668777466秒，仅合成张量/区域/指标/配置，无真实RGB标签/权重/CUDA。初评分在构建指标后由paired_state.summary不接受静态数组形状而exit1，24.781968355178833秒；没有最终公开科学结果。初ledger明确score FAILED/no_retry=true，保留。

修复只在静态模块接受(8,5)/(8,5,2)/(8,2)，仍调用同interval、相同seed/2000次8内容重采样、相同主比较和门槛。新增1回归使6CPU PASS2.6782021522521973秒，检验各静态形状及重复相同顺序不增加bootstrap单位。没有改旧paired_state或任何预测。修复代码和结果单独在repair_code/scoring_repair，原GPU输入和96bit预测只读引用；不重跑GPU、不依赖已看到的结果改权重或区域。

重建scorer exit0/inactive，SCORE_COMPLETE8标签/96唯一metric/3绝对/24区域/24条件头/8配对对比，原controller与独立修复controller原身份均退休。原GPU worker exit0、初score exit1、重建score exit0分别有回执，不将初失败抹掉。本轮科学结果来自独立CPU重建，不能标“原score正常通过”。

## 数据与统计边界

推理只读8登记RGB和既有checkpoint，拒E1整root/scorer旧评分；适应标签/未来图/新源训练0。独立CPU在GPU退役后读取8标签，GT区域仅用于评分，不回传推理。32唯一输入/96bit预测，240逻辑重复读出只保留E1权重；重复条件/顺序非新证据。24区域行的C/H/D原始总数只计32唯一输入，宏净效应按8图×五块条件权重，C+H=D检查通过。区域有重叠不相加。空预测的表面指标保留undefined并报告覆盖。

## 成本与失败保留

GPU完整12.040014505386353秒（主体9.664519309997559），初CPU评分24.781968355178833秒，重建完整23.209264516830444秒（主体20.444620370864868），修复回归2.6782021522521973秒。初score+修复回归+重建合计50.669435024261475秒，不超过原CPU900秒。原GPU1200秒/输出2GiB预算不改。另原GPU前准备CPU2.221911668777466秒。输出6307000字节，GPU peak allocated415404544字节。

历史累计加本轮GPU=4530.565845727862秒，已计CPU=2219.0993401709266秒；旧本地约.631、公式.0010669171、CPU历史基数.0084569158差及未计时阅读/编排继续披露，不清零或混其它聊天成本。取回脚本曾先遇无SCORE_COMPLETE，此编排错误保留、不当新GPU/模型前向或worker成本。

## 公开交付

只公开自有源码、冻结协议、匿名聚合、中文报告、无路径原失败/完整与重建/退役/成本回执；不公开RGB/标签/权重/逐例概率预测评分/身份私有路径或第三方全文。按一次必要覆盖/隐私检查后push并核远程SHA/匿名HTTP200，验证receipt存私有DELIVERY_E2A.json。无参数更新，RDC/ENGRAM关闭，没有TTA/CTTA成功或独立患者/临床结论。
