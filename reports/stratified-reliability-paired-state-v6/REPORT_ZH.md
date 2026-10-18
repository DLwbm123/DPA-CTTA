# V6：同进程严格确定性状态对照结果

2026-10-19；冻结源码41c4445eab11cc4c19bc82ddd27221be2f53e6f9。**执行完整，主比较阳性，但科学推进门槛失败。** 清空跨图像Adam状态相对保留Adam的STRAT PRE提高1.10195pp，8内容组描述性95%区间[.55723,1.70418]，通过预设≥.5pp及下界>0；这不是整体方法成功，更不是TMI创新证明。

## 核心结论

1. Adam历史在当前损失/步幅/宿主上确有不利作用：同进程、相同严格内核下，FRESH仅清空跨图像矩/step而保留仿射，PRE提高1.1020pp，POST提高.6975pp，区间[.3210,1.1390]。但该干预同时清空一二阶矩及计数，不能细分是哪一项作用，不能泛化为所有Adam或CTTA都会有害。
2. 该改善不是分层可靠性独有：RANDOM同样PRE提高1.3522pp，区间[.6433,2.3726]；STRAT状态效应减RANDOM状态效应为−.2502pp，区间[−.8086,.1590]。分层相对随机FRESH PRE仅+.1566pp，区间[−.0285,.3520]，未证明可靠排序优越。
3. 没有首次到达持续收益：FRESH首块PRE对源−.1196pp，区间[−.6450,.4137]；全流PRE+.1982pp，区间[−.4345,.8514]，最差通道条件−5.3415pp，86/160单元下降。返回PRE−.1019pp，区间[−.6871,.4957]，均值保护门槛通过不等于收益成立。
4. FRESH POST92.6432%，RESET POST92.6019%，相差仅+.0413pp，区间[−.1555,.2304]。跨图像仿射积累未证明超过逐图重置。POST对源+.4860pp，区间[−.2319,1.2516]；对RG+.5382pp，区间[−.1333,1.2577]。不能用打败较弱current-image BN/CW基线替代同BN源模型和RESET比较。
5. 结构仍有代价：FRESH POST ASSD3.2512优于源3.2899，但HD95 7.7072劣于7.5831，vCDR MAE .06427劣于.05688。没有新增空预测/嵌套违规、surface160/160和vCDR80/80并不能掩盖这些损害。

因此保留“旧Adam状态会放大漂移”的有限机制结论，停止“当前分层门控可提供稳健CTTA增益”的推进叙事，不继续调门控/步幅/Adam追分。清空优化器、分层及KL均非新概念。

## 完整强基线

|方法|状态|时点|Dice %|ASSD px|HD95 px|vCDR MAE|
|---|---|---|---:|---:|---:|---:|
|SOURCE_EVAL|REFERENCE|STATIC|92.1572|3.290|7.583|0.05688|
|CURRENT_BN|REFERENCE|STATIC|85.3987|8.028|18.310|0.15571|
|D_OLD|REFERENCE|STATIC|85.5856|7.803|17.661|0.15298|
|R_OLD|REFERENCE|STATIC|85.8067|7.476|17.066|0.14426|
|R_SHUFFLE_OLD|REFERENCE|STATIC|86.3431|7.200|17.011|0.14122|
|STRAT|RESET|PRE|92.1572|3.290|7.583|0.05688|
|STRAT|RESET|POST|92.6019|3.231|7.619|0.06722|
|STRAT|RETAIN|PRE|91.2534|3.799|8.789|0.06924|
|STRAT|RETAIN|POST|91.9457|3.529|8.304|0.06759|
|STRAT|FRESH|PRE|92.3553|3.323|7.835|0.06672|
|STRAT|FRESH|POST|92.6432|3.251|7.707|0.06427|
|RANDOM|RESET|PRE|92.1572|3.290|7.583|0.05688|
|RANDOM|RESET|POST|92.4796|3.300|7.890|0.07182|
|RANDOM|RETAIN|PRE|90.8465|3.986|9.112|0.07855|
|RANDOM|RETAIN|POST|91.6995|3.670|8.623|0.07658|
|RANDOM|FRESH|PRE|92.1987|3.420|8.079|0.07123|
|RANDOM|FRESH|POST|92.4735|3.362|8.041|0.07055|
|NOANCHOR|RETAIN|PRE|73.9663|24.440|50.484|0.14054|
|NOANCHOR|RETAIN|POST|73.1292|25.090|49.859|0.13868|
|ENT_NORM|RETAIN|PRE|61.0311|14.197|22.102|0.14696|
|ENT_NORM|RETAIN|POST|58.2301|13.859|21.709|0.16205|
|RG|RETAIN|PRE|92.0589|3.340|7.695|0.05659|
|RG|RETAIN|POST|92.1049|3.312|7.649|0.05693|
|CW1|RESET|PRE|85.3987|8.028|18.310|0.15571|
|CW1|RESET|POST|85.1714|8.604|19.290|0.15498|
|CW1|RETAIN|PRE|84.9251|8.614|19.367|0.15490|
|CW1|RETAIN|POST|84.8892|8.716|19.571|0.15474|
|CW2|RESET|PRE|85.3987|8.028|18.310|0.15571|
|CW2|RESET|POST|84.9093|9.018|19.637|0.15749|
|CW2|RETAIN|PRE|84.1835|9.115|21.227|0.15095|
|CW2|RETAIN|POST|84.0939|9.285|21.592|0.15194|

SOURCE_EVAL使用源BN统计，CW1/CW2及旧D/R前沿使用current-image BN，归一化语义不同。旧D/R为冻结源训练头，本轮没有新增训练。V6按冻结预算仅移除RG/NOANCHOR/ENT_NORM的RESET三个辅助诊断臂；RETAIN与STRAT/RANDOM三状态、CW各RESET/RETAIN、五静态参考完整。先前完整矩阵的负证据仍保留。

## 实际更新、功能与门控

全部1280科学规范更新实际L2匹配，无zero proposal。STRAT FRESH实际步幅范围[.009999544,.010000496]、最小方向cos .999999983；不是“几乎没更新”。其平均raw Adam步幅.011764，实际约.01。

STRAT RETAIN与FRESH的PRE logit RMS分别.40646/.24207，POST .35652/.26127；平均到源仿射距离.18324/.10652。清Adam降低了函数漂移和累计参数距离，但与RESET POST .18884的功能变化及相同Dice比较，不能把参数积累本身视为有益。记录在PRE行的total_affine_norm来自同arrival更新后的诊断，不是PRE模型的参数距离；logit RMS确实对应各自PRE/POST。

FRESH gate在全图排除错误富集OD1.5164/OC1.4435，真值3px边界仅1.0835/1.0085；边界选中31.86%/30.36%。分层补足边界覆盖后，杯缘排序仍几乎没有额外错误区分。

相对SOURCE_EVAL，OD前景corrected/spoiled为.2448%/.4231%，OD边界2.8187%/3.1710%；OC前景4.6198%/.1274%，OC边界6.3302%/5.1882%。OC纠错与OD损害同时存在，必须与尾部及vCDR一起解释。门控对应第二步更新前实际预测，不错配流PRE/POST。

## 冻结判定与范围

通过：主均值/区间、返回均值保护、norm、强POST均值前沿、无新增空/嵌套。失败：首次暴露、全流对源区间、最差保护、同配额随机PRE区间、结构与覆盖联合门槛。RESULTS.advance_development_gate=false，未更改标准。

完整2480读出/26轨迹/1040科学arrival，加额外40arrival/80对logits及末model.state_dict完整RESET精确重演；GPU/CPU均exit0退役，评分后有全部31绝对聚合行。RDC/ENGRAM关闭，源锚/卷积/running stats不变，标签仅在正常worker退出后独立CPU评分时读取。

只有8个反复开发暴露内容、一个checkpoint、两条短合成流。患者关联和预训练成员未知，只有首块是本流首次暴露；bootstrap按8内容组绑定两顺序与重复条件，仅描述性。没有独立患者/临床/TMI创新证据。同进程干预支持当前配置中的状态效应，不应外推其它任务或长流。

详见[执行审计](EXECUTION_AUDIT.md)、[冻结协议](PROTOCOL_ZH.md)、[全部差值](public/contrasts.csv)、[条件通道](public/condition_channel.csv)、[前景转移](public/transitions.csv)、[门控](public/gate_audit.csv)、[实际更新](public/update_audit.csv)和[后续路线判定](NEXT_ROUTE_DECISION_ZH.md)。
