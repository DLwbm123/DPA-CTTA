"""Offline report rendering from sealed public aggregates; R16_PUBLIC_ROOT is private environment input. No inference or label reads."""
from pathlib import Path
import csv,json,statistics as st,math
import os
p=Path(os.environ['R16_PUBLIC_ROOT'])
def rd(n):return json.loads((p/n).read_text())
def csvrd(n):return list(csv.DictReader((p/n).open()))
def csvwr(n,rs):
 with (p/n).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
s=rd('SUMMARY.json');a={r['condition']:r for r in s};ld=rd('RESOURCE_LEDGER.json');src=rd('SOURCE_MEANS.json');pr=rd('PROTOTYPE_CANDIDATE_DIAGNOSTICS.json');z=csvrd('ADAPTER_LATENCY_DIAGNOSTICS.csv');dc=csvrd('DOMAIN_CHANNEL_RESULTS.csv');me=csvrd('MECHANISM_RESULTS.csv')
# Read-only derived adverse distributions retain every registered order, not best-seed selection.
pa=csvrd('PAIRED_RESULTS.csv');tails=[]
for k in sorted({r['condition'] for r in pa}):
 for o in ['0','1']:
  rs=[r for r in pa if r['condition']==k and r['order']==o];v=sorted(float(r['delta_DS_pp']) for r in rs);assert len(v)==1695
  tails.append(dict(condition=k,order=o,n=len(v),mean_imageweighted_delta_DS_pp=st.mean(v),worst_delta_DS_pp=min(v),worst10_n=math.ceil(.1*len(v)),worst10_mean_DS_pp=st.mean(v[:math.ceil(.1*len(v))]),negative_images=sum(x<0 for x in v),positive_images=sum(x>0 for x in v),zero_images=sum(x==0 for x in v)))
csvwr('PAIRED_TAIL_SUMMARY.csv',tails)
# Source displacement was not retained. Separate structural errors from pixel edit errors explicitly.
se=csvrd('SOURCE_ERROR_ANALYSIS.csv');ss=csvrd('TARGET_STRUCTURE_RESULTS.csv');cost=[]
for x in ld['attempts']:
 cost.append(dict(phase=x['phase'],attempt=x['attempt'],status=x['status'],GPU_worker_seconds=x['cost'].get('gpu_seconds',0),wall_seconds=x.get('wall_seconds'),full_forwards=x['cost'].get('model_forwards',0),head_forwards=x['cost'].get('head_forwards',0),BP=x['cost'].get('backward_calls',0),optimizer=x['cost'].get('optimizer_steps',0),VJP=x['cost'].get('vjp_calls',0),zero_order_updates=x['cost'].get('zero_order_updates',0)))
csvwr('COST_BY_ATTEMPT.csv',cost)
# Per-method deployment operation counts are known; standalone elapsed cost was not separately instrumented.
ind=[]
for k in a:
 ind.append(dict(condition=k,backbone_forwards_per_image=1 if k=='C0' else 2 if k!='G' else None,
  head_forwards_per_image=1 if k in ['D_LOGIT','D_CONTEXT','D_VERIFY'] else 0 if k!='G' else None,
  new_source_supervision='2000 optimizer/BP steps per fitted head' if k in ['D_LOGIT','D_CONTEXT','D_VERIFY'] else 'none' if k!='G' else 'historical reference',
  target_updates_per_visit=1 if k in ['Z_CORE','Z_BOUND'] else 0 if k!='G' else None,
  target_BP_optimizer_VJP=0 if k!='G' else None,standalone_wall_latency='NA_NOT_SEPARATELY_MEASURED',peak_VRAM='NA_NOT_INSTRUMENTED',
  origin='matched historical stateful reference; not charged as a new R16 deployment' if k=='G' else 'R16 operations; shared campaign elapsed cost not allocatable per method'))
csvwr('INDEPENDENT_DEPLOYMENT_COST.csv',ind)
na=csvrd('FAILURES_AND_NA.csv')
for metric,reason in [('source_boundary_displacement_ASSD','source boundary distances not retained; no recomputation after retired cache'),('standalone_method_latency_peak_VRAM','shared static deployment and sampled VRAM do not identify standalone costs'),('G_new_structure_edits_and_ops','historical matched scalar reference lacks new topology/edit and R16 deployment operations')]:
 if not any(r['metric']==metric for r in na):
  na.append(dict(condition='ALL' if metric!='G_new_structure_edits_and_ops' else 'G',order='',status='NA_NOT_RECORDED',metric=metric,reason_code=reason))
csvwr('FAILURES_AND_NA.csv',na)
# Keep backend report alongside the reviewed final interpretation.
if not (p/'BACKEND_REPORT.md').exists():
 (p/'BACKEND_REPORT.md').write_text((p/'REPORT.md').read_text())
lines=['# R16 四方向冻结实验：最终报告','',
'本轮 COMPLETE。14 个条件全部覆盖完整开发流，13 个新增物理在线任务及独立 CPU 评分均完成，没有目标恢复或重跑。新方法的最高 Dice 为 D_CONTEXT **75.385885%**（ΔC0 **+0.306465 pp**、ΔDS **+0.146702 pp**、ΔGraTa **−1.843808 pp**）。所有新方向都未达到预注册的 ΔC0≥+0.5 pp 门槛；不能称为超过 GraTa、独立确认或临床有效。', '',
'执行源码 `da25ef2b150d22774b6afcecd1ed542a044becaf`；冻结配置 `2be8aaec7b9e0d3c53d0c7077ac90c8f7d81eb936feca17febbed41d52a44d43`；源选择/目标锁 `161bb2adc9840ce089bc58302cda26df56852ef16df6794a755648b2d85df77a`。本次提交另加的离线统计工具只读取已封存记录，不改变执行代码、预测、条件或评分。公开锁为脱敏视图，锁 SHA 承诺的是私有原始 payload，不能对脱敏 JSON 重算得到同一值。', '',
'## 四方向结论', '',
'| 方向及完成状态 | 相对 DS / GraTa（pp） | 固定机制对照 | 新增源监督、目标更新及实际资源 | 结论 |',
'|---|---|---|---|---|',
'| A 图内原型 P：完成 | +0.104620 / −1.885890 | P−DS 有小开发收益；32 图源端空间打乱后反而 +0.147567，未支持空间对应特征是收益来源 | 无新增训练；共享静态原图+翻转 2F，禁止跨图特征库 | 开发信号，机制支持不足 |',
'| B 结构追踪 S、P_VERIFY：完成 | S +0.120235 / −1.870275；P_VERIFY +0.078926 / −1.911584 | P_VERIFY−P −0.025693；P_VERIFY−P_SIMPLE −0.037874；验证器没有提高平均 Dice | 无新增训练/目标更新；共享静态 2F 与 CPU 候选检查 | S 有小开发信号；验证优越性未获支持 |',
'| C 两前向零阶：完成，2 orders×2 seeds×3 arms | CORE −0.159151 / −2.149661；BOUND −0.157811 / −2.148321 | CORE−PROBE +0.000612；BOUND−CORE +0.001340（order 等权、seed 平均），几乎无效 | 512 个 adapter 参数、2F/图、目标0BP/optimizer/VJP；CORE/BOUND 各1951更新/流；源662.841 s、12条目标4964.429 s | 相对 DS 未见增益，不能泛化否定所有零阶方法 |',
'| D 源端监督残差头：完成 | LOGIT +0.133741 / −1.856769；CONTEXT +0.146702 / −1.843808；VERIFY +0.092163 / −1.898347 | CONTEXT−LOGIT仅 +0.012961；VERIFY−CONTEXT −0.054539；上下文优势很小，验证损害平均 Dice | 同容量各9026参数、各2000源端训练步；目标头冻结，无在线训练；源端D整包5071.853 s，目标共用静态2F+对应1头F | 有小开发信号，未达到确认门槛 |', '',
'静态10条件整个共享目标 worker 为9186.349 s，不能把它逐个算成方法独立耗时。A/B/D 共享成本不能在无计时证据时拆分；`INDEPENDENT_DEPLOYMENT_COST.csv` 给出单方法操作量，并明确独立延迟和峰值显存 NA。GraTa 是匹配旧完整有状态轨迹，不是 R16 新部署，也未把其历史费用重复计入本轮。', '',
'## 全部条件与风险', '',
'| 条件 | Dice % | ΔC0 pp | ΔDS pp | ΔGraTa pp | 最差域/通道 ΔDS pp | 判断 |',
'|---|---:|---:|---:|---:|---:|---|']
for r in s:
 role='匹配参考' if r['condition'] in ['C0','H025','DS','G'] else '开发信号' if r['status']=='DEVELOPMENT_SIGNAL' else '相对DS未见收益'
 lines.append(f"| {r['condition']} | {r['Dice_percent']:.6f} | {r['delta_C0_pp']:+.6f} | {r['delta_DS_pp']:+.6f} | {r['delta_G_pp']:+.6f} | {r['worst_domain_channel_DS_pp']:+.6f} | {role} |")
lines+=['',
'新方向最差退化为 D_LOGIT/D_CONTEXT 在 REFUGE_Valid/OC 的 −1.208402 pp（每 order n=736），相对 C0 −0.509328 pp；虽未超过 −2 pp 风险阈值，也不能只报道均值收益。P 最差 REFUGE_Valid/OD −0.370199 pp；S 最差 REFUGE_Valid/OC −0.491412 pp。所有域/OD/OC、每个 Z seed/order 的结果均在 `DOMAIN_CHANNEL_RESULTS.csv` / `MAIN_RESULTS.csv`。', '',
'预注册判断保持：ΔDS>0且两 order 均非负仅为开发信号；新方向进入新数据确认还需 ΔC0≥+0.5 pp、优于 DS、最差域通道损失≤2 pp。后端 SUMMARY 的 G 数值门槛标志只是既有参考的数值属性，G 不是新候选；没有任何新方向达标。额外列出的最差 ΔC0 不改变本轮结论。', '',
'## 分母、配对、隔离与独立性', '',
'每逻辑流1951在线内容、1695主评分、256 warmup，4域等权、域内逐图OD/OC平均、order等权。34逻辑流66334 scalar rows，其中57630主评分行；13新增物理任务均1951访问。10静态条件共享同一次原图/翻转前向，order1按内容及角色核实后重排，不能算第二次独立运行。Z三个条件×两种子20261003/20261004×两order均实际运行；两个seed仅描述算法随机性。GraTa用原历史seed20260907的完整有状态轨迹，训练seed不可与Z初始化seed混同。', '',
'历史C0/H025逐图 hard intersection/pred/GT/total pixels 与新共享输出完全匹配；DS和H025硬mask逐图精确一致，概率输出不等价。完整锁与轨迹封存由 CPU scorer 核实。主干参数/BN冻结检查由执行器持续实施；Z只改adapter，D只在源端监督训练，目标0BP/optimizer/VJP。所有目标轨迹终态后才读取标签；无目标临时分数用于源选择或修改条件。', '',
'完整数据均为先前已暴露开发集；order共享同批图像，患者依赖未知，不报告伪患者级置信区间。`PAIRED_RESULTS.csv` 保留匿名逐图配对差和最差10%；`PAIRED_TAIL_SUMMARY.csv` 是1695图按图等权的分布，不是4域等权主终点。未知患者依赖及14臂多重比较限制保留；没有解封保护盲测。', '',
'## 源端 hard/soft 与错误机制', '',
'源端16×32验证包含四个模式，D统计/训练仅源fit111，校准23，验证25，已知图像组隔离但未知患者身份。Z仅4个预列(mu,eta)，按源端CORE/BOUND模式均衡hard选择 mu=.001、eta=.0001；D每头2000更新，固定500/1000/1500/2000源端checkpoint，最终LOGIT1000、CONTEXT2000，无目标选点。全部参数候选及checkpoint在 `SOURCE_RESULTS.csv`，Z的soft_Dice由已封存soft_OD/OC算术平均补齐，不新增计算。', '',
'| 源条件（各512访问） | hard Dice % | soft Dice % |', '|---|---:|---:|']
for k in ['C0','H025','DS','P','P_SIMPLE','P_VERIFY','S','D_LOGIT','D_CONTEXT','D_VERIFY']:
 r=src['all_modes_equal_512_visits'][k];lines.append(f"| {k} | {r['hard_percent']:.6f} | {r['soft_percent']:.6f} |")
lines+=['',
'DS相对H025的硬分数不变，而源soft从76.937905%恢复到85.446153%，因为不发生硬决策分歧的位置保留native raw logits；不能据此声称目标校准改善。P/S/D在源hard均低于DS；D_CONTEXT hard−DS=−0.276971 pp、soft−DS=−0.115456 pp。源负结果不算执行故障，也没有据此临时改条件。', '',
'边界偏移与结构错误分开：`SOURCE_ERROR_ANALYSIS.csv` 记录每模式的正确→错误/错误→正确像素，以及包含、碎片、孔洞变化；结构计数下降不等于位置准确。源端ASSD/有方向的边界偏移未保存，退休源缓存后无法补算，明确NA；这是辅助源诊断缺失，不能以轮廓规整替代边界证据。目标独立scorer保存每域/通道ASSD及有效/未定义分母，完整保留在域表；`TARGET_STRUCTURE_RESULTS.csv` 给出1695主评分的结构及编辑统计，G历史拓扑/编辑字段缺失NA。', '',
'32图空间打乱源诊断的P=91.156058%，同32图原P=91.008491%，差+0.147567 pp；与512图总体均值不可混比。这一对照不支持当前图内特征空间对应提供必要纠错证据，不能把它作为效果确认。', '',
'## 原型、候选和adapter诊断', '',
'1951在线图中2图因种子缺失回退DS，1949图有原型；可靠seed平均覆盖背景85.5735%、disc-without-cup2.4413%、cup1.7075%；原型cosine(background,disc)=−0.765389、(background,cup)=−0.905207、(disc,cup)=0.907139，disc/cup相似度高只能作为描述，不能事后改阈值。', '',
'| 选择器（1951图，含warmup） | 候选数 | 接受数 | 最大空间编辑比例 |', '|---|---:|---:|---:|']
for k,r in pr['selectors'].items():lines.append(f"| {k} | {r['candidates']} | {r['accepted']} | {100*r['max_spatial_edit_fraction']:.6f}% |")
lines+=['',
'完整拒绝原因汇总在 `PROTOTYPE_CANDIDATE_DIAGNOSTICS.json`，未公开私人候选support/预测。各选择器均遵守2%空间编辑上限，无种子/带外等保护违规证据。更严格验证接受更少候选、限制碎片孔洞，但在P/D对应对照中降低平均Dice，不能称证实结构验证机制。', '',
'CORE终态adapter RMS范围0.000109336–0.000114417，BOUND为0.000827367–0.000831672，远低于预列0.05投影边界。CORE全部4流无跳过；BOUND有1/7804更新标记跳过，其余执行；PROBE始终0更新。梯度范数为裁剪前值，更新按源码L2clip1执行。损失差、RMS最大值及每seed/order延迟在 `ADAPTER_LATENCY_DIAGNOSTICS.csv`，收益微小与实际参数偏移小一致，但不据此调整eta或重跑。', '',
'## 成本、工程资格与未测项', '',
'| 阶段 | 实际GPU-worker秒 | 说明 |', '|---|---:|---|',
'| v1/v2资格、热点诊断、v3机械预检 | 943.382673 | 前两版本成本未准入零目标，费用保留；v3只8源图16F，复用封存32源资格 |',
'| 源端Z全部4参数选择 | 662.840636 | 8192F，0BP/optimizer/VJP |',
'| 源端D整包及静态源诊断 | 5071.852677 | 2334F，4000BP/optimizer，2头各2000步 |',
'| 共享静态目标10条件 | 9186.349340 | 1951图3902F、3902头F，无目标梯度 |',
'| 12条Z目标 | 4964.429029 | 23412图46824F，无BP/optimizer/VJP |',
'| 总计 | 20828.854354（5.785793小时） | 20个唯一phase/attempt，预检含16丢弃BP/optimizer，不重置计费 |', '',
'原T0北京时间10月4日00:00；全部评分终态06:41:01，计算/评分墙钟6.683653小时，源cache623文件7,452,495,112字节（6.940GiB）在目标前已退休，仅保留冻结头及统计。源阶段已采样私有总量约7.584GB，最终私有产物约6.234GB，未见8GiB限额违规；这些是文件大小快照，不冒充精确全局峰值。独立CPU评分3370.328 s，GPU0；离线报告聚合额外CPU、GPU0，未再次读取图像或标签/模型推理。GPU-worker秒含GPU任务的CPU/IO时间，不是硬件内核活跃秒。', '',
'共享10条件在线trace平均4.437700 s/图、中位4.127112 s、p95=7.069688 s，包含所有静态模块，不可当作每方法单独延迟。Z每流平均0.1479–0.1887 s/图（trace计时，不含全部worker启动/封存）；所有worker总账另列。实际VRAM仅启动/监测采样约854–952MiB量级，源资格/拟合不同；未做峰值设备内存仪表，独立方法显存峰值NA。', '',
'v1/v2资格及CPU等价证明、v3成本准入完整保留在 TEST_REPORT / PROFILE_ADMISSION / OPTIMIZATION_EQUIVALENCE；v1/v2是资格通过但预算不准入，不算方法性能失败。v3九项CPU测试通过；实际源32控制复用v2封存且与v1逐输出指标/原型完全相同。无科学重跑、无目标基础设施恢复。所有主子argv及GPU4/5实际映射在启动核查中满足中性约束，完成后全部本轮worker/watch/supervisor/scorer退出、租约释放，其他进程未干预。', '',
'本轮Inspired/Adapted模块不等于原论文复现；D额外源监督、Z目标无标签更新与A/B无训练不同。目标只封存硬mask，soft Dice/Brier/ECE真实NA；不能从mask当概率编造校准。源端边界偏移、单方法独立延迟/峰值显存及G新编辑拓扑/部署成本未测，详见 `FAILURES_AND_NA.csv`。辅助字段缺失不会用重启科学试验补齐。', '',
'## 交付与结束', '',
'交付代码、冻结配置、资格/成本记录、全部结果和脱敏统计；禁止上传原图、标签、预测、特征、候选support、权重、患者/内容身份、私有路径与凭据。复现依赖有权访问的本地数据/checkpoint；`tools/aggregate_r16_sealed_diagnostics.py` 只在 COMPLETE 后从封存记录生成统计，不打开图像/标签。', '',
'按授权，本轮交付后关闭每小时监测。不追加参数或seed、不恢复RL、不启动后继。下一步只能建议重新审查源/目标语义证据与纠错候选是否有信息增益；本轮最大值不是新数据确认。发布是否已核验以独立交付回执为准，提交不能自证自己的最终SHA。']
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
if 'Final: COMPLETE, sealed v2 native32 reused' not in (p/'TEST_REPORT.md').read_text():
 (p/'TEST_REPORT.md').write_text((p/'TEST_REPORT.md').read_text()+'\nFinal: COMPLETE, sealed v2 native32 reused with exact v3 CPU equivalence. V3 timing8/16F, full source selection/training and all13 target physical jobs completed. CPU scorer66334 scalars/57630 principal; shared historical pixel-count parity PASS, DS/H025 hard identity PASS; target zero BP/optimizer/VJP; no recovery. Offline aggregate review reads sealed scalar/journal files only; no new model or label reads. Auxiliary source boundary displacement and independent per-method latency/peak VRAM are missing, reported NA.\n')
print('Final report written:',len(lines),'lines; paired tails',len(tails),'attempts',len(cost))
