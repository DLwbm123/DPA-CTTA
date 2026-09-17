# CPU 覆盖与证据边界

实现候选：d875f20c11cc7e617c9f39dba04ed46381aba450。每端完整入口 scripts/check_r6_cpu.py，R3_REGRESSION=1；R6 23 个 methods、R5 31 个、继承回归 112 个，共 166。测试数量只作清单，不替代性质覆盖。最终结果与实际成本以 logs/final-local.json、logs/final-server.json 为准。

| 文件 / 检查 | 覆盖 |
| --- | --- |
| test_r6_loss（5 methods） | 空/满/1像素/稀少/半图分区、double 构造、正值/cap/mean、未裁剪质量均等及裁剪非50/50；全通道置乱/直方图/位置/seed/RNG/软q；统一状态 double/float 梯度及 detached scale；p=q零梯度与pos_weight反例；微小残差、非有限值、零能量分支、统一权重退化及生产无VJP |
| test_r6_host（7 methods） | 旧C多步bitwise、实际六视图q/pre/post；每臂改GT/延迟/省略不影响state/RNG、禁止身份参数；读mask晚于所有提交；非有限参数/Adam硬失败不续跑；完整随机ResUNet旧C+四臂各4步；小模型独立目标参数/Adam/RNG比较；有Adam历史时零梯度仍调用并产生moments位移 |
| test_r6_execution（7 methods） | OLD+C+BAL+SCALE+SHUFFLE 每角色第2访问 forward/backward前Adam/Adam后故障共15组合，live独立计数、原异常与首失败；A/B同160/20/20/0 smoke编排；12/8/20及轮转；默认disabled先于查询；SHA/science/data/stream/fingerprint/设备/阶段/recipe拒绝；中性入口 |
| test_r6_analysis（4 methods） | 四臂完整标量字段及join；不可能像素/Dice/GT、重复/缺失/换序、weight/cap/scale/seed/S0/physical污染；A/B所有gate边界与单项失效；12完整A→8完整B复用原SHA，破坏A后A/B均INCOMPLETE；异阶段/错run_id/scope绝不invalidate |
| host_scalar_probe.py | 程序化Toy当前host两访问×四臂的真实非均匀权重trace→独立标量join，64F/8B/8Adam/0VJP；与纯合成完整ledger互补，日志单列 |
| 既有回归 | test_r5_rule / host / analysis / audit_fix；test_r4t_execution、test_r3_reference、test_r3、test_r3_execution、test_r1、test_r1_fixes、test_r2、test_r2_continuation，全部同一最终候选重新执行 |

第一次 core01 的独立 double 参考把整数 count 相除，触发 PyTorch float32 中间值，误差约1.6e-8；生产 double 构造已正确。修正独立参考的 count dtype，保留预定 rtol/atol，没有更改生产公式来通过测试。r6-01 为当时21项通过；host02 为后加两项通过；它们不替代最终166项全套。

EXPECTED_R6_INJECTED_FAILURE 为主动故障测试事件，不是实际套件失败；事件JSON保留在原CPU日志。unittest totals/failures/errors/skip是该次真实结果。B1物理统计只覆盖B1派生路径，不把其他继承方法的成本伪记为零，也不将不同重复套件相加成独立覆盖。

GPU lazy init 被拒绝。完整套件中的旧IO回归仅放行现场新建临时目录和BytesIO程序化fixture，其余真实路径拒绝。新R6模型输入为程序化像素和随机初始化完整网络，没有读取已登记checkpoint。外部 R6_MATH_REFERENCE.py 与 math_history 未提供，不属于本次验收成绩；原始science缺失使阶段I整体仍BLOCKED。

## 后续标量审计修正

静态复核发现 q 评价的前景像素数没有与 loss n_fg 相互约束；audit-gap-before-fix.log 以可实现像素记录复现这一缺口。加入 q.pred_pixels == n_fg 检查，污染测试单独覆盖“指标本身合法但与分区矛盾”的情形；完整合成fixture的q前景计数也修正为实际声明的分区量。audit-fix 为4个分析器methods通过。

旧候选438ff073b05b34d60d3aef4802bbcdc7c3fe1898的本地完整166项曾通过，归档为candidate01-final-local；旧候选服务器套件为节省无效重复成本被主动取消，只取消本任务专用CPU验证进程，退出-15，保留candidate01-final-server.log、candidate01-exit.json、candidate01-superseded.json。该前缀没有最终总计，物理成本未知，不假装通过或零成本。没有终止历史实验/他人任务，也没有自动retry任何真实轨迹。修订后最终候选d875f20c11cc7e617c9f39dba04ed46381aba450在两端重新运行完整166项；其最终日志才用于当前交付。

## 成本计数范围

完整runner的 procedural_B1_physical.forward/backward/Adam 来自 B1 的三个实际hook，仅是这些hook覆盖范围的实测数；不会覆盖所有继承方法。该字典的 jacobian_vjp_calls=0 是继承runner预置字段，**没有给全部继承回归安装VJP计数hook**，不能把它解释为完整166项的实测VJP总数。R6生产路径的0额外VJP由公式实现、禁止autograd.grad测试和其自己的每步配额单独限定；R3等继承方法的VJP总成本在此统一计数器中未知。完整套件的实测wall seconds与B1 hook计数、R6完整网络/额外probe的专用计数分开报告，不伪造全方法总成本。
