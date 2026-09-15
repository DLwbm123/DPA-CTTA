# R4T：三方向联合任务包

**新增的是原第五条G（图教师）的边界锚定版本。** 保留教师×RP与KDG，不替换原十臂，只扩展成十四臂七十轨迹。

现在给Codex：`R4T_STAGE_I_CODEX_PROMPT.md`。主规范：`R4T_COMBINED_EXPERIMENT_PLAN.md`；参数：`R4T_SCIENCE_PROPOSAL.json`。

`R4T_STAGE_II_AFTER_REVIEW.md`仅是未来执行模板，必须等待真实代码review和新的设备授权。当前没有GPU/后台权限。

`inherited/`里的旧R4D文件仅为来源和保持性核对，不是第二个执行计划。本版代码新增相应science摘要，不能谎称总矩阵未改变；`PRESERVATION.json`证明前十臂及teacher/student/RP/kernel等算法块未改。

`PLAN_MATRIX.reference.*`是由公式生成的70job算术计划，不是已经核对NAS后的dry-run；Codex要用现有登记元数据生成真实绑定，不读取阶段I禁止的像素。

`reference/kernel_geometry.py`及`test_reference.py`继承旧包；`boundary_graph.py`/`test_boundary_graph.py`为本次新建的纯张量图teacher参考。代码未接入真实CTTA host，不含数据或权重。测试只证明部分数学和状态性质。

最终选择分A教师/PCA、B kernel参数化、C图teacher三个问题；不能预先把三条路线组合，不能保证增加候选必然提高成功率。
