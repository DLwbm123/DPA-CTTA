# R1 Review Fix Implementation Report

**R1_REVIEW_FIXES_READY**。E1–E4修复已实现，最终服务器CPU套件35/35通过。本次不产生外部复核结论，也不授权或运行阶段II。

- 完整implementation commit：[c93018c0edca2bbe2475117ab0edf6adeece6ff2](https://github.com/DLwbm123/DPA-CTTA/commit/c93018c0edca2bbe2475117ab0edf6adeece6ff2)。
- 外部审阅基线：`f67a4d91c4337fe53e350cc2cf621ab703b9d2c1`；此前材料提交：`d757272f6cd01ed1dbd0a6c55cd14a8ba21f9875`。
- 分支：`experiment/r1-recovery-target-subspace-v1`。完整实现由`1a777a8920ee33ef04f132ad596061273df2c012`的四类修复和最终提交的信号/文件句柄收尾修复组成；以最终完整SHA为复核对象。
- [REVIEW_INDEX](REVIEW_INDEX.md)提供函数行号、35个测试入口和所有材料；[完整patch](implementation.patch)与[diff stat](DIFF_STAT.txt)严格相对f67a4d91，包含此前已发布文档提交的差异。

## 四类修复

| 审阅项 | 最终行为 | CPU证据 |
|---|---|---|
| E1 资产内容绑定 | 恢复既有目标RGB/mask摘要；checkpoint在smoke和formal各自进程内验证后，从同一字节流加载；当前RGB验证后解码，固定预测后才读/验mask；保留IO字节与时间 | 正确size/mtime配错误SHA在两个入口均先于反序列化、GPU环境与模型工作被拒绝；程序化PNG保持原reader输出；mask读取顺序和错误摘要拒绝 |
| E2 独立重算和历史对照 | run/device/job共同身份；completion、JSONL、job预算、smoke和进程退出交叉核对；独立重放controller；PCA配额/累计/refresh/rank/快照时序验证；A必须匹配原receipt及完整流，C0保留明确canonical例外 | 假第1步SENS恢复、trend矛盾、空REGION bank、负配额、错误版本/计数、错误completion、失败与成功并存、错arm/order历史A均拒绝或标UNVERIFIED；合法C0映射及完整24条toy矩阵通过 |
| E3 进程监督和预算 | 父进程直接管理每卡smoke及每条formal轨迹的新process group；轮询全部退出状态；失败后停止派发，独立在途轨迹可结束；超时/共享故障、中断和部分启动失败保留前缀并收尾；无重试 | 程序化sleep/fail子进程覆盖超时、累计/wall/输出上限、禁止新任务、部分启动失败、SIGINT/SIGTERM、孙进程清理及不触碰无关进程；额外覆盖Popen登记和mkstemp中的信号 |
| E4 确定性设备轮换 | 执行前固定`(arm_index+order_index)%k`；每轨迹固定一卡，最多每卡一worker | 1/2/3槽各24条且每job恰好一次；每臂跨四序覆盖全部槽；总调用数不变 |

旧C的归一化、增强、loss、Adam没有修改；`host.py`、`recovery.py`和`region_memory.py`保持原实现。`streaming_pca.py`只增加`last_visit`审计标量，不改变特征、统计、PCA或损失。没有增加臂、模型、源数据需求、额外预训练组件、诊断实验或效果gate。

结果发布采用完整版本目录与原子current指针，report和aggregate来自同一版本。重算开始即将当前索引置为无效；失败不能留下当前有效的旧COMPLETE，旧版本和原JSONL仍保留。描述性摘要显式包含risk、inactive、matched comparisons和副对照状态，不能代替外部研究选择。

每条formal轨迹独立新进程意味着额外的进程创建/checkpoint加载IO；这些成本进入实际时间，不增加网络前向或优化步。过程、状态和预算细节见[PROCESS_AND_BUDGET](PROCESS_AND_BUDGET.md)。

## 配置与登记摘要

[science config](../../../configs/r1_science_v1.json)的字节和所有科学参数保持不变：

`e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2`

六臂、四序、1951组、1695 remaining_dev、seed 20260907及正式预算46824条记录/Adam/backward、398004次前向保持不变。[SCIENCE_DIGEST](SCIENCE_DIGEST.json)和[DRY_RUN_MATRIX](DRY_RUN_MATRIX.json)可直接下载；所有任务仍为NOT_RUN，没有真实GPU分配。

私有registration已从服务器现有metadata恢复1951个image_sha256和1951个mask_sha256。没有重新选图、分组或替换预期摘要。八个order/control条目复用三个旧receipt的metadata绑定，未读取历史真实分数。

| 登记版本 | 摘要 |
|---|---|
| 原Stage I | `8620ed1b0d225ad921d33998123dd322a30c9ed67c2d3de65521100f429c5d4e` |
| 修复后的schema v2 | `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf` |

变化来自恢复hash/身份metadata及旧receipt绑定字段。后续真实授权必须绑定新登记摘要；当前execution默认仍为false，审批字段为null，GPU列表为空、worker为0、后台关闭。公开[解析摘要](ASSET_RESOLUTION.json)不包含真实路径或身份；[REGISTRATION_SCHEMA](REGISTRATION_SCHEMA.md)说明字段和读取边界。

## 真实CPU测试与修复中发现的问题

最终完整套件在服务器原环境运行，使用程序化权重/图像/标量和短时CPU子进程；固定完整网络的原C等价性与PCA梯度测试也保留。没有加载真实源checkpoint来验证本轮内容身份；真实checkpoint/RGB/mask的内容校验代码仅由toy资产测试，真实资产验证留待另行授权的执行阶段。

| 最终环境与结果 | 实测 |
|---|---|
| Python / Torch | 3.10.6 / 2.2.1+cu121 |
| 测试 | **35通过，0失败，0错误，0跳过，exit 0** |
| 原测试保留 | 原17个方法及全部原有assert调用保留，另新增18个工程回归方法 |
| CUDA | `cuda_initialized=false`；可见设备设为空，CUDA初始化被测试入口禁止 |
| 时间 | unittest 80.794秒；外层包装器81.843秒 |

[最终原始日志](CPU_TEST_LOG.txt)与[机器结果](CPU_TEST_RESULTS.json)是真实输出。原17项中完整24条toy汇总fixture增加了新schema所需的receipt、smoke、进程、controller和PCA标量；原有断言未删除或放宽，并新增过期成功索引失效断言。已用AST比较原测试方法与assert调用，确认保留。

本地初版33项通过；补充Popen边界后9项进程检查通过。服务器第一次对中间提交1a777a89运行34项时，出现2个error：SIGINT/SIGTERM用例结束后NAS临时目录清理报Directory not empty。这不是方法效果失败，也没有发生任何正式实验。

随后用只含程序化IO的精确时序探针复现：在mkstemp已取得fd、返回给调用方之前发送SIGTERM，旧handler异步抛异常，fd未关闭；CPU进程退出后遗留目录变空，与打开句柄造成的NFS清理阻塞一致。[FD_ROOT_CAUSE](FD_ROOT_CAUSE.json)保存实际复现结果。最终handler只记录中断请求，先完成原子IO再在安全边界退出；新增同一时序的fd关闭测试和Linux输出fd检查，相关10项本地检查通过，最终服务器35项全部通过。

[初次失败日志](CPU_INITIAL_FAILURE_LOG.txt)保留错误和计数，仅将绝对checkout、解释器、私有审阅路径替换为占位符；原始文件仍在私有服务器。没有通过跳过、忽略清理错误或放宽科学上限获得通过。

最终测试使用现有解释器，不安装依赖或新建环境。逻辑命令如下，路径通过环境变量提供以保持实际进程参数中性：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
```

`RUN_FILE`为`scripts/check_r1_cpu.py`；`PYTHONPATH`含本项目src/tests及原有依赖；`DPA_CTTA_BASE_ROOT`与`DPA_GRATA_ROOT`使用固定外部代码，`CHECKPOINT`未设置，`CHECK_OUTPUT`和`TMPDIR`指向本次私有NAS审阅目录。原始CTTA/GraTa代码仍固定为`dbff0d985c6c95345d9fb78f5b1daef57b392564`与`33ae20d664f305af34739ec54a5bec7da53ffa0b`。

## 未运行项与交付边界

- 真实源checkpoint内容重验、真实目标RGB/mask读取和hash、真实历史标量重算：NOT_RUN。
- GPU mechanical smoke、真实24条目标流、效果指标、PCA实际ready率及恢复次数：NOT_RUN。
- 实际GPU吞吐/显存、共卡争用、混合型号时间可比性与长期预算可达性：NOT_MEASURED。
- 外部差异复核及阶段II执行授权：PENDING_EXTERNAL_REVIEW；本次不生成通过结论。

本轮真实GPU任务、正式记录、真实目标内容访问、源图像/mask/代理/原型访问和后台等待器均为0。测试中的completion/smoke字符串仅用于程序化夹具，不是真实设备smoke或审阅通过证据。短时CPU测试进程已收尾，未创建等待批准后自动启动的任务。

发布范围为修复源码、配置摘要、schema/预算说明、完整patch、[21文件源码归档](r1_fix_source.zip)、真实CPU证据和本报告。源码归档需要基线仓库及固定外部依赖，不含权重、原始数据、私有登记、真实身份/路径或论文PDF；这些材料不在公开边界内。严重NAS/内核不可中断IO可能延迟进程实际回收，该条件不允许标COMPLETE；CPU程序化检查也不代替GPU与真实流复核。

最终状态：**R1_REVIEW_FIXES_READY**。停止，等待对修复差异的外部复核。
