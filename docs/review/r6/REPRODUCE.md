# 复现

在完整实现提交 `c94fff7c05cec38d541c441b62a9381ee46ba076` 的独立 checkout，以已有环境运行，不安装/升级依赖。后续 docs/review/r6 证据提交与实现提交分开。

设好现有固定依赖的本机路径：DPA_CTTA_BASE_ROOT 对应 CTTA dbff0d985c6c95345d9fb78f5b1daef57b392564；DPA_GRATA_ROOT 对应 GraTa 33ae20d664f305af34739ec54a5bec7da53ffa0b。PYTHONPATH 包含当前 checkout 的 src、tests，以及服务器原有依赖 site-packages（如需要）。各端 torch 不升级。

```sh
export RUN_FILE="$PWD/scripts/check_r6_cpu.py"
export PYTHONPATH="$PWD/src:$PWD/tests:$PYTHONPATH"
export R3_REGRESSION=1
export CUDA_VISIBLE_DEVICES=''
# Server: create a new temporary directory under this job's registered NAS root.
export TMPDIR=/path/to/new/NAS/job/tmp
mkdir -p "$TMPDIR"
unset R6_MODULES R5_FAST CHECKPOINT
export CHECK_OUTPUT=/path/to/new/private/cpu-summary.json
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > /path/to/new/private/cpu.log 2>&1
```

命令入口中性，具体项目路径在环境变量中。CPU suite 以前台子进程执行、保持连接至退出，不启动后台等待或GPU监测。日志中的错误注入须以 unittest 结果判断；不能截取预期异常栈冒充真实失败，或删掉它们。

只读真实 registration JSON 生成元数据（不要读取其中的资产路径）：

```python
from dpa_ctta.r6_regional_consistency.plan import dry_run
value = dry_run(registration_metadata)
```

发布的 metadata/DRY_RUN.json 使用当前真实登记和指定 recurrence 摘要；如果缺少登记元数据，应输出准确缺失状态，不能用 tests fixture 替代。本次 science JSON 为原提案字节副本，science_bytes_verified=true；摘要不符仍硬拒绝。

host_scalar_probe.py 是额外 CPU host→scalar 集成复现脚本，放入 RUN_FILE 后使用同一环境入口；程序化输入，无checkpoint和目标读取。它不触发正式 trajectory。

目前 execution.defaults 全部 disabled，设备/caps批准/receipt均空；切勿将本复现说明当成 R6-A/B 授权。真正执行必须另有确切 scope、code/science/registration/stream/fingerprint/device/caps/review-or-waiver binding；B还需重新验证完整A及其原执行绑定。

## 本次原件/共同输入复现

在相同候选checkout和上述环境中，将RUN_FILE设为`$PWD/docs/review/r6/check_originals_cpu.py`，CHECK_OUTPUT及stdout/stderr设为新的输出位置，使用同一个中性python入口。脚本import原参考且只执行ReferenceTests，绝不直接运行原件__main__（其main会写R6_MATH_CHECK.json）。不运行math_history中的首次失败版本。

该入口有原参考8项的新执行和3个核对方法，含240组共同输入及402组gate比较。容差与d875f20生产TOLERANCES相同；它不替代完整166项。所有原件和历史日志不修改，新日志与已提供历史8项分开保存。

本次server设置TMPDIR为专用NAS临时目录；checkout、bundle、CPU日志、exit证据均在既定NAS项目路径。完整回归以前台CPU子进程执行并等待退出，没有后台实验或GPU查询。
