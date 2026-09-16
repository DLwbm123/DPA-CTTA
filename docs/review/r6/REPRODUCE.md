# 复现

在完整实现提交 `d875f20c11cc7e617c9f39dba04ed46381aba450` 的独立 checkout，以已有环境运行，不安装/升级依赖。后续 docs/review/r6 证据提交与实现提交分开。

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

发布的 metadata/DRY_RUN.json 使用当前真实登记和指定 recurrence 摘要；如果缺少登记元数据，应输出准确缺失状态，不能用 tests fixture 替代。本次原始 science JSON 仍缺失，science_bytes_verified=false；代码硬拒绝把重建 JSON 当原件。

host_scalar_probe.py 是额外 CPU host→scalar 集成复现脚本，放入 RUN_FILE 后使用同一环境入口；程序化输入，无checkpoint和目标读取。它不触发正式 trajectory。

目前 execution.defaults 全部 disabled，设备/caps批准/receipt均空；切勿将本复现说明当成 R6-A/B 授权。真正执行必须另有确切 scope、code/science/registration/stream/fingerprint/device/caps/review-or-waiver binding；B还需重新验证完整A及其原执行绑定。
