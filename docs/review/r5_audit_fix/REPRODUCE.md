# CPU 复现

以 implementation SHA b4b71601a5bdf87bb3a7e5d3db352610adcdff74 的源码运行。复用已固定 CTTA/GraTa 源码和 batchgenerators 0.25.2，不升级依赖。

```sh
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$PWD/src:$PWD/tests${DEPENDENCY_PYTHONPATH:+:$DEPENDENCY_PYTHONPATH}"
export RUN_FILE="$PWD/scripts/check_r5_cpu.py"
export R3_REGRESSION=1
export CHECK_OUTPUT="$NEW_CPU_LOG_DIR/check.json"
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > "$NEW_CPU_LOG_DIR/check.log" 2>&1
```

DPA_CTTA_BASE_ROOT、DPA_GRATA_ROOT 指向原锁定依赖；NEW_CPU_LOG_DIR 使用独立新目录，保留每次日志。可见计算进程仅使用中性 python -c，路径通过环境变量传递。服务器同样是前台 CPU 检查，不是后台监测或模型实验。

不设 R3_REGRESSION 时只运行全部 31 项 R5；R5_MODULES=test_r5_audit_fix 是聚焦新增 9 项的诊断方式，不能替代交付所需完整回归。本轮最终不设 R5_FAST，没有省略完整 ResUNet。脚本拒绝 CUDA 初始化、真实 asset loader、source proxy；继承 IO 测试仅允许现场生成的独立临时文件。

科学保持性以 PRESERVATION.json 为准：science 原始字节不变，当前 dry_run(原已授权 registration metadata) 与历史 DRY_RUN.json 相同。公开包不含该私有 registration、逐内容行、RGB、mask、checkpoint 或 optimizer。已有矩阵见 ../r5/A_MATRIX.json、B_NEW_MATRIX.json、AB_MATRIX.json；不是正式运行结果。
