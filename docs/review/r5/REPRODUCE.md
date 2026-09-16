# CPU 与 metadata 复现

使用已安装 Torch 与固定 batchgenerators=0.25.2 的解释器；DPA_CTTA_BASE_ROOT、DPA_GRATA_ROOT 指向锁定源码依赖。无需也不允许加载真实 checkpoint/像素。

```sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/src:$PWD/tests"
export RUN_FILE="$PWD/scripts/check_r5_cpu.py"
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
```

新 R5 套件共 22 个 test methods，包括随机权重完整 ResUNet 四条四步匹配、1951 内容×3 的 A 标量矩阵，以及分开创建的 17-job B fixture。合计20不是正式实验，不产生真实效果证据。R5_FAST=1 仅用于早期轻量检查，会明确 skip 完整网络；交付最终检查不设此值。

R3_REGRESSION=1 加载原 IO/NFS/进程监督/授权与 R1–R4 相关回归。其临时小图/权重是测试当场生成的，只允许专用临时目录/BytesIO；不能把它记为真实 target/checkpoint 读取。测试脚本阻断 CUDA 初始化、真实 asset loader 与 source proxy。日志 JSON 记录实际环境、通过/失败/跳过、CUDA、真实资产数、B1 路径实测 CPU 调用计数；含旧回归时该计数不涵盖所有非 B1 方法，只作明确范围内的物理记录。

无参 scripts/run_r5.py 只给算术计划。带原已授权 registration JSON 的 metadata dry-run 只解析 JSON、核对登记摘要与流：

```sh
export RUN_FILE="$PWD/scripts/run_r5.py"
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' --registration "$REGISTRATION_JSON" --out "$NEW_METADATA_DIRECTORY"
```

实现入口默认 disabled，A/B_NEW scope、完整 code/science/data/stream/C 指纹、新 review 或显式 waiver、设备/caps/smoke 配方均缺一不可。R4 waiver 不可复用。AB 不可作为真实执行 scope。未来 B 还需另行审阅配方和有效 A 的固定校准；当前 p_accept=null。

日志只替换绝对环境路径，保留错误/断言/数量/时间。所有首次检查和最终检查均保留；没有为过测调整门限。
