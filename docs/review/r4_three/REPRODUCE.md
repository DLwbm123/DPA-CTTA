# CPU 与元数据复现

使用已具备 Torch、原锁定 batchgenerators 等依赖的 Python。DPA_CTTA_BASE_ROOT 指向固定 dbff0d985c6c95345d9fb78f5b1daef57b392564 的 CTTA 代码依赖，DPA_GRATA_ROOT 指向 33ae20d664f305af34739ec54a5bec7da53ffa0b。不得替换固定依赖源码或加载权重做 CPU 阶段验收。

从对应实现提交的仓库根目录设置：

```sh
export PYTHONPATH="$PWD/src:$PWD/tests"
export RUN_FILE="$PWD/scripts/check_r4t_cpu.py"
export R3_REGRESSION=1
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
```

检查器禁用 CUDA 初始化、源代理与真实资产读取；旧 IO 回归只允许它自己生成的临时程序化文件。完整模型使用随机初始化的固定结构，真实 checkpoint 读取数为零。测试必须报告实际解释器/Torch，不将本地环境冒充服务器环境。

普通无参数入口只输出算术计划。真实 metadata dry-run 使用原私有 registration JSON 与新的去身份输出目录：

```sh
python scripts/run_r4t.py --registration "$REGISTRATION_METADATA" --out "$METADATA_OUTPUT"
```

只读取 JSON；同一 registration 和 recurrence 摘要匹配后生成 70 个 job。实际执行与源码/科学完整 SHA 绑定，defaults 仍 disabled；不把本例当作 GPU 或 review 授权。

公开 scalar fixture 验证所有 70 job/五流、原子失效和辅助计数污染；公开结果只有构造的数据，不是真实实验分数。CPU 原始日志的本地绝对路径会替换为 <REPO> 或 <PYTHON_ENV>，错误类型、断言、计数和时间不改。
