# CPU verification recipe

Use the existing Python 3.12 / Torch 2.6 environment and the original pinned code dependencies. No installation, registered checkpoint, private registration, source data or target image is needed.

Set `DPA_CTTA_BASE_ROOT` to the CTTA code-only checkout at `dbff0d985c6c95345d9fb78f5b1daef57b392564`, and `DPA_GRATA_ROOT` to the GRATA code-only checkout at `33ae20d664f305af34739ec54a5bec7da53ffa0b`.
From the implementation checkout, with `PYTHON` naming the existing interpreter:

```sh
export PYTHONPATH="$PWD/src:$PWD/tests"
export RUN_FILE="$PWD/scripts/check_r3_cpu.py"
R3_EXECUTION_ONLY=1 CHECK_OUTPUT=targeted.json "$PYTHON" -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > targeted.log 2>&1
R3_REGRESSION=1 CHECK_OUTPUT=cpu-full.json TRACE_OUTPUT=full-model-traces.json "$PYTHON" -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > cpu-full.log 2>&1
```

Run each command in the foreground. Do not set both suite selectors. The checker disables CUDA visibility, forbids CUDA initialization and source proxies, and permits existing IO regressions to decode only assets created under their temporary procedural fixture tree. The real-model smoke uses random procedural weights; it does not load a registered source checkpoint. The future GPU launch entry is never invoked by this recipe.

The full suite includes the targeted tests; adding their totals would double-count. Supervisor tests deliberately create short CPU child processes and inject ENOSPC/EIO. Their captured nonzero subprocess exits and exception tracebacks are expected evidence; the unittest outcome identifies whether the ownership assertions passed. They include fallback cleanup, do not start experiments, and do not retry.

Published logs preserve test names, results, counts, timings, exceptions and procedural PID relationships. Local home/worktree/interpreter/temp paths are replaced with `<HOME>`, `<WORKTREE>`, `<PYTHON_PREFIX>` and `<TEMP_PATH>`. Original logs remain private. These substitutions are the only log editing.
