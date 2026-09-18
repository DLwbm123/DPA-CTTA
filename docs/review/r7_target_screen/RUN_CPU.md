# Reproduce acceptance without real assets

Use the exact IMPLEMENTATION_SHA executable tree with Python 3.12.9 / Torch 2.6.0 on the recorded local CPU runtime. Existing pinned code-only reference dependencies are required for the full R7 regression: DPA_CTTA_BASE_ROOT at dbff0d985c6c95345d9fb78f5b1daef57b392564, DPA_GRATA_ROOT at 33ae20d664f305af34739ec54a5bec7da53ffa0b, existing batchgenerators 0.25.2. No dependency installation or GPU qualification is part of this task.

Set PYTHONPATH=src, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES empty. Use neutral `python -` processes, not a project-named script on the command line. Run separately:

```python
import runpy
runpy.run_path('scripts/r7/check_target_screen_cpu.py', run_name='__main__')
```

```python
import runpy
runpy.run_path('scripts/r7/check_cpu.py', run_name='__main__')
```

```python
import runpy
runpy.run_path('scripts/r7/check_source_prep_cpu.py', run_name='__main__')
```

The target harness blocks GPU discovery/initialization, permits torch/PIL loads only from BytesIO, and restricts verified asset reads to its generated temporary fixture directories and checked-in science specs. Real target/source/checkpoint reads are zero. Availability is stubbed false for unchanged CPU seeding helpers. It prints observed command line and implementation HEAD. Each process has two Torch threads. Logs/results are written outside the checkout during the final fixed-SHA run, then added in a documentation-only publication commit.

Future real launch is deliberately not invoked here. The neutral environment-selected entry is `scripts/r7/target_screen.py` with SCREEN_RECEIPT supplied; absent/disabled receipt raises immediately. Exact bindings, external source-artifact review, execution-layer review and user/resource receipt are still required. No built-in smoke, qualification, source preparation, retry/resume or subsequent-scope flags exist.
