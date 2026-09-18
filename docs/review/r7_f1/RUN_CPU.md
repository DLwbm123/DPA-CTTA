# Reproduce the fixed-SHA CPU checks

Check out the exact commit in `IMPLEMENTATION_SHA`. Configure `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=` and the existing pinned code-only dependency roots via `DPA_CTTA_BASE_ROOT` and `DPA_GRATA_ROOT`. No image, mask or checkpoint paths are inputs. Run through neutral `python -` / `runpy` entrypoints if process-name rules apply.

```python
import os, runpy
os.environ['R7_RESULT'] = '/new-private-dir/implementation.json'
runpy.run_path('scripts/r7/check_cpu.py', run_name='__main__')
```

Run the independent original reference in a separate invocation:

```python
import os, runpy
os.environ['R7_MATH_RESULT'] = '/new-private-dir/reference.json'
runpy.run_path('docs/review/r7/input/checks/test_math_reference.py', run_name='__main__')
```

The implementation runner includes all original 47 methods plus 12 context methods. It counts actual Tensor.backward, autograd.grad, Adam and AdamW calls, old-C forwards, R7 forwards, latent kernels and context hashing. It blocks checkpoint loaders, actual source/target readers and CUDA initialization/discovery, with the legacy availability call replaced by a fixed CPU-only answer. No warning filter was added. Existing numeric comparison tolerances are unchanged.

Local environment: Python3.12.9 / Torch2.6.0 / NumPy2.2.3. Server environment: Python3.10.6 / Torch2.2.1+cu121 / NumPy1.26.4, executed strictly on CPU; its existing pinned `batchgenerators==0.25.2` code dependency was appended to sys.path. No package installation or device migration was performed. Final environment versions and exit codes are in CPU_RESULTS.json and logs.

The archived `reproduce_reviewed_code.py` is intentionally for the *old* reviewed source: run it with that checkout's src and tests/r7 on sys.path. Its two rejection assertions fail on the old API. On fixed code use `test_context`, which actually executes the new trusted-context API and checks successful copies and rejection before forwards; passing an old digest to the new loader would only test legacy-format rejection.

Public logs redact only local/remote account and filesystem roots. Raw logs remain private, with raw and public SHA256 recorded separately. Old failures remain evidence rather than being relabeled as final failures. CPU_RESULTS reports each development run separately; it does not inflate coverage by adding repeated runs. Mocked source-preparation packaging tests do not execute the mock step counters or substitute for real source training.
