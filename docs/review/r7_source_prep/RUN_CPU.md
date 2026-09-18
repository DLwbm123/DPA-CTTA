# Reproduce procedural acceptance

Check out IMPLEMENTATION_SHA, use Python3.12/Torch2.6 or the observed server Python3.10/Torch2.2 environment. Set PYTHONPATH=src, PYTHONDONTWRITEBYTECODE=1, CUDA_VISIBLE_DEVICES empty. Run using neutral `python -` / runpy entrypoints; no real SOURCE_PREP receipt is supplied.

```python
import runpy
runpy.run_path('scripts/r7/check_source_prep_cpu.py', run_name='__main__')
```

This runs17 execution-layer checks against temporary PNGs and random Small checkpoint bytes. It forbids CUDA queries/initialization and only accepts BytesIO for test image decoding and checkpoint deserialization. It records actual model/optimizer/IO/hash counters. Three short synthetic child processes test success, nonzero exit and timeout/owned cleanup. Test receipt/review flags are explicitly SYNTHETIC_TEST_ONLY_NOT_REAL_AUTH fixtures, not external review or real user authority. They do not persist outside test temp directories.

The original59-method R7 suite runs separately with the existing pinned code-only dependency roots in DPA_CTTA_BASE_ROOT and DPA_GRATA_ROOT:

```python
import runpy
runpy.run_path('scripts/r7/check_cpu.py', run_name='__main__')
```

On the server, the existing pinned batchgenerators0.25.2 dependency path is appended to sys.path; no package installation or GPU probe is needed. This regression covers random full ResUNet micro-training/calibration, state/context loading and original-C preservation. It forbids real asset loaders. The independent22-check historical reference and166-test older suite were not rerun in this preparation: equations and tolerances are unchanged, and the59 suite retains its mathematical differential checks.

Actual exit codes, failures, warning lines, per-run counters, time and peak RSS are in CPU_RESULTS and logs. Uncommitted development runs are separate from final-SHA acceptance. The first local development run had6 errors due to macOS /var temporary-root symlinks; fixtures were changed to their actual resolved temporary root without relaxing production path checks. The second had1 process-group cleanup EPERM error; the narrow checked macOS fallback is documented in IMPLEMENTATION. Those original logs remain. No failure is relabeled PASS and no skipped server execution is invented.

The real source metadata audit is separate: existing JSON registrations and known checkpoint raw bytes only. Real checkpoint hashing is not model loading. Source RGB/masks have not been decoded or rehashed this turn. Do not invoke the real source entry to run these tests. SOURCE_PREP and every TARGET scope remain NOT_RUN.
