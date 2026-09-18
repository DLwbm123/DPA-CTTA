# Exact-SHA CPU acceptance

Check out `f719c703087b38c07bdfbe7ce9dcfa62d88a12d9` and require a clean checkout. Use a separate Python process for each command, CUDA_VISIBLE_DEVICES empty, PYTHONPATH=src, PYTHONDONTWRITEBYTECODE=1, two CPU threads, and the previously pinned code-only dependency roots. No execution receipt is supplied.

```python
import runpy
runpy.run_path('scripts/r7/check_source_prep_cpu.py', run_name='__main__')
```

```python
import runpy
runpy.run_path('scripts/r7/check_cpu.py', run_name='__main__')
```

Use neutral `python -` entrypoints. The logs include exact git SHA and observed ps command line. Local Python3.12.9/Torch2.6.0 uses the existing pinned reference/GraTa roots; server Python3.10.6/Torch2.2.1+cu121 uses the same pinned code lineage and existing batchgenerators0.25.2 site-packages appended to sys.path. Torch's CUDA build string is not GPU execution: the harness blocks actual GPU queries/initialization. No dependency installs or real checkpoint/image loads.

Before-fix reproduction: use the base publication's production files and the candidate test file, with tests/r7 on sys.path; run test_SP1_noncanonical_paths_reject_before_assets_or_output, test_SP2_first_error_survives_evidence_failure, test_SP3_already_exited_child_over_tree_cap, and test_SP3_terminal_wall_boundary from PreparationTests. Four methods can yield more than four unittest failures because of subtests. This was an uncommitted test overlay, not a clean final-SHA acceptance.

Raw logs remain private; public copies redact only local/server/temp paths. LOG_MANIFEST binds both distinct byte sets. CPU_RESULTS includes raw final results, real wall/CPU/RSS/hash costs and prior failing development attempts. It does not relabel a failing development run as PASS. No old166 or standalone22 reference rerun.
