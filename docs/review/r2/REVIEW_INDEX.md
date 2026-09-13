# R2 review index

Status: R2_IMPLEMENTATION_READY_FOR_REVIEW after the evidence completion described in IMPLEMENTATION_REPORT.md. This repository does not issue external review approval.

| Review concern | Material |
|---|---|
| Full implementation commit, real tests, exclusions | [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md), [DELIVERY.json](DELIVERY.json), [CPU_TEST_LOG.txt](CPU_TEST_LOG.txt), [CPU_TEST_RESULT.json](CPU_TEST_RESULT.json) |
| Exact scientific contract and digest | [science](../../../configs/r2_science_v1.json), [SCIENCE_DIGEST.txt](SCIENCE_DIGEST.txt) |
| Disabled execution / 20 frozen jobs | [defaults](../../../configs/r2_execution.defaults.json), [DRY_RUN_MATRIX.json](DRY_RUN_MATRIX.json) |
| Provenance and no inherited theoretical claim | [METHOD_PROVENANCE.md](METHOD_PROVENANCE.md) |
| Snapshot / merge timing, empty visit, cleanup | [STATE_LIFECYCLE.md](STATE_LIFECYCLE.md) |
| Weighted algebra | [weighted_pca.py](../../../src/dpa_ctta/r2/weighted_pca.py) |
| Paired loss, reduction, F and shuffled matching | [feature_losses.py](../../../src/dpa_ctta/r2/feature_losses.py), [memory.py](../../../src/dpa_ctta/r2/memory.py) |
| One-step sequence / compatibility | [host.py](../../../src/dpa_ctta/r2/host.py), unchanged [R1 host](../../../src/dpa_ctta/r1/host.py) |
| Matrix / authorization / process and IO reuse | [plan.py](../../../src/dpa_ctta/r2/plan.py), [run.py](../../../src/dpa_ctta/r2/run.py) |
| Historical binding / weighted replay / complete-result validation | [analyze.py](../../../src/dpa_ctta/r2/analyze.py) |
| Procedural tests and guarded CPU entry | [test_r2.py](../../../tests/test_r2.py), [check_r2_cpu.py](../../../scripts/check_r2_cpu.py) |
| Narrow diff against reviewed runtime | [IMPLEMENTATION.patch](IMPLEMENTATION.patch) |
| Metadata-only confirmation | [ASSET_METADATA_STATUS.json](ASSET_METADATA_STATUS.json) |
| Independent provided plan algebra | [PLAN_MATH_LOG.txt](PLAN_MATH_LOG.txt), [PLAN_MATH_RESULT.json](PLAN_MATH_RESULT.json) |

Review the new mathematics, lifecycle, old-control bindings and thin execution entry. Existing 37 R1 regressions remain intact. No performance gate, additional hyperparameter or experiment arm has been introduced.

CPU reproduction uses the existing environment and CTTA / GraTa dependency checkouts (digests in the implementation report). Set PYTHONPATH to src, tests and the existing dependency path; set DPA_CTTA_BASE_ROOT / DPA_GRATA_ROOT to those checkouts, RUN_FILE to scripts/check_r2_cpu.py, and invoke Python with `-c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'`. CHECKPOINT is discarded by this CPU entry. Optional CHECK_OUTPUT writes the machine-readable test status. R2_ONLY=1 runs only new checks during development. No dataset is needed.

Dry-run: use the same neutral entry with RUN_FILE=scripts/run_r2_matrix.py and no arguments. `--run` rejects the shipped disabled config before checkpoint/GPU access. Future `--recompute --assets <private-json> --output <private-run-directory>` recomputes only scalars on CPU; it cannot substitute for review or authorize GPU execution. On a shared server pass sensitive paths via environment/stdin rather than visible arguments.
