# R3 current execution status

**R3_EXECUTION_STOPPED_INCOMPLETE** — first authorized GPU attempt stopped at pre-model process audit. All 85 formal trajectories remain unstarted; both smoke workers and the launcher exited. No automatic retry or source change.

- [Stage II attempt report, failure boundary, physical cost and cleanup](stage2_attempt1/REPORT.md)
- [Exact failure/cleanup summary](stage2_attempt1/logs/failure.summary.private.json), [process matrix](stage2_attempt1/logs/matrix.processes.json)
- [Slot 0 traceback](stage2_attempt1/logs/device0.log), [slot 1 traceback](stage2_attempt1/logs/device1.log), [launcher exit](stage2_attempt1/logs/launcher.exit.json)
- [User-supplied external review pass](stage2_attempt1/input/R3_REVIEW_PASS_d601496a.md), [Stage II scope and no-retry policy](stage2_attempt1/input/R3_STAGE_II_EXECUTION_PROMPT_d601496a.md)

Execution SHA remains `d601496a0827af3e1fe728612a4b9f17a613955d`. This publication contains evidence only. The review packets below are historical; they do not imply successful GPU smoke or a completed experiment.

---

# Environment repair review packet (historical)

**R3_ENV_FIX_READY_FOR_REVIEW** — R3-ENV-01 repaired; external difference review pending. GPU execution remains unauthorized.

New implementation: **`d601496a0827af3e1fe728612a4b9f17a613955d`**, on `experiment/r3-five-region-frameworks-v1`. The following original packet is retained as history; review this environment-only delta first.

- [Environment fix report, entry points, before/after matrix and test counts](env_fix/REPORT.md)
- [Narrow patch relative to 6d8fc731](env_fix/ENV_FIX.patch), [pinned implementation](https://github.com/DLwbm123/DPA-CTTA/tree/d601496a0827af3e1fe728612a4b9f17a613955d)
- [Full 104/104 CPU log](env_fix/logs/cpu-full-01.log), [machine-readable result](env_fix/logs/cpu-full-01.json), [actual backend/counters](env_fix/CPU_BACKEND_EVIDENCE.json), [full-model traces](env_fix/FULL_MODEL_CPU_TRACES.json)
- [Retained first targeted failure](env_fix/logs/targeted-01.log), [corrected 9/9 targeted log](env_fix/logs/targeted-02.log), [CPU reproduction recipe](env_fix/REPRODUCE.md)
- [Frozen-input preservation](env_fix/FROZEN_INPUTS.json), [delivery manifest](env_fix/DELIVERY.json), [supplied external review](env_fix/input/R3_REVIEW_REPORT_6d8fc731.md), [supplied fix prompt](env_fix/input/R3_CODEX_ENV_FIX_PROMPT.md)

104 distinct tests = 98 prior + 6 new; targeted tests overlap. EIO ownership failure: **本次未重现，根因未知**. Shared supervisor and method numerics are unchanged. No GPU, real target/source assets, experiment or background task was used. This index does not issue external review approval or execution authorization. Stop here for external review.

---

# Original Stage I packet (historical)

**R3_IMPLEMENTATION_READY_FOR_REVIEW** — external review pending; GPU execution not authorized.

Review the code at **`6d8fc7317506400b039d3d2ed41bca18523beb27`**, based on `f52f132e4be576ea871467432a6f4b12337209a5`. The commit that contains this index is a separate documentation publication commit, not the runtime implementation SHA.

- [Implementation report and actual checks](IMPLEMENTATION_REPORT.md)
- [Complete base-to-implementation patch](IMPLEMENTATION.patch) and [pinned source tree](https://github.com/DLwbm123/DPA-CTTA/tree/6d8fc7317506400b039d3d2ed41bca18523beb27)
- [Science JSON](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/configs/r3_science_v1.json), [science / registration / stream digests](SCIENCE_DIGEST.json)
- [85-job JSON](DRY_RUN_MATRIX.json), [CSV with 1/2/3-worker rotations](DRY_RUN_MATRIX.csv), [new 33-chunk recurrence summary](STREAM_SUMMARY.json)
- [State lifecycle](STATE_LIFECYCLE.md), [method provenance and explicitly noted interpretations](METHOD_PROVENANCE.md)
- [Final 97-test combined CPU log](logs/cpu-06.log), [4 final focused checks](logs/final-focused.log), [untouched package reference log](logs/reference-01.log), [actual full-model CPU traces](FULL_MODEL_CPU_TRACES.json)
- [All first failures and residual EIO issue](DEVELOPMENT_LOG.md), [unrun scope](UNRUN_ITEMS.md), [delivery manifest](DELIVERY.json)

## Code and test navigation

| Scope | Key implementation | CPU evidence entry |
|---|---|---|
| C / RP | [dispatch and original R1 delegation](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/host.py#L25) | [two-step parity](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L49) |
| Common engine | [phase order / calls / commit](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/host.py#L81) | [real ResUNet 38-update CPU check](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L126) |
| T_LR / ISO / DIAG | [density and full-resolution teacher](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/teachers.py#L22) | [pair readiness / full-resolution / detachment](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L210) |
| U_PCA / RAND / SCALE | [feature-to-affine VJPs](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/displacement.py#L26); [actual Adam displacement](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/displacement.py#L43) | [finite differences](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L157); [Adam moments / norm control](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L171) |
| S_JOINT / SHARED / NOPCA | [small state pool](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/contexts.py#L23); [fixed router](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/kernels.py#L295) | [capacity and independence](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L181); [actual host switch / RNG](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L106) |
| M_TRANSPORT / IDPOST / SHUFFLE | [alignment and post insertion](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/transport.py#L9); [live and cached frame migration](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/stats.py#L41) | [all historical objects and controls](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L194) |
| G_PCA / ISO / ORDER | [graph teacher](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/teachers.py#L42); [fixed solver](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/kernels.py#L255) | [energy / containment](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3_reference.py#L120); [zero-edge full-resolution ORDER parity](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L210) |
| Sampling / statistics | [R1-compatible tokens and snapshots](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/stats.py#L8) | [R1 selection / grid / RNG parity](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L149) |
| 85 trajectories / recurrence | [metadata coverage](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/plan.py#L29); [85 jobs / rotations / budget](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/plan.py#L66) | [matrix and disabled entry](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3.py#L254) |
| Future execution / audit | [authorization](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/execution.py#L14); [unchanged supervisor adapter](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/execution.py#L136); [CPU scalar closeout](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/src/dpa_ctta/r3/analyze.py#L93) | [complete fixture / corruption / missing proof](https://github.com/DLwbm123/DPA-CTTA/blob/6d8fc7317506400b039d3d2ed41bca18523beb27/tests/test_r3_execution.py#L51) |

## Retained execution-layer contracts

No base file was changed. The new adapter reuses `r1.run.current` (mask after fixed output/state), `r1.assets` (verify/decode same bytes), `r1.evidence` (atomic invalidation/publication and ENOENT capacity scan), and `r1.supervise` (owned groups, finite dispatch, cleanup before failure-log writes). Relevant inherited regressions remain in `test_r1_fixes.py` and `test_r2_continuation.py`; no global R1/R2 arm or science constant was patched.

Review attention: one persistent-EIO ownership test failed intermittently in the earlier full run. Nine instrumented reruns and the final suite passed, but its cause is not established. This is a disclosed residual review item; the final green suite does not erase the failed observation.

The full-model CPU test uses random weights with a procedural head scaling to exercise confidence-dependent paths. T corrections are nonzero, U uses six real VJPs per ready tested arm in the 19,136D space, M's proper and shuffled rotations differ from identity, G has 1,984 active edges, and ORDER has zero. S multi-slot behavior is additionally tested by procedural descriptors; this does not predict whether actual data will create more than one context.

This index does not grant review approval. Stop after publishing this Stage I packet and wait for external review of the fixed code and config.
