# R3 process audit repair

**R3_PROCESS_AUDIT_FIX_READY_FOR_REVIEW**. Implementation `185fd440b16920f367c1f5e3096ab495bd85c0ec`; baseline `d601496a0827af3e1fe728612a4b9f17a613955d`. Evidence is published in a separate later commit on `experiment/r3-five-region-frameworks-v1`. No replacement GPU run was started.

## Problem, correction and evidence boundary

The [first authorized GPU attempt](../stage2_attempt1/REPORT.md) stopped both workers at `visible process command audit`, before constructing a host or calling smoke. All 85 formal trajectories remain unstarted. Its raw `ps` and GPU-query values were not persisted, so the exact failing predicate in that incident cannot be retrospectively established.

R3 incorrectly required its GPU PID to be visible before allocating its first model, whereas R1/R2 and the B3/B4/B5 execution paths audit after their existing model allocation. The fix moves R3 smoke's audit into its normal first `OLD_C` host path, **after construction and before the first forward**, once per GPU worker. CPU direct smoke skips the GPU process query. No additional allocation, warm-up, forward, update, retry or polling is introduced. The formal path's audit remains after host allocation, unchanged.

The shared `b3_runtime.process_audit` retains the same acceptance predicate: the worker must appear in the GPU compute-process list, and neither its command nor its parent's command may contain a forbidden substring. It now writes mode-0600 private evidence for accepted and rejected observations: original `ps`, own GPU rows, complete GPU query, `neutral`, and distinct `missing_gpu_process` / `non_neutral_command` reasons. If a rejected audit's evidence write raises OSError, the original shared-code ValueError remains primary with the write error chained as its cause. A successful predicate with a failed write still propagates the original IO error. No failure is swallowed.

The full GPU query may contain unrelated process names/UUIDs. This is private diagnostic evidence, not a public artifact. Future export must redact or withhold those fields; this packet includes only procedural query fixtures and previously de-identified failure logs.

This repairs the misplaced precondition and makes subsequent failures diagnosable. The CPU simulation is not proof that the original server's precise rejection branch was missing PID, nor a successful CUDA smoke result. A future authorized, reviewed attempt must still verify actual GPU visibility and all original 85 trajectories.

## Scope and source navigation

| Change | File / entry |
|---|---|
| Move smoke audit; preserve worker authorization and workspace gate | [execution.py:56](https://github.com/DLwbm123/DPA-CTTA/blob/185fd440b16920f367c1f5e3096ab495bd85c0ec/src/dpa_ctta/r3/execution.py#L56), `smoke` and `worker` |
| Persist audit evidence and distinct rejection reasons | [b3_runtime.py:48](https://github.com/DLwbm123/DPA-CTTA/blob/185fd440b16920f367c1f5e3096ab495bd85c0ec/src/dpa_ctta/b3_runtime.py#L48), `process_audit` |
| Three new regression tests | [test_r3_execution.py:54](https://github.com/DLwbm123/DPA-CTTA/blob/185fd440b16920f367c1f5e3096ab495bd85c0ec/tests/test_r3_execution.py#L54), first three tests |
| Enforce no GPU audit in existing full-model CPU smoke | [test_r3.py:126](https://github.com/DLwbm123/DPA-CTTA/blob/185fd440b16920f367c1f5e3096ab495bd85c0ec/tests/test_r3.py#L126), 38-update test |

[PROCESS_AUDIT_FIX.patch](PROCESS_AUDIT_FIX.patch) contains the complete four-file code/test difference from the reviewed baseline, scoped to `src scripts tests configs`. Historical publication commits between the baseline and repair contain documents only. The shared supervisor and IO/NFS implementations are not changed; only the shared audit's diagnostic behavior changes. All source callers were inspected to keep their existing post-allocation ordering.

## CPU results and negative control

| Run | Actual result | Time | Evidence |
|---|---|---:|---|
| Current-code targeted | 12/12 passed | 2.35 s | [log](logs/targeted-01.log), [JSON](logs/targeted-01.json) |
| Current-code normal full suite (one run) | **107/107 passed**, no failures/errors/skips | 226.35 s | [complete log](logs/cpu-full-01.log), [JSON](logs/cpu-full-01.json) |
| Corrected old-worker negative control | Specific old-order rejection observed; this is intentionally a failing test of old code | 0.008 s unittest time | [log](logs/old-order-witness-02.log) |

The real procedural ResUNet smoke retained **316 forwards, 38 Adam, 38 loss backward, 18 VJPs and 5 parameter replacements**. [Full model traces](FULL_MODEL_CPU_TRACES.json) and [backend/counter evidence](CPU_BACKEND_EVIDENCE.json) are saved. Comparison flags remained strict/false, restoring the original algorithms/warn-only flags; CPU workspace remained absent. This suite ran locally on Python 3.12.9 / Torch 2.6.0 with CUDA uninitialized. It does not replace validation on the original server runtime (Python 3.10.6 / Torch 2.2.1+cu121).

Three new test methods cover: the real worker-to-smoke control flow with a CPU-only host substitute and model-dependent simulated GPU listing; six acceptance/rejection/query-preservation cases; and rejected versus accepted audits under injected evidence-write EIO. Authorization/environment checks and the prior suite remain intact. The direct CPU model test now forbids process-audit invocation while still executing its original complete numerical workload.

107 distinct tests = 104 prior + 3 new. The 12 targeted checks overlap the full suite. The historical ordering negative control reuses one of these tests and is not an additional independent test or GPU experiment.

The [corrected negative-control script](old_order_witness.py) compiles only the exact old worker function from the baseline Git object, using the current module globals so the same explicit CPU collaborators apply. [Its log](logs/old-order-witness-02.log) has the expected old-order errors: missing PID rejection before constructor invocation. The script itself exits zero only if the specific missing-PID and never-constructed-host failures are found. This demonstrates sensitivity to the misplaced audit; it does not run the original source as a full GPU experiment.

The first negative-control harness copied module globals, so later test mocks did not reach the old function. Its `RUN_PACKET` KeyErrors are **invalid control evidence**, despite its too-broad initial status marker. [That original script](old_order_witness_01.py) and [log](logs/old-order-witness-01.log) are retained. The corrected harness binds live globals and asserts the exact expected failure signatures. No production source was changed to address that harness problem. No failures of current-code tests are concealed by this control.

## Reproduction and exclusions

Use the existing CPU environment and code-only dependencies at CTTA `dbff0d985c6c95345d9fb78f5b1daef57b392564` and GRATA `33ae20d664f305af34739ec54a5bec7da53ffa0b`; set `DPA_CTTA_BASE_ROOT` / `DPA_GRATA_ROOT` accordingly. From the implementation checkout:

```sh
export PYTHONPATH="$PWD/src:$PWD/tests"
export RUN_FILE="$PWD/scripts/check_r3_cpu.py"
R3_EXECUTION_ONLY=1 CHECK_OUTPUT=targeted.json "$PYTHON" -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > targeted.log 2>&1
R3_REGRESSION=1 CHECK_OUTPUT=cpu-full.json TRACE_OUTPUT=full-model-traces.json "$PYTHON" -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")' > cpu-full.log 2>&1
```

Run in the foreground with the existing interpreter named by `PYTHON`. The checker blocks CUDA initialization, source proxies and real registered assets; inherited IO tests may use only generated temporary fixtures. The simulated CUDA device string in the new ordering test reaches only a fake constructor; `ps` and `nvidia-smi` are mocked, not executed. Original logs remain private; published logs replace local paths only. No new GPU query, true target RGB/mask, source asset, registered checkpoint load, remote deployment, experiment or automatic monitoring occurred in this repair.

## Frozen science and residuals

[FROZEN_INPUTS.json](FROZEN_INPUTS.json) records unchanged method modules, supervisor/IO code, plan, disabled defaults and original metadata, with a fresh science SHA check:

- Science: `73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`
- Registration: `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`
- Secondary stream: `cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`

Private registration/identity sequences were not re-read or recomputed. The 17-arm / 85-trajectory matrix, 1,951 records per trajectory, four-primary/secondary separation, C/RP, T/U/S/M/G mathematics, numerical tolerances and all frozen budgets remain unchanged. Smoke still uses `:4096:8`, strict determinism with warn-only false; formal removes the workspace variable and keeps its prior policy. The environment regression tests remain in the suite.

The inherited persistent-EIO supervisor case passed in this normal regression: all four owned children returned and were reaped before fallback; the unrelated control remained alive until test fallback. **本次未重现，根因未知。** The new audit-write EIO test checks exception precedence only and is not a diagnosis or repair of the historical supervisor anomaly. No repeated supervisor diagnostics or automatic retry was added.

The failed GPU attempt's 0 formal trajectories, 0 model calls, two checkpoint loads and approximately 6.6153 accumulated worker seconds remain in the earlier failure packet. There is nothing to resume mid-trajectory. Review and any new execution authorization must bind this repair's full implementation SHA, not the historical pass for `d601496a` or this report's publication SHA. The current user authorization is to repair and submit for review; no review pass or renewed execution permission is generated here.
