# R3 Stage II attempt 1: stopped before model smoke

**R3_EXECUTION_STOPPED_INCOMPLETE**. The single authorized attempt exited; no retry or code change followed. This is a failure record, not `R3_EXPERIMENT_COMPLETE`.

Execution SHA: `d601496a0827af3e1fe728612a4b9f17a613955d`. The source/science/85-job matrix are unchanged. External review pass was supplied by the user in [R3_REVIEW_PASS_d601496a.md](input/R3_REVIEW_PASS_d601496a.md). The user explicitly authorized all 85 trajectories, physical GPUs 6 and 7, maximum concurrency 2, and background execution. [The supplied Stage II instructions](input/R3_STAGE_II_EXECUTION_PROMPT_d601496a.md) prohibit automatic retry or modifying code and continuing after failure.

## Verified preparation and actual attempt

A clean detached checkout at the exact execution SHA was created on the existing server. A new private control/output location was used; previous jobs and the repository's disabled defaults were retained. Existing R2 asset configuration was reused with its registration unchanged. Science, registration and secondary stream digests all matched the reviewed bindings; both pinned code dependencies matched. NFS mount/capacity and one write/fsync/read probe passed. Physical GPUs 6/7 each had 24,124 MiB free at the pre-launch check; the reviewed launcher performed its own resource check.

The existing runtime was **Python 3.10.6 / PyTorch 2.2.1+cu121**, not the local CPU review environment's Python 3.12.9 / Torch 2.6.0. No environment installation or replacement was made. The [preflight record](logs/preflight.json) discloses this runtime explicitly; preflight imported code and checked metadata/checkpoint size without loading target pixels or initializing CUDA.

The neutral, environment-driven background launcher started two smoke workers. Both exited with `ValueError: visible process command audit` at the same source location. Supervisor stopped dispatch, recorded both nonzero exits and all 85 jobs as unstarted, then returned failure. The launcher's exit code was 1. No automatic monitoring task, retry, replacement worker batch, method change or extra experiment was created.

## Counts and cleanup evidence

| Item | Observed result |
|---|---:|
| Completed formal trajectories | 0 / 85 |
| Started formal trajectories / scoring records | 0 / 0 |
| Unstarted formal trajectories | 85 |
| Smoke workers started / completed | 2 / 0 |
| Network forwards / loss backward / Adam / VJP | 0 / 0 / 0 / 0 |
| Registered checkpoint loads completed | 2 × 90,388,078 bytes |
| True target RGB / mask reads | 0 / 0 |
| Accumulated active worker time | 6.6153 seconds |
| Supervisor wall time | 3.3821 seconds |
| Whole launcher elapsed time | 5.3118 seconds |

Zero model-call/target-read counts are derived from both exact tracebacks stopping at `process_audit` **before `smoke()` is called**, together with zero formal dispatch. They are not a saved smoke counter block: that function never ran and the failure JSON therefore contains no physical/backend block. The earlier calls on the same worker line loaded the registered checkpoint and queried backend successfully. These startup costs are retained and must not be hidden in any future execution's accounting. There is no completed or partial trajectory prefix to carry forward.

[Failure summary](logs/failure.summary.private.json), [process matrix](logs/matrix.processes.json), [started process ledger](logs/processes.started.json), and [dispatch stop](logs/dispatch.stopped.json) preserve the failed attempt. Worker PIDs/PGIDs were 3539381 and 3539382; both exit codes were 1. The supervisor's records show their exits. A post-failure check found neither PID nor either owned process group present; parent 3539365 had also exited. No unrelated process was signalled. The observer was not the children's parent and does not claim a separate `waitpid` result; the owned supervisor's exit records and process-group absence are the available cleanup evidence.

## Failure location and root-cause boundary

The exact [R3 worker line](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/r3/execution.py#L146) calls `checkpoint`, `environment`, `backend_policy`, then `process_audit`, before calling `smoke` and constructing its first model. The original [audit predicate](https://github.com/DLwbm123/DPA-CTTA/blob/d601496a0827af3e1fe728612a4b9f17a613955d/src/dpa_ctta/b3_runtime.py#L53) rejects either a missing worker PID in the `nvidia-smi` compute-process list or a forbidden substring in parent/worker command evidence. The exact `ps` and `nvidia-smi` values were not written because the guard throws before writing its audit JSON.

Thus the proven failure is the **pre-model GPU process audit rejection**, not a numerical error, workspace mismatch, target-data error, or the old NFS/EIO issue. A missing GPU compute-process listing before the first model allocation is a plausible explanation; it is not established by the saved logs. The original R1 smoke places its audit after constructing the first GPU host, which is a relevant placement difference. No additional GPU probe was run to force a reproduction and no guard was weakened or bypassed.

Next engineering work would need to establish which predicate fired, retain failure-time command/device evidence, and review audit placement while preserving neutral commands and all scientific/model-call budgets. This report does not implement or authorize that work or a replacement attempt. A future successful attempt would still owe both per-device registered smokes and all 85 complete trajectories; startup overhead from this failed attempt remains separately accounted.

## Raw evidence and publication boundary

- [GPU slot 0 traceback](logs/device0.log), [failure JSON](logs/device0/smoke.failure.json), [supervisor failure](logs/device0/supervisor.failure.json)
- [GPU slot 1 traceback](logs/device1.log), [failure JSON](logs/device1/smoke.failure.json), [supervisor failure](logs/device1/supervisor.failure.json)
- [Launcher traceback](logs/launcher.log), [exit](logs/launcher.exit.json), [launch receipt](logs/launch.receipt.json), [storage probe](logs/storage_probe.json)

Files under `logs/` are de-identified copies. Only private directory paths, the private run identifier and device UUIDs were replaced with explicit placeholders; exception text, source lines, process IDs, timestamps, counts and status were retained. The file `failure.summary.private.json` keeps its original filename but its published content is redacted. Original records, enabled authorization, registration, packet and device bindings remain private. No checkpoint, image, mask, private asset config or patient identity is published.

The prior 104/104 CPU review evidence is historical validation of the reviewed source, not proof of this GPU startup. No CPU full-suite rerun or scalar performance recomputation was claimed for this failed attempt. Scientific digests remain:

- Science: `73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`
- Registration: `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`
- Secondary stream: `cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`

**Stopped. Explicit authorization for engineering repair and subsequent execution is required by the supplied Stage II failure policy.**
