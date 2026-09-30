# R10_ATTRIBUTION_SUP_8H_V1 — verified launch

Status: RUNNING, not an experimental result. At the startup check on 2026-09-30 18:13:38 Asia/Shanghai, source counterfactual context 1/32 was saved, with one GPU worker on authorized idle GPU 5. No immediate error. The watchdog, supervisor and worker all use neutral command lines. Other users' processes remain untouched.

Execution SHA: `fa32e7fc40629280d04ed1770979984193bace68`; base `2126c7f81b26b5d49c058c4fae0ddcafa79b3462`.
Frozen config digest: `b1579024e2f4be35a1ffd749d7b56bfdc011c8caacbdf7b892c813ec6833ff9c`.
Preflight execution SHA: `fe07c9f43155b45d7eaddee47d7e25f95f88fbc7`. Final changes add reporting, an exact counterfactual operation check and orchestration bookkeeping; the measured Trainer/AuditTrainer numerical paths are unchanged.

T0: 2026-09-30 17:57:58 Asia/Shanghai. Preflight plus launch took 918.532 seconds, below 30 minutes. Normal compute cutoff: October 1 00:57:58; hard compute cutoff: 01:27:58; absolute close: 01:57:58. The independent watchdog and per-worker process-group deadlines cover descendants. Existing operator/storage caps and zero new VJP/JVP apply.

Budget reconciliation retains distinct categories: 6064.481842329726 seconds prior conservative charge plus 7170.102459669113 seconds completed core actual wall. Neither the core's GPU-worker time nor the delay between its already-completed report and a later user follow-up is added as a second executing phase. Original 12h package remainder is 29965.41569800116 seconds; this run is capped at 28800 seconds.

Five new tests passed: telemetry preserves original SUP updates, no-update source probes leave parameters unchanged, exact control routing/endpoint sharing, paired report/embargo behavior, and the complete 32-context counterfactual schedule including 2496 forward calls. Prior source/target/recovery tests are inherited without re-running unrelated suites.

Both SUP methods passed one complete source-only 32-round schedule block. SUP_RET block took 110.122 seconds; SUP_STATIC 70.257 seconds. Retain the larger of old conservative profile and new timing ×1.3. SUP_RET writer gradient norms were nonzero; STATIC writer was correctly frozen. Five actual control paths passed source-only routing/count checks. Preflight total GPU-worker time was 210.033 seconds, 3130 forwards, 520 backwards, 128 optimizer updates and zero VJP; these are discarded diagnostic copies, not formal model training.

| Admitted work | Conservative seconds |
|---|---:|
| Five controls × two orders plus source counterfactual | 4299.332 |
| SUP_RET1024 plus fixed validation | 6518.203 |
| SUP_STATIC1024 plus fixed validation | 4289.929 |
| Six SUP target trajectories plus CPU scoring | 2360.446 |

Both complete training branches are admitted once, based only on costs. Formal order: fixed source counterfactual; five controls/two orders each; SUP_RET then SUP_STATIC source chains; SUP_RET, its same-checkpoint CONST_HALF, and SUP_STATIC on two orders each. Maximum 16 new trajectories, 16384 visits and 14208 principal-scored visits. Original eight short trajectories were verified for exact sealed identity, order and eligibility and are reused, not rerun.

Directed audit found the actual POST1024 actor in original GR/HALF deployments, shared actor identity and writer membership in all 12 saved optimizer parameter tensors. WARM→POST use/write L2 changes were 0.234612/0.236576; parameter change alone is not proof of useful learning. Original target writer standard deviation was 0.0002234/0.0002224, with max absolute deviation from .5 of 0.001529634. Missing original separate sampled-gate distributions and advantage norms remain explicitly NOT_RECORDED; new frozen-copy source probes are labeled separately and perform no optimizer update.

New target scores remain private until terminal run state. No performance-driven selection, new recipe, seed, WARM rerun, backbone/basis change, or automatic follow-on experiment. The queue produces REPORT plus ATTRIBUTION, WRITER_DIAGNOSTICS, SOURCE_COUNTERFACTUAL, COST_AND_STATUS and resource/score receipts. Final results still require collection after execution. Only local commits; no push.
