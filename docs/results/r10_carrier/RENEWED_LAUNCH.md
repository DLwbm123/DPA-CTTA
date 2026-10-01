# R10_CARRIER_DECISION_3H_V1 — renewed running attempt

Status confirmed RUNNING at 2026-10-01 10:51:33 Asia/Shanghai: source paired validation completed 1 of 16 full episodes; all five controls had completed the first 32 visits. Eight fixed target trajectories remain NOT_RUN at this startup snapshot.

Execution SHA `1084fe1c9ffc030388fbe4c6ebecbd80b1bf64a5`; frozen renewed config digest `746c648a338cca94d48f57211068c43c4de3bc8bd825a9c2bdf653c51bf7d825`.

The source trace duplication was already fixed and is now verified on a complete real episode. Supervisor exclusion now uses a stable per-experiment identity; exact code/config remains checked in the separate authorization seal. The previous inactive config-specific owner remains untouched. The regression verifies mutual exclusion between different configuration revisions and clean acquisition after release. Eight affected checks passed in 0.586 seconds, including source trace, state/readout endpoints, frozen intervention identity, image-only target isolation and owned process-group cutoff. No shared R9 module was changed.

The human expressly requested repairs and an immediate start after the previous closed deadline. This renewed start uses only the unspent cumulative three-hour quota: previous charged wall 2067.405774 seconds, remaining renewed window 8732.594226 seconds. Original T0 (2026-09-30 22:16:06.144414), original deadlines, receipts, failed attempts and closed report are retained. The renewed accounting start is 2026-10-01 10:48:37.318589. Time between closed attempts is recorded separately from executed wall; no prior cost is discarded and no GPU-worker cost is added twice. See RENEWED_AUTHORIZATION.json.

Active normal-compute cutoff: 12:14:09; hard compute cutoff: 12:44:09; absolute exit: 13:14:09 on October 1. Source phase cutoff remains 11:18:37 and target phase is at most one hour after source completion. Independent watchdog and per-worker process groups cover descendants. At most one equivalent infrastructure target-job recovery remains, without extending deadlines.

GPU 5 worker, supervisor and watchdog are alive with neutral command lines. The GPU worker uses 1330 MiB at the startup snapshot. Actual source counters are 327 forwards, zero backward/optimizer/VJP calls. No immediate log error. Filesystem mount/write probe passed. Pre-existing zero parity is reused because this repair changes only supervision, telemetry and wall accounting; model, B recurrence, FiLM/readout, data, BN and scoring are unchanged.

Frozen matrix: five source controls × 16 episodes × 32 visits; four new target conditions × the same two 1024-arrival manifests. Old N/G/C0/B results remain paired reuse. New target scores remain embargoed until terminal state. Old failed receipts and anonymous tables are historical, not new outcomes. No new training, alpha search, seed, dataset, monitor or GitHub push.

The previous incomplete report is retained in REPORT.md with a historical-status banner. Final renewed results are pending; launching does not constitute completion.

## Completed renewed execution

Final status COMPLETE at 2026-10-01 11:36:39 Asia/Shanghai. See [REPORT.md](REPORT.md) and [RUN_RECEIPTS.json](RUN_RECEIPTS.json). New 8 trajectories and independent scoring are complete; prior failed attempts remain retained.
