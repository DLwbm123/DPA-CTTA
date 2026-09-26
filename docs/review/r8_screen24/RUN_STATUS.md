# Screen24 launch status

## Latest: resumed with user-authorized soft timing

2026-09-26 02:49:59 UTC resumed the same run, queue PID 1044358; actual runtime SHA 544640fe51507509a97864fe1a58c8eb0275d3d0, scientific artifact identity remains 2329fb6. The user explicitly removed timing gates. Estimated worker duration and 24-hour wall target no longer stop the queue. Real aggregate caps and scientific validation remain. Scaler recovery passed its anchor-493 boundary and advanced to 495, GPU 5 PID 1044369; old costs and completed artifacts retained. The earlier awaiting-approval/stopped entries below are historical. See TIMING_OVERRIDE.md.


## Latest: stopped at scaler worker deadline

2026-09-25 15:50 UTC monitor confirmed GLOBAL STOP at 493/512 scaler anchors. Oracle/basis/capacity complete; source training and target work not started. Both scaler snapshots validate. No automatic cap override or restart. See SCALER_DEADLINE_REVIEW.md for a fully budgeted proposed continuation requiring the explicit resource-stop exception. Historical launch evidence follows.


- State: **RUNNING**, observed 2026-09-25 12:03 UTC. This is launch verification, not completed results.
- Effective run: `screen24-20260925T120130Z`; launched 2026-09-25 12:02:08 UTC (20:02:08 Asia/Shanghai).
- Runtime code: `2329fb625ee023f461c23d1256162eb5514906c6`; later documentation commits do not change the deployed runtime.
- Queue PID: 722210. Initial workers: 722244 / 722245 / 722246 on physical GPUs 5 / 6 / 7, verified against registered GPU UUIDs.
- Neutral commands: `/tmp/e_8a36/bin/python /tmp/q_23.py` and `/tmp/e_8a36/bin/python /tmp/w_23.py`.
- Initial live evidence: all three workers exceeded 150 backward/optimizer steps; physical journals were growing, queue RUNNING, no worker log errors.
- Scope: 10 source jobs, 40 target trajectories, 78,040 arrivals, 67,800 principal scored records. Original full R8 is not running.
- Validation: 54 original tests, 3 Screen24 tests, 30 concurrent ledger transactions, same-code GPU resource profiles, eight source recovery comparisons, two R7 control recovery comparisons, and real serial/partition oracle comparison passed.
- Projection after repair: 19.23 hours including 1.3 measurement margin and 2 hours reporting reserve. Aggregate admission including failed startup reservations: 57.92 GPU-hours and 9.25 GiB. Limits: 24 hours wall, 60 GPU-hours, 16 GiB. These are estimates, not completion guarantees.
- Recovery: see LOCK_REPAIR.md. Three oracle jobs have already used their allowed infrastructure recovery.
- Hourly monitoring: automation `r8-b-a`, ACTIVE. Quiet unless a stage completes, failure occurs, a new decision is required, or final delivery completes.

Remote private evidence is under the dedicated R8 root: `private/launch-receipt-2329fb6.json`, `private/launch-2329fb6.json`, `private/admission-2329fb6.json`, and `private/recovery-screen24-20260925T120130Z`. Runtime queue/ledger/logs are under `runs/screen24-20260925T120130Z`; source/target artifacts use the same run identifier in their respective output roots. The queue log is `/tmp/o_23q.log`.

At completion, verify the entire reduced coverage, report incomplete/failed jobs without substitution, publish permitted source and aggregate results to the project GitHub repository, verify remote SHA and anonymous access, then pause monitoring.
