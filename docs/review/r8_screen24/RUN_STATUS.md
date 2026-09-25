# Screen24 launch status

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
