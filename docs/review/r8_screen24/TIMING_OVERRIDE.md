# Authorized timing policy amendment — 2026-09-26

User directive: “不要管什么单任务时限，不要因为这些 gate 停止实验”. This authorizes resuming the scaler resource stop and treating estimated per-job time and the 24-hour wall target as soft estimates, not stop conditions. The scientific 10-source/40-target scope is unchanged. Aggregate 60 GPU-hour / 16 GiB / physical-operation caps, GPU 5/6/7, identity, label isolation and numerical validation remain enforced.

Per-attempt GPU-time reservations extend atomically as measured time approaches their estimate, provided the aggregate cap still permits it. No estimate-only worker watchdog or queue wall stop is active in this explicitly enabled mode. Operations and retained costs are still recorded; a stopped attempt is not refunded.

The scaler resumes from the fully validated anchor-493 snapshot with its 62-forward uncommitted tail retained and replayed under an explicit USER_AUTHORIZED_RESOURCE_RESUME classification. This is not relabeled infrastructure. Completed oracle, basis and capacity are reused.

Provenance distinguishes the original scientific artifact identity (code_sha 2329fb625ee023f461c23d1256162eb5514906c6) from the amended scheduler/runtime (runtime_code_sha, bound to its own complete source inventory). Every new worker config and physical receipt records the runtime SHA. This avoids rewriting existing scientific checkpoints or falsely claiming the runtime was unchanged. The admission measurements remain estimates from the scientific base implementation; no repeated GPU profile gate is required for these control-only changes. Original stop/config/queue/ledger evidence is archived before resume.
