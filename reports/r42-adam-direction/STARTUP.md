# R42 startup, not scientific completion

Original T0: 2026-10-10T16:03:39.585395+00:00 (2026-10-11 00:03:39 Asia/Shanghai).
Source: 2ad0b369282457e95426d1f92fc8665781319cc5. Fixed primary CW_LSO_GDP; attribution CW_LSO_GP.
Four conditions C/W/CW_LSO_GP/CW_LSO_GDP, three seeds, two orders;
24 trajectories x 1,951 arrivals = 46,824 planned scores.

14 CPU checks passed, including a concrete Adam reference-dot reversal example,
disabled/first-step retained parity, unchanged moments after post-step projection,
retained cap and full snapshot continuation. All four GPU profiles passed and
whole-matrix admission passed at maximum two GPU workers on physical GPUs 0/1.
At the startup snapshot the first two C jobs reached 18/19 arrivals without GPU
failures. Commands were neutral. Other workloads were not interrupted.

Projected formal GPU-worker time (including profiling timing reserve and I/O
allowance) 17.069444 h, within 48h cap; online 22h/wall
24h/32GiB unchanged. Estimates are admission forecasts, not measured total cost.
Preceding six closed rounds cost 45.08708110955027 GPU-worker h.

One CPU-only preparation dispatch failed because the system Python had an
incomplete Torch installation. The configured runtime was used on the next
preparation attempt; all checks then passed. Failure dispatch elapsed estimate
2.350775s, GPU cost zero; measured successful CPU test time 16.776557s. Failed
preparation evidence is retained privately and disclosed here.

No scientific scores or success claim yet. SEARCH/REVIEW are historically exposed;
REVIEW descriptive only, independent generalization untested. The complete
protocol is in docs/protocols/R42_ADAM_DIRECTION.md. All frozen positive/negative
outcomes and costs will be delivered at actual completion.
