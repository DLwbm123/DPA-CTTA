# R37 startup: C+W orientation transfer

Status: RUNNING_FIXED_MATRIX; scientific results pending. Authorized follow-up after complete, verified R36 delivery.

R36: all42 trajectories and81,942 CPU-scored arrivals complete, no GPU worker failures; 14.369614 GPU-worker hours and5.295474 wall hours to completion. No frozen module passed. W_LSO improved W by+0.119991pp on SEARCH, with6/6 positive trajectories, but remained-0.481921pp below C. W_LSM was negative in6/6 trajectories. R36 results delivery commit14b599ac96103d46fc1a55649e4baefb2920c3fb.

R37 fixed matrix: C, W, CW, CW_LSO; seeds20260907/17011/29009 and both original orders. Exactly24 complete1,951-arrival trajectories (46,824 formal arrivals). Frozen primaryCW_LSO. W and C exactly retain R36 definitions; CW moves the existing W weights to plain consistency with fixed1e-4 learning rate; CW_LSO adds unchanged R36 LSO coefficient0.1. No new source training, labels, RL, views or output ensemble. This is a bundled host transfer; CW isolates LSO's increment.

Scientific source commitdd4c290d85aa2961a8620bff74fa3babb1d3b706. T0 2026-10-10T01:38:48.711646+00:00 (09:38:48 China time). Per-round24h wall/22h online/48 GPU-worker hours/32GiB. Preserve R36 original T0, private receipts and all costs; cumulative cost begins with R36's14.369614 GPU-worker hours.

Two relevant CPU checks passed: unchanged controls and disabled-LSO C-host output/Adam/RNG/frozen-state parity; redaction and decision boundaries. CPU test cost4.352538s, no failed preparation attempt. All four GPU profiles passed (12 arrivals each), zero source/label reads; max reserved memory below1GiB. Entire24-trajectory queue admitted using measured max timing plus I/O/20% reserve, projected7.906391 GPU-worker hours. First three C workers have progressed to47–50 arrivals. Full commands and GPU process names are neutral.

Hourly monitoring continues in the same chat. At completion, retire all online workers before independent CPU mask scoring. SEARCH/legacy REVIEW are both historically exposed; only SEARCH guides development, REVIEW remains descriptive. Primary gate: SEARCH>=0.3pp against both C/W and>=0.1pp against CW, each with both orders positive, >=5/6 positive trajectories, nonnegative imageweighted gain and worst seed-mean cell>=-2pp. All negative cells/costs are delivered whether the gate passes or fails. No independent generalization claim.

No private data, masks, predictions, model states, private paths or third-party PDFs are included. Startup is distinct from execution/scoring and verified final delivery.
