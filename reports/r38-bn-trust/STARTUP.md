# R38 startup: fixed BN displacement trust module

Status: RUNNING_FIXED_MATRIX; scientific results pending. Authorized follow-up after complete R37 delivery52d402ed1894ab6eba86529e0fcffbd77822e241, verified remote commit and anonymous HTTP200.

R37 CW_LSO SEARCH gained+0.354066pp against C,+0.955978pp against W and+0.191136pp against CW, each6/6 trajectories positive. Full gate failed because worst seed-mean cell against W was-5.155135pp (order0 REFUGE_Valid OC). No adverse cell or failed guard was dropped. Mean BN displacement was0.003228 versus W0.001402; the update-size mechanism is a hypothesis, not causal proof.

R38 freezes one radius0.0015: after ordinary C-host Adam, scale the adaptive BN-affine displacement only if its global L2 norm exceeds that cap (within floating-point rounding). Ordinary Adam gradient moments/counters persist. Retain W weights, six views/current target, LSO0.1 and final forward after projection. No extra image query, backward, optimizer step, adaptive radius, new model, output ensemble or RL.

Four frozen conditions C/W/CW_LSO/CW_LSO_TR, three seeds20260907/17011/29009, both original orders, full1,951 arrivals each:24 trajectories/46,824 formal arrivals. Fresh state each job. PrimaryCW_LSO_TR. Exact prior controls and disabled-cap full-state/output parity passed; cap bound, finite final forward, frozen parameters, snapshot-next-step recovery, redaction and unchanged strong-control gate were checked. Five relevant tests passed, measured CPU6.674393s. One local supplemental-test dispatch had an incorrect relative path before SSH; corrected, recorded, CPU cost unmeasured, no GPU attempt.

All4 GPU profiles (12 arrivals each) passed and complete24-trajectory queue admitted. Projected8.322863 GPU-worker hours with measured max timings plus I/O/20% reserve. First3 C workers progressed to85–88 arrivals; no failed worker. Visible process commands and GPU process names were neutral.

Scientific sourceb5591ccc09ec5ce1eee561418456d5e253b57a67. Original T0 2026-10-10T04:56:14.587372+00:00 (12:56:14 China time),22h online/24h wall/48GPU-worker h/32GiB. Original completed-round T0/ledgers retained, prior cumulative GPU22.140055h; add all R38 profile/formal/failed costs. Hourly monitor continues in same chat.

Primary gate against both C/W remains>=0.3pp, both orders positive,>=5/6 positive trajectories, imageweighted gain>=0, worst seed-mean cell>=-2pp. Also preserve CW_LSO SEARCH mean within0.05pp; this extra attribution contrast is mean noninferiority, not required positive per trajectory. Report all of its negative cells anyway. Radius and gates cannot change inside this round.

All online workers retire before independent CPU mask scoring. SEARCH/REVIEW historically exposed; REVIEW descriptive only and not used to choose radius. No independent generalization claim. Deliver all positive/negative anonymous aggregates, cap activation/scales/raw displacement diagnostics, costs/audits/report via proxy and verify GitHub. No private images, labels, predictions, states, content/patient hashes, private paths/logs or third-party PDFs are public.
