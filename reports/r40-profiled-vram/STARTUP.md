# R40 startup: technical recovery using measured worker VRAM

R39 ended with8 completed unscored streams,1 pre-arrival W admission failure,
15 NOT_RUN and0 scores; no primary formal stream or REVIEW release. Its failure
report e38f0c93787ce29ededc28f274d057d333abcb2d is verified via remote SHA and
anonymous HTTP200. Scientific result NOT_EVALUABLE. Cost2.1003391753964955 GPUh;
closed4-round cumulative31.30581171128485 GPUh. No R39 state or ledger is reset.

R40 corrects the concrete startup fault: after same-round profiles, formal worker
memory admission reads the matching measured peak rather than inherited5GiB
fallback; it keeps max(1.2*peak,peak+512MiB) reserve. Cold profiles retain old
fallback; invalid/missing/mismatched profile refuses. This is a separate explicit
technical recovery, not an algorithm improvement or tuning from scored negatives.

R39 scientific matrix and method are unchanged: C/W/CW_LSO/CW_LSO_GC,3 seeds and
2 orders,24 fresh1,951-arrival streams. GC fixed radius0.0015/gradient EMA0.9 and
negative-cosine activation, W/LSO0.1/views/Adam state unchanged. No selective reuse,
new scientific candidate, label access, radius scan, RL or other job interference.

Nine CPU checks PASS, preparation10.074593305587769s/GPU0, no preparation failure.
Four12-arrival GPU profiles PASS; whole queue admitted with20% timing reserve,
projected18.316748 GPU-worker h under48h cap. Three first formal workers have
progress and no failed GPU receipt; neutral PS and nvidia display checked.
Other GPU workloads coexist, and profile timings/duration may vary substantially.

Original T0:2026-10-10T08:56:07.607640+00:00. Own22h online/24h wall/48GPUh/32GiB limits.
All new profiles/failures/formal jobs and old R39 costs retained, no budget reset.
Original C/W gates unchanged: primary gains>=0.3pp each, both orders positive,
>=5/6 positive trajectories, imageweighted>=0, worst seed-mean cell>=-2pp;
primary-minus-CW_LSO mean>=-0.05pp. Gate not yet evaluated; no primary substitution.
All workers retire before CPU scoring, all positive/negative results will be
published. Exposed SEARCH/REVIEW do not establish independent generalization.

Current status RUNNING_FIXED_MATRIX, results pending, hourly necessary monitoring
updated to R40. Runtime correction does not prove the scientific hypothesis.
