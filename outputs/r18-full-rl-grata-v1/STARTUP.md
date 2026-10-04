# R18 startup: full RL use/write FiLM with GraTa

The user authorized this single finite combination experiment. The complete original RL controller, FiLM carrier and cross-image m/q/h memory are retained. Native GraTa updates the current image first; the controller acts on the adapted model. RL policies train on source only and freeze on target.

Execution code: `a0691319b00ba46dd652bd2b61d0849e98a09416`.
Frozen configuration: `467dfb8d3b207ed2281a14b221402273384995fa3e04cb6f988be5c46cdacd7c`.

Three CPU tests and real source exact-parity/continuation qualification passed. Measured 1.3 full-matrix admission passed: remaining source5859.522 GPU seconds, target18066.981 GPU seconds; qualification121.595 GPU seconds (1566 backbone forwards,288 backwards,140 optimizer steps). Formal source training is running on GPU4, with GPU4/5 assigned for target concurrency. No target scores have been read. This startup record is not a completed experiment or a performance result.

New full-stream conditions: RL_ORIGINAL, RL_GRATA, FIXED_GRATA (all actions fixed), WARM_GRATA (supervised warmup only). Each has two 1951-visit/1695-primary orders; new total15608 visits/13560 primary. Matching sealed pure GraTa and C0 references are reused. Two orders are the same images, not independent replication. All development data were previously exposed.

The source policy is re-fitted for the GraTa environment with fixed 1000 WARM updates and1024 GR_RET_EMA rounds. The fixed sixteen-episode source pool and held-BN retention probes are detailed in the protocol and differ from historical source training. GraTa remains action-independent. No target RL reward or policy fitting.

Budget T0 October4 2026 19:45 Beijing. Normal computation deadline October5 18:45; hard19:15, absolute19:45. Maximum12 GPU-worker hours and two concurrent workers in authorized GPU4–7 pool. Hourly monitoring is active and ends after this round's report is delivered. No automatic successor.

See ../../docs/protocols/R18_FULL_RL_GRATA_V1.md for frozen hypotheses, complete controls, admission, stopping rules and limitations.
