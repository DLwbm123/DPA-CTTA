# R48: repaired descent-constrained shared sensitivity

Status: **COMPLETE**. Frozen primary: CW_LSO_JS_SHARED_DESCENT.

Retain the R47 descent constraint on R46 SHARED and add only the CPU/GPU-qualified representable-commit repair. Original tolerance1e-5*max(abs(proposal_dot),1e-10) is unchanged; violating filtered commits restore exact native Adam and remain audited. Same rank4 span/Sensor/probes/four VJPs/strength1/ridge0.01/Adam. All30 trajectories start fresh; no prior-control reuse. Proxy loss descent is not anatomy or Dice protection.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_JS_SHARED | 79.0205 | 78.3859 |
| CW_LSO_JS_SHARED_DESCENT | 79.0463 | 78.4081 |

Primary SEARCH versus C/W each>=0.3pp and versus CW_LSO/R46 SHARED each>=0.1pp. Every comparison requires both orders positive,>=5/6 trajectories positive,imageweighted>=0,worst seed-mean cell>=-2pp. Frozen primary passed: False. All negative cells remain reported.

Completed analysis: [ANALYSIS.md](ANALYSIS.md). DESCENT improved SEARCH macro Dice
over SHARED by+0.025786pp (95% content-cluster CI+0.015596 to+0.034517;6/6
trajectories positive), but its image-weighted increment was-0.009164pp. The
increment over CW_LSO was+0.000650pp (CI-0.015621 to+0.018798;3/6 positive),
which does not establish an average gain. Drishti_GS/ORIGA costs were reduced,
while some REFUGE_Valid benefits were lost. Worst W-relative cell remains-4.704769pp,
better than CW_LSO but worse than SHARED's minimum. All frozen gate failures are
retained in GATE_AUDIT.json. No next experiment is started.

Independent terminal verification confirms30 complete unique jobs,58,530 scored
rows, all processes retired and all65 attempts charged. Original descent tolerance
held for all11,706 primary updates;0 final violations,1,182 commit repairs and234
native non-descent fallbacks. Anonymous full distributions and all paired positive/
negative seed-mean cells accompany the report. This is numerical execution
evidence, not anatomical or Dice protection.

GPU-worker 9.415/48h; CPU-worker 0.198h. Original per-round T0 and attempts retained.

Exactly five frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, output fusion or scientific retries. Only the registered two acquisition probes are added. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.
