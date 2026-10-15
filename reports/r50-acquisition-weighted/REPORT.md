# R50 acquisition weighted structural filtering

Scientific results: **PENDING**. This is the frozen startup registration, not an efficacy result.

Four fixed arms: C, C_JS_SHARED, C_JS_ACQ_WEIGHTED, unchanged CW_LSO.
Three original seeds and two original orders:24 fresh trajectories,46824 independent CPU score rows.
Primary: C_JS_ACQ_WEIGHTED minus C; comparisons to the original C-background shared filter and CW_LSO remain separate.

Same-round four-condition GPU profiles and whole-matrix budget admission must pass before formal work.
Only physical GPU4/5/6/7 on jiangsuiyang; at most4 GPU-workers.22h online/24h wall/48GPU-worker h/32GiB.
All initialization, profiles and failures count. No automatic retry, tuning or subsequent round.
54 CPU checks passed; one Python3.10 report-syntax preparation failure was corrected before GPU work and retained in the audit.

SEARCH/REVIEW have prior exposure; REVIEW is descriptive. The isolated profile was a resource/derivative check, not Dice evidence.
Each positive comparison requires >=0.1pp, paired95% CI lower>0, both orders positive,>=5/6 trajectories positive,
imageweighted>=0 and worst seed-mean domain/channel/order cell>=-2pp.
A gain over C does not establish improvement over the original filter or strong control.

See the frozen R50 protocol in docs/protocols/R50_ACQUISITION_WEIGHTED.md.
The final delivery will include all positive/negative domain/channel/order/seed cells, intervals, mechanism diagnostics, cost and retirement/coverage audit.
No images, labels, content identities/hashes, per-content predictions/scores, model states or private paths/logs are public.
