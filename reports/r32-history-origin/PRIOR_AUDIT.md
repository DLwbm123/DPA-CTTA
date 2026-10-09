# R30 position audit motivating R32

This is a post-hoc reanalysis of already scored R30 predictions; no new target labels, predictions or model fits were obtained. Script:src/dpa_ctta/r32_history_origin/audit.py. Both original streams contain contiguous domain blocks, with identical within-domain image order. Consequently the matched images can be contrasted by prior history, although the original prefixes differ in both length and composition.

R30 C_CONT-C_EPISODIC SEARCH domain means: REFUGE+0.621039pp, ORIGA+12.581827pp, REFUGE_Valid-4.444417pp, Drishti_GS+4.143703pp. Overall equal-domain mean+3.225538pp masks the stable REFUGE_Valid harm. Both channels lose in all six seed/order groups. These are differences from episodic adaptation, not from the source model: REFUGE_Valid SOURCE63.582622%,S_BATCH75.065433%,EPISODIC75.217371%,CONT70.772954%.

| REFUGE_Valid position | Order0 C_CONT-EPISODIC (pp) | Order1 (pp) |
|---|---:|---:|
| First64 images | -5.568614 | -0.439253 |
| First quarter | -5.116541 | -0.575271 |
| Second quarter | -5.737467 | -2.231979 |
| Third quarter | -6.497365 | -3.219781 |
| Fourth quarter | -6.884360 | -4.790004 |

Means average the two channels and three seeds on the SEARCH contents actually present in each fixed positional bin. Bins use all-arrival positions, including unscored context. FIRST64 overlaps the first quarter; these are not disjoint endpoints. Rows differ in image difficulty, so deterioration over position alone is not a causal estimate of harmful updates. Order0 arrives after1050 earlier images; order1 after101. The early contrast does not isolate previous-domain identity from history length or Adam age.

R32 uses matched64-update SAME/CROSS histories and UPDATE/HOLD futures on identical query tails to distinguish specified history-content effects from subsequent within-domain updating. Original full-history and zero-history references anchor this diagnostic. K=64 and target domains are development-informed choices, not a held-out discovery or a deployable rule. Query-tail scores must not be presented as full-stream performance. No LR sweep, larger backbone, new controller or RL is introduced.
