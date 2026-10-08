# Post-hoc SEARCH-only reward component audit

R26's widened action space increased candidate separation without improving the
primary method. This audit uses18,306 existing SEARCH candidate groups (RANDOM4,
GREEDY and PG_RETAIN trajectories); it reads saved scalar scores, not new labels or
images, and performs no training. Eight predefined components are all retained:
cons,structural,retain and the negative consistency,extent,topology,nesting,retention
penalties. No weight grid,sign search or REVIEW component fitting is performed.

For each four-candidate group, compare the selected64px macroOD/OC Dice to the
uniform random-candidate expectation. Equal scores average all tied maxima.
Summaries weight domain,seed and order equally. Files contain576 domain/seed/order
cells and24 summaries. The runnable script includes a check that a constant score
has exactly zero gain, while correct/reversed rankings have positive/negative gain.

| Candidate-trajectory family | Complete retain reward gain(pp) | Nesting-only gain(pp) | Nesting order0/order1(pp) | Positive trajectories |
|---|---:|---:|---:|---:|
| RANDOM4 | +0.001177 | +0.007809 | +0.004261/+0.011356 |5/6|
| GREEDY | -0.001521 | +0.008010 | +0.004788/+0.011232 |5/6|
| PG_RETAIN | -0.000952 | +0.007169 | +0.004338/+0.010000 |5/6|

Nesting is the most consistent small directional signal in this fixed component
set. Worst domain-mean effects are still negative (-0.004122,-0.005603,-0.005006pp).
The eight-way inspection and reused content are development selection; these are
neither independent tests nor multiplicity-adjusted significance claims.64px gains
cannot establish512px improvements or long-run state effects. A constant or empty
mask can satisfy nesting; bounded actions and the native update remain necessary
but do not guarantee correctness.

A justified follow-up is a fixed three-arm full-stream test (ANCHOR,RANDOM4,GREEDY)
with GREEDY selecting by negative nesting only. This isolates the reward change
before further policy learning. It must retain three seeds,both orders,all negative
outcomes and a frozen primary/gate. A positive result remains development evidence.
