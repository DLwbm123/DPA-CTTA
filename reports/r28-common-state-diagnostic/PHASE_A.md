# R28 phase A: exact historical selector audit

One offline pass through R27 SEARCH private scalar records is complete. No new
model forward or online run was used. All aggregate cells and CPU wall cost are
included. Rates and gains below average equally over domain,seed and order; they
are not pooled-image frequencies. Metrics use stored64px hard candidate masks.

| State trajectory | All four rewards zero | Max tie | First argmax index0 | Actual argmax gain vs random (pp) | Uniform-tie gain (pp) | Oracle gain vs random (pp) |
|---|---:|---:|---:|---:|---:|---:|
| GREEDY |79.5588%|81.2710%|85.1326%|+0.007438|+0.004642|+0.079015|
| RANDOM4 |79.2169%|81.0956%|85.1406%|+0.005709|+0.007809|+0.076707|

RANDOM4's argmax columns describe counterfactual greedy selection on its stored
states, not the random action actually selected. GREEDY's stored index was checked
against exact argmax. Ties mean exact equality, without a tolerance sweep.

Rewards are frequently exactly tied, but actual tie handling does not remove the
small positive local aggregate signal. The result does not establish tie handling
as the cause of the negative full online outcome. Candidate oracle gains here are
relative to uniform random within four candidates at64px, NOT relative to ANCHOR
and NOT a closed-loop upper bound. R28 phase B addresses the missing512px common-
state comparison. All data remain exposed development material.
