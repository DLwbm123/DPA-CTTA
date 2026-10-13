# R48 completed analysis

The repaired descent constraint recovered the small SEARCH macro loss of R46
SHARED, but did not establish an incremental average gain over CW_LSO. It retained
some REFUGE_Valid improvements while reducing Drishti_GS/ORIGA costs. All frozen
development criteria remain unchanged; the primary **failed the full gate**.
No automatic next experiment is authorized.

## Complete paired effects

All values are Dice percentage points. Intervals are the registered 2,000-repeat
content-cluster bootstrap, coupling orders and seeds within content. Patient
linkage is unknown, and these are historically exposed development data.

| SEARCH comparison | Macro delta | 95% CI | Orders 0 / 1 | Positive trajectories | Image-weighted delta | Worst seed-mean cell |
|---|---:|---|---|---:|---:|---:|
| DESCENT − C | +0.354716 | [+0.314104, +0.394693] | +0.148437 / +0.560994 | 6/6 | +0.537437 | −0.481492 |
| DESCENT − W | +0.956628 | [+0.739786, +1.177937] | +1.228438 / +0.684817 | 6/6 | +0.387986 | −4.704769 |
| DESCENT − CW_LSO | +0.000650 | [−0.015621, +0.018798] | −0.040674 / +0.041973 | 3/6 | +0.079232 | −0.598777 |
| DESCENT − SHARED | +0.025786 | [+0.015596, +0.034517] | +0.028442 / +0.023130 | 6/6 | −0.009164 | −0.236368 |

SEARCH macro Dice is 79.046269% for DESCENT, 79.045619% for CW_LSO,
and 79.020483% for SHARED. The small resolved improvement over SHARED does not
meet the frozen +0.1pp increment and has a negative image-weighted delta. Versus
CW_LSO the CI includes zero, order0 is negative, and only3/6 trajectories improve.
Versus W the worst-cell requirement fails. `GATE_AUDIT.json` records each test;
`PAIRED_SUMMARY.csv` retains all six seed/order deltas and all intervals.

REVIEW, descriptive only: DESCENT − CW_LSO is −0.004937pp
[−0.023001, +0.016414]; DESCENT − SHARED is +0.022106pp
[+0.010089, +0.032270], with image-weighted −0.020222pp. It is not an
independent confirmation and was not used to change the method or selection.

## Local benefits and remaining costs

| SEARCH domain, average over channels/orders/seeds | SHARED − CW_LSO | DESCENT − CW_LSO | DESCENT − SHARED |
|---|---:|---:|---:|
| REFUGE | +0.124448 | +0.087048 | −0.037400 |
| REFUGE_Valid | +0.380484 | +0.250400 | −0.130084 |
| Drishti_GS | −0.333621 | −0.213441 | +0.120181 |
| ORIGA | −0.271855 | −0.121408 | +0.150447 |

REFUGE_Valid order0 OD/OC remain +0.244001/+0.450366pp above CW_LSO,
but lose −0.195838/−0.236368pp relative to SHARED. Order1 OD/OC remain
+0.157319/+0.149913pp above CW_LSO. The improvement is partly retained,
not preserved without cost.

Drishti_GS order0 OC, the worst incremental CW_LSO cell, improves from
−0.861706pp with SHARED to −0.598777pp with DESCENT. Its OD loss shrinks
from −0.330793 to −0.196964pp. ORIGA order0 OD/OC losses shrink from
−0.196572/−0.310961 to −0.043056/−0.200597pp; order1 OD/OC from
−0.216517/−0.363370 to −0.061776/−0.180201pp. These remain negative.
REFUGE order1 OC retains +0.307147pp versus CW_LSO, less than SHARED's +0.454210pp.

The worst W-relative cell is REFUGE_Valid order0 OC at −4.704769pp.
This is +0.450366pp better than CW_LSO's worst −5.155135pp but −0.235681pp
worse than SHARED's minimum −4.469088pp, whose minimum occurred in OD.
Three W-relative cells remain below −2pp: REFUGE_Valid order0 OD/OC
(−4.664926/−4.704769) and REFUGE order1 OC (−3.221248).
The minimum improved versus CW_LSO but deteriorated versus SHARED.
`ALL_PAIRED_SEED_MEAN_CELLS.csv` includes every positive and negative cell;
`ALL_NEGATIVE_CELLS.csv` also retains individual-seed negative cells.
Do not substitute sample-weighted Dice for the frozen equal-domain macro metric.
Domain order and accumulated adaptation confound domain-specific causal claims.

## Numerical and mechanism evidence

The complete primary arm had11,706 updates,11,472 negative native proposals,
1,182 representable-commit repairs (10.0974%), and234 native non-descent
fallbacks (1.9990%). All roles together had **zero final guard violations** at
the original tolerance. Every repaired commit restored the actual native Adam
parameters, with final derivative exactly equal to the recorded proposal.

SEARCH includes6,102 updates:5,083 constraints triggered (83.3006% of all updates;
84.9432% of5,984 negative proposals),616 commit repairs (10.0951%), and118 native
non-descent fallbacks (1.9338%). Native non-descent proposals are expressly retained;
positive derivatives in that group are not guard violations. No degenerate-constraint
fallback was observed. Final positive gap/tolerance maximum is0.999132 in SEARCH
and below1 in every role. Repairs had candidate gap/tolerance1.000050–122.174569;
those violations were repaired, not hidden or accepted by increasing tolerance.

SEARCH mean native derivative is −0.0004581182; final −0.0004590187.
`JOINT_NUMERICAL_SUMMARY.json` supplies distributions of the native, returned-filter,
candidate-committed, reconstructed unconstrained and final derivatives, multiplier,
gaps, tolerance ratios, fallback reasons, ranks, filter spectra, capture, span energy,
structure evidence and BN/region gradients. It also splits SEARCH by domain/order.
Fields absent on rank0/native fallback steps have explicitly smaller counts.

SHARED has rank4 throughout SEARCH; DESCENT has rank4 on5,984 steps and rank0
on118 native non-descent steps. Capture near1 is expected from the constructed
span and is not evidence of clinical anatomy capture. Candidate metric-norm ratio
averages1.103794 for DESCENT versus0.978109 for SHARED; the constrained candidate
can enlarge the update rather than contract it. These candidate norms and K spectra
are pre-repair quantities; they do not describe every final committed update.
Actual Euclidean BN update norm means are0.003730745 and0.003256125 respectively.
SEARCH OD/OC structure-evidence means are0.009530874/0.006909572 for DESCENT.
These observations show the mechanism acted, not that its proxies preserve true
anatomy. First-order pseudo-loss descent does not guarantee finite-step loss or Dice.
The observed benefit cannot be separated into constraint and repair contributions
because unrepaired R47 failed before scoring; no causal attribution is claimed.

## Execution, resources and limitations

All30 fresh trajectories completed1,951 arrivals. All30 independent CPU scorers
covered58,530 rows, after all online workers retired. All65 attempts (5 profiles,
30 online,30 score) completed with no retry; watch/supervisor/scorers retired.
`COMPLETION_VERIFICATION.json` independently checks receipts, unique job/visit
coverage, retirement, ordering barrier, calls, original tolerance and charged costs.

Both joint arms use10 forwards,1 base backward plus4 VJPs, and1 Adam step per image;
CW_LSO uses8 forwards/1 backward/1 Adam. Native W separately retains9 forwards/
2 backwards/1 Adam. These counts are verified against all formal receipts.
Formal average GPU residence per image: CW_LSO0.530171s, SHARED0.735670s,
DESCENT0.739281s. DESCENT is approximately+0.491% versus SHARED and+39.44%
versus CW_LSO in this shared-server run, not an isolated hardware benchmark.
Same-round and earlier bounded profiles qualify memory admission; no new memory
benchmark was run after completion.

R48 cost is9.415259016354879 GPU-worker h, including100.749329 profile seconds;
CPU scoring711.097096s, server preparation24.812832s, local CPU tests4.685s,
GPU preparation0. Original T0 and budgets were not reset. All13 closed/stopped
rounds total108.22563222454656 GPU-worker h; including the three separately
accounted resource profiles,108.41145762470036h. Old ledgers remain unchanged.

SEARCH/REVIEW were historically exposed; patient linkage is unknown. No untouched
cohort, ASSD, Brier or soft Dice was added. This round supports a small correction
of the shared-space tradeoff, not a robust incremental Dice gain over CW_LSO,
anatomical protection, clinical effectiveness, or a generalization claim.
