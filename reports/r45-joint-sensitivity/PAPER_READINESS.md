# Paper direction after the complete R45 result

Status: evidence review and **proposal only**. No new experiment, dataset access,
parameter tuning, GPU/CPU scientific task or queue was started. All figures below
come from completed anonymous reports; the existing prediction-analysis proposal
has not been executed. Review of publication novelty is incomplete.

## Decision

The minimal joint acquisition/anatomical sensitivity implementation is not an
accuracy-improving contribution in this development matrix. Do not lead the
paper with anatomical protection through joint sensitivity, general safe updates,
RL, GC/GP, SCNP or spectral entropy. Their engineering validity or gains over weak
controls do not establish incremental efficacy over CW_LSO.

The defensible candidate direction is **reliability-weighted structural orientation
consistency for continual medical segmentation adaptation**, centered on unchanged
CW_LSO. Its scientific question is whether stable pixel reliability and predicted
structure orientation give complementary adaptation evidence, and where average
benefit trades off against local domain/channel damage. This is a candidate paper
argument, not a verified novel algorithm or robust/clinical result. A new name or
objective notation does not establish novelty. The existing paper literature and
compatibility with prior anatomy TTA still require a precise comparison before a
novelty claim. A standard efficacy/algorithm paper needs the confirmation below;
a broader diagnostic paper would need broader coverage rather than this one setup.

## Established evidence and boundaries

R37 SEARCH macro Dice: C78.691553,W78.089641,CW78.854483,CW_LSO79.045619.
The sequential increments are CW-minus-C+0.162930pp and LSO-minus-CW+0.191136pp.
CW_LSO-minus-C+0.354066pp, content-cluster interval[+0.303809,+0.402994]pp;
LSO-minus-CW interval[+0.159125,+0.225917]pp. Each of the latter comparisons has
both orders and6/6 trajectories positive. This is the strongest current component
benefit. The C-host transfer from the original W host is a bundle of LR and other
native host differences; CW versus W is not an isolated weighting ablation.

CW_LSO-minus-W is+0.955978pp on average but its worst seed-mean cell is-5.155135pp
(order0 REFUGE_Valid OC); REFUGE_Valid OD-4.908927pp and order1 REFUGE OC-3.528396pp
remain adverse. Thus average improvement is supported while uniformly safe
adaptation is not. The registered -2pp guard failed. Development thresholds are
not publication or clinical significance thresholds.

| Round/module | Paired incremental comparison | SEARCH delta pp | Interpretation |
|---|---|---:|---|
| R36 W_LSO | W_LSO-W | +0.119991 | Helps original W, still below C by0.481921pp |
| R36 W_LSM | W_LSM-W | -0.170578 | Negative; orientation and magnitude not interchangeable |
| R37 LSO | CW_LSO-CW | +0.191136 | Positive component evidence; interaction with W still untested |
| R38 TR | TR-CW_LSO | -1.288808 | Nearly universal cap loses useful adaptation |
| R39 GC | not evaluated | NA | Resource interruption with0 scores; not a scientific negative |
| R40 GC | GC-CW_LSO | -0.015707 | Worst W cell improves to-3.638530pp, still fails guard |
| R41 GP | GP-GC | +0.006058 | Interval includes0; no persuasive increment |
| R42 GDP | GDP-GP | 0 | Actual Adam direction projection never triggered |
| R43 SCNP | SCNP-CW_LSO | +0.005466 | Interval[-0.001187,+0.013486]pp;4/6 positive |
| R44 SE | SE-CW_LSO | +0.000715 | Interval[-0.000837,+0.002422]pp;3/6 positive |
| R45 JS | JS-CW_LSO | -0.006898 | Interval[-0.007949,-0.005819]pp;0/6 positive |

Rows use their own fixed paired controls and do not form a pooled independent
meta-analysis. Repeated identical baselines on the same contents/seed orders
mostly confirm reproducibility, not independent generalization.

R45 JS reaches79.038721 SEARCH macro Dice. Gains versus C/W are+0.347168/+0.949080pp,
retaining CW_LSO gains; both JS-minus-CW_LSO orders are negative, imageweighted
-0.006179pp. Worst W cell-5.138166pp improves only0.016969pp from paired CW_LSO,
not meaningful protection under the frozen guard. All6,102 SEARCH observations
have rank2, no metric contraction violations; mean filtered/proposal norm ratio
0.998812 and median0.999901. These establish active implementation with small
whole-update norm attenuation, not the cause of failure. The short isolated
profile's1.44-1.50x latency and+38.91% allocated/+43.59% reserved memory have no
accuracy benefit in R45 to offset them.

Earlier alternatives explain route closure without proving general impossibility:
R20 W exceeded native GraTa on then-campaign REVIEW but lost to C; R25 PG_RETAIN
78.2248 versus ANCHOR78.2244 REVIEW gives essentially no incremental policy gain;
R27 GREEDY-minus-ANCHOR-0.025883pp; R28 same-state four-candidate oracle headroom
+0.080122pp; R29 FULL/ZERO/HALF oracle headroom+0.010225pp; R35 HOLD256-minus-UPDATE
-0.765089pp with oracle-minus-best-fixed+0.234880pp. Immediate state probes and
prospective windows are distinct endpoints, not upper bounds on every RL policy.

All SEARCH/legacy REVIEW contents are historically exposed. Patient linkage and
ROI provenance remain unresolved. Three seeds and two orders do not create six
independent cohorts; bootstrap couples repeated content and does not resolve
patient dependence or adaptive development selection. Current primary evidence
is one retinal OD/OC setup with the retained model. There is no general medical
segmentation, clinical calibration, anatomical-preservation or SOTA proof.

## Minimum supplements, in decision order

These are conditional proposals, not registered or approved runs. Avoid adding
all variants, many tasks, models or parameter grids at once. Fixed parameters
would remain unchanged; publication criteria must be frozen before new data.

### P0: structure endpoints from already completed predictions (zero new GPU work)

Question: does the established CW_LSO Dice increment have corresponding boundary
quality, rather than only overlap improvement? Use existing C/CW/CW_LSO predictions
and authorized masks; retain W for adverse-cell context. Score one boundary metric
(ASSD is sufficient for this first check), empty-prediction rates and OC-outside-OD
violations by domain/channel/order, alongside existing Dice. Specify undefined
empty-surface distances and report their counts; never silently remove empty cases
or assign zero distance. This is a new CPU analysis proposal, not already done.

Primary contrast CW_LSO-minus-CW, secondary versus C; inspect the existing adverse
cells rather than selecting favorable images. Improvement in overlap without
boundary benefit supports an overlap claim only. Do not infer clinical anatomy
from low nesting violations. No new online trajectories or coefficient tuning.

### P1: fill the single missing fixed loss ablation

Question: is W reliability weighting complementary to LSO, or does LSO help plain
consistency equally? Add **C+LSO**, fixed LSO0.1 with the same C host, source model,
views, LR, seeds and orders. Complete the2x2 design C/CW/C+LSO/CW_LSO. The key
interaction is

    (CW_LSO-CW) - (C+LSO-C).

At least one new condition x3 seeds x2 orders =6 trajectories. Existing controls
may be reused only if original model/data/order/seed/preprocessing/code and
optimizer bindings are compatible and the reused outcomes are prespecified; if
not, rerun the four conditions together (24 trajectories). Do not choose a
favorable old control or average duplicate round baselines. Report the interaction
interval, both orders and all adverse cells; missing interaction positivity means
no synergy claim. The existing positive LSO increment can still support a simpler
structural-consistency claim. No JS-only, anatomy-only, radius or strength scan is
needed for this CW_LSO paper decision.

### P2: one untouched confirmation setting with one fair recent baseline

Question: does the frozen CW_LSO effect survive independent content and exceed a
credible competitor? After P0/P1, freeze unchanged C/W/CW/CW_LSO and **one** recent
medical TTA baseline that can use the same model/checkpoint, data access and update
budget. The precise external baseline requires a separate literature and code
compatibility check; this report has not established a name or a runnable baseline.
Five conditions x3 seeds x2 preregistered stream orders =30 fresh trajectories on
one previously unused, patient-auditable retinal OD/OC cohort or held-out center.
Select the cohort by provenance/availability before seeing results. Do not relabel
old REVIEW or an exposed random split as independent; first resolve patient/ROI
provenance and disjointness. New data access is not authorized by this proposal.

This one matrix contains the strong-simple C and W controls, the CW attribution
control and one relevant external comparator, avoiding separate ablation and
comparison sweeps. Freeze macro Dice as primary, the existing mean/order/seed and
worst-cell protection criteria as development gates, and the P0 structure metrics
as secondary. Require positive CW_LSO-minus-CW and superiority to the strong
simple controls for a component-efficacy claim; assess the external comparator
without moving thresholds after results. A failed -2pp guard precludes a robust
protection claim even with improved means. No parameter change based on the new
cohort. Whole-matrix profiles, budget and authorization must precede execution.

If independent confirmation fails, stop the positive algorithm storyline rather
than expanding JS/SCNP/SE searches to rescue it. A new mechanism would require a
separate evidence-supported hypothesis and explicit authorization. A second task
or backbone is optional for the narrow retinal claim and necessary only if the
paper later claims broader transfer; it is not part of this minimum first package.

## Writing scope now

Can draft: the CW_LSO objective/implementation, complete development evidence,
strong baseline lesson, average/local-damage tradeoff, and clearly scoped negative
interventions. Cannot write: successful joint-sensitivity protection, every module
improves performance, independent validation, broad medical generalization or a
confirmed novel/SOTA method. JS belongs in the mechanism/negative-results section,
not the positive headline. Final paper direction should depend on P0-P2 rather
than aesthetic complexity of the new update formula.

## Source reports

- [R20](../r20-model-only-search/REPORT.md), [R25](../r25-parameter-policy/REPORT.md),
  [R27](../r27-nesting-selection/REPORT.md), [R28](../r28-common-state-diagnostic/REPORT.md),
  [R29](../r29-update-strength-diagnostic/REPORT.md), [R35](../r35-prospective-hold/REPORT.md).
- [R36](../r36-incremental-modules/REPORT.md), [R37](../r37-cw-orientation/REPORT.md),
  [R38](../r38-bn-trust/REPORT.md), [R39](../r39-conflict-trust/REPORT.md),
  [R40](../r40-profiled-vram/REPORT.md), [R41](../r41-conflict-projection/REPORT.md),
  [R42](../r42-adam-direction/REPORT.md), [R43](../r43-scnp/REPORT.md).
- [R44 complete report](https://github.com/DLwbm123/DPA-CTTA/blob/f5d394a427d2599cf1ac494a703e19abbe51d7a7/reports/r44-spectral-entropy/REPORT.md).
- [R45](REPORT.md), [R45 mechanism](JOINT_MECHANISM_SUMMARY.csv),
  [isolated resource profile](../joint-sensitivity-profile/REPORT.md).
