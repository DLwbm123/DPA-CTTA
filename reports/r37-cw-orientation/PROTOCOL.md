# R37: fixed C+W orientation transfer

User authorization on 2026-10-09: autonomously analyze and improve unsuccessful
incremental module results. R36 is complete and delivered before this follow-up.
RL remains paused. This protocol is fixed before R37 execution and target labels.

## Hypothesis and evidence boundary

R36 SEARCH W_LSO-W +0.1200pp, 6/6 positive and both orders positive, but
W_LSO-C -0.4819pp: no R36 module passed the gate. W loses in ORIGA and
Drishti while gaining in REFUGE_Valid; LSO only mildly changes that tradeoff.
R20 C-host W transfer had a one-seed positive signal, not full confirmation.
Test whether unchanged W weighting and LSO work together on plain-consistency C.
The host transfer bundles removal of G's entropy/perturbation and dynamic learning
rate with fixed C learning rate; it is not isolated causal evidence for any part.
R36 REVIEW was descriptive and is not used to choose this design.

## Frozen matrix and implementation

Four conditions, exactly three seeds [20260907,17011,29009], both original
orders [0,1], full 1,951-image streams: 24 trajectories, 46,824 formal arrivals.
Fresh original checkpoint/BN/Adam/RNG state each trajectory, condition-outer order.

- C: exact R36 strong plain-consistency control, original fixed Adam LR1e-4.
- W: exact R36 W06_LR15 G-host control, six weak views, variance temperature0.1,
  boundary boost2 and native GraTa final learning-rate multiplier1.5.
- CW: same W pixel weights on native C, final LR multiplier1.0, fixed LR1e-4.
- CW_LSO (frozen primary): CW plus exactly the existing R36 LSO coefficient0.1.
  Detached current six-view target, pool4/smooth3/Sobel structure tensor,
  doubled-angle axial orientation; target anisotropy times W reliability weights
  orientation loss. Independent nested sigmoid channels, no magnitude term.

Reuse R36 Host and guarded runtime; no new optimizer, backbone, teacher,
memory, source images, source training, RL, extra weak views, pseudo-label editing,
output ensemble, topology/balance/magnitude modules or coefficient search.
No pruning, extra seeds or automatic scientific retry. All four conditions finish.

## Evaluation and decision

All online workers retire before independent CPU scoring reads any masks.
SEARCH1017 and legacy REVIEW678 identities unchanged and historically exposed.
SEARCH guided this follow-up; neither role is independent validation.
Hard Dice2TP/(prediction+GT), both empty1; equal domain×OD/OC then seed/order.
Report every domain/channel/seed/order, imageweighted score, empty predictions,
containment, paired content-cluster bootstrap2000 and all negative cells.
Patient linkage UNKNOWN; no ASSD/Brier/soft Dice in this bounded round.

Stage success: CW_LSO-C>=0.3pp and CW_LSO-W>=0.3pp, plus CW_LSO-CW>=0.1pp
to attribute a material increment to LSO. Each SEARCH comparison requires both
orders positive, >=5/6 trajectories positive, imageweighted gain>=0 and worst
seed-averaged domain/channel/order cell>=-2pp. Fixed primary only; no retrospective
replacement by CW. Report CW comparisons as an ablation even if primary fails.
These are development investment thresholds, not clinical/statistical guarantees.

## Execution, cost and delivery

CPU tests: exact unchanged controls and disabled-LSO C-host output/parameter/Adam/RNG
parity, frozen parameters, public receipt redaction and decision boundaries.
Existing R36 structure gradient/flat-target/channel tests remain applicable because
their implementation is unchanged. All four real GPU profiles12 arrivals, all charged.
Admit all24 full trajectories from measured max time plus I/O20% reserve; if they
cannot fit, stop BUDGET_BLOCKED rather than choose a score-driven subset.

Three workers maximum, one per GPU after free-memory check. Original per-round T0
before profiles, online22h, total24h, GPU-worker48h, disk32GiB. Preserve all failures
and preparation costs. R36 original T0, budgets and receipts are never overwritten;
cumulative multi-round GPU cost includes R36's14.369614h plus all R37 attempts.
Use neutral visible entrypoints/arguments and prescribed mounted storage.

Hourly necessary monitoring, no restart of healthy workers. At completion, deliver
all positive and negative anonymous aggregates plus own code/report to DPA-CTTA
branch experiment/r37-cw-orientation-v1. Explicit local proxy for every GitHub
operation; verify remote commit and anonymous report access. Update DELIVERY.
No private images, masks, predictions, model states, content/patient IDs, private
paths, logs or third-party PDFs. Failed gate permits only a new evidence-driven,
pre-registered bounded hypothesis under the user's continuation authorization.
