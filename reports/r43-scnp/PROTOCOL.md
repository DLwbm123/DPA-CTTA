# R43: same-class neighbor loss on retained CW_LSO

User authorized this round on 2026-10-11: retain C/W/CW_LSO, add SCNP,
freeze before running, use only physical GPUs 0/1, analyze and deliver all results.
Seven preceding closed rounds cost 52.744737940364416 GPU-worker hours.

## Evidence and single hypothesis

CW_LSO historically gains +0.354066/+0.955978pp against C/W on SEARCH, all six
trajectories positive, but worst W seed-mean cell -5.155135pp fails robustness.
GC/GP do not provide a successful new segmentation objective. CVPR 2026 SCNP,
"Towards High-Quality Image Segmentation: Improving Topology Accuracy by
Penalizing Neighbor Pixels", arXiv 2603.18671, uses ground-truth same-class
neighbors to route segmentation gradients through their worst logits.

Hypothesis: its neighborhood gradient allocation complements LSO orientation
consistency under weak-view pseudo supervision. This is our unlabeled adaptation,
not a supervised reproduction or a proven topology guarantee. Incorrect pseudo
membership may propagate errors; soft pseudo targets may damp extreme logits.
No guarantee of improvement or independent generalization is assumed.

## Frozen module and matrix

Conditions C/W/CW_LSO/CW_LSO_SCNP; seeds [20260907,17011,29009]; two retained
original orders, 1951 arrivals each: 24 new trajectories, 46824 CPU scores.
C/W/CW_LSO are literally the R37 candidates. All start from original checkpoint;
no reused trajectories or terminal state. CW_LSO_SCNP is primary, CW_LSO the
attribution control. Preserve W variance temperature 0.1, boundary boost 2,
CW fixed LR and multiplier 1, LSO coefficient 0.1, six aligned weak views,
BN-affine-only update, Adam settings/counts, preprocessing, stream and evaluator.
No GC, GP, GDP, entropy addition, source retraining or RL in the CW candidates.
W retains its original G host/multiplier 1.5; C its plain host.

q is the existing detached six-view mean sigmoid target. Independently per OD/OC
channel, let Y=(q>=0.5), including the center in a 3x3 neighborhood. For Y_i=1,
z_tilde_i=min{z_j: j in neighborhood(i),Y_j=1}; for Y_i=0 use max over Y_j=0.
Use replicate border extension, no channel mixing. Masked pools exclude the other
class exactly; center membership guarantees a nonempty selected pool.

L=L_CW+0.1 L_LSO+0.1 mean(w * BCEWithLogits(z_tilde,q)), where w is the unchanged
detached W reliability weight. Original logits retain base/LSO losses and inference;
SCNP is an additional loss branch, not output postprocessing. No hard pseudo-target
BCE, confidence cutoff, anatomy constraint, extra state, forward/backward/optimizer
step, coefficient/kernel scan, label/domain-dependent action or external model.

## Frozen evaluation and diagnostics

Same historical SEARCH/REVIEW split, hard Dice, both-empty=1, equal domain/channel
then seed/order means. SEARCH is development; REVIEW descriptive only, never
future tuning. Primary SEARCH gain against each C/W >=0.3pp; additionally gain
against CW_LSO >=0.1pp. Each of these three comparisons requires both orders
positive, >=5/6 trajectories positive, imageweighted gain >=0, and worst seed-mean
domain/channel/order cell >=-2pp. Do not replace primary or relax any threshold.
Report mean gain and all negative cells even if the complete stage gate fails.
Coupled content bootstrap 2000; patient linkage UNKNOWN; no new evaluation set,
ASSD/Brier/soft Dice or clinical claim.

Mechanics: exact small-neighborhood value/gradient oracle including mixed channels,
isolated pixels/borders/ties/uniform masks, detached pseudo targets, channel
symmetry, disabled full-state/output parity, active finite gradients, snapshot
continuation and frozen parameters. Retain C/W/LSO gates and profiled memory guard.
Anonymous diagnostics: SCNP weighted loss/coefficient, changed-logit fraction,
absolute logit displacement and foreground fraction by channel, LSO/base loss,
BN gradient norms and retained regional gradients. Correlations do not prove cause.

## Budget, admission and delivery

Only physical GPUs 0/1, at most two GPU workers; no interference with other jobs.
All four 12-arrival GPU profiles must PASS; measured same-condition peak admission
uses max(1.2*peak,peak+512MiB), invalid profiles refuse, cold profile retains existing
5GiB fallback. Whole fixed matrix admission has 20% timing reserve. New immutable
T0 before GPU profiles; 22h online/24h wall/48 GPU-worker hours/32GiB. Charge all
profiles/failures; no retries, old ledger rewriting, budget reset, matrix pruning
or automatic follow-up round. Neutral process commands and mounted storage required.

Reliable detached queue retires all online workers before separate CPU scoring.
After actual finish verify unique 24x1951 coverage, attempts and budgets, analyze
all positive/negative results and mechanism diagnostics, and publish own code,
protocol and anonymous aggregates to DLwbm123/DPA-CTTA on experiment/r43-scnp-v1.
Every GitHub operation explicitly uses local proxy http://127.0.0.1:7897;
verify remote SHA and anonymous report access, then local/remote DELIVERY.
Exclude images/labels, per-content predictions/scores/identifiers/hashes,
model states, private paths/logs and third-party PDFs. Startup is not completion.
