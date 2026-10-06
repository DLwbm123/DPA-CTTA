# R25: conditional parameter exploration on C + adapter

User authorization (2026-10-06, Asia/Shanghai): execute the prioritized mechanism
experiments for about 24 hours, allowed to exceed 24 but never 48 hours; monitor
hourly. This is a new finite campaign, not a continuation of R24's six-hour clock.

## Scope and evidence

Reuse the original model-only fundus checkpoint, BN-affine C adaptation and the
R24 rank-4 up3 residual adapter with original-image anchored descriptor. No source
images, source training, external pretrained networks, online labels, task/domain
identifiers, future images or threshold tuning. Past unlabeled target images are
allowed and explicitly bounded. Baseline C retains its exact original update.

This is a mechanism-transfer study, not complete reproduction of 3PO, RaPO,
TopoOT, SOAP-Bubbles or RN-DPO. In particular, the RL arms implement a contextual
bandit over four continuous adapter gate logits, not multi-step RL, RLVR or a
Bayesian posterior with calibrated uncertainty. Covariance learning does not
implement EVON. The structure score uses threshold-component OT chains including
holes; it does not compute persistent homology diagrams or reproduce TopoOT.

Mechanism references: https://arxiv.org/abs/2608.09805 (3PO),
https://arxiv.org/abs/2605.09640 (RaPO), https://arxiv.org/abs/2601.20333 (TopoOT),
https://arxiv.org/abs/2606.23357 (SOAP-Bubbles),
https://arxiv.org/abs/2601.23222 (RN-DPO).

## Frozen conditions, in priority order

| Arm | Intervention |
|---|---|
| C | Original plain consistency, BN affine LR 1e-4 |
| ANCHOR | R24 anchored rank-4 adapter, adapter LR 3e-4 |
| REPEAT5 | Five consecutive ANCHOR updates per arrival, a compute/update control |
| RANDOM4 | Four temporary parameter proposals; uniformly commit one, no policy learning |
| GREEDY | Same fixed Gaussian proposals, choose best structure+retention reward, no policy learning |
| PG_CONS | Learn conditional Gaussian using consistency+extent reward |
| PG_STRUCT | Add threshold-component transport and nesting reward |
| PG_RETAIN | Add bounded historical prediction retention; registered primary |
| PG_COV | PG_RETAIN with learned full 4x4 Cholesky covariance |
| RN_DPO | GREEDY candidates, then replay chosen branch with region-normalized preference loss |

C and ANCHOR are exact code-path controls. All four-proposal arms run the same
reward probes, including disabled reward components, to make their physical
comparison meaningful. REPEAT5 is close in forward count but not exactly matched
in FLOPs/Adam updates; do not claim equal-cost superiority. RN_DPO requires one
additional replay update. All actual operations and worker time are charged.

## Algorithm

The frozen checkpoint's first-BN input mean/variance descriptor conditions the
existing adapter. A separate CPU linear policy receives the descriptor and five
historical scalars (previous selected tanh action plus its reward). Gaussian mean
is 0.5*tanh(raw), standard deviation 0.1+0.4*sigmoid(raw), initially zero mean and
0.3 SD. PG_COV additionally uses six lower-triangle terms 0.1*tanh(raw). The
adapter's gate preactivation receives 0.75*tanh(action). Four independent actions
are drawn with a separate checkpointed generator. Each candidate starts from the
same pre-arrival BN, adapter, Adam and augmentation RNG state. Each executes one
C update, followed by read-only reward probes; only the chosen branch is committed.
No candidate can inherit another candidate's update. Final threshold stays 0.5.

Rewards are detached, label-free proxies. On 64x64 area-pooled probabilities:
- consistency: foreground/background balanced absolute difference from an
  independent current-image gamma=1.2 probe;
- extent: mean absolute log ratio of candidate versus pre-update soft area,
  epsilon=1e-3 and clipped to 2;
- structure: component/finite-hole sets at thresholds [0.2,0.35,0.5,0.65,0.8],
  4-connected, minimum area 2, largest eight each, descriptors normalized centroid,
  sqrt relative area and confidence. Entropic OT epsilon=0.1, 32 Sinkhorn steps,
  plus component-count mismatch; cross-view mean + 0.25 adjacent-threshold mean;
- nesting: mean relu(P_OC-P_OD);
- retention: balanced prediction difference from one stored historical image and
  its insertion-time prediction, round-robin among a FIFO of four images inserted
  every 16 arrivals, up to 64-arrival horizon. No files are reopened for these images.

R_cons = -consistency - 0.05*extent;
R_struct = R_cons - 0.2*structure - 0.1*nesting;
R_retain = R_struct - 0.1*retention.
These signals may preserve wrong structures or old errors; they are not correctness
or verifiable task rewards. No fixed circle, area, cup/disc ratio or disease-free
shape is imposed. Topological stability is not assumed equivalent to Dice.

Policy learning uses a leave-one-out group baseline and prior-arrival EMA scale
(initial 0.05, beta 0.99, denominator floor 0.01), one score-function gradient
step, Adam 3e-4, gradient clipping 1, and 0.01 KL to N(0,0.09 I). Policy actions
and reward computation are detached. Deterministic best-of-four deployment is
shared with GREEDY; this separates learned proposals from search itself.

RN_DPO uses best/worst R_retain candidates as detached binary preferences. It
replays from the pre-arrival state with the selected action. C loss gains 0.1 times
-log sigmoid(0.1*mean_R[(winner-loser)*(strong_logits-reference_logits)]), where
R is the actual pixel/channel disagreement region and the reference is the
pre-arrival zero-action prediction. Empty R contributes zero. This tests the
region normalization mechanism with an untrained structural judge and local
reference, not the original paper's separately supervised judge/offline protocol.

## Matrix and budget

Each full trajectory has 1,951 arrivals; identical two orders, all original
checkpoint resets between trajectories. No short performance screen and no
performance-based pruning. Three seeds are mandatory: 20260907,17011,29009.
Before ANY formal execution/label access, ten 12-arrival engineering profiles
measure speed and memory. If the full three-seed matrix has conservative estimated
wall under 16h AND all five seeds fit the hard resource cap, add fixed seeds
41017,53003. Otherwise use three. No later seed/arm/hyperparameter changes.
All seeds for the first eight arms precede PG_COV/RN_DPO. The ten-arm matrix has
60 or 100 full trajectories. If the complete three-seed matrix cannot fit, stop
before formal work and report the engineering/budget issue; do not silently shrink.

Runtime T0 is immutable at background launch and includes profiles, failures and
scoring. Target around 24h; finish early when registered work ends rather than
padding runtime. Online cutoff T0+46h, absolute cutoff T0+48h, cumulative GPU-worker
cap 184h, campaign disk cap 64GiB. Only assigned GPUs 4-7 on the authorized NAS
host, neutral process command lines. Watchdog and worker deadlines enforce caps
without a foreground chat. Existing unrelated processes are untouched.

CPU qualification precedes launch. One evidenced transient I/O/network recovery
per job is allowed using the existing journal and original cost ledger. Numerical
failure, weak performance and exhausted budget do not authorize scientific retries.
Engineering fixes require receipts and code-version tracking; no live code edit
under running workers or reset of the original campaign clock.

## Sealed scoring and conclusions

No online worker opens labels. Both SEARCH and REVIEW labels remain closed until
ALL formal workers retire. There is no label-based candidate selection this round.
The existing split is 1,017 SEARCH / 678 REVIEW / 256 context identities. Content
was historically exposed; REVIEW is campaign-held, not independent generalization.
Patient linkage and ROI crop provenance remain UNKNOWN.

Primary outcome: equal-domain/equal-OD-OC 512px hard Dice, then equal order/seed.
Dice=2TP/(pred+GT), empty/empty=1. Report image-weighted Dice, all domain/channel
cells, negative cells, actual operations, timing, VRAM and all failed attempts.
ASSD and soft metrics are not computed. Content-cluster bootstrap couples orders
and seeds and does not resolve patient dependence or multiplicity.

Registered contrasts: PG_RETAIN vs C, ANCHOR, REPEAT5; GREEDY vs RANDOM4;
PG_RETAIN vs GREEDY; PG_STRUCT vs PG_CONS; PG_RETAIN vs PG_STRUCT;
PG_COV vs PG_RETAIN; RN_DPO vs GREEDY. Primary never changes retrospectively.

A development signal requires a complete registered matrix, primary REVIEW gain
>=0.3 pp vs C, positive mean in both orders, >=80% positive seed/order trajectories,
nonnegative image-weighted gain, worst seed-averaged domain/channel cell >=-2 pp,
and positive mean gains over ANCHOR, REPEAT5 and GREEDY. Failure is retained, not
retuned. This is not a claim of novelty, clinical validity or general superiority.
Post-hoc 64px candidate-mask Dice measures reward ranking and oracle headroom;
it is only a mechanism diagnostic and never feeds training or replaces primary Dice.

## Monitoring and delivery

Explicitly authorized hourly monitoring checks identities, logs, receipts, coverage,
resources and errors. Notify on meaningful progress, failure, recovery or completion;
remain quiet on unchanged healthy state. The monitor cannot append new experiments.
After completion, publish code/protocol/aggregate results/report to the existing
public GitHub repository via the required proxy; verify push and anonymous access.
Never publish images, labels, proposal masks, sample identities, checkpoint, private
paths or credentials. Pause this monitor after terminal-state delivery.
